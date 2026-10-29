"""Held-out evaluation runner: cases x generators x budgets x seeds -> JSONL metrics + summary tables."""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from brickcore.catalog import load_catalog
from brickcore.generators import get as get_generator
from brickcore.mesh import MeshError, load_mesh, normalize, voxelize
from brickcore.schema import Bounds, Constraints
from brickcore.stability import analyze, analyze_build
from brickcore.validate import validate_build

ROOT = Path(__file__).resolve().parent.parent

# Frozen procedural suite with expected behaviour. Do not tune the generator against these after freezing.
PROCEDURAL = {
    "cube": ("simple", {"status": "validated_candidate", "min_coverage": 0.8}),
    "slab": ("simple", {"status": "validated_candidate", "min_coverage": 0.8}),
    "tower": ("simple", {"status": "validated_candidate", "min_coverage": 0.8}),
    "l_shape": ("simple", {"status": "validated_candidate", "min_coverage": 0.8}),
    "t_shape": ("simple", {"status": "validated_candidate", "min_coverage": 0.8}),
    "wedge": ("simple", {"status": "validated_candidate", "min_coverage": 0.8}),
    "pyramid": ("overhang", {"min_coverage": 0.8}),
    "sphere": ("overhang", {"min_coverage": 0.8}),
    "cylinder": ("budget", {"min_coverage": 0.5}),
    "bridge": ("bridge", {"min_coverage": 0.8}),
    "flat_sheet_impossible": ("impossible", {"status": "partial_candidate"}),
    "thin_pole": ("impossible", {"status": "no_valid_candidate"}),
    "open_box_invalid": ("invalid_input", {"status": "failed", "error_code": "non_watertight_mesh"}),
}


@dataclass
class Case:
    id: str
    mesh: str
    source: str
    group: str
    license: str = ""
    up: str = "+z"
    size: tuple[int, int, int] = (16, 16, 8)
    expected: dict = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)


def procedural_cases() -> list[Case]:
    from brickcore.fixtures import FIXTURE_DIR, write_fixtures
    if not (FIXTURE_DIR / "cube.stl").exists():
        write_fixtures(FIXTURE_DIR)
    return [Case(id=f"proc_{n}", mesh=str((FIXTURE_DIR / f"{n}.stl").relative_to(ROOT)), source="procedural", group=g,
                 license="project fixture", size=(8, 8, 4) if n == "cube" else (16, 16, 8), expected=e) for n, (g, e) in PROCEDURAL.items()]


def thingi_cases() -> list[Case]:
    manifest = ROOT / "data" / "thingi10k" / "manifest.json"
    if not manifest.exists():
        raise SystemExit("no Thingi10K manifest; run `brickforge-bench fetch-thingi` first")
    return [Case(id=f"thingi_{m['file_id']}", mesh=m["path"], source="thingi10k", group="thingi10k", license=m["license"], tags=m.get("tags", []))
            for m in json.loads(manifest.read_text())["models"]]


def hardware() -> dict:
    try:
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        cpu = platform.processor()
    return {"platform": platform.platform(), "cpu": cpu, "python": platform.python_version()}


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _check_expected(expected: dict, rec: dict) -> bool | None:
    if not expected:
        return None
    if "status" in expected and rec["status"] != expected["status"]:
        return False
    if "error_code" in expected and rec.get("error_code") != expected["error_code"]:
        return False
    if "min_coverage" in expected and (rec.get("coverage") or 0) < expected["min_coverage"]:
        return False
    return True


def run(cases: list[Case], generators: list[str], budgets: list[int], seeds: list[int], out_jsonl: Path,
        stability_mode: str = "final", color_id: str = "4", progress: bool = True) -> list[dict]:
    catalog = load_catalog()
    hw, commit = hardware(), git_commit()
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with out_jsonl.open("w") as fh:
        for ci, case in enumerate(cases):
            bounds = Bounds(x_studs=case.size[0], y_studs=case.size[1], z_plates=case.size[2] * 3)
            base = {"case": case.id, "source": case.source, "group": case.group, "license": case.license, "mesh": case.mesh,
                    "grid": list(case.size), "catalog": catalog.version, "commit": commit, "hardware": hw}
            t0 = time.perf_counter()
            try:
                mesh, meta = load_mesh(ROOT / case.mesh)
                m, _ = normalize(mesh, bounds, catalog, up=case.up)
                target = voxelize(m, bounds, catalog)
                prep = {"triangles": meta["triangles"], "target_cells": int(target.sum()), "prep_s": round(time.perf_counter() - t0, 3)}
                err = None
            except MeshError as e:
                err, prep = e, {"prep_s": round(time.perf_counter() - t0, 3)}
            for gen_name in generators:
                for budget in budgets:
                    for seed in seeds:
                        rec = {**base, **prep, "generator": gen_name, "budget": budget, "seed": seed}
                        if err is not None:
                            rec.update({"status": "failed", "error_code": err.code, "message": err.message})
                        else:
                            constraints = Constraints(max_pieces=budget, bounds=bounds, color_id=color_id)
                            t1 = time.perf_counter()
                            res = get_generator(gen_name)(target, catalog, constraints, seed=seed, build_id=f"{case.id}_{gen_name}_{budget}_{seed}")
                            rec.update({"status": res.status, "iou": round(res.fidelity, 4), "coverage": round(res.coverage, 4),
                                        "extraneous": round(res.extraneous, 4), "generate_s": round(time.perf_counter() - t1, 3),
                                        "pieces": len(res.build.placements) if res.build else 0, "instruction_status": res.instruction_status})
                            if res.build is not None:
                                report = validate_build(res.build, catalog)
                                rec["valid"] = report["ok"]
                                rec["planned"] = report["instruction_status"] == "planned"
                                t2 = time.perf_counter()
                                if stability_mode == "steps":
                                    st = analyze_build(res.build.placements, res.build.steps, catalog)
                                    rec.update({"stable": st["final"]["stable"], "stability_mode": st["final"]["mode"], "utilization": st["final"]["utilization"],
                                                "all_steps_stable": st["all_steps_stable"], "max_step_utilization": st["max_step_utilization"]})
                                else:
                                    st = analyze(res.build.placements, catalog)
                                    rec.update({"stable": st["stable"], "stability_mode": st["mode"], "utilization": st["utilization"]})
                                rec["stability_s"] = round(time.perf_counter() - t2, 3)
                            else:
                                rec.update({"valid": False, "planned": False, "stable": None})
                        rec["meets_expected"] = _check_expected(case.expected, rec)
                        fh.write(json.dumps(rec) + "\n")
                        records.append(rec)
            if progress:
                print(f"[{ci + 1}/{len(cases)}] {case.id}: {records[-1]['status']}", file=sys.stderr)
    return records


def summarize(records: list[dict]) -> dict:
    groups: dict[tuple[str, int], list[dict]] = {}
    for r in records:
        groups.setdefault((r["generator"], r["budget"]), []).append(r)
    rows = []
    for (gen, budget), rs in sorted(groups.items()):
        n = len(rs)
        built = [r for r in rs if r.get("valid")]
        ious = [r["iou"] for r in built]
        rows.append({
            "generator": gen, "budget": budget, "cases": n,
            "input_failed": sum(r["status"] == "failed" for r in rs),
            "valid_rate": round(len(built) / n, 3),
            "planned_rate": round(sum(bool(r.get("planned")) for r in rs) / n, 3),
            "stable_rate": round(sum(bool(r.get("stable")) for r in rs) / n, 3),
            "mean_iou": round(statistics.mean(ious), 4) if ious else None,
            "median_iou": round(statistics.median(ious), 4) if ious else None,
            "mean_coverage": round(statistics.mean(r["coverage"] for r in built), 4) if built else None,
            "mean_pieces": round(statistics.mean(r["pieces"] for r in built), 1) if built else None,
            "mean_iou_per_piece_x100": round(statistics.mean(r["iou"] / r["pieces"] * 100 for r in built if r["pieces"]), 3) if built else None,
            "mean_generate_s": round(statistics.mean(r.get("generate_s", 0) for r in rs), 3),
            "expected_met": sum(r["meets_expected"] is True for r in rs), "expected_total": sum(r["meets_expected"] is not None for r in rs),
        })
    return {"by_generator_budget": rows, "records": len(records)}


def to_markdown(summary: dict, records: list[dict]) -> str:
    cols = ["generator", "budget", "cases", "valid_rate", "planned_rate", "stable_rate", "mean_iou", "median_iou", "mean_coverage", "mean_pieces", "mean_iou_per_piece_x100", "mean_generate_s", "expected_met", "expected_total"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in summary["by_generator_budget"]:
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    lines += ["", "Per case (IoU by budget):", ""]
    cases = sorted({r["case"] for r in records})
    budgets = sorted({r["budget"] for r in records})
    gens = sorted({r["generator"] for r in records})
    lines.append("| case | group | " + " | ".join(f"{g}@{b}" for g in gens for b in budgets) + " | note |")
    lines.append("|---|---|" + "---|" * (len(gens) * len(budgets)) + "---|")
    idx = {(r["case"], r["generator"], r["budget"]): r for r in records if r["seed"] == min(x["seed"] for x in records)}
    for c in cases:
        cells, note = [], ""
        for g in gens:
            for b in budgets:
                r = idx.get((c, g, b))
                if r is None:
                    cells.append("")
                elif r["status"] == "failed":
                    cells.append("fail"); note = r.get("error_code", "")
                else:
                    cells.append(f"{r['iou']:.3f}" + ("" if r.get("valid") else "!"))
                    if r["status"] != "validated_candidate":
                        note = r["status"]
        lines.append(f"| {c} | {idx.get((c, gens[0], budgets[0]), {}).get('group', '')} | " + " | ".join(cells) + f" | {note} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="brickforge-bench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch-thingi", help="select and download the licensed Thingi10K subset")
    f.add_argument("--n", type=int, default=40)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--reselect", action="store_true", help="rewrite the committed selection (changes the benchmark)")
    r = sub.add_parser("run", help="run generators over a suite")
    r.add_argument("--suite", choices=["procedural", "thingi", "all"], default="all")
    r.add_argument("--generators", nargs="+", default=["greedy_v1"])
    r.add_argument("--budgets", nargs="+", type=int, default=[50, 100, 200])
    r.add_argument("--seeds", nargs="+", type=int, default=[42])
    r.add_argument("--stability", choices=["final", "steps"], default="final")
    r.add_argument("--out", required=True, help="output prefix, e.g. docs/benchmarks/baseline_v1")
    s = sub.add_parser("summarize")
    s.add_argument("jsonl")
    args = ap.parse_args(argv)

    if args.cmd == "fetch-thingi":
        from . import thingi10k as t
        if args.reselect or not t.SUBSET_FILE.exists():
            sel = t.write_subset(n=args.n, seed=args.seed)
            print(f"selected {len(sel)} models -> {t.SUBSET_FILE}")
        man = t.download_subset()
        print(f"downloaded {len(man['models'])} meshes -> {t.DATA_DIR / 'meshes'}")
        return 0
    if args.cmd == "run":
        cases = []
        if args.suite in ("procedural", "all"):
            cases += procedural_cases()
        if args.suite in ("thingi", "all"):
            cases += thingi_cases()
        out = Path(args.out)
        records = run(cases, args.generators, args.budgets, args.seeds, out.with_suffix(".jsonl"), stability_mode=args.stability)
        summary = summarize(records)
        out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        out.with_suffix(".md").write_text(to_markdown(summary, records))
        print(to_markdown(summary, records))
        return 0
    if args.cmd == "summarize":
        records = [json.loads(l) for l in Path(args.jsonl).read_text().splitlines() if l.strip()]
        print(to_markdown(summarize(records), records))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
