"""CPU-only command line: fixtures, generate, validate, export."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import __version__
from .catalog import load_catalog
from .export import bom_csv, ldraw_text
from .generate import generate
from .mesh import MeshError, load_mesh, normalize, occupancy_cells, voxelize
from .plan import plan_assembly
from .schema import Bounds, Build, Constraints
from .validate import validate_build


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _dump(obj) -> str:
    return json.dumps(obj, indent=2, sort_keys=True) + "\n"


def cmd_fixtures(args) -> int:
    from .fixtures import write_fixtures
    for p in write_fixtures(Path(args.out)):
        print(p)
    return 0


def cmd_generate(args) -> int:
    catalog = load_catalog(args.catalog)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bounds = Bounds(x_studs=args.size[0], y_studs=args.size[1], z_plates=args.size[2] * 3)
    constraints = Constraints(max_pieces=args.max_pieces, bounds=bounds, color_id=args.color)
    if args.color not in catalog.supported_color_ids:
        print(f"error: color {args.color} not in catalog {catalog.version}", file=sys.stderr)
        return 2
    timings = {}
    t = time.perf_counter()
    try:
        mesh, meta = load_mesh(Path(args.mesh))
        timings["load"] = round(time.perf_counter() - t, 3)
        t = time.perf_counter()
        mesh, norm = normalize(mesh, bounds, catalog, up=args.up, yaw_deg=args.yaw)
        timings["normalize"] = round(time.perf_counter() - t, 3)
        t = time.perf_counter()
        occ = voxelize(mesh, bounds, catalog)
        timings["voxelize"] = round(time.perf_counter() - t, 3)
    except MeshError as e:
        result = {"status": "failed", "error_code": e.code, "message": e.message, "mesh": str(args.mesh)}
        _write(out / "result.json", _dump(result))
        print(_dump(result), end="")
        return 1
    _write(out / "normalization.json", _dump({**meta, **norm.to_dict(), "bounds": bounds.model_dump()}))
    _write(out / "voxels.json", _dump({"grid": list(occ.shape), "cells": occupancy_cells(occ)}))
    build_id = args.build_id or f"build_{meta['source_sha256'][:8]}_{args.max_pieces}"
    t = time.perf_counter()
    res = generate(occ, catalog, constraints, seed=args.seed, build_id=build_id)
    timings["generate"] = round(time.perf_counter() - t, 3)
    result = {
        "status": res.status, "iou": round(res.fidelity, 4), "coverage": round(res.coverage, 4), "extraneous": round(res.extraneous, 4),
        "pieces": len(res.build.placements) if res.build else 0, "instruction_status": res.instruction_status,
        "timings_s": timings, "diagnostics": res.diagnostics,
        "label": "checks passed for the simplified brick model" if res.build else "no valid candidate",
    }
    if res.build is not None:
        report = validate_build(res.build, catalog)
        _write(out / "build.json", res.build.to_json())
        _write(out / "validation.json", _dump(report))
        _write(out / "bom.csv", bom_csv(res.build, catalog))
        _write(out / "build.ldr", ldraw_text(res.build, catalog))
        _write(out / "plan.json", _dump({"status": res.instruction_status, "steps": len(res.build.steps), **res.plan}))
    _write(out / "result.json", _dump(result))
    print(_dump({k: v for k, v in result.items() if k != "diagnostics"}), end="")
    return 0


def cmd_validate(args) -> int:
    catalog = load_catalog(args.catalog)
    build = Build.from_json(Path(args.build).read_text())
    report = validate_build(build, catalog)
    print(_dump(report), end="")
    return 0 if report["ok"] else 1


def cmd_plan(args) -> int:
    catalog = load_catalog(args.catalog)
    build = Build.from_json(Path(args.build).read_text())
    plan = plan_assembly(build.placements, catalog, node_budget=args.node_budget, time_budget_s=args.time_budget)
    print(_dump({"status": plan.status, "steps": len(plan.steps), **plan.diagnostics}), end="")
    if plan.status == "planned" and args.write:
        planned = build.model_copy(update={"steps": plan.steps})
        report = validate_build(planned, catalog)
        if not report["ok"]:
            print("planned order failed replay validation; not writing", file=sys.stderr)
            return 1
        Path(args.build).write_text(planned.to_json())
    return 0 if plan.status == "planned" else 1


def cmd_export(args) -> int:
    catalog = load_catalog(args.catalog)
    build = Build.from_json(Path(args.build).read_text())
    report = validate_build(build, catalog)
    if not report["ok"]:
        print("refusing to export a structurally invalid build; run `validate` for details", file=sys.stderr)
        return 1
    out = Path(args.out)
    _write(out / "bom.csv", bom_csv(build, catalog))
    _write(out / "build.ldr", ldraw_text(build, catalog))
    print(out / "bom.csv")
    print(out / "build.ldr")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="brickforge", description=f"BrickForge {__version__}")
    ap.add_argument("--catalog", default="basic-bricks-v1")
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fixtures", help="write procedural fixture meshes")
    f.add_argument("--out", default="fixtures/meshes")
    f.set_defaults(fn=cmd_fixtures)

    g = sub.add_parser("generate", help="mesh -> build.json, validation.json, bom.csv, build.ldr")
    g.add_argument("--mesh", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--max-pieces", type=int, default=100)
    g.add_argument("--size", type=int, nargs=3, default=[16, 16, 8], metavar=("X_STUDS", "Y_STUDS", "LAYERS"))
    g.add_argument("--color", default="4")
    g.add_argument("--up", default="+z", help="source axis that points up: +x -x +y -y +z -z")
    g.add_argument("--yaw", type=int, default=0, help="rotation about up in 90 degree steps")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--build-id", default=None)
    g.set_defaults(fn=cmd_generate)

    v = sub.add_parser("validate", help="validate a build.json")
    v.add_argument("build")
    v.set_defaults(fn=cmd_validate)

    pl = sub.add_parser("plan", help="compute a checked assembly order for a build.json")
    pl.add_argument("build")
    pl.add_argument("--write", action="store_true", help="store the planned steps back into the build file")
    pl.add_argument("--node-budget", type=int, default=1_000_000)
    pl.add_argument("--time-budget", type=float, default=30.0)
    pl.set_defaults(fn=cmd_plan)

    e = sub.add_parser("export", help="BOM + LDraw from a validated build.json")
    e.add_argument("build")
    e.add_argument("--out", required=True)
    e.set_defaults(fn=cmd_export)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
