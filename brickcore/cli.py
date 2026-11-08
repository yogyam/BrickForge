"""CPU-only command line: fixtures, generate, validate, plan, stability, export.

The greedy generator needs nothing beyond the default dependencies. Learned generators (names starting with
`sft` or `rl`) are loaded on demand and need the `ml` dependency group (torch) plus a checkpoint under checkpoints/.
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from pydantic import ValidationError

from . import __version__
from .catalog import load_catalog
from .export import bom_csv, ldraw_text
from .generators import get as get_generator
from .mesh import MeshError, load_mesh, normalize, occupancy_cells, voxelize
from .plan import plan_assembly
from .schema import Bounds, Build, Constraints
from .stability import analyze_build
from .validate import validate_build

ML_HINT = "learned generators need torch: run `make setup-ml` (or `uv sync --group ml`) and use `uv run --group ml brickforge ...`"
TRAIN_TARGETS = {"sft_v1": "make sft-train", "sft_v2": "make sft-train-v2", "rl_v1": "make rl-train", "rl_v2": "make rl-train-v2"}


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _dump(obj) -> str:
    return json.dumps(obj, indent=2, sort_keys=True) + "\n"


def _err(msg: str) -> int:
    print(f"error: {msg}", file=sys.stderr)
    return 2


def _validation_lines(e: ValidationError) -> str:
    return "; ".join(f"{'.'.join(str(x) for x in err['loc']) or 'input'}: {err['msg']}" for err in e.errors())


def _load_build(path: str) -> Build | None:
    """Read a build.json or explain in one line why it is not one."""
    p = Path(path)
    try:
        return Build.from_json(p.read_text())
    except FileNotFoundError:
        _err(f"no such file: {p}")
    except ValidationError as e:
        _err(f"{p} is not a build.json ({_validation_lines(e)})")
    except (UnicodeDecodeError, ValueError) as e:
        _err(f"{p} is not valid JSON ({e})")
    return None


class StageTimeout(Exception):
    pass


@contextmanager
def _deadline(seconds: float | None, stage: str):
    """Wall-clock limit for a stage. Uses SIGALRM, so it only applies on the main thread of a POSIX process;
    elsewhere it is a no-op and the stage runs unbounded (documented limitation)."""
    if not seconds or threading.current_thread() is not threading.main_thread() or not hasattr(signal, "setitimer"):
        yield
        return

    def _raise(signum, frame):
        raise StageTimeout(stage)

    previous = signal.signal(signal.SIGALRM, _raise)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def cmd_fixtures(args) -> int:
    from .fixtures import write_fixtures
    for p in write_fixtures(Path(args.out)):
        print(p)
    return 0


def _resolve_generator(name: str):
    """Return the generator callable, or None after printing a one-line reason."""
    try:
        return get_generator(name)
    except KeyError as e:
        _err(f"{e.args[0]}; learned policies are sft_v1, sft_v2, rl_v1, rl_v2 and their _nostop variants (see docs/demo.md)")
    except ModuleNotFoundError as e:
        if e.name and e.name.split(".")[0] == "torch":
            _err(ML_HINT)
        else:
            _err(f"generator {name!r} could not be imported: {e}")
    return None


def cmd_generate(args) -> int:
    catalog = load_catalog(args.catalog)
    if not Path(args.mesh).exists():
        return _err(f"no such mesh file: {args.mesh}")
    try:
        bounds = Bounds(x_studs=args.size[0], y_studs=args.size[1], z_plates=args.size[2] * 3)
        constraints = Constraints(max_pieces=args.max_pieces, bounds=bounds, color_id=args.color)
    except ValidationError as e:
        return _err(f"bad arguments ({_validation_lines(e)}); studs 1..24, layers 1..12, pieces 1..200")
    if args.color not in catalog.supported_color_ids:
        return _err(f"color {args.color} not in catalog {catalog.version}; supported: {', '.join(sorted(catalog.supported_color_ids, key=int))}")
    gen = _resolve_generator(args.generator)
    if gen is None:
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    timings = {}
    t = time.perf_counter()
    try:
        with _deadline(args.parse_timeout, "parse"):
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
    except StageTimeout as e:
        result = {"status": "failed", "error_code": "worker_timeout", "stage": str(e), "message": f"{e} stage exceeded {args.parse_timeout} s", "mesh": str(args.mesh)}
        _write(out / "result.json", _dump(result))
        print(_dump(result), end="")
        return 1
    _write(out / "normalization.json", _dump({**meta, **norm.to_dict(), "bounds": bounds.model_dump()}))
    _write(out / "voxels.json", _dump({"grid": list(occ.shape), "cells": occupancy_cells(occ)}))
    build_id = args.build_id or f"build_{meta['source_sha256'][:8]}_{args.max_pieces}"
    t = time.perf_counter()
    try:
        res = gen(occ, catalog, constraints, seed=args.seed, build_id=build_id, time_budget_s=args.generate_timeout)
    except FileNotFoundError as e:
        family = args.generator.split("_nostop")[0].split("_s1")[0]
        hint = TRAIN_TARGETS.get(family, "the matching training target in the Makefile")
        return _err(f"generator {args.generator!r} needs a checkpoint that is not here ({e.filename}); train it with `{hint}`")
    timings["generate"] = round(time.perf_counter() - t, 3)
    result = {
        "status": res.status, "generator": res.build.generator.name if res.build else args.generator,
        "iou": round(res.fidelity, 4), "coverage": round(res.coverage, 4), "extraneous": round(res.extraneous, 4),
        "pieces": len(res.build.placements) if res.build else 0, "instruction_status": res.instruction_status,
        "search_aborted": res.diagnostics.get("aborted"),
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
        t = time.perf_counter()
        stab = analyze_build(res.build.placements, res.build.steps, catalog)
        timings["stability"] = round(time.perf_counter() - t, 3)
        _write(out / "stability.json", _dump(stab))
        result["stability"] = {"final_stable": stab["final"]["stable"], "mode": stab["final"]["mode"], "final_utilization": stab["final"]["utilization"],
                               "all_steps_stable": stab["all_steps_stable"], "first_unstable_step": stab["first_unstable_step"],
                               "max_step_utilization": stab["max_step_utilization"], "label": stab["label"]}
    _write(out / "result.json", _dump(result))
    print(_dump({k: v for k, v in result.items() if k != "diagnostics"}), end="")
    return 0


def cmd_validate(args) -> int:
    catalog = load_catalog(args.catalog)
    build = _load_build(args.build)
    if build is None:
        return 2
    report = validate_build(build, catalog)
    print(_dump(report), end="")
    return 0 if report["ok"] else 1


def cmd_plan(args) -> int:
    catalog = load_catalog(args.catalog)
    build = _load_build(args.build)
    if build is None:
        return 2
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


def cmd_stability(args) -> int:
    catalog = load_catalog(args.catalog)
    build = _load_build(args.build)
    if build is None:
        return 2
    stab = analyze_build(build.placements, build.steps, catalog)
    if args.out:
        _write(Path(args.out), _dump(stab))
    print(_dump({k: v for k, v in stab.items() if k != "per_step"}), end="")
    return 0 if stab["final"]["stable"] else 1


def cmd_export(args) -> int:
    catalog = load_catalog(args.catalog)
    build = _load_build(args.build)
    if build is None:
        return 2
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
    ap = argparse.ArgumentParser(prog="brickforge", description=f"BrickForge {__version__}: turn a 3D mesh into a checked brick build.")
    ap.add_argument("--catalog", default="basic-bricks-v1", help="part catalog name under catalog/ (default: basic-bricks-v1)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fixtures", help="write the 13 procedural fixture meshes")
    f.add_argument("--out", default="fixtures/meshes", help="directory for the .stl files (default: fixtures/meshes)")
    f.set_defaults(fn=cmd_fixtures)

    g = sub.add_parser("generate", help="mesh -> build.json, validation.json, plan.json, stability.json, bom.csv, build.ldr, result.json")
    g.add_argument("--mesh", required=True, help="input mesh: binary or ASCII .stl, or a single-file .glb; must be watertight")
    g.add_argument("--out", required=True, help="output folder (created if missing)")
    g.add_argument("--max-pieces", type=int, default=100, help="piece budget, 1..200 (default 100)")
    g.add_argument("--size", type=int, nargs=3, default=[16, 16, 8], metavar=("X_STUDS", "Y_STUDS", "LAYERS"),
                   help="grid in studs x studs x brick layers; studs 1..24, layers 1..12 (default 16 16 8)")
    g.add_argument("--generator", default="greedy_v1",
                   help="greedy_v1 (default, no torch needed), or a learned policy such as sft_v2 or rl_v2_nostop (needs `make setup-ml`)")
    g.add_argument("--color", default="4", help="LDraw colour code for every brick (default 4 = red)")
    g.add_argument("--up", default="+z", help="source axis that points up: +x -x +y -y +z -z (default +z)")
    g.add_argument("--yaw", type=int, default=0, help="rotation about up in 90 degree steps (default 0)")
    g.add_argument("--seed", type=int, default=42, help="recorded in build.json; greedy_v1 and the learned decoders are deterministic")
    g.add_argument("--build-id", default=None, help="id stored in build.json (default: build_<sha8>_<max_pieces>)")
    g.add_argument("--parse-timeout", type=float, default=60.0, help="seconds for load+normalize+voxelize before worker_timeout (default 60)")
    g.add_argument("--generate-timeout", type=float, default=30.0, help="seconds of search before returning the best candidate so far (default 30)")
    g.set_defaults(fn=cmd_generate)

    v = sub.add_parser("validate", help="re-check a build.json against every rule (exit 0 = ok)")
    v.add_argument("build", help="path to a build.json")
    v.set_defaults(fn=cmd_validate)

    pl = sub.add_parser("plan", help="compute a checked assembly order for a build.json")
    pl.add_argument("build", help="path to a build.json")
    pl.add_argument("--write", action="store_true", help="store the planned steps back into the build file")
    pl.add_argument("--node-budget", type=int, default=1_000_000, help="search nodes before giving up (default 1000000)")
    pl.add_argument("--time-budget", type=float, default=30.0, help="seconds before giving up (default 30)")
    pl.set_defaults(fn=cmd_plan)

    st = sub.add_parser("stability", help="static force-model stability of a build.json, final and per step (exit 0 = stable under the model)")
    st.add_argument("build", help="path to a build.json")
    st.add_argument("--out", default=None, help="write the full report including per-step results to this file")
    st.set_defaults(fn=cmd_stability)

    e = sub.add_parser("export", help="BOM + LDraw from a validated build.json")
    e.add_argument("build", help="path to a build.json")
    e.add_argument("--out", required=True, help="output folder for bom.csv and build.ldr")
    e.set_defaults(fn=cmd_export)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
