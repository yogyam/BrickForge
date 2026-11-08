"""MCP server exposing the BrickForge pipeline as tools for an AI assistant (M2.6, demo only).

Run: scripts/mcp.sh   (stdio transport). Registered for Claude Code in .mcp.json.
All paths are confined to the project directory. Results are the same artifacts the CLI writes.
"""
from __future__ import annotations

import json
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .catalog import load_catalog
from .export import bom_rows
from .geometry import body_of
from .plan import plan_assembly
from .schema import Build
from .stability import analyze, analyze_build
from .validate import validate_build

ROOT = Path(__file__).resolve().parent.parent
mcp = FastMCP("brickforge", instructions=(
    "Tools for converting a 3D mesh into a LEGO-compatible brick build with checked assembly steps. "
    "Every result is labelled 'checks passed for the simplified brick model' or 'stable under the simplified "
    "static force model'; nothing is physically tested. Never describe a build as physically validated."
))


def _safe(path: str) -> Path:
    p = (ROOT / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    if ROOT not in p.parents and p != ROOT:
        raise ValueError(f"path must be inside the project: {path}")
    return p


def _load(build_path: str) -> Build:
    return Build.from_json(_safe(build_path).read_text())


@mcp.tool()
def list_fixtures() -> list[str]:
    """List bundled fixture meshes (relative paths) that can be passed to generate."""
    d = ROOT / "fixtures" / "meshes"
    return sorted(str(p.relative_to(ROOT)) for p in d.glob("*.stl")) if d.exists() else []


@mcp.tool()
def generate(mesh_path: str, out_dir: str, max_pieces: int = 100, x_studs: int = 16, y_studs: int = 16,
             layers: int = 8, color_id: str = "4", up: str = "+z", yaw_deg: int = 0, seed: int = 42,
             generator: str = "greedy_v1") -> dict:
    """Run the full pipeline on a mesh (STL or GLB inside the project). Writes build.json, validation.json,
    plan.json, stability.json, bom.csv, build.ldr and result.json into out_dir. Returns result.json.
    generator is greedy_v1 (default) or a learned policy such as sft_v2 / rl_v2_nostop (needs the ml group)."""
    from .cli import main
    out = _safe(out_dir)
    rc = main(["generate", "--mesh", str(_safe(mesh_path)), "--out", str(out), "--max-pieces", str(max_pieces),
               "--size", str(x_studs), str(y_studs), str(layers), "--color", color_id, "--up", up, "--yaw", str(yaw_deg), "--seed", str(seed), "--generator", generator])
    if rc == 2:
        return {"status": "failed", "error_code": "bad_arguments", "exit_code": rc, "message": "see the server log; check the generator name, torch install and argument ranges"}
    result = json.loads((out / "result.json").read_text())
    result["exit_code"] = rc
    result["out_dir"] = str(out.relative_to(ROOT))
    result.pop("diagnostics", None)
    return result


@mcp.tool()
def validate(build_path: str) -> dict:
    """Authoritative structural validation of a build.json: collisions, bounds, budget, stud connections,
    single component, step replay with insertion-path checks."""
    report = validate_build(_load(build_path), load_catalog())
    report.pop("supports", None)
    return report


@mcp.tool()
def plan(build_path: str) -> dict:
    """Compute a checked one-piece-per-step assembly order (does not modify the file)."""
    r = plan_assembly(_load(build_path).placements, load_catalog())
    d = dict(r.diagnostics)
    d.pop("per_step", None)
    return {"status": r.status, "order": r.order, **d}


@mcp.tool()
def stability(build_path: str) -> dict:
    """Static force-model stability of the final build and each assembly prefix. Returns the summary and
    the first unstable step if any. Not a physical test."""
    b = _load(build_path)
    rep = analyze_build(b.placements, b.steps, load_catalog())
    unstable = [s for s in rep["per_step"] if not s["stable"]]
    rep["per_step"] = unstable[:20]
    rep["unstable_step_count"] = len(unstable)
    return rep


@mcp.tool()
def inspect_step(build_path: str, step: int) -> dict:
    """Describe assembly step k: the piece added, its body bounds, what it attaches to, the assembly so far,
    and the stability of that prefix."""
    b = _load(build_path)
    cat = load_catalog()
    if step < 1 or step > len(b.steps):
        raise ValueError(f"step must be in 1..{len(b.steps)}")
    by_id = {p.id: p for p in b.placements}
    prefix_ids = [i for s in b.steps[:step] for i in s.added_placement_ids]
    added = [by_id[i] for i in b.steps[step - 1].added_placement_ids]
    prefix = [by_id[i] for i in prefix_ids]
    placed_before = prefix[:-len(added)]
    out = {"step": step, "of": len(b.steps), "pieces_placed_after_step": len(prefix), "added": []}
    for p in added:
        body = body_of(p, cat)
        supporters = [q.id for q in placed_before if body_of(q, cat).z1 == body.z0 and body_of(q, cat).footprint_overlaps(body)]
        out["added"].append({"id": p.id, "part": p.part_key, "x": p.x, "y": p.y, "z": p.z, "rotation_deg": p.rotation_deg,
                             "body": [body.x0, body.y0, body.z0, body.x1, body.y1, body.z1], "attaches_to": supporters or ["ground"]})
    out["stability_after_step"] = analyze(prefix, cat)
    return out


@mcp.tool()
def bom(build_path: str) -> list[dict]:
    """Bill of materials derived from stored placements."""
    cat = load_catalog()
    return [{"part_key": k, "ldraw_file": cat.part(k).ldraw_file, "color_id": c, "count": n} for k, c, n in bom_rows(_load(build_path))]


@mcp.tool()
def describe_build(build_path: str) -> dict:
    """Summary of a build: constraints, piece count, pieces per layer, footprint extents, generator provenance."""
    b = _load(build_path)
    cat = load_catalog()
    bodies = [body_of(p, cat) for p in b.placements]
    layers: dict[int, int] = {}
    for bd in bodies:
        layers[bd.z0 // 3] = layers.get(bd.z0 // 3, 0) + 1
    return {
        "build_id": b.build_id, "generator": b.generator.model_dump(), "constraints": b.constraints.model_dump(),
        "pieces": len(b.placements), "steps": len(b.steps), "pieces_per_layer": dict(sorted(layers.items())),
        "extent_studs": [max(x.x1 for x in bodies) - min(x.x0 for x in bodies), max(x.y1 for x in bodies) - min(x.y0 for x in bodies)] if bodies else [0, 0],
        "height_layers": (max(x.z1 for x in bodies) // 3) if bodies else 0,
    }


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
