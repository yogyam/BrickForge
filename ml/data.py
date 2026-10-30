"""Training data for the learned generator.

Targets are occupancy grids. Sources: (a) procedural voxel families generated directly on the grid,
(b) a Thingi10K training pool with every held-out Thingiverse thing excluded. Supervision comes from the
deterministic baseline: its validated build, ordered by the planner, is the teacher action sequence.
Split is by structure instance (all symmetries of one instance share a split), 80/10/10.
The learned model therefore imitates the baseline's style on these shapes; see docs/sft.md for the gap.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from brickcore.catalog import Catalog, load_catalog
from brickcore.generate import generate as greedy_generate
from brickcore.schema import Bounds, Constraints, Placement

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "sft"
GRID = (16, 16, 8)


# ----------------------------------------------------------------------------- synthetic families
def _box(occ, x0, y0, k0, w, d, h):
    nx, ny, nz = occ.shape
    occ[max(x0, 0):min(x0 + w, nx), max(y0, 0):min(y0 + d, ny), max(k0, 0):min(k0 + h, nz)] = True


def _ellipsoid(occ, cx, cy, ck, rx, ry, rk):
    nx, ny, nz = occ.shape
    X, Y, K = np.meshgrid(np.arange(nx) + 0.5, np.arange(ny) + 0.5, np.arange(nz) + 0.5, indexing="ij")
    occ |= ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2 + ((K - ck) / rk) ** 2 <= 1.0


def _ground_and_connect(occ: np.ndarray) -> np.ndarray:
    """Drop to the ground and keep the largest face-connected component."""
    if not occ.any():
        return occ
    ks = np.nonzero(occ.any(axis=(0, 1)))[0]
    if ks[0] > 0:
        occ = np.roll(occ, -ks[0], axis=2)
        occ[:, :, occ.shape[2] - ks[0]:] = False
    # largest component via BFS
    nx, ny, nz = occ.shape
    seen = np.zeros_like(occ)
    best = None
    for start in zip(*np.nonzero(occ), strict=True):
        if seen[start]:
            continue
        comp, stack = [], [start]
        seen[start] = True
        while stack:
            x, y, k = stack.pop()
            comp.append((x, y, k))
            for dx, dy, dk in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                nxx, nyy, nkk = x + dx, y + dy, k + dk
                if 0 <= nxx < nx and 0 <= nyy < ny and 0 <= nkk < nz and occ[nxx, nyy, nkk] and not seen[nxx, nyy, nkk]:
                    seen[nxx, nyy, nkk] = True
                    stack.append((nxx, nyy, nkk))
        if best is None or len(comp) > len(best):
            best = comp
    out = np.zeros_like(occ)
    for c in best:
        out[c] = True
    return out


FAMILIES = ["box", "stack", "capped_tower", "bridge", "polyomino", "ellipsoid", "cylinder", "stepped", "blob", "arch_row"]


def synth_target(family: str, rng: np.random.Generator, grid=GRID) -> np.ndarray:
    nx, ny, nz = grid
    occ = np.zeros(grid, dtype=bool)
    if family == "box":
        w, d, h = rng.integers(2, 13), rng.integers(2, 13), rng.integers(1, nz + 1)
        _box(occ, (nx - w) // 2 + rng.integers(-2, 3), (ny - d) // 2 + rng.integers(-2, 3), 0, w, d, h)
    elif family == "stack":
        x, y, k = rng.integers(1, 6), rng.integers(1, 6), 0
        w, d = rng.integers(5, 12), rng.integers(5, 12)
        while k < nz and w >= 1 and d >= 1:
            h = rng.integers(1, 4)
            _box(occ, x, y, k, w, d, h)
            k += h
            x += rng.integers(-1, 3)
            y += rng.integers(-1, 3)
            w -= rng.integers(0, 3)
            d -= rng.integers(0, 3)
    elif family == "capped_tower":
        w, d = rng.integers(2, 5), rng.integers(2, 5)
        x, y = (nx - w) // 2, (ny - d) // 2
        h = rng.integers(3, nz)
        _box(occ, x, y, 0, w, d, h)
        cap = rng.integers(1, 3)
        _box(occ, x - cap, y - cap, h, w + 2 * cap, d + 2 * cap, rng.integers(1, min(3, nz - h) + 1))
    elif family == "bridge":
        pw = rng.integers(2, 4)
        gap = rng.integers(2, 8)
        depth = rng.integers(2, 6)
        x0 = (nx - (2 * pw + gap)) // 2
        y0 = (ny - depth) // 2
        ph = rng.integers(2, nz - 1)
        span_h = rng.integers(1, min(3, nz - ph) + 1)
        _box(occ, x0, y0, 0, pw, depth, ph)
        _box(occ, x0 + pw + gap, y0, 0, pw, depth, ph)
        _box(occ, x0, y0, ph, 2 * pw + gap, depth, span_h)
    elif family == "polyomino":
        cells = {(nx // 2, ny // 2)}
        for _ in range(rng.integers(2, 6)):
            cx, cy = list(cells)[rng.integers(len(cells))]
            w, d = rng.integers(2, 7), rng.integers(2, 7)
            cells |= {(cx + i, cy + j) for i in range(-rng.integers(0, w), w) for j in range(-rng.integers(0, d), d)}
        h = rng.integers(1, 5)
        for (x, y) in cells:
            if 0 <= x < nx and 0 <= y < ny:
                occ[x, y, :h] = True
    elif family == "ellipsoid":
        rx, ry, rk = rng.uniform(2, 6.5), rng.uniform(2, 6.5), rng.uniform(1.5, 4)
        _ellipsoid(occ, nx / 2 + rng.uniform(-1, 1), ny / 2 + rng.uniform(-1, 1), rk, rx, ry, rk)
    elif family == "cylinder":
        r = rng.uniform(2, 6.5)
        h = rng.integers(2, nz + 1)
        X, Y = np.meshgrid(np.arange(nx) + 0.5, np.arange(ny) + 0.5, indexing="ij")
        disc = (X - nx / 2) ** 2 + (Y - ny / 2) ** 2 <= r * r
        occ[:, :, :h] = disc[:, :, None]
    elif family == "stepped":
        w, d = rng.integers(8, 15), rng.integers(8, 15)
        x, y = (nx - w) // 2, (ny - d) // 2
        k = 0
        while w > 0 and d > 0 and k < nz:
            _box(occ, x, y, k, w, d, 1)
            s = rng.integers(1, 3)
            x += s
            y += s
            w -= 2 * s
            d -= 2 * s
            k += 1
    elif family == "blob":
        x, y, k = nx // 2, ny // 2, 0
        for _ in range(rng.integers(4, 12)):
            _box(occ, x - rng.integers(1, 3), y - rng.integers(1, 3), max(k - rng.integers(0, 2), 0), rng.integers(2, 6), rng.integers(2, 6), rng.integers(1, 4))
            x = int(np.clip(x + rng.integers(-3, 4), 1, nx - 2))
            y = int(np.clip(y + rng.integers(-3, 4), 1, ny - 2))
            k = int(np.clip(k + rng.integers(-1, 3), 0, nz - 1))
    elif family == "arch_row":
        n = rng.integers(2, 4)
        pw, gap, depth = 2, rng.integers(2, 4), rng.integers(2, 5)
        total = n * gap + (n + 1) * pw
        x0 = max((nx - total) // 2, 0)
        ph = rng.integers(2, nz - 1)
        for i in range(n + 1):
            _box(occ, x0 + i * (pw + gap), (ny - depth) // 2, 0, pw, depth, ph)
        _box(occ, x0, (ny - depth) // 2, ph, total, depth, rng.integers(1, min(3, nz - ph) + 1))
    else:
        raise ValueError(family)
    return _ground_and_connect(occ)


# ----------------------------------------------------------------------------- symmetries
def apply_symmetry(target: np.ndarray, placements: list[Placement], sym: int, catalog: Catalog) -> tuple[np.ndarray, list[Placement]]:
    """sym in 0..7: rotation by 90*(sym%4) about z, then x-flip if sym>=4. Keeps builds valid."""
    nx, ny = target.shape[:2]
    t = target
    pl = placements
    for _ in range(sym % 4):
        t = np.rot90(t, k=1, axes=(0, 1))  # (x, y) -> (ny-1-y, x)
        new = []
        for p in pl:
            w, d = catalog.part(p.part_key).footprint(p.rotation_deg)
            nxp, nyp = ny - (p.y + d), p.x
            rot = 90 - p.rotation_deg if 90 in catalog.part(p.part_key).allowed_rotations else 0
            new.append(Placement(id=p.id, part_key=p.part_key, x=int(nxp), y=int(nyp), z=p.z, rotation_deg=rot, color_id=p.color_id))
        pl = new
        nx, ny = ny, nx
    if sym >= 4:
        t = t[::-1, :, :].copy()
        new = []
        for p in pl:
            w, d = catalog.part(p.part_key).footprint(p.rotation_deg)
            new.append(Placement(id=p.id, part_key=p.part_key, x=int(nx - (p.x + w)), y=p.y, z=p.z, rotation_deg=p.rotation_deg, color_id=p.color_id))
        pl = new
    return np.ascontiguousarray(t), pl


# ----------------------------------------------------------------------------- examples
@dataclass
class Example:
    id: str
    family: str
    split: str
    grid: tuple[int, int, int]
    target_cells: list[list[int]]
    placements: list[dict]      # in planned order
    iou: float
    budget: int


def _split_for(instance_id: str) -> str:
    h = int(hashlib.sha256(instance_id.encode()).hexdigest()[:8], 16) % 100
    return "train" if h < 80 else ("val" if h < 90 else "test")


def make_example(ex_id: str, family: str, target: np.ndarray, catalog: Catalog, budget: int = 100) -> Example | None:
    bounds = Bounds(x_studs=target.shape[0], y_studs=target.shape[1], z_plates=target.shape[2] * 3)
    res = greedy_generate(target, catalog, Constraints(max_pieces=budget, bounds=bounds, color_id="4"), build_id=ex_id)
    if res.build is None or res.instruction_status != "planned" or res.coverage < 0.5:
        return None
    by_id = {p.id: p for p in res.build.placements}
    ordered = [by_id[i] for s in res.build.steps for i in s.added_placement_ids]
    return Example(ex_id, family, _split_for(ex_id), tuple(target.shape), [list(map(int, c)) for c in np.argwhere(target)],
                   [p.model_dump() for p in ordered], round(res.fidelity, 4), budget)


def build_synthetic(n_per_family: int, seed: int = 0, grid=GRID, catalog: Catalog | None = None, min_cells: int = 8, max_cells: int = 700) -> list[Example]:
    catalog = catalog or load_catalog()
    out = []
    for fam in FAMILIES:
        made = 0
        tries = 0
        while made < n_per_family and tries < n_per_family * 5:
            tries += 1
            rng = np.random.default_rng([seed, FAMILIES.index(fam), tries])
            t = synth_target(fam, rng, grid)
            if not (min_cells <= int(t.sum()) <= max_cells):
                continue
            ex = make_example(f"synth_{fam}_{seed}_{tries}", fam, t, catalog)
            if ex is not None:
                out.append(ex)
                made += 1
    return out


def build_thingi_pool(n: int, catalog: Catalog | None = None, grid=GRID, max_faces: int = 50_000, progress: bool = True) -> list[Example]:
    """Download (if needed) and voxelize Thingi10K training meshes, excluding every held-out thing id."""
    from brickcore.mesh import MeshError, load_mesh, normalize, voxelize

    from . import thingi10k as T
    catalog = catalog or load_catalog()
    meta = T.load_metadata()
    pool = T.training_pool(meta, T.held_out_thing_ids(), max_faces=max_faces)
    rng = np.random.default_rng(0)
    rng.shuffle(pool)
    mesh_dir = T.DATA_DIR / "train_meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    bounds = Bounds(x_studs=grid[0], y_studs=grid[1], z_plates=grid[2] * 3)
    out, tried = [], 0
    for m in pool:
        if len(out) >= n:
            break
        tried += 1
        local = mesh_dir / f"{m['file_id']}.stl"
        try:
            if not local.exists():
                local.write_bytes(T._fetch(f"raw_meshes/{m['file_id']}.stl", timeout=120))
            mesh, _ = load_mesh(local)
            mm, _ = normalize(mesh, bounds, catalog)
            t = voxelize(mm, bounds, catalog)
        except (MeshError, Exception) as e:  # noqa: BLE001
            if progress:
                print(f"skip {m['file_id']}: {type(e).__name__}", file=sys.stderr)
            continue
        if not (8 <= int(t.sum()) <= 700):
            continue
        ex = make_example(f"thingi_{m['thing_id']}_{m['file_id']}", "thingi", t, catalog)
        if ex is not None:
            ex.split = _split_for(f"thing_{m['thing_id']}")  # all files of one thing share a split
            out.append(ex)
        if progress and tried % 25 == 0:
            print(f"thingi pool: tried {tried}, kept {len(out)}", file=sys.stderr)
    return out


def save_examples(examples: list[Example], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for e in examples:
            fh.write(json.dumps(e.__dict__) + "\n")


def load_examples(path: Path, split: str | None = None) -> list[Example]:
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        d["grid"] = tuple(d["grid"])
        if split is None or d["split"] == split:
            out.append(Example(**d))
    return out


def target_from_example(e: Example) -> np.ndarray:
    t = np.zeros(e.grid, dtype=bool)
    for x, y, k in e.target_cells:
        t[x, y, k] = True
    return t


def placements_from_example(e: Example) -> list[Placement]:
    return [Placement(**p) for p in e.placements]


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="build_sft_data")
    ap.add_argument("--synthetic-per-family", type=int, default=200)
    ap.add_argument("--thingi", type=int, default=0, help="number of Thingi10K training meshes to add (downloads)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(DATA_DIR / "examples.jsonl"))
    args = ap.parse_args(argv)
    catalog = load_catalog()
    t0 = __import__("time").perf_counter()
    ex = build_synthetic(args.synthetic_per_family, seed=args.seed, catalog=catalog)
    print(f"synthetic examples: {len(ex)} in {__import__('time').perf_counter() - t0:.1f}s", file=sys.stderr)
    if args.thingi:
        ex += build_thingi_pool(args.thingi, catalog=catalog)
    save_examples(ex, Path(args.out))
    from collections import Counter
    print(json.dumps({"examples": len(ex), "by_split": dict(Counter(e.split for e in ex)), "by_family": dict(Counter(e.family for e in ex)),
                      "mean_pieces": round(sum(len(e.placements) for e in ex) / max(len(ex), 1), 1), "mean_teacher_iou": round(sum(e.iou for e in ex) / max(len(ex), 1), 4)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
