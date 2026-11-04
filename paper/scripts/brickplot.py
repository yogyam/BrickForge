"""Matplotlib renderer for brick builds and voxel targets (paper figures).

Every visible face is emitted as a unit-cell quad in ONE Poly3DCollection, so matplotlib's painter's sort (which only
works reliably for same-size lattice-aligned polygons) never punches a large face through a small one. Brick outlines
are 2-vertex polygons in the same collection, nudged toward the camera so they sort just in front of their faces.
"""
from __future__ import annotations

import math

import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from brickcore.catalog import Catalog
from brickcore.geometry import body_of
from brickcore.schema import BRICK_HEIGHT_PLATES, Placement

ELEV, AZIM = 28.0, -55.0
Z_ASPECT = 1.2  # brick height 9.6 mm over stud pitch 8 mm
SHADE = {"+z": 1.0, "+y": 0.80, "-y": 0.80, "+x": 0.62, "-x": 0.62}
# five print-safe hues (validated with the dataviz palette checker on a white surface); neighbouring bricks never share one
BRICK_PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e34948"]
TARGET_FACE = (0.16, 0.47, 0.84, 0.45)
GROUND = (0.90, 0.90, 0.88, 1.0)
EDGE = (0.08, 0.08, 0.08, 1.0)
EDGE_LW = 0.45


def _hex_to_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def owner_grid(placements: list[Placement], catalog: Catalog, grid: tuple[int, int, int]) -> np.ndarray:
    nx, ny, nz = grid
    own = -np.ones(grid, dtype=int)
    for i, p in enumerate(placements):
        b = body_of(p, catalog)
        k = b.z0 // BRICK_HEIGHT_PLATES
        if 0 <= k < nz:
            own[max(b.x0, 0):min(b.x1, nx), max(b.y0, 0):min(b.y1, ny), k] = i
    return own


def brick_colours(placements: list[Placement], catalog: Catalog, grid: tuple[int, int, int], palette: list[str] = BRICK_PALETTE) -> list[tuple]:
    """Deterministic greedy graph colouring: face-adjacent bricks (side contact in a layer, or stacked with footprint
    overlap) get different colours; ties are broken by placement order."""
    own = owner_grid(placements, catalog, grid)
    nx, ny, nz = grid
    adj: list[set[int]] = [set() for _ in placements]
    for (x, y, k), a in np.ndenumerate(own):
        if a < 0:
            continue
        for dx, dy, dk in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            xx, yy, kk = x + dx, y + dy, k + dk
            if xx < nx and yy < ny and kk < nz:
                b = own[xx, yy, kk]
                if b >= 0 and b != a:
                    adj[a].add(b)
                    adj[b].add(a)
    colour = [-1] * len(placements)
    for i in range(len(placements)):
        used = {colour[j] for j in adj[i] if colour[j] >= 0}
        colour[i] = next(c for c in range(len(palette) + 1) if c not in used) % len(palette)
    return [_hex_to_rgb(palette[c]) for c in colour]


def _visible_normals(elev: float = ELEV, azim: float = AZIM) -> list[tuple[int, int, int]]:
    a = math.radians(azim)
    normals = [(0, 0, 1 if elev >= 0 else -1)]
    cx, sy = math.cos(a), math.sin(a)
    if abs(cx) > 1e-9:
        normals.append((1 if cx > 0 else -1, 0, 0))
    if abs(sy) > 1e-9:
        normals.append((0, 1 if sy > 0 else -1, 0))
    return normals


def _camera_vector(elev: float = ELEV, azim: float = AZIM) -> np.ndarray:
    e, a = math.radians(elev), math.radians(azim)
    return np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])


def _face_quad(x: int, y: int, k: int, n: tuple[int, int, int]) -> np.ndarray:
    """Unit quad of cell (x, y, k) on the side with outward normal n, in (stud, stud, layer) units."""
    nxn, nyn, nzn = n
    if nzn:
        z = k + (1 if nzn > 0 else 0)
        return np.array([(x, y, z), (x + 1, y, z), (x + 1, y + 1, z), (x, y + 1, z)], dtype=float)
    if nxn:
        xx = x + (1 if nxn > 0 else 0)
        return np.array([(xx, y, k), (xx, y + 1, k), (xx, y + 1, k + 1), (xx, y, k + 1)], dtype=float)
    yy = y + (1 if nyn > 0 else 0)
    return np.array([(x, yy, k), (x + 1, yy, k), (x + 1, yy, k + 1), (x, yy, k + 1)], dtype=float)


def _in_plane_dirs(n: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    return [d for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)) if np.dot(d, n) == 0]


def _edge_of_quad(quad: np.ndarray, d: tuple[int, int, int]) -> np.ndarray:
    """The quad edge on the side of in-plane direction d (the two vertices extreme along d)."""
    proj = quad @ np.array(d, dtype=float)
    idx = np.argsort(proj)[-2:]
    return quad[sorted(idx)]


def _setup_axes(ax, grid: tuple[int, int, int]) -> None:
    nx, ny, nz = grid
    ax.set_proj_type("ortho")
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_zlim(0, nz)
    ax.set_box_aspect((nx, ny, nz * Z_ASPECT))
    ax.set_axis_off()


def draw_build(ax, placements: list[Placement], catalog: Catalog, grid: tuple[int, int, int], ground: bool = True) -> None:
    nx, ny, nz = grid
    own = owner_grid(placements, catalog, grid)
    cols = brick_colours(placements, catalog, grid)
    normals = _visible_normals()
    cam = _camera_vector() * 0.02
    polys, fcs, ecs, lws = [], [], [], []
    if ground:
        for x in range(nx):
            for y in range(ny):
                if own[x, y, 0] < 0:
                    polys.append(_face_quad(x, y, 0, (0, 0, 1)) - np.array([0, 0, 1.0]) + np.array([0, 0, 1.0]))
                    fcs.append(GROUND)
                    ecs.append((0.82, 0.82, 0.80, 1.0))
                    lws.append(0.15)
    for (x, y, k), a in np.ndenumerate(own):
        if a < 0:
            continue
        base = cols[a]
        for n in normals:
            xx, yy, kk = x + n[0], y + n[1], k + n[2]
            if 0 <= xx < nx and 0 <= yy < ny and 0 <= kk < nz and own[xx, yy, kk] >= 0:
                continue  # covered by another cell
            quad = _face_quad(x, y, k, n)
            key = "+z" if n[2] else ("+x" if n[0] else "+y")
            shade = SHADE[key]
            fc = (base[0] * shade, base[1] * shade, base[2] * shade, 1.0)
            polys.append(quad)
            fcs.append(fc)
            ecs.append(fc)
            lws.append(0.2)
            for d in _in_plane_dirs(n):
                px, py, pk = x + d[0], y + d[1], k + d[2]
                inside = 0 <= px < nx and 0 <= py < ny and 0 <= pk < nz
                if inside and own[px, py, pk] == a:
                    continue  # same brick continues across this edge
                polys.append(_edge_of_quad(quad, d) + cam)
                fcs.append((0, 0, 0, 0))
                ecs.append(EDGE)
                lws.append(EDGE_LW)
    coll = Poly3DCollection(polys, facecolors=fcs, edgecolors=ecs, linewidths=lws, zsort="average")
    ax.add_collection3d(coll)
    _setup_axes(ax, grid)


def draw_target(ax, target: np.ndarray, ground: bool = True) -> None:
    """Voxel target drawn with the same single-collection quad emitter as the bricks, so the ground plane and the
    cells depth-sort together. Every exposed cell face gets a light edge so the voxel grid stays readable."""
    grid = tuple(int(v) for v in target.shape)
    nx, ny, nz = grid
    normals = _visible_normals()
    cam = _camera_vector() * 0.02
    polys, fcs, ecs, lws = [], [], [], []
    if ground:
        for x in range(nx):
            for y in range(ny):
                if not target[x, y, 0]:
                    polys.append(_face_quad(x, y, 0, (0, 0, 1)) - np.array([0, 0, 1.0]))
                    fcs.append(GROUND)
                    ecs.append((0.82, 0.82, 0.80, 1.0))
                    lws.append(0.15)
    base = TARGET_FACE[:3]
    for (x, y, k), filled in np.ndenumerate(target):
        if not filled:
            continue
        for n in normals:
            xx, yy, kk = x + n[0], y + n[1], k + n[2]
            if 0 <= xx < nx and 0 <= yy < ny and 0 <= kk < nz and target[xx, yy, kk]:
                continue
            quad = _face_quad(x, y, k, n)
            shade = SHADE["+z" if n[2] else ("+x" if n[0] else "+y")]
            polys.append(quad)
            fcs.append((base[0] * shade, base[1] * shade, base[2] * shade, 0.9))
            ecs.append((1, 1, 1, 0.5))
            lws.append(0.25)
    coll = Poly3DCollection(polys, facecolors=fcs, edgecolors=ecs, linewidths=lws, zsort="average")
    ax.add_collection3d(coll)
    _ = cam
    _setup_axes(ax, grid)
