"""Procedural fixture meshes with known expected behaviour. Written as binary STL (mm units, z up)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from shapely.geometry import Polygon

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "meshes"


def _box(x, y, z):
    return trimesh.creation.box(extents=(x, y, z))


def _extrude(points, height):
    return trimesh.creation.extrude_polygon(Polygon(points), height)


def make_fixtures() -> dict[str, trimesh.Trimesh]:
    f: dict[str, trimesh.Trimesh] = {}
    f["cube"] = _box(64, 64, 38.4)                       # 8x8 studs, 4 bricks: trivially exact
    f["slab"] = _box(96, 64, 19.2)                       # 12x8 studs, 2 bricks tall after fit
    # Flat sheet: one layer of side-by-side bricks has no stud connections -> cannot be one assembly.
    f["flat_sheet_impossible"] = _box(96, 64, 9.6)
    f["tower"] = _box(16, 16, 76.8)                      # 2x2 studs, 8 bricks tall
    f["l_shape"] = _extrude([(0, 0), (96, 0), (96, 32), (32, 32), (32, 96), (0, 96)], 28.8)
    f["t_shape"] = _extrude([(0, 64), (96, 64), (96, 96), (64, 96), (64, 0), (32, 0), (32, 96), (0, 96)], 19.2)
    # Bridge: two pillars joined by a span on top. Extruded along y so it is a single watertight solid.
    bridge = _extrude([(0, 0), (16, 0), (16, 48), (80, 48), (80, 0), (96, 0), (96, 76.8), (0, 76.8)], 32)
    bridge.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))  # profile in xz, depth along -y
    f["bridge"] = bridge
    f["sphere"] = trimesh.creation.icosphere(subdivisions=3, radius=50)
    f["cylinder"] = trimesh.creation.cylinder(radius=24, height=48, sections=48)
    f["pyramid"] = trimesh.creation.cone(radius=64, height=76.8, sections=4)
    f["wedge"] = trimesh.creation.extrude_triangulation(
        vertices=np.array([[0, 0], [96, 0], [0, 48]], dtype=float), faces=np.array([[0, 1, 2]]), height=19.2)
    # Thin pole: too thin at the chosen scale to leave a connected column -> expected partial/fail diagnostics.
    f["thin_pole"] = _box(4, 4, 120)
    # Open box: one face removed -> non-watertight -> intentional input rejection.
    open_box = _box(40, 40, 40)
    open_box = trimesh.Trimesh(vertices=open_box.vertices, faces=open_box.faces[:-2], process=False)
    f["open_box_invalid"] = open_box
    for m in f.values():
        if m.is_watertight and m.volume < 0:
            m.invert()
    return f


def write_fixtures(out_dir: Path = FIXTURE_DIR) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, m in make_fixtures().items():
        p = out_dir / f"{name}.stl"
        m.export(p, file_type="stl")
        paths.append(p)
    return paths
