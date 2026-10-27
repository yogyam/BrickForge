"""Tested transforms from canonical grid coordinates to LDraw and Three.js scene coordinates.

Canonical: x, y horizontal in studs; z up in plates; right-handed.
LDraw:     1 LDU = 0.4 mm; stud pitch 20 LDU; plate 8 LDU; -Y is up; right-handed.
           Mapping: X = x, Y = -z, Z = y (then scaled). Brick parts have their origin at the
           top-center of the body and extend +24 LDU downward (towards +Y).
Three.js:  mm units; Y up; right-handed. Mapping: X = x, Y = z, Z = -y (then scaled).
Never copy canonical integers into either renderer directly.
"""
from __future__ import annotations

from dataclasses import dataclass

from .catalog import Catalog
from .geometry import Body, body_of
from .schema import Placement

IDENTITY = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
# 90 degrees about the LDraw vertical (Y) axis: p' = (p.z, p.y, -p.x)
ROT90_Y = ((0, 0, 1), (0, 1, 0), (-1, 0, 0))


@dataclass(frozen=True)
class LDrawPlacement:
    color: str
    x: float
    y: float
    z: float
    matrix: tuple[tuple[int, int, int], ...]
    file: str

    def line(self) -> str:
        m = " ".join(str(v) for row in self.matrix for v in row)
        return f"1 {self.color} {_fmt(self.x)} {_fmt(self.y)} {_fmt(self.z)} {m} {self.file}"


def _fmt(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else f"{v:g}"


def to_ldraw(p: Placement, catalog: Catalog) -> LDrawPlacement:
    part = catalog.part(p.part_key)
    w, d = part.footprint(p.rotation_deg)
    s = catalog.ldraw_stud_pitch_ldu
    h = catalog.ldraw_plate_height_ldu
    cx = (p.x + w / 2) * s
    cz = (p.y + d / 2) * s
    top_y = -(p.z + part.height_plates) * h
    matrix = IDENTITY if p.rotation_deg == 0 else ROT90_Y
    return LDrawPlacement(p.color_id, cx, top_y, cz, matrix, part.ldraw_file)


def ldraw_body_bounds(lp: LDrawPlacement, p: Placement, catalog: Catalog) -> Body:
    """Push the part's native LDraw extents through its matrix and translation, then map back to
    canonical integer bounds. Used by tests to prove the export transform agrees with body_of()."""
    part = catalog.part(p.part_key)
    s = catalog.ldraw_stud_pitch_ldu
    h = catalog.ldraw_plate_height_ldu
    hx, hz = part.w * s / 2, part.d * s / 2
    corners = [(sx * hx, y, sz * hz) for sx in (-1, 1) for sz in (-1, 1) for y in (0, part.height_plates * h)]
    m = lp.matrix
    pts = []
    for px, py, pz in corners:
        X = m[0][0] * px + m[0][1] * py + m[0][2] * pz + lp.x
        Y = m[1][0] * px + m[1][1] * py + m[1][2] * pz + lp.y
        Z = m[2][0] * px + m[2][1] * py + m[2][2] * pz + lp.z
        pts.append((X, Y, Z))
    xs, ys, zs = zip(*pts)
    x0, x1 = min(xs) / s, max(xs) / s
    y0, y1 = min(zs) / s, max(zs) / s
    z0, z1 = -max(ys) / h, -min(ys) / h
    vals = (x0, y0, z0, x1, y1, z1)
    if any(not float(v).is_integer() for v in vals):
        raise ValueError(f"non-integer LDraw bounds {vals}")
    return Body(*(int(v) for v in vals))


@dataclass(frozen=True)
class SceneBox:
    """Axis-aligned box in Three.js scene millimetres (Y up)."""
    cx: float
    cy: float
    cz: float
    sx: float
    sy: float
    sz: float


def to_scene(p: Placement, catalog: Catalog) -> SceneBox:
    b = body_of(p, catalog)
    sp, ph = catalog.stud_pitch_mm, catalog.plate_height_mm
    return SceneBox(
        cx=(b.x0 + b.x1) / 2 * sp,
        cy=(b.z0 + b.z1) / 2 * ph,
        cz=-(b.y0 + b.y1) / 2 * sp,
        sx=(b.x1 - b.x0) * sp,
        sy=(b.z1 - b.z0) * ph,
        sz=(b.y1 - b.y0) * sp,
    )


def scene_to_body(box: SceneBox, catalog: Catalog) -> Body:
    sp, ph = catalog.stud_pitch_mm, catalog.plate_height_mm
    vals = (
        (box.cx - box.sx / 2) / sp, (-box.cz - box.sz / 2) / sp, (box.cy - box.sy / 2) / ph,
        (box.cx + box.sx / 2) / sp, (-box.cz + box.sz / 2) / sp, (box.cy + box.sy / 2) / ph,
    )
    return Body(*(int(round(v)) for v in vals))
