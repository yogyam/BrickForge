"""Versioned part catalog loaded from catalog/<name>.json."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

CATALOG_DIR = Path(__file__).resolve().parent.parent / "catalog"


@dataclass(frozen=True)
class Part:
    key: str
    w: int  # native extent along x, in studs
    d: int  # native extent along y, in studs
    height_plates: int
    allowed_rotations: tuple[int, ...]
    ldraw_file: str
    cost: int

    def footprint(self, rotation_deg: int) -> tuple[int, int]:
        """Return (w, d) after rotation. 90 swaps the native extents."""
        if rotation_deg == 0:
            return self.w, self.d
        if rotation_deg == 90:
            return self.d, self.w
        raise ValueError(f"unsupported rotation {rotation_deg}")


@dataclass(frozen=True)
class Catalog:
    version: str
    coordinate_system: str
    parts: dict[str, Part]
    colors: dict[str, dict]
    supported_color_ids: frozenset[str]
    stud_pitch_mm: float
    plate_height_mm: float
    ldraw_stud_pitch_ldu: int
    ldraw_plate_height_ldu: int

    def part(self, key: str) -> Part:
        return self.parts[key]

    @property
    def brick_height_mm(self) -> float:
        return 3 * self.plate_height_mm


def load_catalog(name: str = "basic-bricks-v1") -> Catalog:
    path = CATALOG_DIR / f"{name}.json"
    raw = json.loads(path.read_text())
    if raw["catalog_version"] != name:
        raise ValueError(f"catalog_version mismatch in {path}")
    parts = {}
    for key, p in raw["parts"].items():
        parts[key] = Part(
            key=key,
            w=int(p["footprint"][0]),
            d=int(p["footprint"][1]),
            height_plates=int(p["height_plates"]),
            allowed_rotations=tuple(int(r) for r in p["allowed_rotations"]),
            ldraw_file=p["ldraw_file"],
            cost=int(p.get("cost", 1)),
        )
    units = raw["units"]
    return Catalog(
        version=raw["catalog_version"],
        coordinate_system=raw["coordinate_system"],
        parts=parts,
        colors=raw["colors"],
        supported_color_ids=frozenset(raw["part_defaults"]["supported_color_ids"]),
        stud_pitch_mm=float(units["stud_pitch_mm"]),
        plate_height_mm=float(units["plate_height_mm"]),
        ldraw_stud_pitch_ldu=int(units["ldraw_stud_pitch_ldu"]),
        ldraw_plate_height_ldu=int(units["ldraw_plate_height_ldu"]),
    )
