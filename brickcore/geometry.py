"""Integer-grid body geometry. x/y in studs, z in plates, z up. Bounds are half-open."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .catalog import Catalog
from .schema import BRICK_HEIGHT_PLATES, Placement


@dataclass(frozen=True)
class Body:
    x0: int
    y0: int
    z0: int
    x1: int
    y1: int
    z1: int

    def overlaps(self, other: "Body") -> bool:
        return (
            self.x0 < other.x1 and other.x0 < self.x1
            and self.y0 < other.y1 and other.y0 < self.y1
            and self.z0 < other.z1 and other.z0 < self.z1
        )

    def footprint_overlaps(self, other: "Body") -> bool:
        return self.x0 < other.x1 and other.x0 < self.x1 and self.y0 < other.y1 and other.y0 < self.y1

    @property
    def cells(self) -> list[tuple[int, int]]:
        return [(x, y) for x in range(self.x0, self.x1) for y in range(self.y0, self.y1)]


def body_of(p: Placement, catalog: Catalog) -> Body:
    part = catalog.part(p.part_key)
    w, d = part.footprint(p.rotation_deg)
    return Body(p.x, p.y, p.z, p.x + w, p.y + d, p.z + part.height_plates)


def stud_connected(upper: Body, lower: Body) -> bool:
    """True when `upper` rests directly on `lower` with footprint overlap. Side contact is not a connection."""
    return upper.z0 == lower.z1 and upper.footprint_overlaps(lower)


def occupancy_grid(placements: list[Placement], catalog: Catalog, nx: int, ny: int, layers: int) -> np.ndarray:
    """Boolean [nx, ny, layers] occupancy at one-brick vertical resolution. Out-of-bounds cells are ignored."""
    grid = np.zeros((nx, ny, layers), dtype=bool)
    for p in placements:
        b = body_of(p, catalog)
        k = b.z0 // BRICK_HEIGHT_PLATES
        if 0 <= k < layers:
            grid[max(b.x0, 0):min(b.x1, nx), max(b.y0, 0):min(b.y1, ny), k] = True
    return grid


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0
