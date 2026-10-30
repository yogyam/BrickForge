"""Bounded legal-placement enumeration shared by generators, the learned policy and the RL environment.

A candidate is an upright placement (part, rotation, x, y, layer) that is inside the grid, collision-free,
supported (layer 0, or at least one filled cell directly below) and covers at least `min_new_target` target
cells that are not yet filled. The same conventions are used at training and inference time.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .catalog import Catalog

PART_ORDER = ["brick_1x1", "brick_1x2", "brick_1x3", "brick_1x4", "brick_2x2", "brick_2x4"]
PART_INDEX = {k: i for i, k in enumerate(PART_ORDER)}


def window_sums(a: np.ndarray, w: int, d: int) -> np.ndarray:
    """Sum of a[x:x+w, y:y+d] for every anchor via a padded integral image. Shape (nx-w+1, ny-d+1)."""
    s = np.zeros((a.shape[0] + 1, a.shape[1] + 1), dtype=np.int32)
    s[1:, 1:] = a.astype(np.int32).cumsum(0).cumsum(1)
    return s[w:, d:] - s[:-w, d:] - s[w:, :-d] + s[:-w, :-d]


@dataclass
class Candidates:
    """Struct-of-arrays. All arrays have length n."""
    part: np.ndarray      # int, index into PART_ORDER
    rot: np.ndarray       # int, 0 or 90
    x: np.ndarray
    y: np.ndarray
    k: np.ndarray         # layer index (z = 3k)
    w: np.ndarray
    d: np.ndarray
    new_target: np.ndarray
    filler: np.ndarray
    support: np.ndarray   # filled cells directly below

    def __len__(self) -> int:
        return int(self.part.shape[0])

    def find(self, part_key: str, rot: int, x: int, y: int, k: int) -> int:
        m = (self.part == PART_INDEX[part_key]) & (self.rot == rot) & (self.x == x) & (self.y == y) & (self.k == k)
        idx = np.flatnonzero(m)
        return int(idx[0]) if len(idx) else -1

    def footprint_cells(self, i: int) -> list[tuple[int, int]]:
        return [(xx, yy) for xx in range(self.x[i], self.x[i] + self.w[i]) for yy in range(self.y[i], self.y[i] + self.d[i])]


def enumerate_candidates(target: np.ndarray, filled: np.ndarray, catalog: Catalog, min_new_target: int = 1,
                         max_filler_fraction: float = 0.5) -> Candidates:
    nx, ny, layers = target.shape
    cols = {f: [] for f in ("part", "rot", "x", "y", "k", "w", "d", "new_target", "filler", "support")}
    for k in range(layers):
        tk, fk = target[:, :, k], filled[:, :, k]
        if not tk.any():
            continue
        below = filled[:, :, k - 1] if k > 0 else None
        for key in PART_ORDER:
            part = catalog.part(key)
            for rot in part.allowed_rotations:
                w, d = part.footprint(rot)
                if w > nx or d > ny:
                    continue
                area = w * d
                new_t = window_sums(tk & ~fk, w, d)
                coll = window_sums(fk, w, d)
                filler = area - window_sums(tk, w, d)
                ok = (coll == 0) & (new_t >= min_new_target) & (filler <= int(max_filler_fraction * area))
                if below is not None:
                    sup = window_sums(below, w, d)
                    ok &= sup > 0
                else:
                    sup = np.zeros_like(new_t)
                xs, ys = np.nonzero(ok)
                if len(xs) == 0:
                    continue
                n = len(xs)
                cols["part"].append(np.full(n, PART_INDEX[key], dtype=np.int16))
                cols["rot"].append(np.full(n, rot, dtype=np.int16))
                cols["x"].append(xs.astype(np.int16))
                cols["y"].append(ys.astype(np.int16))
                cols["k"].append(np.full(n, k, dtype=np.int16))
                cols["w"].append(np.full(n, w, dtype=np.int16))
                cols["d"].append(np.full(n, d, dtype=np.int16))
                cols["new_target"].append(new_t[xs, ys].astype(np.int16))
                cols["filler"].append(filler[xs, ys].astype(np.int16))
                cols["support"].append(sup[xs, ys].astype(np.int16))
    if not cols["part"]:
        return Candidates(*(np.zeros(0, dtype=np.int16) for _ in range(10)))
    return Candidates(*(np.concatenate(cols[f]) for f in ("part", "rot", "x", "y", "k", "w", "d", "new_target", "filler", "support")))
