"""Deterministic greedy baseline (greedy_v1). Layer-by-layer, bottom-up, stud-connected placements.

Canonical generator interface:
    generate(target, catalog, constraints, seed, cancellation) -> CandidateResult
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .catalog import Catalog
from .geometry import iou, occupancy_grid
from .plan import plan_assembly
from .schema import BRICK_HEIGHT_PLATES, Build, Constraints, GeneratorInfo, Placement
from .validate import components, validate_build

GENERATOR_NAME = "greedy_v1"
GENERATOR_VERSION = "0.1.0"


class Cancelled(Exception):
    pass


@dataclass
class Strategy:
    name: str
    min_target_fraction: float  # fraction of the footprint that must be new target cells
    allow_filler: bool          # permit cells outside the target (penalized)


STRATEGIES = [
    Strategy("exact", 1.0, False),
    Strategy("mostly", 0.75, True),
    Strategy("half", 0.5, True),
]


@dataclass
class CandidateResult:
    status: str
    build: Build | None
    fidelity: float
    coverage: float
    extraneous: float
    diagnostics: dict = field(default_factory=dict)
    instruction_status: str = "none"
    plan: dict = field(default_factory=dict)


def _window_sums(a: np.ndarray, w: int, d: int) -> np.ndarray:
    """Sum of a[x:x+w, y:y+d] for every anchor, via a padded 2D integral image."""
    s = np.zeros((a.shape[0] + 1, a.shape[1] + 1), dtype=np.int32)
    s[1:, 1:] = a.astype(np.int32).cumsum(0).cumsum(1)
    return s[w:, d:] - s[:-w, d:] - s[w:, :-d] + s[:-w, :-d]


def _generate_with_strategy(target: np.ndarray, catalog: Catalog, constraints: Constraints, strategy: Strategy,
                            cancel: Callable[[], bool] | None) -> tuple[list[Placement], dict]:
    nx, ny, layers = target.shape
    filled = np.zeros_like(target)
    owner = np.full(target.shape, -1, dtype=np.int32)
    placements: list[Placement] = []
    cap = constraints.max_pieces
    parts = sorted(catalog.parts.values(), key=lambda p: (-(p.w * p.d), p.key))
    stalls = 0
    for k in range(layers):
        if not target[:, :, k].any():
            continue
        while len(placements) < cap:
            if cancel and cancel():
                raise Cancelled()
            best = None
            for part in parts:
                for rot in part.allowed_rotations:
                    w, d = part.footprint(rot)
                    if w > nx or d > ny:
                        continue
                    area = w * d
                    new_t = _window_sums(target[:, :, k] & ~filled[:, :, k], w, d)
                    coll = _window_sums(filled[:, :, k], w, d)
                    filler = area - _window_sums(target[:, :, k], w, d)
                    ok = (coll == 0) & (new_t >= max(1, int(np.ceil(strategy.min_target_fraction * area))))
                    if not strategy.allow_filler:
                        ok &= filler == 0
                    if k > 0:
                        support = _window_sums(filled[:, :, k - 1], w, d)
                        ok &= support > 0
                        # "Hard" cells have no brick beneath them and can only be reached by a brick that
                        # also overlaps the layer below. Cover them first (most constrained first).
                        hard = _window_sums(target[:, :, k] & ~filled[:, :, k] & ~filled[:, :, k - 1], w, d)
                    else:
                        support = np.zeros_like(new_t)
                        hard = np.zeros_like(new_t)
                    if not ok.any():
                        continue
                    # Vectorized primary key, then exact distinct-lower-brick count on the short list.
                    score = np.where(ok, hard * 1_000_000 + new_t * 10_000 - filler * 100 + np.minimum(support, 99), -1)
                    top = int(score.max())
                    for x, y in np.argwhere(score == top):
                        x, y = int(x), int(y)
                        lower = set(owner[x:x + w, y:y + d, k - 1].ravel().tolist()) - {-1} if k > 0 else set()
                        key = (-int(hard[x, y]), -int(new_t[x, y]), int(filler[x, y]), -len(lower), part.cost, part.key, x, y, rot)
                        if best is None or key < best[0]:
                            best = (key, part, rot, x, y, w, d)
            if best is None:
                break
            _, part, rot, x, y, w, d = best
            pid = f"b{len(placements) + 1:04d}"
            placements.append(Placement(id=pid, part_key=part.key, x=x, y=y, z=k * BRICK_HEIGHT_PLATES, rotation_deg=rot, color_id=constraints.color_id))
            filled[x:x + w, y:y + d, k] = True
            owner[x:x + w, y:y + d, k] = len(placements) - 1
        if len(placements) >= cap:
            stalls = layers - k - 1
            break
    return placements, {"layers_unreached_due_to_budget": stalls}


def _largest_ground_component(placements: list[Placement], catalog: Catalog) -> tuple[list[Placement], int]:
    from .geometry import body_of, stud_connected
    bodies = {p.id: body_of(p, catalog) for p in placements}
    ids = list(bodies)
    edges = [(a, b) for a in ids for b in ids if a < b and (stud_connected(bodies[a], bodies[b]) or stud_connected(bodies[b], bodies[a]))]
    comps = components(ids, edges)
    grounded = [c for c in comps if any(bodies[i].z0 == 0 for i in c)]
    if not grounded:
        return [], len(comps)
    keep = max(grounded, key=lambda c: (len(c), -min(int(i[1:]) for i in c)))
    return [p for p in placements if p.id in keep], len(comps)


def generate(target: np.ndarray, catalog: Catalog, constraints: Constraints, seed: int = 0,
             cancellation: Callable[[], bool] | None = None, build_id: str = "build",
             time_budget_s: float | None = None) -> CandidateResult:
    """Run every strategy, validate each, and return the best valid candidate by IoU.
    Seed is recorded for provenance; greedy_v1 is fully deterministic and does not consume it.
    A wall-clock budget or external cancellation stops the search; the best candidate found so far is returned
    and the abort is recorded in diagnostics. Never relaxes the cap or validation."""
    t0 = time.perf_counter()
    nx, ny, layers = target.shape
    target_cells = int(target.sum())
    tried = []
    best: CandidateResult | None = None
    aborted = None

    def cancel() -> bool:
        if cancellation and cancellation():
            return True
        return time_budget_s is not None and (time.perf_counter() - t0) > time_budget_s

    for strat in STRATEGIES:
        ts = time.perf_counter()
        try:
            placements, diag = _generate_with_strategy(target, catalog, constraints, strat, cancel)
        except Cancelled:
            aborted = "time_budget" if time_budget_s is not None and (time.perf_counter() - t0) > time_budget_s else "cancelled"
            tried.append({"strategy": strat.name, "aborted": aborted, "seconds": round(time.perf_counter() - ts, 3)})
            break
        kept, n_comps = _largest_ground_component(placements, catalog)
        dropped = len(placements) - len(kept)
        # Renumber so ids stay dense and steps are the retained generation order.
        kept = [Placement(id=f"b{i + 1:04d}", part_key=p.part_key, x=p.x, y=p.y, z=p.z, rotation_deg=p.rotation_deg, color_id=p.color_id) for i, p in enumerate(kept)]
        plan = plan_assembly(kept, catalog)
        build = Build(
            build_id=f"{build_id}_{strat.name}",
            catalog_version=catalog.version,
            generator=GeneratorInfo(name=GENERATOR_NAME, version=GENERATOR_VERSION, seed=seed, strategy=strat.name),
            constraints=constraints,
            placements=kept,
            steps=plan.steps,  # empty when planning did not succeed: no invented instructions
        )
        report = validate_build(build, catalog)
        occ = occupancy_grid(kept, catalog, nx, ny, layers)
        inter = int(np.logical_and(occ, target).sum())
        gen_cells = int(occ.sum())
        coverage = inter / target_cells if target_cells else 0.0
        extraneous = (gen_cells - inter) / gen_cells if gen_cells else 0.0
        fid = iou(occ, target)
        entry = {"strategy": strat.name, "pieces": len(kept), "dropped_disconnected": dropped, "components_before_trim": n_comps,
                 "valid": report["ok"], "iou": round(fid, 4), "coverage": round(coverage, 4), "extraneous": round(extraneous, 4),
                 "instruction_status": plan.status, "seconds": round(time.perf_counter() - ts, 3), **diag}
        if plan.status != "planned":
            entry["plan_diagnostics"] = plan.diagnostics
        if not report["ok"]:
            entry["errors"] = [e["code"] for e in report["errors"]]
        tried.append(entry)
        if report["ok"] and kept and (best is None or fid > best.fidelity):
            best = CandidateResult("validated_candidate", build, fid, coverage, extraneous, instruction_status=plan.status, plan=plan.diagnostics)
    diagnostics = {"strategies": tried, "target_cells": target_cells, "grid": [nx, ny, layers], "seconds": round(time.perf_counter() - t0, 3)}
    if aborted:
        diagnostics["aborted"] = aborted
        diagnostics["strategies_not_run"] = [st.name for st in STRATEGIES if st.name not in {t["strategy"] for t in tried}]
    if best is None:
        warnings = []
        if target_cells == 0:
            warnings.append("target occupies zero grid cells at this scale; increase the target size or use a thicker mesh")
        if aborted:
            warnings.append(f"search {aborted} before any valid candidate was found")
        if warnings:
            diagnostics["warnings"] = warnings
        return CandidateResult("no_valid_candidate", None, 0.0, 0.0, 0.0, diagnostics)
    best.diagnostics = diagnostics
    if best.coverage < 0.8 or any(s["dropped_disconnected"] for s in tried if s["strategy"] == best.build.generator.strategy):
        best.status = "partial_candidate"
        diagnostics["warnings"] = [f"coverage {best.coverage:.3f} below 0.80" if best.coverage < 0.8 else "disconnected pieces were dropped"]
    return best
