"""Assembly planner (M2). One piece per step, upright insertion from above, horizontal work surface.

A piece may be added when
  (a) it does not collide with placed bodies,
  (b) it attaches to the assembled component: it touches the ground or rests on a placed piece via studs,
  (c) its downward insertion path is clear: no placed body overlaps its footprint anywhere above it,
  (d) it does not seal off an unplaced piece: no *unplaced* body lies below it with footprint overlap,
      because that piece could then never be inserted from above (an exact dead-end test).

Completeness under these rules: if any full order exists, the greedy never gets stuck. Proof sketch: let S be
the placed set and O a complete valid order; the first piece r of O not in S has its supporter earlier in O,
hence in S (attachable), and every piece below r with footprint overlap precedes r in O, hence is in S (safe).
So backtracking is unnecessary; a node budget is kept only to honour the planning_unknown contract.

Planner outcomes: `planned`, `planning_failed` (no order exists under these rules; the remaining pieces are
listed), `planning_unknown` (budget exhausted). planning_failed is NOT a proof of physical unbuildability.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .catalog import Catalog
from .geometry import Body, body_of, stud_connected
from .schema import Placement, Step


@dataclass
class PlanResult:
    status: str
    order: list[str]
    steps: list[Step]
    diagnostics: dict = field(default_factory=dict)


def path_blocked(b: Body, placed: list[Body]) -> bool:
    """True when some placed body sits above `b` within its footprint, blocking top-down insertion."""
    return any(q.z0 >= b.z1 and q.footprint_overlaps(b) for q in placed)


def plan_assembly(placements: list[Placement], catalog: Catalog, node_budget: int = 1_000_000, time_budget_s: float = 30.0) -> PlanResult:
    t0 = time.perf_counter()
    bodies = {p.id: body_of(p, catalog) for p in placements}
    remaining = {p.id for p in placements}
    placed_ids: list[str] = []
    placed_bodies: list[Body] = []
    per_step: list[dict] = []
    nodes = 0
    while remaining:
        best = None
        for pid in sorted(remaining):
            nodes += 1
            if nodes > node_budget or time.perf_counter() - t0 > time_budget_s:
                return PlanResult("planning_unknown", placed_ids, [], {"reason": "search budget exhausted", "placed": len(placed_ids), "remaining": sorted(remaining), "nodes": nodes})
            b = bodies[pid]
            if any(b.overlaps(q) for q in placed_bodies):
                continue
            supporters = [q for q in placed_ids if stud_connected(b, bodies[q])]
            if b.z0 != 0 and not supporters:
                continue
            if path_blocked(b, placed_bodies):
                continue
            if any(bodies[r].z1 <= b.z0 and bodies[r].footprint_overlaps(b) for r in remaining if r != pid):
                continue  # would seal an unplaced piece beneath it
            key = (b.z0, -len(supporters), pid)
            if best is None or key < best[0]:
                best = (key, pid, supporters)
        if best is None:
            return PlanResult("planning_failed", placed_ids, [], {
                "reason": "no remaining piece can be inserted from above while attaching to the assembly",
                "placed": len(placed_ids), "remaining": sorted(remaining), "nodes": nodes,
                "note": "this is a limit of the supported insertion model, not a proof of physical unbuildability",
            })
        _, pid, supporters = best
        remaining.remove(pid)
        placed_ids.append(pid)
        placed_bodies.append(bodies[pid])
        per_step.append({"index": len(placed_ids), "placement_id": pid, "attaches_to": sorted(supporters) or ["ground"]})
    steps = [Step(index=i + 1, added_placement_ids=[pid]) for i, pid in enumerate(placed_ids)]
    return PlanResult("planned", placed_ids, steps, {"nodes": nodes, "seconds": round(time.perf_counter() - t0, 4), "per_step": per_step})
