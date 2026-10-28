"""Authoritative, generator-independent structural validator and step replay."""
from __future__ import annotations

from collections import defaultdict

from .catalog import Catalog
from .geometry import Body, body_of, stud_connected
from .schema import Build


def components(ids: list[str], edges: list[tuple[str, str]]) -> list[set[str]]:
    adj: dict[str, set[str]] = {i: set() for i in ids}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    seen: set[str] = set()
    out = []
    for i in ids:
        if i in seen:
            continue
        stack, comp = [i], set()
        while stack:
            n = stack.pop()
            if n in comp:
                continue
            comp.add(n)
            stack.extend(adj[n] - comp)
        seen |= comp
        out.append(comp)
    return out


def validate_build(build: Build, catalog: Catalog) -> dict:
    errors: list[dict] = []
    warnings: list[dict] = []

    def err(code: str, msg: str, **extra):
        errors.append({"code": code, "message": msg, **extra})

    if build.catalog_version != catalog.version:
        err("catalog_mismatch", f"build uses {build.catalog_version}, validator has {catalog.version}")

    bounds = build.constraints.bounds
    bodies: dict[str, Body] = {}
    ids = [p.id for p in build.placements]
    if len(set(ids)) != len(ids):
        err("duplicate_id", "placement ids are not unique")

    for p in build.placements:
        if p.part_key not in catalog.parts:
            err("unknown_part", f"{p.id}: unknown part {p.part_key}", placement=p.id)
            continue
        part = catalog.part(p.part_key)
        if p.rotation_deg not in part.allowed_rotations:
            err("illegal_rotation", f"{p.id}: rotation {p.rotation_deg} not allowed for {p.part_key}", placement=p.id)
            continue
        if p.color_id not in catalog.supported_color_ids:
            err("illegal_color", f"{p.id}: color {p.color_id} not in catalog", placement=p.id)
        if p.color_id != build.constraints.color_id:
            err("color_mismatch", f"{p.id}: color {p.color_id} differs from constraint {build.constraints.color_id}", placement=p.id)
        if p.z % 3 != 0:
            err("illegal_z", f"{p.id}: z={p.z} is not a multiple of 3", placement=p.id)
        b = body_of(p, catalog)
        if b.x0 < 0 or b.y0 < 0 or b.z0 < 0 or b.x1 > bounds.x_studs or b.y1 > bounds.y_studs or b.z1 > bounds.z_plates:
            err("out_of_bounds", f"{p.id}: body {b} exceeds region", placement=p.id)
        bodies[p.id] = b

    if len(build.placements) > build.constraints.max_pieces:
        err("budget_exceeded", f"{len(build.placements)} pieces > cap {build.constraints.max_pieces}")

    # Collision: pairwise on bodies (n <= 100 so O(n^2) is fine and exact).
    keys = list(bodies)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if bodies[keys[i]].overlaps(bodies[keys[j]]):
                err("collision", f"{keys[i]} overlaps {keys[j]}", a=keys[i], b=keys[j])

    # Connection graph: only true stud-to-underside contact creates an edge.
    edges: list[tuple[str, str]] = []
    supports: dict[str, list[str]] = defaultdict(list)
    for i in range(len(keys)):
        for j in range(len(keys)):
            if i != j and stud_connected(bodies[keys[i]], bodies[keys[j]]):
                edges.append((keys[i], keys[j]))
                supports[keys[i]].append(keys[j])
    ground = [k for k in keys if bodies[k].z0 == 0]
    comps = components(keys, edges)
    if keys and len(comps) != 1:
        err("disconnected", f"{len(comps)} connection components, expected 1", components=[sorted(c) for c in comps])
    for k in keys:
        if bodies[k].z0 > 0 and not supports[k]:
            err("unsupported", f"{k} has no stud connection below", placement=k)
    if keys and not ground:
        err("no_ground_contact", "no piece touches z=0")

    # Steps, when present, must partition placements exactly once. Replay must keep the assembly connected
    # and every insertion path from above must be clear of previously placed pieces.
    has_steps = bool(build.steps)
    step_ids = [i for s in build.steps for i in s.added_placement_ids]
    if has_steps and sorted(step_ids) != sorted(ids):
        err("steps_not_partition", "steps do not cover every placement exactly once")
    if not has_steps and ids:
        warnings.append({"code": "no_instructions", "message": "build has no checked assembly steps"})
    if [s.index for s in build.steps] != list(range(1, len(build.steps) + 1)):
        err("step_index", "step indices must be 1..N in order")
    placed: set[str] = set()
    replay_ok = has_steps and not errors
    for s in build.steps:
        for pid in s.added_placement_ids:
            if pid not in bodies:
                continue
            b = bodies[pid]
            attached = b.z0 == 0 or any(stud_connected(b, bodies[q]) for q in placed)
            if not attached:
                err("replay_detached", f"step {s.index}: {pid} does not attach to the assembly", step=s.index, placement=pid)
                replay_ok = False
            blockers = [q for q in placed if bodies[q].z0 >= b.z1 and bodies[q].footprint_overlaps(b)]
            if blockers:
                err("replay_blocked", f"step {s.index}: {pid} cannot be inserted from above; blocked by {sorted(blockers)}", step=s.index, placement=pid)
                replay_ok = False
            placed.add(pid)

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "piece_count": len(build.placements),
        "max_pieces": build.constraints.max_pieces,
        "component_count": len(comps),
        "ground_piece_ids": sorted(ground),
        "edge_count": len(edges),  # edges are directed upper -> lower, one per connected pair
        "supports": {k: sorted(v) for k, v in supports.items()},
        "replay_ok": replay_ok,
        "instruction_status": "planned" if replay_ok else ("none" if not has_steps else "invalid"),
        "label": "checks passed for the simplified brick model" if not errors else "structural checks failed",
    }
