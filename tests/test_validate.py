
import pytest
from pydantic import ValidationError

from brickcore.schema import Build, Placement, Step
from brickcore.validate import validate_build


def _with(build: Build, placements, steps=None) -> Build:
    b = build.model_dump()
    b["placements"] = [p.model_dump() for p in placements]
    b["steps"] = [s.model_dump() for s in (steps or [Step(index=i + 1, added_placement_ids=[p.id]) for i, p in enumerate(placements)])]
    return Build.model_validate(b)


def P(id, part, x, y, z, rot=0):
    return Placement(id=id, part_key=part, x=x, y=y, z=z, rotation_deg=rot, color_id="4")


def codes(report):
    return sorted({e["code"] for e in report["errors"]})


def test_two_brick_fixture_is_valid(two_brick, catalog):
    r = validate_build(two_brick, catalog)
    assert r["ok"], r["errors"]
    assert r["component_count"] == 1 and r["edge_count"] == 1 and r["replay_ok"]
    assert r["supports"]["b0002"] == ["b0001"]
    assert r["label"] == "checks passed for the simplified brick model"


def test_collision_detected(two_brick, catalog):
    r = validate_build(_with(two_brick, [P("a", "brick_2x4", 0, 0, 0), P("b", "brick_1x1", 3, 1, 0)]), catalog)
    assert "collision" in codes(r)


def test_floating_piece_is_unsupported_and_disconnected(two_brick, catalog):
    r = validate_build(_with(two_brick, [P("a", "brick_2x4", 0, 0, 0), P("b", "brick_2x2", 0, 0, 6)]), catalog)
    assert {"unsupported", "disconnected", "replay_detached"} <= set(codes(r))


def test_side_by_side_ground_bricks_are_disconnected(two_brick, catalog):
    r = validate_build(_with(two_brick, [P("a", "brick_2x2", 0, 0, 0), P("b", "brick_2x2", 2, 0, 0)]), catalog)
    assert codes(r) == ["disconnected"]


def test_budget_and_bounds(two_brick, catalog):
    r = validate_build(_with(two_brick, [P("a", "brick_2x4", 2, 0, 0)]), catalog)
    assert "out_of_bounds" in codes(r)
    many = [P(f"p{i}", "brick_1x1", i, 0, 0) for i in range(4)] + [P(f"q{i}", "brick_1x1", i, 1, 0) for i in range(4)]
    many += [P(f"r{i}", "brick_1x2", 0, i, 3) for i in range(2)] + [P("s", "brick_1x1", 2, 0, 3), P("t", "brick_1x1", 3, 0, 3), P("u", "brick_1x1", 2, 1, 3)]
    r = validate_build(_with(two_brick, many), catalog)
    assert "budget_exceeded" in codes(r)


def test_illegal_rotation_and_z(two_brick, catalog):
    r = validate_build(_with(two_brick, [P("a", "brick_2x2", 0, 0, 0, rot=90)]), catalog)
    assert "illegal_rotation" in codes(r)
    r = validate_build(_with(two_brick, [P("a", "brick_2x4", 0, 0, 1)]), catalog)
    assert "illegal_z" in codes(r)


def test_steps_must_partition_placements(two_brick, catalog):
    bad = _with(two_brick, list(two_brick.placements), steps=[Step(index=1, added_placement_ids=["b0001", "b0001"])])
    assert "steps_not_partition" in codes(validate_build(bad, catalog))
    wrong_order = _with(two_brick, list(two_brick.placements), steps=[Step(index=1, added_placement_ids=["b0002"]), Step(index=2, added_placement_ids=["b0001"])])
    r = validate_build(wrong_order, catalog)
    assert "replay_detached" in codes(r) and not r["replay_ok"]


def test_schema_rejects_unknown_fields(two_brick):
    raw = two_brick.model_dump()
    raw["placements"][0]["extra"] = 1
    with pytest.raises(ValidationError):
        Build.model_validate(raw)
