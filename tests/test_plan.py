import json

from brickcore.cli import main
from brickcore.generate import generate
from brickcore.mesh import load_mesh, normalize, voxelize
from brickcore.plan import plan_assembly
from brickcore.schema import Bounds, Build, Constraints, Placement, Step
from brickcore.validate import validate_build


def P(id, part, x, y, z, rot=0):
    return Placement(id=id, part_key=part, x=x, y=y, z=z, rotation_deg=rot, color_id="4")


def test_two_brick_plans_bottom_up(two_brick, catalog):
    r = plan_assembly(two_brick.placements, catalog)
    assert r.status == "planned" and r.order == ["b0001", "b0002"]
    assert r.diagnostics["per_step"][1]["attaches_to"] == ["b0001"]


def test_lower_piece_under_an_overhang_is_placed_before_the_overhang(catalog):
    # A on the ground, B rests on A and overhangs to the right, C sits on the ground under B's overhang.
    # Ordering by id alone would place B before C and seal C off. The planner must emit A, C, B.
    a, b, c = P("a", "brick_2x4", 0, 0, 0), P("b", "brick_2x4", 2, 0, 3), P("c", "brick_1x2", 4, 0, 0)
    r = plan_assembly([a, b, c], catalog)
    assert r.status == "planned" and r.order == ["a", "c", "b"]


def test_hanging_piece_fails_planning_with_explicit_state(catalog):
    # C has nothing below it and is held only by B above it: it would need insertion from below.
    a, a2 = P("a", "brick_2x4", 0, 0, 0), P("a2", "brick_2x4", 0, 0, 3)
    b, c = P("b", "brick_2x4", 2, 0, 6), P("c", "brick_2x2", 4, 0, 3)
    r = plan_assembly([a, a2, b, c], catalog)
    assert r.status == "planning_failed" and r.steps == []
    assert r.diagnostics["remaining"] == ["b", "c"] and "not a proof" in r.diagnostics["note"]


def test_budget_exhaustion_is_unknown_not_failed(two_brick, catalog):
    r = plan_assembly(two_brick.placements, catalog, node_budget=1)
    assert r.status == "planning_unknown"


def test_replay_rejects_blocked_insertion(two_brick, catalog):
    a, b, c = P("a", "brick_2x4", 0, 0, 0), P("b", "brick_2x4", 2, 0, 3), P("c", "brick_1x2", 4, 0, 0)
    raw = two_brick.model_dump()
    raw["constraints"]["bounds"] = {"x_studs": 8, "y_studs": 2, "z_plates": 6}
    raw["placements"] = [p.model_dump() for p in (a, b, c)]
    raw["steps"] = [Step(index=i + 1, added_placement_ids=[pid]).model_dump() for i, pid in enumerate(["a", "b", "c"])]
    report = validate_build(Build.model_validate(raw), catalog)
    assert any(e["code"] == "replay_blocked" and e["placement"] == "c" for e in report["errors"])
    raw["steps"] = [Step(index=i + 1, added_placement_ids=[pid]).model_dump() for i, pid in enumerate(["a", "c", "b"])]
    assert validate_build(Build.model_validate(raw), catalog)["ok"]


def test_empty_steps_are_a_warning_not_an_error(two_brick, catalog):
    raw = two_brick.model_dump()
    raw["steps"] = []
    report = validate_build(Build.model_validate(raw), catalog)
    assert report["ok"] and report["instruction_status"] == "none"
    assert any(w["code"] == "no_instructions" for w in report["warnings"])


def test_generated_builds_carry_planned_replayable_steps(fixture_meshes, catalog):
    bounds = Bounds(x_studs=16, y_studs=16, z_plates=24)
    for name in ["sphere", "bridge", "pyramid"]:
        mesh, _ = load_mesh(fixture_meshes / f"{name}.stl")
        m, _ = normalize(mesh, bounds, catalog)
        res = generate(voxelize(m, bounds, catalog), catalog, Constraints(max_pieces=100, bounds=bounds, color_id="4"))
        assert res.instruction_status == "planned", name
        report = validate_build(res.build, catalog)
        assert report["ok"] and report["replay_ok"] and report["instruction_status"] == "planned"
        zs = [next(p.z for p in res.build.placements if p.id == s.added_placement_ids[0]) for s in res.build.steps]
        assert zs == sorted(zs), "planned order must be layer by layer for upright bricks"
        assert len(res.build.steps) == len(res.build.placements)


def test_cli_plan_command_round_trips(fixture_meshes, tmp_path):
    out = tmp_path / "cube"
    assert main(["generate", "--mesh", str(fixture_meshes / "cube.stl"), "--out", str(out), "--size", "8", "8", "4"]) == 0
    plan = json.loads((out / "plan.json").read_text())
    assert plan["status"] == "planned" and plan["steps"] == json.loads((out / "result.json").read_text())["pieces"]
    assert main(["plan", str(out / "build.json"), "--write"]) == 0
    assert main(["validate", str(out / "build.json")]) == 0
