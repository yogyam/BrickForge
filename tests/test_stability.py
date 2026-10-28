import json

from brickcore.cli import main
from brickcore.schema import Placement, Step
from brickcore.stability import StabilityParams, analyze, analyze_build


def P(id, part, x, y, z, rot=0):
    return Placement(id=id, part_key=part, x=x, y=y, z=z, rotation_deg=rot, color_id="4")


def test_two_brick_fixture_is_stable_with_no_tension(two_brick, catalog):
    r = analyze(two_brick.placements, catalog)
    assert r["stable"] and r["mode"] == "ok" and r["utilization"] == 0.0


def test_empty_and_single_ground_piece(catalog):
    assert analyze([], catalog)["stable"]
    assert analyze([P("a", "brick_1x1", 0, 0, 0)], catalog)["stable"]


def test_symmetric_t_is_stable_and_cantilever_loads_clutch(catalog):
    column = [P(f"c{i}", "brick_2x2", 3, 0, 3 * i) for i in range(3)]
    sym = analyze(column + [P("top", "brick_2x4", 2, 0, 9, 0)], catalog)  # top centred over the column
    assert sym["stable"] and sym["utilization"] == 0.0
    off = analyze(column + [P("top", "brick_2x4", 3, 0, 9, 0)], catalog)  # top overhangs to the right
    assert off["stable"] and off["utilization"] > 0 and off["worst_joint"]["upper"] == "top"


def test_staircase_eventually_topples(catalog):
    base = P("g", "brick_2x4", 0, 0, 0, 90)
    results = []
    for n in range(1, 6):
        pl = [base] + [P(f"s{i}", "brick_1x4", i, 0, 3 * (i + 1)) for i in range(n)]
        results.append(analyze(pl, catalog))
    utils = [r["utilization"] for r in results if r["mode"] == "ok"]
    assert utils == sorted(utils) and len(utils) >= 2
    assert results[-1]["mode"] == "topple" and not results[-1]["stable"]


def test_clutch_exceeded_mode_when_capacity_is_tiny(catalog):
    # A 1x4 held by one stud at the base's edge. The combined centre of mass stays over the base, so it cannot
    # topple; with near-zero clutch it must fail by clutch instead, and with the default clutch it holds.
    pl = [P("base", "brick_2x4", 0, 0, 0), P("hang", "brick_1x4", 3, 0, 3)]
    weak = analyze(pl, catalog, StabilityParams(clutch_newton_per_stud=1e-6))
    assert weak["mode"] == "clutch_exceeded" and not weak["stable"]
    assert analyze(pl, catalog)["stable"]


def test_per_step_report_flags_first_unstable_prefix(catalog):
    base = P("g", "brick_2x4", 0, 0, 0, 90)
    stairs = [P(f"s{i}", "brick_1x4", i, 0, 3 * (i + 1)) for i in range(5)]
    steps = [Step(index=i + 1, added_placement_ids=[p.id]) for i, p in enumerate([base] + stairs)]
    rep = analyze_build([base] + stairs, steps, catalog)
    assert rep["first_unstable_step"] is not None and not rep["all_steps_stable"]
    assert rep["per_step"][0]["stable"] and rep["model"] == "stud_joint_lp_v1"
    assert "not physically tested" in rep["label"]


def test_cli_writes_stability_report(fixture_meshes, tmp_path):
    out = tmp_path / "sphere"
    assert main(["generate", "--mesh", str(fixture_meshes / "sphere.stl"), "--out", str(out)]) == 0
    stab = json.loads((out / "stability.json").read_text())
    result = json.loads((out / "result.json").read_text())
    assert stab["final"]["stable"] and stab["all_steps_stable"] and len(stab["per_step"]) == result["pieces"]
    assert result["stability"]["final_stable"] and result["timings_s"]["stability"] < 10
    assert main(["stability", str(out / "build.json")]) == 0
