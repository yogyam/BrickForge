import json

from brickcore.cli import main
from brickcore.generate import generate
from brickcore.mesh import load_mesh, normalize, voxelize
from brickcore.schema import Bounds, Build, Constraints
from brickcore.validate import validate_build

SIMPLE = ["cube", "slab", "tower", "l_shape", "t_shape", "bridge", "sphere", "cylinder", "pyramid", "wedge"]


def _target(fixture_meshes, catalog, name, bounds):
    mesh, _ = load_mesh(fixture_meshes / f"{name}.stl")
    m, _ = normalize(mesh, bounds, catalog)
    return voxelize(m, bounds, catalog)


def test_cube_is_reconstructed_exactly(fixture_meshes, catalog):
    bounds = Bounds(x_studs=8, y_studs=8, z_plates=12)
    res = generate(_target(fixture_meshes, catalog, "cube", bounds), catalog, Constraints(max_pieces=100, bounds=bounds, color_id="4"))
    assert res.status == "validated_candidate" and res.fidelity == 1.0
    assert validate_build(res.build, catalog)["ok"]


def test_output_is_deterministic(fixture_meshes, catalog):
    bounds = Bounds(x_studs=16, y_studs=16, z_plates=24)
    t = _target(fixture_meshes, catalog, "sphere", bounds)
    c = Constraints(max_pieces=100, bounds=bounds, color_id="4")
    a = generate(t, catalog, c, seed=42, build_id="x").build.to_json()
    b = generate(t, catalog, c, seed=42, build_id="x").build.to_json()
    assert a == b


def test_piece_cap_is_never_exceeded(fixture_meshes, catalog):
    bounds = Bounds(x_studs=16, y_studs=16, z_plates=24)
    t = _target(fixture_meshes, catalog, "sphere", bounds)
    for cap in (10, 25, 50):
        res = generate(t, catalog, Constraints(max_pieces=cap, bounds=bounds, color_id="4"))
        assert res.build is not None and len(res.build.placements) <= cap
        assert validate_build(res.build, catalog)["ok"]


def test_release_gate_eight_of_ten_simple_fixtures(fixture_meshes, catalog):
    """R1 gate: >= 8 of 10 frozen simple fixtures give a valid candidate with coverage >= 0.80 at the default budget."""
    passed, detail = 0, {}
    for name in SIMPLE:
        bounds = Bounds(x_studs=8, y_studs=8, z_plates=12) if name == "cube" else Bounds(x_studs=16, y_studs=16, z_plates=24)
        res = generate(_target(fixture_meshes, catalog, name, bounds), catalog, Constraints(max_pieces=100, bounds=bounds, color_id="4"))
        ok = res.build is not None and validate_build(res.build, catalog)["ok"] and res.coverage >= 0.80
        detail[name] = (res.status, round(res.coverage, 3))
        passed += ok
    assert passed >= 8, detail


def test_impossible_fixtures_return_explicit_states(fixture_meshes, catalog):
    bounds = Bounds(x_studs=16, y_studs=16, z_plates=24)
    pole = generate(_target(fixture_meshes, catalog, "thin_pole", bounds), catalog, Constraints(max_pieces=100, bounds=bounds, color_id="4"))
    assert pole.status == "no_valid_candidate" and pole.build is None and pole.diagnostics["warnings"]
    sheet = generate(_target(fixture_meshes, catalog, "flat_sheet_impossible", bounds), catalog, Constraints(max_pieces=100, bounds=bounds, color_id="4"))
    assert sheet.status == "partial_candidate" and sheet.coverage < 0.8


def test_cli_success_writes_all_artifacts(fixture_meshes, tmp_path):
    out = tmp_path / "cube"
    rc = main(["generate", "--mesh", str(fixture_meshes / "cube.stl"), "--out", str(out), "--size", "8", "8", "4"])
    assert rc == 0
    for f in ["build.json", "validation.json", "bom.csv", "build.ldr", "result.json", "normalization.json", "voxels.json"]:
        assert (out / f).exists(), f
    result = json.loads((out / "result.json").read_text())
    assert result["status"] == "validated_candidate" and result["iou"] == 1.0
    build = Build.from_json((out / "build.json").read_text())
    assert json.loads((out / "validation.json").read_text())["ok"]
    assert main(["validate", str(out / "build.json")]) == 0
    assert sum(1 for line in (out / "build.ldr").read_text().splitlines() if line.startswith("1 ")) == len(build.placements)


def test_cli_intentional_failure_on_open_mesh(fixture_meshes, tmp_path):
    out = tmp_path / "open"
    rc = main(["generate", "--mesh", str(fixture_meshes / "open_box_invalid.stl"), "--out", str(out)])
    assert rc == 1
    result = json.loads((out / "result.json").read_text())
    assert result["status"] == "failed" and result["error_code"] == "non_watertight_mesh"
    assert not (out / "build.json").exists()


def test_export_refuses_invalid_build(two_brick, tmp_path):
    raw = two_brick.model_dump()
    raw["placements"][1]["z"] = 6  # floating
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(raw))
    assert main(["export", str(p), "--out", str(tmp_path / "exp")]) == 1
    assert not (tmp_path / "exp" / "build.ldr").exists()
