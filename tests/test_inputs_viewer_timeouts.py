import json
import shutil
import subprocess
import time

import numpy as np
import pytest
import trimesh

from brickcore.cli import main
from brickcore.generate import generate
from brickcore.mesh import load_mesh, normalize, voxelize
from brickcore.schema import Bounds, Constraints
from brickcore.transforms import to_scene
from tests.conftest import ROOT

B = Bounds(x_studs=16, y_studs=16, z_plates=24)


def test_glb_input_matches_stl(fixture_meshes, catalog, tmp_path):
    stl, _ = load_mesh(fixture_meshes / "l_shape.stl")
    glb_path = tmp_path / "l_shape.glb"
    trimesh.Scene(stl).export(glb_path)
    glb, meta = load_mesh(glb_path)
    assert meta["triangles"] > 0
    a = voxelize(normalize(stl, B, catalog)[0], B, catalog)
    b = voxelize(normalize(glb, B, catalog)[0], B, catalog)
    assert np.array_equal(a, b)


def test_gltf_with_external_buffers_is_rejected(tmp_path):
    from brickcore.mesh import MeshError
    p = tmp_path / "x.gltf"
    p.write_text("{}")
    with pytest.raises(MeshError) as e:
        load_mesh(p)
    assert e.value.code == "unsupported_input"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_viewer_js_transform_matches_python(two_brick, catalog):
    """Extract the viewer's footprint() and sceneBox() and evaluate them in Node against to_scene()."""
    html = (ROOT / "viewer" / "index.html").read_text()
    start = html.index("const STUD = 8")
    end = html.index("// --- end pure transform ---")
    snippet = html[start:end]
    placements = [p.model_dump() for p in two_brick.placements]
    script = snippet + f"\nconsole.log(JSON.stringify({json.dumps(placements)}.map(sceneBox)));\n"
    out = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True).stdout
    js = json.loads(out)
    for p, box in zip(two_brick.placements, js, strict=True):
        py = to_scene(p, catalog)
        assert (box["cx"], box["cy"], box["cz"], box["sx"], box["sy"], box["sz"]) == pytest.approx((py.cx, py.cy, py.cz, py.sx, py.sy, py.sz))


def test_generate_time_budget_returns_best_so_far_or_explicit_failure(fixture_meshes, catalog):
    mesh, _ = load_mesh(fixture_meshes / "sphere.stl")
    target = voxelize(normalize(mesh, B, catalog)[0], B, catalog)
    c = Constraints(max_pieces=200, bounds=B, color_id="4")
    res = generate(target, catalog, c, time_budget_s=0.0)
    assert res.diagnostics["aborted"] == "time_budget"
    assert res.status == "no_valid_candidate" and res.build is None
    assert any("time_budget" in w for w in res.diagnostics["warnings"])
    full = generate(target, catalog, c)
    assert "aborted" not in full.diagnostics and full.status == "validated_candidate"


def test_external_cancellation(fixture_meshes, catalog):
    mesh, _ = load_mesh(fixture_meshes / "sphere.stl")
    target = voxelize(normalize(mesh, B, catalog)[0], B, catalog)
    calls = {"n": 0}

    def cancel():
        calls["n"] += 1
        return calls["n"] > 5

    res = generate(target, catalog, Constraints(max_pieces=100, bounds=B, color_id="4"), cancellation=cancel)
    assert res.diagnostics["aborted"] == "cancelled"


def test_cli_parse_timeout_reports_worker_timeout(fixture_meshes, tmp_path, monkeypatch):
    import brickcore.cli as cli
    slow = cli.voxelize

    def slow_voxelize(*a, **k):
        time.sleep(0.5)
        return slow(*a, **k)

    monkeypatch.setattr(cli, "voxelize", slow_voxelize)
    out = tmp_path / "t"
    rc = main(["generate", "--mesh", str(fixture_meshes / "cube.stl"), "--out", str(out), "--parse-timeout", "0.1"])
    assert rc == 1
    r = json.loads((out / "result.json").read_text())
    assert r["status"] == "failed" and r["error_code"] == "worker_timeout" and r["stage"] == "parse"
    assert not (out / "build.json").exists()


def test_held_out_guard_excludes_whole_things():
    from ml.thingi10k import held_out_thing_ids, training_pool
    held = held_out_thing_ids()
    assert len(held) == 40
    ok = {"Closed": "TRUE", "Edge manifold": "TRUE", "Vertex manifold": "TRUE", "Single Component": "TRUE", "PWN": "TRUE", "No degenerate faces": "TRUE"}
    some_held = sorted(held)[0]
    meta = {"input_summary": [{"ID": "1", "Thing ID": some_held, "License": "Creative Commons - Attribution", "Link": "", **ok},
                              {"ID": "2", "Thing ID": "999999999", "License": "Creative Commons - Attribution", "Link": "", **ok}],
            "geometry_data": [{"file_id": "1", "num_faces": "500"}, {"file_id": "2", "num_faces": "500"}],
            "contextual_data": [], "tag_data": []}
    pool = training_pool(meta, held)
    assert [m["file_id"] for m in pool] == ["2"]
