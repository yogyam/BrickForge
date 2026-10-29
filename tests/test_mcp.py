import asyncio

import pytest

from brickcore import mcp_server as srv


def test_tools_are_registered():
    names = {t.name for t in asyncio.run(srv.mcp.list_tools())}
    assert {"list_fixtures", "generate", "validate", "plan", "stability", "inspect_step", "bom", "describe_build"} <= names


def test_paths_are_confined_to_project():
    with pytest.raises(ValueError):
        srv._safe("/etc/passwd")
    with pytest.raises(ValueError):
        srv._safe("../outside.json")


def test_generate_then_inspect(fixture_meshes, tmp_path):
    # Tools take project-relative paths; point them at an in-project scratch dir under runs/.
    import shutil
    out = srv.ROOT / "runs" / "_mcp_test"
    shutil.rmtree(out, ignore_errors=True)
    (srv.ROOT / "runs").mkdir(exist_ok=True)
    shutil.copy(fixture_meshes / "bridge.stl", srv.ROOT / "runs" / "_mcp_test_bridge.stl")
    try:
        res = srv.generate("runs/_mcp_test_bridge.stl", "runs/_mcp_test", max_pieces=60)
        assert res["exit_code"] == 0 and res["status"] in ("validated_candidate", "partial_candidate")
        build = "runs/_mcp_test/build.json"
        assert srv.validate(build)["ok"]
        p = srv.plan(build)
        assert p["status"] == "planned" and len(p["order"]) == res["pieces"]
        s = srv.stability(build)
        assert s["final"]["stable"] and s["unstable_step_count"] == 0
        step = srv.inspect_step(build, 1)
        assert step["added"][0]["attaches_to"] == ["ground"] and step["stability_after_step"]["stable"]
        last = srv.inspect_step(build, res["pieces"])
        assert last["pieces_placed_after_step"] == res["pieces"]
        assert sum(r["count"] for r in srv.bom(build)) == res["pieces"]
        d = srv.describe_build(build)
        assert d["pieces"] == res["pieces"] and d["height_layers"] == 8
        with pytest.raises(ValueError):
            srv.inspect_step(build, 0)
    finally:
        shutil.rmtree(out, ignore_errors=True)
        (srv.ROOT / "runs" / "_mcp_test_bridge.stl").unlink(missing_ok=True)
