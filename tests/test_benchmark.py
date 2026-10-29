import json

from brickcore.generators import REGISTRY, get, register
from ml import benchmark as B
from ml.thingi10k import license_ok, select_subset


def test_generator_registry():
    assert "greedy_v1" in REGISTRY and get("greedy_v1") is REGISTRY["greedy_v1"]
    register("dummy", lambda *a, **k: None)
    assert get("dummy") is not None
    del REGISTRY["dummy"]


def test_license_filter():
    assert license_ok("Creative Commons - Attribution")
    assert license_ok("Creative Commons - Attribution - Share Alike")
    assert license_ok("Creative Commons - Public Domain Dedication")
    assert not license_ok("Creative Commons - Attribution - Non-Commercial")
    assert not license_ok("Creative Commons - Attribution - No Derivatives")
    assert not license_ok("GNU - GPL") and not license_ok("")


def test_subset_selection_is_deterministic_and_filtered():
    ok = {"Closed": "TRUE", "Edge manifold": "TRUE", "Vertex manifold": "TRUE", "Single Component": "TRUE", "PWN": "TRUE", "No degenerate faces": "TRUE"}
    summary = [{"ID": str(i), "Thing ID": str(i // 2), "License": "Creative Commons - Attribution", "Link": "x", **ok} for i in range(100)]
    summary[5]["License"] = "Creative Commons - Attribution - Non-Commercial"
    summary[7]["Closed"] = "FALSE"
    meta = {"input_summary": summary,
            "geometry_data": [{"file_id": str(i), "num_faces": str(1000 if i != 9 else 10)} for i in range(100)],
            "contextual_data": [{"Thing ID": str(t), "Name": f"thing{t}", "Author": "a", "Category": ""} for t in range(50)],
            "tag_data": [{"Thing ID": "0", "Tag": "cat"}]}
    a = select_subset(meta, n=10, seed=3)
    b = select_subset(meta, n=10, seed=3)
    assert a == b and len(a) == 10
    ids = {m["file_id"] for m in a}
    assert "5" not in ids and "7" not in ids and "9" not in ids
    assert len({m["thing_id"] for m in a}) == 10  # one file per thing


def test_procedural_benchmark_runs_and_summarises(tmp_path):
    cases = [c for c in B.procedural_cases() if c.id in ("proc_cube", "proc_bridge", "proc_thin_pole", "proc_open_box_invalid")]
    out = tmp_path / "bench.jsonl"
    records = B.run(cases, ["greedy_v1"], [50, 100], [42], out, progress=False)
    assert len(records) == 8 and out.exists()
    by = {(r["case"], r["budget"]): r for r in records}
    assert by[("proc_cube", 100)]["iou"] == 1.0 and by[("proc_cube", 100)]["meets_expected"]
    assert by[("proc_open_box_invalid", 50)]["status"] == "failed" and by[("proc_open_box_invalid", 50)]["meets_expected"]
    assert by[("proc_thin_pole", 50)]["status"] == "no_valid_candidate" and by[("proc_thin_pole", 50)]["meets_expected"]
    assert by[("proc_bridge", 50)]["pieces"] <= 50 and by[("proc_bridge", 100)]["commit"]
    summary = B.summarize(records)
    rows = {(r["generator"], r["budget"]): r for r in summary["by_generator_budget"]}
    assert rows[("greedy_v1", 100)]["cases"] == 4 and 0 < rows[("greedy_v1", 100)]["valid_rate"] < 1
    md = B.to_markdown(summary, records)
    assert "proc_cube" in md and "greedy_v1@100" in md
    assert all(json.loads(l)["case"] for l in out.read_text().splitlines())
