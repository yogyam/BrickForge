"""M6: RL environment, padded policy loading, reward, generator registration, and a two-iteration PPO smoke run."""
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # the ml group is optional; a plain `make setup` has no torch

from brickcore.generate import _largest_ground_component  # noqa: E402 (after the torch skip)
from brickcore.schema import Bounds, Build, Constraints, GeneratorInfo  # noqa: E402 (after the torch skip)
from brickcore.validate import validate_build  # noqa: E402 (after the torch skip)
from ml.data import synth_target  # noqa: E402 (after the torch skip)
from ml.policy import CAND_FEATS, CONN_FEATS, BrickPolicy, candidate_tensors, collate, extend_state_dict  # noqa: E402 (after the torch skip)
from ml.rl_env import AssemblyState, RewardConfig, load_policy_for_rl  # noqa: E402 (after the torch skip)

ARCH = {"channels": 8, "hidden": 16, "layers": 2}


def _random_build(target, catalog, rng, n_steps=25):
    st = AssemblyState(target, catalog)
    for _ in range(n_steps):
        c, extra = st.candidates()
        if len(c) == 0:
            break
        assert extra.shape == (len(c), CONN_FEATS)
        st.step(int(rng.integers(len(c))))
    return st


@pytest.mark.parametrize("family", ["box", "bridge", "arch_row", "polyomino", "blob"])
def test_component_tracking_matches_finalize_trim(catalog, family):
    rng = np.random.default_rng(3)
    t = synth_target(family, rng)
    st = _random_build(t, catalog, rng)
    kept, n_comps = _largest_ground_component(st.placements, catalog)
    roots = st.roots_array()
    assert len(set(roots.tolist())) == n_comps
    main = st.main_root()
    assert [p.id for i, p in enumerate(st.placements) if roots[i] == main] == [p.id for p in kept]
    cells = sum(np.prod(catalog.part(p.part_key).footprint(p.rotation_deg)) for p in kept)
    assert int(st.main_mask().sum()) == cells


def test_connectivity_features_mark_joining_and_merging(catalog):
    # two separate 2x4 ground pieces; a 2x4 bridging both at layer 1 merges them, a piece over only one joins the main one
    t = np.zeros((16, 16, 8), dtype=bool)
    t[0:8, 0:2, 0] = True
    t[0:8, 0:2, 1] = True
    st = AssemblyState(t, catalog)
    c, _ = st.candidates()
    st.step(c.find("brick_2x4", 0, 0, 0, 0))
    c, _ = st.candidates()
    st.step(c.find("brick_2x4", 0, 4, 0, 0))
    assert len(set(st.roots_array().tolist())) == 2
    c, extra = st.candidates()
    i_bridge = c.find("brick_2x4", 0, 2, 0, 1)
    i_over_first = c.find("brick_2x4", 0, 0, 0, 1)
    i_over_second = c.find("brick_2x4", 0, 4, 0, 1)
    assert extra[i_bridge].tolist() == [1.0, 1.0]       # touches the main component and merges two
    assert extra[i_over_first].tolist() == [1.0, 0.0]   # main = earliest piece on ties
    assert extra[i_over_second].tolist() == [0.0, 0.0]
    st.step(i_bridge)
    assert len(set(st.roots_array().tolist())) == 1


def test_padded_policy_matches_base_logits_and_loads(tmp_path, catalog):
    base = BrickPolicy(**ARCH)
    sd = {k: v for k, v in base.state_dict().items() if not k.startswith("value_mlp")}
    ck = tmp_path / "sft_tiny.pt"
    torch.save({"arch": ARCH, "state_dict": sd, "version": "test"}, ck)
    ext, _ = load_policy_for_rl(ck, torch.device("cpu"))
    assert ext.in_channels == 4 and ext.extra_feats == CONN_FEATS
    t = synth_target("box", np.random.default_rng(5))
    st = AssemblyState(t, catalog)
    it_ext = st.item()
    c, _ = st.candidates()
    cell_idx, cell_mask, feats = candidate_tensors(c, t.shape)
    it_base = {"vox": it_ext["vox"][:3], "cell_idx": cell_idx, "cell_mask": cell_mask, "feats": feats, "target_index": 0}
    def run(m, item):
        b = collate([item], torch.device("cpu"))
        return m(b["vox"], b["cell_idx"], b["cell_mask"], b["feats"], b["cand_mask"])

    with torch.no_grad():
        lb, le = run(base, it_base), run(ext, it_ext)
    assert torch.allclose(lb, le, atol=1e-5)
    assert it_ext["feats"].shape[1] == CAND_FEATS + CONN_FEATS
    # extend_state_dict is idempotent on an already-extended checkpoint
    sd2 = extend_state_dict(ext.state_dict(), 4, CONN_FEATS)
    assert sd2["encoder.0.weight"].shape == ext.state_dict()["encoder.0.weight"].shape


def test_reward_is_after_trim_iou_plus_stability(catalog):
    rng = np.random.default_rng(9)
    t = synth_target("stack", rng)
    st = _random_build(t, catalog, rng)
    bounds = Bounds(x_studs=16, y_studs=16, z_plates=24)
    fin = st.finish(Constraints(max_pieces=100, bounds=bounds, color_id="4"), GeneratorInfo(name="t", version="0", seed=0), "t",
                    RewardConfig(stability_weight=0.25))
    assert fin["valid"] and validate_build(fin["build"], catalog)["ok"]
    assert fin["reward"] == pytest.approx(fin["iou"] + 0.25 * float(fin["stable"]))
    assert fin["pieces"] == len(fin["build"].placements) <= fin["raw_pieces"]


def test_rl_generator_registers_from_checkpoint(tmp_path, catalog, monkeypatch):
    from brickcore.generators import REGISTRY, get
    model = BrickPolicy(**ARCH, in_channels=4, extra_feats=CONN_FEATS)
    ck = tmp_path / "rl_tiny.pt"
    torch.save({"arch": model.arch, "state_dict": model.state_dict(), "version": "test"}, ck)
    REGISTRY.pop("rl_v1", None)
    gen = get("rl_v1")
    t = synth_target("box", np.random.default_rng(11))
    res = gen(t, catalog, Constraints(max_pieces=30, bounds=Bounds(x_studs=16, y_studs=16, z_plates=24), color_id="4"), seed=0, ckpt=ck)
    assert res.status in ("validated_candidate", "partial_candidate", "no_valid_candidate")
    assert res.diagnostics["checkpoint"] == "rl_tiny.pt" and "components_before_trim" in res.diagnostics
    if res.build is not None:
        assert res.build.generator.name == "rl_v1" and validate_build(res.build, catalog)["ok"]
    # masked STOP: the episode can only end on an empty candidate set or the cap
    res2 = gen(t, catalog, Constraints(max_pieces=30, bounds=Bounds(x_studs=16, y_studs=16, z_plates=24), color_id="4"), seed=0, ckpt=ck,
               stop_mode="never")
    assert res2.diagnostics["stop_reason"] in ("no_legal_candidate", "cap")
    if res2.build is not None:
        assert res2.build.generator.strategy == "greedy_decode_stop_never"


def test_ppo_smoke_two_iterations(tmp_path, catalog, monkeypatch):
    from ml import train_rl
    from ml.data import Example
    rng = np.random.default_rng(2)
    rows = []
    for i, fam in enumerate(["box", "stack", "box", "stepped"]):
        t = synth_target(fam, rng)
        cells = np.argwhere(t).tolist()
        rows.append(json.dumps({"id": f"e{i}", "family": fam, "split": "train", "grid": [16, 16, 8], "target_cells": cells, "placements": [],
                                "iou": 0.9, "budget": 40}))
    data = tmp_path / "ex.jsonl"
    data.write_text("\n".join(rows) + "\n")
    base = BrickPolicy(**ARCH)
    init = tmp_path / "init.pt"
    torch.save({"arch": ARCH, "state_dict": {k: v for k, v in base.state_dict().items() if not k.startswith("value_mlp")}, "version": "test"}, init)
    monkeypatch.setattr(train_rl, "ROOT", tmp_path)
    rc = train_rl.main(["--data", str(data), "--init", str(init), "--name", "smoke", "--iters", "2", "--episodes", "4", "--workers", "0",
                        "--epochs", "1", "--minibatch", "16", "--eval-every", "1", "--eval-n", "4", "--device", "cpu", "--overfit", "4"])
    assert rc in (0, 1)  # the gate may or may not pass in two iterations; the pipeline must run end to end
    lines = [json.loads(line) for line in (tmp_path / "runs" / "rl" / "smoke" / "metrics.jsonl").read_text().splitlines()]
    assert lines[0]["iter"] == 0 and "eval" in lines[0]
    assert lines[-1]["update"]["n_steps"] > 0 and (tmp_path / "runs" / "rl" / "smoke" / "final_eval.json").exists()
    assert Example  # imported for clarity of the JSONL schema above


def test_cli_runs_committed_rl_checkpoint(tmp_path, catalog):
    """The headline checkpoint is committed so `brickforge generate --generator rl_v2_nostop` works from a clone."""
    from pathlib import Path

    from brickcore.cli import main
    from brickcore.fixtures import write_fixtures
    ckpt = Path(__file__).resolve().parent.parent / "checkpoints" / "rl_v2.pt"
    if not ckpt.exists():
        pytest.skip("checkpoints/rl_v2.pt not present")
    meshes = tmp_path / "meshes"
    write_fixtures(meshes)
    out = tmp_path / "rl"
    assert main(["generate", "--mesh", str(meshes / "cube.stl"), "--out", str(out), "--size", "8", "8", "4", "--generator", "rl_v2_nostop"]) == 0
    result = json.loads((out / "result.json").read_text())
    assert result["generator"] == "rl_v2_nostop" and result["status"] == "validated_candidate" and result["iou"] > 0.9
    assert validate_build(Build.from_json((out / "build.json").read_text()), catalog)["ok"]
