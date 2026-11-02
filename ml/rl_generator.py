"""Inference wrapper for PPO-trained policies, registered as `rl_v1`. Greedy decode with the policy's own STOP,
through the same AssemblyState and the shared finalize step as every other generator."""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch

from brickcore.catalog import Catalog
from brickcore.generate import CandidateResult, finalize_candidate
from brickcore.generators import register
from brickcore.schema import Constraints, GeneratorInfo

from .rl_env import AssemblyState, load_policy_for_rl, rollout

GENERATOR_NAME = "rl_v1"
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CKPT = Path(os.environ.get("BRICKFORGE_RL_CKPT", ROOT / "checkpoints" / "rl_v1.pt"))
DEFAULT_DEVICE = os.environ.get("BRICKFORGE_RL_DEVICE", "cpu")


class RLGenerator:
    def __init__(self, ckpt: Path = DEFAULT_CKPT, device: str | None = None, name: str = GENERATOR_NAME):
        self.device = torch.device(device or DEFAULT_DEVICE)
        self.ckpt_path, self.name = ckpt, name
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        arch = ck["arch"]
        self.conn_feats = arch.get("extra_feats", 0) > 0
        self.main_channel = arch.get("in_channels", 3) == 4
        self.model, self.ck = load_policy_for_rl(ckpt, self.device, self.conn_feats, self.main_channel)
        self.model.eval()
        self.version = self.ck.get("version", "0.1.0")

    def __call__(self, target: np.ndarray, catalog: Catalog, constraints: Constraints, seed: int = 0, cancellation=None,
                 build_id: str = "build", time_budget_s: float | None = None) -> CandidateResult:
        t0 = time.perf_counter()
        def cancel() -> bool:
            return bool(cancellation and cancellation()) or (time_budget_s is not None and time.perf_counter() - t0 > time_budget_s)

        st = AssemblyState(target, catalog, constraints.color_id, self.conn_feats, self.main_channel)
        out = rollout(self.model, st, constraints.max_pieces, self.device, cancel=cancel)
        info = GeneratorInfo(name=self.name, version=self.version, seed=seed, strategy="greedy_decode")
        build, report, m = finalize_candidate(st.placements, target, catalog, constraints, info, build_id)
        diagnostics = {"checkpoint": str(self.ckpt_path.name), "raw_pieces": len(st.placements), "stop_reason": out["stop_reason"],
                       **{k: v for k, v in m.items() if k != "plan_diagnostics"}, "target_cells": int(target.sum()), "seconds": round(time.perf_counter() - t0, 3)}
        if not report["ok"] or not build.placements:
            diagnostics["errors"] = [e["code"] for e in report["errors"]]
            return CandidateResult("no_valid_candidate", None, 0.0, 0.0, 0.0, diagnostics)
        res = CandidateResult("validated_candidate", build, m["iou"], m["coverage"], m["extraneous"], diagnostics, m["instruction_status"], m["plan_diagnostics"])
        if res.coverage < 0.8 or m["dropped_disconnected"]:
            res.status = "partial_candidate"
        return res


_cached: dict[str, RLGenerator] = {}


def rl_generate(target, catalog, constraints, seed=0, cancellation=None, build_id="build", time_budget_s=None, ckpt: Path = DEFAULT_CKPT):
    key = str(ckpt)
    if key not in _cached:
        _cached[key] = RLGenerator(ckpt)
    return _cached[key](target, catalog, constraints, seed, cancellation, build_id, time_budget_s)


register(GENERATOR_NAME, rl_generate)
