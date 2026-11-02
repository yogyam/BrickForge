"""Inference wrapper: a trained BrickPolicy exposed through the canonical generator interface as `sft_v1`.
Free-running greedy decode (or seeded sampling) over legal candidates, then the shared finalize step."""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch

from brickcore.candidates import PART_ORDER, enumerate_candidates
from brickcore.catalog import Catalog
from brickcore.generate import CandidateResult, finalize_candidate
from brickcore.generators import register
from brickcore.schema import BRICK_HEIGHT_PLATES, Constraints, GeneratorInfo, Placement

from .policy import BrickPolicy, candidate_tensors, collate, voxel_input

GENERATOR_NAME = "sft_v1"
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CKPT = Path(os.environ.get("BRICKFORGE_SFT_CKPT", ROOT / "checkpoints" / "sft_v1.pt"))  # override per run
# Inference runs on CPU by default: the policy is tiny (97k params), CPU decode is as fast as Metal for one target,
# and CPU results are reproducible across processes and machines. Training still uses Metal.
DEFAULT_DEVICE = os.environ.get("BRICKFORGE_SFT_DEVICE", "cpu")


def load_policy(path: Path, device: torch.device) -> tuple[BrickPolicy, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    model = BrickPolicy(**ck["arch"])
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    return model, ck


@torch.no_grad()
def rollout(model: BrickPolicy, target: np.ndarray, catalog: Catalog, max_pieces: int, device: torch.device,
            temperature: float = 0.0, rng: np.random.Generator | None = None, color_id: str = "4",
            cancel=None, stop_mode: str = "learned") -> tuple[list[Placement], dict]:
    """stop_mode: "learned" lets the policy choose STOP; "never" masks STOP so decoding runs until the legal
    candidate set is empty or the piece cap is reached (the baseline's own stopping rule)."""
    grid = target.shape
    filled = np.zeros_like(target)
    placements: list[Placement] = []
    stop_reason = "cap"
    for _ in range(max_pieces):
        if cancel and cancel():
            stop_reason = "cancelled"
            break
        c = enumerate_candidates(target, filled, catalog)
        if len(c) == 0:
            stop_reason = "no_legal_candidate"
            break
        cell_idx, cell_mask, feats = candidate_tensors(c, grid)
        batch = collate([{"vox": voxel_input(target, filled), "cell_idx": cell_idx, "cell_mask": cell_mask, "feats": feats, "target_index": 0}], device)
        logits = model(batch["vox"], batch["cell_idx"], batch["cell_mask"], batch["feats"], batch["cand_mask"])[0]
        if stop_mode == "never":
            logits = logits.clone()
            logits[-1] = float("-inf")
        if temperature > 0 and rng is not None:
            probs = torch.softmax(logits / temperature, -1).cpu().numpy().astype(np.float64)
            probs = np.nan_to_num(probs) / max(np.nan_to_num(probs).sum(), 1e-12)
            choice = int(rng.choice(len(probs), p=probs))
        else:
            choice = int(logits.argmax().item())
        if choice == len(c):
            stop_reason = "stop"
            break
        i = choice
        w, d = int(c.w[i]), int(c.d[i])
        x, y, k = int(c.x[i]), int(c.y[i]), int(c.k[i])
        placements.append(Placement(id=f"b{len(placements) + 1:04d}", part_key=PART_ORDER[int(c.part[i])], x=x, y=y, z=k * BRICK_HEIGHT_PLATES, rotation_deg=int(c.rot[i]), color_id=color_id))
        filled[x:x + w, y:y + d, k] = True
    return placements, {"stop_reason": stop_reason}


class SFTGenerator:
    def __init__(self, ckpt: Path = DEFAULT_CKPT, device: str | None = None, stop_mode: str = "learned", name: str = GENERATOR_NAME):
        self.stop_mode, self.name = stop_mode, name
        self.device = torch.device(device or DEFAULT_DEVICE)
        self.ckpt_path = ckpt
        self.model, self.ck = load_policy(ckpt, self.device)
        self.version = self.ck.get("version", "0.1.0")

    def __call__(self, target: np.ndarray, catalog: Catalog, constraints: Constraints, seed: int = 0, cancellation=None,
                 build_id: str = "build", time_budget_s: float | None = None, temperature: float = 0.0) -> CandidateResult:
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        cancel = (lambda: (cancellation and cancellation()) or (time_budget_s is not None and time.perf_counter() - t0 > time_budget_s))
        raw, diag = rollout(self.model, target, catalog, constraints.max_pieces, self.device, temperature, rng, constraints.color_id, cancel,
                            stop_mode=self.stop_mode)
        strategy = ("greedy_decode" if temperature == 0 else f"sample_t{temperature}") + ("" if self.stop_mode == "learned" else f"_stop_{self.stop_mode}")
        info = GeneratorInfo(name=self.name, version=self.version, seed=seed, strategy=strategy)
        build, report, m = finalize_candidate(raw, target, catalog, constraints, info, build_id)
        diagnostics = {"checkpoint": str(self.ckpt_path.name), "raw_pieces": len(raw), **diag, **{k: v for k, v in m.items() if k != "plan_diagnostics"},
                       "target_cells": int(target.sum()), "seconds": round(time.perf_counter() - t0, 3)}
        if not report["ok"] or not build.placements:
            diagnostics["errors"] = [e["code"] for e in report["errors"]]
            return CandidateResult("no_valid_candidate", None, 0.0, 0.0, 0.0, diagnostics)
        res = CandidateResult("validated_candidate", build, m["iou"], m["coverage"], m["extraneous"], diagnostics, m["instruction_status"], m["plan_diagnostics"])
        if res.coverage < 0.8 or m["dropped_disconnected"]:
            res.status = "partial_candidate"
        return res


_cached: dict[tuple, SFTGenerator] = {}


def sft_generate(target, catalog, constraints, seed=0, cancellation=None, build_id="build", time_budget_s=None, ckpt: Path = DEFAULT_CKPT,
                 stop_mode: str = "learned", name: str = GENERATOR_NAME):
    key = (str(ckpt), stop_mode, name)
    if key not in _cached:
        _cached[key] = SFTGenerator(ckpt, stop_mode=stop_mode, name=name)
    return _cached[key](target, catalog, constraints, seed, cancellation, build_id, time_budget_s)


def sft_generate_nostop(target, catalog, constraints, seed=0, cancellation=None, build_id="build", time_budget_s=None, ckpt: Path = DEFAULT_CKPT):
    return sft_generate(target, catalog, constraints, seed, cancellation, build_id, time_budget_s, ckpt, stop_mode="never", name="sft_v1_nostop")


register(GENERATOR_NAME, sft_generate)
register("sft_v1_nostop", sft_generate_nostop)
