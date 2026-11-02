"""Assembly environment for RL: the same legal candidate action set as SFT, plus connectivity tracking
(union-find over stud connections) that exposes the main ground-connected component to the policy, and a
terminal reward computed *after* the shared finalize step (component trim, planner, validator) plus a stability
term. Nothing here changes the rules; the validator stays authoritative."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from brickcore.candidates import PART_ORDER, Candidates, enumerate_candidates
from brickcore.catalog import Catalog
from brickcore.generate import finalize_candidate
from brickcore.schema import BRICK_HEIGHT_PLATES, Constraints, GeneratorInfo, Placement
from brickcore.stability import analyze

from .policy import CONN_FEATS, BrickPolicy, candidate_tensors, collate, extend_state_dict, voxel_input

DX = np.repeat(np.arange(4), 4)[None, :]
DY = np.tile(np.arange(4), 4)[None, :]


@dataclass
class RewardConfig:
    stability_weight: float = 0.1   # added when the final trimmed build is stable under the static force model
    invalid_penalty: float = 0.0    # subtracted when the final build fails validation (cannot happen under the candidate rules)


class AssemblyState:
    """Mutable build state. Pieces are stored in placement order; `root[i]` is the union-find root of piece i.
    A component is grounded if any member rests on the ground; the main component is the largest grounded one
    (ties: the one containing the earliest piece), matching `brickcore.generate._largest_ground_component`."""

    def __init__(self, target: np.ndarray, catalog: Catalog, color_id: str = "4", conn_feats: bool = True, main_channel: bool = True):
        self.target = target.astype(bool)
        self.catalog = catalog
        self.color_id = color_id
        self.conn_feats, self.main_channel = conn_feats, main_channel
        self.grid = target.shape
        self.filled = np.zeros_like(self.target)
        self.piece_grid = np.full(self.grid, -1, dtype=np.int32)
        self.placements: list[Placement] = []
        self.parent: list[int] = []
        self.size: list[int] = []
        self.grounded: list[bool] = []
        self.min_id: list[int] = []
        self._cands: Candidates | None = None
        self._extra: np.ndarray | None = None

    # ---- union-find
    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        self.grounded[ra] = self.grounded[ra] or self.grounded[rb]
        self.min_id[ra] = min(self.min_id[ra], self.min_id[rb])

    def main_root(self) -> int:
        best = -1
        for i in range(len(self.parent)):
            if self.parent[i] == i and self.grounded[i]:
                if best < 0 or (self.size[i], -self.min_id[i]) > (self.size[best], -self.min_id[best]):
                    best = i
        return best

    def roots_array(self) -> np.ndarray:
        return np.array([self.find(i) for i in range(len(self.parent))], dtype=np.int64) if self.parent else np.zeros(0, dtype=np.int64)

    def main_mask(self) -> np.ndarray:
        if not self.parent:
            return np.zeros(self.grid, dtype=bool)
        roots = self.roots_array()
        main = self.main_root()
        pg = self.piece_grid
        out = np.zeros(self.grid, dtype=bool)
        has = pg >= 0
        out[has] = roots[pg[has]] == main
        return out

    # ---- candidates and features
    def candidates(self) -> tuple[Candidates, np.ndarray | None]:
        if self._cands is None:
            c = enumerate_candidates(self.target, self.filled, self.catalog)
            self._cands = c
            self._extra = self._connectivity_features(c) if self.conn_feats else None
        return self._cands, self._extra

    def _support_roots(self, c: Candidates) -> np.ndarray:
        """[n, 16] root id of the piece under each footprint cell on the layer below (-1 none / out of footprint / ground layer)."""
        n = len(c)
        out = np.full((n, 16), -1, dtype=np.int64)
        if n == 0 or not self.parent:
            return out
        x, y, k = c.x.astype(np.int64), c.y.astype(np.int64), c.k.astype(np.int64)
        w, d = c.w.astype(np.int64), c.d.astype(np.int64)
        valid = (DX < w[:, None]) & (DY < d[:, None]) & (k[:, None] > 0)
        xx = np.minimum(x[:, None] + DX, self.grid[0] - 1)
        yy = np.minimum(y[:, None] + DY, self.grid[1] - 1)
        kk = np.maximum(k[:, None] - 1, 0)
        pid = self.piece_grid[xx, yy, np.broadcast_to(kk, xx.shape)]
        roots = self.roots_array()
        got = np.where(valid & (pid >= 0), roots[np.maximum(pid, 0)], -1)
        return got

    def _connectivity_features(self, c: Candidates) -> np.ndarray:
        n = len(c)
        feats = np.zeros((n, CONN_FEATS), dtype=np.float32)
        if n == 0:
            return feats
        sr = self._support_roots(c)
        main = self.main_root()
        joins_main = (sr == main).any(1) if main >= 0 else np.zeros(n, dtype=bool)
        srt = np.sort(sr, axis=1)
        distinct = ((srt[:, 1:] != srt[:, :-1]) & (srt[:, 1:] >= 0)).sum(1) + (srt[:, 0] >= 0)
        # a ground-layer piece starts a new component; it "joins" nothing and merges nothing
        feats[:, 0] = joins_main
        feats[:, 1] = distinct >= 2
        return feats

    def item(self) -> dict:
        """Policy input for the current state (same dict layout as SFT items)."""
        c, extra = self.candidates()
        cell_idx, cell_mask, feats = candidate_tensors(c, self.grid, extra)
        vox = voxel_input(self.target, self.filled, self.main_mask() if self.main_channel else None)
        return {"vox": vox, "cell_idx": cell_idx, "cell_mask": cell_mask, "feats": feats, "target_index": 0}

    # ---- transitions
    def step(self, i: int) -> None:
        c, _ = self.candidates()
        w, d = int(c.w[i]), int(c.d[i])
        x, y, k = int(c.x[i]), int(c.y[i]), int(c.k[i])
        pid = len(self.placements)
        self.placements.append(Placement(id=f"b{pid + 1:04d}", part_key=PART_ORDER[int(c.part[i])], x=x, y=y, z=k * BRICK_HEIGHT_PLATES,
                                         rotation_deg=int(c.rot[i]), color_id=self.color_id))
        self.parent.append(pid)
        self.size.append(1)
        self.grounded.append(k == 0)
        self.min_id.append(pid)
        # stud connections to pieces directly below and (when a piece is slid under an existing one) directly above
        if k > 0:
            below = np.unique(self.piece_grid[x:x + w, y:y + d, k - 1])
            for b in below[below >= 0]:
                self.union(pid, int(b))
        if k + 1 < self.grid[2]:
            above = np.unique(self.piece_grid[x:x + w, y:y + d, k + 1])
            for b in above[above >= 0]:
                self.union(pid, int(b))
        self.filled[x:x + w, y:y + d, k] = True
        self.piece_grid[x:x + w, y:y + d, k] = pid
        self._cands, self._extra = None, None

    def finish(self, constraints: Constraints, generator: GeneratorInfo, build_id: str, cfg: RewardConfig = RewardConfig()) -> dict:
        """Shared finalize (trim, plan, validate) then the terminal reward."""
        build, report, m = finalize_candidate(self.placements, self.target, self.catalog, constraints, generator, build_id)
        stable = False
        if report["ok"] and build.placements:
            stable = bool(analyze(build.placements, self.catalog)["stable"])
        reward = m["iou"] + cfg.stability_weight * float(stable) - (0.0 if report["ok"] else cfg.invalid_penalty)
        return {"reward": float(reward), "iou": m["iou"], "stable": stable, "valid": report["ok"], "pieces": m["pieces"],
                "raw_pieces": len(self.placements), "dropped": m["dropped_disconnected"], "components": m["components_before_trim"],
                "planned": m["instruction_status"] == "planned", "build": build, "report": report, "metrics": m}


def load_policy_for_rl(ckpt_path, device: torch.device, conn_feats: bool = True, main_channel: bool = True) -> tuple[BrickPolicy, dict]:
    """Load an SFT or RL checkpoint into the RL input layout. SFT weights are zero-padded (identical logits at init)."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch = dict(ck["arch"])
    want_in = 4 if main_channel else 3
    want_extra = CONN_FEATS if conn_feats else 0
    arch.setdefault("in_channels", 3)
    arch.setdefault("extra_feats", 0)
    if arch["in_channels"] > want_in or arch["extra_feats"] > want_extra:
        raise ValueError(f"checkpoint expects inputs {arch} but requested in_channels={want_in} extra_feats={want_extra}")
    sd = extend_state_dict(ck["state_dict"], want_in, want_extra)
    arch["in_channels"], arch["extra_feats"] = want_in, want_extra
    model = BrickPolicy(**arch)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected and all(k.startswith("value_mlp") for k in missing), (missing, unexpected)
    return model.to(device), ck


@torch.no_grad()
def rollout(model: BrickPolicy, state: AssemblyState, max_pieces: int, device: torch.device, sample: bool = False,
            rng: np.random.Generator | None = None, cancel=None, record: bool = False, stop_mode: str = "learned") -> dict:
    """Run one episode. With record=True returns per-step items/actions/logps/values for PPO.
    stop_mode "never" masks the STOP action (decode-time choice, made on the SFT test split; see docs/rl.md)."""
    steps = []
    stop_reason = "cap"
    for _ in range(max_pieces):
        if cancel and cancel():
            stop_reason = "cancelled"
            break
        c, _ = state.candidates()
        if len(c) == 0:
            stop_reason = "no_legal_candidate"
            break
        it = state.item()
        batch = collate([it], device)
        logits, value = model.forward_with_value(batch["vox"], batch["cell_idx"], batch["cell_mask"], batch["feats"], batch["cand_mask"])
        if stop_mode == "never":
            logits = logits.clone()
            logits[0, len(c)] = float("-inf")
        logp = torch.log_softmax(logits[0], -1)
        if sample:
            probs = logp.exp().cpu().numpy().astype(np.float64)
            probs = np.nan_to_num(probs)
            probs /= probs.sum()
            a = int((rng or np.random.default_rng()).choice(len(probs), p=probs))
        else:
            a = int(logits[0].argmax().item())
        if record:
            steps.append({**it, "action": a, "logp": float(logp[a].item()), "value": float(value[0].item()), "n_cands": len(c)})
        if a == len(c):
            stop_reason = "stop"
            break
        state.step(a)
    return {"steps": steps, "stop_reason": stop_reason}
