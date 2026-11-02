"""Compact shape-conditioned policy: 3D CNN over (target, assembly, remaining) scores a bounded set of legal
placements plus STOP. Legal-action masking is inherent: only enumerated candidates receive logits."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from brickcore.candidates import PART_ORDER, Candidates

N_PARTS = len(PART_ORDER)
CAND_FEATS = N_PARTS + 7  # part one-hot, rot, x, y, k, new_target, filler, support (normalised)
CONN_FEATS = 2            # RL extension: joins_main_component, merges_two_components


class BrickPolicy(nn.Module):
    """in_channels: 3 (target, assembly, remaining) or 4 (+ main ground-connected component).
    extra_feats: 0, or CONN_FEATS for the connectivity features. value head is used by RL only."""

    def __init__(self, channels: int = 32, hidden: int = 64, layers: int = 4, in_channels: int = 3, extra_feats: int = 0):
        super().__init__()
        convs, c_in = [], in_channels
        for _ in range(layers):
            convs += [nn.Conv3d(c_in, channels, 3, padding=1), nn.GELU()]
            c_in = channels
        self.encoder = nn.Sequential(*convs)
        self.cand_mlp = nn.Sequential(nn.Linear(channels + CAND_FEATS + extra_feats, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.stop_mlp = nn.Sequential(nn.Linear(2 * channels, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.value_mlp = nn.Sequential(nn.Linear(2 * channels, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.channels, self.in_channels, self.extra_feats = channels, in_channels, extra_feats

    @property
    def arch(self) -> dict:
        return {"channels": self.channels, "hidden": self.cand_mlp[0].out_features, "layers": len(self.encoder) // 2,
                "in_channels": self.in_channels, "extra_feats": self.extra_feats}

    def value(self, vox: torch.Tensor) -> torch.Tensor:
        flat = self.encode(vox).flatten(2).transpose(1, 2)
        return self.value_mlp(torch.cat([flat.mean(1), flat.amax(1)], dim=-1)).squeeze(-1)

    def forward_with_value(self, vox, cell_idx, cell_mask, cand_feats, cand_mask) -> tuple[torch.Tensor, torch.Tensor]:
        logits, g = self._forward(vox, cell_idx, cell_mask, cand_feats, cand_mask)
        return logits, self.value_mlp(g).squeeze(-1)

    def encode(self, vox: torch.Tensor) -> torch.Tensor:
        return self.encoder(vox)  # [B, C, X, Y, Z]

    def forward(self, vox: torch.Tensor, cell_idx: torch.Tensor, cell_mask: torch.Tensor, cand_feats: torch.Tensor, cand_mask: torch.Tensor) -> torch.Tensor:
        """vox [B,3,X,Y,Z]; cell_idx [B,N,8] flat voxel indices (-1 pad); cell_mask [B,N,8]; cand_feats [B,N,F];
        cand_mask [B,N]. Returns logits [B, N+1] with the last column = STOP; padded candidates get -inf."""
        return self._forward(vox, cell_idx, cell_mask, cand_feats, cand_mask)[0]

    def _forward(self, vox, cell_idx, cell_mask, cand_feats, cand_mask) -> tuple[torch.Tensor, torch.Tensor]:
        B = vox.shape[0]
        fmap = self.encode(vox)                                  # [B,C,X,Y,Z]
        flat = fmap.flatten(2).transpose(1, 2)                   # [B,V,C]
        idx = cell_idx.clamp(min=0)
        gathered = torch.gather(flat, 1, idx.flatten(1).unsqueeze(-1).expand(-1, -1, self.channels))  # [B,N*8,C]
        gathered = gathered.view(B, cell_idx.shape[1], cell_idx.shape[2], self.channels)
        m = cell_mask.unsqueeze(-1).to(gathered.dtype)
        pooled = (gathered * m).sum(2) / m.sum(2).clamp(min=1)   # [B,N,C]
        cand_logits = self.cand_mlp(torch.cat([pooled, cand_feats], dim=-1)).squeeze(-1)  # [B,N]
        cand_logits = cand_logits.masked_fill(~cand_mask, float("-inf"))
        g = torch.cat([flat.mean(1), flat.amax(1)], dim=-1)      # [B,2C]
        stop_logit = self.stop_mlp(g)                            # [B,1]
        return torch.cat([cand_logits, stop_logit], dim=-1), g


def extend_state_dict(sd: dict, in_channels: int, extra_feats: int) -> dict:
    """Zero-pad a 3-channel / base-feature checkpoint to the RL input layout, so the extended policy computes
    exactly the same logits as the SFT policy until the new weights move. The value head stays freshly initialised."""
    sd = dict(sd)
    w = sd["encoder.0.weight"]
    if w.shape[1] < in_channels:
        pad = torch.zeros(w.shape[0], in_channels - w.shape[1], *w.shape[2:], dtype=w.dtype)
        sd["encoder.0.weight"] = torch.cat([w, pad], dim=1)
    w = sd["cand_mlp.0.weight"]
    want = sd["encoder.0.weight"].shape[0] + CAND_FEATS + extra_feats  # channels + base features + connectivity features
    if w.shape[1] < want:
        sd["cand_mlp.0.weight"] = torch.cat([w, torch.zeros(w.shape[0], want - w.shape[1], dtype=w.dtype)], dim=1)
    return sd


def voxel_input(target: np.ndarray, filled: np.ndarray, main: np.ndarray | None = None) -> np.ndarray:
    chans = [target, filled, target & ~filled] + ([main] if main is not None else [])
    return np.stack(chans).astype(np.float32)


def candidate_tensors(c: Candidates, grid: tuple[int, int, int], extra: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-candidate flat voxel indices (padded to 8 cells), mask, and normalised scalar features
    (optionally concatenated with `extra` [n, E] connectivity features)."""
    nx, ny, nz = grid
    n = len(c)
    n_feats = CAND_FEATS + (extra.shape[1] if extra is not None else 0)
    if n == 0:
        return np.full((0, 8), -1, dtype=np.int64), np.zeros((0, 8), dtype=bool), np.zeros((0, n_feats), dtype=np.float32)
    x, y, k = c.x.astype(np.int64), c.y.astype(np.int64), c.k.astype(np.int64)
    w, d = c.w.astype(np.int64), c.d.astype(np.int64)
    # Footprints are at most 4x4 with w*d <= 8 cells: enumerate a 4x4 offset grid, mask, and compact valid cells to the front.
    dx = np.repeat(np.arange(4), 4)[None, :]
    dy = np.tile(np.arange(4), 4)[None, :]
    valid = (dx < w[:, None]) & (dy < d[:, None])  # [n,16], row-major over (dx, dy) like the reference loop
    cells = ((x[:, None] + dx) * ny + (y[:, None] + dy)) * nz + k[:, None]
    order = np.argsort(~valid, axis=1, kind="stable")[:, :8]
    cell_mask = np.take_along_axis(valid, order, axis=1)
    cell_idx = np.where(cell_mask, np.take_along_axis(cells, order, axis=1), -1)
    feats = np.zeros((n, n_feats), dtype=np.float32)
    feats[np.arange(n), c.part.astype(np.int64)] = 1.0
    feats[:, N_PARTS:CAND_FEATS] = np.stack([c.rot / 90.0, x / nx, y / ny, k / nz, c.new_target / 8.0, c.filler / 8.0, c.support / 8.0], axis=1)
    if extra is not None:
        feats[:, CAND_FEATS:] = extra
    return cell_idx, cell_mask, feats


def collate(items: list[dict], device: torch.device) -> dict:
    """items: dicts with vox [3,X,Y,Z], cell_idx [N,8], cell_mask [N,8], feats [N,F], target_index (int, N = STOP)."""
    B = len(items)
    N = max(max(it["cell_idx"].shape[0] for it in items), 1)
    n_feats = max(it["feats"].shape[1] for it in items) if items else CAND_FEATS
    vox = torch.from_numpy(np.stack([it["vox"] for it in items]))
    cell_idx = torch.full((B, N, 8), -1, dtype=torch.long)
    cell_mask = torch.zeros((B, N, 8), dtype=torch.bool)
    feats = torch.zeros((B, N, n_feats))
    cand_mask = torch.zeros((B, N), dtype=torch.bool)
    labels = torch.zeros(B, dtype=torch.long)
    for b, it in enumerate(items):
        n = it["cell_idx"].shape[0]
        if n:
            cell_idx[b, :n] = torch.from_numpy(it["cell_idx"].astype(np.int64))
            cell_mask[b, :n] = torch.from_numpy(it["cell_mask"])
            feats[b, :n] = torch.from_numpy(it["feats"])
            cand_mask[b, :n] = True
        labels[b] = N if it["target_index"] < 0 else it["target_index"]  # STOP is always the last column
    return {"vox": vox.to(device), "cell_idx": cell_idx.to(device), "cell_mask": cell_mask.to(device), "feats": feats.to(device),
            "cand_mask": cand_mask.to(device), "labels": labels.to(device)}


def loss_and_acc(model: BrickPolicy, batch: dict) -> tuple[torch.Tensor, float]:
    logits = model(batch["vox"], batch["cell_idx"], batch["cell_mask"], batch["feats"], batch["cand_mask"])
    loss = F.cross_entropy(logits, batch["labels"])
    acc = (logits.argmax(-1) == batch["labels"]).float().mean().item()
    return loss, acc
