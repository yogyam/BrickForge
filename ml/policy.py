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


class BrickPolicy(nn.Module):
    def __init__(self, channels: int = 32, hidden: int = 64, layers: int = 4):
        super().__init__()
        convs, c_in = [], 3
        for _ in range(layers):
            convs += [nn.Conv3d(c_in, channels, 3, padding=1), nn.GELU()]
            c_in = channels
        self.encoder = nn.Sequential(*convs)
        self.cand_mlp = nn.Sequential(nn.Linear(channels + CAND_FEATS, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.stop_mlp = nn.Sequential(nn.Linear(2 * channels, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.channels = channels

    def encode(self, vox: torch.Tensor) -> torch.Tensor:
        return self.encoder(vox)  # [B, C, X, Y, Z]

    def forward(self, vox: torch.Tensor, cell_idx: torch.Tensor, cell_mask: torch.Tensor, cand_feats: torch.Tensor, cand_mask: torch.Tensor) -> torch.Tensor:
        """vox [B,3,X,Y,Z]; cell_idx [B,N,8] flat voxel indices (-1 pad); cell_mask [B,N,8]; cand_feats [B,N,F];
        cand_mask [B,N]. Returns logits [B, N+1] with the last column = STOP; padded candidates get -inf."""
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
        return torch.cat([cand_logits, stop_logit], dim=-1)


def voxel_input(target: np.ndarray, filled: np.ndarray) -> np.ndarray:
    return np.stack([target, filled, target & ~filled]).astype(np.float32)


def candidate_tensors(c: Candidates, grid: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-candidate flat voxel indices (padded to 8 cells), mask, and normalised scalar features."""
    nx, ny, nz = grid
    n = len(c)
    cell_idx = np.full((n, 8), -1, dtype=np.int64)
    cell_mask = np.zeros((n, 8), dtype=bool)
    feats = np.zeros((n, CAND_FEATS), dtype=np.float32)
    for i in range(n):
        cells = [(xx * ny + yy) * nz + int(c.k[i]) for xx in range(c.x[i], c.x[i] + c.w[i]) for yy in range(c.y[i], c.y[i] + c.d[i])]
        cell_idx[i, :len(cells)] = cells
        cell_mask[i, :len(cells)] = True
        feats[i, int(c.part[i])] = 1.0
        feats[i, N_PARTS:] = [c.rot[i] / 90.0, c.x[i] / nx, c.y[i] / ny, c.k[i] / nz, c.new_target[i] / 8.0, c.filler[i] / 8.0, c.support[i] / 8.0]
    return cell_idx, cell_mask, feats


def collate(items: list[dict], device: torch.device) -> dict:
    """items: dicts with vox [3,X,Y,Z], cell_idx [N,8], cell_mask [N,8], feats [N,F], target_index (int, N = STOP)."""
    B = len(items)
    N = max(max(it["cell_idx"].shape[0] for it in items), 1)
    vox = torch.from_numpy(np.stack([it["vox"] for it in items]))
    cell_idx = torch.full((B, N, 8), -1, dtype=torch.long)
    cell_mask = torch.zeros((B, N, 8), dtype=torch.bool)
    feats = torch.zeros((B, N, CAND_FEATS))
    cand_mask = torch.zeros((B, N), dtype=torch.bool)
    labels = torch.zeros(B, dtype=torch.long)
    for b, it in enumerate(items):
        n = it["cell_idx"].shape[0]
        if n:
            cell_idx[b, :n] = torch.from_numpy(it["cell_idx"])
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
