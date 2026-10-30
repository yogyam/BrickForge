"""Supervised training of BrickPolicy on baseline-generated action sequences (teacher forcing, legal-action masks).

Gate 1 (--overfit N): fit N training examples, then free-run every one of them through the real decoder,
planner and validator and report how many reconstruct. Do not scale data before this passes.
Metrics go to runs/sft/<name>/metrics.jsonl; checkpoints to checkpoints/<name>.pt.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from brickcore.candidates import enumerate_candidates
from brickcore.catalog import load_catalog
from brickcore.schema import BRICK_HEIGHT_PLATES

from .data import (
    DATA_DIR,
    Example,
    apply_symmetry,
    load_examples,
    placements_from_example,
    target_from_example,
)
from .policy import BrickPolicy, candidate_tensors, collate, loss_and_acc, voxel_input

ROOT = Path(__file__).resolve().parent.parent


def pick_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def make_items(ex: Example, catalog, sym: int = 0, steps: list[int] | None = None) -> tuple[list[dict], int]:
    """One training item per decision step (including STOP). Returns (items, n_teacher_not_in_candidates)."""
    target = target_from_example(ex)
    placements = placements_from_example(ex)
    if sym:
        target, placements = apply_symmetry(target, placements, sym, catalog)
    grid = target.shape
    filled = np.zeros_like(target)
    items, missing = [], 0
    idxs = steps if steps is not None else range(len(placements) + 1)
    for t in range(len(placements) + 1):
        if t in idxs:
            c = enumerate_candidates(target, filled, catalog)
            if t < len(placements):
                p = placements[t]
                ti = c.find(p.part_key, p.rotation_deg, p.x, p.y, p.z // BRICK_HEIGHT_PLATES)
                if ti < 0:
                    missing += 1
                    ti = None
            else:
                ti = -1  # STOP
            if ti is not None:
                cell_idx, cell_mask, feats = candidate_tensors(c, grid)
                items.append({"vox": voxel_input(target, filled), "cell_idx": cell_idx, "cell_mask": cell_mask, "feats": feats, "target_index": ti})
        if t < len(placements):
            p = placements[t]
            w, d = catalog.part(p.part_key).footprint(p.rotation_deg)
            filled[p.x:p.x + w, p.y:p.y + d, p.z // BRICK_HEIGHT_PLATES] = True
    return items, missing


def free_run_eval(model, examples: list[Example], catalog, device, max_n: int | None = None) -> dict:
    from brickcore.generate import finalize_candidate
    from brickcore.schema import Bounds, Constraints, GeneratorInfo

    from .sft_generator import rollout
    model.eval()
    rows = []
    for ex in examples[:max_n]:
        target = target_from_example(ex)
        bounds = Bounds(x_studs=target.shape[0], y_studs=target.shape[1], z_plates=target.shape[2] * 3)
        cons = Constraints(max_pieces=ex.budget, bounds=bounds, color_id="4")
        raw, diag = rollout(model, target, catalog, ex.budget, device)
        build, report, m = finalize_candidate(raw, target, catalog, cons, GeneratorInfo(name="sft_eval", version="0", seed=0), ex.id)
        rows.append({"id": ex.id, "teacher_iou": ex.iou, "iou": round(m["iou"], 4), "valid": m["valid"], "pieces": m["pieces"], "raw_pieces": len(raw),
                     "stop_reason": diag["stop_reason"], "planned": m["instruction_status"] == "planned", "teacher_pieces": len(ex.placements)})
    model.train()
    n = max(len(rows), 1)
    return {"n": len(rows), "mean_iou": round(sum(r["iou"] for r in rows) / n, 4), "mean_teacher_iou": round(sum(r["teacher_iou"] for r in rows) / n, 4),
            "valid_rate": round(sum(r["valid"] for r in rows) / n, 3), "planned_rate": round(sum(r["planned"] for r in rows) / n, 3),
            "reconstructed": sum(r["valid"] and r["iou"] >= r["teacher_iou"] - 1e-6 for r in rows), "rows": rows}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="train_sft")
    ap.add_argument("--data", default=str(DATA_DIR / "examples.jsonl"))
    ap.add_argument("--name", default="sft_v1")
    ap.add_argument("--overfit", type=int, default=0, help="gate 1: use only the first N train examples and verify reconstruction")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--augment", action="store_true", help="random 90-degree rotations / flips")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-n", type=int, default=32)
    ap.add_argument("--profile", action="store_true", help="time 50 steps on the chosen device and exit")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)
    catalog = load_catalog()
    train = load_examples(Path(args.data), "train")
    val = load_examples(Path(args.data), "val")
    if args.overfit:
        train = train[:args.overfit]
        val = train
    run_dir = ROOT / "runs" / "sft" / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    ck_dir = ROOT / "checkpoints"
    ck_dir.mkdir(exist_ok=True)
    log = (run_dir / "metrics.jsonl").open("a")

    # Pre-build teacher items (cheap for small sets; for large sets sample a subset of steps per example per epoch).
    t0 = time.perf_counter()
    items, missing = [], 0
    syms = range(8) if args.augment else [0]
    for ex in train:
        for sym in syms:
            its, miss = make_items(ex, catalog, sym=sym)
            items += its
            missing += miss
    print(f"device={device} train_examples={len(train)} decision_items={len(items)} teacher_not_in_candidates={missing} prep_s={time.perf_counter() - t0:.1f}", file=sys.stderr)

    arch = {"channels": args.channels, "hidden": args.hidden, "layers": args.layers}
    model = BrickPolicy(**arch).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.steps, 1))
    print(f"params={n_params}", file=sys.stderr)

    rng = random.Random(args.seed)
    if args.profile:
        batch = collate(rng.sample(items, min(args.batch, len(items))), device)
        for _ in range(5):
            loss, _ = loss_and_acc(model, batch); opt.zero_grad(); loss.backward(); opt.step()
        if device.type == "mps":
            torch.mps.synchronize()
        t1 = time.perf_counter()
        for _ in range(50):
            loss, _ = loss_and_acc(model, batch); opt.zero_grad(); loss.backward(); opt.step()
        if device.type == "mps":
            torch.mps.synchronize()
        dt = (time.perf_counter() - t1) / 50
        print(json.dumps({"device": str(device), "params": n_params, "batch": args.batch, "step_s": round(dt, 4), "steps_per_s": round(1 / dt, 2), "items_per_s": round(args.batch / dt, 1)}))
        return 0

    best_iou = -1.0
    for step in range(1, args.steps + 1):
        batch_items = rng.sample(items, min(args.batch, len(items)))
        batch = collate(batch_items, device)
        loss, acc = loss_and_acc(model, batch)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 25 == 0 or step == 1:
            rec = {"step": step, "loss": round(loss.item(), 4), "acc": round(acc, 4), "lr": sched.get_last_lr()[0], "t": round(time.perf_counter() - t0, 1)}
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(rec, file=sys.stderr)
        if step % args.eval_every == 0 or step == args.steps:
            ev = free_run_eval(model, val, catalog, device, args.eval_n)
            rec = {"step": step, "eval": {k: v for k, v in ev.items() if k != "rows"}}
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(rec, file=sys.stderr)
            if ev["mean_iou"] > best_iou:
                best_iou = ev["mean_iou"]
                torch.save({"arch": arch, "state_dict": model.state_dict(), "version": "0.1.0", "name": args.name, "step": step, "eval": rec["eval"],
                            "data": str(args.data), "n_train_examples": len(train), "seed": args.seed}, ck_dir / f"{args.name}.pt")
    final = free_run_eval(model, val, catalog, device, None if args.overfit else args.eval_n)
    (run_dir / "final_eval.json").write_text(json.dumps(final, indent=2) + "\n")
    summary = {k: v for k, v in final.items() if k != "rows"}
    print(json.dumps(summary))
    if args.overfit:
        gate = final["reconstructed"] >= int(0.9 * final["n"])
        print(f"OVERFIT GATE {'PASSED' if gate else 'FAILED'}: {final['reconstructed']}/{final['n']} reconstructed at or above teacher IoU", file=sys.stderr)
        return 0 if gate else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
