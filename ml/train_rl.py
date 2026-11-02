"""PPO fine-tuning of the candidate-scoring policy (M6). Episodes run in forked CPU workers; updates run on Metal.

Reward is terminal: IoU of the build *after* the shared finalize (largest ground-connected component, planner,
validator) plus a stability bonus, so the policy is scored on exactly what the user receives. Advantage is
reward minus the value head's estimate (gamma = 1, terminal reward only). A KL penalty to the frozen initial
(SFT) policy keeps the fine-tune on-distribution.

Gate (--overfit N): train on N examples and verify the after-trim IoU on those examples rises above the initial policy.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from brickcore.catalog import load_catalog
from brickcore.schema import Bounds, Constraints, GeneratorInfo

from .data import DATA_DIR, Example, apply_symmetry, load_examples, target_from_example
from .policy import BrickPolicy, collate
from .rl_env import AssemblyState, RewardConfig, load_policy_for_rl, rollout

ROOT = Path(__file__).resolve().parent.parent
_W: dict = {}  # worker globals, populated before fork


def _episode(model: BrickPolicy, ex: Example, catalog, sym: int, seed: int, cfg: dict, sample: bool, record: bool) -> dict:
    target = target_from_example(ex)
    if sym:
        target, _ = apply_symmetry(target, [], sym, catalog)
    st = AssemblyState(target, catalog, conn_feats=cfg["conn_feats"], main_channel=cfg["main_channel"])
    out = rollout(model, st, ex.budget, torch.device("cpu"), sample=sample, rng=np.random.default_rng(seed), record=record)
    bounds = Bounds(x_studs=target.shape[0], y_studs=target.shape[1], z_plates=target.shape[2] * 3)
    fin = st.finish(Constraints(max_pieces=ex.budget, bounds=bounds, color_id="4"), GeneratorInfo(name="rl_train", version="0", seed=seed), ex.id,
                    RewardConfig(stability_weight=cfg["stab_weight"]))
    info = {k: v for k, v in fin.items() if k not in ("build", "report", "metrics")}
    info.update({"id": ex.id, "family": ex.family, "teacher_iou": ex.iou, "stop_reason": out["stop_reason"], "sym": sym})
    return {"steps": out["steps"], "info": info}


def _worker(task: list[tuple[int, int, int]]) -> list[dict]:
    torch.set_num_threads(1)
    model = BrickPolicy(**_W["arch"])
    model.load_state_dict(_W["state_dict"])
    model.eval()
    return [_episode(model, _W["examples"][i], _W["catalog"], sym, seed, _W["cfg"], _W["sample"], _W["record"]) for i, sym, seed in task]


def collect(model: BrickPolicy, examples: list[Example], catalog, tasks: list[tuple[int, int, int]], cfg: dict, workers: int,
            sample: bool, record: bool) -> list[dict]:
    _W.update({"arch": model.arch, "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "examples": examples,
               "catalog": catalog, "cfg": cfg, "sample": sample, "record": record})
    if workers <= 0:
        return _worker(tasks)
    chunks = [tasks[i::workers] for i in range(workers)]
    chunks = [c for c in chunks if c]
    with mp.get_context("fork").Pool(len(chunks)) as pool:
        parts = pool.map(_worker, chunks)
    return [e for part in parts for e in part]


def summarize(eps: list[dict]) -> dict:
    infos = [e["info"] for e in eps]
    n = max(len(infos), 1)
    stops = {}
    for i in infos:
        stops[i["stop_reason"]] = stops.get(i["stop_reason"], 0) + 1
    return {"n": len(infos), "reward": round(sum(i["reward"] for i in infos) / n, 4), "iou": round(sum(i["iou"] for i in infos) / n, 4),
            "teacher_iou": round(sum(i["teacher_iou"] for i in infos) / n, 4), "valid_rate": round(sum(i["valid"] for i in infos) / n, 3),
            "stable_rate": round(sum(i["stable"] for i in infos) / n, 3), "fragmented_rate": round(sum(i["components"] > 1 for i in infos) / n, 3),
            "pieces": round(sum(i["pieces"] for i in infos) / n, 1), "dropped": round(sum(i["dropped"] for i in infos) / n, 2), "stops": stops,
            "at_or_above_teacher": sum(i["valid"] and i["iou"] >= i["teacher_iou"] - 1e-6 for i in infos)}


def ppo_update(model: BrickPolicy, ref: BrickPolicy, opt, eps: list[dict], args, device) -> dict:
    steps = []
    for e in eps:
        R = e["info"]["reward"]
        for s in e["steps"]:
            steps.append({**s, "ret": R, "adv": R - s["value"]})
    if not steps:
        return {"n_steps": 0}
    adv = np.array([s["adv"] for s in steps], dtype=np.float32)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    for s, a in zip(steps, adv, strict=True):
        s["adv_n"] = float(a)
    rng = random.Random(args.seed)
    tot = {"pg": 0.0, "vf": 0.0, "ent": 0.0, "kl": 0.0, "approx_kl": 0.0, "clipfrac": 0.0}
    n_mb = 0
    for _ in range(args.epochs):
        rng.shuffle(steps)
        for i in range(0, len(steps), args.minibatch):
            mb = steps[i:i + args.minibatch]
            batch = collate(mb, device)
            acts = torch.tensor([mb_i["action"] for mb_i in mb], dtype=torch.long, device=device)
            old_logp = torch.tensor([mb_i["logp"] for mb_i in mb], dtype=torch.float32, device=device)
            A = torch.tensor([mb_i["adv_n"] for mb_i in mb], dtype=torch.float32, device=device)
            ret = torch.tensor([mb_i["ret"] for mb_i in mb], dtype=torch.float32, device=device)
            logits, values = model.forward_with_value(batch["vox"], batch["cell_idx"], batch["cell_mask"], batch["feats"], batch["cand_mask"])
            logp_all = torch.log_softmax(logits, -1)
            logp = logp_all.gather(1, acts[:, None]).squeeze(1)
            ratio = (logp - old_logp).exp()
            pg = -torch.min(ratio * A, ratio.clamp(1 - args.clip, 1 + args.clip) * A).mean()
            vf = ((values - ret) ** 2).mean()
            # masked (padded) actions have logp = -inf; zero them out *before* multiplying so no NaN reaches the backward pass
            finite = torch.isfinite(logp_all)
            logp_safe = torch.where(finite, logp_all, torch.zeros_like(logp_all))
            p = torch.where(finite, logp_all.exp(), torch.zeros_like(logp_all))
            ent = -(p * logp_safe).sum(-1).mean()
            with torch.no_grad():
                ref_logp = torch.log_softmax(ref(batch["vox"], batch["cell_idx"], batch["cell_mask"], batch["feats"], batch["cand_mask"]), -1)
                ref_safe = torch.where(finite, ref_logp, torch.zeros_like(ref_logp))
            kl = (p * (logp_safe - ref_safe)).sum(-1).mean()
            loss = pg + args.vf * vf + args.kl * kl - args.ent * ent
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                for key, val in (("pg", pg), ("vf", vf), ("ent", ent), ("kl", kl), ("approx_kl", (old_logp - logp).mean()),
                                 ("clipfrac", ((ratio - 1).abs() > args.clip).float().mean())):
                    tot[key] += val.item()
            n_mb += 1
    return {"n_steps": len(steps), "minibatches": n_mb, **{k: round(v / max(n_mb, 1), 5) for k, v in tot.items()}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA_DIR / "examples.jsonl"))
    ap.add_argument("--init", default=str(ROOT / "checkpoints" / "sft_v1.pt"))
    ap.add_argument("--name", default="rl_v1")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--episodes", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--kl", type=float, default=0.05)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--stab-weight", type=float, default=0.1)
    ap.add_argument("--no-conn-feats", action="store_true")
    ap.add_argument("--no-main-channel", action="store_true")
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--overfit", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-n", type=int, default=64)
    ap.add_argument("--real-frac", type=float, default=0.0, help="fraction of episodes (and of the eval subset) drawn from real meshes; 0 = natural mix")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device or ("mps" if torch.backends.mps.is_available() else "cpu"))
    catalog = load_catalog()
    train = load_examples(Path(args.data), "train")
    val = load_examples(Path(args.data), "val")
    if args.overfit:
        train = train[:args.overfit]
        val = train
    real_idx = [i for i, e in enumerate(train) if e.family == "thingi"]
    synth_idx = [i for i, e in enumerate(train) if e.family != "thingi"]
    real_frac = args.real_frac if (real_idx and synth_idx) else 0.0
    if real_frac:
        # the eval subset mirrors the episode mix, so checkpoint selection is made on the distribution being trained
        vr, vs = [e for e in val if e.family == "thingi"], [e for e in val if e.family != "thingi"]
        k = min(round(args.eval_n * real_frac), len(vr))
        er = random.Random(1234)
        eval_set = er.sample(vr, k) + er.sample(vs, min(args.eval_n - k, len(vs)))
    else:
        eval_set = val if len(val) <= args.eval_n else random.Random(1234).sample(val, args.eval_n)
    print(f"train={len(train)} real={len(real_idx)} real_frac={real_frac} eval_n={len(eval_set)} eval_real={sum(e.family == 'thingi' for e in eval_set)}", file=sys.stderr)
    cfg = {"conn_feats": not args.no_conn_feats, "main_channel": not args.no_main_channel, "stab_weight": args.stab_weight}

    model, init_ck = load_policy_for_rl(Path(args.init), device, cfg["conn_feats"], cfg["main_channel"])
    ref, _ = load_policy_for_rl(Path(args.init), device, cfg["conn_feats"], cfg["main_channel"])
    ref.eval()
    for p_ in ref.parameters():
        p_.requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    run_dir = ROOT / "runs" / "rl" / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    log = (run_dir / "metrics.jsonl").open("a")
    ck_dir = ROOT / "checkpoints"
    ck_dir.mkdir(exist_ok=True)

    def evaluate(tag: str) -> dict:
        model.eval()
        eps = collect(model, eval_set, catalog, [(i, 0, 0) for i in range(len(eval_set))], cfg, args.workers, sample=False, record=False)
        model.train()
        return summarize(eps)

    t0 = time.perf_counter()
    init_eval = evaluate("init")
    print(json.dumps({"iter": 0, "eval": init_eval, "note": "initial policy (SFT weights, padded)"}), file=sys.stderr)
    log.write(json.dumps({"iter": 0, "eval": init_eval}) + "\n")
    best = init_eval["iou"]
    rng = random.Random(args.seed)
    for it in range(1, args.iters + 1):
        t1 = time.perf_counter()
        def pick() -> int:
            if real_frac and rng.random() < real_frac:
                return rng.choice(real_idx)
            return rng.choice(synth_idx) if real_frac else rng.randrange(len(train))

        tasks = [(pick(), rng.randrange(8) if args.augment else 0, rng.randrange(1 << 30)) for _ in range(args.episodes)]
        model.eval()
        eps = collect(model, train, catalog, tasks, cfg, args.workers, sample=True, record=True)
        model.train()
        t2 = time.perf_counter()
        upd = ppo_update(model, ref, opt, eps, args, device)
        rec = {"iter": it, "rollout": summarize(eps), "update": upd, "rollout_s": round(t2 - t1, 1), "update_s": round(time.perf_counter() - t2, 1),
               "t": round(time.perf_counter() - t0, 1)}
        if it % args.eval_every == 0 or it == args.iters:
            ev = evaluate(f"iter{it}")
            rec["eval"] = ev
            if ev["iou"] > best:
                best = ev["iou"]
                torch.save({"arch": model.arch, "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "version": "0.1.0",
                            "name": args.name, "iter": it, "eval": ev, "init": str(args.init), "cfg": cfg, "args": vars(args), "seed": args.seed},
                           ck_dir / f"{args.name}.pt")
                rec["saved"] = True
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), file=sys.stderr)
    final = evaluate("final")
    out = {"init_eval": init_eval, "final_eval": final, "best_eval_iou": best}
    (run_dir / "final_eval.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out))
    if args.overfit:
        gate = final["iou"] > init_eval["iou"] + 1e-6 or best > init_eval["iou"] + 1e-6
        print(f"RL GATE {'PASSED' if gate else 'FAILED'}: after-trim IoU init {init_eval['iou']} -> best {best} (final {final['iou']})", file=sys.stderr)
        return 0 if gate else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
