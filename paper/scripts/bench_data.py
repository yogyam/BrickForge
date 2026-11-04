"""Benchmark and training-log loaders for the paper figures.

The paper's fidelity convention: mean after-trim IoU over the attempted cases (every case except those whose status is
`failed` for all generators, i.e. the invalid-input fixture), with a build that is missing or invalid scored 0. This
differs from `ml.benchmark.summarize`, whose `mean_iou` averages over valid builds only. See docs/benchmark.md.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "docs" / "benchmarks"
RUNS = ROOT / "runs"
BUDGETS = (50, 100, 200)

# training seeds live in the generator name; the record's `seed` field is the decode seed (always 42)
SEED_FAMILIES = {
    "sft_v2": ["sft_v2", "sft_v2_s1"],
    "rl_v2": ["rl_v2", "rl_v2_s1"],
    "rl_v2_nostop": ["rl_v2_nostop", "rl_v2_s1_nostop"],
}


def family_of(generator: str) -> str:
    return re.sub(r"_s\d+", "", generator)


def load_records(*names: str) -> list[dict]:
    """Concatenate benchmark JSONL files; a (generator, case, budget) key seen twice keeps the first copy
    (greedy_v1 rows are repeated verbatim in every file, as the audits check)."""
    seen: dict[tuple, dict] = {}
    for name in names:
        for line in (BENCH / f"{name}.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            seen.setdefault((r["generator"], r["case"], r["budget"]), r)
    return list(seen.values())


def all_records() -> list[dict]:
    return load_records("rl_v2", "rl_v2_s1", "rl_v2_ablations", "rl_v1", "sft_v1", "baseline_v1")


def attempted_cases(records: list[dict]) -> list[str]:
    return sorted({r["case"] for r in records if r["status"] != "failed"})


def score(r: dict | None) -> float:
    if r is None or not r.get("valid") or not r.get("pieces"):
        return 0.0
    return float(r.get("iou") or 0.0)


def index(records: list[dict]) -> dict[tuple, dict]:
    return {(r["generator"], r["case"], r["budget"]): r for r in records}


def case_group(records: list[dict]) -> dict[str, str]:
    return {r["case"]: ("thingi10k" if r["group"] == "thingi10k" else "procedural") for r in records}


def attempted_mean(records: list[dict], generator: str, budget: int, group: str | None = None, cases: list[str] | None = None) -> float:
    by = index(records)
    grp = case_group(records)
    cs = [c for c in (cases or attempted_cases(records)) if group is None or grp[c] == group]
    return sum(score(by.get((generator, c, budget))) for c in cs) / len(cs)


def family_stats(records: list[dict], family: str, budget: int, group: str | None = None) -> tuple[float, float, float]:
    """(mean, min, max) over the training seeds of a generator family."""
    gens = SEED_FAMILIES.get(family, [family])
    vals = [attempted_mean(records, g, budget, group) for g in gens]
    return sum(vals) / len(vals), min(vals), max(vals)


def head_to_head(records: list[dict], a: str, b: str, budget: int = 200, tol: float = 1e-3) -> tuple[int, int, int]:
    by = index(records)
    w = lose = t = 0
    for c in attempted_cases(records):
        d = score(by.get((a, c, budget))) - score(by.get((b, c, budget)))
        if d > tol:
            w += 1
        elif d < -tol:
            lose += 1
        else:
            t += 1
    return w, lose, t


def counts(records: list[dict], generator: str, budget: int = 200) -> dict:
    by = index(records)
    rows = [by.get((generator, c, budget)) for c in attempted_cases(records)]
    n_total = len({r["case"] for r in records})  # 53, the invalid-input fixture included
    return {
        "valid": sum(1 for r in rows if r and r.get("valid")),
        "stable": sum(1 for r in rows if r and r.get("stable")),
        "n": n_total,
        "seconds": sum((r or {}).get("generate_s", 0.0) or 0.0 for r in rows) / len(rows),
        "stops": _stop_counts(rows),
        "fragmented_real": sum(1 for r in rows if r and r["group"] == "thingi10k" and (r.get("dropped_disconnected") or 0) > 0),
    }


def _stop_counts(rows: list[dict | None]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        if r and r.get("stop_reason"):
            out[r["stop_reason"]] = out.get(r["stop_reason"], 0) + 1
    return out


# ---- training curves -------------------------------------------------------------------------------------------

def rl_curve(run: str) -> dict:
    """Validation points every 10 iterations plus the per-iteration rollout IoU."""
    path = RUNS / "rl" / run / "metrics.jsonl"
    evals, rollout = [], []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if "rollout" in d:
            rollout.append((d["iter"], d["rollout"]["iou"]))
        if "eval" in d:
            e = d["eval"]
            evals.append({"iter": d.get("iter", 0), "iou": e["iou"], "teacher_iou": e["teacher_iou"],
                          "fragmented_rate": e["fragmented_rate"], "stable_rate": e["stable_rate"]})
    return {"evals": evals, "rollout": rollout}


def sft_curve(run: str) -> dict:
    path = RUNS / "sft" / run / "metrics.jsonl"
    evals, loss = [], []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if "eval" in d:
            evals.append({"step": d["step"], "iou": d["eval"]["mean_iou"], "teacher_iou": d["eval"]["mean_teacher_iou"]})
        elif "loss" in d:
            loss.append((d["step"], d["loss"]))
    return {"evals": evals, "loss": loss}
