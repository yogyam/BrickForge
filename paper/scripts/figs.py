"""Paper figures and numbers, scripted from docs/benchmarks/*.jsonl, runs/ metrics and cached builds.

    uv run --group ml --group paper python paper/scripts/figs.py <builds|pick-gallery|budget|scatter|ladder|curves|gallery|stops|numbers|check|all>

`builds` regenerates the gallery builds from the checkpoints (needs torch, checkpoints/ and data/thingi10k/) and
caches them under paper/figures/builds/; every other command works from the committed caches alone.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_data as bd  # noqa: E402

ROOT = bd.ROOT
FIG = ROOT / "paper" / "figures"
BUILDS = FIG / "builds"
CURVES_JSON = FIG / "curves_data.json"

# one fixed hue per generator stage; data version is carried by line style, seeds by a band
HUE = {"greedy": "#2a78d6", "sft": "#eb6834", "rl": "#1baf7a", "nostab": "#6250d6", "noconn": "#e34948"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
STAGE = {"greedy_v1": "greedy", "sft_v1": "sft", "sft_v2": "sft", "rl_v1": "rl", "rl_v1_nostop": "rl", "rl_v2": "rl", "rl_v2_nostop": "rl",
         "rl_v2_nostab": "nostab", "rl_v2_nostab_nostop": "nostab", "rl_v2_noconn": "noconn", "rl_v2_noconn_nostop": "noconn"}
LABEL = {"greedy_v1": "greedy search", "sft_v1": "SFT v1", "rl_v1": "RL v1", "rl_v1_nostop": "RL v1 (STOP masked)",
         "sft_v2": "SFT v2", "rl_v2": "RL v2", "rl_v2_nostop": "RL v2 (STOP masked)",
         "rl_v2_nostab": "no stability reward", "rl_v2_nostab_nostop": "no stability reward (STOP masked)",
         "rl_v2_noconn": "no connectivity inputs", "rl_v2_noconn_nostop": "no connectivity inputs (STOP masked)"}
# rows chosen with `pick-gallery`: two procedural shapes, an RL-over-SFT gain, the largest RL win, a shared loss, the honeycomb pot
GALLERY = ["proc_pyramid", "proc_sphere", "thingi_42035", "thingi_966906", "thingi_87521", "thingi_39108"]
GALLERY_GENS = ["greedy_v1", "sft_v2", "rl_v2_nostop"]
WIDE, COL = 6.9, 3.3


def style() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.6, "lines.markersize": 4, "legend.frameon": False, "pdf.fonttype": 42, "savefig.dpi": 220,
    })


def save(fig, name: str, png: bool = False) -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / (f"{name}.png" if png else f"{name}.pdf")
    if png:
        fig.savefig(out, dpi=220, metadata={"Software": None})
    else:
        fig.savefig(out, metadata={"CreationDate": None, "Producer": None, "Creator": None})
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")
    return out


# ---- builds (needs torch) ---------------------------------------------------------------------------------------

def _cases():
    from ml.benchmark import procedural_cases, thingi_cases
    return {c.id: c for c in procedural_cases() + thingi_cases()}


def cmd_builds(args) -> None:
    import os
    import ml.rl_generator  # noqa: F401  (registers rl_* generators)
    import ml.sft_generator  # noqa: F401
    from brickcore.catalog import load_catalog
    from brickcore.generators import get as get_generator
    from brickcore.mesh import load_mesh, normalize, occupancy_cells, voxelize
    from brickcore.schema import Bounds, Constraints
    for var in ("BRICKFORGE_SFT_DEVICE", "BRICKFORGE_RL_DEVICE"):
        assert os.environ.get(var, "cpu") == "cpu", f"{var} must be unset or cpu for reproducible builds"
    records = bd.index(bd.all_records())
    catalog = load_catalog()
    cases = _cases()
    BUILDS.mkdir(parents=True, exist_ok=True)
    for cid in args.cases or GALLERY:
        case = cases[cid]
        bounds = Bounds(x_studs=case.size[0], y_studs=case.size[1], z_plates=case.size[2] * 3)
        mesh, _ = load_mesh(ROOT / case.mesh)
        m, _ = normalize(mesh, bounds, catalog, up=case.up)
        target = voxelize(m, bounds, catalog)
        (BUILDS / f"{cid}.target.json").write_text(json.dumps({"grid": list(case.size), "cells": occupancy_cells(target)}))
        for gen in args.generators or GALLERY_GENS:
            budget = args.budget
            res = get_generator(gen)(target, catalog, Constraints(max_pieces=budget, bounds=bounds, color_id="4"), seed=42, build_id=f"{cid}_{gen}_{budget}_42")
            rec = records[(gen, cid, budget)]
            assert abs(res.fidelity - (rec.get("iou") or 0.0)) < 1e-4 or res.build is None and not rec.get("valid"), (cid, gen, res.fidelity, rec.get("iou"))
            out = {"case": cid, "generator": gen, "budget": budget, "status": res.status, "iou": round(res.fidelity, 4),
                   "pieces": len(res.build.placements) if res.build else 0, "build": res.build.model_dump() if res.build else None}
            (BUILDS / f"{cid}_{gen}_{budget}.json").write_text(json.dumps(out))
            print(f"{cid:>16} {gen:<14} iou={res.fidelity:.3f} pieces={out['pieces']} status={res.status}")


def cmd_pick_gallery(args) -> None:
    recs = bd.all_records()
    by, grp = bd.index(recs), bd.case_group(recs)
    rows = []
    for c in bd.attempted_cases(recs):
        g, s, r = (bd.score(by.get((x, c, 200))) for x in GALLERY_GENS)
        rows.append((r - g, c, grp[c], g, s, r, (by.get(("rl_v2_nostop", c, 200)) or {}).get("dropped_disconnected")))
    for d, c, g, a, b, r, dropped in sorted(rows):
        print(f"{d:+.3f} {c:>18} {g:<10} greedy={a:.3f} sft={b:.3f} rl={r:.3f} dropped={dropped}")


# ---- figures ----------------------------------------------------------------------------------------------------

def cmd_budget(args) -> None:
    recs = bd.all_records()
    panels = [("all 52 attempted cases", None), ("12 procedural shapes", "procedural"), ("40 Thingi10K meshes", "thingi10k")]
    series = [("greedy_v1", "-"), ("sft_v1", "--"), ("rl_v1_nostop", "--"), ("sft_v2", "-"), ("rl_v2_nostop", "-")]
    fig, axes = plt.subplots(1, 3, figsize=(WIDE, 2.3), sharey=True)
    for ax, (title, group) in zip(axes, panels):
        for gen, ls in series:
            fam = bd.family_of(gen)
            stats = [bd.family_stats(recs, fam, b, group) for b in bd.BUDGETS]
            ys = [s[0] for s in stats]
            ax.plot(bd.BUDGETS, ys, ls, color=HUE[STAGE[gen]], marker="o", markersize=3.2, label=LABEL[gen])
            if fam in bd.SEED_FAMILIES:
                ax.fill_between(bd.BUDGETS, [s[1] for s in stats], [s[2] for s in stats], color=HUE[STAGE[gen]], alpha=0.18, linewidth=0)
        ax.set_xticks(bd.BUDGETS)
        ax.set_xlim(40, 212)
        ax.set_title(title, loc="left")
        ax.set_xlabel("piece budget")
    axes[0].set_ylabel("mean IoU after trim")
    axes[0].set_ylim(0.44, 0.86)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.02), handlelength=2.4)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.34, wspace=0.12)
    save(fig, "budget")


def cmd_scatter(args) -> None:
    recs = bd.all_records()
    by, grp = bd.index(recs), bd.case_group(recs)
    fig, ax = plt.subplots(figsize=(COL, COL))
    ax.plot([0, 1], [0, 1], color="#c3c2b7", linewidth=0.8, zorder=1)
    pts = {"thingi10k": ([], [], []), "procedural": ([], [], [])}
    for c in bd.attempted_cases(recs):
        x, y = bd.score(by.get(("greedy_v1", c, 200))), bd.score(by.get(("rl_v2_nostop", c, 200)))
        pts[grp[c]][0].append(x)
        pts[grp[c]][1].append(y)
        pts[grp[c]][2].append(c)
    ax.scatter(*pts["thingi10k"][:2], s=16, color=HUE["greedy"], label="Thingi10K mesh", zorder=3, edgecolor="white", linewidth=0.5)
    ax.scatter(*pts["procedural"][:2], s=18, color=HUE["sft"], marker="s", label="procedural shape", zorder=3, edgecolor="white", linewidth=0.5)
    for g in pts.values():
        for x, y, c in zip(*g):
            if abs(x - y) > 0.12:
                right = x > 0.85
                ax.annotate(c.replace("thingi_", "T").replace("proc_", ""), (x, y), xytext=(-4 if right else 4, -3), textcoords="offset points",
                            fontsize=6, color=MUTED, ha="right" if right else "left")
    ax.set_xlabel("greedy search IoU (200-piece cap)")
    ax.set_ylabel("RL v2, STOP masked, IoU (200-piece cap)")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    fig.subplots_adjust(left=0.16, right=0.98, top=0.98, bottom=0.14)
    save(fig, "scatter")


def cmd_ladder(args) -> None:
    recs = bd.all_records()
    arms = ["sft_v1", "rl_v1", "rl_v1_nostop", "sft_v2", "rl_v2", "rl_v2_nostop"]
    fig, ax = plt.subplots(figsize=(COL, 2.2))
    xs = np.arange(len(arms))
    for i, gen in enumerate(arms):
        fam = bd.family_of(gen)
        gens = bd.SEED_FAMILIES.get(fam, [gen])
        vals = [bd.attempted_mean(recs, g, 200) for g in gens]
        mean = sum(vals) / len(vals)
        hatch = "////" if gen in ("rl_v1", "rl_v2") else None
        ax.bar(i, mean, width=0.7, color=HUE[STAGE[gen]], hatch=hatch, edgecolor="white", linewidth=0.8, alpha=0.95 if hatch is None else 0.6)
        if len(vals) > 1:
            ax.scatter([i] * len(vals), vals, s=10, color=INK, zorder=4)
        ax.text(i, max(vals) + 0.006, f"{mean:.3f}", ha="center", va="bottom", fontsize=6.5, color=INK)
    g = bd.attempted_mean(recs, "greedy_v1", 200)
    ax.axhline(g, color=HUE["greedy"], linewidth=1.2, linestyle="-")
    ax.text(len(arms) - 0.55, g + 0.004, f"greedy search {g:.3f}", ha="right", va="bottom", fontsize=6.5, color=HUE["greedy"])
    ax.set_xticks(xs)
    ax.set_xticklabels(["SFT", "RL", "RL\nSTOP masked", "SFT", "RL", "RL\nSTOP masked"], fontsize=6.5)
    ax.text(1, 0.735, "v1 data (7 % real meshes)", ha="center", va="top", fontsize=6.5, color=MUTED)
    ax.text(4, 0.735, "v2 data (24 % real meshes)", ha="center", va="top", fontsize=6.5, color=MUTED)
    ax.set_ylim(0.5, 0.74)
    ax.set_ylabel("mean IoU, 200-piece cap")
    ax.grid(axis="x", visible=False)
    fig.subplots_adjust(left=0.17, right=0.98, top=0.97, bottom=0.2)
    save(fig, "ladder")


def _curves_data() -> dict:
    runs_present = (bd.RUNS / "rl" / "rl_v2" / "metrics.jsonl").exists()
    if runs_present:
        data = {"sft": {r: bd.sft_curve(r) for r in ("sft_v1", "sft_v2", "sft_v2_s1")},
                "rl": {r: bd.rl_curve(r) for r in ("rl_v1", "rl_v2", "rl_v2_s1", "rl_v2_nostab", "rl_v2_noconn")}}
        CURVES_JSON.write_text(json.dumps(data))
        return data
    return json.loads(CURVES_JSON.read_text())


def cmd_curves(args) -> None:
    data = _curves_data()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDE, 2.3))
    sft_style = {"sft_v1": ("--", "SFT v1"), "sft_v2": ("-", "SFT v2 (seed 0)"), "sft_v2_s1": ("-", "SFT v2 (seed 1)")}
    for run, (ls, lab) in sft_style.items():
        ev = data["sft"][run]["evals"]
        a1.plot([e["step"] / 1000 for e in ev], [e["iou"] for e in ev], ls, color=HUE["sft"], alpha=1.0 if run != "sft_v2_s1" else 0.55, label=lab)
    t = data["sft"]["sft_v2"]["evals"][-1]["teacher_iou"]
    a1.axhline(t, color=HUE["greedy"], linewidth=0.9, linestyle=":")
    a1.text(0.3, t + 0.006, f"greedy teacher on this pool {t:.3f}", fontsize=6.5, color=HUE["greedy"], va="bottom")
    a1.set_xlabel("SFT step (thousands)")
    a1.set_ylabel("validation IoU after trim (free-running decode)")
    a1.set_title("imitation", loc="left")
    a1.legend(loc="lower right")
    rl_style = {"rl_v1": ("--", "rl", "RL v1"), "rl_v2": ("-", "rl", "RL v2 (seed 0)"), "rl_v2_s1": ("-", "rl", "RL v2 (seed 1)"),
                "rl_v2_nostab": ("-", "nostab", "no stability reward"), "rl_v2_noconn": ("-", "noconn", "no connectivity inputs")}
    for run, (ls, hue, lab) in rl_style.items():
        ev = data["rl"][run]["evals"]
        a2.plot([e["iter"] for e in ev], [e["iou"] for e in ev], ls, color=HUE[hue], alpha=1.0 if run != "rl_v2_s1" else 0.55, label=lab)
    t2 = data["rl"]["rl_v2"]["evals"][-1]["teacher_iou"]
    a2.axhline(t2, color=HUE["greedy"], linewidth=0.9, linestyle=":")
    a2.text(3, t2 + 0.006, f"greedy teacher on this pool {t2:.3f}", fontsize=6.5, color=HUE["greedy"], va="bottom")
    a2.set_xlabel("PPO iteration (128 episodes each)")
    a2.set_title("reinforcement learning", loc="left")
    a2.legend(loc="lower left", ncol=2, columnspacing=0.8)
    a1.set_ylim(0.6, 1.0)
    a2.set_ylim(0.58, 0.96)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.9, bottom=0.2, wspace=0.15)
    save(fig, "curves")


def _load_build(path: Path):
    from brickcore.schema import Build
    d = json.loads(path.read_text())
    return d, (Build.model_validate(d["build"]) if d["build"] else None)


def cmd_gallery(args) -> None:
    from brickcore.catalog import load_catalog
    from brickcore.mesh import occupancy_from_cells
    from brickcore.schema import Bounds
    import brickplot as bp
    catalog = load_catalog()
    cases = args.cases or GALLERY
    gens = GALLERY_GENS
    titles = ["target (voxels)", "greedy search", "SFT v2", "RL v2, STOP masked"]
    fig = plt.figure(figsize=(WIDE, 1.25 * len(cases)))
    for r, cid in enumerate(cases):
        t = json.loads((BUILDS / f"{cid}.target.json").read_text())
        grid = tuple(t["grid"])
        target = occupancy_from_cells(t["cells"], Bounds(x_studs=grid[0], y_studs=grid[1], z_plates=grid[2] * 3))
        ax = fig.add_subplot(len(cases), 4, r * 4 + 1, projection="3d")
        bp.draw_target(ax, target)
        name = cid.replace("thingi_", "Thingi10K ").replace("proc_", "")
        ax.text2D(0.02, 0.98, f"{name}\n{int(target.sum())} cells", transform=ax.transAxes, fontsize=6.5, va="top", color=INK)
        if r == 0:
            ax.set_title(titles[0], fontsize=7.5)
        for j, gen in enumerate(gens):
            d, build = _load_build(BUILDS / f"{cid}_{gen}_200.json")
            ax = fig.add_subplot(len(cases), 4, r * 4 + 2 + j, projection="3d")
            if build is not None:
                bp.draw_build(ax, build.placements, catalog, grid)
            else:
                bp.draw_target(ax, np.zeros(grid, dtype=bool))
            ax.text2D(0.02, 0.98, f"IoU {d['iou']:.2f}, {d['pieces']} pieces" if build else "no valid build", transform=ax.transAxes, fontsize=6.5, va="top", color=INK)
            if r == 0:
                ax.set_title(titles[j + 1], fontsize=7.5)
    fig.subplots_adjust(left=0.0, right=1.0, top=0.96, bottom=0.0, wspace=0.0, hspace=0.02)
    save(fig, "gallery", png=True)


def cmd_teaser(args) -> None:
    """One Thingi10K mesh: target, greedy, RL (STOP masked) in a row; used as Fig. 1."""
    from brickcore.catalog import load_catalog
    from brickcore.mesh import occupancy_from_cells
    from brickcore.schema import Bounds
    import brickplot as bp
    catalog = load_catalog()
    cid = args.cases[0] if args.cases else "thingi_75090"
    t = json.loads((BUILDS / f"{cid}.target.json").read_text())
    grid = tuple(t["grid"])
    target = occupancy_from_cells(t["cells"], Bounds(x_studs=grid[0], y_studs=grid[1], z_plates=grid[2] * 3))
    fig = plt.figure(figsize=(WIDE, 2.0))
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    bp.draw_target(ax, target)
    ax.set_title(f"voxel target\n{int(target.sum())} cells, {grid[0]}x{grid[1]}x{grid[2]} grid", fontsize=7.5)
    for j, (gen, title) in enumerate((("greedy_v1", "greedy search"), ("rl_v2_nostop", "learned policy (RL v2, STOP masked)"))):
        d, build = _load_build(BUILDS / f"{cid}_{gen}_200.json")
        ax = fig.add_subplot(1, 3, 2 + j, projection="3d")
        bp.draw_build(ax, build.placements, catalog, grid)
        ax.set_title(f"{title}\nIoU {d['iou']:.2f}, {d['pieces']} pieces", fontsize=7.5)
    fig.subplots_adjust(left=0.0, right=1.0, top=0.84, bottom=0.0, wspace=0.0)
    save(fig, "teaser", png=True)


def cmd_stops(args) -> None:
    recs = bd.all_records()
    gens = ["rl_v2", "rl_v2_nostop", "rl_v2_s1", "rl_v2_s1_nostop", "rl_v2_nostab", "rl_v2_nostab_nostop", "rl_v2_noconn", "rl_v2_noconn_nostop"]
    reasons = [("stop", "policy chose STOP", "#6250d6"), ("no_legal_candidate", "no legal placement", "#2a78d6"), ("cap", "cap reached", "#1baf7a")]
    fig, ax = plt.subplots(figsize=(COL, 2.1))
    bottoms = np.zeros(len(gens))
    for key, lab, col in reasons:
        vals = np.array([bd.counts(recs, g, 200)["stops"].get(key, 0) for g in gens], dtype=float)
        ax.bar(np.arange(len(gens)), vals, bottom=bottoms, color=col, width=0.7, edgecolor="white", linewidth=0.8, label=lab)
        bottoms += vals
    ax.set_xticks(np.arange(len(gens)))
    ax.set_xticklabels(["learned", "masked"] * 4, fontsize=6.5)
    for i, name in enumerate(["RL v2 s0", "RL v2 s1", "no stab.", "no conn."]):
        ax.text(2 * i + 0.5, 56, name, ha="center", fontsize=6.5, color=MUTED)
    ax.set_ylim(0, 60)
    ax.set_ylabel("cases (200-piece cap)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=3, fontsize=6.5, columnspacing=0.9, handlelength=1.2)
    ax.grid(axis="x", visible=False)
    fig.subplots_adjust(left=0.15, right=0.98, top=0.95, bottom=0.3)
    save(fig, "stops")


# ---- numbers.tex ------------------------------------------------------------------------------------------------

_WORDS = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}
_BUDGET = {50: "Fifty", 100: "Hundred", 200: "TwoHundred"}


def macro(*parts: str) -> str:
    out = ""
    for part in parts:
        for tok in str(part).split("_"):
            tok = "".join(_WORDS[ch] if ch.isdigit() else ch for ch in tok)
            out += tok[:1].upper() + tok[1:]
    return out


def cmd_numbers(args) -> None:
    recs = bd.all_records()
    lines = ["% generated by paper/scripts/figs.py numbers; do not edit"]
    gens = sorted({r["generator"] for r in recs})
    for g in gens:
        for b in bd.BUDGETS:
            lines.append(f"\\newcommand{{\\mean{macro(g, _BUDGET[b])}}}{{{bd.attempted_mean(recs, g, b):.3f}}}")
        for grp, tag in (("procedural", "Proc"), ("thingi10k", "Real")):
            lines.append(f"\\newcommand{{\\mean{macro(g)}{tag}}}{{{bd.attempted_mean(recs, g, 200, grp):.3f}}}")
        c = bd.counts(recs, g, 200)
        lines.append(f"\\newcommand{{\\valid{macro(g)}}}{{{c['valid']}}}")
        lines.append(f"\\newcommand{{\\stable{macro(g)}}}{{{c['stable']}}}")
        lines.append(f"\\newcommand{{\\secs{macro(g)}}}{{{c['seconds']:.2f}}}")
    for fam in bd.SEED_FAMILIES:
        for b in bd.BUDGETS:
            m, lo, hi = bd.family_stats(recs, fam, b)
            lines.append(f"\\newcommand{{\\fam{macro(fam, _BUDGET[b])}}}{{{m:.3f}}}")
            lines.append(f"\\newcommand{{\\famlo{macro(fam, _BUDGET[b])}}}{{{lo:.3f}}}")
            lines.append(f"\\newcommand{{\\famhi{macro(fam, _BUDGET[b])}}}{{{hi:.3f}}}")
    pairs = [("rl_v2_nostop", "greedy_v1"), ("rl_v2_nostop", "sft_v2"), ("rl_v2_s1_nostop", "greedy_v1"), ("rl_v2_s1_nostop", "sft_v2_s1"),
             ("rl_v1_nostop", "greedy_v1"), ("sft_v2", "greedy_v1"), ("sft_v1", "greedy_v1"), ("rl_v2_nostab_nostop", "rl_v2_nostop"),
             ("rl_v2_noconn_nostop", "rl_v2_nostop"), ("rl_v2_nostab_nostop", "greedy_v1"), ("rl_v2_noconn_nostop", "greedy_v1"), ("rl_v2", "sft_v2")]
    for a, b in pairs:
        w, lose, t = bd.head_to_head(recs, a, b)
        lines.append(f"\\newcommand{{\\hh{macro(a)}Vs{macro(b)}}}{{{w} / {lose} / {t}}}")
    n_att = len(bd.attempted_cases(recs))
    lines.append(f"\\newcommand{{\\nAttempted}}{{{n_att}}}")
    lines.append(f"\\newcommand{{\\nCases}}{{{len({r['case'] for r in recs})}}}")
    (FIG / "numbers.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {(FIG / 'numbers.tex').relative_to(ROOT)} ({len(lines)} macros)")


def cmd_tables(args) -> None:
    """Per-case table at the 200-piece cap for the appendix."""
    recs = bd.all_records()
    by, grp = bd.index(recs), bd.case_group(recs)
    gens = ["greedy_v1", "sft_v2", "rl_v2_nostop", "rl_v2_s1_nostop", "rl_v2_nostab_nostop", "rl_v2_noconn_nostop"]
    lines = ["% generated by paper/scripts/figs.py tables; do not edit",
             "\\begin{center}\\footnotesize\\setlength{\\tabcolsep}{4pt}",
             "\\begin{tabular}{llr" + "c" * len(gens) + "}", "\\toprule",
             "case & source & cells & search & SFT v2 & RL v2 s0 & RL v2 s1 & no stab. & no conn. \\\\", "\\midrule"]
    cases = sorted(bd.attempted_cases(recs), key=lambda c: (grp[c] != "procedural", c))
    for c in cases:
        r0 = by[("greedy_v1", c, 200)]
        name = c.replace("thingi_", "T").replace("proc_", "").replace("_", "\\_")
        vals = " & ".join(f"{bd.score(by.get((g, c, 200))):.3f}" for g in gens)
        lines.append(f"{name} & {grp[c]} & {r0.get('target_cells', 0)} & {vals} \\\\")
    lines += ["\\midrule", "mean & & & " + " & ".join(f"{bd.attempted_mean(recs, g, 200):.3f}" for g in gens) + " \\\\",
              "\\bottomrule", "\\end{tabular}", "\\end{center}"]
    lines.insert(1, "Per-case IoU after trim at the 200-piece cap (0 for a missing or invalid build). RL columns use the masked stop; "
                    "cells is the number of occupied target cells out of 2048 (512 for the cube).")
    (FIG / "table_percase.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {(FIG / 'table_percase.tex').relative_to(ROOT)} ({len(cases)} rows)")


# ---- checks -----------------------------------------------------------------------------------------------------

def cmd_check(args) -> None:
    from brickcore.catalog import load_catalog
    from brickcore.geometry import iou, occupancy_grid
    from brickcore.mesh import occupancy_from_cells
    from brickcore.schema import Bounds
    from brickcore.validate import validate_build
    recs = bd.all_records()
    by = bd.index(recs)
    catalog = load_catalog()
    n = 0
    for path in sorted(BUILDS.glob("*_200.json")):
        d, build = _load_build(path)
        t = json.loads((BUILDS / f"{d['case']}.target.json").read_text())
        grid = tuple(t["grid"])
        target = occupancy_from_cells(t["cells"], Bounds(x_studs=grid[0], y_studs=grid[1], z_plates=grid[2] * 3))
        rec = by[(d["generator"], d["case"], d["budget"])]
        if build is None:
            assert not rec.get("valid"), path
        else:
            assert validate_build(build, catalog)["ok"], path
            got = iou(occupancy_grid(build.placements, catalog, *grid), target)
            assert abs(got - rec["iou"]) < 1e-4, (path, got, rec["iou"])
        n += 1
    spot = {("greedy_v1", 200): 0.705, ("rl_v2_nostop", 200): 0.693, ("rl_v2_s1_nostop", 200): 0.693, ("sft_v2", 200): 0.672,
            ("sft_v2_s1", 200): 0.652, ("rl_v2_noconn", 200): 0.595, ("rl_v2_nostab_nostop", 200): 0.686, ("sft_v1", 200): 0.614, ("rl_v1", 200): 0.654}
    for (g, b), want in spot.items():
        got = bd.attempted_mean(recs, g, b)
        assert abs(got - want) < 0.0006, (g, b, got, want)
    digest = hashlib.sha256()
    for p in sorted(FIG.glob("*.p*")):
        digest.update(p.read_bytes())
    print(f"check ok: {n} cached builds match their benchmark records; {len(spot)} spot means match docs/rl.md; figures sha256 {digest.hexdigest()[:12]}")


def cmd_all(args) -> None:
    for fn in (cmd_budget, cmd_scatter, cmd_ladder, cmd_curves, cmd_stops, cmd_numbers, cmd_tables, cmd_teaser, cmd_gallery):
        fn(args)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["builds", "pick-gallery", "budget", "scatter", "ladder", "curves", "gallery", "teaser", "stops", "numbers", "tables", "check", "all"])
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--generators", nargs="*", default=None)
    ap.add_argument("--budget", type=int, default=200)
    args = ap.parse_args(argv)
    style()
    globals()[f"cmd_{args.command.replace('-', '_')}"](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
