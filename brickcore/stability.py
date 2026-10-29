"""Static stability analysis (M2.5): rigid-body force equilibrium at stud joints under gravity.

Model (after StableLego, arXiv 2402.10711, and the rigid block equilibrium formulation it adopts):
- Every brick is a rigid body with weight proportional to its footprint area.
- Where an upper brick rests on a lower brick, each shared stud cell transmits vertical force at four
  sub-points near the cell corners. Compression is unbounded; tension (pull-apart) is limited by the stud
  clutch capacity T, split evenly over the four sub-points. Four points per stud give a single stud a small
  moment capacity, as a real stud-in-tube fit has.
- Ground contact cells transmit compression only (a brick on a table has no clutch).
- Per brick: vertical force balance and torque balance about x and y. Only gravity loads are modelled.

We solve a linear program for the smallest tension utilisation t such that every joint force stays within
[-t*T/4, +inf). The assembly is "stable under the model" iff t <= 1. The same LP is run for every prefix of
the assembly steps so intermediate states are checked too.

Not modelled: lateral loads, friction/shear, plastic deformation, stud wear, hand forces, dynamic effects.
This is a diagnostic under a simplified model with a literature constant, not a physical test.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import linprog

from .catalog import Catalog
from .geometry import body_of, stud_connected
from .schema import Placement, Step

MODEL_NAME = "stud_joint_lp_v1"


@dataclass(frozen=True)
class StabilityParams:
    clutch_newton_per_stud: float = 0.98      # StableLego (arXiv 2402.10711) measured vertical capacity T
    brick_2x4_mass_g: float = 2.32            # common reference mass; weight scales with footprint area
    gravity_m_s2: float = 9.81
    sub_point_offset_mm: float = 2.0          # four sub-points per stud at (+-2, +-2) mm from the cell centre
    source: str = "T=0.98 N/stud from StableLego (Liu et al. 2024); mass 2.32 g per 2x4 brick"

    @property
    def weight_per_cell_mN(self) -> float:
        return self.brick_2x4_mass_g / 8 * self.gravity_m_s2  # grams * m/s^2 = mN


DEFAULT_PARAMS = StabilityParams()


def _points(cells: list[tuple[int, int]], pitch: float, off: float) -> list[tuple[float, float]]:
    pts = []
    for x, y in cells:
        cx, cy = (x + 0.5) * pitch, (y + 0.5) * pitch
        pts += [(cx + sx * off, cy + sy * off) for sx in (-1, 1) for sy in (-1, 1)]
    return pts


def analyze(placements: list[Placement], catalog: Catalog, params: StabilityParams = DEFAULT_PARAMS) -> dict:
    """Minimum tension utilisation for one assembly state. Returns stable flag, utilisation, worst joint."""
    if not placements:
        return {"stable": True, "utilization": 0.0, "worst_joint": None, "pieces": 0}
    pitch = catalog.stud_pitch_mm
    bodies = {p.id: body_of(p, catalog) for p in placements}
    ids = list(bodies)
    idx = {pid: i for i, pid in enumerate(ids)}
    n = len(ids)
    cap_point = params.clutch_newton_per_stud * 1000 / 4  # mN per sub-point

    # Columns: force at each sub-point. Rows: 3 equations per brick (Fz, Mx, My).
    cols: list[tuple[int, int | None, float, float, bool]] = []  # (upper, lower or None=ground, x, y, is_joint)
    joint_label: list[tuple[str, str] | None] = []
    for u in ids:
        bu = bodies[u]
        if bu.z0 == 0:
            for x, y in _points(bu.cells, pitch, params.sub_point_offset_mm):
                cols.append((idx[u], None, x, y, False))
                joint_label.append(None)
        for low in ids:
            if u != low and stud_connected(bu, bodies[low]):
                bl = bodies[low]
                shared = [(x, y) for x in range(max(bu.x0, bl.x0), min(bu.x1, bl.x1)) for y in range(max(bu.y0, bl.y0), min(bu.y1, bl.y1))]
                for x, y in _points(shared, pitch, params.sub_point_offset_mm):
                    cols.append((idx[u], idx[low], x, y, True))
                    joint_label.append((u, low))
    m = len(cols)
    A_eq = np.zeros((3 * n, m + 1))
    b_eq = np.zeros(3 * n)
    centres = {i: (((bodies[pid].x0 + bodies[pid].x1) / 2) * pitch, ((bodies[pid].y0 + bodies[pid].y1) / 2) * pitch) for pid, i in idx.items()}
    for i, pid in enumerate(ids):
        b = bodies[pid]
        b_eq[3 * i] = (b.x1 - b.x0) * (b.y1 - b.y0) * params.weight_per_cell_mN
    for j, (u, low, x, y, _) in enumerate(cols):
        cx, cy = centres[u]
        A_eq[3 * u, j] += 1.0
        A_eq[3 * u + 1, j] += x - cx
        A_eq[3 * u + 2, j] += y - cy
        if low is not None:
            cx, cy = centres[low]
            A_eq[3 * low, j] -= 1.0
            A_eq[3 * low + 1, j] -= x - cx
            A_eq[3 * low + 2, j] -= y - cy
    # Inequalities: -f_j - t*cap <= 0 for joint points. Bounds: ground points >= 0, joint points free, t >= 0.
    joint_cols = [j for j, c in enumerate(cols) if c[4]]
    A_ub = np.zeros((len(joint_cols), m + 1))
    b_ub = np.zeros(len(joint_cols))
    for r, j in enumerate(joint_cols):
        A_ub[r, j] = -1.0
        A_ub[r, m] = -cap_point
    bounds = [(None, None) if c[4] else (0, None) for c in cols] + [(0, None)]
    c = np.zeros(m + 1)
    c[m] = 1.0
    res = linprog(c, A_ub=A_ub if joint_cols else None, b_ub=b_ub if joint_cols else None, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        # No equilibrium exists even with unlimited clutch: the structure topples (ground gives compression only).
        return {"stable": False, "mode": "topple", "utilization": None, "worst_joint": None, "pieces": n, "lp_status": res.message}
    t = float(res.x[m])
    worst = None
    if joint_cols:
        tensions = [(float(-res.x[j] / cap_point), j) for j in joint_cols]
        val, j = max(tensions)
        if val > 1e-9:
            worst = {"upper": joint_label[j][0], "lower": joint_label[j][1], "tension_utilization": round(val, 4)}
    stable = t <= 1.0 + 1e-9
    return {"stable": stable, "mode": "ok" if stable else "clutch_exceeded", "utilization": round(t, 4), "worst_joint": worst, "pieces": n}


def analyze_build(placements: list[Placement], steps: list[Step], catalog: Catalog, params: StabilityParams = DEFAULT_PARAMS) -> dict:
    """Final-state analysis plus one analysis per assembly prefix when steps are present."""
    final = analyze(placements, catalog, params)
    by_id = {p.id: p for p in placements}
    per_step, first_unstable, prefix = [], None, []
    for s in steps:
        prefix += [by_id[i] for i in s.added_placement_ids]
        r = analyze(prefix, catalog, params)
        per_step.append({"step": s.index, "stable": r["stable"], "mode": r["mode"], "utilization": r["utilization"]})
        if not r["stable"] and first_unstable is None:
            first_unstable = s.index
    out = {
        "model": MODEL_NAME, "params": asdict(params), "final": final,
        "per_step": per_step, "first_unstable_step": first_unstable,
        "all_steps_stable": bool(steps) and first_unstable is None,
        "max_step_utilization": max((r["utilization"] for r in per_step if r["utilization"] is not None), default=final["utilization"]),
    }
    out["label"] = ("stable under the simplified static force model" if final["stable"] else f"UNSTABLE ({final['mode']}) under the simplified static force model") + "; not physically tested"
    return out
