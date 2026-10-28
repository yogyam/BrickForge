# Stability analysis (M2.5)

`brickcore/stability.py` checks static equilibrium of the final build and of every assembly prefix.

## Model
- Rigid bricks under gravity only. Weight scales with footprint area, 2.32 g per 2x4 brick.
- Each shared stud cell between an upper and a lower brick carries vertical force at four sub-points near the
  cell corners. Compression is unbounded; tension is capped by the clutch capacity T split over the four
  points. Ground cells carry compression only (no clutch against the table).
- Per brick: vertical force balance and torque balance about x and y.
- A linear program finds the smallest tension utilisation `t` such that every joint force is within
  `[-t*T/4, +inf)`. `t <= 1` means stable under the model. If the LP is infeasible even with unlimited tension
  the structure topples: the ground cannot pull, so the centre of mass has left the support.

Outcome modes: `ok`, `clutch_exceeded` (equilibrium exists but needs more pull-apart force than the studs
provide), `topple` (no equilibrium at all).

## Constants
- `T = 0.98 N per stud`, the measured vertical capacity used by StableLego (Liu et al., arXiv 2402.10711) and
  reused by ResemBrick (arXiv 2608.09597). It is a literature value, not measured on this project's bricks.
- All constants live in `StabilityParams` and are written into every report.

## What it is not
No lateral loads, no shear or friction, no deformation, no hand forces, no stud wear, no dynamics. Reports say
"stable under the simplified static force model; not physically tested" and nothing stronger. Structural
validity (collisions, connections, budget) is a separate, authoritative check; stability is a diagnostic
and, from M6 on, a reported metric and RL reward term.

## Cost
About 20 ms per LP on the M5 Pro for a 75-piece build; per-step analysis of 75 steps takes about 1.5 s.
