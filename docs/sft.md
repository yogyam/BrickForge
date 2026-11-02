# Supervised generator (M5)

## Data (`ml/data.py`)
- Targets are occupancy grids on the 16x16x8 canonical grid.
- Ten procedural families (box, stack, capped tower, bridge, polyomino, ellipsoid, cylinder, stepped, blob,
  arch row) generated directly on the grid, grounded and face-connected. 200 instances per family.
- A Thingi10K training pool: eligible CC models with every held-out Thingiverse thing excluded
  (`ml.thingi10k.held_out_thing_ids`), voxelized with the production pipeline, at most 50k faces.
- Supervision: the baseline's validated build on each target, ordered by the planner, is the teacher action
  sequence; STOP is the final action. Examples with coverage under 0.5 or no planned order are dropped.
- Split by structure instance via a hash of the instance id (80/10/10); all symmetries of an instance, and all
  files of one Thingiverse thing, share a split.
- Known domain gap: the model imitates the greedy baseline's style on these shapes. It can only beat the
  baseline through better choices under the same rules, not through shapes it never saw.

## Policy (`ml/policy.py`)
- Input: three voxel channels (target, current assembly, remaining target). Four 3D conv layers, 32 channels,
  full resolution (no pooling; the grid is small).
- Actions: the bounded legal candidate set from `brickcore/candidates.py` (inside bounds, collision-free,
  supported, at least one new target cell, at most half filler) plus STOP. Each candidate pools the feature
  map over its footprint cells and concatenates part one-hot, rotation, position and overlap counts into a
  small MLP; STOP is scored from a global pool. Softmax over candidates + STOP. Legal-action masking is
  therefore inherent, at training and inference alike.
- 97k parameters.

## Training (`ml/train_sft.py`)
- Teacher forcing with cross-entropy; AdamW, cosine schedule, grad clipping. Optional 8-fold symmetry
  augmentation. Free-running evaluation (decode, finalize, validate, plan) is logged alongside teacher-forced
  loss, because teacher-forced accuracy alone hides exposure bias.
- Profiling on the M5 Pro, batch 32: Metal 22.7 steps/s, CPU 5.8 steps/s. Metal is used.

## Gate 1 (passed 2026-10-29)
`--overfit 32`: after 1500 steps (86 s on Metal) all 32 training examples free-run to a valid, planned build at
or above the teacher's IoU (mean 0.985). Checkpoint `checkpoints/sft_overfit32.pt` (not committed).

## Inference (`ml/sft_generator.py`)
Registered as generator `sft_v1`. Greedy decode by default; seeded sampling with a temperature is available
for best-of-N ablations. Output goes through the same `finalize_candidate` as the baseline (component trim,
planner, validator, metrics), so no learned output bypasses validation. Select a checkpoint with
`BRICKFORGE_SFT_CKPT=checkpoints/<name>.pt`.

## Full training (sft_v1, 2026-10-29)
- Data: 2000 synthetic + 150 Thingi10K training-pool examples (`data/sft/examples.jsonl`); train 1750 / val 194 / test 206.
  Teacher mean IoU 0.941 on the test split.
- 20,000 steps, batch 32, AdamW 1e-3 cosine, 8-fold symmetry augmentation, 23 minutes on Metal. Items are
  streamed by a background producer (`ItemPool`) because precomputing every augmented decision would need ~40 GB.
  No teacher action ever fell outside the candidate set.
- Checkpoint selection used free-run IoU on a 32-example val subset every 1000 steps; best at step 8000
  (`checkpoints/sft_v1.pt`, not committed). Caveat: that subset was the first 32 val examples, which are the easiest
  families; the trainer now samples the subset with a fixed seed. Val IoU moved between 0.88 and 0.97 across the run,
  so a lower learning rate or longer tail is the first thing to change in a retrain.
- Inference runs on CPU by default (`BRICKFORGE_SFT_DEVICE`): same speed as Metal for one target, and builds are
  bit-identical across runs, devices and processes (checked on 14 cases).

### Held-out SFT test split (206 examples, not the frozen benchmark)
| decode | mean IoU | teacher IoU | valid | fragmented before trim |
|---|---|---|---|---|
| learned STOP (`sft_v1`) | 0.858 | 0.941 | 206/206 | 59/206 |
| STOP masked (`sft_v1_nostop`) | 0.861 | 0.941 | 206/206 | |

**Before the single-component trim the policy's occupancy IoU is 0.949, above the teacher's 0.941.** The trim
(keep the largest ground-connected component, shared with the baseline) removes 2.6 pieces per build on average and
costs 0.09 IoU. The policy learned *what to cover* but not *how to interlock*: it often repeats one layer's tiling on
the next layer, so rows that only touch side-on never join the main body (arch_row 1.00 -> 0.72, bridge 0.97 -> 0.79,
box 0.99 -> 0.87 after trim). Connectivity is not part of candidate legality (only "rests on something"), so
imitation has no signal for it beyond the teacher's incidental staggering.

The STOP-masking variant was chosen on this split, not on the benchmark, and only then run on the benchmark.

### Frozen benchmark (`docs/benchmarks/sft_v1.{jsonl,md,summary.json}`, 53 cases, seed 42)
| generator | budget | valid | stable | mean IoU | median IoU | mean pieces | gen s |
|---|---|---|---|---|---|---|---|
| greedy_v1 | 50 | 0.962 | 0.925 | 0.633 | 0.698 | 28.4 | 0.04 |
| greedy_v1 | 100 | 0.962 | 0.925 | 0.709 | 0.893 | 37.3 | 0.06 |
| greedy_v1 | 200 | 0.962 | 0.925 | 0.719 | 0.913 | 39.1 | 0.07 |
| sft_v1 | 50 | 0.925 | 0.868 | 0.555 | 0.617 | 24.4 | 0.11 |
| sft_v1 | 100 | 0.925 | 0.868 | 0.639 | 0.785 | 33.1 | 0.13 |
| sft_v1 | 200 | 0.925 | 0.868 | 0.651 | 0.809 | 35.3 | 0.14 |
| sft_v1_nostop | 50 | 0.962 | 0.925 | 0.558 | 0.620 | 24.8 | 0.11 |
| sft_v1_nostop | 100 | 0.962 | 0.925 | 0.641 | 0.785 | 33.5 | 0.15 |
| sft_v1_nostop | 200 | 0.962 | 0.925 | 0.652 | 0.809 | 35.7 | 0.16 |

By source at 200 pieces: procedural greedy 0.828 / sft 0.812 / sft_nostop 0.813; Thingi10K greedy 0.668 /
sft 0.554 / sft_nostop 0.588. Head-to-head at 200: sft_v1 beats greedy on 1 case, loses on 29, ties on 21;
sft_v1_nostop wins 3, loses 24, ties 24. The greedy rows match `baseline_v1` exactly (159/159), and `./bench audit` regenerates all 468 records with no mismatch.

**Result: supervised imitation does not beat its teacher on the held-out benchmark.** Three causes, each visible in the
per-record fields (`stop_reason`, `components_before_trim`, `dropped_disconnected`):
1. Fragmentation (24 of 40 Thingi cases): the dominant loss, as on the test split. Example `thingi_1417953`, a rounded
   column: the policy covers all 256 target cells with 42 pieces, but as 4 components; the trim keeps 18 and IoU is 0.50
   where greedy gets 1.00.
2. Dead ends (`no_legal_candidate` with target cells left, 24 of 52 cases at 200): the policy fills around a cell whose only
   legal cover was a large piece it already blocked. The baseline's hard-cells-first ordering avoids this by construction;
   imitation reproduces it only approximately, and one wrong step is unrecoverable (exposure bias).
3. Early learned STOP on out-of-distribution shapes (28 of 52 stop by STOP; 2 cases stop at zero pieces). Masking STOP
   restores the baseline's validity rate (0.962) and gains 0.03 IoU on Thingi, but does not touch causes 1 and 2.

Distribution shift compounds all three: the Thingi training pool keeps only shapes where the baseline reaches coverage
0.5, so the policy saw few hard shapes, and the benchmark Thingi mean IoU (0.55) is far below the Thingi test-split
IoU (0.81). Stability: every SFT build that is valid on the procedural suite is stable; on Thingi, three SFT builds
and two greedy builds are valid but unstable under the simplified static force model (not physically tested).

## What this means for M6 (RL)
The reward must be computed *after* the trim (or include a connectivity term), because occupancy imitation is already
at teacher level and all the headroom is in interlocking and dead-end avoidance. Candidate features should expose
"would join the main component" so the policy can act on it. Starting RL from `sft_v1` is still right: it is valid,
plannable, and fast, and PPO over the same candidate action set needs no new machinery.
