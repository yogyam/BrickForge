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

## Next
Full training on all synthetic + Thingi training examples with augmentation, then the frozen benchmark with
`--generators greedy_v1 sft_v1`, then `./bench audit`.
