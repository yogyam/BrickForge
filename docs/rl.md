# Reinforcement learning generator (M6)

## Why RL here
M5 showed that supervised imitation reaches the teacher's *occupancy* IoU but loses 0.09 IoU to the single-component
trim and dead ends (`docs/sft.md`). Those failures are invisible to a per-step imitation loss and only show up in the
finished build. RL scores the finished build directly.

## Environment (`ml/rl_env.py`)
- State: target grid, filled grid, the placed pieces, and a union-find over stud connections (pieces directly below
  and directly above each new piece). The main component is the largest grounded one, ties to the earliest piece,
  exactly as `brickcore.generate._largest_ground_component` decides the trim (tested on random builds).
- Actions: the same legal candidate set as SFT (`brickcore/candidates.py`) plus STOP. Legality is still inherent.
- Observation extensions over SFT: a fourth voxel channel marking cells of the main component, and two per-candidate
  features, *joins main component* and *merges two components*. SFT weights are zero-padded into this layout
  (`ml.policy.extend_state_dict`), so the initial RL policy produces exactly the SFT policy's logits (checked on 40
  rollouts). Both extensions can be disabled for ablations (`--no-conn-feats`, `--no-main-channel`).
- Terminal reward: after the shared finalize step (trim, planner, validator),
  `reward = IoU_after_trim + 0.1 * stable`, where `stable` is the static force model's verdict on the final trimmed
  build. No per-step shaping. Validity cannot fail under the candidate rules, so no penalty is needed.

## Algorithm (`ml/train_rl.py`)
PPO with a value head on the same encoder; advantage = reward minus value (gamma 1, terminal reward only), normalised
per batch; clipped ratio 0.2; entropy bonus 0.01; KL penalty 0.05 to the frozen initial policy; Adam 1e-4; gradient
clipping 1.0. Episodes are sampled (temperature 1) in forked CPU workers, 8 at a time; updates run on Metal. Masked
actions are removed from the entropy and KL terms before multiplication, otherwise -inf logits poison the backward
pass with NaNs (this bug was caught by the tests and by the first gate run).

## Gate (passed 2026-11-01)
`--overfit 16 --iters 20 --episodes 48`: after-trim IoU on the 16 examples 0.952 -> 1.000, fragmented builds
2/16 -> 0/16, all stable, 16/16 at or above the teacher. Rollouts 2 s and update 12 s per iteration on the M5 Pro.
Run log: `runs/rl/rl_gate16/metrics.jsonl` (not committed).

## Baselines the RL policy is compared against
- `greedy_v1`: the frozen deterministic baseline.
- `sft_v1`, `sft_v1_nostop`: imitation, greedy decode.
- `sft_v1_bo8`: best-of-8 sampling from the SFT policy, keeping the best validated after-trim IoU. This is a
  test-time-compute baseline at ~8x decode cost; on a 30-example test-split sample it lifts sft_v1 from 0.923 to 0.968.
  RL has to beat it, or at least match it at 1x cost, to be worth its training.

## Full run
`make rl-train` (250 iterations, 128 episodes each, symmetry augmentation; best checkpoint by after-trim IoU on a fixed
64-example validation subset -> `checkpoints/rl_v1.pt`), then `make rl-bench` (frozen benchmark with all generators,
then audit).

## Full-run result (2026-11-01)

Training: 250 iterations x 128 episodes, 2 h 25 min on the M5 Pro (rollouts 5-7 s, PPO update 30-35 s per
iteration; the update dominates because every step's candidate set is re-scored on Metal). Policy entropy stayed
near 1.3-1.4 nats and the KL to the frozen SFT policy settled around 0.2, so the policy moved but did not collapse.
Log: `runs/rl/rl_v1/metrics.jsonl` (not committed); checkpoint: `checkpoints/rl_v1.pt` from iteration 220.

Validation subset (64 fixed examples from the SFT validation split, greedy decode, after-trim IoU):

| iteration | IoU | fragmented | stable | at or above teacher |
|---|---|---|---|---|
| 0 (SFT init, padded) | 0.870 | 25 % | 100 % | 40/64 |
| 80 | 0.916 | 17 % | 100 % | 50/64 |
| 150 | 0.921 | 16 % | 100 % | 50/64 |
| 220 (best, saved) | 0.925 | 12 % | 100 % | 51/64 |
| 250 (final) | 0.918 | 14 % | 100 % | 49/64 |

The gain is where M5 said it would be: fragmentation halves and the mean number of pieces dropped by the trim falls
from 3.7 to 1.7 per build. The teacher's IoU on the same subset is 0.923, so the RL policy matches the search-based
teacher it was distilled from, in-distribution.

### Frozen benchmark (`docs/benchmarks/rl_v1.md`, 53 held-out cases, 795 records, audit passed)

Mean after-trim IoU (procedural 13 cases / Thingi10K 40 cases), greedy decode unless stated:

| generator | 50 | 100 | 200 | procedural @200 | Thingi10K @200 | stable @200 | decode s/case |
|---|---|---|---|---|---|---|---|
| greedy_v1 (frozen baseline) | 0.633 | 0.709 | **0.719** | 0.764 | **0.668** | 49/53 | 0.07 |
| sft_v1 | 0.555 | 0.639 | 0.651 | 0.749 | 0.554 | 46/53 | 0.11 |
| sft_v1_nostop | 0.558 | 0.641 | 0.652 | 0.750 | 0.588 | 49/53 | 0.12 |
| sft_v1_bo8 (8 samples) | 0.612 | 0.688 | 0.699 | 0.764 | 0.643 | 49/53 | 0.91 |
| **rl_v1** | 0.597 | 0.682 | 0.694 | 0.762 | 0.603 | 45/53 | 0.11 |

Head-to-head at the 200 cap (rl_v1 wins / loses / ties, 0.001 tolerance): vs sft_v1 22/5/26, vs sft_v1_nostop
21/10/22, vs sft_v1_bo8 8/17/28, vs greedy_v1 7/24/22. Greedy rows are identical to `baseline_v1` (159/159).
The audit regenerated all 780 non-failed records; its first pass exposed a process-dependent voxelization of one
mesh (fixed and documented in `docs/benchmark.md`), after which it passed.

What the numbers say:

1. **RL fixes what imitation got wrong, at no decode cost.** +0.043 mean IoU over sft_v1 at every budget, 22 wins to 5
   losses, fragmented builds 27 -> 22 of 52 and mean pieces dropped by the trim 6.0 -> 3.2. On procedural shapes rl_v1
   ties the baseline (0.762 vs 0.764). It reaches the best-of-8 policy's mean within 0.005 at one eighth of its decode
   time, which is the comparison that justifies training rather than sampling.
2. **It still trails the frozen baseline on real meshes (0.603 vs 0.668).** Twenty of the 24 losses to greedy end with the
   policy's own STOP action while target cells remain, and three of the five losses larger than 0.2 IoU
   (`thingi_39108`, `thingi_39922`, `thingi_48124`) are early stops on shapes unlike anything in the synthetic training
   families; on `thingi_48124` and `thingi_42025` the policy stops before placing a single piece, as sft_v1 does.
   Training never shows the policy such targets, so the terminal reward never penalised stopping on them. The 64-example
   validation set, drawn from the same synthetic families, cannot see this either: it reports 0.925 while the benchmark's
   Thingi10K subset reports 0.603. That gap, not the PPO recipe, is the limiting factor.
3. **Stability is unchanged where it matters and slightly worse at the tail.** Every procedural build is stable under the
   simplified static force model (not physically tested). rl_v1 has two more unstable Thingi10K builds than the
   baseline, both tiny fragments (3 and 4 pieces) left after trimming an early-stopped build, the same failure as above.

### What this means for the paper and the next step

The research question's SFT+RL arm is now in place: greedy 0.719 / SFT 0.651 / SFT+RL 0.694 at the 200 cap, with the
budget curve (50/100/200) for every arm in `docs/benchmarks/rl_v1.md`. The honest headline is that RL recovers two
thirds of the imitation gap in-distribution and matches a test-time-compute baseline at one eighth the cost, while the
search baseline keeps its lead on out-of-distribution meshes because the learned policies stop early on them.

The next lever is the training distribution, not the algorithm: add Thingi10K-like targets (voxelised CC meshes
*outside* the frozen benchmark subset) to the RL episode pool so the STOP action is trained on real shapes. A cheaper
diagnostic is decoding rl_v1 with STOP masked, as `sft_v1_nostop` does, but that choice must be made on the SFT
test split, not on the benchmark, and has not been made.

Ablations planned for the paper, each a full run (`--stab-weight 0`, `--no-conn-feats --no-main-channel`): about
2.5 h each on the M5 Pro.
