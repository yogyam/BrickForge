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
then audit). Results are appended below when the run completes.
