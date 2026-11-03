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

Mean after-trim IoU over the 52 attempted cases (the invalid-input fixture is excluded for every generator; a build
that fails validation or is empty scores 0, so a generator cannot improve its mean by failing). Note that the
summarizer's `mean_iou` column in `docs/benchmarks/*.md` averages over *valid* builds only and therefore reads higher
for generators with failures; all comparisons in this document use the attempted-case mean. Greedy decode unless stated:

| generator | 50 | 100 | 200 | procedural @200 (12) | Thingi10K @200 (40) | valid @200 | stable @200 | decode s/case |
|---|---|---|---|---|---|---|---|---|
| greedy_v1 (frozen baseline) | 0.621 | 0.696 | **0.705** | 0.828 | **0.668** | 51/53 | 49/53 | 0.07 |
| sft_v1 | 0.523 | 0.602 | 0.614 | 0.812 | 0.554 | 49/53 | 46/53 | 0.11 |
| sft_v1_nostop | 0.547 | 0.628 | 0.639 | 0.813 | 0.588 | 51/53 | 49/53 | 0.12 |
| sft_v1_bo8 (8 samples) | 0.601 | 0.675 | 0.686 | 0.827 | 0.643 | 51/53 | 49/53 | 0.91 |
| **rl_v1** | 0.563 | 0.643 | 0.654 | 0.826 | 0.603 | 49/53 | 45/53 | 0.11 |

Head-to-head at the 200 cap (rl_v1 wins / loses / ties, 0.001 tolerance): vs sft_v1 22/5/26, vs sft_v1_nostop
21/10/22, vs sft_v1_bo8 8/17/28, vs greedy_v1 7/24/22. Greedy rows are identical to `baseline_v1` (159/159).
The audit regenerated all 780 non-failed records; its first pass exposed a process-dependent voxelization of one
mesh (fixed and documented in `docs/benchmark.md`), after which it passed.

What the numbers say:

1. **RL fixes what imitation got wrong, at no decode cost.** +0.040 mean IoU over sft_v1 at every budget, 22 wins to 5
   losses, fragmented builds 27 -> 22 of 52 and mean pieces dropped by the trim 6.0 -> 3.2. On procedural shapes rl_v1
   ties the baseline (0.826 vs 0.828). Against best-of-8 sampling it is 0.03 behind (8 wins, 17 losses) at one eighth
   of the decode time; on the builds both produce they are close, the gap is two real meshes where rl_v1 stops before
   placing a piece and best-of-8 finds a sample that does not.
2. **It still trails the frozen baseline on real meshes (0.603 vs 0.668).** Twenty of the 24 losses to greedy end with the
   policy's own STOP action while target cells remain, and three of the five losses larger than 0.2 IoU
   (`thingi_39108`, `thingi_39922`, `thingi_48124`) are early stops on shapes unlike the training distribution; on
   `thingi_48124` and `thingi_42025` the policy stops before placing a single piece, as sft_v1 does. The training set is
   93 % procedural families; its 117 real-mesh training examples are only shapes on which the teacher reached coverage
   0.5, because `make_example` needs a usable teacher build. So the hard real meshes, exactly where the baseline scores
   0.05-0.4 and where RL could do better, are absent from training, and the terminal reward never penalised stopping on
   them. The 64-example validation subset, drawn from the same distribution, cannot see this either: it reports 0.925
   while the benchmark's Thingi10K subset reports 0.603. That gap, not the PPO recipe, is the limiting factor.
3. **Stability is unchanged where it matters and slightly worse at the tail.** Every procedural build is stable under the
   simplified static force model (not physically tested). rl_v1 has two more unstable Thingi10K builds than the
   baseline, both tiny fragments (3 and 4 pieces) left after trimming an early-stopped build, the same failure as above.

### What this means for the paper and the next step

The research question's SFT+RL arm is now in place: greedy 0.705 / SFT 0.614 / SFT+RL 0.654 at the 200 cap, with the
budget curve (50/100/200) for every arm in `docs/benchmarks/rl_v1.md`. The honest headline is that RL recovers almost
half of the imitation gap at no decode cost, while the search baseline keeps its lead on out-of-distribution meshes
because the learned policies stop early on them.

The next lever is the training distribution, not the algorithm: RL needs no teacher, so it can train on every eligible
Thingi10K mesh outside the frozen benchmark subset, including the ones the baseline handles badly and the SFT data
had to drop, and real meshes can be weighted up in the episode mix so the STOP action is trained on real shapes. A cheaper
diagnostic is decoding rl_v1 with STOP masked, as `sft_v1_nostop` does, but that choice must be made on the SFT
test split, not on the benchmark, and has not been made.

Ablations planned for the paper, each a full run (`--stab-weight 0`, `--no-conn-feats --no-main-channel`): about
2.5 h each on the M5 Pro.

## Part 3: training on the shapes the benchmark is made of (rl_v2)

Diagnosis from part 2, checked on the SFT split: on the 33 real-mesh validation and test examples rl_v1 scores 0.826
against the teacher's 0.890 (11/33 at or above it), so the learned policies trail the search even on real meshes they
trained on, and those were 7 % of training. A census of the 290 downloaded training meshes: 150 kept, 87 dropped only
because the teacher's coverage was under 0.5, 32 too small on the grid, 9 too large, 12 unusable.

Changes (code: `ml/data.py`, `ml/train_sft.py`, `ml/train_rl.py`; `make data-v2 sft-train-v2 rl-train-v2 rl-bench-v2`):
- Dataset v2: the same 2000 procedural examples plus every eligible Thingi10K training mesh (809 files, one per
  Thingiverse thing, every held-out thing excluded, up to 200k faces). Targets whose teacher build covers under half
  the target are kept with `coverage` recorded; RL trains on them (it needs only the reward), SFT drops them
  (`--min-teacher-coverage 0.5`, the previous rule).
- Sampling: `--real-frac 0.5` draws half of SFT pool examples and half of RL episodes from real meshes; the RL eval
  subset mirrors that mix so checkpoint selection is made on the distribution being trained.
- Ladder: `sft_v2` (same recipe as sft_v1) and `rl_v2` (same PPO recipe, initialised from sft_v2), so the comparison
  with sft_v1 / rl_v1 isolates the data change and the comparison between sft_v2 and rl_v2 isolates RL.
- Decode choices for rl_v2 (learned STOP or masked) are made on the SFT test split before the benchmark is run.

### Decode choice, made on the SFT test split before the benchmark (2026-11-02)

v2 test split, 260 examples (188 procedural, 72 real meshes of which 30 have a teacher below coverage 0.5), 100-piece
cap, after-trim IoU:

| checkpoint | STOP | all | procedural | real | real, weak teacher |
|---|---|---|---|---|---|
| teacher (greedy) | | 0.843 | 0.943 | 0.584 | |
| sft_v1 | learned | 0.770 | 0.863 | 0.527 | 0.175 |
| sft_v1 | masked | 0.775 | 0.864 | 0.543 | 0.184 |
| sft_v2 | learned | 0.766 | 0.870 | 0.493 | 0.123 |
| sft_v2 | masked | 0.782 | 0.876 | 0.535 | 0.179 |
| rl_v1 | learned | 0.824 | 0.936 | 0.531 | 0.190 |
| rl_v1 | masked | 0.835 | 0.936 | 0.569 | 0.206 |
| rl_v2 | learned | 0.828 | 0.937 | 0.542 | 0.147 |
| rl_v2 | masked | 0.840 | 0.939 | 0.581 | 0.208 |
| rl_v2_s1 (seed 1, added 2026-11-03) | learned | 0.821 | 0.931 | 0.533 | 0.141 |
| rl_v2_s1 | masked | 0.835 | 0.935 | 0.574 | 0.204 |

Masking STOP never costs anything on procedural shapes and gains about 0.04 on real meshes for all three RL
checkpoints, so `rl_v1_nostop`, `rl_v2_nostop` and `rl_v2_s1_nostop` are registered and benchmarked alongside the
learned-STOP decodes. The learned
STOP remains miscalibrated on real meshes even after RL on them (v2); the reward does penalise early stopping, but the
policy still prefers STOP on shapes where every remaining candidate looks poor. Also visible: the data change alone
(sft_v1 -> sft_v2) gains little at the SFT stage; RL lifts both to within 0.01 of the teacher on this split.

### Frozen benchmark, part 3 (`docs/benchmarks/rl_v2.md`, 53 cases, 1092 records, audit passed 2026-11-02)

Training: sft_v2 20k steps (28 min, best val 0.840 at step 12k); rl_v2 250 PPO iterations from sft_v2 (1 h 35 min,
best checkpoint at iteration 170: validation IoU 0.713 -> 0.781 on a half-real-mesh subset, fragmentation 50 % -> 36 %,
98 % stable, teacher 0.805 on the same subset).

Mean after-trim IoU over the 52 attempted cases, failed builds scored 0:

| generator | 50 | 100 | 200 | procedural @200 (12) | Thingi10K @200 (40) | valid @200 | stable @200 | decode s/case |
|---|---|---|---|---|---|---|---|---|
| greedy_v1 (frozen baseline) | 0.621 | 0.696 | **0.705** | 0.828 | **0.668** | 51/53 | 49/53 | 0.07 |
| sft_v1 | 0.523 | 0.602 | 0.614 | 0.812 | 0.554 | 49/53 | 46/53 | 0.12 |
| rl_v1 | 0.563 | 0.643 | 0.654 | 0.826 | 0.603 | 49/53 | 45/53 | 0.12 |
| rl_v1_nostop | 0.593 | 0.676 | 0.688 | 0.826 | 0.646 | 51/53 | 48/53 | 0.14 |
| sft_v2 | 0.573 | 0.663 | 0.672 | 0.793 | 0.636 | 49/53 | 48/53 | 0.11 |
| rl_v2 | 0.573 | 0.648 | 0.659 | 0.803 | 0.615 | 47/53 | 46/53 | 0.12 |
| **rl_v2_nostop** | 0.607 | 0.681 | **0.693** | 0.826 | **0.653** | 51/53 | 49/53 | 0.13 |

Head-to-head at the 200 cap (wins / losses / ties, 0.001 tolerance):

| | vs greedy_v1 | vs sft_v1 | vs sft_v2 | vs rl_v1_nostop |
|---|---|---|---|---|
| rl_v2_nostop | 10 / 16 / 27 | 31 / 7 / 15 | 26 / 3 / 24 | 15 / 14 / 24 |
| rl_v1_nostop | 9 / 20 / 24 | 30 / 1 / 22 | 21 / 12 / 20 | |
| sft_v2 | 4 / 26 / 23 | 28 / 12 / 13 | | |

What the numbers say:

1. **The learned pipeline is now within 0.012 IoU of the hand-written search at the same decode cost.** rl_v2_nostop
   0.693 vs greedy 0.705 overall, 0.653 vs 0.668 on real meshes, a tie on procedural shapes, identical validity (51/53)
   and stability (49/53, under the simplified static force model; not physically tested). Head to head it wins 10 cases,
   loses 16 and ties 27. The median case is a tie: the remaining gap sits in a handful of real meshes.
2. **Each stage earned its place.** Data rebalancing moved SFT from 0.614 to 0.672; RL on top moved it to 0.693 and
   beats its own SFT initialisation 26 to 3. On the two unchanged-data arms RL added +0.040 (sft_v1 -> rl_v1) and the
   decode fix added another +0.034 (rl_v1 -> rl_v1_nostop).
3. **The learned STOP is the one thing RL did not fix.** rl_v2 with its own STOP is *worse* than rl_v2_nostop by 0.034
   and loses validity on four real meshes (`thingi_39108`, `110201`, `111878`, `567026`) by stopping before placing a
   single piece, even though the reward penalises exactly that and half its training episodes were real meshes. The
   policy learned that on shapes where every candidate looks poor, STOP has the highest expected value under its own
   future behaviour; masking it forces tiling and the trim keeps the largest connected part. Masked decode fragments
   more (25/40 real builds vs 14/40) but ends with more target covered.
4. **The remaining loss is one shape class.** `thingi_39108` (a honeycomb pot) is the only case any learned generator
   loses by more than 0.35: greedy 0.80, rl_v2_nostop 0.45 with 19 of 39 pieces trimmed as unconnected. Thin-walled
   shells with interior structure are where the search's exhaustive candidate scan still beats a learned preference.

### Second training seed (`docs/benchmarks/rl_v2_s1.md`, 53 cases, 636 records, audit passed 2026-11-03)

The v2 ladder was retrained end to end with training seed 1 (`make v2-seed1 rl-bench-v2-seeds`): sft_v2_s1 20k steps
(25 min, best val 0.851 at step 15k), rl_v2_s1 250 PPO iterations from it (2 h 22 min, best checkpoint at iteration
180: validation IoU 0.724 -> 0.791 on the half-real-mesh subset, fragmentation 38 % -> 22 %, 95 % stable, teacher
0.805). The STOP decision was re-made on the SFT test split (table above, not on the benchmark): masking again gains
0.04 on real meshes and nothing on procedural shapes, so `rl_v2_s1_nostop` is the decode. Same frozen suite; greedy
rows identical to `baseline_v1` (159/159).

Mean after-trim IoU over the 52 attempted cases, both seeds: mean, with the seed-0 to seed-1 range in brackets.

| arm | 50 | 100 | 200 | procedural @200 | Thingi10K @200 | W / L vs greedy @200 (s0; s1) |
|---|---|---|---|---|---|---|
| greedy_v1 (frozen baseline) | 0.621 | 0.696 | **0.705** | 0.828 | 0.668 | |
| sft_v2 | 0.566 (0.559-0.573) | 0.653 (0.643-0.663) | 0.662 (0.652-0.672) | 0.788 (0.783-0.793) | 0.624 (0.613-0.636) | 4/26; 4/28 |
| rl_v2 (learned STOP) | 0.583 (0.573-0.593) | 0.657 (0.648-0.666) | 0.667 (0.659-0.675) | 0.803 (0.803-0.803) | 0.626 (0.615-0.637) | 7/27; 7/25 |
| **rl_v2_nostop** | 0.609 (0.607-0.611) | 0.682 (0.681-0.684) | **0.693** (0.693-0.693) | 0.825 (0.824-0.826) | 0.653 (0.653-0.654) | 10/16; 11/19 |

Seed 1 alone at the 200 cap: sft_v2_s1 0.652 (valid 48/53, stable 44/53), rl_v2_s1 0.675 (48/53, 46/53),
rl_v2_s1_nostop 0.693 (51/53, 47/53), decode 0.13 to 0.15 s/case. Head to head rl_v2_s1_nostop beats sft_v2_s1
27 / 9 / 16 and splits 14 / 16 / 22 against the seed-0 rl_v2_nostop.

What the second seed changes:

1. **The headline reproduces.** rl_v2_nostop ends at 0.693 for both seeds (range 0.000 at the 200 cap, 0.004 at 50)
   and sits 0.012 behind the search either way. The per-case picture is also stable: 12 cases are lost to greedy in
   both seeds (`proc_sphere`, `proc_wedge`, and ten real meshes including `thingi_39108` at 0.80 vs 0.45 / 0.42) and
   the same 6 real meshes are won in both.
2. **RL removes the SFT seed spread.** The SFT stage differs by 0.020 between seeds (0.672 vs 0.652); RL then adds
   +0.021 to seed 0 and +0.041 to seed 1 and both land on 0.693. The RL gain over its own initialisation is therefore
   +0.02 to +0.04 (positive in both seeds, 26/3 and 27/9 head to head). The data-rebalancing gain at the SFT stage,
   +0.058 and +0.038 against the single sft_v1 seed, is less precisely known than the seed-0 table suggested.
3. **The learned STOP fails on different meshes per seed.** Seed 1 stops before placing anything on `thingi_108578`,
   `42025` and `567026` (seed 0: `39108`, `110201`, `111878`, `567026`); only `567026` is shared. The cost of the
   learned STOP versus masking is 0.018 (seed 1) to 0.034 (seed 0).
4. **Stability is the one count where seed 1 is slightly worse**: 47/53 stable at the 200 cap against 49/53 for
   seed 0 and for greedy, with identical validity (51/53). Stable here means stable under the simplified static force
   model; not physically tested.

Two seeds give a range, not a confidence interval; the paper states both per-seed numbers and the range. The v1 arms
(sft_v1, rl_v1) and the ablations remain single-seed.

### What this means for the paper

Piece budget vs fidelity across greedy / SFT / SFT+RL is complete at 50/100/200 for every arm, with the controlled
ladder sft_v1 -> rl_v1 and sft_v2 -> rl_v2 isolating the RL effect at each data level, and rl_v1 -> rl_v2 isolating
the data effect. The defensible claim: a 100k-parameter policy trained in under five hours on a laptop reaches 98 % of
a hand-engineered search's fidelity at equal decode cost, with RL rather than more supervised data providing most of
the gain beyond imitation; the search keeps a small lead on thin shells, and learned stopping remains an open problem.
The endpoint is reproduced by a second training seed (0.693 in both), so the sft -> rl difference of +0.02 to +0.04 is
stated with its range rather than as one observation.

Remaining for the paper: the two ablations (`--stab-weight 0`, `--no-conn-feats --no-main-channel`, about 2 h each
at the v2 settings, one seed each), and optionally a second seed for the v1 arms.
