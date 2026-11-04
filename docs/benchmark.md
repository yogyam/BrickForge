# Evaluation suite (M4)

`brickforge-bench` runs every registered generator over a frozen held-out suite at several piece budgets and
seeds, writing one JSONL record per (case, generator, budget, seed) plus summary tables.

## Suites
- **Procedural** (13 cases, `ml/benchmark.py:PROCEDURAL`): shapes with known expected behaviour, including
  impossible and invalid inputs. Frozen at M4; the generator must not be tuned against them afterwards.
- **Thingi10K subset** (`fixtures/benchmark/thingi10k_subset.json`): models drawn deterministically (seed 0)
  from the Hugging Face mirror of Thingi10K, eligible only if CC BY, CC BY-SA or public domain and closed,
  manifold, single-component solids with 100 to 200k faces, one file per Thingiverse thing. License, author,
  name and source link are recorded per model. Meshes download to `data/` and are never committed.
  Thingi10K has no semantic categories; coverage of vehicles/furniture comes from tags where present and from
  the procedural suite. All Thingi meshes use `+z` up.

## Metrics per record
Target IoU, coverage, extraneous occupancy, structural validity, planning success, stability under the force
model (final state by default; `--stability steps` for every prefix), piece count, IoU per piece, per-stage
seconds, hardware, git commit, catalog version.

## Commands
```sh
./bench fetch-thingi --n 40          # one-time download (~tens of MB)
./bench run --suite all --generators greedy_v1 --budgets 50 100 200 --out docs/benchmarks/baseline_v1
./bench summarize docs/benchmarks/baseline_v1.jsonl
```

## Learned generators
`sft_v1` (learned STOP) and `sft_v1_nostop` (STOP masked; stops on empty candidate set or cap) load
`checkpoints/sft_v1.pt` (override with `BRICKFORGE_SFT_CKPT`) and decode on CPU by default
(`BRICKFORGE_SFT_DEVICE`), which is as fast as Metal for one target and reproducible across processes. Their records
carry additive diagnostic fields absent from `baseline_v1`: `raw_pieces`, `dropped_disconnected`,
`components_before_trim`, `stop_reason`. Results: `docs/benchmarks/sft_v1.*` (M5), `docs/benchmarks/rl_v1.*`, `docs/benchmarks/rl_v2.*`, `docs/benchmarks/rl_v2_s1.*` and `docs/benchmarks/rl_v2_ablations.*` (M6: main run, second training seed, ablations), analysed in `docs/sft.md` and `docs/rl.md`.
The summarizer's `mean_iou` averages over valid builds only; comparisons between generators with different validity rates use the mean over attempted cases with failed builds scored 0 (computed from the JSONL, as in `docs/rl.md`).
`sft_v1_bo8` is best-of-8 sampling from the same SFT checkpoint (seeded; ~8x decode cost). `rl_v1` loads
`checkpoints/rl_v1.pt` (`BRICKFORGE_RL_CKPT`, `BRICKFORGE_RL_DEVICE`); see `docs/rl.md`.

## Audit
`./bench audit docs/benchmarks/baseline_v1.jsonl` regenerates every non-failed record, validates it with the
authoritative validator, recomputes IoU from occupancy, and compares status, piece count, validity and IoU
with what the JSONL claims. Any mismatch fails the audit. Run it after any change to brickcore to prove the
frozen baseline is still reproducible, and before quoting numbers in the paper.

Voxelization is pinned (M6, 2026-11-01): trimesh's point-containment test re-checks ambiguous cell centres with a ray
direction drawn from numpy's global RNG, so one benchmark mesh (`thingi_225935`, a surface through a cell centre)
voxelized to 288, 289 or 290 cells depending on the process. `brickcore.mesh.voxelize` now runs that test under a
fixed, restored RNG state (`VOXELIZE_RNG_SEED = 1`, the value that reproduces the 290-cell target every frozen
record was scored against). No recorded number changed; the audit of `rl_v1` first surfaced this as 15 IoU
mismatches of at most 0.013 on that single case, all with identical builds.

## Rules
- Report every case, including failures. Never evaluate only curated successes.
- Learned generators register under a new name and run with identical cases, budgets and seeds.
- Stochastic generators run several seeds; report mean and spread.
- Latency numbers are tied to the recorded hardware.
