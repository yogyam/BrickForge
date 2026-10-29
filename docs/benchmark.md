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

## Rules
- Report every case, including failures. Never evaluate only curated successes.
- Learned generators register under a new name and run with identical cases, budgets and seeds.
- Stochastic generators run several seeds; report mean and spread.
- Latency numbers are tied to the recorded hardware.
