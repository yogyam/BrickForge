# BrickForge

Converts a 3D mesh into a LEGO-compatible brick build, a parts list, and checked step-by-step assembly order.
Research goal: compare a deterministic baseline, a supervised model, and RL under identical geometry and
evaluation rules, across piece budgets and a structural stability model. See `BrickForge_PRD_Design_Doc.md`.

**Status: M0 through M6 complete (baseline, SFT, PPO RL generators benchmarked; learned pipeline within 0.012 IoU of the search baseline at equal cost); ablations, seeds and photo input next.** CPU-only CLI with catalog, build schema, tested LDraw/Three.js transforms,
mesh normalization, voxelizer, greedy baseline, authoritative validator, assembly planner with insertion-path
checks and replay validation, static force-model stability analysis, BOM/LDraw export, and a static viewer.
Learned generators run through the same validator and planner as the baseline. No server. Nothing here is physically tested: every label reads
"checks passed for the simplified brick model".

## Setup

Requires `uv` and Python 3.12. If the project lives in an iCloud-synced folder (for example `~/Desktop`),
the virtualenv must live outside it, because iCloud marks files under dot-directories as hidden and Python 3.12
then refuses to load the editable-install `.pth`. The Makefile handles this; for direct `uv` use, export
`UV_PROJECT_ENVIRONMENT=$HOME/.venvs/brickforge` first.

```sh
make setup      # uv sync into ~/.venvs/brickforge
make test       # pytest
make demo       # fixtures + one success (sphere) + one intentional failure (open mesh)
```

## CLI

Use `./bf` instead of `uv run brickforge`: it points uv at the external virtualenv. A bare `uv run` without
`UV_PROJECT_ENVIRONMENT` set creates a `.venv` inside the project, which iCloud then breaks. To make plain
`uv run` work everywhere, add `export UV_PROJECT_ENVIRONMENT=$HOME/.venvs/brickforge` to your `~/.zshrc`.

```sh
./bf fixtures                                   # write 13 procedural STL fixtures
./bf generate --mesh fixtures/meshes/sphere.stl --out runs/sphere \
    --size 16 16 8 --max-pieces 100 --color 4 --up +z --yaw 0 --seed 42
./bf validate runs/sphere/build.json
./bf plan runs/sphere/build.json --write    # recompute checked steps
./bf stability runs/sphere/build.json       # force-model stability, final and per step
./bf export runs/sphere/build.json --out runs/sphere/exports
```

`generate` writes `normalization.json`, `voxels.json`, `build.json`, `validation.json`, `plan.json`, `stability.json`, `bom.csv`,
`build.ldr`, and `result.json`. Parsing (`--parse-timeout`, default 60 s) and search (`--generate-timeout`, default
30 s) have independent wall-clock limits; a parse timeout fails with `worker_timeout`, a search timeout returns the
best candidate found so far and records `search_aborted`. `result.json` carries one of `validated_candidate`, `partial_candidate`,
`no_valid_candidate`, or `failed` with an error code, plus IoU, coverage, extraneous occupancy, per-stage
timings, and per-strategy diagnostics. A failed or partial result never produces an invented build.

## Viewer

Open `viewer/index.html` in a browser (no server needed), load a `build.json`, optionally the matching
`voxels.json` to see the target cells, optionally `stability.json` to see per-step stability, and scrub
through steps with the slider or arrow keys. The page draws
only the stored placements using the documented canonical-to-scene transform.

## Learned generator (M5)

```sh
make sft-data                      # synthetic families -> data/sft/synthetic.jsonl (add --thingi N for the Thingi pool)
make sft-gate                      # 32-example overfit gate on Metal, ~2 minutes
make sft-train                     # full run: 20k steps, augmentation, ~25 minutes on Metal -> checkpoints/sft_v1.pt
make sft-bench                     # frozen benchmark with greedy_v1, sft_v1, sft_v1_nostop, then audit
```

Result (`docs/benchmarks/sft_v1.md`, 53 held-out cases, 200-piece cap, mean IoU over attempted cases with failed
builds scored 0): greedy 0.705, sft_v1 0.614, sft_v1_nostop 0.639. The supervised policy matches the baseline on procedural shapes (0.81 vs 0.83) and trails on real
meshes (0.59 vs 0.67). Before the single-component trim its occupancy IoU exceeds the teacher's; the gap is interlocking
and dead-end avoidance, not coverage. Full analysis and the implications for RL are in `docs/sft.md`.

## RL generator (M6)

```sh
make rl-gate                       # 16-example PPO gate (~5 minutes): after-trim IoU must rise above the SFT init
make rl-train                      # 250 PPO iterations from sft_v1 (~2 hours on the M5 Pro) -> checkpoints/rl_v1.pt
make rl-bench                      # frozen benchmark with greedy, sft, sft best-of-8 and rl, then audit
```

The reward is the IoU of the build *after* the component trim plus a stability bonus, so the policy is trained on what
the user actually receives.

Result (`docs/benchmarks/rl_v2.md`, 53 held-out cases, 200-piece cap, mean IoU over attempted cases with failed
builds scored 0): greedy 0.705, sft_v1 0.614, rl_v1 0.654, sft_v2 (rebalanced real-mesh data) 0.672,
**rl_v2_nostop 0.693**. The learned pipeline ties the frozen search baseline on procedural shapes (0.826 vs 0.828),
trails it by 0.015 on real meshes (0.653 vs 0.668), and matches its validity and stability counts, at the same decode
time. RL beats its own SFT initialisation 26 cases to 3; the learned STOP action stays miscalibrated on real meshes and
is masked at decode time (a choice made on the SFT test split, not the benchmark). All stability statements are under
the simplified static force model (not physically tested). Design, training curves, the decode decision and the failure
analysis: `docs/rl.md`.

```sh
make data-v2 sft-train-v2 rl-train-v2 rl-bench-v2   # the rebalanced-data ladder (~1.5 GB of CC meshes, ~2.5 h)
```

## MCP server

Start Claude Code in this folder and approve the `brickforge` project server from `.mcp.json`; it can then
generate, validate, plan, check stability, and inspect individual steps. See `docs/mcp.md`.

## Frozen baseline benchmark (M4)

`docs/benchmarks/baseline_v1.{jsonl,md,summary.json}`: greedy_v1 over 13 procedural cases and 40 CC-licensed
Thingi10K models, budgets 50/100/200, seed 42, 16x16x8 grid, commit recorded per record. Whole run: 60 s.

| budget | valid | planned | stable | mean IoU | median IoU | mean pieces |
|---|---|---|---|---|---|---|
| 50 | 0.962 | 0.962 | 0.925 | 0.633 | 0.698 | 28.4 |
| 100 | 0.962 | 0.962 | 0.925 | 0.709 | 0.893 | 37.3 |
| 200 | 0.962 | 0.962 | 0.925 | 0.719 | 0.913 | 39.1 |

What the numbers say: the two invalid cases are the intentional ones (open mesh, zero-cell pole). Doubling the
budget from 100 to 200 adds almost nothing because the greedy rarely reaches the cap; it stops when no
supported placement remains. The hard tail is thin or hollow Thingi models that occupy only 20 to 90 of the
2048 cells and fragment into unconnected islands, where the retained component covers under 20 % of the
target. Two of those tiny fragments topple under the force model (2 and 5 pieces with overhangs), the first
sign that the generator should consider stability, which it currently does not. Grid resolution is the main
lever for thin objects and is a documented limitation, not something to tune after the freeze.

Reproduce: `make bench` (downloads ~100 MB of meshes once into `data/`, never committed). Verify: `./bench audit docs/benchmarks/baseline_v1.jsonl`.

## Fixture results (M1, default 16x16x8 grid, 100-piece cap, M5 Pro)

| Fixture | Status | Coverage | Pieces | Generate (s) |
|---|---|---|---|---|
| cube (8x8x4) | validated | 1.000 | 44 | 0.05 |
| slab | validated | 1.000 | 68 | 0.13 |
| tower | validated | 1.000 | 8 | 0.01 |
| l_shape | validated | 1.000 | 90 | 0.15 |
| t_shape | validated | 1.000 | 43 | 0.07 |
| bridge | validated | 0.964 | 31 | 0.04 |
| sphere | validated | 0.975 | 75 | 0.09 |
| cylinder | validated | 0.901 | 100 | 0.13 |
| pyramid | partial (disconnected pieces dropped) | 0.946 | 77 | 0.08 |
| wedge | validated | 1.000 | 42 | 0.04 |
| flat_sheet_impossible | partial: one layer cannot interlock | 0.050 | 1 | 0.04 |
| thin_pole | no_valid_candidate: zero cells at this scale | 0 | 0 | 0.00 |
| open_box_invalid | failed: `non_watertight_mesh` | | | |

All ten simple fixtures are stable under the static force model, final state and every step. The bridge
peaks at 5.8 % of clutch capacity mid-build (cantilevering the span) and the sphere at 2.4 %; everything else
sits at zero tension. Per-step analysis costs 0.1 to 1.5 s per build.

Nine of the ten frozen simple fixtures pass the R1 gate (coverage >= 0.80 with a valid candidate); the
gate requires eight. `tests/test_generate_cli.py::test_release_gate_eight_of_ten_simple_fixtures` enforces it.

## Known limits

- Greedy baseline fills bottom-up and cannot reach cells more than one brick-width beyond the layer below.
  Overhangs beyond that are reported as uncovered, not faked.
- Budget-starved shapes stop at the cap with lower layers complete and upper layers missing.
- Stability is a static force-model diagnostic with a literature clutch constant (`docs/stability.md`), never
  a physical test. Labels say so.
- The planner checks a restricted insertion model only: straight-down insertion, no hand clearance, no
  insertion force, no clutch or gravity. See `docs/planner.md` for the rules and why backtracking is unnecessary.
- Solid fill only; hollow shells, multiple colors, plates, and other part types are later work.
- Viewer uses procedural boxes, not LDraw part geometry. LDraw part files are not bundled (CC BY 2.0 upstream).

## Layout

```
brickcore/   catalog, schema, geometry, transforms, mesh, generate, validate, export, fixtures, cli
catalog/     basic-bricks-v1.json manifest
fixtures/    builds/two_brick.json (+ expected LDraw); meshes/ are regenerated by `brickforge fixtures`
viewer/      static Three.js page
tests/       pytest suite (collision, stud connections, transforms, serialization, budgets, determinism, CLI)
ml/          Thingi10K fetcher, benchmark runner, SFT data/policy/trainer/generator (later: RL)
docs/        geometry, planner, stability, MCP and benchmark notes; docs/benchmarks/ holds frozen results
scripts/     mcp.sh launcher; ./bf and ./bench wrap the two CLIs
```
