# BrickForge

Converts a 3D mesh into a LEGO-compatible brick build, a parts list, and checked step-by-step assembly order.
Research goal: compare a deterministic baseline, a supervised model, and RL under identical geometry and
evaluation rules, across piece budgets and a structural stability model. See `BrickForge_PRD_Design_Doc.md`.

**Status: M0 + M1 + M2 + M2.5 + M2.6 (MCP) complete.** CPU-only CLI with catalog, build schema, tested LDraw/Three.js transforms,
mesh normalization, voxelizer, greedy baseline, authoritative validator, assembly planner with insertion-path
checks and replay validation, static force-model stability analysis, BOM/LDraw export, and a static viewer.
No ML yet. No server. Nothing here is physically tested: every label reads
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
`build.ldr`, and `result.json`. `result.json` carries one of `validated_candidate`, `partial_candidate`,
`no_valid_candidate`, or `failed` with an error code, plus IoU, coverage, extraneous occupancy, per-stage
timings, and per-strategy diagnostics. A failed or partial result never produces an invented build.

## Viewer

Open `viewer/index.html` in a browser (no server needed), load a `build.json`, optionally the matching
`voxels.json` to see the target cells, optionally `stability.json` to see per-step stability, and scrub
through steps with the slider or arrow keys. The page draws
only the stored placements using the documented canonical-to-scene transform.

## MCP server

Start Claude Code in this folder and approve the `brickforge` project server from `.mcp.json`; it can then
generate, validate, plan, check stability, and inspect individual steps. See `docs/mcp.md`.

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
docs/        geometry, planner, stability and MCP notes
scripts/     mcp.sh launcher
```
