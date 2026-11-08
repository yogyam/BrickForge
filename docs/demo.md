# BrickForge, step by step

This is the hands-on tour. It assumes nothing beyond a terminal and a browser. Every command is run from the
folder you cloned into, and every output shown here is what the commands produced on a laptop in November 2026.

## 1. What you need

- [`uv`](https://docs.astral.sh/uv/) (it installs Python 3.12 for you if needed).
- A browser for the viewer. The viewer loads Three.js from a CDN, so it needs internet the first time.
- No GPU. Everything here runs on the CPU in seconds.

## 2. Install

```sh
git clone https://github.com/yogyam/BrickForge.git
cd BrickForge
make setup       # the core: mesh loading, the greedy search, the checker, the viewer files (about a minute)
make setup-ml    # optional: adds PyTorch so the two learned policies can run too (a few hundred MB)
```

`make setup` creates the virtualenv in `~/.venvs/brickforge`, outside the project folder. That is deliberate: if
the folder is synced by iCloud (`~/Desktop` and `~/Documents` usually are), a virtualenv inside it breaks. Use the
`./bf` wrapper for every command below and you never have to think about it.

## 3. First run

```sh
make demo
```

This does four things:

1. Regenerates the 13 fixture meshes in `fixtures/meshes/` from code (they are tiny, and byte-identical to the
   committed ones).
2. Builds the sphere with the greedy search into `runs/sphere/`.
3. Tries to build `open_box_invalid.stl`, which has a face missing. **This is supposed to fail.** You will see
   `"status": "failed"` and `"error_code": "non_watertight_mesh"`. BrickForge does not guess what the inside of an
   open mesh looks like, so it refuses, and the demo shows you what that refusal looks like.
4. Builds the same sphere with the learned policy into `runs/sphere_rl/`. If you skipped `make setup-ml` this step
   prints one line saying torch is missing and moves on.

The sphere summary that step 2 prints:

```json
{
  "coverage": 0.9745,
  "extraneous": 0.0,
  "generator": "greedy_v1",
  "instruction_status": "planned",
  "iou": 0.9745,
  "label": "checks passed for the simplified brick model",
  "pieces": 75,
  "stability": {
    "all_steps_stable": true,
    "final_stable": true,
    "label": "stable under the simplified static force model; not physically tested",
    ...
  },
  "status": "validated_candidate",
  "timings_s": { "generate": 0.17, "load": 0.005, "normalize": 0.001, "stability": 2.25, "voxelize": 0.11 }
}
```

How to read it: the build covers 97 % of the target shape (`coverage`), adds nothing outside it (`extraneous`),
uses 75 of the 100 allowed pieces, has a checked assembly order (`instruction_status: planned`), and every one of
the 75 intermediate stages holds together under the force model. Most of the three seconds goes to that per-step
stability check.

## 4. What is in the output folder

```sh
ls runs/sphere
```

| file | what it is | open it with |
|---|---|---|
| `result.json` | the summary above, plus a `diagnostics` block with what each search strategy tried | any text editor |
| `build.json` | the build itself: every brick (`part_key`, `x`, `y`, `z`, `rotation_deg`, `color_id`) and the assembly `steps` | the viewer |
| `voxels.json` | the target shape as a list of grid cells `[x, y, layer]` | the viewer (optional) |
| `stability.json` | the force model's verdict for the final build and after each step | the viewer (optional) |
| `plan.json` | the planner's status and step count | text editor |
| `validation.json` | the independent checker's report: `ok`, `errors`, `warnings`, how many pieces touch the ground, stud connections | text editor |
| `normalization.json` | how the mesh was rotated, scaled and placed on the grid (so you can map coordinates back) | text editor |
| `bom.csv` | parts list: part, LDraw file, colour, count | spreadsheet |
| `build.ldr` | the build as an LDraw model, one `0 STEP` per assembly step | LDView, Stud.io, LeoCAD |

Coordinates in `build.json` are in studs for `x` and `y` and in plates for `z` (a brick is 3 plates tall). The
default grid is 16 studs by 16 studs by 8 bricks.

## 5. The four outcomes

`status` in `result.json` is one of four words, and the fixtures include a shape for each:

| status | meaning | try it |
|---|---|---|
| `validated_candidate` | legal build, covers at least 80 % of the shape, nothing dropped | `./bf generate --mesh fixtures/meshes/cube.stl --out runs/cube --size 8 8 4` gives IoU 1.0 with 44 bricks |
| `partial_candidate` | legal, but covers under 80 %, or disconnected pieces had to be dropped | `./bf generate --mesh fixtures/meshes/flat_sheet_impossible.stl --out runs/sheet` keeps 1 brick: a single layer of side-by-side bricks has no stud connections, so only one can stay |
| `no_valid_candidate` | no legal build at all | `./bf generate --mesh fixtures/meshes/thin_pole.stl --out runs/pole`: a 4 mm pole is thinner than one cell at this scale, so it voxelizes to nothing |
| `failed` | the input was rejected before any search | `./bf generate --mesh fixtures/meshes/open_box_invalid.stl --out runs/open` (exit code 1) |

Only `failed` and `no_valid_candidate` leave you without a `build.json`. The exit code is 0 for the first three
statuses and 1 for `failed`; a bad command line (unknown generator, piece budget over 200, grid over 24 studs) exits
with 2 and a one-line message.

The other fixtures are `slab`, `tower`, `l_shape`, `t_shape`, `bridge`, `cylinder`, `pyramid` and `wedge`. The
pyramid and sphere have overhangs, the bridge has a span, the cylinder is the one that runs out of budget first.

## 6. Look at it in the viewer

Open [`viewer/index.html`](../viewer/index.html) by double-clicking it. No server, no install. Then:

1. **build.json**: pick `runs/sphere/build.json`. The bricks appear, built up to the last step.
2. **voxels.json** (optional): pick `runs/sphere/voxels.json`. The target cells show as small translucent blue
   cubes, so you can see where the build misses. Untick "show target cells" to hide them.
3. **stability.json** (optional): pick `runs/sphere/stability.json`. The line under the buttons now reports, for
   the current step, whether that partial build is stable under the force model and how close its worst stud joint
   is to its limit.

Drag to orbit, scroll to zoom, right-drag to pan. The slider, the Previous/Next buttons and the left/right arrow
keys step through the assembly one brick at a time; the newest brick is drawn slightly brighter. "All" jumps to
the finished build.

The viewer draws plain boxes at the stored positions. It never shows anything that is not in `build.json`.

## 7. Three generators, same rules

Everything after brick placement (trimming, planning, checking, stability) is shared. The only thing that changes
with `--generator` is who picks the bricks:

```sh
./bf generate --mesh fixtures/meshes/sphere.stl --out runs/sphere_greedy --generator greedy_v1
uv run --group ml brickforge generate --mesh fixtures/meshes/sphere.stl --out runs/sphere_sft --generator sft_v2
uv run --group ml brickforge generate --mesh fixtures/meshes/sphere.stl --out runs/sphere_rl  --generator rl_v2_nostop
```

(`./bf` does not add the `ml` group, so the learned generators are run with `uv run --group ml brickforge`. If you
set `UV_PROJECT_ENVIRONMENT=$HOME/.venvs/brickforge` in your shell, that is all you need.)

On the sphere at 100 pieces:

| generator | what it is | IoU | pieces | time |
|---|---|---|---|---|
| `greedy_v1` | hand-written search, layer by layer | 0.974 | 75 | 0.17 s |
| `sft_v2` | 100k-parameter network trained to imitate the search | 0.959 | 69 | 1.3 s |
| `rl_v2_nostop` | the same network after reinforcement learning, stop action masked | 0.957 | 67 | 0.35 s |

The two checkpoints `checkpoints/sft_v2.pt` and `checkpoints/rl_v2.pt` are committed (about 400 KB each) so this
comparison works from a fresh clone. All three builds are stable under the simplified static force model; none has
been physically tested. The full comparison on 53 held-out shapes is in the README and the paper; the fixtures
above are not part of that benchmark.

Other names that work once you have trained them (see the README's reproduction section): `sft_v1`, `rl_v1`,
`rl_v2`, `rl_v2_s1`, and `_nostop` variants of the RL ones. Checkpoints not present print a one-line message that
names the `make` target that trains them.

## 8. Your own model

```sh
./bf generate --mesh path/to/model.stl --out runs/my_model --max-pieces 120 --up +y
```

- **Formats:** `.stl` (binary or ASCII) and single-file `.glb`. Anything else is rejected with
  `unsupported_input`. Export from Blender, MeshLab or your CAD tool as STL.
- **Watertight only.** The mesh must be a closed solid. Open meshes, scans with holes and "shells" fail with
  `non_watertight_mesh`. Most tools have a "make manifold" or "close holes" command.
- **Limits:** 20 MB, 200 000 triangles.
- `--up`: which axis of your file points up (`+z` default; many exporters use `+y`).
- `--yaw`: rotate about up in 90 degree steps if the long side lands the wrong way.
- `--size X Y LAYERS`: the grid, up to 24 x 24 studs and 12 layers. Bigger grids mean finer detail and more bricks.
- `--max-pieces`: 1 to 200. The default is 100.
- `--color`: LDraw colour code (`4` red, `1` blue, `2` green, `14` yellow, `15` white, `0` black).

The mesh is scaled uniformly to fit the grid, so a long thin object gets few cells. If the result is
`no_valid_candidate` or badly fragmented, try a bigger `--size`.

## 9. The other commands

```sh
./bf validate runs/sphere/build.json             # re-check every rule; exit 0 means ok
./bf plan runs/sphere/build.json --write         # recompute the assembly order and store it in the file
./bf stability runs/sphere/build.json --out s.json   # force model, final and per step; exit 0 means stable under the model
./bf export runs/sphere/build.json --out out/    # bom.csv + build.ldr (refuses an invalid build)
./bf fixtures --out somewhere/                   # write the 13 fixture meshes elsewhere
./bf generate --help                             # every flag with its range
```

These all take a `build.json`, so you can hand-edit a build and see what the checker says about it.

## 10. Driving it from Claude Code

The repo ships an MCP server (`.mcp.json` points at `scripts/mcp.sh`). Start Claude Code in this folder, approve the
`brickforge` server when asked, and try:

> generate fixtures/meshes/bridge.stl into runs/bridge with 40 pieces, then tell me which step has the highest
> tension and what it attaches to

The tools are `list_fixtures`, `generate` (same flags as the CLI, including `generator`), `validate`, `plan`,
`stability`, `inspect_step`, `bom` and `describe_build`. Paths are confined to the project folder. See
[`docs/mcp.md`](mcp.md).

## 11. When something goes wrong

| you see | why | fix |
|---|---|---|
| `learned generators need torch` | `make setup` does not install PyTorch | `make setup-ml`, then use `uv run --group ml brickforge ...` |
| `needs a checkpoint that is not here` | only `sft_v2.pt` and `rl_v2.pt` are committed | use those names, or train the others with the named `make` target |
| `non_watertight_mesh` | the mesh has holes or is a surface, not a solid | close it in your modelling tool and re-export |
| `unsupported_input` | not `.stl` or `.glb` | export as STL |
| `no_valid_candidate` with `target occupies zero grid cells` | the object is too thin at this scale | increase `--size` |
| blank viewer, no bricks | offline (Three.js comes from a CDN), or the wrong file in a picker | reconnect, reload, pick `build.json` in the first box |
| `uv` says it cannot find a project | you are not in the repo folder | `cd` into it; `make` and `./bf` both assume that |
| Python cannot import `brickcore` after `uv sync` | a `.venv` inside an iCloud-synced folder | use `./bf`, or `export UV_PROJECT_ENVIRONMENT=$HOME/.venvs/brickforge` |
| `warning: VIRTUAL_ENV=... does not match` | another Python environment (often conda) is active in your shell | harmless; `make` and `./bf` silence it, or `conda deactivate` |
| `make test` fails on `test_rl.py` | an old checkout; the test now skips without torch | `git pull`, or `make test-ml` |

## 12. Where next

- The README's research section and results table.
- [`docs/rl.md`](rl.md) and [`docs/sft.md`](sft.md) for how the learned policies were trained and what did not work.
- [`docs/stability.md`](stability.md) and [`docs/planner.md`](planner.md) for the two checkers.
- The paper in [`paper/`](../paper/) (`make paper` builds the PDF).
