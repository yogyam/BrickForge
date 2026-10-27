# Geometry decisions (M0)

## Canonical grid
- `x`, `y`: integer stud coordinates of the minimum footprint corner. `z`: integer plate units, up.
- A brick body occupies the half-open box `[x, x+w) x [y, y+d) x [z, z+3)`. R1 `z` is always a multiple of 3.
- Physical: stud pitch 8 mm, plate 3.2 mm, brick body 9.6 mm. Studs are connector geometry, not body.
- Rotation 90 swaps `w` and `d`; the anchor stays the minimum corner. Square parts allow only rotation 0.
- A stud connection exists only when `upper.z == lower.z + 3` and the footprints overlap. Side contact never connects.

## LDraw export (`brickcore/transforms.py`)
- 1 LDU = 0.4 mm, stud pitch 20 LDU, plate 8 LDU. `-Y` is up. Right-handed.
- Canonical to LDraw: `X = x`, `Y = -z`, `Z = y`, then scaled.
- Brick parts (3001, 3003, 3004, 3005, 3010, 3622) have their origin at the top centre and extend 24 LDU toward `+Y`.
- Translation: `X = (x + w/2) * 20`, `Y = -(z + 3) * 8`, `Z = (y + d/2) * 20` using the rotated footprint.
- Rotation 90 is the matrix `0 0 1 / 0 1 0 / -1 0 0` (90 degrees about LDraw Y). All six parts are symmetric so the sign is immaterial.
- Verified by `tests/test_transforms.py`: pushing each part's native extents through the exported matrix and translation reproduces `body_of()` exactly, for every part, rotation and several positions.

## Three.js viewer (`viewer/index.html`)
- Scene millimetres, `Y` up. Canonical to scene: `X = x`, `Y = z`, `Z = -y`, then scaled. Right-handed.
- Each placement is drawn as a box centred at `((x0+x1)/2 * 8, (z0+z1)/2 * 3.2, -(y0+y1)/2 * 8)`.
- The page contains the same formula as `brickcore.transforms.to_scene`, which is unit-tested; the two-brick fixture's expected centres are listed in `tests/test_transforms.py`.

## Two-brick fixture
`fixtures/builds/two_brick.json`: a 2x4 at the origin and a 2x2 on top. Expected LDraw lines are in
`fixtures/builds/two_brick.expected.ldr` and are asserted byte-for-byte.
