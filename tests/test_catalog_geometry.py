from brickcore.geometry import Body, body_of, stud_connected
from brickcore.schema import Placement


def test_catalog_has_six_parts_with_expected_footprints(catalog):
    fp = {k: (p.w, p.d) for k, p in catalog.parts.items()}
    assert fp == {"brick_1x1": (1, 1), "brick_1x2": (2, 1), "brick_1x3": (3, 1), "brick_1x4": (4, 1), "brick_2x2": (2, 2), "brick_2x4": (4, 2)}
    assert all(p.height_plates == 3 for p in catalog.parts.values())
    assert catalog.part("brick_2x2").allowed_rotations == (0,)
    assert catalog.part("brick_1x4").allowed_rotations == (0, 90)


def test_rotation_swaps_footprint_but_keeps_min_corner(catalog):
    p = Placement(id="a", part_key="brick_1x4", x=2, y=5, z=0, rotation_deg=90, color_id="4")
    assert body_of(p, catalog) == Body(2, 5, 0, 3, 9, 3)


def test_overlap_is_strict_on_half_open_bounds():
    a, b = Body(0, 0, 0, 4, 2, 3), Body(4, 0, 0, 8, 2, 3)
    assert not a.overlaps(b)
    assert a.overlaps(Body(3, 1, 0, 5, 3, 3))
    assert not a.overlaps(Body(0, 0, 3, 4, 2, 6))


def test_side_contact_is_not_a_connection():
    a, b = Body(0, 0, 0, 4, 2, 3), Body(4, 0, 0, 8, 2, 3)
    assert not stud_connected(a, b) and not stud_connected(b, a)


def test_true_stud_connection_requires_vertical_adjacency_and_footprint_overlap():
    lower = Body(0, 0, 0, 4, 2, 3)
    assert stud_connected(Body(3, 1, 3, 5, 3, 6), lower)   # one-cell overlap counts
    assert not stud_connected(Body(4, 0, 3, 6, 2, 6), lower)  # touching edge only
    assert not stud_connected(Body(0, 0, 6, 2, 2, 9), lower)  # gap of one brick
