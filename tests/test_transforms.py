import pytest

from brickcore.export import ldraw_text
from brickcore.geometry import body_of
from brickcore.schema import Placement
from brickcore.transforms import ldraw_body_bounds, scene_to_body, to_ldraw, to_scene
from tests.conftest import ROOT


@pytest.mark.parametrize("part", ["brick_1x1", "brick_1x2", "brick_1x3", "brick_1x4", "brick_2x2", "brick_2x4"])
@pytest.mark.parametrize("rot", [0, 90])
@pytest.mark.parametrize("pos", [(0, 0, 0), (3, 5, 3), (7, 1, 9)])
def test_ldraw_transform_reproduces_body(catalog, part, rot, pos):
    if rot not in catalog.part(part).allowed_rotations:
        pytest.skip("rotation not allowed")
    p = Placement(id="t", part_key=part, x=pos[0], y=pos[1], z=pos[2], rotation_deg=rot, color_id="4")
    assert ldraw_body_bounds(to_ldraw(p, catalog), p, catalog) == body_of(p, catalog)


@pytest.mark.parametrize("part", ["brick_1x1", "brick_1x4", "brick_2x4"])
@pytest.mark.parametrize("rot", [0, 90])
def test_scene_transform_round_trips(catalog, part, rot):
    if rot not in catalog.part(part).allowed_rotations:
        pytest.skip("rotation not allowed")
    p = Placement(id="t", part_key=part, x=4, y=6, z=6, rotation_deg=rot, color_id="4")
    assert scene_to_body(to_scene(p, catalog), catalog) == body_of(p, catalog)


def test_two_brick_fixture_scene_boxes(catalog, two_brick):
    a, b = (to_scene(p, catalog) for p in two_brick.placements)
    assert (a.cx, a.cy, a.cz, a.sx, a.sy, a.sz) == pytest.approx((16.0, 4.8, -8.0, 32.0, 9.6, 16.0))
    assert (b.cx, b.cy, b.cz, b.sx, b.sy, b.sz) == pytest.approx((8.0, 14.4, -8.0, 16.0, 9.6, 16.0))


def test_two_brick_fixture_ldraw_matches_expected_file(catalog, two_brick):
    expected = (ROOT / "fixtures" / "builds" / "two_brick.expected.ldr").read_text()
    assert ldraw_text(two_brick, catalog) == expected


def test_rotated_part_ldraw_line(catalog):
    p = Placement(id="t", part_key="brick_1x4", x=2, y=0, z=0, rotation_deg=90, color_id="4")
    assert to_ldraw(p, catalog).line() == "1 4 50 -24 40 0 0 1 0 1 0 -1 0 0 3010.dat"
