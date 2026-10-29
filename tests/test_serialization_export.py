from brickcore.export import bom_csv, bom_rows, ldraw_text
from brickcore.schema import Build
from tests.conftest import ROOT


def test_json_round_trip_is_byte_identical(two_brick):
    text = two_brick.to_json()
    assert Build.from_json(text).to_json() == text


def test_fixture_file_parses_and_reserialises_equivalently(two_brick):
    raw = (ROOT / "fixtures" / "builds" / "two_brick.json").read_text()
    assert Build.from_json(raw).model_dump() == two_brick.model_dump()


def test_bom_counts_equal_placements(two_brick, catalog):
    rows = bom_rows(two_brick)
    assert rows == [("brick_2x2", "4", 1), ("brick_2x4", "4", 1)]
    assert sum(n for _, _, n in rows) == len(two_brick.placements)
    csv_text = bom_csv(two_brick, catalog)
    assert "brick_2x4,3001.dat,4,red,1" in csv_text


def test_ldraw_has_one_part_line_and_one_step_per_placement(two_brick, catalog):
    lines = ldraw_text(two_brick, catalog).splitlines()
    assert sum(line.startswith("1 ") for line in lines) == len(two_brick.placements)
    assert sum(line == "0 STEP" for line in lines) == len(two_brick.steps)
