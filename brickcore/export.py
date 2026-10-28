"""BOM CSV and LDraw exports derived strictly from stored placements."""
from __future__ import annotations

import csv
import io
from collections import Counter

from .catalog import Catalog
from .schema import Build
from .transforms import to_ldraw


def bom_rows(build: Build) -> list[tuple[str, str, int]]:
    counts = Counter((p.part_key, p.color_id) for p in build.placements)
    return sorted((k, c, n) for (k, c), n in counts.items())


def bom_csv(build: Build, catalog: Catalog) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["part_key", "ldraw_file", "color_id", "color_name", "count"])
    for part_key, color_id, n in bom_rows(build):
        w.writerow([part_key, catalog.part(part_key).ldraw_file, color_id, catalog.colors[color_id]["name"], n])
    return buf.getvalue()


def ldraw_text(build: Build, catalog: Catalog) -> str:
    by_id = {p.id: p for p in build.placements}
    lines = [
        f"0 FILE {build.build_id}.ldr",
        f"0 {build.build_id}",
        f"0 Name: {build.build_id}.ldr",
        "0 Author: BrickForge",
        "0 !LDRAW_ORG Unofficial_Model",
        f"0 !BRICKFORGE catalog={build.catalog_version} generator={build.generator.name}/{build.generator.version} seed={build.generator.seed}",
    ]
    if build.steps:
        for step in build.steps:
            for pid in step.added_placement_ids:
                lines.append(to_ldraw(by_id[pid], catalog).line())
            lines.append("0 STEP")
    else:
        lines.append("0 !BRICKFORGE no checked assembly steps; parts listed in storage order")
        for p in build.placements:
            lines.append(to_ldraw(p, catalog).line())
    return "\n".join(lines) + "\n"
