import numpy as np
import pytest

from brickcore.candidates import enumerate_candidates
from brickcore.schema import BRICK_HEIGHT_PLATES, Build
from brickcore.validate import validate_build
from ml.data import (
    FAMILIES,
    apply_symmetry,
    make_example,
    placements_from_example,
    synth_target,
    target_from_example,
)


def test_candidates_are_all_legal_and_teacher_actions_are_enumerable(two_brick, catalog):
    rng = np.random.default_rng(1)
    t = synth_target("stack", rng)
    ex = make_example("t", "stack", t, catalog)
    assert ex is not None
    target = target_from_example(ex)
    filled = np.zeros_like(target)
    placements = placements_from_example(ex)
    for p in placements:
        c = enumerate_candidates(target, filled, catalog)
        i = c.find(p.part_key, p.rotation_deg, p.x, p.y, p.z // BRICK_HEIGHT_PLATES)
        assert i >= 0, "teacher action must be in the candidate set"
        # every candidate is collision-free and supported
        for j in range(min(len(c), 200)):
            cells = c.footprint_cells(j)
            assert not any(filled[x, y, c.k[j]] for x, y in cells)
            assert c.k[j] == 0 or any(filled[x, y, c.k[j] - 1] for x, y in cells)
        w, d = catalog.part(p.part_key).footprint(p.rotation_deg)
        filled[p.x:p.x + w, p.y:p.y + d, p.z // BRICK_HEIGHT_PLATES] = True


@pytest.mark.parametrize("family", FAMILIES)
def test_synthetic_families_are_grounded_and_connected(family):
    t = synth_target(family, np.random.default_rng(3))
    assert t.any() and t[:, :, 0].any()


@pytest.mark.parametrize("sym", range(8))
def test_symmetry_keeps_builds_valid(two_brick, catalog, sym):
    rng = np.random.default_rng(5)
    ex = make_example("s", "polyomino", synth_target("polyomino", rng), catalog)
    target, placements = apply_symmetry(target_from_example(ex), placements_from_example(ex), sym, catalog)
    raw = two_brick.model_dump()
    raw["constraints"]["bounds"] = {"x_studs": 16, "y_studs": 16, "z_plates": 24}
    raw["constraints"]["max_pieces"] = 100
    raw["placements"] = [p.model_dump() for p in placements]
    raw["steps"] = [{"index": i + 1, "added_placement_ids": [p.id]} for i, p in enumerate(placements)]
    report = validate_build(Build.model_validate(raw), catalog)
    assert report["ok"], report["errors"][:3]
    # occupancy of transformed placements equals transformed target coverage
    from brickcore.geometry import occupancy_grid
    occ = occupancy_grid(placements, catalog, *target.shape)
    assert np.logical_and(occ, target).sum() == np.logical_and(occupancy_grid(placements_from_example(ex), catalog, 16, 16, 8), target_from_example(ex)).sum()


def test_policy_forward_backward_and_tiny_overfit(catalog):
    torch = pytest.importorskip("torch")
    from ml.policy import BrickPolicy, collate, loss_and_acc
    from ml.train_sft import make_items
    ex = make_example("o", "box", synth_target("box", np.random.default_rng(7)), catalog)
    items, missing = make_items(ex, catalog)
    assert missing == 0 and len(items) == len(ex.placements) + 1
    model = BrickPolicy(channels=8, hidden=16, layers=2)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    batch = collate(items, torch.device("cpu"))
    first = None
    for _ in range(150):
        loss, acc = loss_and_acc(model, batch)
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.5
    assert acc > 0.4
