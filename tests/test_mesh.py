import numpy as np
import pytest

from brickcore.mesh import MeshError, load_mesh, normalize, voxelize
from brickcore.schema import Bounds

B = Bounds(x_studs=16, y_studs=16, z_plates=24)


def test_fixture_meshes_load(fixture_meshes):
    for name in ["cube", "bridge", "sphere", "l_shape"]:
        mesh, meta = load_mesh(fixture_meshes / f"{name}.stl")
        assert mesh.is_watertight and len(meta["source_sha256"]) == 64 and meta["triangles"] > 0


def test_open_mesh_is_rejected_with_specific_code(fixture_meshes):
    with pytest.raises(MeshError) as e:
        load_mesh(fixture_meshes / "open_box_invalid.stl")
    assert e.value.code == "non_watertight_mesh"


def test_unsupported_suffix(tmp_path):
    p = tmp_path / "x.obj"
    p.write_text("v 0 0 0\n")
    with pytest.raises(MeshError) as e:
        load_mesh(p)
    assert e.value.code == "unsupported_input"


def test_normalize_fits_box_and_floors_at_zero(fixture_meshes, catalog):
    mesh, _ = load_mesh(fixture_meshes / "cube.stl")
    m, norm = normalize(mesh, Bounds(x_studs=8, y_studs=8, z_plates=12), catalog)
    assert np.allclose(m.bounds[0][2], 0.0)
    assert np.allclose(m.extents, [64, 64, 38.4])
    assert np.allclose((m.bounds[0][:2] + m.bounds[1][:2]) / 2, [32, 32])


def test_up_axis_choice_changes_result_consistently(fixture_meshes, catalog):
    mesh, _ = load_mesh(fixture_meshes / "tower.stl")  # tall along z in the file
    z_up, _ = normalize(mesh, B, catalog, up="+z")
    y_up, _ = normalize(mesh, B, catalog, up="+y")  # treat y as up: the tower lies down
    assert np.argmax(z_up.extents) == 2
    assert np.argmax(y_up.extents) != 2


def test_voxelize_cube_exactly(fixture_meshes, catalog):
    mesh, _ = load_mesh(fixture_meshes / "cube.stl")
    bounds = Bounds(x_studs=8, y_studs=8, z_plates=12)
    m, _ = normalize(mesh, bounds, catalog)
    occ = voxelize(m, bounds, catalog)
    assert occ.shape == (8, 8, 4) and occ.all()


def test_voxelize_respects_unequal_units(fixture_meshes, catalog):
    # slab: 96 x 64 x 19.2 mm. Fit to 16x16x8 scales by 4/3 -> 25.6 mm tall. Layer centres sit at 4.8, 14.4 and
    # 24.0 mm, all inside, so 3 occupied layers; the 4th centre at 33.6 mm is outside.
    mesh, _ = load_mesh(fixture_meshes / "slab.stl")
    m, _ = normalize(mesh, B, catalog)
    occ = voxelize(m, B, catalog)
    assert occ.sum(axis=(0, 1)).tolist()[:4] == [160, 160, 160, 0]
