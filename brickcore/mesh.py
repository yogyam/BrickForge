"""Mesh input limits, normalization to the canonical grid, and brick-resolution voxelization."""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import trimesh

from .catalog import Catalog
from .schema import BRICK_HEIGHT_PLATES, Bounds

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_TRIANGLES = 200_000
SUPPORTED_SUFFIXES = {".stl", ".glb"}


class MeshError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


UP_AXES = {
    "+x": np.array([1, 0, 0]), "-x": np.array([-1, 0, 0]),
    "+y": np.array([0, 1, 0]), "-y": np.array([0, -1, 0]),
    "+z": np.array([0, 0, 1]), "-z": np.array([0, 0, -1]),
}


def _rotation_taking(src_up: np.ndarray) -> np.ndarray:
    """Proper rotation matrix mapping src_up to +z (3x3)."""
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(src_up, z)
    c = float(np.dot(src_up, z))
    if np.allclose(v, 0):
        if c > 0:
            return np.eye(3)
        return np.diag([1.0, -1.0, -1.0])  # 180 degrees about x
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))


def _yaw(deg: int) -> np.ndarray:
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def load_mesh(path: Path) -> tuple[trimesh.Trimesh, dict]:
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise MeshError("unsupported_input", f"only binary STL and single-file GLB are supported, got {path.suffix}")
    data = path.read_bytes()
    if len(data) > MAX_UPLOAD_BYTES:
        raise MeshError("input_limit_exceeded", f"{len(data)} bytes exceeds {MAX_UPLOAD_BYTES}")
    sha = hashlib.sha256(data).hexdigest()
    try:
        loaded = trimesh.load(path, force="scene", process=False)
    except Exception as e:  # noqa: BLE001
        raise MeshError("invalid_mesh", f"parse failed: {e}") from e
    geoms = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
    if not geoms:
        raise MeshError("invalid_mesh", "no triangle geometry found")
    if len(loaded.graph.nodes_geometry) > 256:
        raise MeshError("input_limit_exceeded", "too many scene nodes")
    mesh = loaded.to_mesh()  # flattens node transforms into one mesh
    mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=True)
    if len(mesh.faces) > MAX_TRIANGLES:
        raise MeshError("input_limit_exceeded", f"{len(mesh.faces)} triangles exceeds {MAX_TRIANGLES}")
    if not np.isfinite(mesh.vertices).all():
        raise MeshError("invalid_mesh", "non-finite vertex coordinates")
    if np.any(mesh.extents <= 0):
        raise MeshError("invalid_mesh", "mesh has zero extent on some axis")
    if not mesh.is_watertight:
        raise MeshError("non_watertight_mesh", "mesh is not watertight; R1 requires a closed solid and will not fill unknown interiors")
    if mesh.volume < 0:
        mesh.invert()
    meta = {"source_sha256": sha, "source_bytes": len(data), "triangles": int(len(mesh.faces)), "source_extents": mesh.extents.tolist()}
    return mesh, meta


@dataclass
class Normalization:
    up: str
    yaw_deg: int
    scale: float
    rotation: list[list[float]]
    translation_mm: list[float]
    target_mm: list[float]
    fitted_mm: list[float]

    def to_dict(self) -> dict:
        return asdict(self)


def normalize(mesh: trimesh.Trimesh, bounds: Bounds, catalog: Catalog, up: str = "+z", yaw_deg: int = 0) -> tuple[trimesh.Trimesh, Normalization]:
    """Rotate the chosen up axis to +z, apply yaw, center horizontally, floor at z=0, and scale uniformly
    to fit the physical target box (unequal horizontal/vertical unit lengths respected)."""
    if up not in UP_AXES:
        raise MeshError("unsupported_input", f"up must be one of {sorted(UP_AXES)}")
    if yaw_deg % 90 != 0:
        raise MeshError("unsupported_input", "yaw must be a multiple of 90")
    R = _yaw(yaw_deg % 360) @ _rotation_taking(UP_AXES[up].astype(float))
    m = mesh.copy()
    m.apply_transform(np.block([[R, np.zeros((3, 1))], [np.zeros((1, 3)), 1.0]]))
    target = np.array([bounds.x_studs * catalog.stud_pitch_mm, bounds.y_studs * catalog.stud_pitch_mm, bounds.z_plates * catalog.plate_height_mm])
    scale = float(np.min(target / m.extents))
    m.apply_scale(scale)
    lo, hi = m.bounds
    center_xy = (lo[:2] + hi[:2]) / 2
    t = np.array([target[0] / 2 - center_xy[0], target[1] / 2 - center_xy[1], -lo[2]])
    m.apply_translation(t)
    norm = Normalization(up, yaw_deg, scale, R.tolist(), t.tolist(), target.tolist(), m.extents.tolist())
    return m, norm


VOXELIZE_RNG_SEED = 1


def voxelize(mesh: trimesh.Trimesh, bounds: Bounds, catalog: Catalog) -> np.ndarray:
    """Occupancy [x_studs, y_studs, layers] sampled at cell centres of 1 stud x 1 stud x 1 brick cells."""
    nx, ny, nz = bounds.x_studs, bounds.y_studs, bounds.layers
    sp, bh = catalog.stud_pitch_mm, catalog.brick_height_mm
    xs = (np.arange(nx) + 0.5) * sp
    ys = (np.arange(ny) + 0.5) * sp
    zs = (np.arange(nz) + 0.5) * bh
    pts = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
    # trimesh's parity test re-checks cell centres whose forward and backward rays disagree using a direction drawn
    # from numpy's *global* RNG, so meshes with a surface through a cell centre (Thingi10K 225935 flips 288-290
    # cells) would voxelize differently from process to process. Pin that draw and restore the caller's RNG state.
    # The seed is the one that reproduces the frozen benchmark targets recorded in docs/benchmarks/baseline_v1.
    state = np.random.get_state()
    try:
        np.random.seed(VOXELIZE_RNG_SEED)
        inside = mesh.contains(pts)
    finally:
        np.random.set_state(state)
    return inside.reshape(nx, ny, nz)


def occupancy_cells(occ: np.ndarray) -> list[list[int]]:
    return [[int(i), int(j), int(k)] for i, j, k in np.argwhere(occ)]


def occupancy_from_cells(cells: list[list[int]], bounds: Bounds) -> np.ndarray:
    occ = np.zeros((bounds.x_studs, bounds.y_studs, bounds.layers), dtype=bool)
    for i, j, k in cells:
        occ[i, j, k] = True
    return occ


__all__ = ["load_mesh", "normalize", "voxelize", "MeshError", "Normalization", "occupancy_cells", "occupancy_from_cells", "BRICK_HEIGHT_PLATES"]
