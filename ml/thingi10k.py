"""Fetch a licensed, reproducible Thingi10K subset for the held-out benchmark.

Source: Hugging Face mirror Thingi10K/Thingi10K (revision v1.5.0), which hosts per-file raw meshes and metadata.
Only CC BY / CC BY-SA / public-domain models that are closed, manifold, single-component solids are eligible.
The selection is deterministic (seed) and written to fixtures/benchmark/thingi10k_subset.json with license,
author, name and source link per model. Mesh files are downloaded to data/ and never committed.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HF_BASE = "https://huggingface.co/datasets/Thingi10K/Thingi10K/resolve/v1.5.0/"
DATA_DIR = ROOT / "data" / "thingi10k"
SUBSET_FILE = ROOT / "fixtures" / "benchmark" / "thingi10k_subset.json"
META_FILES = ["metadata/input_summary.csv", "metadata/contextual_data.csv", "metadata/tag_data.csv", "metadata/geometry_data.csv"]


def license_ok(text: str) -> bool:
    t = (text or "").lower()
    if "noncommercial" in t or "non-commercial" in t or "no derivatives" in t or "nd" == t.strip():
        return False
    return "creative commons" in t or "public domain" in t or "cc0" in t


def _fetch(rel: str, timeout: int = 120) -> bytes:
    return urllib.request.urlopen(HF_BASE + rel, timeout=timeout).read()


def load_metadata(cache_dir: Path = DATA_DIR / "metadata") -> dict[str, list[dict]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for rel in META_FILES:
        local = cache_dir / Path(rel).name
        if not local.exists():
            local.write_bytes(_fetch(rel))
        out[local.stem] = list(csv.DictReader(io.StringIO(local.read_text(encoding="utf-8", errors="replace"))))
    return out


def select_subset(meta: dict[str, list[dict]], n: int = 40, seed: int = 0, min_faces: int = 100, max_faces: int = 200_000) -> list[dict]:
    faces = {r["file_id"]: int(r["num_faces"]) for r in meta["geometry_data"]}
    context = {r["Thing ID"]: r for r in meta["contextual_data"]}
    tags: dict[str, list[str]] = {}
    for r in meta["tag_data"]:
        tags.setdefault(r["Thing ID"], []).append(r["Tag"])
    flags = ["Closed", "Edge manifold", "Vertex manifold", "Single Component", "PWN", "No degenerate faces"]
    eligible = []
    for r in meta["input_summary"]:
        fid = r["ID"]
        if not all(r.get(f) == "TRUE" for f in flags) or not license_ok(r["License"]):
            continue
        if fid not in faces or not (min_faces <= faces[fid] <= max_faces):
            continue
        c = context.get(r["Thing ID"], {})
        eligible.append({
            "file_id": fid, "thing_id": r["Thing ID"], "license": r["License"], "link": r["Link"],
            "name": (c.get("Name") or "").strip(), "author": (c.get("Author") or "").strip(),
            "category": c.get("Category") or "", "tags": sorted(tags.get(r["Thing ID"], []))[:8], "faces": faces[fid],
            "source": f"{HF_BASE}raw_meshes/{fid}.stl",
        })
    eligible.sort(key=lambda e: int(e["file_id"]))
    rng = random.Random(seed)
    # One file per Thing ID so near-duplicates from one upload do not dominate.
    by_thing: dict[str, list[dict]] = {}
    for e in eligible:
        by_thing.setdefault(e["thing_id"], []).append(e)
    things = sorted(by_thing)
    rng.shuffle(things)
    picked = [by_thing[t][0] for t in things[:n]]
    return sorted(picked, key=lambda e: int(e["file_id"]))


def write_subset(n: int = 40, seed: int = 0) -> list[dict]:
    meta = load_metadata()
    subset = select_subset(meta, n=n, seed=seed)
    SUBSET_FILE.parent.mkdir(parents=True, exist_ok=True)
    SUBSET_FILE.write_text(json.dumps({"dataset": "Thingi10K", "revision": "v1.5.0", "mirror": HF_BASE, "seed": seed, "n": n,
                                       "eligibility": "CC BY / CC BY-SA / public domain; closed, manifold, single component, PWN solid; 100..200k faces",
                                       "models": subset}, indent=2) + "\n")
    return subset


def download_subset(mesh_dir: Path = DATA_DIR / "meshes") -> dict:
    subset = json.loads(SUBSET_FILE.read_text())["models"]
    mesh_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"models": []}
    for m in subset:
        local = mesh_dir / f"{m['file_id']}.stl"
        if not local.exists():
            local.write_bytes(_fetch(f"raw_meshes/{m['file_id']}.stl", timeout=300))
        sha = hashlib.sha256(local.read_bytes()).hexdigest()
        manifest["models"].append({**m, "path": str(local.relative_to(ROOT)), "sha256": sha, "bytes": local.stat().st_size})
    (DATA_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def held_out_thing_ids(subset_file: Path = SUBSET_FILE) -> set[str]:
    """Thingiverse thing ids in the frozen held-out subset. Any training pool must exclude these entirely,
    including other files uploaded under the same thing (rotations, variants, remixes of one upload)."""
    return {m["thing_id"] for m in json.loads(subset_file.read_text())["models"]}


def training_pool(meta: dict[str, list[dict]], exclude_things: set[str], max_faces: int = 200_000) -> list[dict]:
    """Eligible Thingi10K files for training with the held-out things removed (split by structure family)."""
    pool = select_subset(meta, n=10**9, seed=0, max_faces=max_faces)
    return [m for m in pool if m["thing_id"] not in exclude_things]
