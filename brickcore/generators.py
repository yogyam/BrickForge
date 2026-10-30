"""Generator registry. Every generator implements
    generate(target, catalog, constraints, seed, cancellation, build_id) -> CandidateResult
Validators, planner, stability and exports never change with the generator."""
from __future__ import annotations

from typing import Callable

from . import generate as _greedy

REGISTRY: dict[str, Callable] = {_greedy.GENERATOR_NAME: _greedy.generate}


def register(name: str, fn: Callable) -> None:
    REGISTRY[name] = fn


def get(name: str) -> Callable:
    if name not in REGISTRY and name.startswith("sft"):
        import importlib
        importlib.import_module("ml.sft_generator")  # registers sft_v1 on import
    if name not in REGISTRY:
        raise KeyError(f"unknown generator {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name]
