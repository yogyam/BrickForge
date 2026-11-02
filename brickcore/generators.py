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
    if name not in REGISTRY:
        import importlib
        for prefix, module in (("sft", "ml.sft_generator"), ("rl", "ml.rl_generator")):
            if name.startswith(prefix):
                importlib.import_module(module)  # registers its generators on import
    if name not in REGISTRY:
        raise KeyError(f"unknown generator {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name]
