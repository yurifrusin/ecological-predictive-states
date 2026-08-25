"""Deterministic, namespaced random-seed utilities."""

import hashlib

import numpy as np


def derive_seed(root_seed: int, namespace: str) -> int:
    payload = f"epsbench-v0:{root_seed}:{namespace}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def rng_for(root_seed: int, namespace: str) -> np.random.Generator:
    return np.random.default_rng(derive_seed(root_seed, namespace))
