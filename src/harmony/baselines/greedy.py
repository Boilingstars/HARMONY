"""Greedy and random-valid assignment baselines."""

from __future__ import annotations

import numpy as np

from harmony.env.features import FeatureSpec, access_index


def greedy_action(obs: np.ndarray, masks: np.ndarray, spec: FeatureSpec) -> int:
    """Pick the valid satellite with the longest remaining access; else defer."""
    _, sats, _ = spec.split(np.asarray(obs, dtype=np.float32))
    idx = access_index(spec)
    valid = np.flatnonzero(np.asarray(masks[: spec.n_sats], dtype=bool))
    if valid.size == 0:
        return spec.n_sats
    access = sats[valid, idx]
    return int(valid[int(np.argmax(access))])


def random_masked_action(masks: np.ndarray, rng: np.random.Generator | None = None) -> int:
    rng = rng or np.random.default_rng()
    valid = np.flatnonzero(np.asarray(masks, dtype=bool))
    if valid.size == 0:
        return int(len(masks) - 1)
    return int(rng.choice(valid))
