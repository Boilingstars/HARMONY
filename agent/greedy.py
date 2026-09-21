"""Жадный диспетчер на том же наблюдении, что и сеть.

Правило простое: самые дорогие срочные задания разбираются первыми, свободные
аппараты уходят на калибровку, когда срок допуска подходит к концу. Это и есть
планка, которую RL должен обойти.
"""

from __future__ import annotations

import numpy as np

# Индексы признаков наблюдения (см. agent/env/ops_env.py)
_JOB_PRIO = slice(0, 3)
_JOB_VALUE = 3
_JOB_REMAINING = 5
_JOB_DEADLINE = 7
_JOB_DOWNLINK = 8
_SAT_SOC = 0
_SAT_CAL_AGE = 3
_EDGE_LAST_EXECUTOR = 4

CALIBRATE_AT = 0.85  # доля израсходованного срока допуска


def greedy_action(
    obs: dict[str, np.ndarray],
    downlink_limit: int = 2,
    calibrate_at: float = CALIBRATE_AT,
) -> np.ndarray:
    """Матрица наблюдения -> вектор действий длины N."""
    mask = np.asarray(obs["mask"], dtype=bool)
    jobs = np.asarray(obs["jobs"], dtype=np.float32)
    sats = np.asarray(obs["sats"], dtype=np.float32)
    edges = np.asarray(obs["edges"], dtype=np.float32)
    n, width = mask.shape
    k = width - 2
    idle, calibrate = k, k + 1

    action = np.full(n, idle, dtype=np.int64)
    priority = jobs[:, _JOB_PRIO].argmax(axis=1) + 1
    has_job = jobs[:, _JOB_PRIO].sum(axis=1) > 0
    remaining = np.maximum(jobs[:, _JOB_REMAINING], 1e-3)
    rate = jobs[:, _JOB_VALUE] / remaining

    order = sorted(
        (slot for slot in range(k) if has_job[slot] and mask[:, slot].any()),
        key=lambda s: (-priority[s], -rate[s], jobs[s, _JOB_DEADLINE]),
    )

    taken = np.zeros(n, dtype=bool)
    downlinks = 0
    for slot in order:
        if jobs[slot, _JOB_DOWNLINK] > 0.5 and downlinks >= downlink_limit:
            continue
        free = np.flatnonzero(mask[:, slot] & ~taken)
        if free.size == 0:
            continue
        # Преемственность важнее запаса энергии: переброс задания без выигрыша
        # только сбивает прогресс и штрафуется наградой.
        pick = int(free[np.lexsort((-sats[free, _SAT_SOC], -edges[free, slot, _EDGE_LAST_EXECUTOR]))[0]])
        action[pick] = slot
        taken[pick] = True
        downlinks += int(jobs[slot, _JOB_DOWNLINK] > 0.5)

    for i in range(n):
        if taken[i] or not mask[i, calibrate]:
            continue
        if sats[i, _SAT_CAL_AGE] >= calibrate_at:
            action[i] = calibrate
    return action


def random_masked_action(
    obs: dict[str, np.ndarray], rng: np.random.Generator | None = None
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    mask = np.asarray(obs["mask"], dtype=bool)
    n, width = mask.shape
    action = np.full(n, width - 2, dtype=np.int64)
    for i in range(n):
        valid = np.flatnonzero(mask[i])
        if valid.size:
            action[i] = int(rng.choice(valid))
    return action


__all__ = ["greedy_action", "random_masked_action", "CALIBRATE_AT"]
