"""Прежний контур attach/defer поверх `TaskAssignmentEnv` и Kepler-мира.

Учебным контуром больше не является: обучение идёт на модели организаторов
(`agent/env/ops_env.py`). Модуль остаётся ради демонстрации Basilisk-бэкенда и
старых тестов.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from agent.env.features import FeatureSpec, access_index
from agent.env.task_assignment_env import TaskAssignmentEnv


def greedy_action(obs: np.ndarray, masks: np.ndarray, spec: FeatureSpec) -> int:
    """Взять допустимый КА с самым длинным остатком видимости, иначе отложить."""
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


def run_episode(env: TaskAssignmentEnv, actor: Callable, seed: int | None = None) -> dict:
    obs, info = env.reset(seed=seed)
    total = 0.0
    steps = 0
    terminated = truncated = False
    while not (terminated or truncated):
        masks = env.action_masks()
        action = actor(obs, masks)
        obs, reward, terminated, truncated, info = env.step(action)
        total += float(reward)
        steps += 1
    info = dict(info)
    info["return"] = total
    info["steps"] = steps
    return info


__all__ = ["greedy_action", "random_masked_action", "run_episode"]
