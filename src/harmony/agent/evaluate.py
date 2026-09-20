"""Evaluate a trained policy or a heuristic baseline."""

from __future__ import annotations

from typing import Callable

import numpy as np

from harmony.baselines.greedy import greedy_action, random_masked_action
from harmony.env.task_assignment_env import TaskAssignmentEnv, load_config


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


def evaluate(
    n_episodes: int = 8,
    backend: str = "kepler",
    config_path: str | None = None,
    model_path: str | None = None,
    policy: str = "greedy",
    seed: int = 0,
) -> dict:
    cfg = load_config(config_path)
    env = TaskAssignmentEnv(config=cfg, backend=backend)
    spec = env.feat_spec
    rng = np.random.default_rng(seed)

    if model_path:
        from sb3_contrib import MaskablePPO

        model = MaskablePPO.load(model_path)

        def actor(obs, masks):
            action, _ = model.predict(obs, action_masks=masks, deterministic=True)
            return int(action)

    elif policy == "random":

        def actor(obs, masks):
            return random_masked_action(masks, rng)

    else:

        def actor(obs, masks):
            return greedy_action(obs, masks, spec)

    rows = [run_episode(env, actor, seed=seed + i) for i in range(n_episodes)]
    summary = {
        "n_episodes": n_episodes,
        "mean_return": float(np.mean([r["return"] for r in rows])),
        "mean_complete": float(np.mean([r.get("n_complete", 0) for r in rows])),
        "mean_handoffs": float(np.mean([r.get("n_handoffs", 0) for r in rows])),
        "mean_work_s": float(np.mean([r.get("work_s", 0.0) for r in rows])),
        "backend": backend,
        "policy": "ppo" if model_path else policy,
    }
    return summary
