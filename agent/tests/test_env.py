"""Прежний Kepler-контур attach/defer: маски, частичное покрытие, handoff, defer."""

from __future__ import annotations

import numpy as np

from agent.env.features import FeatureSpec
from agent.env.task_assignment_env import TaskAssignmentEnv, load_config


def _cfg(**overrides) -> dict:
    cfg = load_config()
    cfg.update(overrides)
    return cfg


def test_obs_and_action_shapes():
    env = TaskAssignmentEnv(config=_cfg(n_tasks=6))
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert env.action_space.n == env.n_sats + 1
    masks = env.action_masks()
    assert masks.shape == (env.n_sats + 1,)
    assert masks[-1]
    assert np.isfinite(obs).all()


def test_mask_allows_partial_window():
    env = TaskAssignmentEnv(config=_cfg(n_tasks=4, min_work_slice_s=5.0))
    env.reset(seed=3)
    lat, lon, _ = env.world.lla(0)
    env.current.lat = lat
    env.current.lon = lon
    env.current.duration = 10_000.0
    env.current.duration_original = 10_000.0
    access = env.world.access_remaining(0, lat, lon, env.world.t)
    assert 0.0 < access < env.current.duration
    masks = env.action_masks()
    assert masks[0], "partial access must still allow attach"


def test_mask_blocks_far_target_and_full_queue():
    env = TaskAssignmentEnv(config=_cfg(n_tasks=4, max_queue=1))
    env.reset(seed=4)
    env.current.lat = np.deg2rad(85.0)
    env.current.lon = 0.0
    masks = env.action_masks()
    assert not masks[: env.n_sats].any()
    assert masks[-1]


def test_handoff_reduces_duration_and_returns_to_front():
    env = TaskAssignmentEnv(
        config=_cfg(n_tasks=1, max_step_duration=900.0, episode_orbits=2.0)
    )
    env.reset(seed=8)
    lat, lon, _ = env.world.lla(0)
    task = env.current
    task.lat = lat
    task.lon = lon
    task.duration = 2_000.0
    task.duration_original = 2_000.0
    original_id = task.task_id
    obs, reward, terminated, truncated, info = env.step(0)
    assert info["work_s"] > 30.0
    assert info["n_handoffs"] >= 1
    leftover = None
    if env.current is not None and env.current.task_id == original_id:
        leftover = env.current
    elif env.stack and env.stack[0].task_id == original_id:
        leftover = env.stack[0]
    else:
        for sat in env.sats:
            for queued in sat.queue:
                if queued.task_id == original_id:
                    leftover = queued
    assert leftover is not None
    assert leftover.duration < leftover.duration_original
    assert leftover.duration > 0.0
    assert reward > 0.0


def test_defer_sends_task_to_back():
    env = TaskAssignmentEnv(config=_cfg(n_tasks=5))
    env.reset(seed=1)
    first = env.current.task_id
    second = env.stack[0].task_id
    env.step(env.n_sats)
    assert env.current.task_id == second
    assert env.stack[-1].task_id == first


def test_defer_cycle_advances_time():
    env = TaskAssignmentEnv(config=_cfg(n_tasks=3, max_defer_advance=120.0))
    env.reset(seed=2)
    t0 = env.world.t
    for _ in range(env.n_tasks + 2):
        env.step(env.n_sats)
        if env.world.t > t0:
            break
    assert env.world.t > t0


def test_greedy_episode_runs():
    from agent.legacy_kepler import greedy_action, run_episode

    env = TaskAssignmentEnv(config=_cfg(n_tasks=8, episode_orbits=0.4))
    info = run_episode(
        env,
        lambda obs, masks: greedy_action(obs, masks, env.feat_spec),
        seed=0,
    )
    assert info["steps"] > 0
    assert "n_complete" in info


def test_pointer_forward_shapes():
    import torch

    from agent.policy.pointer_policy import TaskAssignmentNet

    env = TaskAssignmentEnv(config=_cfg(n_tasks=4))
    spec: FeatureSpec = env.feat_spec
    net = TaskAssignmentNet(spec.task_dim, spec.sat_dim, spec.n_sats, spec.global_dim, hidden=32)
    obs, _ = env.reset(seed=0)
    logits, value = net(torch.from_numpy(obs).unsqueeze(0))
    assert logits.shape == (1, env.n_sats + 1)
    assert value.shape == (1, 1)


def test_maskable_ppo_predict_and_short_learn():
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.monitor import Monitor

    from agent.policy.pointer_policy import TaskPointerPolicy, make_pointer_policy_kwargs
    from agent.train import mask_fn

    cfg = _cfg(n_tasks=4, episode_orbits=0.25)
    raw = TaskAssignmentEnv(config=cfg)
    env = ActionMasker(Monitor(raw), mask_fn)
    kwargs = make_pointer_policy_kwargs(raw, hidden=32)
    model = MaskablePPO(
        TaskPointerPolicy,
        env,
        policy_kwargs=kwargs,
        n_steps=8,
        batch_size=4,
        n_epochs=1,
        verbose=0,
    )
    obs, _ = env.reset(seed=0)
    masks = env.action_masks()
    action, _ = model.predict(obs, action_masks=masks, deterministic=True)
    assert 0 <= int(action) <= raw.n_sats
    model.learn(total_timesteps=8)
