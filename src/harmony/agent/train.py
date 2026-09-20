"""Train MaskablePPO with the pointer policy."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from harmony.agent.pointer_policy import TaskPointerPolicy, make_pointer_policy_kwargs
from harmony.env.task_assignment_env import TaskAssignmentEnv, load_config


def mask_fn(env) -> np.ndarray:
    return env.unwrapped.action_masks()


def make_env(cfg: dict, backend: str = "kepler", seed: int = 0):
    def _thunk():
        env = TaskAssignmentEnv(config=cfg, backend=backend)
        env.reset(seed=seed)
        env = Monitor(env)
        return ActionMasker(env, mask_fn)

    return _thunk


def train(
    timesteps: int = 50_000,
    backend: str = "kepler",
    config_path: str | None = None,
    save_path: str = "models/pointer_ppo.zip",
    seed: int = 0,
) -> MaskablePPO:
    cfg = load_config(config_path)
    ppo_cfg = cfg.get("ppo", {})
    env = DummyVecEnv([make_env(cfg, backend=backend, seed=seed)])
    eval_env = DummyVecEnv([make_env(cfg, backend=backend, seed=seed + 17)])
    proto = TaskAssignmentEnv(config=cfg, backend="kepler")
    policy_kwargs = make_pointer_policy_kwargs(proto, hidden=int(ppo_cfg.get("hidden", 128)))
    proto.close()

    model = MaskablePPO(
        TaskPointerPolicy,
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=float(ppo_cfg.get("learning_rate", 3e-4)),
        n_steps=int(ppo_cfg.get("n_steps", 1024)),
        batch_size=int(ppo_cfg.get("batch_size", 128)),
        n_epochs=int(ppo_cfg.get("n_epochs", 10)),
        gamma=float(ppo_cfg.get("gamma", 0.995)),
        gae_lambda=float(ppo_cfg.get("gae_lambda", 0.98)),
        clip_range=float(ppo_cfg.get("clip_range", 0.2)),
        ent_coef=float(ppo_cfg.get("ent_coef", 0.02)),
        vf_coef=float(ppo_cfg.get("vf_coef", 0.5)),
        max_grad_norm=float(ppo_cfg.get("max_grad_norm", 0.5)),
        verbose=1,
        seed=seed,
    )
    eval_cb = MaskableEvalCallback(
        eval_env,
        best_model_save_path=str(Path(save_path).parent / "best"),
        n_eval_episodes=4,
        eval_freq=max(ppo_cfg.get("n_steps", 1024) * 4, 2048),
        deterministic=True,
    )
    model.learn(total_timesteps=int(timesteps), callback=eval_cb)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path)
    return model
