"""Обучение MaskablePPO на модели кейса.

Один PPO-таймстеп = один тик 5 минут на всю группировку, поэтому эпизод P02 —
это 288 таймстепов, а не 48 x 288.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from agent.env.ops_env import N_MAX, EventConfig, OpsEnv
from agent.policy.ops_policy import OpsAttentionPolicy, count_parameters
from agent.telemetry import OpsMetricsCallback, configure_run_logger, dump_config, make_run_dir
from sim.ops import load, resolve_scenario

_AGENT_DIR = Path(__file__).resolve().parent
MODELS_DIR = _AGENT_DIR / "models"

# `full` — размеры из плана. `light` — та же архитектура в меньшем масштабе,
# чтобы обучение шло на обычном 4-ядерном CPU: слои и маски те же, веса уже.
PROFILES = {
    "full": dict(d=128, heads=4, blocks=2, ff=512, d_pointer=64),
    "light": dict(d=64, heads=4, blocks=1, ff=128, d_pointer=32),
}


def mask_fn(env) -> np.ndarray:
    return env.unwrapped.action_masks()


def make_env_fn(
    scenario: dict,
    goal: str,
    events: EventConfig,
    top_k: int,
    max_steps: int | None,
    seed: int,
):
    def _thunk():
        env = OpsEnv(
            scenario=scenario,
            goal=goal,
            events=events,
            top_k=top_k,
            max_steps=max_steps,
            seed=seed,
        )
        return ActionMasker(Monitor(env), mask_fn)

    return _thunk


def train(
    scenario: str | Path = "p02",
    goal: str = "priority",
    timesteps: int = 200_000,
    n_envs: int = 4,
    top_k: int = 32,
    max_steps: int | None = None,
    seed: int = 0,
    tensorboard: bool = False,
    no_events: bool = False,
    save_path: str | Path | None = None,
    run_dir: str | Path | None = None,
    subproc: bool = False,
    profile: str = "full",
) -> MaskablePPO:
    path = resolve_scenario(scenario)
    spec = load(path)
    events = EventConfig.off() if no_events else EventConfig()

    thunks = [
        make_env_fn(spec, goal, events, top_k, max_steps, seed + i) for i in range(n_envs)
    ]
    vec_cls = SubprocVecEnv if (subproc and n_envs > 1) else DummyVecEnv
    env = vec_cls(thunks)

    episode_len = max_steps or spec["time"]["steps"]
    n_steps = episode_len  # один rollout = один эпизод на каждом воркере

    policy_kwargs = dict(PROFILES[profile])
    model = MaskablePPO(
        OpsAttentionPolicy,
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=n_steps,
        batch_size=128,
        n_epochs=4,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        seed=seed,
    )

    run_dir = Path(run_dir) if run_dir is not None else make_run_dir(spec["meta"]["id"], goal)
    model.set_logger(configure_run_logger(run_dir, tensorboard=tensorboard))
    dump_config(
        run_dir,
        {
            "scenario": str(path),
            "scenario_id": spec["meta"]["id"],
            "n_satellites": len(spec["satellites"]),
            "steps": spec["time"]["steps"],
            "goal": goal,
            "top_k": top_k,
            "n_max": N_MAX,
            "timesteps": timesteps,
            "n_envs": n_envs,
            "seed": seed,
            "max_steps": max_steps,
            "events": {} if no_events else events.__dict__,
            "profile": profile,
            "policy": policy_kwargs,
            "policy_parameters": count_parameters(model.policy.net),
            "ppo": {
                "learning_rate": 3e-4,
                "n_steps": n_steps,
                "batch_size": 128,
                "n_epochs": 4,
                "gamma": 0.995,
                "gae_lambda": 0.95,
                "ent_coef": 0.01,
            },
        },
    )

    model.learn(total_timesteps=int(timesteps), callback=OpsMetricsCallback())

    save_path = Path(save_path) if save_path is not None else MODELS_DIR / f"ops_{goal}.zip"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    (run_dir / "model.txt").write_text(str(save_path), encoding="utf-8")
    print(f"Модель: {save_path}\nЖурнал обучения: {run_dir}")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучение диспетчера группировки")
    parser.add_argument("--scenario", default="p02", help="p01..p04 или путь к JSON")
    parser.add_argument("--goal", default="priority", choices=["priority", "revenue"])
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--no-events", action="store_true")
    parser.add_argument("--subproc", action="store_true")
    parser.add_argument("--profile", default="full", choices=sorted(PROFILES))
    parser.add_argument("--save", default=None)
    args = parser.parse_args()
    train(
        scenario=args.scenario,
        goal=args.goal,
        timesteps=args.timesteps,
        n_envs=args.n_envs,
        top_k=args.top_k,
        max_steps=args.max_steps,
        seed=args.seed,
        tensorboard=args.tensorboard,
        no_events=args.no_events,
        save_path=args.save,
        subproc=args.subproc,
        profile=args.profile,
    )


if __name__ == "__main__":
    main()
