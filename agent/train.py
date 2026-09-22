"""Обучение MaskablePPO на модели кейса.

Один PPO-таймстеп = один тик 5 минут на всю группировку, поэтому эпизод P02 —
это 288 таймстепов, а не 48 x 288.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from agent.env.ops_env import N_MAX, EventConfig, OpsEnv
from agent.evaluate import _load_ppo
from agent.policy.ops_policy import OpsAttentionPolicy, count_parameters
from agent.plots import plot_run
from agent.telemetry import (
    OpsMetricsCallback,
    configure_run_logger,
    dump_config,
    last_logged_timesteps,
    make_run_dir,
)
from sim.ops import load, resolve_scenario

_AGENT_DIR = Path(__file__).resolve().parent
MODELS_DIR = _AGENT_DIR / "models"

# `full` — размеры из плана. `light` — та же архитектура в меньшем масштабе,
# чтобы обучение шло на обычном 4-ядерном CPU: слои и маски те же, веса уже.
PROFILES = {
    "full": dict(d=128, heads=4, blocks=2, ff=512, d_pointer=64),
    "light": dict(d=64, heads=4, blocks=1, ff=128, d_pointer=32),
}


# Веса и графики — каждые 20 минут по настенным часам.
CHECKPOINT_EVERY_S = 20 * 60
PLOT_EVERY_S = 20 * 60


class RunArtifactsCallback(BaseCallback):
    """Чекпоинт весов и перерисовка графиков по ходу обучения."""

    def __init__(self, run_dir: Path | str) -> None:
        super().__init__()
        self.run_dir = Path(run_dir)
        self._next_checkpoint = 0.0
        self._next_plot = 0.0

    def _on_training_start(self) -> None:
        now = time.monotonic()
        self._next_checkpoint = now + CHECKPOINT_EVERY_S
        self._next_plot = now + PLOT_EVERY_S
        (self.run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        now = time.monotonic()
        if now >= self._next_plot:
            self._write_plots()
            while self._next_plot <= now:
                self._next_plot += PLOT_EVERY_S
        if now >= self._next_checkpoint:
            self._save_checkpoint()
            while self._next_checkpoint <= now:
                self._next_checkpoint += CHECKPOINT_EVERY_S
        return True

    def _write_plots(self) -> None:
        progress = self.run_dir / "progress.csv"
        if not progress.exists() or progress.stat().st_size == 0:
            return
        made = plot_run(self.run_dir)
        print(f"Графики обновлены: {len(made)} файлов в {self.run_dir / 'plots'}", flush=True)

    def _save_checkpoint(self) -> None:
        path = self.run_dir / "checkpoints" / f"ckpt_{self.num_timesteps}.zip"
        self.model.save(str(path))
        print(f"Чекпоинт весов: {path}", flush=True)


def _split_value_learning_rate(model: MaskablePPO, policy_lr: float, value_lr: float) -> None:
    """Две группы Adam. SB3 каждый rollout ставит всем группам шаг политики."""
    net = model.policy.net
    value_params = [
        p
        for module in (net.value_head, net.pool_sat, net.pool_job, net.pool_plan)
        for p in module.parameters()
    ]
    value_ids = {id(p) for p in value_params}
    policy_params = [p for p in net.parameters() if id(p) not in value_ids]
    model.policy.optimizer = model.policy.optimizer_class(
        [
            {"params": policy_params, "lr": policy_lr},
            {"params": value_params, "lr": value_lr},
        ],
        **model.policy.optimizer_kwargs,
    )
    scale = value_lr / policy_lr
    original = model._update_learning_rate

    def _update(optimizers) -> None:
        original(optimizers)
        optimizer = model.policy.optimizer
        base = optimizer.param_groups[0]["lr"]
        optimizer.param_groups[1]["lr"] = base * scale

    def _excluded_save_params() -> list[str]:
        # Замыкание держит модель с открытым train.log. cloudpickle на Python 3.14
        # из-за этого роняет процесс на чекпоинте, поэтому в zip его не кладём.
        names = type(model)._excluded_save_params(model)
        return list(names) + ["_update_learning_rate", "_excluded_save_params"]

    model._update_learning_rate = _update
    model._excluded_save_params = _excluded_save_params


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
    load_path: str | Path | None = None,
    resume: bool = False,
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
    # Политика остаётся на 3e-5: на 1e-4 одна эпоха уже давала KL около 1–2.
    # Голова ценности и её пулы — на 1e-4, иначе explained_variance стоит на 0.02
    # и преимущество остаётся шумом. Общий энкодер сидит в группе политики,
    # чтобы крупный шаг критика не двигал логиты.
    learning_rate = 3e-5
    value_learning_rate = 1e-4
    ent_coef = 0.01
    target_kl = 1.5
    model = MaskablePPO(
        OpsAttentionPolicy,
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=128,
        n_epochs=4,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=target_kl,
        verbose=1,
        seed=seed,
    )
    _split_value_learning_rate(model, learning_rate, value_learning_rate)
    if load_path is not None:
        loaded = _load_ppo(load_path)
        model.policy.load_state_dict(loaded.policy.state_dict())
        print(f"Стартовые веса: {load_path}", flush=True)

    run_dir = Path(run_dir) if run_dir is not None else make_run_dir(spec["meta"]["id"], goal)
    started = last_logged_timesteps(run_dir) if resume else 0
    if resume:
        model.num_timesteps = started
        remaining = max(int(timesteps) - started, 0)
        print(f"Продолжение {run_dir}: уже {started} шагов, осталось {remaining}", flush=True)
    else:
        remaining = int(timesteps)
    model.set_logger(configure_run_logger(run_dir, tensorboard=tensorboard, resume=resume))
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
            "resumed_from": started if resume else None,
            "n_envs": n_envs,
            "seed": seed,
            "max_steps": max_steps,
            "events": {} if no_events else events.__dict__,
            "profile": profile,
            "load": str(load_path) if load_path is not None else None,
            "policy": policy_kwargs,
            "policy_parameters": count_parameters(model.policy.net),
            "ppo": {
                "learning_rate": learning_rate,
                "value_learning_rate": value_learning_rate,
                "n_steps": n_steps,
                "batch_size": 128,
                "n_epochs": 4,
                "gamma": 0.995,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": ent_coef,
                "target_kl": target_kl,
            },
        },
    )

    model.learn(
        total_timesteps=remaining,
        callback=[OpsMetricsCallback(), RunArtifactsCallback(run_dir)],
        progress_bar=True,
        reset_num_timesteps=not resume,
    )

    save_path = Path(save_path) if save_path is not None else MODELS_DIR / f"ops_{goal}.zip"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    (run_dir / "model.txt").write_text(str(save_path), encoding="utf-8")
    print(f"Модель: {save_path}\nЖурнал обучения: {run_dir}")
    for path in plot_run(run_dir):
        print(f"График: {path}")
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
    parser.add_argument("--load", default=None, help="zip весов политики для старта")
    parser.add_argument("--run-dir", default=None, help="каталог журнала; с --resume дописывается")
    parser.add_argument("--resume", action="store_true", help="продолжить progress.csv и счётчик шагов")
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
        run_dir=args.run_dir,
        subproc=args.subproc,
        profile=args.profile,
        load_path=args.load,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
