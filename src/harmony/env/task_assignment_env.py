"""Gymnasium environment: sequential attach / defer over a constellation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Literal

import gymnasium as gym
import numpy as np
import yaml

from harmony.env.features import (
    STATUS_BUSY,
    STATUS_IDLE,
    STATUS_SLEEP,
    FeatureSpec,
    encode_sat,
    encode_task,
    pack_obs,
)
from harmony.sim.coverage import footprint_radius
from harmony.sim.kepler import KeplerWorld, kepler_from_config


def _default_config_path() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "configs" / "default.yaml"
        if candidate.exists():
            return candidate
    return Path.cwd() / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path is not None else _default_config_path()
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@dataclass
class GroundTask:
    task_id: int
    lat: float
    lon: float
    duration: float
    duration_original: float
    power_need: float
    capability: np.ndarray
    attempts: int = 0


@dataclass
class SatRuntime:
    sat_id: int
    capability: np.ndarray
    queue: Deque[GroundTask] = field(default_factory=deque)


class TaskAssignmentEnv(gym.Env):
    """
    Discrete(N+1): actions 0..N-1 attach the current task to that satellite,
    action N defers it to the back of the stack.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: dict | None = None,
        config_path: str | Path | None = None,
        backend: Literal["kepler", "basilisk"] = "kepler",
        world=None,
    ) -> None:
        super().__init__()
        self.cfg = config if config is not None else load_config(config_path)
        self.backend_name = backend
        self.n_sats = int(self.cfg["n_sats"])
        self.n_tasks = int(self.cfg["n_tasks"])
        self.max_queue = int(self.cfg["max_queue"])
        self.max_attempts = int(self.cfg["max_attempts"])
        self.sim_dt = float(self.cfg["sim_dt"])
        self.max_step_duration = float(self.cfg["max_step_duration"])
        self.max_defer_advance = float(self.cfg["max_defer_advance"])
        self.min_work_slice_s = float(self.cfg["min_work_slice_s"])
        self.capability_dim = int(self.cfg.get("capability_dim", 4))
        self.long_task_frac = float(self.cfg.get("long_task_frac", 0.35))

        if world is not None:
            self.world = world
        elif backend == "basilisk":
            from harmony.sim.basilisk_world import basilisk_from_config

            self.world = basilisk_from_config(self.cfg)
        else:
            self.world = kepler_from_config(self.cfg)

        period = float(getattr(self.world, "period", 5700.0))
        self.episode_limit = period * float(self.cfg.get("episode_orbits", 1.5))
        dur = self.cfg.get("duration_s", [90.0, 720.0])
        pwr = self.cfg.get("power_need_w", [20.0, 45.0])
        self.feat_spec = FeatureSpec(
            n_sats=self.n_sats,
            capability_dim=self.capability_dim,
            orbit_period=period,
            max_queue=self.max_queue,
            max_attempts=self.max_attempts,
            max_duration=float(dur[1]),
            max_power=float(pwr[1]),
        )
        self.observation_space = gym.spaces.Box(
            low=-2.0, high=2.0, shape=(self.feat_spec.obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(self.n_sats + 1)

        self.stack: Deque[GroundTask] = deque()
        self.sats: list[SatRuntime] = []
        self.current: GroundTask | None = None
        self._defer_seen: set[int] = set()
        self._np_random = np.random.default_rng()
        self._masks = np.ones(self.n_sats + 1, dtype=bool)
        self.info_stats: dict[str, Any] = {}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._np_random = np.random.default_rng(seed)
        self.world.reset(seed=seed)
        self.sats = [
            SatRuntime(sat_id=i, capability=self._sat_capability(i))
            for i in range(self.n_sats)
        ]
        self.stack = deque(self._spawn_tasks())
        self.current = self.stack.popleft() if self.stack else None
        self._defer_seen = set()
        self.info_stats = {
            "n_complete": 0,
            "n_handoffs": 0,
            "n_defers": 0,
            "n_attaches": 0,
            "work_s": 0.0,
        }
        obs = self._observe()
        return obs, dict(self.info_stats)

    def action_masks(self) -> np.ndarray:
        return self._compute_masks()

    def step(self, action: int):
        if self.current is None:
            obs = self._observe()
            return obs, 0.0, True, False, dict(self.info_stats)

        action = int(action)
        masks = self._compute_masks()
        if not masks[action]:
            action = self.n_sats

        reward = 0.0
        terminated = False
        truncated = False

        if action == self.n_sats:
            reward += self._defer_current()
        else:
            reward += self._attach_current(action)
            reward += self._simulate(self.max_step_duration)

        reward += self._drain_empty_stack()
        dead = self._any_dead()
        if dead:
            reward -= 1.0
            terminated = True
        if self.world.t >= self.episode_limit:
            truncated = True
            reward += self._incomplete_penalty()
        if self.current is None and not self.stack and not self._any_working():
            terminated = True

        obs = self._observe()
        info = dict(self.info_stats)
        info["t"] = self.world.t
        info["stack"] = len(self.stack) + (1 if self.current is not None else 0)
        return obs, float(reward), bool(terminated), bool(truncated), info

    def _sat_capability(self, sat_i: int) -> np.ndarray:
        cap = np.ones(self.capability_dim, dtype=np.float32)
        if self.capability_dim > 1 and sat_i % 2 == 1:
            cap[1] = 0.0
        return cap

    def _spawn_tasks(self) -> list[GroundTask]:
        dur_lo, dur_hi = self.cfg.get("duration_s", [90.0, 720.0])
        p_lo, p_hi = self.cfg.get("power_need_w", [20.0, 45.0])
        tasks: list[GroundTask] = []
        for i in range(self.n_tasks):
            lat = float(np.arcsin(self._np_random.uniform(-0.85, 0.85)))
            lon = float(self._np_random.uniform(-np.pi, np.pi))
            if self._np_random.random() < self.long_task_frac:
                duration = float(self._np_random.uniform(0.65 * dur_hi, dur_hi))
            else:
                duration = float(self._np_random.uniform(dur_lo, 0.55 * dur_hi))
            power = float(self._np_random.uniform(p_lo, p_hi))
            cap = np.zeros(self.capability_dim, dtype=np.float32)
            cap[0] = 1.0
            if self.capability_dim > 1 and self._np_random.random() < 0.2:
                cap[1] = 1.0
            tasks.append(
                GroundTask(
                    task_id=i,
                    lat=lat,
                    lon=lon,
                    duration=duration,
                    duration_original=duration,
                    power_need=power,
                    capability=cap,
                )
            )
        return tasks

    def _defer_current(self) -> float:
        task = self.current
        assert task is not None
        self.info_stats["n_defers"] += 1
        anyone = bool(self._compute_masks()[: self.n_sats].any())
        reward = 0.0 if not anyone else -0.05
        self.stack.append(task)
        self._defer_seen.add(task.task_id)
        self.current = None
        remaining_ids = {t.task_id for t in self.stack}
        if remaining_ids and remaining_ids.issubset(self._defer_seen):
            reward += self._simulate(self.max_defer_advance)
            self._defer_seen.clear()
        self._pull_next()
        return reward

    def _attach_current(self, sat_i: int) -> float:
        task = self.current
        assert task is not None
        sat = self.sats[sat_i]
        task.attempts += 1
        self.info_stats["n_attaches"] += 1
        qlen = len(sat.queue)
        sat.queue.append(task)
        self.current = None
        self._defer_seen.clear()
        self._pull_next()
        return -0.02 * qlen

    def _pull_next(self) -> None:
        if self.current is None and self.stack:
            self.current = self.stack.popleft()

    def _drain_empty_stack(self) -> float:
        reward = 0.0
        if self.current is not None:
            return reward
        while (
            self.current is None
            and not self.stack
            and self._any_working()
            and self.world.t < self.episode_limit
            and not self._any_dead()
        ):
            reward += self._simulate(self.max_step_duration)
        return reward

    def _any_working(self) -> bool:
        return any(sat.queue for sat in self.sats)

    def _any_dead(self) -> bool:
        return any(self.world.battery_frac(i) <= 1e-8 for i in range(self.n_sats))

    def _incomplete_penalty(self) -> float:
        leftover = []
        if self.current is not None:
            leftover.append(self.current)
        leftover.extend(self.stack)
        for sat in self.sats:
            leftover.extend(sat.queue)
        penalty = 0.0
        for task in leftover:
            penalty -= task.duration / max(task.duration_original, 1e-6)
        return penalty

    def _simulate(self, horizon: float) -> float:
        reward = 0.0
        elapsed = 0.0
        while elapsed < horizon - 1e-9 and self.world.t < self.episode_limit:
            dt = min(self.sim_dt, horizon - elapsed, self.episode_limit - self.world.t)
            if dt <= 0.0:
                break
            payload = np.zeros(self.n_sats, dtype=float)
            working: list[tuple[int, GroundTask, float]] = []
            for i, sat in enumerate(self.sats):
                if not sat.queue:
                    continue
                task = sat.queue[0]
                if self.world.elevation(i, task.lat, task.lon) < self.world.min_elev:
                    continue
                if self.world.battery_ws(i) <= 0.0:
                    continue
                payload[i] = task.power_need
                working.append((i, task, dt))

            self.world.step(dt, payload)
            elapsed += dt

            handed: list[GroundTask] = []
            for i, task, planned_dt in working:
                energy_left = self.world.battery_ws(i)
                if energy_left <= 0.0 and planned_dt > 0:
                    continue
                work_dt = planned_dt
                if self.world.elevation(i, task.lat, task.lon) < self.world.min_elev:
                    work_dt = 0.0
                work_dt = min(work_dt, task.duration)
                if work_dt <= 0.0:
                    continue
                task.duration -= work_dt
                self.info_stats["work_s"] += work_dt
                reward += work_dt / task.duration_original
                if task.duration <= 1e-6:
                    task.duration = 0.0
                    self.sats[i].queue.popleft()
                    self.info_stats["n_complete"] += 1
                    reward += 0.2

            for i, sat in enumerate(self.sats):
                if not sat.queue:
                    continue
                task = sat.queue[0]
                in_view = self.world.elevation(i, task.lat, task.lon) >= self.world.min_elev
                if not in_view and task.duration > 1e-6:
                    sat.queue.popleft()
                    if task.attempts >= self.max_attempts:
                        reward -= task.duration / task.duration_original
                    else:
                        handed.append(task)
                        self.info_stats["n_handoffs"] += 1

            if handed:
                for task in reversed(handed):
                    self.stack.appendleft(task)
                if self.current is not None:
                    self.stack.appendleft(self.current)
                    self.current = None
                self._pull_next()
                break

            if any(t.duration <= 0.0 for _, t, _ in working):
                break
        return reward

    def _time_until_free(self, sat_i: int) -> float:
        sat = self.sats[sat_i]
        if not sat.queue:
            return 0.0
        total = 0.0
        t_cursor = self.world.t
        for k, task in enumerate(sat.queue):
            acc = self.world.access_remaining(sat_i, task.lat, task.lon, t_cursor)
            slice_s = min(task.duration, acc) if acc > 0 else task.duration
            total += slice_s
            t_cursor += max(slice_s, self.min_work_slice_s)
            if k >= 3:
                break
        return total

    def _status(self, sat_i: int) -> int:
        if self.sats[sat_i].queue:
            return STATUS_BUSY
        return STATUS_IDLE if self.world.in_sun(sat_i) else STATUS_SLEEP

    def _compute_masks(self) -> np.ndarray:
        masks = np.zeros(self.n_sats + 1, dtype=bool)
        masks[self.n_sats] = True
        task = self.current
        if task is None:
            self._masks = masks
            return masks
        for i, sat in enumerate(self.sats):
            if len(sat.queue) >= self.max_queue:
                continue
            if float(np.dot(sat.capability, task.capability)) <= 0.0:
                continue
            t_free = self.world.t + self._time_until_free(i)
            access = self.world.access_remaining(i, task.lat, task.lon, t_free)
            if access <= 0.0:
                continue
            slice_s = min(access, task.duration, self.min_work_slice_s)
            energy_need = task.power_need * slice_s
            if self.world.battery_ws(i) < energy_need:
                continue
            masks[i] = True
        self._masks = masks
        return masks

    def _observe(self) -> np.ndarray:
        if self.current is None:
            task_vec = np.zeros(self.feat_spec.task_dim, dtype=np.float32)
            lat = lon = 0.0
        else:
            tsk = self.current
            lat, lon = tsk.lat, tsk.lon
            task_vec = encode_task(
                self.feat_spec,
                tsk.lat,
                tsk.lon,
                tsk.duration,
                tsk.duration_original,
                tsk.power_need,
                tsk.attempts,
                tsk.capability,
            )
        sats = []
        for i, sat in enumerate(self.sats):
            lat_now, lon_now, alt = self.world.lla(i)
            t_free = self._time_until_free(i)
            lat_f, lon_f, _ = self.world.lla(i, self.world.t + t_free)
            access = 0.0
            if self.current is not None:
                access = self.world.access_remaining(i, lat, lon, self.world.t + t_free)
            sats.append(
                encode_sat(
                    self.feat_spec,
                    lat_now,
                    lon_now,
                    alt,
                    lat_f,
                    lon_f,
                    footprint_radius(alt, self.world.min_elev),
                    self.world.battery_frac(i),
                    self._status(i),
                    len(sat.queue),
                    t_free,
                    sat.capability,
                    access,
                    self.world.time_to_sun_change(i),
                )
            )
        n_stack = len(self.stack) + (1 if self.current is not None else 0)
        glob = np.array(
            [
                np.clip(self.world.t / self.episode_limit, 0.0, 1.5),
                np.clip(n_stack / max(self.n_tasks, 1), 0.0, 2.0),
            ],
            dtype=np.float32,
        )
        return pack_obs(task_vec, np.stack(sats, axis=0), glob)
