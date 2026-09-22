"""Gymnasium-среда поверх `Session` кейса.

Один шаг среды = один тик 5 минут на всю группировку. Действие факторизовано по
аппаратам: `MultiDiscrete([K + 2] * N_max)`, где первые `K` индексов — слоты
заданий-кандидатов этого тика, `K` — ожидание, `K + 1` — калибровка.

Физика не дублируется: заряд, температура, калибровка и допуск команд считаются
в `sim.ops.resource_env.Environment`, среда только кодирует наблюдение,
чинит конфликты и начисляет награду.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import gymnasium as gym
import numpy as np

from sim.ops import Session, load, resolve_scenario

N_MAX = 48
TOP_K = 32
PLAN_MAX = 8192
SAT_FEATURES = 14
JOB_FEATURES = 13
EDGE_FEATURES = 5
GLOBAL_FEATURES = 8

GOALS = ("priority", "revenue")

_HORIZON = 24.0  # шкала нормировки «шагов до события», 2 часа
_WORK_SCALE = 4.0


@dataclass
class RewardConfig:
    """Веса награды. Физика от `goal` не зависит, зависит только начисление."""

    complete_priority: tuple[float, float, float] = (1.5, 4.0, 10.0)
    miss_priority: tuple[float, float, float] = (0.5, 2.0, 8.0)
    value_scale_priority: float = 0.01
    value_scale_revenue: float = 0.04
    miss_value_revenue: float = 0.02
    progress: float = 0.15
    below_reserve: float = 0.02
    brownout: float = 0.5
    blocked: float = 0.02
    executor_switch: float = 0.05
    # Дубль задания repair гасит в idle, но log_prob дубля иначе получает ту же награду.
    conflict: float = 0.15
    # Калибровка бесплатна только у конца допуска — тот же порог, что у greedy (0.85).
    early_calibrate: float = 0.15
    calibrate_free_at: float = 0.85


@dataclass
class EventConfig:
    """Вероятности стохастических сообщений на тик (только train)."""

    p_add_jobs: float = 0.12
    p_outage: float = 0.05
    p_close_downlink: float = 0.03
    add_jobs_count: tuple[int, int] = (1, 2)
    outage_sats: tuple[int, int] = (1, 3)
    outage_len: tuple[int, int] = (2, 8)
    close_downlink_sats: tuple[int, int] = (1, 6)
    close_downlink_len: tuple[int, int] = (2, 6)

    @classmethod
    def off(cls) -> "EventConfig":
        return cls(p_add_jobs=0.0, p_outage=0.0, p_close_downlink=0.0)


@dataclass
class _Candidate:
    job_id: str
    job: dict
    sats: list[int] = field(default_factory=list)


class OpsEnv(gym.Env):
    """Диспетчер группировки на модели организаторов."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: str | Path | dict = "p02",
        goal: Literal["priority", "revenue"] = "priority",
        top_k: int = TOP_K,
        n_max: int = N_MAX,
        events: EventConfig | None = None,
        reward: RewardConfig | None = None,
        scripted_events: list[dict] | None = None,
        max_steps: int | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if goal not in GOALS:
            raise ValueError(f"goal must be one of {GOALS}")
        self.scenario = scenario if isinstance(scenario, dict) else load(resolve_scenario(scenario))
        self.goal = goal
        self.K = int(top_k)
        self.N = int(n_max)
        self.events_cfg = events if events is not None else EventConfig()
        self.reward_cfg = reward if reward is not None else RewardConfig()
        self.scripted_events = list(scripted_events or [])
        self.max_steps = max_steps

        self.sat_ids: list[str] = sorted(v["id"] for v in self.scenario["satellites"])
        self.n_sats = len(self.sat_ids)
        if self.n_sats > self.N:
            raise ValueError(f"В сценарии {self.n_sats} КА, а n_max={self.N}")
        self.steps_total = int(self.scenario["time"]["steps"])
        self.model = self.scenario["model"]
        self.downlink_limit = int(self.model["downlink_parallel_limit"])
        self.calibration_valid = int(self.model["calibration_valid_steps"])
        self.reserve_pct = float(self.model["reserve_soc_pct"])

        self.idle_index = self.K
        self.calibrate_index = self.K + 1
        self.action_space = gym.spaces.MultiDiscrete([self.K + 2] * self.N)
        self.observation_space = gym.spaces.Dict(
            {
                "sats": gym.spaces.Box(-5.0, 5.0, (self.N, SAT_FEATURES), np.float32),
                "jobs": gym.spaces.Box(-5.0, 5.0, (self.K, JOB_FEATURES), np.float32),
                "plan": gym.spaces.Box(-5.0, 5.0, (PLAN_MAX, JOB_FEATURES), np.float32),
                "edges": gym.spaces.Box(0.0, 1.0, (self.N, self.K, EDGE_FEATURES), np.float32),
                "global": gym.spaces.Box(-5.0, 5.0, (GLOBAL_FEATURES,), np.float32),
                "mask": gym.spaces.Box(0.0, 1.0, (self.N, self.K + 2), np.float32),
            }
        )

        self._precompute_solar()
        self.rng = np.random.default_rng(seed)
        self.session: Session | None = None
        self._mask = np.zeros((self.N, self.K + 2), dtype=bool)
        self._candidates: list[_Candidate] = []
        self._last_executor: dict[str, str] = {}
        self._event_counter = 0
        self.ops_stats: dict[str, float] = {}

    # ------------------------------------------------------------------ setup

    def _precompute_solar(self) -> None:
        env = self.scenario["environment"]
        t = self.steps_total
        specs = {v["id"]: v for v in self.scenario["satellites"]}

        def column(key: str) -> np.ndarray:
            return np.array([float(specs[sid][key]) for sid in self.sat_ids], dtype=np.float32)

        self.capacity = column("capacity_wh")
        self.base_w = column("base_w")
        self.heater_w = column("heater_w")
        self.solar = np.array([env[sid]["solar_w"] for sid in self.sat_ids], dtype=np.float32)
        # «Сколько шагов до тени» и «до солнца»: аккумулятор заряжается только при
        # solar_w > 0, поэтому это главный горизонт планирования для агента.
        self.to_dark = np.full((self.n_sats, t), t, dtype=np.float32)
        self.to_sun = np.full((self.n_sats, t), t, dtype=np.float32)
        for i in range(self.n_sats):
            dark = t
            sun = t
            for k in range(t - 1, -1, -1):
                lit = self.solar[i, k] > 0.0
                dark = 0 if not lit else dark + 1
                sun = 0 if lit else sun + 1
                self.to_dark[i, k] = dark
                self.to_sun[i, k] = sun

    # ------------------------------------------------------------------- gym

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.session = Session(
            self.scenario,
            run_metadata={"goal": self.goal, "scenario": self.scenario["meta"]["id"]},
        )
        self._last_executor = {}
        self._event_counter = 0
        self.ops_stats = {
            "act_idle": 0.0,
            "act_calibrate": 0.0,
            "act_job": 0.0,
            "conflicts_repaired": 0.0,
            "downlink_slots_used": 0.0,
            "executor_switches": 0.0,
            "events_add_jobs": 0.0,
            "events_outage": 0.0,
            "events_close_downlink": 0.0,
            "events_skipped": 0.0,
            "reward_total": 0.0,
        }
        self._deliver_events()
        obs = self._observe()
        return obs, {"step": 0}

    def action_masks(self) -> np.ndarray:
        return self._mask.reshape(-1).copy()

    def step(self, action):
        assert self.session is not None, "reset() перед step()"
        env = self.session.env
        commands, stats = self._repair(np.asarray(action, dtype=np.int64))
        prev_executor = dict(self._last_executor)
        rows = self.session.advance(commands)
        reward = self._reward(rows, prev_executor, stats)

        self.ops_stats["act_idle"] += stats["idle"]
        self.ops_stats["act_calibrate"] += stats["calibrate"]
        self.ops_stats["act_job"] += stats["job"]
        self.ops_stats["conflicts_repaired"] += stats["conflicts"]
        self.ops_stats["downlink_slots_used"] += stats["downlinks"]
        self.ops_stats["reward_total"] += reward

        terminated = env.k >= self.steps_total
        truncated = bool(self.max_steps is not None and env.k >= self.max_steps)
        info: dict[str, Any] = {"step": env.k}
        if not (terminated or truncated):
            self._deliver_events()
        else:
            info["summary"] = self.session.summary()
            info["ops"] = self._episode_metrics()
        obs = self._observe()
        return obs, float(reward), bool(terminated), bool(truncated), info

    # --------------------------------------------------------------- события

    def _deliver_events(self) -> None:
        """Сообщения приходят перед действием текущего шага — как в `apply_event`."""
        assert self.session is not None
        k = self.session.env.k
        for event in self.scripted_events:
            if event.get("at_step") == k:
                self.session.apply_event(event)
                self.ops_stats[f"events_{event['type']}"] = (
                    self.ops_stats.get(f"events_{event['type']}", 0.0) + 1.0
                )
        cfg = self.events_cfg
        # «Новое задание» в учебном шуме — пропуск: план сценария и так известен
        # с первого шага, а случайная добавка забивала короткую смену.
        # Операторское add_jobs по-прежнему приходит через apply_event.
        if cfg.p_add_jobs and self.rng.random() < cfg.p_add_jobs:
            self.ops_stats["events_skipped"] += 1.0
        if cfg.p_outage and self.rng.random() < cfg.p_outage:
            self._emit_outage(k, "satellite_outage")
        if cfg.p_close_downlink and self.rng.random() < cfg.p_close_downlink:
            self._emit_outage(k, "close_downlink")

    def _next_event_id(self) -> str:
        self._event_counter += 1
        return f"RND-{self._event_counter:04d}"

    def _emit_add_jobs(self, k: int) -> None:
        assert self.session is not None
        cfg = self.events_cfg
        lo, hi = cfg.add_jobs_count
        jobs: list[dict] = []
        for _ in range(int(self.rng.integers(lo, hi + 1))):
            work = int(self.rng.integers(1, 4))
            slack = int(self.rng.integers(1, 7))
            deadline = k + work + slack
            if deadline > self.steps_total:
                continue
            kind = "downlink" if self.rng.random() < 0.25 else "relay"
            if kind == "downlink":
                eligible = [self.sat_ids[int(self.rng.integers(0, self.n_sats))]]
            else:
                size = int(self.rng.integers(2, min(5, self.n_sats) + 1))
                picks = self.rng.choice(self.n_sats, size=size, replace=False)
                eligible = [self.sat_ids[int(p)] for p in picks]
            priority = int(self.rng.choice([1, 2, 3], p=[0.3, 0.4, 0.3]))
            jobs.append(
                {
                    "id": f"{self._next_event_id()}-J{len(jobs)}",
                    "kind": kind,
                    "release_step": k,
                    "deadline_step": deadline,
                    "work_steps": work,
                    "eligible_satellites": eligible,
                    "priority": priority,
                    "value_usd": round(float(self.rng.uniform(15.0, 40.0 * priority)), 2),
                }
            )
        if not jobs:
            return
        self.session.apply_event(
            {"id": self._next_event_id(), "at_step": k, "type": "add_jobs", "jobs": jobs}
        )
        self.ops_stats["events_add_jobs"] += 1.0

    def _emit_outage(self, k: int, kind: str) -> None:
        assert self.session is not None
        cfg = self.events_cfg
        if kind == "satellite_outage":
            # Выводим из строя только незанятых — занятый КА иначе теряет прогресс
            # задания, а по условию сбой приходит на свободные аппараты.
            busy = set(self._last_executor.values())
            pool = [i for i, sid in enumerate(self.sat_ids) if sid not in busy]
            lo, hi = cfg.outage_sats
            length = cfg.outage_len
        else:
            pool = list(range(self.n_sats))
            lo, hi = cfg.close_downlink_sats
            length = cfg.close_downlink_len
        if not pool:
            return
        size = int(self.rng.integers(lo, min(hi, len(pool)) + 1))
        picks = self.rng.choice(pool, size=size, replace=False)
        end = k + int(self.rng.integers(length[0], length[1] + 1))
        if end > self.steps_total or end <= k:
            return
        self.session.apply_event(
            {
                "id": self._next_event_id(),
                "at_step": k,
                "type": kind,
                "satellite_ids": [self.sat_ids[int(p)] for p in picks],
                "end_step": end,
            }
        )
        self.ops_stats["events_outage" if kind == "satellite_outage" else "events_close_downlink"] += 1.0

    # ------------------------------------------------------------- наблюдение

    def _known_jobs(self) -> list[dict]:
        """Все незавершённые задания, включая те, чьё окно ещё не открылось.

        Плановый список по постановке известен с начала расчёта. Выполнять
        задание до `release_step` по-прежнему нельзя — это решает маска.
        """
        assert self.session is not None
        out = []
        for j in self.session.env.jobs.values():
            if j["completed_step"] is not None or j["remaining_steps"] <= 0:
                continue
            out.append(j)
        return out

    def _open_jobs(self) -> list[dict]:
        assert self.session is not None
        k = self.session.env.k
        return [j for j in self._known_jobs() if j["release_step"] <= k < j["deadline_step"]]

    def _has_contact(self, job: dict) -> bool:
        assert self.session is not None
        env = self.session.env
        k = env.k
        key = job["kind"] + "_available"
        return any(
            env.available(sid) and env.s["environment"][sid][key][k]
            for sid in job["eligible_satellites"]
        )

    def _job_rank(self, job: dict) -> tuple:
        """Сначала исполнимые сейчас; среди остальных relay раньше downlink без окна."""
        assert self.session is not None
        k = self.session.env.k
        open_now = job["release_step"] <= k < job["deadline_step"]
        if open_now and self._has_contact(job):
            return (0, -job["priority"], job["deadline_step"], job["remaining_steps"], job["id"])
        if job["release_step"] > k or open_now:
            # Неисполнимые downlink не занимают слоты впереди будущих relay.
            kind_penalty = 0 if job["kind"] == "relay" else 1
            future = 0 if job["release_step"] > k else 1
            return (
                1,
                kind_penalty,
                future,
                job["release_step"],
                -job["priority"],
                job["deadline_step"],
                job["id"],
            )
        return (2, job["deadline_step"], -job["priority"], job["id"])

    def _job_features(self, job: dict, k: int) -> tuple[float, ...]:
        done = 1.0 - job["remaining_steps"] / job["work_steps"]
        slack = job["deadline_step"] - k - job["remaining_steps"]
        to_release = max(job["release_step"] - k, 0)
        priority = job["priority"]
        return (
            float(priority == 1),
            float(priority == 2),
            float(priority == 3),
            job["value_usd"] / 100.0,
            done,
            min(job["remaining_steps"], _WORK_SCALE) / _WORK_SCALE,
            float(np.clip(slack / _WORK_SCALE, -1.0, 1.0)),
            min(max(job["deadline_step"] - k, 0), _HORIZON) / _HORIZON,
            float(job["kind"] == "downlink"),
            float(job["kind"] == "relay"),
            len(job["eligible_satellites"]) / self.n_sats,
            float(self._last_executor.get(job["id"]) is not None),
            min(to_release, _HORIZON) / _HORIZON,
        )

    def _select_candidates(self, known: list[dict]) -> list[_Candidate]:
        """Top-K: исполнимые сейчас, затем ближайшие будущие (маска их закроет)."""
        ranked = sorted(known, key=self._job_rank)
        index = {sid: i for i, sid in enumerate(self.sat_ids)}
        return [
            _Candidate(job_id=j["id"], job=j, sats=[index[s] for s in j["eligible_satellites"]])
            for j in ranked[: self.K]
        ]

    def _observe(self) -> dict[str, np.ndarray]:
        assert self.session is not None
        env = self.session.env
        k = min(env.k, self.steps_total - 1)
        finished = env.k >= self.steps_total

        known = [] if finished else self._known_jobs()
        open_jobs = [j for j in known if j["release_step"] <= env.k < j["deadline_step"]]
        self._candidates = [] if finished else self._select_candidates(known)

        # Строки сверх n_sats остаются нулевыми: нулевая ёмкость — то, по чему
        # политика отличает padding от реального аппарата.
        sats = np.zeros((self.N, SAT_FEATURES), dtype=np.float32)
        jobs = np.zeros((self.K, JOB_FEATURES), dtype=np.float32)
        plan = np.zeros((PLAN_MAX, JOB_FEATURES), dtype=np.float32)
        edges = np.zeros((self.N, self.K, EDGE_FEATURES), dtype=np.float32)
        mask = np.zeros((self.N, self.K + 2), dtype=bool)
        # Ожидание допустимо всегда, в том числе на padding-строках: иначе у
        # категориального распределения не осталось бы ни одного валидного класса.
        mask[:, self.idle_index] = True

        m = self.model
        soc = np.zeros(self.n_sats, dtype=np.float32)
        for i, sid in enumerate(self.sat_ids):
            st = env.state[sid]
            cap = self.capacity[i]
            soc[i] = st["energy_wh"] / cap
            temp = st["temp_c"]
            age = st["calibration_age_steps"]
            avail = env.available(sid)
            e = env.s["environment"][sid]
            solar = float(e["solar_w"][k])
            heater = self.heater_w[i] if temp < m["heater_below_c"] else 0.0
            sats[i] = (
                soc[i],
                soc[i] - self.reserve_pct / 100.0,
                temp / 50.0,
                age / self.calibration_valid,
                np.clip((self.calibration_valid - age) / self.calibration_valid, -1.0, 1.0),
                float(avail),
                float(e["downlink_available"][k]),
                float(e["relay_available"][k]),
                solar / 100.0,
                min(self.to_dark[i, k], _HORIZON) / _HORIZON,
                min(self.to_sun[i, k], _HORIZON) / _HORIZON,
                cap / 200.0,
                float(heater > 0.0),
                (solar - self.base_w[i] - heater) / 100.0,
            )

        if not finished:
            for i, sid in enumerate(self.sat_ids):
                ok, _, _ = env.can_execute(sid, {"action": "calibrate"})
                mask[i, self.calibrate_index] = ok

            n_prio3 = sum(j["priority"] == 3 for j in open_jobs)
            downlink_demand = 0
            ranked_plan = sorted(known, key=self._job_rank)
            for row, job in enumerate(ranked_plan[:PLAN_MAX]):
                plan[row] = self._job_features(job, k)
            for slot, cand in enumerate(self._candidates):
                j = cand.job
                jobs[slot] = self._job_features(j, k)
                key = j["kind"] + "_available"
                for i in cand.sats:
                    sid = self.sat_ids[i]
                    ok, reason, _ = env.can_execute(sid, {"action": "job", "job_id": j["id"]})
                    edges[i, slot] = (
                        1.0,
                        float(ok),
                        float(reason != "energy_reserve"),
                        float(env.s["environment"][sid][key][k]),
                        float(self._last_executor.get(j["id"]) == sid),
                    )
                    mask[i, slot] = ok
                if j["kind"] == "downlink" and mask[:, slot].any():
                    downlink_demand += 1
        else:
            n_prio3 = 0
            downlink_demand = 0

        glob = np.array(
            [
                env.k / self.steps_total,
                min(len(open_jobs), 64) / 64.0,
                min(n_prio3, 32) / 32.0,
                min(downlink_demand / self.downlink_limit, 2.0),
                float(soc.mean()) if self.n_sats else 0.0,
                float(soc.min()) if self.n_sats else 0.0,
                float(self.goal == "priority"),
                float(self.goal == "revenue"),
            ],
            dtype=np.float32,
        )

        self._mask = mask
        return {
            "sats": sats,
            "jobs": jobs,
            "plan": plan,
            "edges": edges,
            "global": glob,
            "mask": mask.astype(np.float32),
        }

    # ------------------------------------------------------------------ repair

    def _repair(self, action: np.ndarray) -> tuple[dict[str, dict], dict[str, float]]:
        """Жёсткие ограничения чинятся детерминированно, без второго прохода сети.

        Совпадает с поведением `Environment.step`, который сам гасит
        `duplicate_job_in_step` и `ground_capacity` в `idle`, но чинит заранее,
        чтобы в журнал смены не попадали заведомо отклонённые команды.
        """
        assert self.session is not None
        env = self.session.env
        decoded: list[tuple[str, str | None]] = []
        conflicts = 0
        for i, sid in enumerate(self.sat_ids):
            a = int(action[i]) if i < action.shape[0] else self.idle_index
            if a == self.calibrate_index and self._mask[i, self.calibrate_index]:
                decoded.append(("calibrate", None))
            elif a < self.K and a < len(self._candidates) and self._mask[i, a]:
                decoded.append(("job", self._candidates[a].job_id))
            else:
                decoded.append(("idle", None))

        by_job: dict[str, list[int]] = {}
        for i, (kind, job_id) in enumerate(decoded):
            if kind == "job" and job_id is not None:
                by_job.setdefault(job_id, []).append(i)

        forced_idle: set[int] = set()
        for job_id, holders in by_job.items():
            if len(holders) == 1:
                continue
            last = self._last_executor.get(job_id)
            holders.sort(
                key=lambda i: (
                    0 if self.sat_ids[i] == last else 1,
                    -(env.state[self.sat_ids[i]]["energy_wh"] / self.capacity[i]),
                    i,
                )
            )
            for i in holders[1:]:
                decoded[i] = ("idle", None)
                forced_idle.add(i)
                conflicts += 1
            by_job[job_id] = holders[:1]

        downlinks = [
            (job_id, holders[0])
            for job_id, holders in by_job.items()
            if holders and env.jobs[job_id]["kind"] == "downlink"
        ]
        if len(downlinks) > self.downlink_limit:
            downlinks.sort(
                key=lambda p: (
                    -env.jobs[p[0]]["priority"],
                    env.jobs[p[0]]["deadline_step"],
                    -env.jobs[p[0]]["value_usd"],
                )
            )
            for _, i in downlinks[self.downlink_limit :]:
                decoded[i] = ("idle", None)
                forced_idle.add(i)
                conflicts += 1
            downlinks = downlinks[: self.downlink_limit]

        self._yield_unique_downlinks(decoded, forced_idle)
        self._fill_forced_idle_with_relays(decoded, forced_idle)

        downlinks = [
            (job_id, i)
            for i, (kind, job_id) in enumerate(decoded)
            if kind == "job" and job_id is not None and env.jobs[job_id]["kind"] == "downlink"
        ]

        commands: dict[str, dict] = {}
        counts = {"idle": 0, "calibrate": 0, "job": 0}
        for i, (kind, job_id) in enumerate(decoded):
            counts[kind] += 1
            if kind == "calibrate":
                commands[self.sat_ids[i]] = {"action": "calibrate"}
            elif kind == "job" and job_id is not None:
                commands[self.sat_ids[i]] = {"action": "job", "job_id": job_id}
        # Возраст читаем до advance: калибровка на этом шаге его обнулит.
        early = 0
        valid = max(self.calibration_valid, 1)
        free_at = self.reward_cfg.calibrate_free_at
        for i, (kind, _) in enumerate(decoded):
            if kind != "calibrate":
                continue
            age = env.state[self.sat_ids[i]]["calibration_age_steps"]
            if age / valid < free_at:
                early += 1
        stats = {
            "idle": float(counts["idle"]),
            "calibrate": float(counts["calibrate"]),
            "job": float(counts["job"]),
            "conflicts": float(conflicts),
            "early_calibrations": float(early),
            "downlinks": float(len(downlinks)),
        }
        return commands, stats

    def _slot_by_id(self) -> dict[str, int]:
        return {c.job_id: slot for slot, c in enumerate(self._candidates)}

    @staticmethod
    def _job_pref(job: dict) -> tuple:
        return (-job["priority"], job["deadline_step"], -job["value_usd"], job["id"])

    def _yield_unique_downlinks(
        self,
        decoded: list[tuple[str, str | None]],
        forced_idle: set[int],
    ) -> None:
        """Единственный исполнитель незанятого сброса забирает КА у relay."""
        assert self.session is not None
        env = self.session.env
        slots = self._slot_by_id()
        taken = {job_id for kind, job_id in decoded if kind == "job" and job_id is not None}
        n_downlinks = sum(1 for job_id in taken if env.jobs[job_id]["kind"] == "downlink")
        unique: list[tuple[dict, int, int]] = []
        for slot, cand in enumerate(self._candidates):
            job = cand.job
            if job["kind"] != "downlink" or job["id"] in taken:
                continue
            holders = [i for i in range(self.n_sats) if self._mask[i, slot]]
            if len(holders) != 1:
                continue
            unique.append((job, holders[0], slot))
        unique.sort(key=lambda row: self._job_pref(row[0]))
        for job, i, _slot in unique:
            if n_downlinks >= self.downlink_limit:
                break
            kind, current = decoded[i]
            if kind != "job" or current is None or env.jobs[current]["kind"] != "relay":
                continue
            decoded[i] = ("job", job["id"])
            taken.add(job["id"])
            taken.discard(current)
            n_downlinks += 1
            relay_slot = slots.get(current)
            if relay_slot is None:
                continue
            last = self._last_executor.get(current)
            recipients = [
                j
                for j, (other_kind, _) in enumerate(decoded)
                if j != i and other_kind == "idle" and self._mask[j, relay_slot]
            ]
            recipients.sort(
                key=lambda j: (
                    0 if j in forced_idle else 1,
                    0 if self.sat_ids[j] == last else 1,
                    j,
                )
            )
            if not recipients:
                continue
            other = recipients[0]
            decoded[other] = ("job", current)
            taken.add(current)
            forced_idle.discard(other)

    def _fill_forced_idle_with_relays(
        self,
        decoded: list[tuple[str, str | None]],
        forced_idle: set[int],
    ) -> None:
        """Погашенный дублем или третьим сбросом КА берёт свободный контактный relay."""
        taken = {job_id for kind, job_id in decoded if kind == "job" and job_id is not None}
        relays = [
            (cand.job, slot)
            for slot, cand in enumerate(self._candidates)
            if cand.job["kind"] == "relay" and cand.job_id not in taken
        ]
        relays.sort(key=lambda row: self._job_pref(row[0]))
        for i in sorted(forced_idle):
            if decoded[i][0] != "idle":
                continue
            for job, slot in relays:
                if job["id"] in taken or not self._mask[i, slot]:
                    continue
                decoded[i] = ("job", job["id"])
                taken.add(job["id"])
                break

    # ------------------------------------------------------------------ награда

    def _reward(self, rows: list[dict], prev_executor: dict[str, str], stats: dict[str, float]) -> float:
        assert self.session is not None
        env = self.session.env
        cfg = self.reward_cfg
        k_done = env.k  # шаг уже увеличен внутри Environment.step

        reward = 0.0
        switches = 0
        executed: dict[str, str] = {}
        for r in rows:
            if r["executed"] == "job":
                job_id = r["requested"]["job_id"]
                executed[job_id] = r["satellite_id"]
                reward += cfg.progress
                if prev_executor.get(job_id) not in (None, r["satellite_id"]):
                    switches += 1
            if r["below_reserve"]:
                reward -= cfg.below_reserve
            if r["brownout"]:
                reward -= cfg.brownout
            if r["reason"] not in ("accepted", "idle"):
                reward -= cfg.blocked
        reward -= cfg.executor_switch * switches
        reward -= cfg.conflict * stats.get("conflicts", 0.0)
        reward -= cfg.early_calibrate * stats.get("early_calibrations", 0.0)
        self.ops_stats["executor_switches"] += switches
        self._last_executor = executed

        for r in rows:
            job_id = r["completed_job"]
            if job_id is None:
                continue
            j = env.jobs[job_id]
            if self.goal == "priority":
                reward += cfg.complete_priority[j["priority"] - 1]
                reward += cfg.value_scale_priority * j["value_usd"]
            else:
                reward += cfg.value_scale_revenue * j["value_usd"]

        for j in env.jobs.values():
            if j["deadline_step"] == k_done and j["completed_step"] is None:
                if self.goal == "priority":
                    reward -= cfg.miss_priority[j["priority"] - 1]
                else:
                    reward -= cfg.miss_value_revenue * j["value_usd"]
        return reward

    # ------------------------------------------------------------------ метрики

    def _episode_metrics(self) -> dict[str, float]:
        assert self.session is not None
        total = max(self.ops_stats["act_idle"] + self.ops_stats["act_calibrate"] + self.ops_stats["act_job"], 1.0)
        out = dict(self.ops_stats)
        out["idle_share"] = self.ops_stats["act_idle"] / total
        out["calibrate_share"] = self.ops_stats["act_calibrate"] / total
        out["job_share"] = self.ops_stats["act_job"] / total
        return out

    def journal(self) -> dict:
        """Журнал смены в схеме `cosmo-B-ops-result-1.0`."""
        assert self.session is not None
        return self.session.result()


__all__ = [
    "EDGE_FEATURES",
    "EventConfig",
    "GLOBAL_FEATURES",
    "JOB_FEATURES",
    "N_MAX",
    "PLAN_MAX",
    "OpsEnv",
    "RewardConfig",
    "SAT_FEATURES",
    "TOP_K",
]
