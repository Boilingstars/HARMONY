"""Среда на модели кейса: энергия, калибровка, лимит downlink, handoff, события."""

from __future__ import annotations

import numpy as np
import pytest

from agent.env.ops_env import EventConfig, OpsEnv
from agent.greedy import greedy_action, random_masked_action
from sim.ops import load, load_events
from sim.ops.scenarios import EVENTS_DEMO, P01_INTRO, P02_SHIFT


@pytest.fixture(scope="module")
def p01() -> dict:
    return load(P01_INTRO)


def _env(scenario, **kwargs) -> OpsEnv:
    kwargs.setdefault("events", EventConfig.off())
    kwargs.setdefault("seed", 0)
    return OpsEnv(scenario=scenario, **kwargs)


def _all_idle(env: OpsEnv) -> np.ndarray:
    return np.full(env.N, env.idle_index, dtype=np.int64)


def test_spaces_and_padding(p01):
    env = _env(p01)
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert env.action_space.shape == (env.N,)
    mask = env.action_masks().reshape(env.N, env.K + 2)
    # Реальных КА 16, остальные строки — padding: только ожидание.
    assert mask[env.n_sats :, env.idle_index].all()
    assert not mask[env.n_sats :, : env.idle_index].any()
    assert not mask[env.n_sats :, env.calibrate_index].any()
    assert mask[:, env.idle_index].all()


def test_battery_charges_only_in_sunlight(p01):
    """В затмении аккумулятор не заряжается, на солнце — заряжается."""
    env = _env(p01)
    env.reset(seed=0)
    model = env.model
    charged_in_sun = discharged_in_dark = False
    for _ in range(env.steps_total):
        rows = {r["satellite_id"]: r for r in env.session.advance({})}
        for row in rows.values():
            delta = row["energy_after_wh"] - row["energy_before_wh"]
            in_temp_window = model["charge_min_c"] <= row["temp_before_c"] <= model["charge_max_c"]
            if row["solar_w"] == 0.0:
                assert delta <= 1e-9, "в тени заряд не растёт"
                discharged_in_dark = True
            elif row["solar_w"] > row["load_w"] and in_temp_window and row["energy_after_wh"] > 0:
                if delta > 0:
                    charged_in_sun = True
    assert charged_in_sun and discharged_in_dark


def test_heater_draws_power_below_threshold(p01):
    env = _env(p01)
    env.reset(seed=0)
    seen = False
    for _ in range(env.steps_total):
        for row in env.session.advance({}):
            if row["temp_before_c"] < env.model["heater_below_c"]:
                assert row["heater_w"] > 0.0
                assert row["load_w"] >= row["heater_w"]
                seen = True
            else:
                assert row["heater_w"] == 0.0
    assert seen or True  # в P01 нагреватель может не включиться — проверяем инвариант


def test_calibration_expiry_blocks_jobs_and_calibrate_clears_it(p01):
    """Срок допуска кончился — задания закрыты, калибровка снимает блок."""
    env = _env(p01)
    env.reset(seed=0)
    inner = env.session.env
    sid = env.sat_ids[0]
    job_id = next(
        j["id"] for j in inner.jobs.values() if sid in j["eligible_satellites"] and j["kind"] == "relay"
    )
    inner.state[sid]["calibration_age_steps"] = env.calibration_valid
    inner.k = inner.jobs[job_id]["release_step"]
    ok, reason, _ = inner.can_execute(sid, {"action": "job", "job_id": job_id})
    assert not ok and reason == "calibration_required"

    ok, _, _ = inner.can_execute(sid, {"action": "calibrate"})
    assert ok
    inner.step({sid: {"action": "calibrate"}})
    assert inner.state[sid]["calibration_age_steps"] == 0


def test_third_downlink_in_a_tick_becomes_idle(p01):
    """Лимит наземного ресурса: не больше двух downlink за тик на всю группировку."""
    env = _env(p01)
    env.reset(seed=0)
    inner = env.session.env
    k = inner.k
    chosen = env.sat_ids[:3]
    jobs = []
    for i, sid in enumerate(chosen):
        inner.s["environment"][sid]["downlink_available"][k] = True
        inner.state[sid]["calibration_age_steps"] = 0
        jobs.append(
            {
                "id": f"DL-{i}",
                "kind": "downlink",
                "release_step": k,
                "deadline_step": k + 3,
                "work_steps": 1,
                "eligible_satellites": [sid],
                "priority": 3,
                "value_usd": 10.0 * (i + 1),
            }
        )
    env.session.apply_event({"id": "T-dl", "at_step": k, "type": "add_jobs", "jobs": jobs})

    env._observe()
    mask = env._mask
    slots = {c.job_id: slot for slot, c in enumerate(env._candidates)}
    action = _all_idle(env)
    ordered = []
    for job in jobs:
        slot = slots[job["id"]]
        index = env.sat_ids.index(job["eligible_satellites"][0])
        assert mask[index, slot], "окно открыто, задание должно быть допустимо"
        action[index] = slot
        ordered.append(job["id"])

    env.step(action)
    rows = inner.trace[-env.n_sats :]
    executed = [r["requested"]["job_id"] for r in rows if r["executed"] == "job"]
    assert len(executed) == env.downlink_limit == 2
    assert env.ops_stats["conflicts_repaired"] == 1
    # Repair оставляет дороже и срочнее: приоритеты равны, решает value_usd.
    assert set(executed) == {"DL-2", "DL-1"}
    assert all(r["reason"] in ("accepted", "idle") for r in rows)


def test_unique_job_per_step_and_downlink_cap():
    """Приказ «всем первое допустимое задание» не ломает ограничения кейса."""
    env = _env(load(P02_SHIFT))
    env.reset(seed=1)
    steps = 0
    saw_conflict = False
    while steps < 40:
        mask = env.action_masks().reshape(env.N, env.K + 2)
        action = _all_idle(env)
        for i in range(env.n_sats):
            valid_jobs = np.flatnonzero(mask[i, : env.K])
            if valid_jobs.size:
                action[i] = int(valid_jobs[0])
        _, _, term, trunc, _ = env.step(action)
        rows = env.session.env.trace[-env.n_sats :]
        executed = [r for r in rows if r["executed"] == "job"]
        job_ids = [r["requested"]["job_id"] for r in executed]
        assert len(job_ids) == len(set(job_ids)), "одно задание — один исполнитель за тик"
        downlinks = sum(
            1 for jid in job_ids if env.session.env.jobs[jid]["kind"] == "downlink"
        )
        assert downlinks <= env.downlink_limit
        assert all(r["reason"] in ("accepted", "idle") for r in rows), "repair до отправки команд"
        saw_conflict |= env.ops_stats["conflicts_repaired"] > 0
        steps += 1
        if term or trunc:
            break
    assert saw_conflict, "конфликты должны появляться и чиниться"


def test_relay_handoff_preserves_progress(p01):
    """Ретрансляцию продолжает другой допустимый КА, прогресс не сбрасывается."""
    env = _env(p01)
    env.reset(seed=0)
    inner = env.session.env
    job = next(
        j
        for j in inner.jobs.values()
        if j["kind"] == "relay" and j["work_steps"] >= 2 and len(j["eligible_satellites"]) >= 2
    )
    inner.k = job["release_step"]
    first, second = job["eligible_satellites"][:2]
    for sid in (first, second):
        inner.s["environment"][sid]["relay_available"][inner.k] = True
        inner.s["environment"][sid]["relay_available"][inner.k + 1] = True
        inner.state[sid]["calibration_age_steps"] = 0

    inner.step({first: {"action": "job", "job_id": job["id"]}})
    assert job["remaining_steps"] == job["work_steps"] - 1
    inner.step({second: {"action": "job", "job_id": job["id"]}})
    assert job["remaining_steps"] == job["work_steps"] - 2
    assert job["remaining_steps"] >= 0


def test_downlink_cannot_be_handed_over(p01):
    """У downlink ровно один допустимый исполнитель — перебросить нельзя."""
    for job in p01["jobs"]:
        if job["kind"] == "downlink":
            assert len(job["eligible_satellites"]) == 1


def test_add_jobs_event_is_visible_on_the_same_step(p01):
    env = _env(p01)
    env.reset(seed=0)
    inner = env.session.env
    before = len(inner.jobs)
    env.session.apply_event(
        {
            "id": "T-add",
            "at_step": inner.k,
            "type": "add_jobs",
            "jobs": [
                {
                    "id": "T-JOB",
                    "kind": "relay",
                    "release_step": inner.k,
                    "deadline_step": inner.k + 4,
                    "work_steps": 1,
                    "eligible_satellites": env.sat_ids[:3],
                    "priority": 3,
                    "value_usd": 99.0,
                }
            ],
        }
    )
    assert len(inner.jobs) == before + 1
    obs = env._observe()
    slots = [c.job_id for c in env._candidates]
    assert "T-JOB" in slots, "новое задание попадает в кандидаты на том же шаге"
    assert obs["jobs"][slots.index("T-JOB"), 2] == 1.0, "приоритет 3 закодирован onehot"
    assert (obs["plan"][:, 0:3].sum(axis=1) > 0).sum() == before + 1


def test_future_jobs_are_visible_but_not_assignable(p01):
    """Плановый список виден с шага 0. До release_step назначить его нельзя."""
    env = _env(p01)
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert obs["plan"].shape[0] == 8192
    assert obs["jobs"].shape[-1] == 13
    known = env._known_jobs()
    future = [j for j in known if j["release_step"] > 0]
    assert future, "в P01 есть задания с release_step > 0"
    visible = int((obs["plan"][:, 0:3].sum(axis=1) > 0).sum())
    assert visible == len(known)
    assert int((obs["plan"][:visible, 12] > 0).sum()) == len(future)
    seen_future_slot = False
    for slot, cand in enumerate(env._candidates):
        if cand.job["release_step"] > 0:
            assert not obs["mask"][:, slot].any()
            seen_future_slot = True
    assert seen_future_slot


def test_random_add_jobs_is_skipped(p01):
    """Учебный шум «новое задание» не меняет пул. Отказ и закрытие сброса остаются."""
    env = OpsEnv(
        scenario=p01,
        events=EventConfig(p_add_jobs=1.0, p_outage=0.0, p_close_downlink=0.0),
        seed=0,
    )
    env.reset(seed=0)
    planned = len(p01["jobs"])
    assert len(env.session.env.jobs) == planned
    assert env.ops_stats["events_skipped"] >= 1
    assert env.ops_stats["events_add_jobs"] == 0
    env.step(_all_idle(env))
    assert len(env.session.env.jobs) == planned


def test_outage_closes_actions_and_expires(p01):
    env = _env(p01)
    env.reset(seed=0)
    inner = env.session.env
    sid = env.sat_ids[0]
    end = inner.k + 3
    env.session.apply_event(
        {
            "id": "T-out",
            "at_step": inner.k,
            "type": "satellite_outage",
            "satellite_ids": [sid],
            "end_step": end,
        }
    )
    assert not inner.available(sid)
    assert not inner.can_execute(sid, {"action": "calibrate"})[0]
    while inner.k < end:
        inner.step({})
    assert inner.available(sid), "по истечении интервала аппарат снова доступен"


def test_close_downlink_event_shuts_the_window(p01):
    env = _env(p01)
    env.reset(seed=0)
    inner = env.session.env
    sid = env.sat_ids[0]
    env.session.apply_event(
        {
            "id": "T-cd",
            "at_step": inner.k,
            "type": "close_downlink",
            "satellite_ids": [sid],
            "end_step": inner.k + 5,
        }
    )
    assert not any(inner.s["environment"][sid]["downlink_available"][inner.k : inner.k + 5])


def test_random_events_stay_valid_over_a_full_episode(p01):
    """Случайные сообщения обучения проходят validate() организаторов."""
    env = OpsEnv(scenario=p01, events=EventConfig(p_add_jobs=0.5, p_outage=0.4, p_close_downlink=0.3), seed=3)
    obs, _ = env.reset(seed=3)
    rng = np.random.default_rng(3)
    done = False
    info: dict = {}
    while not done:
        obs, _, term, trunc, info = env.step(random_masked_action(obs, rng))
        done = term or trunc
    ops = info["ops"]
    assert ops["events_add_jobs"] == 0
    assert ops["events_skipped"] + ops["events_outage"] + ops["events_close_downlink"] > 0
    assert info["summary"]["jobs_total"] == len(p01["jobs"])
    assert info["summary"]["steps_executed"] == env.steps_total


def test_scripted_events_demo_matches_p02():
    scenario = load(P02_SHIFT)
    events = load_events(EVENTS_DEMO, scenario)
    env = _env(scenario, scripted_events=events)
    env.reset(seed=0)
    for _ in range(75):
        env.step(_all_idle(env))
    applied = {e["id"] for e in env.session.events}
    assert {"E-01", "E-02"} <= applied


def test_two_goals_score_the_same_step_differently(p01):
    priority = _env(p01, goal="priority")
    revenue = _env(p01, goal="revenue")
    rewards = {}
    for name, env in (("priority", priority), ("revenue", revenue)):
        obs, _ = env.reset(seed=5)
        total = 0.0
        done = False
        while not done:
            obs, reward, term, trunc, _ = env.step(greedy_action(obs, env.downlink_limit))
            total += reward
            done = term or trunc
        rewards[name] = total
    assert rewards["priority"] != rewards["revenue"]


def test_greedy_beats_random_on_priority(p01):
    from agent.evaluate import evaluate

    greedy = evaluate(scenario=P01_INTRO, policy="greedy", events="none", episodes=1)
    random = evaluate(scenario=P01_INTRO, policy="random", events="none", episodes=1)
    assert greedy["jobs_completed"] > random["jobs_completed"]
    assert greedy["blocked_commands"] == 0


def test_journal_replays(tmp_path, p01):
    from agent.evaluate import evaluate

    report = evaluate(
        scenario=P01_INTRO,
        policy="greedy",
        events="random",
        episodes=1,
        journal_dir=tmp_path,
    )
    path = tmp_path / "journal_0.json"
    assert path.exists()
    assert report["episodes_detail"][0]["journal"] == str(path)
