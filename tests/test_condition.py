"""Shared world snapshot used by RL and autonomous CNP."""

from __future__ import annotations

import numpy as np

from harmony.sim.condition import TaskCondition, sat_condition
from harmony.sim.kepler import KeplerWorld


def test_nadir_target_has_access_and_can_work():
    world = KeplerWorld(n_sats=4, n_planes=2)
    world.reset(seed=0)
    lat, lon, _ = world.lla(0)
    task = TaskCondition(
        lat=lat,
        lon=lon,
        duration=120.0,
        duration_original=120.0,
        power_need=20.0,
    )
    cond = sat_condition(world, 0, task)
    assert cond.access_remaining > 0.0
    assert cond.can_work
    assert cond.in_view_now
    assert cond.footprint_radius > 0.0


def test_far_target_cannot_work():
    world = KeplerWorld(n_sats=4, n_planes=2)
    world.reset(seed=0)
    lat, lon, _ = world.lla(0)
    task = TaskCondition(
        lat=lat + np.deg2rad(80.0),
        lon=lon,
        duration=120.0,
        duration_original=120.0,
        power_need=20.0,
    )
    cond = sat_condition(world, 0, task)
    assert cond.access_remaining == 0.0
    assert not cond.can_work


def test_predicted_lla_matches_now_when_idle():
    world = KeplerWorld(n_sats=4, n_planes=2)
    world.reset(seed=1)
    cond = sat_condition(world, 0, time_until_free=0.0)
    assert abs(cond.lat - cond.lat_free) < 1e-9
    assert abs(cond.lon - cond.lon_free) < 1e-9


def test_autonomous_cnp_runs_on_kepler(monkeypatch):
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from autonomous import autonomous

    monkeypatch.setattr(autonomous, "ENABLE_VIZARD", False)

    def kepler_only(cfg, prefer_basilisk=True):
        from harmony.sim.condition import open_world

        return open_world(cfg, prefer_basilisk=False)

    def two_tasks():
        return [
            TaskCondition(
                name="Тест-1",
                lat=0.0,
                lon=0.0,
                duration=40.0,
                duration_original=40.0,
                power_need=20.0,
            )
        ]

    monkeypatch.setattr(autonomous, "open_world", kepler_only)
    monkeypatch.setattr(autonomous, "make_named_tasks", two_tasks)
    summary = autonomous.run_cnp(seed=0)
    assert summary["backend"] == "kepler"
    assert summary["t"] >= 0.0
    assert len(summary["tasks_taken"]) == 4
