"""Shared world snapshot for the RL dispatcher and later web visualization.

This is the single place to extend when the simulated world gains new fields.
Kepler and Basilisk backends expose the same numbers: coverage, predicted LLA,
battery, eclipse.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.coverage import footprint_radius
from sim.kepler import kepler_from_config

STATUS_IDLE = 0
STATUS_SLEEP = 1
STATUS_BUSY = 2


@dataclass
class TaskCondition:
    lat: float
    lon: float
    duration: float
    duration_original: float
    power_need: float
    capability: np.ndarray | None = None
    name: str = ""
    attempts: int = 0


@dataclass
class SatCondition:
    sat_i: int
    lat: float
    lon: float
    alt: float
    lat_free: float
    lon_free: float
    footprint_radius: float
    battery: float
    battery_ws: float
    in_sun: bool
    time_to_sun_change: float
    status: int
    queue_length: int
    time_until_free: float
    elevation_to_task: float
    access_remaining: float
    in_view_now: bool
    can_work: bool


@dataclass
class Heartbeat:
    sat_i: int
    battery_ws: float
    battery_wh: float
    lat_deg: float
    lon_deg: float
    alt_km: float
    in_sun: bool
    shadow_factor: float


def as_task(task) -> TaskCondition | None:
    if task is None:
        return None
    if isinstance(task, TaskCondition):
        return task
    cap = getattr(task, "capability", None)
    return TaskCondition(
        lat=float(task.lat),
        lon=float(task.lon),
        duration=float(task.duration),
        duration_original=float(getattr(task, "duration_original", task.duration)),
        power_need=float(task.power_need),
        capability=None if cap is None else np.asarray(cap, dtype=np.float32),
        name=str(getattr(task, "name", "")),
        attempts=int(getattr(task, "attempts", 0)),
    )


def sat_condition(
    world,
    sat_i: int,
    task=None,
    *,
    time_until_free: float = 0.0,
    queue_length: int = 0,
    capability: np.ndarray | None = None,
    max_queue: int = 4,
    min_work_slice_s: float = 10.0,
) -> SatCondition:
    """Snapshot of one satellite relative to an optional ground task."""
    task = as_task(task)
    lat, lon, alt = world.lla(sat_i)
    t_free = max(0.0, float(time_until_free))
    lat_f, lon_f, _ = world.lla(sat_i, world.t + t_free)
    footprint = footprint_radius(alt, world.min_elev)
    in_sun = bool(world.in_sun(sat_i))
    if queue_length > 0:
        status = STATUS_BUSY
    else:
        status = STATUS_IDLE if in_sun else STATUS_SLEEP

    elevation = -np.pi / 2.0
    access = 0.0
    in_view_now = False
    if task is not None:
        elevation = float(world.elevation(sat_i, task.lat, task.lon, world.t + t_free))
        in_view_now = float(world.elevation(sat_i, task.lat, task.lon)) >= world.min_elev
        access = float(world.access_remaining(sat_i, task.lat, task.lon, world.t + t_free))

    can = True
    if task is None:
        can = False
    elif queue_length >= max_queue:
        can = False
    elif access <= 0.0:
        can = False
    else:
        if capability is not None and task.capability is not None:
            if float(np.dot(np.asarray(capability, dtype=float), task.capability)) <= 0.0:
                can = False
        slice_s = min(access, task.duration, min_work_slice_s)
        if world.battery_ws(sat_i) < task.power_need * slice_s:
            can = False

    return SatCondition(
        sat_i=sat_i,
        lat=lat,
        lon=lon,
        alt=alt,
        lat_free=lat_f,
        lon_free=lon_f,
        footprint_radius=footprint,
        battery=world.battery_frac(sat_i),
        battery_ws=world.battery_ws(sat_i),
        in_sun=in_sun,
        time_to_sun_change=float(world.time_to_sun_change(sat_i)),
        status=status,
        queue_length=int(queue_length),
        time_until_free=t_free,
        elevation_to_task=elevation,
        access_remaining=access,
        in_view_now=in_view_now,
        can_work=can,
    )


def world_heartbeat(world) -> list[Heartbeat]:
    rows = []
    for i in range(world.n_sats):
        lat, lon, alt = world.lla(i)
        in_sun = bool(world.in_sun(i))
        rows.append(
            Heartbeat(
                sat_i=i,
                battery_ws=world.battery_ws(i),
                battery_wh=world.battery_ws(i) / 3600.0,
                lat_deg=float(np.degrees(lat)),
                lon_deg=float(np.degrees(lon)),
                alt_km=float(alt / 1000.0),
                in_sun=in_sun,
                shadow_factor=1.0 if in_sun else 0.0,
            )
        )
    return rows


def open_world(cfg: dict, prefer_basilisk: bool = True):
    """Build the shared world: Basilisk if installed, otherwise Kepler."""
    if prefer_basilisk:
        from sim.basilisk_world import basilisk_available, basilisk_from_config

        if basilisk_available():
            world = basilisk_from_config(cfg)
            world.reset(seed=0)
            return world, "basilisk"
    world = kepler_from_config(cfg)
    world.reset(seed=0)
    return world, "kepler"
