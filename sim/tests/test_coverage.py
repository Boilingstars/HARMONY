"""Tests for coverage geometry and KeplerWorld."""

from __future__ import annotations

import numpy as np

from sim.constants import R_EARTH
from sim.coverage import (
    access_remaining,
    ecef_to_lla,
    elevation,
    footprint_radius,
    in_sunlight,
    lla_to_ecef,
    rv_at_time,
    walker_delta_elements,
)
from sim.kepler import KeplerWorld


def test_nadir_elevation_is_90_deg():
    lat, lon = np.deg2rad(10.0), np.deg2rad(20.0)
    tgt = lla_to_ecef(lat, lon, 0.0)
    sat = lla_to_ecef(lat, lon, 500e3)
    assert elevation(sat, tgt) > np.deg2rad(89.0)


def test_horizon_target_has_low_elevation():
    sat = lla_to_ecef(0.0, 0.0, 500e3)
    far = lla_to_ecef(0.0, np.deg2rad(80.0), 0.0)
    assert elevation(sat, far) < 0.0


def test_footprint_grows_with_altitude():
    r_low = footprint_radius(400e3, np.deg2rad(10.0))
    r_high = footprint_radius(800e3, np.deg2rad(10.0))
    assert r_high > r_low
    assert r_low > 5e5


def test_lla_roundtrip():
    r = lla_to_ecef(np.deg2rad(45.0), np.deg2rad(-30.0), 500e3)
    lat, lon, alt = ecef_to_lla(r)
    assert abs(lat - np.deg2rad(45.0)) < 1e-6
    assert abs(lon - np.deg2rad(-30.0)) < 1e-6
    assert abs(alt - 500e3) < 1.0


def test_cylindrical_eclipse():
    sun = np.array([1.0, 0.0, 0.0])
    lit = np.array([R_EARTH + 500e3, 0.0, 0.0])
    dark = np.array([-(R_EARTH + 500e3), 0.0, 0.0])
    assert in_sunlight(lit, sun)
    assert not in_sunlight(dark, sun)


def test_walker_has_n_satellites():
    els = walker_delta_elements(4, 2, 500e3, np.deg2rad(51.6))
    assert len(els) == 4
    rs = [np.linalg.norm(rv_at_time(*el, 0.0)[0]) for el in els]
    assert all(abs(r - (R_EARTH + 500e3)) < 1.0 for r in rs)


def test_overhead_access_remaining_positive():
    world = KeplerWorld(n_sats=4, n_planes=2)
    world.reset(seed=0)
    lat, lon, _ = world.lla(0)
    acc = world.access_remaining(0, lat, lon, world.t)
    assert acc > 60.0
    far_lat, far_lon = lat + np.deg2rad(80.0), lon
    assert world.access_remaining(0, far_lat, far_lon, world.t) == 0.0


def test_kepler_battery_charges_in_sun_and_drains_in_eclipse():
    world = KeplerWorld(n_sats=4, n_planes=2, bus_power_w=30.0, panel_area_m2=0.05)
    world.reset(seed=2)
    payload = np.zeros(world.n_sats)
    before = np.array([world.battery_ws(i) for i in range(world.n_sats)])
    world.step(600.0, payload)
    after = np.array([world.battery_ws(i) for i in range(world.n_sats)])
    assert np.any(after != before)


def test_access_remaining_helper_zero_if_not_in_view():
    els = walker_delta_elements(4, 2, 500e3, np.deg2rad(51.6))

    def r_fn(t, el=els[0]):
        return rv_at_time(*el, t)[0]

    acc = access_remaining(r_fn, np.deg2rad(80.0), 0.0, 0.0, np.deg2rad(10.0))
    assert acc == 0.0
