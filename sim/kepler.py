"""Analytic constellation: two-body Kepler, cylindrical eclipse, battery ODE."""

from __future__ import annotations

import numpy as np

import sim.coverage as geo
from sim.constants import MU_EARTH, R_EARTH, SOLAR_CONSTANT, orbital_period


class KeplerWorld:
    def __init__(
        self,
        n_sats: int = 4,
        n_planes: int = 2,
        altitude_m: float = 500e3,
        inclination: float = np.deg2rad(51.6),
        min_elev: float = np.deg2rad(10.0),
        battery_capacity_ws: float = 720000.0,
        panel_area_m2: float = 0.25,
        panel_efficiency: float = 0.2,
        solar_constant_w: float = SOLAR_CONSTANT,
        bus_power_w: float = 10.0,
        battery_init_frac: tuple[float, float] = (0.45, 0.95),
        raan_offset: float = 0.0,
    ) -> None:
        self.n_sats = n_sats
        self.n_planes = n_planes
        self.altitude_m = altitude_m
        self.inclination = inclination
        self.min_elev = min_elev
        self.battery_capacity_ws = battery_capacity_ws
        self.panel_area_m2 = panel_area_m2
        self.panel_efficiency = panel_efficiency
        self.solar_constant_w = solar_constant_w
        self.bus_power_w = bus_power_w
        self.battery_init_frac = battery_init_frac
        self.raan_offset = raan_offset

        self.t = 0.0
        self.gmst0 = 0.0
        self.sun_eci = np.array([1.0, 0.0, 0.0])
        self.elements: list[tuple[float, float, float, float, float, float]] = []
        self._battery = np.zeros(n_sats, dtype=float)
        self._rng = np.random.default_rng()
        self._access_cache: dict = {}
        self._sun_cache: dict = {}

    @property
    def semi_major_axis(self) -> float:
        return R_EARTH + self.altitude_m

    @property
    def period(self) -> float:
        return orbital_period(self.semi_major_axis)

    @property
    def p_panel_max(self) -> float:
        return self.panel_area_m2 * self.panel_efficiency * self.solar_constant_w

    def reset(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)
        self.t = 0.0
        self.gmst0 = float(self._rng.uniform(0.0, 2.0 * np.pi))
        lon = float(self._rng.uniform(0.0, 2.0 * np.pi))
        lat = float(np.arcsin(self._rng.uniform(-0.4, 0.4)))
        self.sun_eci = np.array(
            [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
        )
        self.elements = geo.walker_delta_elements(
            self.n_sats,
            self.n_planes,
            self.altitude_m,
            self.inclination,
            phasing=1,
            raan_offset=self.raan_offset + float(self._rng.uniform(0.0, 2.0 * np.pi)),
        )
        lo, hi = self.battery_init_frac
        fracs = self._rng.uniform(lo, hi, size=self.n_sats)
        self._battery = fracs * self.battery_capacity_ws
        self._access_cache = {}
        self._sun_cache = {}

    def r_eci(self, sat_i: int, t: float | None = None) -> np.ndarray:
        t_use = self.t if t is None else t
        a, e, inc, raan, argp, m0 = self.elements[sat_i]
        r, _ = geo.rv_at_time(a, e, inc, raan, argp, m0, t_use, 0.0)
        return r

    def r_ecef(self, sat_i: int, t: float | None = None) -> np.ndarray:
        t_use = self.t if t is None else t
        return geo.eci_to_ecef(self.r_eci(sat_i, t_use), t_use, self.gmst0)

    def lla(self, sat_i: int, t: float | None = None) -> tuple[float, float, float]:
        return geo.ecef_to_lla(self.r_ecef(sat_i, t))

    def in_sun(self, sat_i: int, t: float | None = None) -> bool:
        t_use = self.t if t is None else t
        return geo.in_sunlight(self.r_eci(sat_i, t_use), self.sun_eci)

    def elevation(self, sat_i: int, lat: float, lon: float, t: float | None = None) -> float:
        t_use = self.t if t is None else t
        return geo.elevation(self.r_ecef(sat_i, t_use), geo.lla_to_ecef(lat, lon))

    def access_remaining(self, sat_i: int, lat: float, lon: float, t_query: float) -> float:
        key = (sat_i, round(lat, 5), round(lon, 5), round(float(t_query), 1))
        cached = self._access_cache.get(key)
        if cached is not None:
            return cached
        value = geo.access_remaining(
            lambda tau, i=sat_i: self.r_eci(i, tau),
            lat,
            lon,
            t_query,
            self.min_elev,
            gmst0=self.gmst0,
            max_horizon=0.25 * self.period,
            dt=8.0,
        )
        self._access_cache[key] = value
        if len(self._access_cache) > 4096:
            self._access_cache.clear()
        return value

    def time_to_sun_change(self, sat_i: int) -> float:
        key = (sat_i, round(self.t, 1))
        cached = self._sun_cache.get(key)
        if cached is not None:
            return cached
        lit = self.in_sun(sat_i)
        value = geo.time_to_sun_change(
            lambda tau, i=sat_i: self.r_eci(i, tau),
            self.sun_eci,
            self.t,
            lit,
            max_horizon=self.period,
            dt=20.0,
        )
        self._sun_cache[key] = value
        if len(self._sun_cache) > 512:
            self._sun_cache.clear()
        return value

    def battery_frac(self, sat_i: int) -> float:
        return float(np.clip(self._battery[sat_i] / self.battery_capacity_ws, 0.0, 1.0))

    def battery_ws(self, sat_i: int) -> float:
        return float(self._battery[sat_i])

    def set_battery_ws(self, sat_i: int, value: float) -> None:
        self._battery[sat_i] = float(np.clip(value, 0.0, self.battery_capacity_ws))

    def step(self, dt: float, payload_w: np.ndarray) -> None:
        if dt <= 0.0:
            return
        payload_w = np.asarray(payload_w, dtype=float)
        for i in range(self.n_sats):
            p_in = self.p_panel_max if self.in_sun(i) else 0.0
            p_out = self.bus_power_w + float(payload_w[i])
            self._battery[i] = float(
                np.clip(self._battery[i] + (p_in - p_out) * dt, 0.0, self.battery_capacity_ws)
            )
        self.t += float(dt)


def kepler_from_config(cfg: dict) -> KeplerWorld:
    lo, hi = cfg.get("battery_init_frac", [0.45, 0.95])
    return KeplerWorld(
        n_sats=int(cfg["n_sats"]),
        n_planes=int(cfg["n_planes"]),
        altitude_m=float(cfg["altitude_km"]) * 1e3,
        inclination=np.deg2rad(float(cfg["inclination_deg"])),
        min_elev=np.deg2rad(float(cfg["min_elevation_deg"])),
        battery_capacity_ws=float(cfg["battery_capacity_ws"]),
        panel_area_m2=float(cfg["panel_area_m2"]),
        panel_efficiency=float(cfg["panel_efficiency"]),
        solar_constant_w=float(cfg.get("solar_constant_w", SOLAR_CONSTANT)),
        bus_power_w=float(cfg["bus_power_w"]),
        battery_init_frac=(float(lo), float(hi)),
        raan_offset=np.deg2rad(float(cfg.get("raan_offset_deg", 0.0))),
    )
