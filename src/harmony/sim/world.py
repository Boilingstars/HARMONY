"""Shared world-backend contract for Kepler and Basilisk simulations."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class WorldBackend(Protocol):
    n_sats: int
    t: float
    gmst0: float
    min_elev: float
    battery_capacity_ws: float

    def reset(self, seed: int | None = None) -> None: ...

    def step(self, dt: float, payload_w: np.ndarray) -> None:
        """Advance dynamics; payload_w[i] is extra Watts drawn by sat i."""

    def r_eci(self, sat_i: int, t: float | None = None) -> np.ndarray: ...

    def lla(self, sat_i: int, t: float | None = None) -> tuple[float, float, float]: ...

    def in_sun(self, sat_i: int, t: float | None = None) -> bool: ...

    def elevation(self, sat_i: int, lat: float, lon: float, t: float | None = None) -> float: ...

    def access_remaining(self, sat_i: int, lat: float, lon: float, t_query: float) -> float: ...

    def time_to_sun_change(self, sat_i: int) -> float: ...

    def battery_frac(self, sat_i: int) -> float: ...

    def battery_ws(self, sat_i: int) -> float: ...

    def set_battery_ws(self, sat_i: int, value: float) -> None: ...
