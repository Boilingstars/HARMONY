"""Observation packing: task + N satellites + global context."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

STATUS_IDLE = 0
STATUS_SLEEP = 1
STATUS_BUSY = 2


@dataclass(frozen=True)
class FeatureSpec:
    n_sats: int
    capability_dim: int = 4
    orbit_period: float = 5700.0
    max_queue: int = 4
    max_attempts: int = 12
    max_duration: float = 720.0
    max_power: float = 45.0
    max_alt: float = 8.0e5
    max_footprint: float = 2.5e6

    @property
    def task_dim(self) -> int:
        return 6 + self.capability_dim

    @property
    def sat_dim(self) -> int:
        # lat, lon, alt, lat_free, lon_free, footprint, battery,
        # status(3), queue, t_free, capability, access, t_sun
        return 14 + self.capability_dim

    @property
    def global_dim(self) -> int:
        return 2

    @property
    def obs_dim(self) -> int:
        return self.task_dim + self.n_sats * self.sat_dim + self.global_dim

    def split(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        obs = np.asarray(obs, dtype=np.float32)
        t0 = self.task_dim
        t1 = t0 + self.n_sats * self.sat_dim
        return obs[:t0], obs[t0:t1].reshape(self.n_sats, self.sat_dim), obs[t1:]


def _ang(lat: float, lon: float) -> tuple[float, float]:
    return float(lat / (0.5 * np.pi)), float(lon / np.pi)


def encode_task(
    spec: FeatureSpec,
    lat: float,
    lon: float,
    duration: float,
    duration_original: float,
    power_need: float,
    attempts: int,
    capability: np.ndarray,
) -> np.ndarray:
    lat_n, lon_n = _ang(lat, lon)
    frac = duration / max(duration_original, 1e-6)
    vec = np.zeros(spec.task_dim, dtype=np.float32)
    vec[0] = lat_n
    vec[1] = lon_n
    vec[2] = np.clip(duration / spec.max_duration, 0.0, 2.0)
    vec[3] = np.clip(frac, 0.0, 1.0)
    vec[4] = np.clip(power_need / spec.max_power, 0.0, 2.0)
    vec[5] = np.clip(attempts / spec.max_attempts, 0.0, 2.0)
    cap = np.asarray(capability, dtype=np.float32).reshape(-1)
    n = min(spec.capability_dim, cap.size)
    vec[6 : 6 + n] = cap[:n]
    return vec


def encode_sat(
    spec: FeatureSpec,
    lat: float,
    lon: float,
    alt: float,
    lat_free: float,
    lon_free: float,
    footprint: float,
    battery: float,
    status: int,
    queue_length: int,
    time_until_free: float,
    capability: np.ndarray,
    access_remaining: float,
    time_to_sun_change: float,
) -> np.ndarray:
    vec = np.zeros(spec.sat_dim, dtype=np.float32)
    lat_n, lon_n = _ang(lat, lon)
    lat_f, lon_f = _ang(lat_free, lon_free)
    vec[0], vec[1], vec[2] = lat_n, lon_n, np.clip(alt / spec.max_alt, 0.0, 2.0)
    vec[3], vec[4] = lat_f, lon_f
    vec[5] = np.clip(footprint / spec.max_footprint, 0.0, 2.0)
    vec[6] = np.clip(battery, 0.0, 1.0)
    status = int(np.clip(status, 0, 2))
    vec[7 + status] = 1.0
    vec[10] = np.clip(queue_length / spec.max_queue, 0.0, 2.0)
    vec[11] = np.clip(time_until_free / spec.orbit_period, 0.0, 2.0)
    cap = np.asarray(capability, dtype=np.float32).reshape(-1)
    n = min(spec.capability_dim, cap.size)
    vec[12 : 12 + n] = cap[:n]
    base = 12 + spec.capability_dim
    vec[base] = np.clip(access_remaining / spec.orbit_period, 0.0, 1.0)
    vec[base + 1] = np.clip(time_to_sun_change / spec.orbit_period, 0.0, 1.0)
    return vec


def access_index(spec: FeatureSpec) -> int:
    return 12 + spec.capability_dim


def pack_obs(task: np.ndarray, sats: np.ndarray, glob: np.ndarray) -> np.ndarray:
    return np.concatenate([task, sats.reshape(-1), glob]).astype(np.float32)
