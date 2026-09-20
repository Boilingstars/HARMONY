"""Ground-track geometry: LLA, elevation, footprint, access remaining."""

from __future__ import annotations

import numpy as np

from harmony.sim.constants import MU_EARTH, OMEGA_EARTH, R_EARTH


def rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def eci_to_ecef(r_eci: np.ndarray, t: float, gmst0: float = 0.0) -> np.ndarray:
    return rot_z(gmst0 + OMEGA_EARTH * t) @ np.asarray(r_eci, dtype=float)


def ecef_to_eci(r_ecef: np.ndarray, t: float, gmst0: float = 0.0) -> np.ndarray:
    return rot_z(gmst0 + OMEGA_EARTH * t).T @ np.asarray(r_ecef, dtype=float)


def ecef_to_lla(r_ecef: np.ndarray) -> tuple[float, float, float]:
    x, y, z = np.asarray(r_ecef, dtype=float)
    lon = np.arctan2(y, x)
    hyp = np.hypot(x, y)
    lat = np.arctan2(z, hyp)
    alt = np.linalg.norm(r_ecef) - R_EARTH
    return float(lat), float(lon), float(alt)


def lla_to_ecef(lat: float, lon: float, alt: float = 0.0) -> np.ndarray:
    r = R_EARTH + alt
    clat, slat = np.cos(lat), np.sin(lat)
    clon, slon = np.cos(lon), np.sin(lon)
    return np.array([r * clat * clon, r * clat * slon, r * slat], dtype=float)


def elevation(r_sat_ecef: np.ndarray, r_tgt_ecef: np.ndarray) -> float:
    """Elevation of the satellite above the local horizon at the target, radians."""
    rho = np.asarray(r_sat_ecef, dtype=float) - np.asarray(r_tgt_ecef, dtype=float)
    up = np.asarray(r_tgt_ecef, dtype=float)
    up_norm = np.linalg.norm(up)
    rho_norm = np.linalg.norm(rho)
    if rho_norm < 1.0 or up_norm < 1.0:
        return -np.pi / 2.0
    sine = np.clip(np.dot(rho, up) / (rho_norm * up_norm), -1.0, 1.0)
    return float(np.arcsin(sine))


def earth_central_angle(alt_m: float, min_elev: float) -> float:
    r = R_EARTH + max(alt_m, 1.0)
    arg = np.clip((R_EARTH / r) * np.cos(min_elev), -1.0, 1.0)
    return float(np.arccos(arg) - min_elev)


def footprint_radius(alt_m: float, min_elev: float) -> float:
    """Approximate geodetic coverage radius along the surface, metres."""
    return R_EARTH * earth_central_angle(alt_m, min_elev)


def koe_to_rv(
    a: float,
    e: float,
    inc: float,
    raan: float,
    argp: float,
    nu: float,
    mu: float = MU_EARTH,
) -> tuple[np.ndarray, np.ndarray]:
    """Classical orbital elements to ECI position/velocity."""
    p = a * (1.0 - e**2) if e < 0.999 else a
    if e < 1e-12:
        p = a
    r_pqw = p / (1.0 + e * np.cos(nu))
    pos_pqw = np.array([r_pqw * np.cos(nu), r_pqw * np.sin(nu), 0.0])
    vel_pqw = np.sqrt(mu / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])
    cO, sO = np.cos(raan), np.sin(raan)
    ci, si = np.cos(inc), np.sin(inc)
    cw, sw = np.cos(argp), np.sin(argp)
    rot = np.array(
        [
            [cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci, sO * si],
            [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
            [sw * si, cw * si, ci],
        ]
    )
    return rot @ pos_pqw, rot @ vel_pqw


def true_anomaly_from_mean(M: float, e: float, iters: int = 12) -> float:
    if e < 1e-12:
        return float(np.mod(M, 2.0 * np.pi))
    E = M if e < 0.8 else np.pi
    for _ in range(iters):
        E = E - (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
    sin_nu = (np.sqrt(1.0 - e**2) * np.sin(E)) / (1.0 - e * np.cos(E))
    cos_nu = (np.cos(E) - e) / (1.0 - e * np.cos(E))
    return float(np.arctan2(sin_nu, cos_nu))


def rv_at_time(
    a: float,
    e: float,
    inc: float,
    raan: float,
    argp: float,
    m0: float,
    t: float,
    t0: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    n = np.sqrt(MU_EARTH / a**3)
    M = m0 + n * (t - t0)
    nu = true_anomaly_from_mean(M, e)
    return koe_to_rv(a, e, inc, raan, argp, nu)


def walker_delta_elements(
    n_sats: int,
    n_planes: int,
    altitude_m: float,
    inclination: float,
    phasing: int = 1,
    raan_offset: float = 0.0,
) -> list[tuple[float, float, float, float, float, float]]:
    """Return list of (a, e, i, RAAN, argp, M0) for a Walker-Delta constellation."""
    if n_sats % n_planes != 0:
        raise ValueError("n_sats must be divisible by n_planes")
    per_plane = n_sats // n_planes
    a = R_EARTH + altitude_m
    elements = []
    for p in range(n_planes):
        raan = raan_offset + 2.0 * np.pi * p / n_planes
        for k in range(per_plane):
            m0 = 2.0 * np.pi * k / per_plane + 2.0 * np.pi * phasing * p / n_sats
            elements.append((a, 0.0, inclination, raan, 0.0, float(np.mod(m0, 2.0 * np.pi))))
    return elements


def in_sunlight(r_eci: np.ndarray, sun_eci: np.ndarray, radius: float = R_EARTH) -> bool:
    """Cylindrical eclipse: False when the spacecraft is in Earth's shadow."""
    s = np.asarray(sun_eci, dtype=float)
    s_norm = np.linalg.norm(s)
    if s_norm < 1.0:
        return True
    s_hat = s / s_norm
    r = np.asarray(r_eci, dtype=float)
    if np.dot(r, s_hat) >= 0.0:
        return True
    perp = r - np.dot(r, s_hat) * s_hat
    return float(np.linalg.norm(perp)) > radius


def access_remaining(
    r_sat_fn,
    lat: float,
    lon: float,
    t0: float,
    min_elev: float,
    gmst0: float = 0.0,
    max_horizon: float = 900.0,
    dt: float = 10.0,
) -> float:
    """
    Seconds the target stays in view after t0.

    r_sat_fn(t) -> ECI position. Returns 0 if the target is not in view at t0.
    """
    tgt = lla_to_ecef(lat, lon, 0.0)

    def elev_at(t: float) -> float:
        r_ecef = eci_to_ecef(r_sat_fn(t), t, gmst0)
        return elevation(r_ecef, tgt)

    if elev_at(t0) < min_elev:
        return 0.0

    t = t0
    t_limit = t0 + max_horizon
    while t < t_limit:
        t_next = min(t + dt, t_limit)
        if elev_at(t_next) < min_elev:
            lo, hi = t, t_next
            for _ in range(18):
                mid = 0.5 * (lo + hi)
                if elev_at(mid) >= min_elev:
                    lo = mid
                else:
                    hi = mid
            return max(0.0, lo - t0)
        t = t_next
    return max_horizon


def time_to_sun_change(
    r_sat_fn,
    sun_eci: np.ndarray,
    t0: float,
    currently_in_sun: bool,
    max_horizon: float = 6000.0,
    dt: float = 20.0,
) -> float:
    t = t0
    t_limit = t0 + max_horizon
    while t < t_limit:
        t_next = min(t + dt, t_limit)
        lit = in_sunlight(r_sat_fn(t_next), sun_eci)
        if lit != currently_in_sun:
            lo, hi = t, t_next
            for _ in range(16):
                mid = 0.5 * (lo + hi)
                if in_sunlight(r_sat_fn(mid), sun_eci) == currently_in_sun:
                    lo = mid
                else:
                    hi = mid
            return max(0.0, hi - t0)
        t = t_next
    return max_horizon
