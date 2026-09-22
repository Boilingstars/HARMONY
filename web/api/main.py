"""One loaded shift in memory. Orbits are a Kepler picture, not the shift physics."""

from __future__ import annotations

import json
import math

import numpy as np
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from sim.constants import MU_EARTH, OMEGA_EARTH, R_EARTH
from sim.coverage import eci_to_ecef
from sim.kepler import KeplerWorld
from sim.ops.resource_env import validate

ALTITUDE_M = 500e3
INCLINATION = math.radians(51.6)
ORBIT_SAMPLES = 64
STEP_S = 300

app = FastAPI()
_shift: dict | None = None


def plane_count(n_sats: int) -> int:
    """Divisor of n closest to sqrt(n). Walker-Delta needs n_sats % n_planes == 0."""
    target = max(1, round(math.sqrt(n_sats)))
    divisors = [d for d in range(1, n_sats + 1) if n_sats % d == 0]
    return min(divisors, key=lambda d: (abs(d - target), -d))


def _world_for(n_sats: int) -> KeplerWorld:
    world = KeplerWorld(
        n_sats=n_sats,
        n_planes=plane_count(n_sats),
        altitude_m=ALTITUDE_M,
        inclination=INCLINATION,
    )
    world.reset(seed=0)
    return world


def _lla(world: KeplerWorld, sat_i: int, t: float) -> dict:
    lat, lon, alt = world.lla(sat_i, t)
    return {
        "lat": round(math.degrees(lat), 5),
        "lon": round(math.degrees(lon), 5),
        "alt_m": round(float(alt), 1),
    }


def _batch_lla(world: KeplerWorld, times: np.ndarray) -> np.ndarray:
    """Lat/lon degrees and altitude for every satellite at every time. Shape (T, N, 3)."""
    el = np.asarray(world.elements, dtype=float)
    a, e, inc, raan, argp, m0 = (el[:, i] for i in range(6))
    mean_n = np.sqrt(MU_EARTH / a**3)
    nu = np.mod(m0 + mean_n * times[:, None], 2.0 * np.pi)
    c_o, s_o = np.cos(raan), np.sin(raan)
    ci, si = np.cos(inc), np.sin(inc)
    cw, sw = np.cos(argp), np.sin(argp)
    rot = np.stack(
        [
            np.stack([c_o * cw - s_o * sw * ci, -c_o * sw - s_o * cw * ci, s_o * si], axis=-1),
            np.stack([s_o * cw + c_o * sw * ci, -s_o * sw + c_o * cw * ci, -c_o * si], axis=-1),
            np.stack([sw * si, cw * si, ci], axis=-1),
        ],
        axis=-2,
    )
    p = np.where(e < 1e-12, a, a * (1.0 - e**2))
    radius = p / (1.0 + e * np.cos(nu))
    pos_pqw = np.stack([radius * np.cos(nu), radius * np.sin(nu), np.zeros_like(nu)], axis=-1)
    eci = np.einsum("nij,tnj->tni", rot, pos_pqw)
    theta = world.gmst0 + OMEGA_EARTH * times
    c_t, s_t = np.cos(theta), np.sin(theta)
    x, y, z = eci[..., 0], eci[..., 1], eci[..., 2]
    ex = c_t[:, None] * x + s_t[:, None] * y
    ey = -s_t[:, None] * x + c_t[:, None] * y
    lat = np.degrees(np.arctan2(z, np.hypot(ex, ey)))
    lon = np.degrees(np.arctan2(ey, ex))
    alt = np.sqrt(ex * ex + ey * ey + z * z) - R_EARTH
    return np.stack([lat, lon, alt], axis=-1)


def _sun_rows(world: KeplerWorld, times: np.ndarray) -> np.ndarray:
    sun = np.asarray(world.sun_eci, dtype=float)
    theta = world.gmst0 + OMEGA_EARTH * times
    c_t, s_t = np.cos(theta), np.sin(theta)
    ex = c_t * sun[0] + s_t * sun[1]
    ey = -s_t * sun[0] + c_t * sun[1]
    ez = np.full(times.shape, sun[2])
    vec = np.stack([ex, ey, ez], axis=-1)
    return vec / np.linalg.norm(vec, axis=-1)[:, None]


def _ndjson(payload: dict) -> bytes:
    return (json.dumps(payload, separators=(",", ":")) + "\n").encode()


@app.post("/api/scenario")
def load_scenario(scenario: dict) -> dict:
    global _shift
    try:
        validate(scenario)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    satellites = [sat["id"] for sat in scenario["satellites"]]
    _shift = {
        "scenario": scenario,
        "ids": satellites,
        "world": _world_for(len(satellites)),
    }
    return {
        "scenario_id": scenario["meta"]["id"],
        "satellites": satellites,
        "jobs": [job["id"] for job in scenario["jobs"]],
    }


@app.get("/api/orbits")
def orbits(step: int = 0) -> dict:
    if _shift is None:
        raise HTTPException(status_code=409, detail="Сценарий ещё не загружен")
    steps = int(_shift["scenario"]["time"]["steps"])
    if step < 0 or step > steps:
        raise HTTPException(status_code=400, detail="Шаг вне смены")
    world: KeplerWorld = _shift["world"]
    t = float(step * STEP_S)
    sun = eci_to_ecef(world.sun_eci, t, world.gmst0)
    sun = sun / np.linalg.norm(sun)
    period = float(world.period)
    satellites = []
    for i, sat_id in enumerate(_shift["ids"]):
        polyline = [
            _lla(world, i, t + period * k / ORBIT_SAMPLES)
            for k in range(ORBIT_SAMPLES)
        ]
        satellites.append({"id": sat_id, **_lla(world, i, t), "orbit": polyline})
    return {
        "step": step,
        "t_s": step * STEP_S,
        "sun_ecef": [round(float(v), 5) for v in sun],
        "satellites": satellites,
    }


@app.get("/api/tracks")
def tracks():
    """Stream the whole shift: positions first, orbit lines once."""
    if _shift is None:
        raise HTTPException(status_code=409, detail="Сценарий ещё не загружен")
    world: KeplerWorld = _shift["world"]
    ids = list(_shift["ids"])
    steps = int(_shift["scenario"]["time"]["steps"])
    times = np.arange(steps + 1, dtype=float) * STEP_S
    table = _batch_lla(world, times)
    suns = _sun_rows(world, times)
    period = float(world.period)
    line_times = period * np.arange(ORBIT_SAMPLES, dtype=float) / ORBIT_SAMPLES
    lines = _batch_lla(world, line_times)

    def gen():
        yield _ndjson({"type": "meta", "ids": ids, "steps": steps})
        yield _ndjson(_frame(0, table[0], suns[0]))
        yield _ndjson({
            "type": "orbits",
            "path": [
                [[round(float(p[0]), 5), round(float(p[1]), 5), round(float(p[2]), 1)] for p in lines[:, i, :]]
                for i in range(len(ids))
            ],
        })
        for step in range(1, steps + 1):
            yield _ndjson(_frame(step, table[step], suns[step]))

    return StreamingResponse(
        gen(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _frame(step: int, row: np.ndarray, sun: np.ndarray) -> dict:
    return {
        "type": "frame",
        "step": step,
        "sun": [round(float(v), 5) for v in sun],
        "pos": [[round(float(p[0]), 5), round(float(p[1]), 5), round(float(p[2]), 1)] for p in row],
    }


_WEB = Path(__file__).resolve().parents[1]


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_WEB / "index.html")


app.mount("/static", StaticFiles(directory=_WEB / "static"), name="static")
app.mount("/src", StaticFiles(directory=_WEB / "src"), name="src")
