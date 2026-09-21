"""Basilisk constellation backend. Imported lazily; Kepler remains the trainer."""

from __future__ import annotations

import numpy as np

import sim.coverage as geo
from sim.constants import AU, R_EARTH, SOLAR_CONSTANT, orbital_period
from sim.kepler import KeplerWorld


def basilisk_available() -> bool:
    try:
        import Basilisk  # noqa: F401

        return True
    except ImportError:
        return False


class BasiliskUnavailableError(ImportError):
    pass


class BasiliskWorld:
    """
    N spacecraft, shared Earth gravity + eclipse, per-sat panel/battery/sink.

    Access geometry is computed from spacecraft states with the same coverage
    helpers as KeplerWorld. Future-time queries use two-body elements matching
    the initial conditions (Basilisk is also two-body if J2 is off).
    """

    def __init__(self, **kwargs) -> None:
        if not basilisk_available():
            raise BasiliskUnavailableError(
                "Basilisk is not installed. Use backend='kepler' or `pip install bsk`."
            )
        self._kw = kwargs
        self._analytic = KeplerWorld(**kwargs)
        self.n_sats = self._analytic.n_sats
        self.min_elev = self._analytic.min_elev
        self.battery_capacity_ws = self._analytic.battery_capacity_ws
        self.t = 0.0
        self.gmst0 = 0.0
        self.sun_eci = np.array([AU, 0.0, 0.0])
        self.elements = []
        self._sim = None
        self._scs = []
        self._bats = []
        self._payload_msgs = []
        self._eclipses = []
        self._macros = None
        self._stop_ns = 0

    @property
    def period(self) -> float:
        return orbital_period(R_EARTH + self._analytic.altitude_m)

    def reset(self, seed: int | None = None) -> None:
        self._analytic.reset(seed)
        self.t = 0.0
        self.gmst0 = self._analytic.gmst0
        self.sun_eci = self._analytic.sun_eci * AU
        self.elements = list(self._analytic.elements)
        self._build_sim()

    def _build_sim(self) -> None:
        from Basilisk.architecture import messaging
        from Basilisk.simulation import (
            eclipse,
            simpleBattery,
            simplePowerSink,
            simpleSolarPanel,
            spacecraft,
        )
        from Basilisk.utilities import SimulationBaseClass, macros, simIncludeGravBody

        self._macros = macros
        dt_ns = macros.sec2nano(1.0)
        sim = SimulationBaseClass.SimBaseClass()
        sim.SetProgressBar(False)
        sim.CreateNewProcess("dyn")
        sim.CreateNewTask("dynTask", dt_ns)

        grav_factory = simIncludeGravBody.gravBodyFactory()
        earth = grav_factory.createEarth()
        earth.isCentralBody = True

        sun_pl = messaging.SpicePlanetStateMsgPayload()
        sun_pl.PositionVector = [float(x) for x in self.sun_eci]
        sun_pl.VelocityVector = [0.0, 0.0, 0.0]
        sun_pl.PlanetName = "sun"
        sun_msg = messaging.SpicePlanetStateMsg()
        sun_msg.write(sun_pl)

        earth_pl = messaging.SpicePlanetStateMsgPayload()
        earth_pl.PositionVector = [0.0, 0.0, 0.0]
        earth_pl.PlanetName = "earth"
        earth_msg = messaging.SpicePlanetStateMsg()
        earth_msg.write(earth_pl)

        ecl = eclipse.Eclipse()
        ecl.ModelTag = "eclipse"
        ecl.sunInMsg.subscribeTo(sun_msg)
        ecl.addPlanetToModel(earth_msg)

        self._scs = []
        self._bats = []
        self._payload_msgs = []
        self._eclipses = []

        for i, el in enumerate(self.elements):
            a, e, inc, raan, argp, m0 = el
            r_n, v_n = geo.rv_at_time(a, e, inc, raan, argp, m0, 0.0, 0.0)
            sc = spacecraft.Spacecraft()
            sc.ModelTag = f"sat{i}"
            sc.hub.mHub = 330.0
            sc.hub.IHubPntBc_B = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
            sc.hub.r_CN_NInit = r_n.tolist()
            sc.hub.v_CN_NInit = v_n.tolist()
            self._attach_gravity(grav_factory, sc, earth)
            sim.AddModelToTask("dynTask", sc, ModelPriority=2000 - i)

            ecl.addSpacecraftToModel(sc.scStateOutMsg)

            panel = simpleSolarPanel.SimpleSolarPanel()
            panel.ModelTag = f"panel{i}"
            panel.stateInMsg.subscribeTo(sc.scStateOutMsg)
            panel.sunInMsg.subscribeTo(sun_msg)
            panel.setPanelParameters(
                [0.0, 0.0, 1.0],
                self._analytic.panel_area_m2,
                self._analytic.panel_efficiency,
            )
            try:
                panel.sunEclipseInMsg.subscribeTo(ecl.eclipseOutMsgs[-1])
            except Exception:
                pass
            sim.AddModelToTask("dynTask", panel, ModelPriority=1500 - i)

            bus = simplePowerSink.SimplePowerSink()
            bus.ModelTag = f"bus{i}"
            bus.nodePowerOut = -float(self._analytic.bus_power_w)

            payload_pl = messaging.PowerNodeUsageMsgPayload()
            payload_pl.netPower = 0.0
            payload_msg = messaging.PowerNodeUsageMsg()
            payload_msg.write(payload_pl)

            bat = simpleBattery.SimpleBattery()
            bat.ModelTag = f"bat{i}"
            bat.storageCapacity = self.battery_capacity_ws
            bat.storedCharge_Init = float(self._analytic.battery_ws(i))
            bat.addPowerNodeToModel(panel.nodePowerOutMsg)
            bat.addPowerNodeToModel(bus.nodePowerOutMsg)
            bat.addPowerNodeToModel(payload_msg)
            sim.AddModelToTask("dynTask", bus, ModelPriority=1400 - i)
            sim.AddModelToTask("dynTask", bat, ModelPriority=1300 - i)

            self._scs.append(sc)
            self._bats.append(bat)
            self._payload_msgs.append((payload_msg, messaging))

        self._eclipses = [ecl]
        sim.AddModelToTask("dynTask", ecl, ModelPriority=1800)
        sim.InitializeSimulation()
        self._sim = sim
        self._stop_ns = 0
        self._sync_from_bsk()

    def _attach_gravity(self, grav_factory, sc, earth) -> None:
        if hasattr(grav_factory, "addBodiesToSpacecraft"):
            grav_factory.addBodiesToSpacecraft(sc)
            return
        try:
            from Basilisk.simulation import spacecraft as sc_mod

            bodies = [earth]
            if hasattr(sc.gravField, "gravBodies"):
                sc.gravField.gravBodies = sc_mod.GravBodyVector(bodies)
        except Exception:
            if hasattr(earth, "addSpacecraft"):
                earth.addSpacecraft(sc)

    def step(self, dt: float, payload_w: np.ndarray) -> None:
        if dt <= 0.0:
            return
        from Basilisk.architecture import messaging

        payload_w = np.asarray(payload_w, dtype=float)
        for i, (msg, _) in enumerate(self._payload_msgs):
            pl = messaging.PowerNodeUsageMsgPayload()
            pl.netPower = -float(payload_w[i])
            msg.write(pl)
        self._stop_ns += int(self._macros.sec2nano(float(dt)))
        self._sim.ConfigureStopTime(self._stop_ns)
        self._sim.ExecuteSimulation()
        self.t += float(dt)
        self._analytic.t = self.t
        self._sync_from_bsk()

    def _sync_from_bsk(self) -> None:
        for i, bat in enumerate(self._bats):
            try:
                level = float(bat.batPowerOutMsg.read().storageLevel)
            except Exception:
                level = float(self._analytic.battery_ws(i))
            self._analytic.set_battery_ws(i, level)

    def r_eci(self, sat_i: int, t: float | None = None) -> np.ndarray:
        if t is None or abs(t - self.t) < 1e-9:
            try:
                return np.array(self._scs[sat_i].scStateOutMsg.read().r_BN_N, dtype=float)
            except Exception:
                pass
        return self._analytic.r_eci(sat_i, self.t if t is None else t)

    def lla(self, sat_i: int, t: float | None = None) -> tuple[float, float, float]:
        t_use = self.t if t is None else t
        r_ecef = geo.eci_to_ecef(self.r_eci(sat_i, t_use), t_use, self.gmst0)
        return geo.ecef_to_lla(r_ecef)

    def in_sun(self, sat_i: int, t: float | None = None) -> bool:
        if (t is None or abs(t - self.t) < 1e-9) and self._eclipses:
            try:
                factor = float(self._eclipses[0].eclipseOutMsgs[sat_i].read().shadowFactor)
                return factor > 0.05
            except Exception:
                pass
        return self._analytic.in_sun(sat_i, t)

    def elevation(self, sat_i: int, lat: float, lon: float, t: float | None = None) -> float:
        t_use = self.t if t is None else t
        r_ecef = geo.eci_to_ecef(self.r_eci(sat_i, t_use), t_use, self.gmst0)
        return geo.elevation(r_ecef, geo.lla_to_ecef(lat, lon))

    def access_remaining(self, sat_i: int, lat: float, lon: float, t_query: float) -> float:
        return self._analytic.access_remaining(sat_i, lat, lon, t_query)

    def time_to_sun_change(self, sat_i: int) -> float:
        return self._analytic.time_to_sun_change(sat_i)

    def battery_frac(self, sat_i: int) -> float:
        return self._analytic.battery_frac(sat_i)

    def battery_ws(self, sat_i: int) -> float:
        return self._analytic.battery_ws(sat_i)

    def set_battery_ws(self, sat_i: int, value: float) -> None:
        self._analytic.set_battery_ws(sat_i, value)


def basilisk_from_config(cfg: dict) -> BasiliskWorld:
    lo, hi = cfg.get("battery_init_frac", [0.45, 0.95])
    return BasiliskWorld(
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
