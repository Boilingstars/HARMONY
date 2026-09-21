"""Shared physics: Kepler trainer, optional Basilisk validation, world snapshots."""

from pathlib import Path

from sim.condition import SatCondition, TaskCondition, open_world, sat_condition, world_heartbeat
from sim.kepler import KeplerWorld
from sim.world import WorldBackend

CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "default.yaml"

__all__ = [
    "CONFIG_PATH",
    "KeplerWorld",
    "WorldBackend",
    "SatCondition",
    "TaskCondition",
    "open_world",
    "sat_condition",
    "world_heartbeat",
]
