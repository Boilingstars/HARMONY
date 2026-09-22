"""Модель кейса как есть.

`resource_env.py` и `operations.py` скопированы из `task/model/` без правок:
формулы заряда, температуры, калибровки и допуска действий — источник истины,
по ним же сверяется журнал смены. Менять их нельзя, иначе RL разъедется с оценкой.
"""

from sim.ops.operations import EVENT_SCHEMA, RESULT_SCHEMA, Session, digest, replay_episode
from sim.ops.resource_env import Environment, load, validate
from sim.ops.scenarios import (
    DATA_DIR,
    EVENTS_DEMO,
    P01_INTRO,
    P02_SHIFT,
    P03_ENERGY,
    P04_DEMAND,
    load_events,
    resolve_scenario,
)

__all__ = [
    "DATA_DIR",
    "EVENTS_DEMO",
    "EVENT_SCHEMA",
    "Environment",
    "P01_INTRO",
    "P02_SHIFT",
    "P03_ENERGY",
    "P04_DEMAND",
    "RESULT_SCHEMA",
    "Session",
    "digest",
    "load",
    "load_events",
    "replay_episode",
    "resolve_scenario",
    "validate",
]
