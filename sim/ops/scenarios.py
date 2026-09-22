"""Пути к сценариям кейса и разбор файла событий."""

from __future__ import annotations

import json
from pathlib import Path

from sim.ops.operations import EVENT_SCHEMA

DATA_DIR = Path(__file__).resolve().parents[2] / "task" / "data"
EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "task" / "examples"

P01_INTRO = DATA_DIR / "P01_intro.json"
P02_SHIFT = DATA_DIR / "P02_shift.json"
P03_ENERGY = DATA_DIR / "P03_energy.json"
P04_DEMAND = DATA_DIR / "P04_demand.json"
EVENTS_DEMO = EXAMPLES_DIR / "events_demo.json"

_ALIASES = {
    "p01": P01_INTRO,
    "p01_intro": P01_INTRO,
    "intro": P01_INTRO,
    "p02": P02_SHIFT,
    "p02_shift": P02_SHIFT,
    "shift": P02_SHIFT,
    "p03": P03_ENERGY,
    "p03_energy": P03_ENERGY,
    "energy": P03_ENERGY,
    "p04": P04_DEMAND,
    "p04_demand": P04_DEMAND,
    "demand": P04_DEMAND,
}


def resolve_scenario(name: str | Path) -> Path:
    """Короткое имя (`p02`, `shift`) или путь к JSON."""
    key = str(name).strip().lower()
    if key in _ALIASES:
        return _ALIASES[key]
    path = Path(name)
    if not path.exists():
        raise FileNotFoundError(f"Сценарий не найден: {name}")
    return path


def load_events(path: str | Path, scenario: dict | None = None) -> list[dict]:
    """Прочитать файл `cosmo-B-events-1.0` и вернуть список событий."""
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if payload.get("schema_version") != EVENT_SCHEMA:
        raise ValueError("Unsupported events schema")
    if scenario is not None and payload.get("base_scenario") != scenario["meta"]["id"]:
        raise ValueError("Events refer to a different base scenario")
    return list(payload["events"])
