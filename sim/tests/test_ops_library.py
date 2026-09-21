"""sim/ops — копия модели организаторов. Формулы править нельзя."""

from __future__ import annotations

from pathlib import Path

import pytest

from sim.ops import P01_INTRO, P02_SHIFT, Environment, Session, load, validate
from sim.ops.scenarios import DATA_DIR, EVENTS_DEMO, load_events, resolve_scenario

_ROOT = Path(__file__).resolve().parents[2]
_ORIGINAL = _ROOT / "Кейс" / "model"
_COPIED = _ROOT / "sim" / "ops"


@pytest.mark.parametrize("name", ["resource_env.py", "operations.py"])
def test_copy_is_byte_identical_to_the_case_model(name: str):
    """Журнал смены сверяется с моделью кейса — расхождение недопустимо."""
    assert _COPIED.joinpath(name).read_bytes() == _ORIGINAL.joinpath(name).read_bytes()


def test_all_scenarios_validate():
    for path in sorted(DATA_DIR.glob("P0*.json")):
        scenario = load(path)
        validate(scenario)
        assert scenario["time"]["step_s"] == 300


def test_resolve_scenario_accepts_alias_and_path():
    assert resolve_scenario("p02") == P02_SHIFT
    assert resolve_scenario("shift") == P02_SHIFT
    assert resolve_scenario(str(P01_INTRO)) == P01_INTRO
    with pytest.raises(FileNotFoundError):
        resolve_scenario("no_such_scenario")


def test_events_demo_belongs_to_p02():
    scenario = load(P02_SHIFT)
    events = load_events(EVENTS_DEMO, scenario)
    assert [e["id"] for e in events] == ["E-01", "E-02", "E-03", "E-04"]
    with pytest.raises(ValueError):
        load_events(EVENTS_DEMO, load(P01_INTRO))


def test_session_advance_matches_environment_step():
    scenario = load(P01_INTRO)
    session = Session(scenario)
    direct = Environment(scenario)
    for _ in range(5):
        assert session.advance({}) == direct.step({})
    assert session.summary()["steps_executed"] == direct.summary()["steps_executed"]
