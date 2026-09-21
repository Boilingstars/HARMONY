"""Сравнение подходов HARMONY на одном мире.

Сюда сводим greedy, автономный CNP и RL (если есть сохранённая модель).
Таблицы, графики и ролики визуализации кладём в figures/.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from harmony.agent.evaluate import evaluate
from harmony.sim.basilisk_world import basilisk_available

FIGURES = ROOT / "figures"


def run_greedy(episodes: int = 4, seed: int = 0) -> dict:
    return evaluate(
        n_episodes=episodes,
        backend="kepler",
        policy="greedy",
        seed=seed,
    )


def run_cnp(seed: int = 0) -> dict:
    from autonomous.autonomous import run_cnp as cnp_run
    import autonomous.autonomous as auto

    auto.ENABLE_VIZARD = False
    return cnp_run(seed=seed)


def run_rl(episodes: int = 4, seed: int = 0) -> dict | None:
    model = ROOT / "models" / "pointer_ppo.zip"
    if not model.exists():
        return None
    return evaluate(
        n_episodes=episodes,
        backend="kepler",
        model_path=str(model),
        seed=seed,
    )


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    report = {
        "when": datetime.now(timezone.utc).isoformat(),
        "basilisk_installed": basilisk_available(),
        "greedy": run_greedy(),
        "cnp": run_cnp(),
        "rl": run_rl(),
    }
    out = FIGURES / "comparison.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nСохранено: {out}")
    print("Ролики и картинки кладите рядом в figures/.")


if __name__ == "__main__":
    main()
