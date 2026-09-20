#!/usr/bin/env python3
"""Run one episode on Basilisk if installed, otherwise KeplerWorld."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from harmony.agent.evaluate import run_episode
from harmony.baselines.greedy import greedy_action
from harmony.env.task_assignment_env import TaskAssignmentEnv, load_config
from harmony.sim.basilisk_world import basilisk_available


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    backend = "basilisk" if basilisk_available() else "kepler"
    print(f"backend={backend} (Basilisk installed={basilisk_available()})")
    cfg = load_config(args.config)
    env = TaskAssignmentEnv(config=cfg, backend=backend)
    info = run_episode(
        env,
        lambda obs, masks: greedy_action(obs, masks, env.feat_spec),
        seed=args.seed,
    )
    print(json.dumps({k: info[k] for k in info if k in (
        "return", "steps", "n_complete", "n_handoffs", "n_defers", "n_attaches", "work_s", "t", "stack"
    )}, indent=2))


if __name__ == "__main__":
    main()
