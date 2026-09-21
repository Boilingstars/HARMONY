"""Run one episode on Basilisk if installed, otherwise KeplerWorld."""

from __future__ import annotations

import argparse
import json

from agent.env.task_assignment_env import TaskAssignmentEnv, load_config
from agent.legacy_kepler import greedy_action, run_episode
from sim.basilisk_world import basilisk_available


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
    keys = (
        "return",
        "steps",
        "n_complete",
        "n_handoffs",
        "n_defers",
        "n_attaches",
        "work_s",
        "t",
        "stack",
    )
    print(json.dumps({k: info[k] for k in info if k in keys}, indent=2))


if __name__ == "__main__":
    main()
