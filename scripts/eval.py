#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from harmony.agent.evaluate import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate dispatcher policy")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--backend", default="kepler", choices=["kepler", "basilisk"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--policy", default="greedy", choices=["greedy", "random"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    summary = evaluate(
        n_episodes=args.episodes,
        backend=args.backend,
        config_path=args.config,
        model_path=args.model,
        policy=args.policy,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
