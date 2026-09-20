#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from harmony.agent.train import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the constellation dispatcher")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--backend", default="kepler", choices=["kepler", "basilisk"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--save", default=str(ROOT / "models" / "pointer_ppo.zip"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train(
        timesteps=args.timesteps,
        backend=args.backend,
        config_path=args.config,
        save_path=args.save,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
