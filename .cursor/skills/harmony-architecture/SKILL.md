---
name: harmony-architecture
description: >-
  HARMONY project layout and shared simulation contract. Use when editing
  sim/, sim/ops, agent/, web/, OpsEnv, the attention policy, Basilisk/Kepler
  worlds, coverage, condition snapshots, rewards, telemetry, or RL assignment.
---

# HARMONY architecture

One RL dispatcher over the organizers' model. `web/` is an empty placeholder for later browser visualization.

## Layout

- `sim/ops/` — **verbatim copies** of `Кейс/model/resource_env.py` and `operations.py` plus `scenarios.py` (paths, event loading)
- `sim/` — Kepler (future web, scenario generation), Basilisk (validate), coverage, condition
- `agent/` — Gym env on `Session`, attention policy, greedy, train/eval/baseline, telemetry, plots
- `web/` — empty; no server yet

## Hard rules

- **Never edit the formulas in `sim/ops/resource_env.py` or `operations.py`.** They are the source of truth for energy, temperature, calibration, command admissibility and the shift journal; the case evaluation replays against them.
- Kepler/Basilisk must not feed charge or contact windows. `solar_w`, `downlink_available`, `relay_available` come from the scenario JSON.
- `Environment.observation()` and `Session.fork()` deep-copy; in hot loops read `env.state` / `env.jobs` directly instead.
- Events must be applied at `env.k` **before** that step's action. `apply_event` runs full `validate()` on a deep copy — it is expensive, keep probabilities modest.

## OpsEnv contract (`agent/env/ops_env.py`)

- One step = one 5-minute tick. `MultiDiscrete([K + 2] * N_max)`: slots `0..K-1` are job candidates, `K` is idle, `K + 1` is calibrate.
- Dict-obs keys: `sats 48x14`, `jobs 32x13`, `plan 8192x13`, `edges 48x32x5`, `global 8`, `mask 48x34`.
- `plan` holds every unfinished job from step 0, including those not yet released. Action slots stay top-32; a future job is visible but its mask is off until `can_execute`.
- Training noise does not emit `add_jobs` (the draw is a skip). Operator and scripted `add_jobs` still go through `apply_event`.
- **Padding rows sats/edges stay exactly zero.** Zero `capacity` (column 11) is how the policy tells padding from a real satellite. Do not fill padding rows.
- Idle must stay valid in every mask row, including padding, or the categorical distribution gets no valid class.
- Repair runs before `Session.advance`: unique job per tick, at most `downlink_parallel_limit` downlinks. Commands sent must never be rejected, so `blocked_command_count` stays 0.
- `goal` changes reward accrual only, never physics.

## Policy contract (`agent/policy/ops_policy.py`)

- Weights must not depend on `N` or `K` — one model trains on P02 (48 sats) and plays on P01 (16 sats).
- Attention masks are `(B, 1, L, S)` for `F.scaled_dot_product_attention`; do not expand to `B * heads`.
- Degenerate rows (no valid keys) must be opened and then gated to zero, otherwise softmax returns NaN.
- `--profile full` is the shipped architecture; `light` is only for smoke runs on a weak CPU.

## Telemetry

- Run dir `agent/runs/<stamp>_<scenario>_<goal>/` with `train.log`, `progress.csv`, `config.json`, optional `tb/`.
- Domain metrics come from `Session.summary()` via `OpsMetricsCallback`, never recomputed by hand.

## Legacy

`agent/env/task_assignment_env.py`, `agent/policy/pointer_policy.py`, `agent/legacy_kepler.py`, `agent/demo_basilisk.py` are the old attach/defer Kepler loop. Kept for the Basilisk check, not the training contour.

## Commands

```bash
py -3 -m pytest -q
py -3 -m agent.train --scenario p02 --goal priority --timesteps 200000
py -3 -m agent.plots --run agent/runs/<run>
py -3 -m agent.evaluate --scenario p02 --policy ppo --model agent/models/ops_priority.zip --journal agent/runs/eval
py -3 -m agent.baseline --scenario p02
```

User-facing docs: `README.md`.
