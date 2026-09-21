"""Автономное распределение задач по спутникам (CNP), без нейросети.

Физика мира та же, что у RL-агента HARMONY: орбита, покрытие цели,
заряд батареи, тень. Здесь вместо сети работает абстрактный менеджер
и заявки спутников.
"""

from __future__ import annotations

import math
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

# Скрипт в autonomous/ — корень репозитория на уровень выше.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harmony.env.task_assignment_env import load_config
from harmony.sim.condition import TaskCondition, open_world, sat_condition, world_heartbeat
from harmony.sim.coverage import ecef_to_lla

random.seed(42)

# Включите True, если рядом запущен Vizard. Без него симуляция всё равно считается.
ENABLE_VIZARD = False

# Кольцо соседей из sat_comms — отдельный модуль тиммейтов, его может не быть.
try:
    from sat_comms import SatelliteComms
except ImportError:
    SatelliteComms = None


def make_named_tasks() -> list[TaskCondition]:
    """Те же 12 сюжетов, что были у тиммейтов.

    Раньше цель задавалась точкой в метрах (x, y, z). Теперь это широта и
    долгота на Земле, плюс сколько секунд работать и сколько ватт брать
    с батареи — как в RL-среде.
    """
    raw = [
        {"name": "Снять паводок", "pos": [5000000.0, 4000000.0, 0.0]},
        {"name": "Передать данные", "pos": [6000000.0, 3000000.0, 0.0]},
        {"name": "Сфотографировать лес", "pos": [4000000.0, 5000000.0, 0.0]},
        {"name": "Снять разлив", "pos": [5500000.0, 4500000.0, 0.0]},
        {"name": "Мониторинг озера", "pos": [6500000.0, 2500000.0, 0.0]},
        {"name": "Снять наводнение", "pos": [5000000.0, 3000000.0, 2000000.0]},
        {"name": "Сфотографировать горы", "pos": [4000000.0, 4000000.0, 3000000.0]},
        {"name": "Мониторинг реки", "pos": [6000000.0, 2000000.0, 1500000.0]},
        {"name": "Снять тайгу", "pos": [3500000.0, 5500000.0, 2500000.0]},
        {"name": "Снять побережье", "pos": [4500000.0, 3500000.0, -2000000.0]},
        {"name": "Мониторинг болот", "pos": [5500000.0, 2500000.0, -1500000.0]},
        {"name": "Снять ледник", "pos": [3000000.0, 4500000.0, 3000000.0]},
    ]
    tasks = []
    for item in raw:
        lat, lon, _ = ecef_to_lla(np.asarray(item["pos"], dtype=float))
        tasks.append(
            TaskCondition(
                name=item["name"],
                lat=lat,
                lon=lon,
                duration=240.0,
                duration_original=240.0,
                power_need=30.0,
            )
        )
    return tasks


def chance_from_condition(cond) -> float:
    """Оценка заявки: не расстояние по прямой, а «видит ли цель и надолго ли».

    access_remaining — сколько секунд цель ещё будет в зоне покрытия,
    когда спутник освободится. battery — запас энергии (0…1).
    """
    cover = min(cond.access_remaining / 400.0, 1.0)
    sun_bonus = 0.2 if cond.in_sun else 0.0
    # Штраф, если сейчас цель не в зоне, но окно появится после текущей работы.
    wait_penalty = 0.0 if cond.in_view_now else 0.1
    return 0.45 * cover + 0.35 * cond.battery + sun_bonus - wait_penalty


def maybe_enable_vizard(world) -> None:
    if not ENABLE_VIZARD:
        return
    handles = world.sim_handles() if hasattr(world, "sim_handles") else None
    if not handles:
        print("Vizard: нет Basilisk-симулятора, пропускаем.")
        return
    try:
        from Basilisk.utilities import vizSupport

        viz = vizSupport.enableUnityVisualization(
            handles["sim"],
            handles["task_name"],
            handles["spacecraft"],
            liveStream=True,
        )
        viz.reqComAddress = "127.0.0.1"
        viz.reqPortNumber = "5556"
        print("Vizard: стрим на 127.0.0.1:5556")
    except Exception as exc:
        print(f"Vizard не запущен ({exc}). CNP продолжается без картинки.")


def maybe_wire_comms(world) -> None:
    """Кольцо соседних аппаратов. Если модуля sat_comms нет — просто не подключаем."""
    if SatelliteComms is None:
        return
    handles = world.sim_handles() if hasattr(world, "sim_handles") else None
    if not handles:
        return
    sats = handles["spacecraft"]
    sim = handles["sim"]
    mods = []
    for i, sc in enumerate(sats):
        m = SatelliteComms.SatelliteComms()
        m.ModelTag = f"Comms-{i + 1}"
        m.satId = i + 1
        sim.AddModelToTask(handles["task_name"], m)
        m.selfStateInMsg.subscribeTo(sc.scStateOutMsg)
        mods.append(m)
    for i, m in enumerate(mods):
        nxt = sats[(i + 1) % len(sats)]
        m.neighborStateInMsg.subscribeTo(nxt.scStateOutMsg)


def run_cnp(seed: int = 42) -> dict:
    random.seed(seed)
    cfg = load_config()
    # Один и тот же мир, что у RL: Walker-Δ, батарея, покрытие. Не собираем Basilisk вручную.
    world, backend = open_world(cfg, prefer_basilisk=True)
    maybe_wire_comms(world)
    maybe_enable_vizard(world)

    print()
    print(f"=== HEARTBEAT (мир={backend}) ===")
    for row in world_heartbeat(world):
        sun = "на солнце" if row.in_sun else "в тени"
        print(
            f"[Sat-{row.sat_i + 1}] заряд={row.battery_wh:.1f} Вт*ч, "
            f"широта={row.lat_deg:.2f}°, долгота={row.lon_deg:.2f}°, "
            f"высота={row.alt_km:.1f} км, {sun}"
        )
    print()

    stack = deque(make_named_tasks())
    n = world.n_sats
    until_free = [0.0] * n
    tasks_taken = [0] * n
    refusals = 0
    max_attempts = int(cfg.get("max_attempts", 12))

    print("=== CNP: ПОЛНЫЙ ПРОТОКОЛ ===")
    print("Заявка смотрит на окно покрытия и батарею, не на метры по прямой.")
    print()

    while stack:
        task = stack.popleft()
        print(f"--- Задача: {task.name} (осталось {task.duration:.0f} с) ---")

        manager = random.randint(0, n - 1)
        print(f"  [Земля] Задача отправлена Sat-{manager + 1}")
        print(f"  [Менеджер] Sat-{manager + 1} объявляет задачу соседям.")

        bids = []
        for i in range(n):
            cond = sat_condition(
                world,
                i,
                task,
                time_until_free=until_free[i],
                queue_length=1 if until_free[i] > 0 else 0,
            )
            chance = chance_from_condition(cond) if cond.can_work else 0.0
            bids.append({"sat": i, "chance": chance, "can": cond.can_work, "cond": cond})
            status = "могу" if cond.can_work else "отказ"
            sun_str = "на солнце" if cond.in_sun else "в тени"
            # lat_free / lon_free — куда спутник приедет, когда доделает текущую работу.
            print(
                f"    [Sat-{i + 1}] Шанс={chance:.3f}, окно={cond.access_remaining:.0f} с, "
                f"батарея={cond.battery:.2f}, {sun_str}, "
                f"потом шир={math.degrees(cond.lat_free):.1f}° дол={math.degrees(cond.lon_free):.1f}° "
                f"({status})"
            )

        candidates = [b for b in bids if b["can"]]
        if not candidates:
            print("  [Менеджер] Нет заявок. Задача отложена в конец очереди.")
            refusals += 1
            task.attempts += 1
            if task.attempts < max_attempts:
                stack.append(task)
            print()
            continue

        winner = max(candidates, key=lambda b: b["chance"])

        if random.random() < 0.1:
            print(f"  [Менеджер] Победитель Sat-{winner['sat'] + 1} — СБОЙ! Новый турнир...")
            candidates.remove(winner)
            if candidates:
                winner = max(candidates, key=lambda b: b["chance"])
                print(f"  [Менеджер] Новый победитель: Sat-{winner['sat'] + 1}")
            else:
                print("  [Менеджер] Нет запасных. Задача провалена.")
                refusals += 1
                print()
                continue

        sat_i = winner["sat"]
        cond = winner["cond"]
        print(f"  [Менеджер] Победитель: Sat-{sat_i + 1} (шанс={winner['chance']:.3f})")
        print(f"  [Sat-{sat_i + 1}] Сам себя назначаю. Беру задачу.")

        # Мир крутится вперёд: заряд падает, спутник летит. Не вычитаем «cost» из списка.
        dt = min(cond.access_remaining, task.duration)
        dt = max(dt, float(cfg.get("sim_dt", 5.0)))
        payload = np.zeros(n, dtype=float)
        payload[sat_i] = task.power_need
        world.step(dt, payload)
        task.duration = max(0.0, task.duration - dt)
        task.attempts += 1
        for i in range(n):
            until_free[i] = max(0.0, until_free[i] - dt)

        if task.duration <= 1e-6:
            tasks_taken[sat_i] += 1
            print(f"  [Sat-{sat_i + 1}] Задача закрыта.")
        else:
            # Окно кончилось раньше, чем работа — остаток в конец, как отложенная задача.
            print(
                f"  [Sat-{sat_i + 1}] Зона кончилась, осталось {task.duration:.0f} с. "
                "Кладу в конец очереди."
            )
            if task.attempts < max_attempts:
                stack.append(task)
            tasks_taken[sat_i] += 1
        print()

    print("=== ИТОГИ ===")
    for row in world_heartbeat(world):
        print(
            f"Sat-{row.sat_i + 1}: кусков={tasks_taken[row.sat_i]}, "
            f"заряд={row.battery_wh:.1f} Вт*ч"
        )
    print(f"Отказов: {refusals}")
    return {
        "backend": backend,
        "tasks_taken": tasks_taken,
        "refusals": refusals,
        "t": world.t,
    }


if __name__ == "__main__":
    run_cnp()
