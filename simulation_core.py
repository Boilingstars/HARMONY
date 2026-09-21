from Basilisk.utilities import SimulationBaseClass, macros, vizSupport
from Basilisk.simulation import spacecraft, simpleBattery, simplePowerSink, eclipse
from Basilisk.utilities import simIncludeGravBody
from Basilisk.architecture import messaging
from sat_comms import SatelliteComms
import math
import random
import json
import os
import tkinter as tk
from tkinter import ttk
import threading

random.seed(42)

# === ЗАГРУЗКА JSON ===
JSON_PATH = r"C:\Users\user\Desktop\data\P01_intro.json"

if os.path.exists(JSON_PATH):
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        scenario = json.load(f)
    print(f"Загружен сценарий: {scenario.get('meta', {}).get('title', 'unknown')}")
    print(f"Аппаратов: {len(scenario.get('satellites', []))}")
    print(f"Заданий: {len(scenario.get('jobs', []))}")
else:
    print(f"JSON не найден: {JSON_PATH}")
    scenario = None

if scenario and "model" in scenario:
    model = scenario["model"]
    RESERVE_SOC_PCT = model.get("reserve_soc_pct", 30.0)
else:
    RESERVE_SOC_PCT = 30.0

if scenario and "time" in scenario:
    STEP_S = scenario["time"].get("step_s", 300)
    TOTAL_STEPS = scenario["time"].get("steps", 48)
else:
    STEP_S = 300
    TOTAL_STEPS = 48

# === СОСТОЯНИЕ ДЛЯ ЛЕГЕНДЫ ===
legend_state = {
    "sats": [],
    "tasks_total": 0,
    "tasks_done": 0,
    "goal": "приоритетное обслуживание",
    "current_task": "—",
    "current_step": 0,
    "revenue_usd": 0.0,
    "total_steps": TOTAL_STEPS,
}

def legend_window():
    root = tk.Tk()
    root.title("Статус — Автономное управление")
    root.geometry("600x800")
    root.configure(bg="#1e1e1e")

    # === ФИКСИРОВАННАЯ ШАПКА ===
    header = tk.Label(root, text="СТАТУС", font=("Arial", 16, "bold"), fg="white", bg="#1e1e1e")
    header.pack(pady=5)

    summary = tk.Label(root, text="", justify="left", anchor="nw", font=("Consolas", 10), fg="#00ff00", bg="#1e1e1e")
    summary.pack(anchor="nw", padx=15, pady=5, fill="x")

    # === ПРОКРУЧИВАЕМАЯ ОБЛАСТЬ ===
    frame = tk.Frame(root, bg="#1e1e1e")
    frame.pack(fill="both", expand=True, padx=5, pady=5)

    canvas = tk.Canvas(frame, bg="#1e1e1e", highlightthickness=0)
    scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas, bg="#1e1e1e")

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # === ПРОКРУТКА КОЛЁСИКОМ ===
    def on_mousewheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    canvas.bind_all("<MouseWheel>", on_mousewheel)

    # === ФУТЕР ===
    footer = tk.Label(
        root,
        text="● работает  ○ idle  ◐ калибровка  ✕ недоступен",
        font=("Consolas", 8), fg="#aaaaaa", bg="#1e1e1e"
    )
    footer.pack(side="bottom", pady=5)

    # === СПИСОК СПУТНИКОВ ===
    sat_labels = []
    for s in legend_state["sats"]:
        lbl = tk.Label(scrollable_frame, text="", justify="left", anchor="nw",
                       font=("Consolas", 9), fg="white", bg="#1e1e1e")
        lbl.pack(anchor="nw", padx=10, pady=3, fill="x")
        sat_labels.append(lbl)

    def update():
        # Обновляем шапку
        text = ""
        text += f"Цель: {legend_state['goal']}\n"
        text += f"Шаг: {legend_state['current_step']} / {legend_state['total_steps']}\n"
        text += f"Задач: {legend_state['tasks_done']} / {legend_state['tasks_total']}\n"
        text += f"Выручка: {legend_state['revenue_usd']:.2f} USD\n"
        text += f"Текущая: {legend_state['current_task']}"
        summary.config(text=text)

        # Обновляем спутников
        for i, s in enumerate(legend_state["sats"]):
            if i >= len(sat_labels):
                break
            status = s["status"]
            if status == "работает":
                marker = "●"
            elif status == "калибровка":
                marker = "◐"
            elif status == "недоступен":
                marker = "✕"
            else:
                marker = "○"

            t = f"{marker} {s['id']}: {status}\n"
            t += f"    Заряд: {s['charge']:.1f} Вт·ч ({s['soc_pct']:.1f}%)\n"
            t += f"    Темп: {s['temp']:.1f} °C  Калибр: {s['calib_age']}\n"
            t += f"    Задача: {s['task']}"
            sat_labels[i].config(text=t)

        root.after(1000, update)

    update()
    root.mainloop()

legend_thread = threading.Thread(target=legend_window, daemon=True)
legend_thread.start()

# === СИМУЛЯЦИЯ ===
scSim = SimulationBaseClass.SimBaseClass()
dynProcess = scSim.CreateNewProcess("simProcess")
dynProcess.addTask(scSim.CreateNewTask("simTask", macros.sec2nano(10.0)))

gravFactory = simIncludeGravBody.gravBodyFactory()
earth = gravFactory.createEarth()
earth.isCentralBody = True

spiceObject = gravFactory.createSpiceInterface(time="2026-09-20T00:00:00")
spiceObject.zeroBase = 'Earth'
spiceObject.referenceBase = 'J2000'
scSim.AddModelToTask("simTask", spiceObject)

sunFactory = simIncludeGravBody.gravBodyFactory()
sunFactory.createSun()
sunSpice = sunFactory.createSpiceInterface(time="2026-09-20T00:00:00")
sunSpice.zeroBase = 'Earth'
sunSpice.referenceBase = 'J2000'
scSim.AddModelToTask("simTask", sunSpice)

sunMsg = sunFactory.spiceObject.planetStateOutMsgs[0]

def cartesian_to_geodetic(r_BN_N):
    x, y, z = r_BN_N[0], r_BN_N[1], r_BN_N[2]
    R_EARTH = 6371000.0
    r = math.sqrt(x**2 + y**2 + z**2)
    lon = math.atan2(y, x)
    lat = math.asin(z / r) if r > 0 else 0
    alt = r - R_EARTH
    return math.degrees(lat), math.degrees(lon), alt

def make_sat(tag, r_init, v_init, power_draw, capacity_wh, soc_pct):
    sat = spacecraft.Spacecraft()
    sat.ModelTag = tag
    scSim.AddModelToTask("simTask", sat)
    sat.gravField.gravBodies = spacecraft.GravBodyVector([earth])
    sat.hub.r_CN_NInit = r_init
    sat.hub.v_CN_NInit = v_init

    sink = simplePowerSink.SimplePowerSink()
    sink.ModelTag = tag + "-sink"
    sink.nodePowerOut = -power_draw
    scSim.AddModelToTask("simTask", sink)

    battery = simpleBattery.SimpleBattery()
    battery.ModelTag = tag + "-battery"
    battery.storageCapacity = capacity_wh * 3600.0
    battery.storedCharge_Init = capacity_wh * soc_pct / 100.0 * 3600.0
    scSim.AddModelToTask("simTask", battery)

    battery.addPowerNodeToModel(sink.nodePowerOutMsg)

    ecl = eclipse.Eclipse()
    ecl.ModelTag = tag + "-eclipse"
    ecl.addSpacecraftToModel(sat.scStateOutMsg)
    ecl.addPlanetToModel(gravFactory.spiceObject.planetStateOutMsgs[0])
    ecl.sunInMsg.subscribeTo(sunMsg)
    scSim.AddModelToTask("simTask", ecl)

    reader = messaging.SCStatesMsgReader()
    reader.subscribeTo(sat.scStateOutMsg)

    ecl_reader = messaging.EclipseMsgReader()
    ecl_reader.subscribeTo(ecl.eclipseOutMsgs[0])

    return sat, battery, reader, ecl_reader

def orbit_params(altitude_km, inclination_deg, theta_deg):
    r_mag = 6371000.0 + altitude_km * 1000.0
    v_mag = math.sqrt(3.986e14 / r_mag)
    inc = math.radians(inclination_deg)
    theta = math.radians(theta_deg)

    x = r_mag * math.cos(theta)
    y = r_mag * math.sin(theta)

    vx = -v_mag * math.sin(theta)
    vy = v_mag * math.cos(theta)

    y_inc = y * math.cos(inc)
    z_inc = y * math.sin(inc)
    vy_inc = vy * math.cos(inc)
    vz_inc = vy * math.sin(inc)

    r = [[x], [y_inc], [z_inc]]
    v = [[vx], [vy_inc], [vz_inc]]
    return r, v

# === АППАРАТЫ ИЗ JSON ===
if scenario and "satellites" in scenario:
    sats_data = scenario["satellites"]
    n_sats = len(sats_data)
    print(f"\nСоздаём {n_sats} аппаратов из JSON...")
else:
    sats_data = None
    n_sats = 4

sats = []
bats = []
readers = []
ecls = []
sat_ids = []
sat_capacities = []
sat_temp = []
sat_calib_age = []

for i in range(n_sats):
    if sats_data:
        sat_info = sats_data[i]
        sat_id = sat_info.get("id", f"Sat-{i+1}")
        cap = sat_info.get("capacity_wh", 100.0)
        soc = sat_info.get("initial_soc_pct", 100.0)
        init_temp = sat_info.get("initial_temp_c", 20.0)
        init_calib = sat_info.get("initial_calibration_age_steps", 0)
    else:
        sat_id = f"Sat-{i+1}"
        cap = 100.0
        soc = 50.0
        init_temp = 20.0
        init_calib = 0

    alt = 600 + i * 50
    inc = (i * 22.5) % 180
    theta = (i * 22.5) % 360

    r, v = orbit_params(alt, inc, theta)
    sat, bat, rd, ecl = make_sat(sat_id, r, v, 18.0, cap, soc)
    sats.append(sat)
    bats.append(bat)
    readers.append(rd)
    ecls.append(ecl)
    sat_ids.append(sat_id)
    sat_capacities.append(cap)
    sat_temp.append(init_temp)
    sat_calib_age.append(init_calib)

    legend_state["sats"].append({
        "id": sat_id,
        "status": "idle",
        "charge": soc * cap / 100.0,
        "soc_pct": soc,
        "temp": init_temp,
        "calib_age": init_calib,
        "task": "—",
    })

# === МОДУЛИ КОММУНИКАЦИИ ===
mods = []
for i in range(n_sats):
    m = SatelliteComms.SatelliteComms()
    m.ModelTag = f"Comms-{i+1}"
    m.satId = i + 1
    scSim.AddModelToTask("simTask", m)
    m.selfStateInMsg.subscribeTo(sats[i].scStateOutMsg)
    m.selfPowerInMsg.subscribeTo(bats[i].batPowerOutMsg)
    mods.append(m)

for i in range(n_sats):
    next_i = (i + 1) % n_sats
    mods[i].neighborStateInMsg.subscribeTo(sats[next_i].scStateOutMsg)
    mods[i].neighborPowerInMsg.subscribeTo(bats[next_i].batPowerOutMsg)

viz = vizSupport.enableUnityVisualization(
    scSim, "simTask", sats, liveStream=True
)
viz.reqComAddress = "127.0.0.1"
viz.reqPortNumber = "5556"

scSim.InitializeSimulation()
scSim.ConfigureStopTime(macros.sec2nano(600.0))
scSim.ExecuteSimulation()

print()
print("=== HEARTBEAT ===")
for i in range(n_sats):
    b = bats[i].batPowerOutMsg.read()
    p = readers[i]().r_BN_N
    e = ecls[i]()
    lat, lon, alt = cartesian_to_geodetic(p)
    charge_wh = b.storageLevel / 3600.0
    soc_pct = charge_wh / sat_capacities[i] * 100.0
    print(f"[{sat_ids[i]}] заряд={charge_wh:.1f} Вт*ч ({soc_pct:.1f}%), широта={lat:.2f}°, долгота={lon:.2f}°, высота={alt/1000:.1f} км, затмение={e.shadowFactor:.2f}")
    legend_state["sats"][i]["charge"] = charge_wh
    legend_state["sats"][i]["soc_pct"] = soc_pct
print()

# === ЗАДАЧИ ИЗ JSON ===
if scenario and "jobs" in scenario:
    jobs_data = scenario["jobs"]
    print(f"Задач из JSON: {len(jobs_data)}")
    tasks = []
    for job in jobs_data:
        tasks.append({
            "id": job.get("id", "task"),
            "kind": job.get("kind", "relay"),
            "release_step": job.get("release_step", 0),
            "deadline_step": job.get("deadline_step", TOTAL_STEPS),
            "work_steps": job.get("work_steps", 1),
            "duration": job.get("work_steps", 1) * STEP_S,
            "power_need": 30.0,
            "value": job.get("value_usd", 10.0),
            "priority": job.get("priority", 1),
            "eligible": job.get("eligible_satellites", sat_ids),
        })
else:
    tasks = []

charges = [bats[i].batPowerOutMsg.read().storageLevel for i in range(n_sats)]

tasks_taken = [0] * n_sats
refusals = 0
revenue = 0.0

legend_state["tasks_total"] = len(tasks)

print("=== CNP: ПОЛНЫЙ ПРОТОКОЛ ===")
print(f"Резерв: {RESERVE_SOC_PCT}% от ёмкости каждого аппарата")
print()

for task in tasks:
    legend_state["current_step"] = task["release_step"]
    legend_state["current_task"] = task["id"]

    print(f"--- Задача: {task['id']} ({task['kind']}, приоритет {task['priority']}, выручка {task['value']} USD) ---")

    manager = random.randint(0, n_sats - 1)
    print(f"  [Земля] Задача отправлена {sat_ids[manager]}")
    print(f"  [Менеджер] {sat_ids[manager]} объявляет задачу соседям.")

    bids = []
    for i in range(n_sats):
        if sat_ids[i] not in task["eligible"]:
            continue

        charge_norm = charges[i] / (sat_capacities[i] * 3600.0)
        chance = 0.7 * charge_norm + 0.3 * random.random()

        energy_needed = task["power_need"] * task["duration"]
        min_charge = RESERVE_SOC_PCT / 100.0 * sat_capacities[i] * 3600.0
        can = charges[i] - energy_needed >= min_charge
        bids.append({"sat": i, "chance": chance, "can": can})

        status = "могу" if can else "отказ"
        print(f"    [{sat_ids[i]}] Шанс={chance:.3f}, заряд={charges[i]/3600:.1f} Вт*ч, порог={min_charge/3600:.1f} Вт*ч ({status})")

    candidates = [b for b in bids if b["can"]]
    if not candidates:
        print(f"  [Менеджер] Нет заявок. Задача отложена.")
        refusals += 1
        print()
        continue

    winner = max(candidates, key=lambda b: b["chance"])

    if random.random() < 0.1:
        print(f"  [Менеджер] Победитель {sat_ids[winner['sat']]} — СБОЙ! Новый турнир...")
        legend_state["sats"][winner["sat"]]["status"] = "недоступен"
        candidates.remove(winner)
        if candidates:
            winner = max(candidates, key=lambda b: b["chance"])
            print(f"  [Менеджер] Новый победитель: {sat_ids[winner['sat']]}")
        else:
            print(f"  [Менеджер] Нет запасных. Задача провалена.")
            refusals += 1
            print()
            continue

    print(f"  [Менеджер] Победитель: {sat_ids[winner['sat']]}")
    print(f"  [{sat_ids[winner['sat']]}] Сам себя назначаю. Беру задачу.")
    energy_used = task["power_need"] * task["duration"]
    charges[winner["sat"]] -= energy_used
    tasks_taken[winner["sat"]] += 1
    revenue += task["value"]

    legend_state["sats"][winner["sat"]]["status"] = "работает"
    legend_state["sats"][winner["sat"]]["task"] = task["id"]
    legend_state["sats"][winner["sat"]]["charge"] = charges[winner["sat"]] / 3600.0
    legend_state["sats"][winner["sat"]]["soc_pct"] = charges[winner["sat"]] / (sat_capacities[winner["sat"]] * 3600.0) * 100.0
    legend_state["tasks_done"] += 1
    legend_state["revenue_usd"] = revenue

    for i in range(n_sats):
        if i != winner["sat"] and legend_state["sats"][i]["status"] != "недоступен":
            legend_state["sats"][i]["status"] = "idle"

    print()

print("=== ИТОГИ ===")
for i in range(n_sats):
    print(f"{sat_ids[i]}: задач={tasks_taken[i]}, заряд={charges[i]/3600:.1f} Вт*ч")
print(f"Отказов: {refusals}")
print(f"Выручка: {revenue:.2f} USD")

input("Нажми Enter, чтобы закрыть окно статуса...")