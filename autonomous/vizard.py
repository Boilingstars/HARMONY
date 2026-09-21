from Basilisk.utilities import SimulationBaseClass, macros, vizSupport
from Basilisk.simulation import spacecraft, simpleBattery, simplePowerSink, eclipse
from Basilisk.utilities import simIncludeGravBody
from Basilisk.architecture import messaging
from sat_comms import SatelliteComms
import math
import random
import tkinter as tk
import threading

random.seed(42)

# === СОСТОЯНИЕ ДЛЯ СТАТУСА ===
legend_state = {
    "sats": [
        {"status": "idle", "charge": 500.0, "task": "—", "calib_age": 0,
         "temp": 20.0, "work_time": 0, "time_left": 0},
        {"status": "idle", "charge": 500.0, "task": "—", "calib_age": 0,
         "temp": 20.0, "work_time": 0, "time_left": 0},
        {"status": "idle", "charge": 500.0, "task": "—", "calib_age": 0,
         "temp": 20.0, "work_time": 0, "time_left": 0},
        {"status": "idle", "charge": 500.0, "task": "—", "calib_age": 0,
         "temp": 20.0, "work_time": 0, "time_left": 0},
    ],
    "tasks_total": 0,
    "tasks_done": 0,
    "goal": "приоритетное обслуживание",
    "current_task": "—",
    "current_step": 0,
    "revenue_usd": 0.0,
    "total_steps": 0,
}

def legend_window():
    root = tk.Tk()
    root.title("Статус — Автономное управление")
    root.geometry("560x900")
    root.configure(bg="#1e1e1e")

    title = tk.Label(root, text="СТАТУС", font=("Arial", 16, "bold"), fg="white", bg="#1e1e1e")
    title.pack(pady=10)

    info = tk.Label(root, text="", justify="left", anchor="nw", font=("Consolas", 10), fg="white", bg="#1e1e1e")
    info.pack(anchor="nw", padx=15, pady=5, fill="both", expand=True)

    legend_help = tk.Label(
        root,
        text="● работает  ○ idle  ◐ калибровка  ✕ недоступен\n"
             "Данные: заряд, температура, работа, задача",
        font=("Consolas", 8), fg="#aaaaaa", bg="#1e1e1e", justify="left"
    )
    legend_help.pack(side="bottom", pady=10)

    def update():
        text = ""
        text += f"Цель: {legend_state['goal']}\n"
        text += f"Шаг: {legend_state['current_step']} / {legend_state['total_steps']}\n"
        text += f"Задач всего: {legend_state['tasks_total']}\n"
        text += f"Выполнено: {legend_state['tasks_done']}\n"
        text += f"Выручка: {legend_state['revenue_usd']:.1f} USD\n"
        text += f"Текущая задача: {legend_state['current_task']}\n"
        text += "\n" + "=" * 48 + "\n\n"

        for i, s in enumerate(legend_state["sats"]):
            status = s["status"]
            if status == "работает":
                marker = "●"
            elif status == "калибровка":
                marker = "◐"
            elif status == "недоступен":
                marker = "✕"
            else:
                marker = "○"

            text += f"{marker} Sat-{i+1}: {status}\n"
            text += f"    Заряд: {s['charge']:.1f} Вт·ч\n"
            text += f"    Температура: {s['temp']:.1f} °C\n"
            text += f"    Калибровка: {s['calib_age']} шагов\n"

            if status == "работает":
                text += f"    Задача: {s['task']}\n"
                text += f"    Работает: {s['work_time']} шагов ({s['work_time']*5} мин)\n"
                text += f"    До конца: {s['time_left']} шагов ({s['time_left']*5} мин)\n"
            else:
                text += f"    Задача: —\n"

            text += "\n"

        info.config(text=text)
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

def make_sat(tag, r_init, v_init, power_draw):
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
    battery.storageCapacity = 1000.0 * 3600.0
    battery.storedCharge_Init = 500.0 * 3600.0
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

r1, v1 = orbit_params(600, 0, 0)
r2, v2 = orbit_params(1000, 30, 90)
r3, v3 = orbit_params(1400, 60, 180)
r4, v4 = orbit_params(800, 90, 270)

sat1, bat1, rd1, e1 = make_sat("Sat-1", r1, v1, 10.0)
sat2, bat2, rd2, e2 = make_sat("Sat-2", r2, v2, 5.0)
sat3, bat3, rd3, e3 = make_sat("Sat-3", r3, v3, 3.0)
sat4, bat4, rd4, e4 = make_sat("Sat-4", r4, v4, 7.0)

sats = [sat1, sat2, sat3, sat4]
bats = [bat1, bat2, bat3, bat4]
readers = [rd1, rd2, rd3, rd4]
ecls = [e1, e2, e3, e4]

mods = []
for i in range(4):
    m = SatelliteComms.SatelliteComms()
    m.ModelTag = f"Comms-{i+1}"
    m.satId = i + 1
    scSim.AddModelToTask("simTask", m)
    m.selfStateInMsg.subscribeTo(sats[i].scStateOutMsg)
    m.selfPowerInMsg.subscribeTo(bats[i].batPowerOutMsg)
    mods.append(m)

for i in range(4):
    next_i = (i + 1) % 4
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
for i in range(4):
    b = bats[i].batPowerOutMsg.read()
    p = readers[i]().r_BN_N
    e = ecls[i]()
    lat, lon, alt = cartesian_to_geodetic(p)
    print(f"[Sat-{i+1}] заряд={b.storageLevel/3600:.1f} Вт*ч, широта={lat:.2f}°, долгота={lon:.2f}°, высота={alt/1000:.1f} км, затмение={e.shadowFactor:.2f}")
    legend_state["sats"][i]["charge"] = b.storageLevel / 3600.0
print()

# === ЗАДАЧИ С ДЛИТЕЛЬНОСТЬЮ И МОЩНОСТЬЮ ===
tasks = [
    {"name": "Снять паводок",        "pos": [5000000.0, 4000000.0, 0.0],        "duration": 240.0, "power_need": 30.0, "value": 40.0, "priority": 3},
    {"name": "Передать данные",      "pos": [6000000.0, 3000000.0, 0.0],        "duration": 180.0, "power_need": 25.0, "value": 10.0, "priority": 1},
    {"name": "Сфотографировать лес", "pos": [4000000.0, 5000000.0, 0.0],        "duration": 300.0, "power_need": 35.0, "value": 25.0, "priority": 2},
    {"name": "Снять разлив",         "pos": [5500000.0, 4500000.0, 0.0],        "duration": 240.0, "power_need": 30.0, "value": 45.0, "priority": 3},
    {"name": "Мониторинг озера",     "pos": [6500000.0, 2500000.0, 0.0],        "duration": 360.0, "power_need": 20.0, "value": 30.0, "priority": 2},
    {"name": "Снять наводнение",     "pos": [5000000.0, 3000000.0, 2000000.0],  "duration": 240.0, "power_need": 30.0, "value": 35.0, "priority": 3},
    {"name": "Сфотографировать горы","pos": [4000000.0, 4000000.0, 3000000.0],  "duration": 300.0, "power_need": 35.0, "value": 20.0, "priority": 2},
    {"name": "Мониторинг реки",      "pos": [6000000.0, 2000000.0, 1500000.0],  "duration": 200.0, "power_need": 25.0, "value": 15.0, "priority": 1},
    {"name": "Снять тайгу",          "pos": [3500000.0, 5500000.0, 2500000.0],  "duration": 400.0, "power_need": 40.0, "value": 50.0, "priority": 3},
    {"name": "Снять побережье",      "pos": [4500000.0, 3500000.0, -2000000.0], "duration": 220.0, "power_need": 30.0, "value": 25.0, "priority": 2},
    {"name": "Мониторинг болот",     "pos": [5500000.0, 2500000.0, -1500000.0], "duration": 280.0, "power_need": 25.0, "value": 20.0, "priority": 1},
    {"name": "Снять ледник",         "pos": [3000000.0, 4500000.0, 3000000.0],  "duration": 320.0, "power_need": 35.0, "value": 40.0, "priority": 3},
]

def distance(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

MIN_CHARGE = 0.2 * 500.0 * 3600.0

charges = [bats[i].batPowerOutMsg.read().storageLevel for i in range(4)]
positions = [readers[i]().r_BN_N for i in range(4)]
shadow = [ecls[i]().shadowFactor for i in range(4)]

tasks_taken = [0, 0, 0, 0]
refusals = 0
current_step = 0
total_steps = len(tasks)

legend_state["tasks_total"] = total_steps
legend_state["total_steps"] = total_steps

print("=== CNP: ПОЛНЫЙ ПРОТОКОЛ ===")
print(f"Порог заряда: {MIN_CHARGE/3600:.1f} Вт*ч")
print()

for task in tasks:
    current_step += 1
    legend_state["current_step"] = current_step
    legend_state["current_task"] = task["name"]

    print(f"--- Задача: {task['name']} (длит {task['duration']:.0f} с, мощн {task['power_need']:.0f} Вт) ---")

    manager = random.randint(0, 3)
    print(f"  [Земля] Задача отправлена Sat-{manager+1}")
    print(f"  [Менеджер] Sat-{manager+1} объявляет задачу соседям.")

    bids = []
    for i in range(4):
        d = distance(positions[i], task["pos"])
        charge_norm = charges[i] / 1800000.0
        dist_norm = 1.0 - min(d / 10000000.0, 1.0)
        sun_bonus = 0.2 if shadow[i] < 0.5 else 0.0
        chance = 0.4 * charge_norm + 0.4 * dist_norm + sun_bonus

        energy_needed = task["power_need"] * task["duration"]
        can = charges[i] - energy_needed >= MIN_CHARGE
        bids.append({"sat": i, "chance": chance, "can": can, "dist": d})

        status = "могу" if can else "отказ"
        sun_str = "на солнце" if shadow[i] < 0.5 else "в тени"
        print(f"    [Sat-{i+1}] Шанс={chance:.3f}, дистанция={d:.0f} м, заряд={charges[i]/3600:.1f} Вт*ч, {sun_str} ({status})")

        legend_state["sats"][i]["charge"] = charges[i] / 3600.0

    candidates = [b for b in bids if b["can"]]
    if not candidates:
        print(f"  [Менеджер] Нет заявок. Задача отложена.")
        for i in range(4):
            legend_state["sats"][i]["status"] = "idle"
        refusals += 1
        print()
        continue

    winner = max(candidates, key=lambda b: b["chance"])

    if random.random() < 0.1:
        print(f"  [Менеджер] Победитель Sat-{winner['sat']+1} — СБОЙ! Новый турнир...")
        legend_state["sats"][winner["sat"]]["status"] = "недоступен"
        candidates.remove(winner)
        if candidates:
            winner = max(candidates, key=lambda b: b["chance"])
            print(f"  [Менеджер] Новый победитель: Sat-{winner['sat']+1}")
        else:
            print(f"  [Менеджер] Нет запасных. Задача провалена.")
            refusals += 1
            print()
            continue

    print(f"  [Менеджер] Победитель: Sat-{winner['sat']+1} (шанс={winner['chance']:.3f})")
    print(f"  [Sat-{winner['sat']+1}] Сам себя назначаю. Беру задачу.")

    energy_used = task["power_need"] * task["duration"]
    charges[winner["sat"]] -= energy_used
    tasks_taken[winner["sat"]] += 1

    legend_state["sats"][winner["sat"]]["status"] = "работает"
    legend_state["sats"][winner["sat"]]["task"] = task["name"]
    legend_state["sats"][winner["sat"]]["charge"] = charges[winner["sat"]] / 3600.0
    legend_state["sats"][winner["sat"]]["work_time"] = int(task["duration"] / 300)
    legend_state["sats"][winner["sat"]]["time_left"] = max(0, total_steps - current_step)
    legend_state["tasks_done"] += 1
    legend_state["revenue_usd"] += task["value"]

    for i in range(4):
        if i != winner["sat"] and legend_state["sats"][i]["status"] != "недоступен":
            legend_state["sats"][i]["status"] = "idle"

    print()

print("=== ИТОГИ ===")
for i in range(4):
    print(f"Sat-{i+1}: задач={tasks_taken[i]}, заряд={charges[i]/3600:.1f} Вт*ч")
print(f"Отказов: {refusals}")
print(f"Выручка: {legend_state['revenue_usd']:.1f} USD")

input("Нажми Enter, чтобы закрыть окно статуса...")
