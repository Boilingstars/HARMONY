# HARMONY

**Hybrid Autonomous Resource Management for Orbital Networks**

Два диспетчера на **одном мире**: RL-агент и автономный CNP без нейросети. Наземные задания — точка на Земле, длительность и мощность. Если одно окно доступа короче `duration`, задачу дорабатывает эстафета нескольких аппаратов.

Физика общая: орбита, покрытие цели, батарея, тень. Снимок состояния — [`src/harmony/sim/condition.py`](src/harmony/sim/condition.py). Новые поля мира добавляют туда.

Сравнение методов и визуализация — [`baseline.py`](baseline.py) и папка [`figures/`](figures/). Автономный контур живёт отдельно в [`autonomous/`](autonomous/).

Обучение RL — KeplerWorld. [Basilisk](https://avslab.github.io/basilisk/Learn.html) — проверка политики и опциональный бэкенд CNP.

## Два контура

| Контур | Где | Как решает |
|--------|-----|------------|
| RL | `scripts/train.py`, среда `TaskAssignmentEnv` | pointer-сеть, `attach i` или `defer` |
| Автономный CNP | [`autonomous/autonomous.py`](autonomous/autonomous.py) | менеджер и заявки спутников, без ЦУ и без сети |
| Сравнение | [`baseline.py`](baseline.py) → [`figures/`](figures/) | greedy + CNP + RL (если есть модель) |

Оба диспетчера читают `sat_condition(...)`: видит ли спутник цель, сколько ещё будет видеть, где окажется когда освободится, хватает ли батареи.

## Задача RL

Есть **N** спутников на LEO (Walker-Δ) и **M** задач в стеке.

| Действие | Смысл |
|----------|--------|
| `attach i` | отдать задачу спутнику `i` |
| `defer` | положить задачу в **конец** стека и взять следующую |

Жёсткого reject нет. Если спутник ушёл из зоны до конца работы, остаток `duration` уходит **наверх** стека (handoff). В CNP остаток кладётся в **конец** их очереди.

## Наблюдения (общий снимок)

**Спутник:** широта/долгота/высота, прогноз (`lat_free`, `lon_free`), радиус покрытия, батарея, статус (`idle` / `sleep` / `busy`), очередь, `time_until_free`, capability, окно доступа, время до смены затмения.

**Задача:** широта/долгота, остаток `duration`, доля от исходной, мощность, попытки.

**Маска / `can_work`:** capability, очередь не полна, батарея на короткий срез, `access_remaining > 0`. Полное окно не требуется.

## Установка

Python 3.10+. На Windows: `py -3`.

```bash
cd "Harmony Framework"
py -3 -m pip install -r requirements.txt
```

Basilisk опционален: `py -3 -m pip install bsk`. Документация: [Learning Basilisk](https://avslab.github.io/basilisk/Learn.html).

## Использование

Сценарий: [`configs/default.yaml`](configs/default.yaml) — 4 КА, 24 задачи, ~500 км.

### Тесты

```bash
py -3 -m pytest -q
```

### Автономный CNP

```bash
py -3 autonomous/autonomous.py
```

Мир — Basilisk, если есть `bsk`, иначе Kepler. Vizard: в файле `ENABLE_VIZARD = True`. `sat_comms` подключается, только если модуль лежит рядом в `autonomous/`.

### Сравнение подходов

```bash
py -3 baseline.py
```

Пишет [`figures/comparison.json`](figures/comparison.json). Сюда же кладите графики и ролики визуализации.

### Обучение RL

```bash
py -3 scripts/train.py --timesteps 50000
```

Модель: `models/pointer_ppo.zip`.

```bash
py -3 scripts/train.py --config configs/default.yaml --save models/pointer_ppo.zip --seed 0
py -3 scripts/eval.py --policy greedy --episodes 8
py -3 scripts/eval.py --model models/pointer_ppo.zip --episodes 8
py -3 scripts/demo_basilisk.py --seed 1
```

Для обучения оставляйте Kepler. `--backend basilisk` — только если установлен `bsk`.

## Конфигурация

| Параметр | По умолчанию | Назначение |
|----------|----------------|------------|
| `n_sats` / `n_planes` | 4 / 2 | Walker-Δ |
| `n_tasks` | 24 | стек RL |
| `altitude_km` | 500 | круговая орбита |
| `min_elevation_deg` | 10 | порог покрытия |
| `duration_s` | 90…720 | часть задач длиннее одного окна |
| `power_need_w` | 20…45 | мощность нагрузки |
| `episode_orbits` | 1.5 | горизонт RL |
| `max_queue` | 4 | очередь на борту |
| `ppo.*` | — | MaskablePPO |

## Структура

```
autonomous/autonomous.py          CNP-менеджер (цельный файл, зона тиммейтов)
baseline.py                       сводка greedy / CNP / RL
figures/                          JSON, графики, видео сравнения
configs/default.yaml              сценарий группировки и PPO
scripts/train.py                  обучение RL
scripts/eval.py                   оценка greedy / random / PPO
scripts/demo_basilisk.py          один эпизод RL
src/harmony/sim/condition.py      общий снимок мира
src/harmony/sim/world.py          контракт бэкенда
src/harmony/sim/kepler.py         быстрый мир
src/harmony/sim/basilisk_world.py Basilisk
src/harmony/sim/coverage.py       покрытие и окна
src/harmony/env/                  Gymnasium RL-среда
src/harmony/agent/                pointer-политика
src/harmony/baselines/            greedy / random (внутри RL)
tests/
```

Новое поле симуляции — в `condition.py`. Не копируйте сборку мира в `autonomous/`.

## Награда RL (кратко)

- `+ dt / duration_original` за покрытие (полная сборка цели `+1`)
- `+0.2` за закрытие задачи
- `0` за defer, если никто не видит цель; `-0.05`, если зря отложили
- нет штрафа за уход из зоны при прогрессе
- `-1` при нулевой батарее; незакрытый остаток — в конце эпизода
