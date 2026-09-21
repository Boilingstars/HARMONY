# HARMONY

**Hybrid Autonomous Resource Management for Orbital Networks**

Два диспетчера на **одном мире**: RL-агент и автономный CNP без нейросети. Наземные задания — точка на Земле, длительность и мощность. Если одно окно доступа короче `duration`, задачу дорабатывает эстафета нескольких аппаратов.

Физика общая: орбита, покрытие цели, батарея, тень. Снимок состояния для обоих контуров — [`src/harmony/sim/condition.py`](src/harmony/sim/condition.py). Новые поля мира добавляют туда, а не копируют в каждый скрипт.

Обучение RL идёт на KeplerWorld. [Basilisk](https://avslab.github.io/basilisk/Learn.html) — проверка той же политики и опциональный бэкенд для CNP.

## Два контура

| Контур | Файл | Как решает |
|--------|------|------------|
| RL | `scripts/train.py`, среда `TaskAssignmentEnv` | pointer-сеть, `attach i` или `defer` |
| Автономный CNP | [`autonomous.py`](autonomous.py) | абстрактный менеджер и заявки спутников, без ЦУ и без сети |

Оба читают `sat_condition(...)`: видит ли спутник цель, сколько ещё будет видеть, где окажется когда освободится, хватает ли батареи.

## Задача RL

Есть **N** спутников на LEO (Walker-Δ) и **M** задач в стеке. Действия:

| Действие | Смысл |
|----------|--------|
| `attach i` | отдать задачу спутнику `i` |
| `defer` | положить задачу в **конец** стека и взять следующую |

Жёсткого reject нет. Если все КА далеко, выгоднее отложить задачу, чем забить очередь.

Если спутник ушёл из зоны до конца работы, остаток `duration` уменьшается, задача возвращается **наверх** стека — штатный handoff, не провал. В CNP остаток кладётся в **конец** очереди (их протокол).

## Наблюдения (общий снимок)

**Спутник:** широта/долгота/высота, прогноз позиции к моменту освобождения (`lat_free`, `lon_free`), радиус покрытия, батарея, статус (`idle` / `sleep` / `busy`), длина очереди, `time_until_free`, маска возможностей, оставшееся окно доступа к цели, время до смены затмения.

**Задача:** широта/долгота, остаток `duration`, доля от исходной длительности, мощность, число попыток.

**Маска / заявка `can_work`:** capability, очередь не полна, батарея тянет короткий срез, `access_remaining > 0`. Полное окно **не** требуется.

## Установка

Python 3.10+. На Windows удобнее `py -3`.

```bash
cd "Harmony Framework"
py -3 -m pip install -r requirements.txt
```

Basilisk опционален:

```bash
py -3 -m pip install bsk
```

Документация симулятора: [Learning Basilisk](https://avslab.github.io/basilisk/Learn.html).

## Использование

Сценарий по умолчанию: [`configs/default.yaml`](configs/default.yaml) — 4 КА, 24 задачи, ~500 км, эпизод 1.5 витка.

### Тесты

```bash
py -3 -m pytest -q
```

### Автономный CNP (без нейросети)

```bash
py -3 autonomous.py
```

Мир — Basilisk, если установлен `bsk`, иначе Kepler. Vizard по умолчанию выключен: в файле поставьте `ENABLE_VIZARD = True`, если стрим нужен. Модуль `sat_comms` подключается только если он есть рядом.

### Обучение RL

```bash
py -3 scripts/train.py --timesteps 50000
```

Модель: `models/pointer_ppo.zip`.

```bash
py -3 scripts/train.py --timesteps 50000 --config configs/default.yaml --save models/pointer_ppo.zip --seed 0
py -3 scripts/train.py --timesteps 20000 --backend basilisk
```

`--backend basilisk` только при установленном `bsk`. Для обучения оставляйте Kepler.

### Оценка RL

```bash
py -3 scripts/eval.py --policy greedy --episodes 8
py -3 scripts/eval.py --policy random --episodes 8
py -3 scripts/eval.py --model models/pointer_ppo.zip --episodes 8
py -3 scripts/eval.py --model models/pointer_ppo.zip --backend basilisk --episodes 4
```

JSON: средний return, закрытые цели, handoff, секунды покрытия.

### Демо одного эпизода RL

```bash
py -3 scripts/demo_basilisk.py --seed 1
```

Без Basilisk берётся KeplerWorld.

## Конфигурация

| Параметр | Значение по умолчанию | Назначение |
|----------|------------------------|------------|
| `n_sats` / `n_planes` | 4 / 2 | Walker-Δ группировка |
| `n_tasks` | 24 | размер стека на эпизод RL |
| `altitude_km` | 500 | круговая орбита |
| `min_elevation_deg` | 10 | порог покрытия цели |
| `duration_s` | 90…720 | длительность задачи; часть длиннее одного окна |
| `power_need_w` | 20…45 | мощность полезной нагрузки |
| `episode_orbits` | 1.5 | горизонт эпизода RL |
| `max_queue` | 4 | очередь на борту |
| `ppo.*` | — | гиперпараметры MaskablePPO |

## Структура

```
autonomous.py                     CNP-менеджер тиммейтов (цельный файл)
configs/default.yaml              сценарий группировки и PPO
scripts/train.py                  обучение RL
scripts/eval.py                   оценка greedy / random / PPO
scripts/demo_basilisk.py          один эпизод RL на Basilisk или Kepler
src/harmony/sim/condition.py      общий снимок мира (RL + CNP)
src/harmony/sim/world.py          контракт бэкенда
src/harmony/sim/kepler.py         быстрый мир для обучения
src/harmony/sim/basilisk_world.py Basilisk: орбита, eclipse, батарея
src/harmony/sim/coverage.py       elevation, покрытие, окна доступа
src/harmony/env/                  Gymnasium-среда, признаки, маска
src/harmony/agent/                pointer-политика, train, evaluate
src/harmony/baselines/            greedy и random
tests/                            покрытие, маска, handoff, condition, CNP
```

Новое поле симуляции (покрытие, энергия, прогноз позиции) добавляйте в `condition.py`. Не дублируйте сборку Basilisk в `autonomous.py`.

## Награда RL (кратко)

- `+ dt / duration_original` за каждую секунду покрытия (полная сборка цели даёт `+1`, неважно сколькими КА)
- `+0.2` за закрытие задачи
- `0` за defer, если никто не видит цель; `-0.05`, если зря отложили
- нет штрафа за уход из зоны при ненулевом прогрессе
- `-1` при разряде батареи до нуля; незакрытый остаток штрафуется в конце эпизода
