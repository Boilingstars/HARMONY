# HARMONY

**Hybrid Autonomous Resource Management for Orbital Networks**

RL-диспетчер задач для спутниковой группировки. Наземные задания приходят стеком: агент в реальном времени либо назначает задачу спутнику (`attach`), либо откладывает её в конец очереди (`defer`). Если одно окно доступа короче `duration`, задача дорабатывается эстафетой нескольких аппаратов.

Обучение идёт на быстром двухтельном мире (KeplerWorld). [Basilisk](https://avslab.github.io/basilisk/Learn.html) используется для проверки той же политики, не как основной тренажёр.

## Задача

Есть **N** спутников на LEO (Walker-Δ) и **M** задач в стеке. Каждая задача — точка на Земле, длительность работы и потребляемая мощность. Агент видит состояние всех аппаратов и текущую задачу и выбирает одно действие:

| Действие | Смысл |
|----------|--------|
| `attach i` | отдать задачу спутнику `i` |
| `defer` | положить задачу в **конец** стека и взять следующую |

Жёсткого reject нет. Если все КА далеко, выгоднее отложить задачу, чем забить очередь.

Если спутник ушёл из зоны до конца работы, остаток `duration` уменьшается, задача возвращается **наверх** стека — это штатный handoff, не провал.

## Наблюдения

**Спутник:** широта/долгота/высота, прогноз позиции к моменту освобождения, радиус покрытия, батарея, статус (`idle` / `sleep` / `busy`), длина очереди, `time_until_free`, маска возможностей, оставшееся окно доступа к текущей цели, время до смены затмения.

**Задача:** широта/долгота цели, остаток `duration`, доля от исходной длительности, мощность, число попыток attach.

**Маска:** attach разрешён, если есть capability, очередь не полна, батарея тянет хотя бы короткий срез и `access_remaining > 0`. Полное окно **не** требуется. `defer` всегда доступен.

## Установка

Python 3.10+. На Windows удобнее `py -3`.

```bash
cd "Harmony Framework"
py -3 -m pip install -r requirements.txt
```

Basilisk опционален (валидация и демо):

```bash
py -3 -m pip install bsk
```

Документация симулятора: [Learning Basilisk](https://avslab.github.io/basilisk/Learn.html).

## Использование

Сценарий по умолчанию задаётся в [`configs/default.yaml`](configs/default.yaml): 4 КА, 24 задачи, орбита ~500 км, эпизод 1.5 витка.

### Тесты

```bash
py -3 -m pytest -q
```

### Обучение

По умолчанию — KeplerWorld, политика pointer-network + MaskablePPO:

```bash
py -3 scripts/train.py --timesteps 50000
```

Модель сохраняется в `models/pointer_ppo.zip`. Другой конфиг или бэкенд:

```bash
py -3 scripts/train.py --timesteps 50000 --config configs/default.yaml --save models/pointer_ppo.zip --seed 0
py -3 scripts/train.py --timesteps 20000 --backend basilisk
```

`--backend basilisk` имеет смысл только если установлен `bsk`; для обычного обучения оставляйте Kepler.

### Оценка

Жадный baseline (максимальное `access_remaining` среди валидных КА):

```bash
py -3 scripts/eval.py --policy greedy --episodes 8
```

Случайный выбор среди разрешённых действий:

```bash
py -3 scripts/eval.py --policy random --episodes 8
```

Обученная политика:

```bash
py -3 scripts/eval.py --model models/pointer_ppo.zip --episodes 8
```

Прогон в Basilisk (если пакет установлен):

```bash
py -3 scripts/eval.py --model models/pointer_ppo.zip --backend basilisk --episodes 4
```

Скрипт печатает JSON: средний return, число закрытых целей, handoff, наработанные секунды покрытия.

### Демо одного эпизода

```bash
py -3 scripts/demo_basilisk.py --seed 1
```

Если Basilisk недоступен, автоматически берётся KeplerWorld.

## Конфигурация

Главные поля `configs/default.yaml`:

| Параметр | Значение по умолчанию | Назначение |
|----------|------------------------|------------|
| `n_sats` / `n_planes` | 4 / 2 | Walker-Δ группировка |
| `n_tasks` | 24 | размер стека на эпизод |
| `altitude_km` | 500 | круговая орбита |
| `min_elevation_deg` | 10 | порог покрытия цели |
| `duration_s` | 90…720 | длительность задачи; часть длиннее одного окна |
| `power_need_w` | 20…45 | мощность полезной нагрузки |
| `episode_orbits` | 1.5 | горизонт эпизода |
| `max_queue` | 4 | очередь на борту |
| `ppo.*` | — | гиперпараметры MaskablePPO |

## Структура

```
configs/default.yaml              сценарий и PPO
scripts/train.py                  обучение
scripts/eval.py                   оценка greedy / random / PPO
scripts/demo_basilisk.py          один эпизод на Basilisk или Kepler
src/harmony/env/                  Gymnasium-среда, признаки, маска
src/harmony/sim/kepler.py         быстрый мир для обучения
src/harmony/sim/basilisk_world.py Basilisk: орбита, eclipse, батарея
src/harmony/sim/coverage.py       elevation, footprint, окна доступа
src/harmony/agent/                pointer-политика, train, evaluate
src/harmony/baselines/            greedy и random
tests/                            покрытие, маска, handoff, defer
```

## Награда (кратко)

- `+ dt / duration_original` за каждую секунду покрытия (полная сборка цели даёт `+1`, неважно сколькими КА)
- `+0.2` за закрытие задачи
- `0` за defer, если никто не видит цель; `-0.05`, если зря отложили
- нет штрафа за уход из зоны при ненулевом прогрессе
- `-1` при разряде батареи до нуля; незакрытый остаток штрафуется в конце эпизода
