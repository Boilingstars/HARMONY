import { formatShiftDuration } from '../status.js';

export function renderTopBar(el, state, actions) {
  const steps = state.scenario.steps;
  const hours = formatShiftDuration(steps, state.scenario.step_s);
  el.innerHTML = `
    <div class="shift-meta">
      <span class="scenario-id">${state.scenario.id}</span>
      <span class="step-readout">Длительность смены · <span class="num">${steps}</span> шагов (<span class="num">${hours}</span>)</span>
    </div>
    <div class="obj-toggle" role="group" aria-label="Цель управления">
      <button type="button" data-obj="priorities" class="${state.objective === 'priorities' ? 'is-on' : ''}">Приоритеты</button>
      <button type="button" data-obj="money" class="${state.objective === 'money' ? 'is-on' : ''}">Деньги</button>
    </div>
    <div class="top-actions">
      <button type="button" class="btn btn-go" data-act="toggle-run">${state.running ? 'Пауза' : 'Запустить'}</button>
      <button type="button" class="btn" data-act="step">Шаг вперёд</button>
      <div class="overflow">
        <button type="button" class="btn" data-act="overflow">Ещё ▾</button>
        ${state.overflowOpen ? `
          <div class="overflow-menu">
            <label>До шага N <input class="jump-n" type="number" min="0" max="${state.scenario.steps}" value="${Math.min(state.step + 1, state.scenario.steps)}" data-jump></label>
            <button type="button" data-act="jump">Перейти</button>
            <button type="button" data-act="load">Загрузить сценарий</button>
            <button type="button" data-act="export">Выгрузить результат</button>
            <button type="button" data-act="event">Ввести событие</button>
            <button type="button" data-act="compare">Сравнить варианты</button>
          </div>` : ''}
      </div>
    </div>
  `;

  el.querySelectorAll('[data-obj]').forEach((btn) => {
    btn.addEventListener('click', () => actions.setObjective(btn.dataset.obj));
  });
  el.querySelector('[data-act="toggle-run"]').addEventListener('click', actions.toggleRun);
  el.querySelector('[data-act="step"]').addEventListener('click', actions.stepForward);
  el.querySelector('[data-act="overflow"]').addEventListener('click', actions.toggleOverflow);
  if (state.overflowOpen) {
    el.querySelector('[data-act="jump"]').addEventListener('click', () => {
      actions.jumpTo(Number(el.querySelector('[data-jump]').value));
    });
    el.querySelector('[data-act="load"]').addEventListener('click', actions.openLoad);
    el.querySelector('[data-act="export"]').addEventListener('click', actions.exportResult);
    el.querySelector('[data-act="event"]').addEventListener('click', actions.openEvent);
    el.querySelector('[data-act="compare"]').addEventListener('click', actions.openCompare);
  }
}
