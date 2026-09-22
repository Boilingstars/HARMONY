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
    </div>
  `;

  el.querySelectorAll('[data-obj]').forEach((btn) => {
    btn.addEventListener('click', () => actions.setObjective(btn.dataset.obj));
  });
  el.querySelector('[data-act="toggle-run"]').addEventListener('click', actions.toggleRun);
  el.querySelector('[data-act="step"]').addEventListener('click', actions.stepForward);
}
