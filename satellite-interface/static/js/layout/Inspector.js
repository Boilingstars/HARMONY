import { actionLabel, eventTypeLabel, idleReasonLabel, jobKindLabel, jobPriorityLabel, jobStatusLabel, reasonLabel, socPct, tapeActionLabel } from '../status.js';
import { explainJob } from '../xray.js';
import { operatorEvents, selectedSatellite } from '../state.js';
import { seriesChart } from '../chart.js';

export function renderInspector(el, state, actions) {
  const canAdd = state.inspectorTab === 'event' && state.eventComposeAt === state.step;
  el.innerHTML = `
    <div class="insp-tabs">
      <button type="button" data-tab="satellite" class="${state.inspectorTab === 'satellite' ? 'is-on' : ''}">Спутник</button>
      <button type="button" data-tab="job" class="${state.inspectorTab === 'job' ? 'is-on' : ''}">Задания</button>
      <button type="button" data-tab="event" class="${state.inspectorTab === 'event' ? 'is-on' : ''}">События</button>
      ${state.inspectorTab === 'event' ? `
        <div class="event-add-wrap">
          <button type="button" class="btn event-add" data-add ${canAdd ? '' : 'disabled'} title="${canAdd ? 'Добавить событие' : 'Событие можно ввести после «Шаг вперёд»'}">+</button>
          ${state.eventMenuOpen && canAdd ? `
            <div class="event-add-menu">
              <button type="button" data-etype="add_jobs">Новое срочное задание</button>
              <button type="button" data-etype="satellite_outage">Спутник недоступен</button>
              <button type="button" data-etype="close_downlink">Отмена сеанса</button>
            </div>` : ''}
        </div>` : ''}
    </div>
    <div class="insp-body" data-body></div>
  `;
  el.querySelectorAll('[data-tab]').forEach((btn) => {
    btn.addEventListener('click', () => actions.setTab(btn.dataset.tab));
  });
  const addBtn = el.querySelector('[data-add]');
  if (addBtn && !addBtn.disabled) {
    addBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      actions.toggleEventMenu();
    });
  }
  el.querySelectorAll('[data-etype]').forEach((btn) => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      actions.addEvent(btn.dataset.etype);
    });
  });
  const body = el.querySelector('[data-body]');
  if (state.inspectorTab === 'satellite') renderSat(body, state);
  else if (state.inspectorTab === 'job') renderJob(body, state, actions);
  else renderEvent(body, state, actions);
}

function renderSat(el, state) {
  const sat = selectedSatellite(state) || state.satellites.find((s) => s.id === state.selection.id);
  if (state.selection.kind !== 'satellite' || !sat) {
    el.innerHTML = '<p class="empty">Выберите спутник в дереве или на 3D.</p>';
    return;
  }
  const soc = socPct(sat);
  el.innerHTML = `
    <div class="sat-props">
      <section class="prop-card">
        <h3 class="prop-card-h">Параметры</h3>
        <div class="prop-grid">
          <span class="k">Идентификатор</span><span class="v">${sat.id}</span>
          <span class="k">Ёмкость батареи</span><span class="v">${sat.capacity_wh.toFixed(1)} <span class="u">Вт·ч</span></span>
          <span class="k">Заряд</span><span class="v">${soc.toFixed(2)} <span class="u">%</span></span>
          <span class="k">Начальная температура</span><span class="v">${sat.temp_c.toFixed(2)} <span class="u">°C</span></span>
          <span class="k">Базовая мощность</span><span class="v">${sat.base_w} <span class="u">Вт</span></span>
          <span class="k">Мощность нагревателя</span><span class="v">${sat.heater_w} <span class="u">Вт</span></span>
        </div>
      </section>
      <section class="prop-card">
        <h3 class="prop-card-h">Доп. мощность</h3>
        <div class="prop-grid">
          <span class="k">Калибровка</span><span class="v">${sat.calibration_w} <span class="u">Вт</span></span>
          <span class="k">Передача на Землю</span><span class="v">${sat.downlink_w} <span class="u">Вт</span></span>
          <span class="k">Ретрансляция</span><span class="v">${sat.relay_w} <span class="u">Вт</span></span>
        </div>
      </section>
      <section class="prop-card">
        <h3 class="prop-card-h">Состояние</h3>
        <div class="prop-grid">
          <div class="prop-pair is-long">
            <span class="k">Число шагов после последней калибровки</span>
            <span class="v">${sat.calibration_age_steps}</span>
          </div>
          <span class="k">Текущее действие</span><span class="v">${actionLabel(sat)}</span>
        </div>
      </section>
      ${seriesChart(sat.soc_series, {
        title: 'Заряд, %',
        yMin: 0,
        yMax: 100,
        yTicks: [0, 25, 50, 75, 100],
        yAxis: 'заряд, %',
        xAxis: 'шаги',
        xMin: 0,
        xMax: state.scenario.steps,
        xNow: state.previewStep ?? state.step,
        refs: [
          { value: state.model.reserve_soc_pct, cls: 'is-reserve' },
          { value: state.model.critical_soc_pct, cls: 'is-critical' },
        ],
        caption: [
          `текущее состояние ${soc.toFixed(0)}% · резерв ${state.model.reserve_soc_pct}%`,
          `критическое ${state.model.critical_soc_pct}%`,
        ],
      })}
      ${seriesChart(sat.temp_series, {
        title: 'Температура, °C',
        yMin: -150,
        yMax: 150,
        yTicks: [-150, -75, 0, 75, 150],
        yAxis: 'температура, °C',
        xAxis: 'шаги',
        xMin: 0,
        xMax: state.scenario.steps,
        xNow: state.previewStep ?? state.step,
        refs: [
          { value: 130, cls: 'is-critical' },
          { value: -130, cls: 'is-critical' },
        ],
        caption: `текущее состояние ${sat.temp_c.toFixed(1)}°C · критическое ±130°C`,
      })}
      <section class="prop-card">
        <h3 class="prop-card-h">История</h3>
        ${logGrid('шаг', 'статус', sat.action_tape.map((a) => [a.step, tapeActionLabel(a)]))}
      </section>
      <section class="prop-card">
        <h3 class="prop-card-h">Отклонённые команды</h3>
        ${logGrid('шаг', 'причина', sat.rejected.map((r) => [r.step, reasonLabel(r.reason)]))}
      </section>
      <section class="prop-card">
        <h3 class="prop-card-h">Статус</h3>
        <p class="prop-note">${idleReasonLabel(sat)}</p>
      </section>
    </div>
  `;
}

function logGrid(leftH, rightH, rows) {
  if (!rows.length) return '<p class="empty">нет</p>';
  return `
    <div class="tape-grid">
      <span class="th">${leftH}</span><span class="th is-val">${rightH}</span>
      ${rows.map(([left, right]) => `<span class="k">${left}</span><span class="v">${right}</span>`).join('')}
    </div>`;
}

function renderJob(el, state, actions) {
  const q = (state.jobQuery || '').trim().toLowerCase();
  const jobs = state.jobs.filter((j) => {
    if (state.jobFilter === 'p3' && j.priority !== 3) return false;
    if (state.jobFilter === 'overdue' && j.status !== 'overdue') return false;
    if (state.jobFilter === 'problem' && !['overdue', 'infeasible'].includes(j.status)) return false;
    if (q && !j.id.toLowerCase().includes(q)
      && !String(j.priority).includes(q)
      && !jobStatusLabel(j.status).toLowerCase().includes(q)
      && !(j.kind || '').toLowerCase().includes(q)) return false;
    return true;
  });
  const selectedId = state.selection.kind === 'job' ? state.selection.id : null;
  const job = state.jobs.find((j) => j.id === selectedId) || null;
  const xray = explainJob(job);

  el.innerHTML = `
    <div class="jobs-pane">
      <section class="prop-card job-list-card">
        <div class="tree-toolbar job-toolbar">
          <div class="tree-filters">
            <button type="button" class="filter-btn ${state.jobFilter === 'all' ? 'is-on' : ''}" data-jf="all">все</button>
            <button type="button" class="filter-btn ${state.jobFilter === 'problem' ? 'is-on' : ''}" data-jf="problem">проблемные</button>
            <button type="button" class="filter-btn ${state.jobFilter === 'overdue' ? 'is-on' : ''}" data-jf="overdue">просроченные</button>
            <button type="button" class="filter-btn ${state.jobFilter === 'p3' ? 'is-on' : ''}" data-jf="p3">высший приоритет</button>
          </div>
          <input type="search" placeholder="поиск" value="${state.jobQuery || ''}" data-job-q>
        </div>
        <div class="job-list">
          <div class="job-cols">
            <span>ID</span>
            <span>Приоритет</span>
            <span>Статус</span>
          </div>
          ${jobs.slice(0, 60).map((j) => `
            <button type="button" class="job-row ${j.id === selectedId ? 'is-selected' : ''}" data-jid="${j.id}">
              <span class="num">${j.id}</span>
              <span class="num">${j.priority}</span>
              <span class="status-${j.status}">${jobStatusLabel(j.status)}</span>
            </button>`).join('')}
        </div>
      </section>
      ${job ? jobDetail(job, xray) : ''}
    </div>
  `;
  el.querySelectorAll('[data-jf]').forEach((btn) => {
    btn.addEventListener('click', () => actions.setJobFilter(btn.dataset.jf));
  });
  el.querySelector('[data-job-q]').addEventListener('input', (e) => actions.setJobQuery(e.target.value));
  el.querySelectorAll('[data-jid]').forEach((row) => {
    row.addEventListener('click', () => actions.selectJob(row.dataset.jid));
  });
  el.querySelector('.job-row.is-selected')?.scrollIntoView({ block: 'nearest' });
}

function jobDetail(job, xray) {
  const done = job.work_steps - job.remaining_steps;
  const by = job.progress_by.join(', ') || '—';
  return `
    <section class="prop-card job-detail">
      <h3 class="prop-card-h">${job.id}</h3>
      <div class="prop-grid">
        <span class="k">Тип работы</span><span class="v">${jobKindLabel(job.kind)}</span>
        <span class="k">Минимальный шаг для исполнения</span><span class="v">${job.release_step}</span>
        <span class="k">Максимальный шаг для исполнения</span><span class="v">${job.deadline_step}</span>
        <span class="k">Требуемое кол-во шагов</span><span class="v">${job.work_steps}</span>
        <span class="k">Исполнители</span><span class="v">${job.eligible_satellites.join(', ') || '—'}</span>
        <span class="k">Приоритет</span><span class="v">${jobPriorityLabel(job.priority)}</span>
        <span class="k">Выручка</span><span class="v">$${job.value_usd.toFixed(2)}</span>
        <span class="k">Прогресс</span><span class="v">${done}/${job.work_steps} шагов · ${by}</span>
        <span class="k">Статус</span><span class="v status-${job.status}">${jobStatusLabel(job.status)}</span>
      </div>
    </section>
    <section class="prop-card job-xray">
      <h3 class="prop-card-h">Анализ текущего состояния</h3>
      ${xray ? `<ul class="xray">${xray.map((s) => `<li>${s}</li>`).join('')}</ul>` : '<p class="empty">задание выполняется или выполнено.</p>'}
    </section>
  `;
}

function renderEvent(el, state, actions) {
  const list = operatorEvents(state);
  if (!list.length) {
    el.innerHTML = state.eventComposeAt === state.step
      ? `<p class="empty empty-split"><span>Текущий шаг <span class="num">${state.step}</span> ·</span> <span>Нажмите + и выберите тип события.</span></p>`
      : '<p class="empty empty-break">Кликните метку на таймлайне или нажмите<br>«Шаг вперёд», чтобы ввести событие.</p>';
    return;
  }
  el.innerHTML = list.map((ev) => `
    <section class="prop-card">
      <div class="prop-card-head">
        <h3 class="prop-card-h">СОБЫТИЕ № ${ev.seq}</h3>
        <button type="button" class="event-cancel" data-cancel="${ev.id}">Отмена</button>
      </div>
      <div class="prop-grid">
        <span class="k">Идентификатор</span><span class="v">${ev.id}</span>
        <span class="k">Тип</span><span class="v">${eventTypeLabel(ev.type)}</span>
        <span class="k">Шаг</span><span class="v">${ev.at_step}</span>
        <span class="k">Принято</span><span class="v">${ev.reaction.accepted ? 'да' : 'нет'}</span>
      </div>
    </section>
  `).join('');
  el.querySelectorAll('[data-cancel]').forEach((btn) => {
    btn.addEventListener('click', () => actions.cancelEvent(btn.dataset.cancel));
  });
}
