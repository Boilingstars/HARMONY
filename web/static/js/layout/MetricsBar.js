const AHEAD = 10;

export function renderMetricsBar(el, state, actions) {
  const m = state.metrics;
  const d = state.metricsDelta || {};
  const rev = (m.revenue_usd || 0).toLocaleString('ru-RU');
  const items = [
    {
      delta: d.completed,
      title: 'против второй политики на этом шаге',
      label: `Выполнено <span class="num">${m.critical_jobs_completed_on_time}/${m.critical_jobs_due}</span> задач высшего приоритета`,
    },
    {
      delta: d.overdue,
      title: `через ${AHEAD} шагов по текущей политике`,
      label: `Просрочено <span class="num">${m.jobs_due_missed}</span> задач`,
    },
    {
      delta: d.revenue,
      title: 'против второй политики на этом шаге',
      label: `Выручка $<span class="num">${rev}</span>`,
    },
    {
      delta: d.brownout,
      title: `через ${AHEAD} шагов по текущей политике`,
      label: `Критическое состояние заряда <span class="num">${m.brownout_satellite_steps}</span>`,
    },
    {
      delta: d.reserve,
      title: `через ${AHEAD} шагов по текущей политике`,
      label: `Заряд ниже резерва <span class="num">${m.below_reserve_satellite_steps}</span>`,
    },
    {
      delta: d.blocked,
      title: '',
      label: `Отклонённых команд <span class="num">${m.blocked_command_count}</span>`,
    },
  ];
  el.innerHTML = `
    <div class="metrics-line" data-toggle>
      ${items.map((item, i) => `
        ${i ? '<span class="sep">·</span>' : ''}
        <span class="met-item" title="${item.title}">
          ${deltaHtml(item.delta)}
          <span class="met-label">${item.label}</span>
        </span>
      `).join('')}
    </div>
    <div class="metrics-panel">
      <div class="metrics-grid">
        <div><span class="k">Заданий всего</span><span class="v">${m.jobs_total}</span></div>
        <div><span class="k">Выполнено</span><span class="v">${m.jobs_completed}</span></div>
        <div><span class="k">Потенциал</span><span class="v">$${(m.potential_revenue_usd || 0).toLocaleString('ru-RU')}</span></div>
      </div>
    </div>
  `;
  el.querySelector('[data-toggle]').addEventListener('click', actions.toggleMetrics);
}

function deltaHtml(delta) {
  if (!delta) return '<span class="met-delta is-flat">—</span>';
  if (delta.flat) return `<span class="met-delta is-flat">${delta.pct}%</span>`;
  const arrow = delta.dir === 'up' ? '↗' : '↘';
  return `<span class="met-delta is-${delta.dir}">${delta.pct}% ${arrow}</span>`;
}
