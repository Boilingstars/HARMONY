export function renderMetricsBar(el, state, actions) {
  const m = state.metrics;
  const rev = m.revenue_usd.toLocaleString('ru-RU');
  const items = [
    { pct: 78, dir: 'up', label: `Выполнено <span class="num">${m.critical_jobs_completed_on_time}/${m.critical_jobs_due}</span> задач высшего приоритета` },
    { pct: 14, dir: 'down', label: `Просрочено <span class="num">${m.jobs_due_missed}</span> задач` },
    { pct: 6, dir: 'up', label: `Выручка $<span class="num">${rev}</span>` },
    { pct: 9, dir: 'down', label: `Критическое состояние заряда <span class="num">${m.brownout_satellite_steps}</span>` },
    { pct: 22, dir: 'up', label: `Заряд ниже резерва <span class="num">${m.below_reserve_satellite_steps}</span>` },
    { pct: 4, dir: 'down', label: `Отклонённых команд <span class="num">${m.blocked_command_count}</span>` },
  ];
  el.innerHTML = `
    <div class="metrics-line" data-toggle>
      <span class="met-item met-users">
        <svg class="met-user-icon" viewBox="0 0 16 16" aria-hidden="true">
          <circle cx="8" cy="5" r="2.25"/>
          <path d="M3.6 13.2c.7-2.6 2.2-3.9 4.4-3.9s3.7 1.3 4.4 3.9"/>
        </svg>
        <span class="met-label">Кол-во пользователей <span class="num">8</span></span>
      </span>
      ${items.map((item) => `
        <span class="sep">·</span>
        <span class="met-item">
          <span class="met-delta is-${item.dir}">${item.pct}% ${item.dir === 'up' ? '↗' : '↘'}</span>
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
