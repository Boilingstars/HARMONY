export function renderMetricsBar(el, state, actions) {
  const m = state.metrics;
  const rev = m.revenue_usd.toLocaleString('ru-RU');
  el.innerHTML = `
    <div class="metrics-line" data-toggle>
      <span>Выполнено: <span class="num">${m.critical_jobs_completed_on_time}/${m.critical_jobs_due}</span> задач высшего приоритета</span>
      <span class="sep">|</span>
      <span>Просрочено: <span class="num">${m.jobs_due_missed}</span> задач</span>
      <span class="sep">|</span>
      <span>Выручка: $<span class="num">${rev}</span></span>
      <span class="sep">|</span>
      <span>Критическое состояние заряда: <span class="num">${m.brownout_satellite_steps}</span></span>
      <span class="sep">|</span>
      <span>Заряд ниже резерва: <span class="num">${m.below_reserve_satellite_steps}</span></span>
      <span class="sep">|</span>
      <span>Отклонённых команд: <span class="num">${m.blocked_command_count}</span></span>
    </div>
    <div class="metrics-panel">
      <div class="metrics-grid">
        <div><span class="k">Заданий всего:</span><span class="v">${m.jobs_total}</span></div>
        <div><span class="k">Выполнено:</span><span class="v">${m.jobs_completed}</span></div>
        <div><span class="k">Потенциал:</span><span class="v">$${(m.potential_revenue_usd || 0).toLocaleString('ru-RU')}</span></div>
      </div>
    </div>
  `;
  el.querySelector('[data-toggle]').addEventListener('click', actions.toggleMetrics);
}
