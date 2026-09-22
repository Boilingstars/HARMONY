import { actionLabel, actionTone, satelliteStatus, socPct, treeActionLabel } from '../status.js';

export function renderSatelliteTree(el, state, actions) {
  const q = state.satQuery.trim().toLowerCase();
  const rows = state.satellites.filter((sat) => {
    const st = satelliteStatus(sat, state.model);
    if (state.satFilter === 'critical' && st !== 'critical') return false;
    if (state.satFilter === 'problem' && st === 'ok') return false;
    if (q && !sat.id.toLowerCase().includes(q) && !(sat.job_id || '').toLowerCase().includes(q)) return false;
    return true;
  });

  const selectedId = state.selection.kind === 'satellite' ? state.selection.id : null;

  el.innerHTML = `
    <div class="tree-card">
      <div class="tree-toolbar">
        <div class="tree-filters">
          <button type="button" class="filter-btn ${state.satFilter === 'all' ? 'is-on' : ''}" data-f="all">все</button>
          <button type="button" class="filter-btn ${state.satFilter === 'problem' ? 'is-on' : ''}" data-f="problem">проблемные</button>
          <button type="button" class="filter-btn ${state.satFilter === 'critical' ? 'is-on' : ''}" data-f="critical">критические</button>
        </div>
        <input type="search" placeholder="поиск" value="${state.satQuery}" data-q>
      </div>
      <div class="sat-list">
        <div class="sat-cols">
          <span></span>
          <span>ID</span>
          <span>заряд</span>
          <span title="температура">темп., °C</span>
          <span title="число шагов после калибровки">калибровка</span>
          <span>статус</span>
        </div>
        ${rows.map((sat) => {
          const st = satelliteStatus(sat, state.model);
          const soc = socPct(sat);
          const action = treeActionLabel(sat);
          return `
            <button type="button" class="sat-row ${sat.id === selectedId ? 'is-selected' : ''}" data-id="${sat.id}">
              <span class="dot ${st}" title="${st}"></span>
              <span class="num sat-id">${sat.id}</span>
              <span class="soc-cell is-${st}">
                <span class="soc-bar"><span style="width:${Math.max(0, Math.min(100, soc))}%"></span></span>
                <span class="num">${soc.toFixed(0)}%</span>
              </span>
              <span class="num">${sat.temp_c.toFixed(0)}°</span>
              <span class="num" title="число шагов после калибровки">${sat.calibration_age_steps}</span>
              <span class="action-cell is-${actionTone(sat)}" title="${actionLabel(sat)}">${action}</span>
            </button>`;
        }).join('')}
      </div>
    </div>
  `;

  el.querySelectorAll('[data-f]').forEach((btn) => {
    btn.addEventListener('click', () => actions.setSatFilter(btn.dataset.f));
  });
  el.querySelector('[data-q]').addEventListener('input', (e) => actions.setSatQuery(e.target.value));
  el.querySelectorAll('.sat-row').forEach((row) => {
    row.addEventListener('click', () => actions.selectSatellite(row.dataset.id));
  });
}
