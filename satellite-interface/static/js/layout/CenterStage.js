import { formatClock, satelliteStatus } from '../status.js';
import { operatorEvents } from '../state.js';
import { buildSignalSvg, timelineMarks } from '../timelineSignal.js';

export function renderCenterStage(state, actions, d3El, tlEl) {
  renderPlaceholder(d3El, state, actions);
  renderTimeline(tlEl, state, actions);
}

function renderPlaceholder(el, state, actions) {
  const selected = state.selection.kind === 'satellite' ? state.selection.id : '—';
  const counts = { ok: 0, reserve: 0, critical: 0, unavailable: 0 };
  state.satellites.forEach((s) => { counts[satelliteStatus(s, state.model)] += 1; });
  el.innerHTML = `
    <div class="hud">
      <strong>3D-модель будет подключена</strong>
      · <span class="num">selectSatellite</span>
      · выбран <span class="sel">${selected}</span>
      · ${counts.ok}/${counts.reserve}/${counts.critical}/${counts.unavailable}
    </div>
  `;
  el.onclick = () => {
    if (state.selection.kind === 'satellite') actions.selectSatellite(state.selection.id);
  };
}

function renderTimeline(el, state, actions) {
  const steps = state.scenario.steps || 1;
  const now = state.step;
  const view = state.previewStep ?? state.step;
  const nowT = now / steps;
  const ballT = view / steps;
  const events = operatorEvents(state);
  const marks = timelineMarks();
  el.innerHTML = `
    <div class="timeline-head">
      <span>Ожидается <span class="num">${steps}</span> шагов · Меток событий <span class="num">${marks.length}</span></span>
      <span class="num">${formatClock(view)}</span>
    </div>
    <div class="timeline-track" data-track>
      ${buildSignalSvg(nowT, ballT, events, steps)}
      ${marks.map((m) => `
        <button type="button" class="tl-dot tone-${m.tone}" style="left:${m.left * 100}%" title="${m.title}"></button>
      `).join('')}
      <div class="tl-stick" style="left:${nowT * 100}%"></div>
      <div class="tl-ball" style="left:${ballT * 100}%"></div>
    </div>
    <div class="timeline-hint">Текущий шаг <span class="num">${now}</span> · Вы навелись на <span class="num">${view}</span> шаг</div>
  `;

  const track = el.querySelector('[data-track]');
  const setFromEvent = (e) => {
    const rect = track.getBoundingClientRect();
    const t = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    actions.previewStep(Math.round(t * steps));
  };
  track.addEventListener('click', setFromEvent);
  track.querySelectorAll('.tl-dot').forEach((btn) => {
    btn.addEventListener('click', (e) => e.stopPropagation());
  });
}
