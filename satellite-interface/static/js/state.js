export function createState(snapshot) {
  return {
    scenario: snapshot.scenario,
    model: snapshot.model,
    step: snapshot.step,
    running: false,
    objective: snapshot.objective || 'priorities',
    satellites: snapshot.satellites,
    jobs: snapshot.jobs,
    events: snapshot.events,
    metrics: snapshot.metrics,
    selection: { kind: 'satellite', id: snapshot.satellites[0].id },
    satFilter: 'all',
    satQuery: '',
    jobFilter: 'all',
    jobQuery: '',
    inspectorTab: 'satellite',
    metricsOpen: false,
    previewStep: null,
    overflowOpen: false,
    loadOpen: false,
    eventOpen: false,
    eventMenuOpen: false,
    eventComposeAt: null,
    compareOpen: false,
  };
}

export function selectSatellite(state, id) {
  state.selection = { kind: 'satellite', id };
  state.inspectorTab = 'satellite';
}

export function selectJob(state, id) {
  state.selection = { kind: 'job', id };
  state.inspectorTab = 'job';
}

export function selectEvent(state, id) {
  state.selection = { kind: 'event', id };
  state.inspectorTab = 'event';
}

export function selectedSatellite(state) {
  const id = state.selection.kind === 'satellite' ? state.selection.id : null;
  return state.satellites.find((s) => s.id === id) || null;
}

export function operatorEvents(state) {
  return state.events
    .filter((ev) => ev.origin === 'operator')
    .sort((a, b) => a.seq - b.seq);
}

function nextEventId(events) {
  let max = 0;
  for (const ev of events) {
    const m = /^E-(\d+)$/.exec(ev.id || '');
    if (m) max = Math.max(max, Number(m[1]));
  }
  return `E-${String(max + 1).padStart(2, '0')}`;
}

function nextOperatorSeq(events) {
  let max = 0;
  for (const ev of events) {
    if (ev.origin === 'operator') max = Math.max(max, ev.seq || 0);
  }
  return max + 1;
}

export function addOperatorEvent(state, type) {
  if (state.eventComposeAt !== state.step) return;
  const id = nextEventId(state.events);
  const at_step = state.step;
  const sat = selectedSatellite(state)?.id || state.satellites[0]?.id || 'S01';
  const end = Math.min(state.scenario.steps, at_step + 24);
  let ev;
  if (type === 'add_jobs') {
    ev = {
      id,
      at_step,
      type,
      payload: {
        id,
        at_step,
        type,
        jobs: [{
          id: `URG-${id}`,
          kind: 'relay',
          release_step: at_step,
          deadline_step: Math.min(state.scenario.steps, at_step + 8),
          work_steps: 3,
          eligible_satellites: [sat],
          priority: 3,
          value_usd: 40,
        }],
      },
      reaction: {
        accepted: true,
        changed: 'В план добавлено 1 задание приоритета 3',
        kept: '—',
        lost: '—',
      },
    };
  } else if (type === 'satellite_outage') {
    ev = {
      id,
      at_step,
      type,
      payload: { id, at_step, type, satellite_ids: [sat], end_step: end },
      reaction: {
        accepted: true,
        changed: `${sat} исключён из назначения до шага ${end}`,
        kept: '—',
        lost: '—',
      },
    };
  } else if (type === 'close_downlink') {
    ev = {
      id,
      at_step,
      type,
      payload: { id, at_step, type, satellite_ids: [sat], end_step: end },
      reaction: {
        accepted: true,
        changed: `Сеанс ${sat} закрыт до шага ${end}`,
        kept: '—',
        lost: '—',
      },
    };
  } else {
    return;
  }
  ev.origin = 'operator';
  ev.seq = nextOperatorSeq(state.events);
  state.events.push(ev);
  selectEvent(state, id);
  state.eventMenuOpen = false;
}

export function cancelOperatorEvent(state, id) {
  const i = state.events.findIndex((ev) => ev.id === id && ev.origin === 'operator');
  if (i < 0) return;
  state.events.splice(i, 1);
  const rest = operatorEvents(state);
  if (state.selection.kind === 'event' && state.selection.id === id) {
    const last = rest[rest.length - 1];
    state.selection = last ? { kind: 'event', id: last.id } : { kind: 'event', id: null };
  }
  state.eventMenuOpen = false;
}
