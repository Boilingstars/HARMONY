import {
  createState, selectSatellite, selectJob, selectEvent, addOperatorEvent, cancelOperatorEvent,
} from './state.js';
import { renderTopBar } from './layout/TopBar.js';
import { renderSatelliteTree } from './layout/SatelliteTree.js';
import { renderCenterStage } from './layout/CenterStage.js';
import { renderInspector } from './layout/Inspector.js';
import { renderMetricsBar } from './layout/MetricsBar.js';
import { renderLoadModal } from './views/LoadModal.js';
import { renderEventModal } from './views/EventModal.js';
import { renderCompareView } from './views/CompareView.js';

const els = {
  console: document.getElementById('console'),
  top: document.getElementById('zone-top'),
  left: document.getElementById('zone-left'),
  d3: document.getElementById('zone-3d'),
  timeline: document.getElementById('zone-timeline'),
  right: document.getElementById('zone-right'),
  bottom: document.getElementById('zone-bottom'),
  modal: document.getElementById('modal-root'),
  eventModal: document.getElementById('event-modal-root'),
  compare: document.getElementById('compare-root'),
};

let state;
let playTimer = null;

function stopPlayback() {
  if (playTimer) {
    clearInterval(playTimer);
    playTimer = null;
  }
  if (state) state.running = false;
}

function startPlayback() {
  if (playTimer) {
    clearInterval(playTimer);
    playTimer = null;
  }
  if (state.step >= state.scenario.steps) state.step = 0;
  state.previewStep = null;
  state.running = true;
  playTimer = setInterval(() => {
    if (state.step >= state.scenario.steps) {
      stopPlayback();
      render();
      return;
    }
    state.step += 1;
    state.previewStep = null;
    render();
  }, 450);
}

const actions = {
  selectSatellite(id) {
    selectSatellite(state, id);
    render();
  },
  selectJob(id) {
    selectJob(state, id);
    render();
  },
  selectEvent(id) {
    selectEvent(state, id);
    render();
  },
  setSatFilter(v) { state.satFilter = v; render(); },
  setSatQuery(v) { state.satQuery = v; render(); },
  setJobFilter(v) { state.jobFilter = v; render(); },
  setJobQuery(v) { state.jobQuery = v; render(); },
  setTab(v) { state.inspectorTab = v; state.eventMenuOpen = false; render(); },
  setObjective(v) { state.objective = v; render(); },
  toggleRun() {
    if (state.running) stopPlayback();
    else startPlayback();
    render();
  },
  stepForward() {
    stopPlayback();
    const next = Math.min(state.scenario.steps, state.step + 1);
    state.step = next;
    state.previewStep = null;
    state.eventComposeAt = state.step;
    state.eventMenuOpen = false;
    render();
  },
  jumpTo(n) {
    stopPlayback();
    const max = state.scenario.steps;
    const next = Number(n);
    state.step = Number.isFinite(next) ? Math.max(0, Math.min(max, Math.round(next))) : state.step;
    state.previewStep = null;
    state.overflowOpen = false;
    state.eventComposeAt = state.step;
    state.eventMenuOpen = false;
    render();
  },
  toggleOverflow() { state.overflowOpen = !state.overflowOpen; render(); },
  openLoad() { state.overflowOpen = false; state.loadOpen = true; render(); },
  closeLoad() { state.loadOpen = false; render(); },
  resetAndRun() {
    state.step = 0;
    state.previewStep = null;
    state.loadOpen = false;
    startPlayback();
    render();
  },
  exportResult() { state.overflowOpen = false; render(); },
  openEvent() { state.overflowOpen = false; state.eventOpen = true; render(); },
  closeEvent() { state.eventOpen = false; render(); },
  openCompare() { state.overflowOpen = false; state.running = false; state.compareOpen = true; render(); },
  closeCompare() { state.compareOpen = false; render(); },
  toggleMetrics() {
    state.metricsOpen = !state.metricsOpen;
    els.console.classList.toggle('is-metrics-open', state.metricsOpen);
    render();
  },
  previewStep(n) {
    stopPlayback();
    const max = state.scenario.steps;
    const next = Number(n);
    state.previewStep = Number.isFinite(next) ? Math.max(0, Math.min(max, Math.round(next))) : state.step;
    render();
  },
  toggleEventMenu() {
    if (state.eventComposeAt !== state.step) return;
    state.eventMenuOpen = !state.eventMenuOpen;
    render();
  },
  addEvent(type) {
    addOperatorEvent(state, type);
    render();
  },
  cancelEvent(id) {
    cancelOperatorEvent(state, id);
    render();
  },
};

window.selectSatellite = (id) => actions.selectSatellite(id);

function render() {
  const active = document.activeElement;
  const restoreSatQ = active && active.matches && active.matches('[data-q]');
  const restoreJobQ = active && active.matches && active.matches('[data-job-q]');
  const restoreQ = restoreSatQ || restoreJobQ;
  const caret = restoreQ ? active.selectionStart : null;

  renderTopBar(els.top, state, actions);
  renderSatelliteTree(els.left, state, actions);
  renderCenterStage(state, actions, els.d3, els.timeline);
  renderInspector(els.right, state, actions);
  renderMetricsBar(els.bottom, state, actions);
  renderLoadModal(els.modal, state, actions);
  renderEventModal(els.eventModal, state, actions);
  renderCompareView(els.compare, state, actions);

  if (restoreQ) {
    const input = restoreJobQ
      ? els.right.querySelector('[data-job-q]')
      : els.left.querySelector('[data-q]');
    if (input) {
      input.focus();
      input.setSelectionRange(caret, caret);
    }
  }
}

export function startConsole(snapshot) {
  state = createState(snapshot);
  window.__consoleState = state;
  els.console.hidden = false;
  render();
  window.addEventListener('resize', () => renderCenterStage(state, actions, els.d3, els.timeline));
}
