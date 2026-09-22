import { buildSnapshot } from '../../src/mocks/snapshot.js';

export async function loadSession() {
  return buildSnapshot();
}

export async function postScenario(scenario) {
  const res = await fetch('/api/scenario', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(scenario),
  });
  if (!res.ok) {
    let detail = '';
    try {
      const body = await res.json();
      detail = typeof body.detail === 'string' ? body.detail : '';
    } catch {
      detail = '';
    }
    throw new Error(detail || 'Сервер не принял сценарий');
  }
  return res.json();
}

export async function fetchOrbits(step) {
  const res = await fetch(`/api/orbits?step=${encodeURIComponent(step)}`);
  if (!res.ok) throw new Error('Орбиты недоступны');
  return res.json();
}

let active = null;

export function trackSnapshot() {
  return active;
}

export function primeTracks() {
  if (active) active.abort.abort();
  const state = {
    abort: new AbortController(),
    frames: new Map(),
    orbits: null,
    ids: [],
    listeners: new Set(),
    failed: false,
  };
  active = state;
  readTracks(state).catch((err) => {
    if (err && err.name === 'AbortError') return;
    if (active !== state) return;
    state.failed = true;
    state.listeners.forEach((fn) => fn({ type: 'error' }));
  });
}

export function onTrack(fn) {
  if (!active) primeTracks();
  active.listeners.add(fn);
  return () => {
    if (active) active.listeners.delete(fn);
  };
}

async function readTracks(state) {
  const res = await fetch('/api/tracks', { signal: state.abort.signal });
  if (!res.ok || !res.body) throw new Error('Орбиты недоступны');
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let nl = buf.indexOf('\n');
    while (nl >= 0) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      nl = buf.indexOf('\n');
      if (!line || active !== state) continue;
      const msg = JSON.parse(line);
      if (msg.type === 'meta') state.ids = msg.ids;
      else if (msg.type === 'frame') state.frames.set(msg.step, msg);
      else if (msg.type === 'orbits') state.orbits = msg;
      state.listeners.forEach((fn) => fn(msg));
    }
  }
}
