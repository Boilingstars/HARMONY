/** Случайный контур на всю ширину таймлайна.
 *  Шаг двигает только заливку и курсор.
 *  Операторское событие пересобирает кривую строго справа от nowT. */

const KEYS = [
  [0.00, 0.86], [0.05, 0.80], [0.09, 0.74], [0.13, 0.32],
  [0.17, 0.18], [0.21, 0.16], [0.26, 0.52], [0.32, 0.84],
  [0.37, 0.92], [0.41, 0.78], [0.44, 0.36], [0.48, 0.24],
  [0.52, 0.48], [0.56, 0.20], [0.62, 0.36], [0.68, 0.46],
  [0.74, 0.28], [0.80, 0.38], [0.86, 0.26], [0.93, 0.34],
  [1.00, 0.42],
];

export const SIGNAL_MARKS = [
  { t: 0.11, tone: 'ok', title: 'Выручка превысила порог' },
  { t: 0.20, tone: 'warn', title: 'Просроченных заданий больше допустимого' },
  { t: 0.39, tone: 'ok', title: 'Выполнено 80% исходных заданий' },
  { t: 0.49, tone: 'ok', title: 'Выполнено 80% исходных заданий' },
  { t: 0.54, tone: 'warn', title: 'Просроченных заданий больше допустимого' },
  { t: 0.72, tone: 'ok', title: 'Выручка превысила порог' },
  { t: 0.82, tone: 'warn', title: 'КА ниже 30% заряда' },
];

const W = 1000;
const H = 48;
const TOP = 8;
const BASE = 40;

let committedKeys = KEYS.map((p) => p.slice());
let eventSig = 0;
let lastSteps = null;

function clamp01(x) {
  return Math.max(0, Math.min(1, x));
}

function clampY(y) {
  return Math.max(0.08, Math.min(0.96, y));
}

function lerpKeys(t, keys = committedKeys) {
  const x = clamp01(t);
  for (let i = 0; i < keys.length - 1; i += 1) {
    const [x0, y0] = keys[i];
    const [x1, y1] = keys[i + 1];
    if (x >= x0 && x <= x1) {
      const u = (x - x0) / (x1 - x0 || 1);
      const s = u * u * (3 - 2 * u);
      return y0 + (y1 - y0) * s;
    }
  }
  return keys[keys.length - 1][1];
}

export function curveY(t, keys = committedKeys) {
  return BASE - lerpKeys(t, keys) * (BASE - TOP);
}

export function timelineMarks(steps) {
  const n = Math.max(1, steps || 1);
  const seen = new Set();
  const marks = [];
  for (const m of SIGNAL_MARKS) {
    const step = Math.max(0, Math.min(n, Math.round(m.t * n)));
    if (seen.has(step)) continue;
    seen.add(step);
    marks.push({
      step,
      left: step / n,
      tone: m.tone,
      title: m.title,
    });
  }
  return marks;
}

function hashEvents(events) {
  let h = 2166136261;
  for (const ev of events || []) {
    const s = `${ev.id || ''}:${ev.seq}:${ev.type}:${ev.at_step}`;
    for (let i = 0; i < s.length; i += 1) h = Math.imul(h ^ s.charCodeAt(i), 16777619);
  }
  return h >>> 0;
}

function mulberry32(a) {
  return () => {
    a |= 0;
    a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function eventBias(t, events, steps) {
  let y = 0;
  for (const ev of events) {
    const et = (ev.at_step || 0) / Math.max(1, steps);
    const dt = t - et;
    if (dt < -0.02) continue;
    const width = ev.type === 'add_jobs' ? 0.08 : 0.11;
    const amp = ev.type === 'add_jobs' ? 0.26
      : ev.type === 'satellite_outage' ? -0.3
      : -0.2;
    y += amp * Math.exp(-(dt * dt) / (2 * width * width));
    if (dt > 0) y += amp * 0.2 * Math.exp(-dt / 0.28);
  }
  return y;
}

function recastKeys(events, steps) {
  if (!events.length) return KEYS.map((p) => p.slice());
  const rnd = mulberry32(hashEvents(events));
  const phase = rnd() * Math.PI * 2;
  const freq = 5.5 + rnd() * 8;
  const amp = 0.2 + rnd() * 0.16;
  const phase2 = rnd() * Math.PI * 2;
  return KEYS.map(([t, y0]) => {
    const wave = Math.sin(t * freq + phase) * amp
      + Math.sin(t * freq * 1.65 + phase2) * amp * 0.45;
    return [t, clampY(y0 + wave + eventBias(t, events, steps))];
  });
}

function ensureWave(events, steps, nowT) {
  if (lastSteps !== steps) {
    committedKeys = KEYS.map((p) => p.slice());
    eventSig = hashEvents([]);
    lastSteps = steps;
  }
  const sig = hashEvents(events);
  if (sig === eventSig) return;
  const cut = clamp01(nowT);
  const yCut = lerpKeys(cut, committedKeys);
  const kept = committedKeys
    .filter(([t]) => t <= cut + 1e-6)
    .map(([t, y]) => [t, y]);
  if (!kept.length) kept.push([0, lerpKeys(0, KEYS)]);
  const lastT = kept[kept.length - 1][0];
  if (cut - lastT > 1e-4) kept.push([cut, yCut]);
  else kept[kept.length - 1] = [cut, yCut];

  const futureSrc = recastKeys(events, steps);
  const offset = yCut - lerpKeys(cut, futureSrc);
  const future = [];
  for (let t = cut + 0.04; t < 0.999; t += 0.04) {
    future.push([Number(t.toFixed(4)), clampY(lerpKeys(t, futureSrc) + offset)]);
  }
  future.push([1, clampY(lerpKeys(1, futureSrc) + offset)]);
  committedKeys = [...kept, ...future];
  eventSig = sig;
}

function catmull(pts) {
  if (!pts.length) return '';
  let d = `M ${pts[0].x.toFixed(1)} ${pts[0].y.toFixed(1)}`;
  for (let i = 0; i < pts.length - 1; i += 1) {
    const p0 = pts[Math.max(0, i - 1)];
    const p1 = pts[i];
    const p2 = pts[i + 1];
    const p3 = pts[Math.min(pts.length - 1, i + 2)];
    const c1x = p1.x + (p2.x - p0.x) / 6;
    const c1y = p1.y + (p2.y - p0.y) / 6;
    const c2x = p2.x - (p3.x - p1.x) / 6;
    const c2y = p2.y - (p3.y - p1.y) / 6;
    d += ` C ${c1x.toFixed(1)} ${c1y.toFixed(1)}, ${c2x.toFixed(1)} ${c2y.toFixed(1)}, ${p2.x.toFixed(1)} ${p2.y.toFixed(1)}`;
  }
  return d;
}

export function buildSignalSvg(nowT, ballT, events = [], steps = 1) {
  ensureWave(events, steps, nowT);
  const pts = committedKeys.map(([t, y]) => ({
    x: clamp01(t) * W,
    y: BASE - y * (BASE - TOP),
  }));
  const line = catmull(pts);
  const fill = `${line} L ${W} ${BASE} L 0 ${BASE} Z`;
  const nowX = clamp01(nowT) * W;
  const uid = `p${Math.round(nowX)}`;
  const stems = timelineMarks(steps).map((m) => {
    const x = m.left * W;
    return `<line class="tl-stem tone-${m.tone}" x1="${x.toFixed(1)}" y1="${curveY(m.left).toFixed(1)}" x2="${x.toFixed(1)}" y2="${BASE}" />`;
  }).join('');

  return `
    <svg class="tl-svg" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" aria-hidden="true">
      <defs>
        <linearGradient id="${uid}-g" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#b8b0f0" stop-opacity="0.7"/>
          <stop offset="100%" stop-color="#8b7fd4" stop-opacity="0.12"/>
        </linearGradient>
        <clipPath id="${uid}-c"><rect x="0" y="0" width="${nowX.toFixed(1)}" height="${H}"/></clipPath>
      </defs>
      <path d="${fill}" fill="url(#${uid}-g)" clip-path="url(#${uid}-c)"/>
      <path d="${line}" fill="none" stroke="#c4bdf5" stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round"/>
      ${stems}
    </svg>
  `;
}
