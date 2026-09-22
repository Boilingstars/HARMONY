/** Декоративная кривая выгодности политики и метки — визуал, не расчёт.
 *  Форма как на макете: палочка ≈ 42% кадра.
 *  После операторских событий ключи пересобираются — «перерасчёт» плана. */

const SPLIT = 0.42;

const KEYS = [
  [0.00, 0.86], [0.05, 0.80], [0.09, 0.74], [0.13, 0.32],
  [0.17, 0.18], [0.21, 0.16], [0.26, 0.52], [0.32, 0.84],
  [0.37, 0.92], [0.41, 0.78], [0.44, 0.36], [0.48, 0.24],
  [0.52, 0.48], [0.56, 0.20], [0.62, 0.36], [0.68, 0.46],
  [0.74, 0.28], [0.80, 0.38], [0.86, 0.26], [0.93, 0.34],
  [1.00, 0.42],
];

export const SIGNAL_MARKS = [
  { t: 0.11, tone: 'revenue', title: 'Выручка превысила порог' },
  { t: 0.20, tone: 'overdue', title: 'Просроченных заданий больше допустимого' },
  { t: 0.39, tone: 'jobs', title: 'Выполнено 80% исходных заданий' },
  { t: 0.49, tone: 'jobs', title: 'Выполнено 80% исходных заданий' },
  { t: 0.54, tone: 'overdue', title: 'Просроченных заданий больше допустимого' },
  { t: 0.72, tone: 'revenue', title: 'Выручка превысила порог' },
  { t: 0.82, tone: 'soc', title: 'КА ниже 30% заряда' },
];

function lerpKeys(t, keys = KEYS) {
  const x = Math.max(0, Math.min(1, t));
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

/** Кадр макета → доля смены, чтобы палочка совпала с текущим шагом. */
export function photoToActual(photoT, nowT) {
  const n = Math.max(0, Math.min(1, nowT));
  if (photoT <= SPLIT) return n * (photoT / SPLIT);
  return n + (1 - n) * ((photoT - SPLIT) / (1 - SPLIT));
}

function actualToPhoto(t, nowT) {
  const n = Math.max(0, Math.min(0.999, nowT));
  if (t <= n) return SPLIT * (t / (n || 1e-6));
  return SPLIT + (1 - SPLIT) * ((t - n) / (1 - n || 1e-6));
}

function hashEvents(events) {
  let h = 2166136261;
  for (const ev of events) {
    const s = `${ev.seq}:${ev.type}:${ev.at_step}`;
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

function eventBias(actual, events, steps) {
  let y = 0;
  for (const ev of events) {
    const et = (ev.at_step || 0) / (steps || 1);
    const dt = actual - et;
    const width = ev.type === 'add_jobs' ? 0.07 : 0.1;
    const amp = ev.type === 'add_jobs' ? 0.28
      : ev.type === 'satellite_outage' ? -0.32
      : -0.2;
    y += amp * Math.exp(-(dt * dt) / (2 * width * width));
    if (dt > 0) y += amp * 0.22 * Math.exp(-dt / 0.28);
  }
  return y;
}

function recastKeys(events, nowT, steps) {
  if (!events.length) return KEYS;
  const rnd = mulberry32(hashEvents(events));
  const phase = rnd() * Math.PI * 2;
  const freq = 6 + rnd() * 9;
  const amp = 0.24 + rnd() * 0.2;
  const phase2 = rnd() * Math.PI * 2;
  return KEYS.map(([t, y0]) => {
    const actual = photoToActual(t, nowT);
    const k = actual <= nowT ? 0.14 : 1;
    const wave = Math.sin(actual * freq + phase) * amp
      + Math.sin(actual * freq * 1.65 + phase2) * amp * 0.5;
    let y = y0 + wave * k + eventBias(actual, events, steps) * k;
    y = Math.max(0.08, Math.min(0.96, y));
    return [t, y];
  });
}

export function curveHeight(t, nowT, keys = KEYS) {
  return lerpKeys(actualToPhoto(t, nowT), keys);
}

const W = 1000;
const H = 48;
const BASE = 41;
const TOP = 4;

export function curveY(t, nowT, keys = KEYS) {
  return BASE - curveHeight(t, nowT, keys) * (BASE - TOP);
}

export function timelineMarks() {
  return SIGNAL_MARKS.map((m) => ({
    left: m.t,
    tone: m.tone,
    title: m.title,
  }));
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
  const keys = recastKeys(events, nowT, steps);
  const n = 64;
  const pts = [];
  for (let i = 0; i <= n; i += 1) {
    const t = i / n;
    pts.push({ x: t * W, y: curveY(t, nowT, keys) });
  }
  const line = catmull(pts);
  const fill = `${line} L ${W} ${BASE} L 0 ${BASE} Z`;
  const nowX = Math.max(0, Math.min(1, nowT)) * W;
  const ballX = Math.max(0, Math.min(1, ballT)) * W;
  const wake = ballX > nowX + 1.5;
  const marks = timelineMarks();
  const stems = marks.map((m) => {
    const x = m.left * W;
    return `<line class="tl-stem tone-${m.tone}" x1="${x.toFixed(1)}" y1="${curveY(m.left, nowT, keys).toFixed(1)}" x2="${x.toFixed(1)}" y2="${BASE}" />`;
  }).join('');

  return `
    <svg class="tl-svg" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" aria-hidden="true">
      <defs>
        <linearGradient id="tl-past-grad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#b8b0f0" stop-opacity="0.7"/>
          <stop offset="100%" stop-color="#8b7fd4" stop-opacity="0.12"/>
        </linearGradient>
        <clipPath id="tl-clip-past"><rect x="0" y="0" width="${nowX.toFixed(1)}" height="${H}"/></clipPath>
        <clipPath id="tl-clip-fut"><rect x="${nowX.toFixed(1)}" y="0" width="${(W - nowX).toFixed(1)}" height="${H}"/></clipPath>
        ${wake ? `<clipPath id="tl-clip-wake"><rect x="${nowX.toFixed(1)}" y="0" width="${(ballX - nowX).toFixed(1)}" height="${H}"/></clipPath>` : ''}
      </defs>
      <line class="tl-base" x1="0" y1="${BASE}" x2="${W}" y2="${BASE}"/>
      <path d="${fill}" fill="url(#tl-past-grad)" clip-path="url(#tl-clip-past)"/>
      ${wake ? `<path d="${fill}" class="tl-wake-fill" clip-path="url(#tl-clip-wake)"/>` : ''}
      <path d="${line}" class="tl-stroke-past" clip-path="url(#tl-clip-past)"/>
      <path d="${line}" class="tl-stroke-fut" clip-path="url(#tl-clip-fut)"/>
      ${stems}
      ${wake ? `<line class="tl-wake" x1="${nowX.toFixed(1)}" y1="${BASE}" x2="${ballX.toFixed(1)}" y2="${BASE}"/>` : ''}
    </svg>
  `;
}
