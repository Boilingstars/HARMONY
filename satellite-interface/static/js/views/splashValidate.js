const SCHEMA = 'cosmo-B-ops-1.0';
const MIN_SATS = 1;
const MAX_SATS = 48;
const MIN_STEPS = 1;
const MAX_STEPS = 288;
const STEP_S = 300;

export function formatBytes(n) {
  if (n < 1024) return `${n} Б`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} КБ`;
  return `${(n / (1024 * 1024)).toFixed(1)} МБ`;
}

export function isScenarioFile(file) {
  if (!file) return false;
  const name = (file.name || '').toLowerCase();
  return name.endsWith('.json') || file.type === 'application/json';
}

export function parseScenarioText(text) {
  let data;
  try {
    data = JSON.parse(text);
  } catch {
    throw new Error('Файл не JSON');
  }
  return validateScenario(data);
}

export function validateScenario(data) {
  if (data == null || typeof data !== 'object' || Array.isArray(data)) {
    throw new Error('Сценарий должен быть JSON-объектом');
  }
  if (data.schema_version !== SCHEMA) {
    throw new Error(`Нет schema_version ${SCHEMA}`);
  }

  const satellites = data.satellites;
  if (!Array.isArray(satellites)) {
    throw new Error('В сценарии нет списка аппаратов');
  }
  if (satellites.length < MIN_SATS || satellites.length > MAX_SATS) {
    throw new Error(`Число аппаратов должно быть от ${MIN_SATS} до ${MAX_SATS}`);
  }
  if (satellites.some((s) => !s || typeof s.id !== 'string' || !s.id)) {
    throw new Error('У каждого аппарата должен быть id');
  }

  const jobs = data.jobs;
  if (jobs == null) {
    throw new Error('В сценарии нет списка заданий');
  }
  if (!Array.isArray(jobs)) {
    throw new Error('Список заданий должен быть массивом');
  }

  const steps = data.time?.steps ?? data.scenario?.steps;
  if (!Number.isInteger(steps) || steps < MIN_STEPS || steps > MAX_STEPS) {
    throw new Error(`Число шагов должно быть от ${MIN_STEPS} до ${MAX_STEPS}`);
  }
  const stepS = data.time?.step_s ?? data.scenario?.step_s;
  if (stepS != null && stepS !== STEP_S) {
    throw new Error(`Шаг времени должен быть ${STEP_S} с`);
  }

  return {
    data,
    preview: {
      satelliteCount: satellites.length,
      steps,
      jobCount: jobs.length,
      potentialRevenue: potentialRevenue(data, jobs),
    },
  };
}

function potentialRevenue(data, jobs) {
  if (typeof data.metrics?.potential_revenue_usd === 'number') {
    return data.metrics.potential_revenue_usd;
  }
  if (!jobs.length) return 0;
  if (jobs.some((j) => typeof j?.value_usd !== 'number')) return null;
  return jobs.reduce((sum, j) => sum + j.value_usd, 0);
}

export function readFileAsText(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ''));
    reader.onerror = () => reject(new Error('Не удалось прочитать файл'));
    reader.readAsText(file);
  });
}
