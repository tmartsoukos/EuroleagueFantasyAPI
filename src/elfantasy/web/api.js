// Πελάτης του API (ίδιο origin, άρα χωρίς CORS). Το δωρεάν πλάνο του Render «κοιμίζει» την
// υπηρεσία: το πρώτο αίτημα μπορεί να πάρει έως ένα λεπτό, γι' αυτό υπάρχει ειδοποίηση «ξυπνάει».

const TIMEOUT_MS = 90000;
const SLOW_MS = 3500;
const RETRY_DELAY_MS = 3000;
const MAX_RETRIES = 2;

let activeSlow = 0;
let wakeListener = () => {};

export function onWake(listener) {
  wakeListener = listener;
}

export class ApiError extends Error {
  constructor(status, detail) {
    super(formatDetail(detail, status));
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }
}

function formatDetail(detail, status) {
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) return detail.map((item) => item.msg ?? JSON.stringify(item)).join('; ');
  return `HTTP ${status}`;
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export async function request(path, { method = 'GET', body, adminKey, okStatuses = [] } = {}) {
  const headers = { Accept: 'application/json' };
  if (body !== undefined) headers['Content-Type'] = 'application/json';
  if (adminKey) headers['X-API-Key'] = adminKey;

  let slow = false;
  const slowTimer = setTimeout(() => {
    slow = true;
    activeSlow += 1;
    wakeListener(true);
  }, SLOW_MS);

  try {
    for (let attempt = 0; ; attempt += 1) {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), TIMEOUT_MS);
      try {
        const response = await fetch(path, {
          method,
          headers,
          body: body === undefined ? undefined : JSON.stringify(body),
          signal: controller.signal,
          cache: 'no-store',
        });
        if (response.status === 204) return null;
        const isJson = (response.headers.get('content-type') ?? '').includes('json');
        const data = isJson ? await response.json() : null;
        if (response.ok || okStatuses.includes(response.status)) return data;
        const transient = response.status === 502 || response.status === 504;
        if (method === 'GET' && transient && attempt < MAX_RETRIES) {
          await delay(RETRY_DELAY_MS);
          continue;
        }
        throw new ApiError(response.status, data?.detail);
      } catch (error) {
        if (error instanceof ApiError) throw error;
        if (method === 'GET' && attempt < MAX_RETRIES) {
          await delay(RETRY_DELAY_MS);
          continue;
        }
        throw new ApiError(0, error.name === 'AbortError' ? 'timeout' : 'network');
      } finally {
        clearTimeout(timeout);
      }
    }
  } finally {
    clearTimeout(slowTimer);
    if (slow) {
      activeSlow -= 1;
      wakeListener(activeSlow > 0);
    }
  }
}

export function describeError(error) {
  if (!(error instanceof ApiError)) return 'Απρόβλεπτο σφάλμα.';
  switch (error.status) {
    case 0:
      return error.detail === 'timeout'
        ? 'Ο server άργησε πολύ να απαντήσει. Δοκίμασε ξανά.'
        : 'Δεν υπάρχει σύνδεση με τον server.';
    case 401:
      return 'Λάθος ή ελλιπές κλειδί διαχειριστή.';
    case 404:
      return 'Δεν βρέθηκε (άγνωστος παίκτης ή ομάδα).';
    case 413:
      return 'Το αίτημα είναι πολύ μεγάλο.';
    case 422:
      return `Μη έγκυρα στοιχεία: ${error.message}`;
    case 503:
      if (typeof error.detail === 'string' && error.detail.includes('admin API is disabled')) {
        return 'Το API διαχείρισης είναι κλειστό (δεν έχει οριστεί κλειδί στον server).';
      }
      return 'Η υπηρεσία δεν είναι διαθέσιμη αυτή τη στιγμή.';
    default:
      return `Σφάλμα του server (${error.status}).`;
  }
}
