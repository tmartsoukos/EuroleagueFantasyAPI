// Είσοδος της εφαρμογής: πλοήγηση με hash (#/rankings, #/player/P003469, #/lineup, #/admin),
// κατάσταση υπηρεσίας, θέμα και service worker.

import { onWake, request } from './api.js';
import { fmtShortDate, h } from './util.js';
import { createAdminView } from './view-admin.js';
import { createLineupView } from './view-lineup.js';
import { createPlayerView } from './view-player.js';
import { createRankingsView } from './view-rankings.js';

const PLAYER_ID = /^P[A-Z0-9]{3,6}$/i;
const TITLES = {
  rankings: 'Κατάταξη',
  player: 'Παίκτης',
  lineup: 'Ρόστερ',
  admin: 'Διαχείριση',
};

const main = document.getElementById('main');
const healthPill = document.getElementById('health');
const wakeBanner = document.getElementById('wake');

const views = {
  rankings: createRankingsView(),
  player: createPlayerView(),
  lineup: createLineupView(),
  admin: createAdminView({ onDataChanged: refreshHealth }),
};

for (const view of Object.values(views)) {
  view.el.hidden = true;
  main.append(view.el);
}

onWake((slow) => {
  wakeBanner.hidden = !slow;
});

function parseHash() {
  const [name, arg] = location.hash.replace(/^#\/?/, '').split('/');
  const view = Object.hasOwn(views, name) ? name : 'rankings';
  let decoded;
  try {
    decoded = arg ? decodeURIComponent(arg).trim().toUpperCase() : undefined;
  } catch {
    decoded = undefined; // κακή ακολουθία %XX στο URL
  }
  return { view, arg: view === 'player' && decoded && PLAYER_ID.test(decoded) ? decoded : undefined };
}

let firstRoute = true;

async function route() {
  const { view, arg } = parseHash();
  for (const [name, entry] of Object.entries(views)) entry.el.hidden = name !== view;
  for (const link of document.querySelectorAll('.tabs a')) {
    const active = link.dataset.view === view;
    link.classList.toggle('active', active);
    if (active) link.setAttribute('aria-current', 'page');
    else link.removeAttribute('aria-current');
  }
  document.title = `${TITLES[view]} · Euroleague Fantasy`;
  if (!firstRoute) main.focus({ preventScroll: true });
  firstRoute = false;
  window.scrollTo(0, 0);
  await views[view].enter(arg);
}

async function refreshHealth() {
  try {
    const data = await request('/health', { okStatuses: [503] });
    const ok = data.status === 'ok';
    healthPill.className = `pill ${ok ? 'ok' : 'bad'}`;
    healthPill.replaceChildren(
      ok ? 'Online' : 'Εκτός λειτουργίας',
      ok ? h('span', { class: 'wide-only' }, ` · δεδομένα έως ${fmtShortDate(data.data_loaded_through)}`) : '',
    );
    const model = data.model;
    healthPill.title = model
      ? `Μοντέλο ${model.version} · MAE ${model.test_mae?.toFixed(2)} · εκπαίδευση έως ${model.trained_through_season}`
      : (data.problems ?? []).join(' · ');
  } catch {
    healthPill.className = 'pill bad';
    healthPill.textContent = 'Χωρίς σύνδεση';
    healthPill.title = '';
  }
}

function setupTheme() {
  const button = document.getElementById('theme-toggle');
  button.addEventListener('click', () => {
    const root = document.documentElement;
    const current =
      root.dataset.theme ?? (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
    const next = current === 'dark' ? 'light' : 'dark';
    root.dataset.theme = next;
    try {
      localStorage.setItem('elfantasy.theme', next);
    } catch {
      // Η επιλογή ισχύει μόνο για αυτή τη συνεδρία.
    }
  });
}

function registerServiceWorker() {
  const secure = location.protocol === 'https:' || location.hostname === 'localhost';
  if ('serviceWorker' in navigator && secure) {
    navigator.serviceWorker.register('sw.js').catch(() => {});
  }
}

window.addEventListener('hashchange', route);
setupTheme();
registerServiceWorker();
refreshHealth();
route();
