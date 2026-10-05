// Κοινή κατάσταση: κατάταξη (μία φόρτωση για όλους τους ενεργούς παίκτες), ρόστερ και ειδοποιήσεις.

import { request } from './api.js';
import { emptyLineup, sanitize } from './lineup.js';

const LINEUP_KEY = 'elfantasy.lineup.v1';

export const store = {
  items: [],
  byId: new Map(),
  meta: null,
  lineup: emptyLineup(),
};

let rankingsPromise = null;
let rawLineup = readLineup();
const lineupListeners = new Set();

function readLineup() {
  try {
    return JSON.parse(localStorage.getItem(LINEUP_KEY));
  } catch {
    return null;
  }
}

function writeLineup() {
  try {
    localStorage.setItem(LINEUP_KEY, JSON.stringify(store.lineup));
  } catch {
    // Ιδιωτικό παράθυρο ή αποκλεισμένο storage: το ρόστερ ισχύει μόνο για αυτή τη συνεδρία.
  }
}

// Όλοι οι ενεργοί παίκτες (262 στο 2026) σε ένα αίτημα· τα φίλτρα γίνονται στον client.
export function loadRankings() {
  if (!rankingsPromise) {
    rankingsPromise = request('/rankings?limit=500&include_unavailable=true')
      .then((data) => {
        store.items = data.items;
        store.byId = new Map(data.items.map((item) => [item.player_id, item]));
        store.meta = data.meta;
        store.lineup = sanitize(rawLineup ?? store.lineup, store.byId);
        rawLineup = null;
        writeLineup();
        return data;
      })
      .catch((error) => {
        rankingsPromise = null;
        throw error;
      });
  }
  return rankingsPromise;
}

// Μετά από αλλαγή διαθεσιμότητας ή ανανέωση δεδομένων η επόμενη φόρτωση φέρνει νέα κατάταξη.
export function invalidateRankings() {
  rawLineup = store.lineup;
  rankingsPromise = null;
}

export function onLineupChange(listener) {
  lineupListeners.add(listener);
}

export function setLineup(lineup) {
  store.lineup = lineup;
  writeLineup();
  for (const listener of lineupListeners) listener(lineup);
}

let toastTimer = null;

export function toast(message, kind = 'info') {
  const el = document.getElementById('toast');
  el.textContent = message;
  el.className = `toast ${kind}`;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    el.hidden = true;
  }, 4000);
}
