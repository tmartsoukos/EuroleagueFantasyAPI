// Οθόνη «Κατάταξη»: όλοι οι ενεργοί παίκτες κατά προβλεπόμενο fantasy score.

import { describeError } from './api.js';
import { addPlayer } from './lineup.js';
import { loadRankings, setLineup, store, toast } from './store.js';
import {
  clear,
  fmt,
  fmtShortDate,
  h,
  matchesQuery,
  norm,
  opponentLabel,
  prettyName,
  statusBadge,
} from './util.js';

const PAGE = 50;

const BLOCKERS = {
  duplicate: 'Ο παίκτης είναι ήδη στο ρόστερ.',
  full: 'Το ρόστερ είναι γεμάτο.',
  'team-limit': 'Έως 6 παίκτες από την ίδια ομάδα.',
};

export function addToLineup(player) {
  const { lineup, error } = addPlayer(store.lineup, player, store.byId);
  if (error) {
    toast(BLOCKERS[error] ?? 'Δεν μπορεί να προστεθεί.', 'error');
    return false;
  }
  setLineup(lineup);
  toast(`${prettyName(player.name)}: προστέθηκε στο ρόστερ.`, 'ok');
  return true;
}

export function createRankingsView() {
  const state = { team: '', query: '', showOut: false, shown: PAGE };
  const el = h('section', { class: 'view', 'aria-labelledby': 'rankings-title' });

  const title = h('h2', { id: 'rankings-title' }, 'Κατάταξη προβλέψεων');
  const intro = h('p', { class: 'muted' });
  const search = h('input', {
    type: 'search',
    placeholder: 'Αναζήτηση παίκτη…',
    'aria-label': 'Αναζήτηση παίκτη',
    autocomplete: 'off',
    oninput: (event) => {
      state.query = event.target.value;
      state.shown = PAGE;
      render();
    },
  });
  const teamSelect = h('select', {
    'aria-label': 'Φίλτρο ομάδας',
    onchange: (event) => {
      state.team = event.target.value;
      state.shown = PAGE;
      render();
    },
  });
  const outToggle = h('input', {
    type: 'checkbox',
    onchange: (event) => {
      state.showOut = event.target.checked;
      state.shown = PAGE;
      render();
    },
  });
  const controls = h(
    'div',
    { class: 'controls' },
    search,
    teamSelect,
    h('label', { class: 'check' }, outToggle, ' Και οι εκτός (out)'),
  );
  const body = h('div', { class: 'view-body' });
  el.append(title, intro, controls, body);

  function populateTeams() {
    const codes = [...new Set(store.items.map((item) => item.team_code))].sort();
    teamSelect.replaceChildren(
      h('option', { value: '' }, 'Όλες οι ομάδες'),
      ...codes.map((code) => h('option', { value: code }, code)),
    );
    teamSelect.value = state.team;
  }

  function filtered() {
    return store.items.filter((item) => {
      if (state.team && item.team_code !== state.team) return false;
      if (!state.showOut && item.availability_status === 'out') return false;
      return !state.query || matchesQuery(norm(item.name), state.query);
    });
  }

  function row(item, maxScore) {
    const bar = h('span', { class: 'bar' });
    bar.style.width = `${Math.max(0, Math.min(100, (item.predicted_fantasy / maxScore) * 100))}%`;
    return h(
      'tr',
      { class: item.availability_status === 'out' ? 'is-out' : null },
      h('td', { class: 'num muted' }, item.rank),
      h(
        'th',
        { scope: 'row' },
        h(
          'a',
          { class: 'player-link', href: `#/player/${encodeURIComponent(item.player_id)}` },
          prettyName(item.name),
        ),
        h('span', { class: 'sub' }, item.team_code, ' ', statusBadge(item.availability_status)),
      ),
      h(
        'td',
        { class: 'hide-sm' },
        opponentLabel(item.next_opponent_code, item.next_home),
        h('span', { class: 'sub' }, fmtShortDate(item.next_game_date)),
      ),
      h('td', { class: 'score' }, h('span', { class: 'num strong' }, fmt(item.predicted_fantasy)), bar),
      h('td', { class: 'num hide-sm' }, fmt(item.predicted_pir)),
      h(
        'td',
        { class: 'act' },
        h(
          'button',
          {
            type: 'button',
            class: 'icon-btn',
            title: 'Προσθήκη στο ρόστερ',
            'aria-label': `Προσθήκη του ${prettyName(item.name)} στο ρόστερ`,
            onclick: () => addToLineup(item),
          },
          '+',
        ),
      ),
    );
  }

  function render() {
    const items = filtered();
    const visible = items.slice(0, state.shown);
    const maxScore = Math.max(1, ...store.items.map((item) => item.predicted_fantasy));
    clear(body);
    if (items.length === 0) {
      body.append(h('p', { class: 'empty' }, 'Κανένας παίκτης δεν ταιριάζει με τα φίλτρα.'));
      return;
    }
    body.append(
      h(
        'div',
        { class: 'table-wrap' },
        h(
          'table',
          { class: 'data' },
          h(
            'thead',
            {},
            h(
              'tr',
              {},
              h('th', { scope: 'col' }, '#'),
              h('th', { scope: 'col' }, 'Παίκτης'),
              h('th', { scope: 'col', class: 'hide-sm' }, 'Επόμενος αγώνας'),
              h('th', { scope: 'col' }, 'Fantasy'),
              h('th', { scope: 'col', class: 'hide-sm' }, 'PIR'),
              h('th', { scope: 'col' }, h('span', { class: 'sr-only' }, 'Ενέργειες')),
            ),
          ),
          h('tbody', {}, visible.map((item) => row(item, maxScore))),
        ),
      ),
      h('p', { class: 'muted small' }, `Εμφανίζονται ${visible.length} από ${items.length} παίκτες.`),
    );
    if (items.length > visible.length) {
      body.append(
        h(
          'button',
          {
            type: 'button',
            class: 'btn',
            onclick: () => {
              state.shown += PAGE;
              render();
            },
          },
          'Περισσότεροι παίκτες',
        ),
      );
    }
  }

  function showError(error) {
    clear(body);
    body.append(
      h(
        'div',
        { class: 'error-box', role: 'alert' },
        h('p', {}, describeError(error)),
        h('button', { type: 'button', class: 'btn', onclick: () => enter() }, 'Δοκίμασε ξανά'),
      ),
    );
  }

  async function enter() {
    if (store.items.length === 0) {
      clear(body);
      body.append(h('p', { class: 'loading' }, 'Φόρτωση κατάταξης…'));
    }
    try {
      const data = await loadRankings();
      intro.textContent =
        `Προβλεπόμενο fantasy score για τον επόμενο αγώνα κάθε παίκτη (${data.meta.total} ενεργοί, ` +
        `πρόβλεψη της ${fmtShortDate(data.meta.as_of)}). Χωρίς captain ή πάγκο.`;
      populateTeams();
      render();
    } catch (error) {
      showError(error);
    }
  }

  return { el, enter };
}
