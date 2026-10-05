// Οθόνη «Ρόστερ»: 10 παίκτες (πεντάδα, έκτος, πάγκος), captain ×2 και συνολική πρόβλεψη.

import { describeError } from './api.js';
import {
  BENCH_MULTIPLIER,
  CAPTAIN_MULTIPLIER,
  MAX_PER_TEAM,
  SIXTH_INDEX,
  STARTER_COUNT,
  addBlocker,
  addPlayer,
  autoFill,
  emptyLineup,
  filledCount,
  lineupTotal,
  removeSlot,
  setCaptain,
  slotRole,
  slotWeight,
  teamCounts,
} from './lineup.js';
import { loadRankings, onLineupChange, setLineup, store, toast } from './store.js';
import { clear, fmt, h, matchesQuery, norm, prettyName, statusBadge } from './util.js';

const PICKER_LIMIT = 40;

const SECTIONS = [
  { title: 'Βασική πεντάδα', note: '100% · ο captain διπλασιάζεται', start: 0, end: STARTER_COUNT },
  { title: 'Έκτος παίκτης', note: '100%', start: SIXTH_INDEX, end: SIXTH_INDEX + 1 },
  { title: 'Πάγκος', note: '50%', start: SIXTH_INDEX + 1, end: 10 },
];

const SLOT_LABELS = {
  starter: (i) => `Βασικός ${i + 1}`,
  sixth: () => 'Έκτος παίκτης',
  bench: (i) => `Πάγκος ${i - SIXTH_INDEX}`,
};

function multiplierLabel(weight) {
  if (weight === CAPTAIN_MULTIPLIER) return '×2';
  return weight === BENCH_MULTIPLIER ? '×0,5' : '×1';
}

export function createLineupView() {
  const el = h('section', { class: 'view', 'aria-labelledby': 'lineup-title' });
  const body = h('div', { class: 'view-body' });
  const picker = h('dialog', { class: 'picker', 'aria-labelledby': 'picker-title' });
  document.body.append(picker);
  el.append(h('h2', { id: 'lineup-title' }, 'Ρόστερ'), body);

  picker.addEventListener('click', (event) => {
    if (event.target === picker) picker.close();
  });

  function openPicker(slotIndex) {
    const search = h('input', {
      type: 'search',
      placeholder: 'Αναζήτηση παίκτη…',
      'aria-label': 'Αναζήτηση παίκτη',
      autocomplete: 'off',
      oninput: () => renderList(),
    });
    const list = h('ul', { class: 'pick-list' });

    function renderList() {
      const items = store.items
        .filter((p) => p.availability_status !== 'out')
        .filter((p) => !search.value || matchesQuery(norm(p.name), search.value))
        .slice(0, PICKER_LIMIT);
      list.replaceChildren(
        ...items.map((player) => {
          const blocker = addBlocker(store.lineup, player, store.byId, slotIndex);
          const reason =
            blocker === 'team-limit'
              ? `Έως ${MAX_PER_TEAM} παίκτες από την ίδια ομάδα`
              : blocker === 'duplicate'
                ? 'Ήδη στο ρόστερ'
                : null;
          return h(
            'li',
            {},
            h(
              'button',
              {
                type: 'button',
                class: 'pick-row',
                disabled: blocker !== null,
                title: reason,
                onclick: () => {
                  const { lineup } = addPlayer(store.lineup, player, store.byId, slotIndex);
                  setLineup(lineup);
                  picker.close();
                },
              },
              h('span', { class: 'strong' }, prettyName(player.name)),
              h('span', { class: 'sub' }, player.team_code, ' ', statusBadge(player.availability_status)),
              h('span', { class: 'num' }, fmt(player.predicted_fantasy)),
            ),
          );
        }),
      );
      if (items.length === 0) list.append(h('li', { class: 'empty' }, 'Κανένας παίκτης.'));
    }

    picker.replaceChildren(
      h(
        'div',
        { class: 'picker-head' },
        h('h3', { id: 'picker-title' }, `Επιλογή παίκτη · ${SLOT_LABELS[slotRole(slotIndex)](slotIndex)}`),
        h('button', { type: 'button', class: 'icon-btn', 'aria-label': 'Κλείσιμο', onclick: () => picker.close() }, '×'),
      ),
      search,
      list,
    );
    renderList();
    picker.showModal();
    search.focus();
  }

  function slotRow(index) {
    const id = store.lineup.slots[index];
    const player = id === null ? null : store.byId.get(id);
    const label = SLOT_LABELS[slotRole(index)](index);
    if (!player) {
      return h(
        'li',
        { class: 'slot empty-slot' },
        h('span', { class: 'slot-label' }, label),
        h('button', { type: 'button', class: 'btn', onclick: () => openPicker(index) }, '+ Επιλογή παίκτη'),
      );
    }
    const isCaptain = id === store.lineup.captain;
    const weight = slotWeight(index, isCaptain);
    const name = prettyName(player.name);
    return h(
      'li',
      { class: 'slot' },
      h('span', { class: 'slot-label' }, label),
      h(
        'span',
        { class: 'slot-player' },
        h('a', { class: 'player-link', href: `#/player/${encodeURIComponent(id)}` }, name),
        h('span', { class: 'sub' }, player.team_code, ' ', statusBadge(player.availability_status)),
      ),
      h(
        'span',
        { class: 'slot-points' },
        h('span', { class: 'num strong' }, fmt(player.predicted_fantasy * weight)),
        h('span', { class: 'sub' }, `${fmt(player.predicted_fantasy)} ${multiplierLabel(weight)}`),
      ),
      index < STARTER_COUNT
        ? h(
            'button',
            {
              type: 'button',
              class: `icon-btn captain${isCaptain ? ' on' : ''}`,
              'aria-pressed': String(isCaptain),
              title: 'Captain (×2)',
              'aria-label': `Captain: ${name}`,
              onclick: () => setLineup(setCaptain(store.lineup, id)),
            },
            'C',
          )
        : h('span', { class: 'icon-spacer' }),
      h(
        'button',
        {
          type: 'button',
          class: 'icon-btn',
          title: 'Αφαίρεση',
          'aria-label': `Αφαίρεση του ${name}`,
          onclick: () => setLineup(removeSlot(store.lineup, index)),
        },
        '×',
      ),
    );
  }

  function summary() {
    const lineup = store.lineup;
    const counts = [...teamCounts(lineup, store.byId).entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
    const filled = filledCount(lineup);
    const warnings = [];
    if (filled > 0 && lineup.captain === null) warnings.push('Διάλεξε captain από τη βασική πεντάδα (×2).');
    if (filled < 10) warnings.push(`Λείπουν ${10 - filled} παίκτες.`);
    for (const id of lineup.slots) {
      const player = id === null ? null : store.byId.get(id);
      if (player?.availability_status === 'out') {
        warnings.push(`${prettyName(player.name)} είναι εκτός (out): μετράει 0.`);
      } else if (player?.availability_status === 'doubtful') {
        warnings.push(`${prettyName(player.name)} είναι αμφίβολος.`);
      }
    }
    return h(
      'div',
      { class: 'card summary' },
      h(
        'div',
        { class: 'total' },
        h('span', { class: 'stat-label' }, 'Συνολική πρόβλεψη'),
        h('span', { class: 'big' }, fmt(lineupTotal(lineup, store.byId))),
        h('span', { class: 'stat-hint' }, `${filled}/10 παίκτες`),
      ),
      counts.length > 0
        ? h(
            'ul',
            { class: 'chips', 'aria-label': 'Παίκτες ανά ομάδα' },
            counts.map(([team, count]) =>
              h('li', { class: `chip${count >= MAX_PER_TEAM ? ' warn' : ''}` }, `${team} ${count}/${MAX_PER_TEAM}`),
            ),
          )
        : null,
      warnings.length > 0 ? h('ul', { class: 'notes' }, warnings.map((w) => h('li', {}, w))) : null,
      h(
        'div',
        { class: 'actions' },
        h('button', { type: 'button', class: 'btn primary', onclick: onAutoFill }, 'Αυτόματη συμπλήρωση'),
        h('button', { type: 'button', class: 'btn', onclick: onClear }, 'Καθαρισμός'),
      ),
    );
  }

  function onAutoFill() {
    if (filledCount(store.lineup) > 0 && !confirm('Η αυτόματη συμπλήρωση αντικαθιστά το τρέχον ρόστερ. Συνέχεια;')) {
      return;
    }
    setLineup(autoFill(store.items));
    toast('Το ρόστερ συμπληρώθηκε με τους υψηλότερους προβλεπόμενους παίκτες.', 'ok');
  }

  function onClear() {
    if (filledCount(store.lineup) === 0) return;
    if (confirm('Να αδειάσει το ρόστερ;')) setLineup(emptyLineup());
  }

  function render() {
    clear(body);
    body.append(
      summary(),
      ...SECTIONS.map((section) =>
        h(
          'section',
          { class: 'card' },
          h('h3', {}, section.title, h('span', { class: 'sub' }, section.note)),
          h(
            'ol',
            { class: 'slots', start: section.start + 1 },
            Array.from({ length: section.end - section.start }, (_unused, k) => slotRow(section.start + k)),
          ),
        ),
      ),
      h(
        'p',
        { class: 'muted small' },
        'Η πρόβλεψη εφαρμόζει captain ×2 και πάγκο ×0,5 πάνω στο προβλεπόμενο fantasy score. Ελέγχεται ' +
          'μόνο το όριο των 6 παικτών ανά ομάδα· οι τιμές (100 credits), οι θέσεις (2C/4F/4G) και ο ' +
          'προπονητής δεν υπάρχουν στο API και δεν υπολογίζονται. Το ρόστερ αποθηκεύεται στη συσκευή σου.',
      ),
    );
  }

  onLineupChange(() => {
    if (!el.hidden) render();
  });

  async function enter() {
    if (store.items.length === 0) {
      clear(body);
      body.append(h('p', { class: 'loading' }, 'Φόρτωση παικτών…'));
    }
    try {
      await loadRankings();
      render();
    } catch (error) {
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
  }

  return { el, enter };
}
