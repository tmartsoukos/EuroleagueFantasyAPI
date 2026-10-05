// Οθόνη «Παίκτης»: αναζήτηση και πρόβλεψη για τον επόμενο αγώνα.

import { describeError, request } from './api.js';
import { loadRankings, store } from './store.js';
import {
  STATUS_LABELS,
  clear,
  fmt,
  fmtDay,
  fmtShortDate,
  fmtTime,
  h,
  opponentLabel,
  prettyName,
  statusBadge,
  translateNote,
} from './util.js';
import { addToLineup } from './view-rankings.js';

const SEARCH_DELAY_MS = 250;
const MAE = 5.89;

export function createPlayerView() {
  const el = h('section', { class: 'view', 'aria-labelledby': 'player-title' });
  const results = h('ul', { class: 'result-list', 'aria-label': 'Αποτελέσματα αναζήτησης' });
  const resultsInfo = h('p', { class: 'muted small', role: 'status' });
  const detail = h('div', { class: 'detail' });
  let searchTimer = null;
  let searchToken = 0;
  let detailToken = 0;

  const input = h('input', {
    type: 'search',
    placeholder: 'Όνομα παίκτη, π.χ. vezenkov',
    'aria-label': 'Αναζήτηση παίκτη',
    autocomplete: 'off',
    maxlength: 100,
    oninput: (event) => {
      clearTimeout(searchTimer);
      searchTimer = setTimeout(() => search(event.target.value.trim()), SEARCH_DELAY_MS);
    },
  });

  el.append(
    h('h2', { id: 'player-title' }, 'Παίκτης'),
    h('div', { class: 'controls' }, input),
    resultsInfo,
    results,
    detail,
  );

  async function search(query) {
    const token = (searchToken += 1);
    if (!query) {
      clear(results);
      resultsInfo.textContent = '';
      return;
    }
    try {
      const data = await request(`/players?search=${encodeURIComponent(query)}&limit=20`);
      if (token !== searchToken) return;
      resultsInfo.textContent =
        data.total === 0 ? 'Κανένας παίκτης δεν βρέθηκε.' : `${data.total} αποτελέσματα.`;
      results.replaceChildren(
        ...data.items.map((player) =>
          h(
            'li',
            {},
            h(
              'a',
              { href: `#/player/${encodeURIComponent(player.player_id)}` },
              h('span', { class: 'strong' }, prettyName(player.name)),
              h('span', { class: 'sub' }, player.team_code, player.is_active ? '' : ' · ανενεργός'),
            ),
          ),
        ),
      );
    } catch (error) {
      if (token === searchToken) resultsInfo.textContent = describeError(error);
    }
  }

  function stat(label, value, hint) {
    return h(
      'div',
      { class: 'stat' },
      h('span', { class: 'stat-label' }, label),
      h('span', { class: 'stat-value' }, value),
      hint ? h('span', { class: 'stat-hint' }, hint) : null,
    );
  }

  function renderPlayer(p) {
    const game = p.next_game;
    const out = p.availability.status === 'out';
    const tipoff = game ? fmtTime(game.tipoff_utc) : null;
    const inRankings = store.byId.get(p.player_id);

    const gameCard = game
      ? h(
          'p',
          { class: 'game' },
          h('strong', {}, opponentLabel(game.opponent_code, game.home)),
          ' ',
          game.opponent_name ?? '',
          h(
            'span',
            { class: 'sub' },
            `${fmtDay(game.game_date)}${tipoff ? ` · ${tipoff} (ώρα συσκευής)` : ''}`,
          ),
        )
      : h('p', { class: 'muted' }, 'Δεν υπάρχει προγραμματισμένος αγώνας.');

    const availability = p.availability;
    const availabilityBlock =
      availability.status === 'available'
        ? null
        : h(
            'div',
            { class: `callout ${availability.status === 'out' ? 'bad' : 'warn'}` },
            h('strong', {}, STATUS_LABELS[availability.status]),
            availability.note ? ` · ${availability.note}` : '',
            availability.expected_return
              ? ` · αναμενόμενη επιστροφή ${fmtShortDate(availability.expected_return)}`
              : '',
            availability.source ? ` (πηγή: ${availability.source})` : '',
          );

    const features = p.features
      ? h(
          'details',
          { class: 'features' },
          h('summary', {}, 'Τεχνικά στοιχεία του μοντέλου (features)'),
          h(
            'dl',
            {},
            Object.entries(p.features).flatMap(([name, value]) => [
              h('dt', {}, name),
              h('dd', {}, typeof value === 'number' ? fmt(value, 2) : '–'),
            ]),
          ),
        )
      : null;

    detail.replaceChildren(
      h(
        'article',
        { class: 'card' },
        h(
          'header',
          { class: 'card-head' },
          h('h3', {}, prettyName(p.name)),
          h('span', { class: 'sub' }, `${p.team_name ?? p.team_code} `, statusBadge(availability.status)),
        ),
        gameCard,
        availabilityBlock,
        h(
          'div',
          { class: 'stats' },
          stat(
            'Προβλεπόμενο fantasy',
            fmt(p.predicted_fantasy),
            out ? `Μοντέλο: ${fmt(p.model_predicted_fantasy)} (εκτός → 0)` : `τυπικό σφάλμα ±${fmt(MAE, 1)}`,
          ),
          stat('Προβλεπόμενο PIR', fmt(p.predicted_pir), 'ανεξάρτητο μοντέλο'),
          stat('Προηγούμενοι αγώνες', String(p.n_prior_appearances), `τελευταίος: ${fmtShortDate(p.last_appearance_date)}`),
        ),
        p.notes.length > 0
          ? h('ul', { class: 'notes' }, p.notes.map((note) => h('li', {}, translateNote(note))))
          : null,
        h(
          'p',
          { class: 'muted small' },
          'Το fantasy είναι «τυπική» τιμή (κοντά στη διάμεσο) με την υπόθεση ότι ο παίκτης αγωνίζεται, ' +
            'χωρίς captain ×2 ή πάγκο ×0,5. Το σφάλμα ενός αγώνα είναι μεγάλο· η αξία της πρόβλεψης ' +
            'είναι στη σύγκριση παικτών.',
        ),
        h(
          'div',
          { class: 'actions' },
          inRankings
            ? h(
                'button',
                { type: 'button', class: 'btn primary', onclick: () => addToLineup(inRankings) },
                'Προσθήκη στο ρόστερ',
              )
            : h('span', { class: 'muted small' }, 'Ο παίκτης δεν είναι ενεργός: δεν μπαίνει στο ρόστερ.'),
          h('a', { class: 'btn', href: '#/rankings' }, 'Κατάταξη'),
        ),
        features,
        h('p', { class: 'muted small' }, `Έκδοση μοντέλου: ${p.model_version}`),
      ),
    );
  }

  async function loadDetail(playerId) {
    const token = (detailToken += 1);
    detail.replaceChildren(h('p', { class: 'loading' }, 'Φόρτωση πρόβλεψης…'));
    try {
      // Η κατάταξη χρειάζεται μόνο για το κουμπί του ρόστερ· αποτυχία της δεν μπλοκάρει την πρόβλεψη.
      const [data] = await Promise.all([
        request(`/predict/${encodeURIComponent(playerId)}?include_features=true`),
        loadRankings().catch(() => null),
      ]);
      if (token === detailToken) renderPlayer(data);
    } catch (error) {
      if (token !== detailToken) return;
      detail.replaceChildren(h('div', { class: 'error-box', role: 'alert' }, describeError(error)));
    }
  }

  async function enter(playerId) {
    if (playerId) {
      await loadDetail(playerId);
    } else {
      detailToken += 1;
      detail.replaceChildren(
        h('p', { class: 'empty' }, 'Γράψε το όνομα ενός παίκτη ή διάλεξε κάποιον από την κατάταξη.'),
      );
    }
  }

  return { el, enter };
}
