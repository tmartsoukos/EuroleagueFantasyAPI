// Οθόνη «Διαχείριση»: διαθεσιμότητα παικτών (τραυματισμοί) και ανανέωση δεδομένων.
// Το κλειδί (ADMIN_API_KEY) μένει μόνο στο sessionStorage αυτής της καρτέλας: ποτέ σε URL,
// σε μόνιμη αποθήκευση ή στο DOM μετά τη σύνδεση.

import { ApiError, describeError, request } from './api.js';
import { invalidateRankings, toast } from './store.js';
import { STATUS_LABELS, clear, fmt, fmtShortDate, h, prettyName, statusBadge } from './util.js';

const KEY_STORAGE = 'elfantasy.adminKey';
// Αίτημα που ΔΕΝ αλλάζει τίποτα: ο έλεγχος κλειδιού γίνεται πρώτος, άρα με σωστό κλειδί ο άγνωστος
// παίκτης δίνει 404, με λάθος 401 και χωρίς ρυθμισμένο κλειδί στον server 503.
const PROBE_PATH = '/availability/PZZZZZZ';

function readKey() {
  try {
    return sessionStorage.getItem(KEY_STORAGE);
  } catch {
    return null;
  }
}

function writeKey(key) {
  try {
    if (key) sessionStorage.setItem(KEY_STORAGE, key);
    else sessionStorage.removeItem(KEY_STORAGE);
  } catch {
    // Χωρίς sessionStorage το κλειδί ισχύει μέχρι να κλείσει η σελίδα.
  }
}

export function createAdminView({ onDataChanged }) {
  const el = h('section', { class: 'view', 'aria-labelledby': 'admin-title' });
  const body = h('div', { class: 'view-body' });
  el.append(h('h2', { id: 'admin-title' }, 'Διαχείριση'), body);
  let adminKey = readKey();

  function logout(message) {
    adminKey = null;
    writeKey(null);
    if (message) toast(message, 'error');
    renderLogin();
  }

  // Κοινή διαχείριση σφαλμάτων των προστατευμένων κλήσεων: το 401 αποσυνδέει.
  async function adminRequest(path, options) {
    try {
      return await request(path, { ...options, adminKey });
    } catch (error) {
      if (error instanceof ApiError && error.status === 401) {
        logout('Το κλειδί δεν έγινε δεκτό. Συνδέσου ξανά.');
      }
      throw error;
    }
  }

  function renderLogin(message = '') {
    const input = h('input', {
      type: 'password',
      autocomplete: 'off',
      'aria-label': 'Κλειδί διαχειριστή',
      placeholder: 'ADMIN_API_KEY',
      required: true,
    });
    const status = h('p', { class: 'error-text', role: 'alert' }, message);
    const submit = h('button', { type: 'submit', class: 'btn primary' }, 'Σύνδεση');
    const form = h(
      'form',
      {
        class: 'card form',
        onsubmit: async (event) => {
          event.preventDefault();
          const candidate = input.value.trim();
          submit.disabled = true;
          status.textContent = '';
          try {
            await request(PROBE_PATH, { method: 'DELETE', adminKey: candidate });
          } catch (error) {
            // 404 σημαίνει «το κλειδί έγινε δεκτό, ο παίκτης PZZZZZZ δεν υπάρχει».
            if (error instanceof ApiError && error.status === 404) {
              adminKey = candidate;
              writeKey(candidate);
              input.value = '';
              await renderPanel();
              return;
            }
            status.textContent = describeError(error);
          }
          submit.disabled = false;
        },
      },
      h('p', {}, 'Για να αλλάξεις τη διαθεσιμότητα παικτών χρειάζεται το κλειδί διαχειριστή (ADMIN_API_KEY).'),
      h('label', {}, 'Κλειδί', input),
      status,
      submit,
      h('p', { class: 'muted small' }, 'Το κλειδί κρατιέται μόνο σε αυτή την καρτέλα μέχρι να την κλείσεις.'),
    );
    clear(body);
    body.append(form);
  }

  async function renderPanel() {
    clear(body);
    const listBox = h('div', {});
    const refreshBox = h('div', {});
    body.append(
      h(
        'div',
        { class: 'card-head' },
        h('p', { class: 'muted' }, 'Συνδεδεμένος διαχειριστής.'),
        h('button', { type: 'button', class: 'btn', onclick: () => logout() }, 'Αποσύνδεση'),
      ),
      buildForm(() => loadList(listBox)),
      listBox,
      refreshBox,
    );
    buildRefresh(refreshBox);
    await loadList(listBox);
  }

  function buildForm(onSaved) {
    let selected = null;
    const query = h('input', {
      type: 'search',
      placeholder: 'Αναζήτηση παίκτη…',
      'aria-label': 'Αναζήτηση παίκτη',
      autocomplete: 'off',
      maxlength: 100,
    });
    const found = h('ul', { class: 'result-list' });
    const chosen = h('p', { class: 'muted' }, 'Δεν έχει επιλεγεί παίκτης.');
    const statusSelect = h(
      'select',
      { 'aria-label': 'Κατάσταση' },
      Object.entries(STATUS_LABELS).map(([value, label]) => h('option', { value }, label)),
    );
    statusSelect.value = 'out';
    const source = h('input', { type: 'text', maxlength: 100, 'aria-label': 'Πηγή' });
    const note = h('textarea', { rows: 2, maxlength: 500, 'aria-label': 'Σημείωση' });
    const returnDate = h('input', { type: 'date', 'aria-label': 'Αναμενόμενη επιστροφή' });
    const message = h('p', { class: 'error-text', role: 'alert' });
    let timer = null;

    function select(player) {
      selected = player;
      chosen.textContent = `Επιλεγμένος: ${prettyName(player.name)} (${player.team_code})`;
      clear(found);
    }

    query.addEventListener('input', () => {
      clearTimeout(timer);
      timer = setTimeout(async () => {
        const text = query.value.trim();
        if (!text) return clear(found);
        try {
          const data = await request(`/players?search=${encodeURIComponent(text)}&limit=8`);
          found.replaceChildren(
            ...data.items.map((player) =>
              h(
                'li',
                {},
                h(
                  'button',
                  { type: 'button', onclick: () => select(player) },
                  h('span', { class: 'strong' }, prettyName(player.name)),
                  h('span', { class: 'sub' }, player.team_code),
                ),
              ),
            ),
          );
        } catch (error) {
          message.textContent = describeError(error);
        }
        return undefined;
      }, 250);
    });

    return h(
      'form',
      {
        class: 'card form',
        onsubmit: async (event) => {
          event.preventDefault();
          message.textContent = '';
          if (!selected) {
            message.textContent = 'Διάλεξε πρώτα παίκτη.';
            return;
          }
          try {
            await adminRequest('/availability', {
              method: 'POST',
              body: {
                player_id: selected.player_id,
                status: statusSelect.value,
                source: source.value.trim() || null,
                note: note.value.trim() || null,
                expected_return: returnDate.value || null,
              },
            });
            invalidateRankings();
            toast(`${prettyName(selected.name)}: αποθηκεύτηκε.`, 'ok');
            note.value = '';
            source.value = '';
            returnDate.value = '';
            await onSaved();
          } catch (error) {
            if (adminKey) message.textContent = describeError(error);
          }
        },
      },
      h('h3', {}, 'Καταχώρηση διαθεσιμότητας'),
      h('label', {}, 'Παίκτης', query),
      found,
      chosen,
      h('label', {}, 'Κατάσταση', statusSelect),
      h('label', {}, 'Πηγή (προαιρετικό)', source),
      h('label', {}, 'Σημείωση (προαιρετικό, έως 500 χαρακτήρες)', note),
      h('label', {}, 'Αναμενόμενη επιστροφή (προαιρετικό)', returnDate),
      message,
      h('button', { type: 'submit', class: 'btn primary' }, 'Αποθήκευση'),
      h(
        'p',
        { class: 'muted small' },
        'Η νέα εγγραφή αντικαθιστά ολόκληρη την προηγούμενη του παίκτη και ισχύει αμέσως στις προβλέψεις.',
      ),
    );
  }

  async function loadList(box) {
    box.replaceChildren(h('p', { class: 'loading' }, 'Φόρτωση διαθεσιμότητας…'));
    try {
      const data = await request('/availability');
      box.replaceChildren(
        h(
          'section',
          { class: 'card' },
          h('h3', {}, `Τρέχουσες εγγραφές (${data.total})`),
          data.items.length === 0
            ? h('p', { class: 'empty' }, 'Δεν υπάρχουν εγγραφές: όλοι θεωρούνται διαθέσιμοι.')
            : h('ul', { class: 'avail-list' }, data.items.map((item) => availabilityRow(item, box))),
        ),
      );
    } catch (error) {
      box.replaceChildren(h('div', { class: 'error-box', role: 'alert' }, describeError(error)));
    }
  }

  function availabilityRow(item, box) {
    return h(
      'li',
      {},
      h(
        'span',
        { class: 'avail-main' },
        h('a', { class: 'player-link', href: `#/player/${encodeURIComponent(item.player_id)}` }, prettyName(item.name)),
        ' ',
        statusBadge(item.status) ?? h('span', { class: 'badge' }, STATUS_LABELS[item.status]),
        h(
          'span',
          { class: 'sub' },
          [item.note, item.expected_return ? `επιστροφή ${fmtShortDate(item.expected_return)}` : null, item.source]
            .filter(Boolean)
            .join(' · '),
        ),
      ),
      h(
        'button',
        {
          type: 'button',
          class: 'btn',
          onclick: async () => {
            if (!confirm(`Να αφαιρεθεί η εγγραφή του ${prettyName(item.name)};`)) return;
            try {
              await adminRequest(`/availability/${encodeURIComponent(item.player_id)}`, { method: 'DELETE' });
              invalidateRankings();
              toast('Η εγγραφή αφαιρέθηκε.', 'ok');
              await loadList(box);
            } catch (error) {
              if (adminKey) toast(describeError(error), 'error');
            }
          },
        },
        'Αφαίρεση',
      ),
    );
  }

  function buildRefresh(box) {
    const result = h('p', { class: 'muted', role: 'status' });
    const button = h(
      'button',
      {
        type: 'button',
        class: 'btn',
        onclick: async () => {
          button.disabled = true;
          result.textContent = 'Ανανέωση…';
          try {
            const data = await adminRequest('/admin/refresh', { method: 'POST' });
            invalidateRankings();
            result.textContent =
              `Έγινε: δεδομένα έως ${fmtShortDate(data.latest_played_game_date)} ` +
              `(πριν ${fmtShortDate(data.previous_latest_played_game_date)}), ` +
              `${data.players} παίκτες, ${fmt(data.duration_seconds)} s.`;
            onDataChanged();
          } catch (error) {
            result.textContent = adminKey ? describeError(error) : '';
          }
          button.disabled = false;
        },
      },
      'Ανανέωση δεδομένων',
    );
    box.replaceChildren(
      h(
        'section',
        { class: 'card' },
        h('h3', {}, 'Ανανέωση δεδομένων'),
        h(
          'p',
          { class: 'muted' },
          'Μετά από νέο ingestion στη βάση, η υπηρεσία δεν βλέπει τους νέους αγώνες μέχρι να πατηθεί αυτό το κουμπί.',
        ),
        button,
        result,
      ),
    );
  }

  async function enter() {
    if (adminKey) await renderPanel();
    else renderLogin();
  }

  return { el, enter };
}
