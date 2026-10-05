// Βοηθητικά: δημιουργία DOM μόνο με κόμβους κειμένου (ποτέ HTML strings), μορφοποίηση, ονόματα και
// μηνύματα στα ελληνικά.
// Κάθε κείμενο που προέρχεται από το API μπαίνει στο DOM ως κείμενο (ποτέ ως HTML).

const PROPERTY_ATTRIBUTES = new Set(['value', 'checked', 'selected']);

function appendChildren(el, children) {
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
}

export function h(tag, props = {}, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(props ?? {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === 'class') el.className = value;
    else if (key.startsWith('on') && typeof value === 'function') {
      el.addEventListener(key.slice(2).toLowerCase(), value);
    } else if (PROPERTY_ATTRIBUTES.has(key)) el[key] = value;
    else el.setAttribute(key, value === true ? '' : String(value));
  }
  appendChildren(el, children);
  return el;
}

export function clear(el) {
  el.replaceChildren();
}

export function fmt(value, digits = 1) {
  if (typeof value !== 'number' || Number.isNaN(value)) return '–';
  return value.toLocaleString('el-GR', { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

// 'ΕΠΩΝΥΜΟ, ΟΝΟΜΑ' -> 'Όνομα Επώνυμο' (λατινικά στο API: 'VEZENKOV, SASHA' -> 'Sasha Vezenkov').
export function prettyName(raw) {
  if (!raw) return '';
  const [last, ...rest] = String(raw).split(',');
  const first = rest.join(',').trim();
  const full = first ? `${first} ${last.trim()}` : last.trim();
  return full.toLowerCase().replace(/(^|[\s\-'’.(])(\p{L})/gu, (_m, sep, ch) => sep + ch.toUpperCase());
}

// Αναζήτηση όπως στο API: χωρίς πεζά/κεφαλαία, τόνους και σημεία στίξης, κάθε λέξη σε οποιαδήποτε σειρά.
export function norm(text) {
  return String(text ?? '')
    .normalize('NFD')
    .replace(/[̀-ͯ]/g, '')
    .replace(/[.,'’-]/g, '')
    .toLowerCase()
    .replace(/\s+/g, ' ')
    .trim();
}

export function matchesQuery(haystackNorm, query) {
  const words = norm(query).split(' ').filter(Boolean);
  return words.every((word) => haystackNorm.includes(word));
}

export function fmtDay(iso) {
  if (!iso) return '–';
  const [, month, day] = iso.split('-');
  const weekday = new Date(`${iso}T00:00:00Z`).toLocaleDateString('el-GR', {
    weekday: 'short',
    timeZone: 'UTC',
  });
  return `${weekday} ${day}/${month}`;
}

export function fmtTime(isoUtc) {
  if (!isoUtc) return null;
  const date = new Date(isoUtc);
  if (Number.isNaN(date.getTime())) return null;
  return date.toLocaleTimeString('el-GR', { hour: '2-digit', minute: '2-digit' });
}

export function fmtShortDate(iso) {
  if (!iso) return '–';
  const [, month, day] = iso.split('-');
  return `${day}/${month}`;
}

const NOTES = [
  [/^player is marked out/, 'Ο παίκτης είναι εκτός (out): η πρόβλεψη θεωρείται 0.'],
  [/^player is marked doubtful/, 'Ο παίκτης είναι αμφίβολος: η πρόβλεψη υποθέτει ότι θα παίξει.'],
  [/^expected return date has passed/, 'Η αναμενόμενη επιστροφή πέρασε: η εγγραφή διαθεσιμότητας μπορεί να είναι παλιά.'],
  [/^no scheduled game/, 'Δεν υπάρχει προγραμματισμένος αγώνας: η πρόβλεψη βασίζεται μόνο στη φόρμα του παίκτη.'],
  [/^player is not active/, 'Ο παίκτης δεν είναι ενεργός (κανένας αγώνας στη νέα σεζόν ή τις τελευταίες 45 ημέρες): η ομάδα και ο αγώνας μπορεί να είναι παλιά.'],
  [/^player has no previous appearances/, 'Δεν υπάρχουν προηγούμενες συμμετοχές: η πρόβλεψη βασίζεται μόνο στο πλαίσιο του αγώνα.'],
];

export function translateNote(note) {
  for (const [pattern, text] of NOTES) if (pattern.test(note)) return text;
  const few = /^player has only (\d+) previous appearance/.exec(note);
  if (few) return `Μόνο ${few[1]} προηγούμενες συμμετοχές: η πρόβλεψη είναι λιγότερο αξιόπιστη.`;
  return note;
}

export const STATUS_LABELS = { out: 'Εκτός', doubtful: 'Αμφίβολος', available: 'Διαθέσιμος' };

export function statusBadge(status) {
  if (!status || status === 'available') return null;
  return h('span', { class: `badge ${status === 'out' ? 'bad' : 'warn'}` }, STATUS_LABELS[status]);
}

export function opponentLabel(code, home) {
  if (!code) return '–';
  return `${home ? 'vs' : '@'} ${code}`;
}
