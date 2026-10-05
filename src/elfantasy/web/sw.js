// Service worker: κρατά μόνο το κέλυφος της εφαρμογής (/app/*) για γρήγορη φόρτωση και λειτουργία
// χωρίς σύνδεση. Οι απαντήσεις του API (προβλέψεις, διαθεσιμότητα) ΔΕΝ αποθηκεύονται ποτέ:
// αλλάζουν συνεχώς και το κλειδί διαχειριστή περνά από αυτά τα αιτήματα.

const CACHE = 'elfantasy-shell-v1';
const SHELL = [
  './',
  'app.css',
  'app.js',
  'api.js',
  'store.js',
  'util.js',
  'lineup.js',
  'theme.js',
  'view-rankings.js',
  'view-player.js',
  'view-lineup.js',
  'view-admin.js',
  'manifest.webmanifest',
  'icon.svg',
];

self.addEventListener('install', (event) => {
  event.waitUntil(caches.open(CACHE).then((cache) => cache.addAll(SHELL)));
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) => Promise.all(keys.filter((key) => key !== CACHE).map((key) => caches.delete(key))))
      .then(() => self.clients.claim()),
  );
});

self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);
  if (request.method !== 'GET' || url.origin !== self.location.origin) return;
  if (!url.pathname.startsWith('/app/')) return;

  // Network-first: πάντα η τελευταία έκδοση όταν υπάρχει σύνδεση, αλλιώς το αντίγραφο.
  event.respondWith(
    fetch(request, { cache: 'no-cache' })
      .then((response) => {
        if (response.ok) {
          const copy = response.clone();
          caches.open(CACHE).then((cache) => cache.put(request, copy));
        }
        return response;
      })
      .catch(() => caches.match(request).then((cached) => cached ?? caches.match('./'))),
  );
});
