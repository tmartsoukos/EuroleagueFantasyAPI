// Εφαρμόζει το αποθηκευμένο θέμα πριν από την πρώτη εμφάνιση (χωρίς αναλαμπή). Χωρίς αποθηκευμένη
// επιλογή ισχύει το θέμα της συσκευής (prefers-color-scheme) από το CSS.
try {
  const saved = localStorage.getItem('elfantasy.theme');
  if (saved === 'light' || saved === 'dark') document.documentElement.dataset.theme = saved;
} catch {
  // Αποκλεισμένο storage: μένει το θέμα της συσκευής.
}
