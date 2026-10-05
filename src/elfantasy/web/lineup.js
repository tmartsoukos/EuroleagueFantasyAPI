// Καθαρή λογική του ρόστερ (χωρίς DOM), ώστε να δοκιμάζεται με `node --test`.
//
// Ρόστερ: 10 θέσεις. Δείκτες 0-4 = βασική πεντάδα, 5 = έκτος παίκτης, 6-9 = πάγκος.
// Βάρη: πεντάδα και έκτος 100%, πάγκος 50%, captain (μόνο από την πεντάδα) ×2.
// Περιορισμός: έως 6 παίκτες από την ίδια ομάδα (FANTASY_RULES.md, ενότητα 5).
// Το API δεν δίνει τιμές (credits) ούτε θέσεις (G/F/C), άρα αυτά δεν ελέγχονται.

export const SLOT_COUNT = 10;
export const STARTER_COUNT = 5;
export const SIXTH_INDEX = 5;
export const MAX_PER_TEAM = 6;
export const CAPTAIN_MULTIPLIER = 2;
export const BENCH_MULTIPLIER = 0.5;

export function emptyLineup() {
  return { slots: Array(SLOT_COUNT).fill(null), captain: null };
}

export function slotRole(index) {
  if (index < STARTER_COUNT) return 'starter';
  return index === SIXTH_INDEX ? 'sixth' : 'bench';
}

export function slotWeight(index, isCaptain) {
  if (slotRole(index) === 'bench') return BENCH_MULTIPLIER;
  return isCaptain && index < STARTER_COUNT ? CAPTAIN_MULTIPLIER : 1;
}

function copy(lineup) {
  return { slots: [...lineup.slots], captain: lineup.captain };
}

export function filledCount(lineup) {
  return lineup.slots.filter((id) => id !== null).length;
}

export function firstEmptySlot(lineup) {
  const index = lineup.slots.indexOf(null);
  return index === -1 ? null : index;
}

export function teamCounts(lineup, byId) {
  const counts = new Map();
  for (const id of lineup.slots) {
    const player = id === null ? null : byId.get(id);
    if (player) counts.set(player.team_code, (counts.get(player.team_code) ?? 0) + 1);
  }
  return counts;
}

export function lineupTotal(lineup, byId) {
  let total = 0;
  lineup.slots.forEach((id, index) => {
    const player = id === null ? null : byId.get(id);
    if (player) total += slotWeight(index, id === lineup.captain) * player.predicted_fantasy;
  });
  return total;
}

// Επιστρέφει null αν ο παίκτης μπορεί να μπει, αλλιώς κωδικό λόγου.
export function addBlocker(lineup, player, byId, slot = null) {
  if (lineup.slots.includes(player.player_id)) return 'duplicate';
  if (slot !== null && lineup.slots[slot] !== null) return 'occupied';
  if (slot === null && firstEmptySlot(lineup) === null) return 'full';
  if ((teamCounts(lineup, byId).get(player.team_code) ?? 0) >= MAX_PER_TEAM) return 'team-limit';
  return null;
}

// Επιστρέφει { lineup, error }. Ο πρώτος παίκτης της πεντάδας γίνεται captain αν δεν υπάρχει.
export function addPlayer(lineup, player, byId, slot = null) {
  const error = addBlocker(lineup, player, byId, slot);
  if (error) return { lineup, error };
  const next = copy(lineup);
  const index = slot ?? firstEmptySlot(lineup);
  next.slots[index] = player.player_id;
  if (next.captain === null && index < STARTER_COUNT) next.captain = player.player_id;
  return { lineup: next, error: null };
}

export function removeSlot(lineup, index) {
  const next = copy(lineup);
  if (next.slots[index] !== null && next.slots[index] === next.captain) next.captain = null;
  next.slots[index] = null;
  return next;
}

export function setCaptain(lineup, playerId) {
  const index = lineup.slots.indexOf(playerId);
  if (index === -1 || index >= STARTER_COUNT) return lineup;
  return { slots: [...lineup.slots], captain: playerId };
}

// Άπληστη επιλογή κατά φθίνον predicted_fantasy με όριο ανά ομάδα. Επειδή ο περιορισμός είναι
// matroid και τα βάρη ταξινομούνται (captain ×2, πεντάδα/έκτος ×1, πάγκος ×0,5), η άπληστη
// επιλογή με ταξινομημένη ανάθεση δίνει το βέλτιστο ρόστερ. Οι παίκτες `out` εξαιρούνται.
export function autoFill(players, maxPerTeam = MAX_PER_TEAM) {
  const sorted = players
    .filter((p) => p.availability_status !== 'out')
    .sort((a, b) => b.predicted_fantasy - a.predicted_fantasy || a.player_id.localeCompare(b.player_id));
  const counts = new Map();
  const slots = [];
  for (const player of sorted) {
    if (slots.length === SLOT_COUNT) break;
    const used = counts.get(player.team_code) ?? 0;
    if (used >= maxPerTeam) continue;
    counts.set(player.team_code, used + 1);
    slots.push(player.player_id);
  }
  while (slots.length < SLOT_COUNT) slots.push(null);
  return { slots, captain: slots[0] };
}

// Καθαρίζει ό,τι διαβάστηκε από το localStorage: σχήμα, άγνωστοι παίκτες, διπλότυπα, captain.
export function sanitize(raw, byId) {
  const lineup = emptyLineup();
  if (!raw || !Array.isArray(raw.slots)) return lineup;
  const seen = new Set();
  for (let i = 0; i < SLOT_COUNT; i += 1) {
    const id = raw.slots[i];
    if (typeof id === 'string' && byId.has(id) && !seen.has(id)) {
      seen.add(id);
      lineup.slots[i] = id;
    }
  }
  const captainIndex = lineup.slots.indexOf(raw.captain);
  lineup.captain = captainIndex !== -1 && captainIndex < STARTER_COUNT ? raw.captain : null;
  return lineup;
}
