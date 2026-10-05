// Tests της καθαρής λογικής του ρόστερ (src/elfantasy/web/lineup.js). Τρέχουν με `node --test`.
// Το module φορτώνεται από data URL, ώστε να μη χρειάζεται package.json με "type": "module".

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const source = readFileSync(new URL('../../src/elfantasy/web/lineup.js', import.meta.url), 'utf8');
const lineup = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);

function player(id, team, score, status = 'available') {
  return { player_id: id, team_code: team, predicted_fantasy: score, availability_status: status };
}

function indexOf(players) {
  return new Map(players.map((p) => [p.player_id, p]));
}

test('τα βάρη: πεντάδα και έκτος ×1, πάγκος ×0,5, captain ×2 μόνο στην πεντάδα', () => {
  assert.equal(lineup.slotWeight(0, false), 1);
  assert.equal(lineup.slotWeight(0, true), 2);
  assert.equal(lineup.slotWeight(5, false), 1);
  assert.equal(lineup.slotWeight(5, true), 1);
  assert.equal(lineup.slotWeight(6, false), 0.5);
  assert.equal(lineup.slotWeight(9, true), 0.5);
});

test('ο πρώτος παίκτης της πεντάδας γίνεται captain και το σύνολο εφαρμόζει τα βάρη', () => {
  const players = [player('P1', 'AAA', 20), player('P2', 'BBB', 10), player('P3', 'CCC', 8)];
  const byId = indexOf(players);
  let state = lineup.emptyLineup();
  state = lineup.addPlayer(state, players[0], byId).lineup;
  assert.equal(state.captain, 'P1');
  state = lineup.addPlayer(state, players[1], byId, 5).lineup; // έκτος
  state = lineup.addPlayer(state, players[2], byId, 9).lineup; // πάγκος
  assert.equal(lineup.lineupTotal(state, byId), 20 * 2 + 10 + 8 * 0.5);
  assert.equal(lineup.filledCount(state), 3);
});

test('απορρίπτει διπλότυπο, κατειλημμένη θέση, γεμάτο ρόστερ και όριο ομάδας', () => {
  const same = Array.from({ length: 7 }, (_u, i) => player(`T${i}`, 'AAA', 10 - i));
  const byId = indexOf(same);
  let state = lineup.emptyLineup();
  for (let i = 0; i < 6; i += 1) {
    const result = lineup.addPlayer(state, same[i], byId);
    assert.equal(result.error, null);
    state = result.lineup;
  }
  assert.equal(lineup.addPlayer(state, same[6], byId).error, 'team-limit');
  assert.equal(lineup.addPlayer(state, same[0], byId).error, 'duplicate');
  assert.equal(lineup.addPlayer(state, player('X', 'BBB', 1), byId, 0).error, 'occupied');

  const full = lineup.autoFill(Array.from({ length: 12 }, (_u, i) => player(`F${i}`, `T${i}`, 20 - i)));
  const fullById = indexOf(Array.from({ length: 12 }, (_u, i) => player(`F${i}`, `T${i}`, 20 - i)));
  assert.equal(lineup.addPlayer(full, player('F11', 'T11', 1), fullById).error, 'full');
});

test('η αφαίρεση του captain τον μηδενίζει, και ο captain επιτρέπεται μόνο από την πεντάδα', () => {
  const players = [player('P1', 'AAA', 20), player('P2', 'BBB', 10)];
  const byId = indexOf(players);
  let state = lineup.addPlayer(lineup.emptyLineup(), players[0], byId).lineup;
  state = lineup.addPlayer(state, players[1], byId, 7).lineup;
  assert.equal(lineup.setCaptain(state, 'P2').captain, 'P1');
  state = lineup.removeSlot(state, 0);
  assert.equal(state.captain, null);
  assert.equal(state.slots[0], null);
});

test('η αυτόματη συμπλήρωση σέβεται το όριο ομάδας, εξαιρεί τους out και βάζει τον καλύτερο captain', () => {
  const players = [
    ...Array.from({ length: 8 }, (_u, i) => player(`A${i}`, 'AAA', 30 - i)),
    player('B0', 'BBB', 22),
    player('B1', 'BBB', 21),
    player('B2', 'BBB', 20),
    player('C0', 'CCC', 40, 'out'),
    player('C1', 'CCC', 5),
  ];
  const result = lineup.autoFill(players);
  const byId = indexOf(players);
  assert.equal(result.captain, 'A0');
  assert.equal(result.slots[0], 'A0');
  assert.equal(result.slots.filter((id) => id !== null).length, 10);
  assert.ok(!result.slots.includes('C0'));
  assert.equal(lineup.teamCounts(result, byId).get('AAA'), 6);
  // Φθίνουσα σειρά τιμών στα slots: το βέλτιστο για ταξινομημένα βάρη.
  const values = result.slots.map((id) => byId.get(id).predicted_fantasy);
  assert.deepEqual(values, [...values].sort((a, b) => b - a));
});

test('η αυτόματη συμπλήρωση με λιγότερους από 10 παίκτες αφήνει κενές θέσεις', () => {
  const result = lineup.autoFill([player('P1', 'AAA', 10), player('P2', 'BBB', 5)]);
  assert.equal(result.slots.filter((id) => id !== null).length, 2);
  assert.equal(result.slots.length, 10);
});

test('το sanitize πετά άγνωστους, διπλούς και λάθος captain', () => {
  const byId = indexOf([player('P1', 'AAA', 10), player('P2', 'BBB', 5)]);
  const raw = { slots: ['P1', 'P1', 'GHOST', null, null, 'P2', 7, null, null, null], captain: 'P2' };
  const clean = lineup.sanitize(raw, byId);
  assert.deepEqual(clean.slots, ['P1', null, null, null, null, 'P2', null, null, null, null]);
  assert.equal(clean.captain, null); // ο P2 είναι έκτος, δεν μπορεί να είναι captain
  assert.deepEqual(lineup.sanitize(null, byId), lineup.emptyLineup());
  assert.deepEqual(lineup.sanitize({ slots: 'x' }, byId), lineup.emptyLineup());
});
