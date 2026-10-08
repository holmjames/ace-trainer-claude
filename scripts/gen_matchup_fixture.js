#!/usr/bin/env node
// Ground truth for the matchup model's damage layer (agent/pokemon/matchup.py: calc_hit): live-catalog cards
// (data/cards.json) attacking each other at full HP in a neutral doubles field, computed by Smogon's official
// calculator (@smogon/calc, a dev dependency of sim/). Writes tests/fixtures/smogon_calc_matchup.json;
// tests/test_matchup.py compares calc_hit with it.
//
//   node scripts/gen_matchup_fixture.js        # deterministic sample + every case of the mechanics below
//
// Always included (these are what calc_hit adds on top of battle.damage_percent): Pixilate-style type changes,
// Adaptability, Levitate and Air Balloon against Ground moves, and Low Kick's weight-based power. Skipped: moves
// the calculator cannot be told about here (Ruination, Beat Up, variable-hit moves are modelled on the item).
const fs = require('fs');
const path = require('path');
const root = path.resolve(__dirname, '..');
const calc = require(path.join(root, 'sim', 'node_modules', '@smogon/calc'));
const { calculate, Pokemon, Move, Field, Generations } = calc;
const gen = Generations.get(9);

let seed = 20261008;
function rand() { seed = (seed + 0x6D2B79F5) | 0; let t = seed; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }

const cards = JSON.parse(fs.readFileSync(path.join(root, 'data', 'cards.json'), 'utf8'));
const SKIP = new Set(['Ruination', 'Beat Up', 'Scale Shot', 'Icicle Spear', 'Population Bomb']);
const ATE = new Set(['Pixilate', 'Aerilate', 'Refrigerate', 'Galvanize']);
const GROUND_PROOF = (card) => card.ability === 'Levitate' || card.item === 'Air Balloon';

function mon(card) {
  // boostedStat 'auto' lets Booster Energy raise the right stat, as battle.build_mon does.
  return new Pokemon(gen, card.species, { level: 50, nature: card.nature, evs: card.evs, item: card.item, ability: card.ability, boostedStat: 'auto' });
}

const cases = [];
for (const atk of cards) {
  for (const def of cards) {
    if (atk === def) continue;
    for (const moveName of atk.moves) {
      if (SKIP.has(moveName)) continue;
      const probe = new Move(gen, moveName);
      if (probe.category === 'Status') continue;
      const forced = ATE.has(atk.ability) || atk.ability === 'Adaptability' || moveName === 'Low Kick'
        || (probe.type === 'Ground' && GROUND_PROOF(def));
      if (!forced && rand() > 0.2) continue; // sample the rest
      const attacker = mon(atk);
      const defender = mon(def);
      const field = new Field({ gameType: 'Doubles' });
      let result;
      try { result = calculate(gen, attacker, defender, new Move(gen, moveName), field); } catch (e) { continue; }
      const range = result.range();
      cases.push({
        a: atk.card_id, d: def.card_id, move: moveName,
        spread: ['allAdjacent', 'allAdjacentFoes'].includes(probe.target),
        max_hp: defender.maxHP(), damage: [range[0], range[1]], forced,
      });
    }
  }
}
const out = path.join(root, 'tests', 'fixtures', 'smogon_calc_matchup.json');
fs.writeFileSync(out, JSON.stringify(cases));
console.log(`wrote ${cases.length} cases (${cases.filter(c => c.forced).length} forced) to ${path.relative(root, out)} `
  + `(calc ${require(path.join(root, 'sim', 'node_modules', '@smogon/calc', 'package.json')).version})`);
