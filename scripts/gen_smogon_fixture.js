#!/usr/bin/env node
// Ground truth for the damage model: every pool card attacking every other with its real ability, item and
// moves, under random field conditions, computed by Smogon's official calculator (@smogon/calc, installed as
// a dev dependency of sim/). Writes tests/fixtures/smogon_calc_full.json; tests/test_damage_vs_smogon_full.py
// compares our estimates with it.
//
//   node scripts/gen_smogon_fixture.js                         # live catalog (sim/cards.json) -> smogon_calc_live.json
//   node scripts/gen_smogon_fixture.js old_pool.json full.json   # any card list -> any fixture name
//
// smogon_calc_full.json was generated (Oct 7) from the old guessed pool and is kept: it still exercises mechanics
// whose cards have left the live catalog (Technician, Hadron Engine, Surging Strikes...).
//
// Skipped on purpose: moves whose power depends on things the calculator cannot be told here (Beat Up, Rage
// Fist, Stomping Tantrum), variable-hit moves (the calculator assumes 3 hits; we model the item), and fixed
// damage (Ruination).
const fs = require('fs');
const path = require('path');
const root = path.resolve(__dirname, '..');
const calc = require(path.join(root, 'sim', 'node_modules', '@smogon/calc'));
const { calculate, Pokemon, Move, Field, Generations } = calc;
const gen = Generations.get(9);

let seed = 20261007;
function rand() { seed = (seed + 0x6D2B79F5) | 0; let t = seed; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }
function pick(list) { return list[Math.floor(rand() * list.length)]; }

const cardsPath = process.argv[2] ? path.resolve(process.argv[2]) : path.join(root, 'sim', 'cards.json');
const outName = process.argv[3] || 'smogon_calc_live.json';
const cards = JSON.parse(fs.readFileSync(cardsPath, 'utf8'));
// Variable-hit moves are skipped: the calculator fixes the hit count, we model the item (Loaded Dice: 4-5 hits).
const SKIP = new Set(['Beat Up', 'Rage Fist', 'Stomping Tantrum', 'Ruination', 'Scale Shot', 'Last Respects', 'Icicle Spear', 'Bullet Seed', 'Rock Blast']);
const RUIN_THIRD = [null, null, null, null, null, null, null, null, { ability: 'Sword of Ruin', species: 'Chien-Pao' }, { ability: 'Beads of Ruin', species: 'Chi-Yu' }];

function mon(card, opts) {
  return new Pokemon(gen, card.species, { level: 50, nature: card.nature, evs: card.evs, item: card.item, ability: card.ability, ...opts });
}

const cases = [];
for (const atk of cards) {
  for (const def of cards) {
    if (atk === def) continue;
    for (const moveName of atk.moves) {
      if (SKIP.has(moveName)) continue;
      const probe = new Move(gen, moveName);
      if (!probe.bp && !['Heavy Slam', 'Heat Crash', 'Eruption', 'Weather Ball', 'Facade', 'Last Respects'].includes(moveName)) continue;
      if (probe.category === 'Status') continue;
      if (rand() > 0.3) continue; // sample
      const hp = pick([1.0, 1.0, 0.6, 0.35]);
      const aBoost = pick([0, 0, 0, -1, 1]);
      const dBoost = pick([0, 0, 0, 1]);
      const weather = pick([undefined, undefined, undefined, 'Rain', 'Sun']);
      const terrain = pick([undefined, undefined, undefined, 'Electric', 'Psychic', 'Grassy']);
      const reflect = rand() < 0.12, lightScreen = rand() < 0.12;
      const third = pick(RUIN_THIRD);
      const status = rand() < 0.1 ? 'brn' : undefined;
      const offensive = probe.category === 'Physical' ? 'atk' : 'spa';
      const defensive = probe.category === 'Physical' ? 'def' : 'spd';
      const attacker = mon(atk, { boosts: { [offensive]: aBoost }, status, boostedStat: 'auto' });
      const defender = mon(def, { boosts: { [defensive]: dBoost }, boostedStat: 'auto' });
      defender.originalCurHP = Math.max(1, Math.floor(defender.maxHP() * hp));
      const field = new Field({
        gameType: 'Doubles', weather, terrain,
        isSwordOfRuin: !!third && third.ability === 'Sword of Ruin',
        isBeadsOfRuin: !!third && third.ability === 'Beads of Ruin',
        defenderSide: { isReflect: reflect, isLightScreen: lightScreen },
      });
      const move = new Move(gen, moveName);
      let result;
      try { result = calculate(gen, attacker, defender, move, field); } catch (e) { continue; }
      const range = result.range();
      cases.push({
        attacker: { species: atk.species, nature: atk.nature, evs: atk.evs, item: atk.item, ability: atk.ability, boosts: { [offensive]: aBoost }, status: status || null },
        defender: { species: def.species, nature: def.nature, evs: def.evs, item: def.item, ability: def.ability, boosts: { [defensive]: dBoost } },
        move: moveName, defender_hp_fraction: hp, defender_max_hp: defender.maxHP(), defender_cur_hp: defender.curHP(),
        field: { weather: weather || null, terrain: terrain || null, defender_reflect: reflect, defender_light_screen: lightScreen, third_ruin: third },
        spread_in_calc: field.gameType !== 'Singles' && ['allAdjacent', 'allAdjacentFoes'].includes(move.target),
        damage: [range[0], range[1]],
      });
    }
  }
}
// Always-present cases for the mechanics the test names, so sampling can never leave one out.
const by = Object.fromEntries(cards.map(c => [c.species, c]));
const forced = [
  ['Hatterene', 'Expanding Force', ['Garchomp', 'Rillaboom', 'Koraidon', 'Zamazenta'], { terrain: 'Psychic' }],
  ['Miraidon', 'Electro Drift', ['Incineroar', 'Pelipper', 'Zamazenta'], { terrain: 'Electric' }],
  ['Koraidon', 'Flare Blitz', ['Rillaboom', 'Amoonguss', 'Zamazenta'], { weather: 'Sun' }],
  ['Pelipper', 'Weather Ball', ['Incineroar', 'Garchomp', 'Koraidon'], { weather: 'Rain' }],
  ['Torkoal', 'Eruption', ['Rillaboom', 'Amoonguss'], { hp: 0.35 }],
  ['Ursaluna', 'Facade', ['Incineroar', 'Zamazenta'], { status: 'brn' }],
  ['Ursaluna-Bloodmoon', 'Blood Moon', ['Flutter Mane', 'Gholdengo', 'Lunala'], {}],
  ['Glimmora', 'Meteor Beam', ['Pelipper', 'Tornadus'], {}],
  ['Zamazenta', 'Body Press', ['Incineroar', 'Kingambit'], {}],
  ['Farigiraf', 'Foul Play', ['Urshifu', 'Kingambit'], {}],
  ['Iron Hands', 'Heavy Slam', ['Flutter Mane', 'Whimsicott'], {}],
  ['Kingambit', 'Sucker Punch', ['Rillaboom', 'Garchomp'], { terrain: 'Psychic' }],
  ['Maushold', 'Population Bomb', ['Archaludon', 'Incineroar', 'Hatterene'], {}],
  ['Chien-Pao', 'Sacred Sword', ['Zamazenta', 'Kingambit'], { dBoost: 1 }],
  // Live pool, Oct 7: Pixilate (Normal -> Fairy, x1.2), sand (Rock-type Sp. Def x1.5), Air Balloon / Levitate vs Ground.
  ['Sylveon', 'Hyper Voice', ['Iron Hands', 'Garchomp', 'Incineroar', 'Whimsicott'], {}],
  ['Tyranitar', 'Rock Slide', ['Gyarados', 'Tornadus', 'Arcanine'], { weather: 'Sand' }],
  ['Thundurus', 'Thunderbolt', ['Tyranitar', 'Gyarados'], { weather: 'Sand' }],
  ['Landorus-Therian', 'Earthquake', ['Gholdengo', 'Cresselia', 'Incineroar'], {}],
  ['Baxcalibur', 'Glaive Rush', ['Dragonite', 'Garchomp'], {}],
  ['Klefki', 'Foul Play', ['Iron Hands', 'Kingambit'], {}],
];
for (const [aName, moveName, defenders, opts] of forced) {
  for (const dName of defenders) {
    const atk = by[aName], def = by[dName];
    if (!atk || !def) continue; // the live catalog changes; a forced case for a card that is gone is just skipped
    const probe = new Move(gen, moveName);
    const offensive = probe.category === 'Physical' ? 'atk' : 'spa', defensive = probe.category === 'Physical' ? 'def' : 'spd';
    const attacker = mon(atk, { status: opts.status, boostedStat: 'auto' });
    if (opts.hp) attacker.originalCurHP = Math.max(1, Math.floor(attacker.maxHP() * opts.hp));
    const defender = mon(def, { boosts: { [defensive]: opts.dBoost || 0 }, boostedStat: 'auto' });
    const field = new Field({ gameType: 'Doubles', weather: opts.weather, terrain: opts.terrain });
    const result = calculate(gen, attacker, defender, new Move(gen, moveName), field);
    const range = result.range();
    cases.push({
      attacker: { species: atk.species, nature: atk.nature, evs: atk.evs, item: atk.item, ability: atk.ability, boosts: {}, status: opts.status || null, hp_fraction: opts.hp || 1.0 },
      defender: { species: def.species, nature: def.nature, evs: def.evs, item: def.item, ability: def.ability, boosts: { [defensive]: opts.dBoost || 0 } },
      move: moveName, defender_hp_fraction: 1.0, defender_max_hp: defender.maxHP(), defender_cur_hp: defender.curHP(),
      field: { weather: opts.weather || null, terrain: opts.terrain || null, defender_reflect: false, defender_light_screen: false, third_ruin: null },
      spread_in_calc: ['allAdjacent', 'allAdjacentFoes'].includes(new Move(gen, moveName).target) || (moveName === 'Expanding Force' && opts.terrain === 'Psychic'),
      damage: [range[0], range[1]],
    });
  }
}
const out = path.join(root, 'tests', 'fixtures', outName);
fs.writeFileSync(out, JSON.stringify(cases));
console.log(`wrote ${cases.length} cases to ${path.relative(root, out)} (calc ${require(path.join(root, 'sim', 'node_modules', '@smogon/calc', 'package.json')).version})`);

// Sanity prints for mechanics the calculator may or may not auto-apply.
const F = new Field({ gameType: 'Doubles' });
const inc = cards.find(c => c.species === 'Incineroar');
const rb = cards.find(c => c.species === 'Raging Bolt');
const target = inc && mon(inc, {});
if (rb && target) console.log('Raging Bolt Thunderbolt w/ Booster:', calculate(gen, mon(rb, { boostedStat: 'auto' }), target, new Move(gen, 'Thunderbolt'), F).range(),
  'w/o ability:', calculate(gen, mon({ ...rb, ability: 'Pressure', item: 'Leftovers' }, {}), target, new Move(gen, 'Thunderbolt'), F).range());
const bas = cards.find(c => c.species === 'Basculegion');
if (bas && target) console.log('Last Respects alliesFainted 2:', calculate(gen, mon(bas, { alliesFainted: 2 }), target, new Move(gen, 'Last Respects'), F).range(), '0:', calculate(gen, mon(bas, { alliesFainted: 0 }), target, new Move(gen, 'Last Respects'), F).range());
