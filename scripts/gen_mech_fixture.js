#!/usr/bin/env node
// Ground truth for the sheet mechanics of Workstream D batch 1, mech lane (tests/test_mech_d.py): weight-based
// moves, spread and field in switch previews, multi-hit moves as a hit-count distribution, Supreme Overlord and
// Intimidate on entry, all from Smogon's official calculator (@smogon/calc). Writes tests/fixtures/smogon_calc_mech.json.
//
//   node scripts/gen_mech_fixture.js                                   # uses sim/node_modules/@smogon/calc
//   SMOGON_CALC_NODE_MODULES=/other/node_modules node scripts/gen_mech_fixture.js   # any other install (same version)
//
// Named cases (a test looks them up by name) plus deterministic searches over the live catalog (data/cards.json, in
// catalog order) for the positions the tests need: a target only the fifth strike of a Loaded Dice multi-hit can KO,
// a pivot that is KO'd on entry at +0 Attack but survives at -1 (Intimidate), a spread move that KOs at single-target
// power but not at the doubles 0.75x, a redirector that survives a hit only behind its side's screen, and a lead that
// Hurricane knocks out from full HP (the 70% accuracy case). Skipped on purpose: Focus Sash holders (we cap a single
// hit at 99% through an intact sash) and Multiscale defenders of multi-hit moves (every strike is halved in our model
// until batch 2; the calculator halves the first only).
'use strict';
const fs = require('fs');
const path = require('path');
const root = path.resolve(__dirname, '..');
const nodeModules = process.env.SMOGON_CALC_NODE_MODULES || path.join(root, 'sim', 'node_modules');
const calc = require(path.join(nodeModules, '@smogon/calc'));
const { calculate, Pokemon, Move, Field, Generations } = calc;
const gen = Generations.get(9);
const cards = JSON.parse(fs.readFileSync(path.join(root, 'data', 'cards.json'), 'utf8'));
const by = Object.fromEntries(cards.map(c => [c.species, c]));
const SPREAD = ['allAdjacent', 'allAdjacentFoes'];

function mon(card, opts = {}) {
  return new Pokemon(gen, card.species, { level: 50, nature: card.nature, evs: card.evs, item: card.item, ability: card.ability, boostedStat: 'auto', ...opts });
}
function spec(card, p, extra = {}) {
  return { species: card.species, nature: card.nature, evs: card.evs, item: card.item, ability: card.ability, boosts: p.boosts, status: p.status || null, ...extra };
}

const cases = [];
// One calculator call -> one fixture case. `a`/`d` are catalog cards (optionally with fields overridden), `opts`:
// aBoosts/dBoosts, aStatus, alliesFainted, hits, hp (defender HP fraction), weather/terrain, reflect/lightScreen (the
// defender's side), gameType ('Doubles' unless 'Singles' is asked for, which turns the spread modifier off).
function run(name, a, moveName, d, opts = {}) {
  const attacker = mon(a, { boosts: opts.aBoosts || {}, status: opts.aStatus, alliesFainted: opts.alliesFainted });
  const defender = mon(d, { boosts: opts.dBoosts || {} });
  const hp = opts.hp == null ? 1.0 : opts.hp;
  defender.originalCurHP = Math.max(1, Math.floor(defender.maxHP() * hp));
  const gameType = opts.gameType || 'Doubles';
  const field = new Field({ gameType, weather: opts.weather, terrain: opts.terrain,
    defenderSide: { isReflect: !!opts.reflect, isLightScreen: !!opts.lightScreen } });
  const move = new Move(gen, moveName, opts.hits ? { hits: opts.hits } : {});
  const result = calculate(gen, attacker, defender, move, field);
  const range = result.range();
  const c = {
    name,
    attacker: spec(a, attacker, { hp_fraction: 1.0, allies_fainted: opts.alliesFainted || 0 }),
    defender: spec(d, defender),
    move: moveName, hits: opts.hits || null,
    defender_hp_fraction: hp, defender_max_hp: defender.maxHP(), defender_cur_hp: defender.curHP(),
    field: { weather: opts.weather || null, terrain: opts.terrain || null, defender_reflect: !!opts.reflect, defender_light_screen: !!opts.lightScreen, game_type: gameType },
    spread_in_calc: gameType !== 'Singles' && SPREAD.includes(move.target),
    damage: [range[0], range[1]],
  };
  cases.push(c);
  return c;
}
const range = (a, moveName, d, opts) => { const c = run('_probe', a, moveName, d, opts); cases.pop(); return c; };
const damaging = (card) => card.moves.filter(m => { const probe = new Move(gen, m); return probe.category !== 'Status' && !['Beat Up', 'Ruination'].includes(m); });
const sash = (card) => card.item === 'Focus Sash';

// -- D1b: weight-based power ---------------------------------------------------------------------------------------
run('lowkick_heavy', by['Tyranitar'], 'Low Kick', by['Ting-Lu']);            // 699.7 kg -> 120 power
run('lowkick_kingambit', by['Tyranitar'], 'Low Kick', by['Kingambit']);      // 120 kg -> 100 power (the invisible 81-97%)
run('lowkick_light', by['Tyranitar'], 'Low Kick', by['Scream Tail']);        // 8 kg -> 20 power
run('grassknot_heavy', by['Rillaboom'], 'Grass Knot', by['Dondozo']);        // 220 kg -> 120 power
run('grassknot_light', by['Rillaboom'], 'Grass Knot', by['Scream Tail']);    // 8 kg -> 20 power
run('heatcrash_light', by['Torkoal'], 'Heat Crash', by['Scream Tail']);      // 80.4 / 8 -> 120 power
run('heatcrash_heavy', by['Torkoal'], 'Heat Crash', by['Dondozo']);          // 80.4 / 220 -> 40 power
run('heavyslam_light', by['Dondozo'], 'Heavy Slam', by['Flutter Mane']);     // 220 / 4 -> 120 power
run('heavyslam_mid', by['Dondozo'], 'Heavy Slam', by['Garchomp']);           // 220 / 95 -> 60 power

// -- D1c: a Hurricane that knocks a lead out from full HP (70% accurate; sure-hit in rain) ------------------------
for (const atkName of ['Salamence', 'Pelipper']) {
  for (const d of cards) {
    if (d.species === atkName || sash(d)) continue;
    const r = range(by[atkName], 'Hurricane', d);
    if (r.damage[0] >= r.defender_max_hp) {
      run(`hurricane_ohko_${atkName.toLowerCase()}`, by[atkName], 'Hurricane', d);
      run(`hurricane_ohko_${atkName.toLowerCase()}_rain`, by[atkName], 'Hurricane', d, { weather: 'Rain' });
      break;
    }
  }
}

// -- D1k: spread moves into a pair under the field --------------------------------------------------------------
run('rockslide_spread_sand_reflect_tornadus', by['Garchomp'], 'Rock Slide', by['Tornadus'], { weather: 'Sand', reflect: true });
run('rockslide_spread_sand_reflect_salamence', by['Garchomp'], 'Rock Slide', by['Salamence'], { weather: 'Sand', reflect: true });
run('gleam_spread_sand_screen_tyranitar', by['Hatterene'], 'Dazzling Gleam', by['Tyranitar'], { weather: 'Sand', lightScreen: true });
run('gleam_spread_sand_screen_kingambit', by['Hatterene'], 'Dazzling Gleam', by['Kingambit'], { weather: 'Sand', lightScreen: true });
// A spread move that KOs an incoming Pokémon at single-target power but not at the doubles 0.75x, from an attacker
// none of whose other moves KOs it either (the sheet's pivot check looks at every move).
const kills = (r) => r.damage[0] >= r.defender_cur_hp;
search: for (const hp of [1.0, 0.8, 0.7, 0.6, 0.5]) {
  for (const a of cards) {
    const moves = damaging(a);
    const spreadMoves = moves.filter(m => SPREAD.includes(new Move(gen, m).target));
    if (!spreadMoves.length) continue;
    for (const d of cards) {
      if (d === a || sash(d)) continue;
      if (moves.some(m => kills(range(a, m, d, { hp })))) continue;  // something still KOs it in doubles
      const m = spreadMoves.find(m => kills(range(a, m, d, { hp, gameType: 'Singles' })));
      if (m) {
        run('spread_pivot_single', a, m, d, { hp, gameType: 'Singles' });
        run('spread_pivot_doubles', a, m, d, { hp });
        break search;
      }
    }
  }
}
// A redirector (Rage Powder / Follow Me) that survives a single-target hit only behind its side's screen: every
// single-target move of the attacker can KO it in the open, and the screen makes one of them survivable.
search: for (const hp of [1.0, 0.8, 0.6]) {
  for (const a of cards) {
    const singles = damaging(a).filter(m => { const p = new Move(gen, m); return !SPREAD.includes(p.target) && p.priority <= 0; });
    if (!singles.length) continue;
    for (const dName of ['Amoonguss', 'Indeedee-F']) {
      const d = by[dName];
      if (!d || d === a) continue;
      const open = singles.map(m => range(a, m, d, { hp }));
      if (!open.every(r => r.damage[1] >= r.defender_cur_hp)) continue;
      const screenFor = (m) => new Move(gen, m).category === 'Physical' ? { reflect: true } : { lightScreen: true };
      const m = singles.find(m => { const r = range(a, m, d, { hp, ...screenFor(m) }); return r.damage[1] < r.defender_cur_hp; });
      if (m) {
        run('redirect_open', a, m, d, { hp });
        run('redirect_screen', a, m, d, { hp, ...screenFor(m) });
        break search;
      }
    }
  }
}

// -- D1e: multi-hit as a distribution: a target only the fifth strike of a Loaded Dice move can KO -----------------
for (const [atkName, moveName] of [['Dragonite', 'Scale Shot'], ['Baxcalibur', 'Icicle Spear']]) {
  const a = by[atkName];
  search: for (const hp of [1.0, 0.85, 0.7, 0.6, 0.5]) {
    for (const d of cards) {
      if (d === a || sash(d) || d.ability === 'Multiscale') continue;
      const four = range(a, moveName, d, { hp, hits: 4 });
      const five = range(a, moveName, d, { hp, hits: 5 });
      if (four.damage[1] < four.defender_cur_hp && five.damage[1] >= five.defender_cur_hp) {
        const tag = moveName.toLowerCase().replace(/\s+/g, '');
        run(`${tag}_1hit`, a, moveName, d, { hp, hits: 1 });
        run(`${tag}_4hits`, a, moveName, d, { hp, hits: 4 });
        run(`${tag}_5hits`, a, moveName, d, { hp, hits: 5 });
        break search;
      }
    }
  }
}

// Multiscale halves only the FIRST strike of a multi-hit move (post-mortem: Icicle Spear into a full-HP Dragonite was
// 116-136% in the sheet with every strike halved; the calculator says 177-212% for four strikes).
for (const hits of [1, 4, 5]) run(`multiscale_iciclespear_dragonite_${hits}hits`, by['Baxcalibur'], 'Icicle Spear', by['Dragonite'], { hits });
run('multiscale_iciclespear_dragonite_calc_default', by['Baxcalibur'], 'Icicle Spear', by['Dragonite'], {});
run('multiscale_moonblast_dragonite', by['Flutter Mane'], 'Moonblast', by['Dragonite'], {});

// -- D1d: Supreme Overlord ---------------------------------------------------------------------------------------
for (const n of [0, 2, 3, 5]) run(`overlord_kowtow_cresselia_${n}`, by['Kingambit'], 'Kowtow Cleave', by['Cresselia'], { alliesFainted: n });
run('overlord_sucker_fluttermane_3', by['Kingambit'], 'Sucker Punch', by['Flutter Mane'], { alliesFainted: 3 });

// -- Intimidate on entry: a pivot that dies at +0 Attack and lives at -1 ------------------------------------------
// The calculator applies the defender's Intimidate only when abilityOn is set (it is not here), so the +0 case is the
// "no Intimidate" baseline whatever ability the incoming Pokémon has; the -1 case is what one Intimidate leaves.
// Some move KOs the incoming Incineroar at +0 and NONE of the attacker's moves does at -1 (the sheet checks them all).
search: for (const hp of [1.0, 0.8, 0.7, 0.6, 0.5]) {
  for (const a of cards) {
    if (a.species === 'Incineroar') continue;
    const moves = damaging(a);
    if (moves.some(m => kills(range(a, m, by['Incineroar'], { hp, aBoosts: { atk: -1 } })))) continue;
    const plain = moves.map(m => [m, range(a, m, by['Incineroar'], { hp })]).filter(([, r]) => kills(r));
    if (plain.length) {
      const [m] = plain.sort((x, y) => y[1].damage[0] - x[1].damage[0])[0];
      run('intimidate_pivot_plain', a, m, by['Incineroar'], { hp });
      run('intimidate_pivot_minus1', a, m, by['Incineroar'], { hp, aBoosts: { atk: -1 } });
      break search;
    }
  }
}

// For lead_ohkos: a FASTER attacker that knocks one of our leads out from full HP at +0 Attack and with none of its
// moves at -1 (the partner lead is Incineroar). No Choice Scarf / Booster Energy (speed the calculator does not
// apply) and no Ruin abilities (field effects our race applies and the calculator is not told about) on either side.
const speedOf = (card) => mon(card).stats.spe;
const skipLead = (card) => ['Choice Scarf', 'Booster Energy'].includes(card.item) || /Ruin$/.test(card.ability) || card.species === 'Incineroar';
search: for (const a of cards) {
  if (skipLead(a)) continue;
  for (const d of cards) {
    if (d === a || skipLead(d) || sash(d) || speedOf(a) <= speedOf(d)) continue;
    const moves = damaging(a);
    if (moves.some(m => kills(range(a, m, d, { aBoosts: { atk: -1 } })))) continue;
    const plain = moves.map(m => [m, range(a, m, d, {})]).filter(([, r]) => kills(r));
    if (plain.length) {
      const [m] = plain.sort((x, y) => y[1].damage[0] - x[1].damage[0])[0];
      run('lead_intimidate_plain', a, m, d, {});
      run('lead_intimidate_minus1', a, m, d, { aBoosts: { atk: -1 } });
      break search;
    }
  }
}

const out = path.join(root, 'tests', 'fixtures', 'smogon_calc_mech.json');
fs.writeFileSync(out, JSON.stringify(cases, null, 0));
const names = cases.map(c => c.name);
console.log(`wrote ${cases.length} cases to ${path.relative(root, out)} (calc ${require(path.join(nodeModules, '@smogon/calc', 'package.json')).version})`);
for (const c of cases) console.log(`  ${c.name}: ${c.attacker.species} ${c.move}${c.hits ? ' x' + c.hits : ''} -> ${c.defender.species} @${c.defender_hp_fraction} ${JSON.stringify(c.damage)} / ${c.defender_cur_hp}`);
const dup = names.filter((n, i) => names.indexOf(n) !== i);
if (dup.length) { console.error('duplicate case names:', dup); process.exit(1); }
