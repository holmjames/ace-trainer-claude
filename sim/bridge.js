// Local Pokémon Showdown battle bridge: JSON lines in, JSON lines out.
//
// The Python harness (sim/harness.py) starts this once and sends commands:
//   {"cmd":"new","id":"g1","format":"gen9vgc2025regi","seed":[1,2,3,4],
//    "p1":{"name":"A","team":"<showdown team text>"},"p2":{"name":"B","team":"..."}}
//   {"cmd":"choose","id":"g1","side":"p1","choice":"move fakeout 2, move tailwind"}
//   {"cmd":"close","id":"g1"}
// Every reply is {"ok":true,"id":...,"state":<snapshot>} or {"ok":false,"error":"..."}.
//
// The snapshot is raw engine state, not the platform's format; sim/translate.py
// turns it into the observation/template shapes our agent sees in a live match.
'use strict';
const readline = require('readline');
const {Battle, Teams} = require('pokemon-showdown');

const battles = new Map();

function pokemonSnapshot(side, mon) {
  const pos = side.active.indexOf(mon);
  return {
    ident: mon.fullname,
    species: mon.species.id,
    name: mon.species.name,
    hp: mon.hp,
    maxhp: mon.maxhp,
    fainted: mon.fainted,
    active: pos >= 0,
    position: pos >= 0 ? pos : null,
    status: mon.status || null,
    item: mon.item || null,
    ability: mon.ability || null,
    types: mon.getTypes(),
    base_stats: mon.species.baseStats,
    boosts: mon.boosts,
    moves: mon.moveSlots.map(m => m.id),
    trapped: !!mon.trapped,
    volatiles: Object.keys(mon.volatiles),
    turns_active: mon.activeTurns,
  };
}

function sideSnapshot(side) {
  return {
    name: side.name,
    request: side.activeRequest || null,
    pokemon: side.pokemon.map(mon => pokemonSnapshot(side, mon)),
    side_conditions: Object.keys(side.sideConditions),
  };
}

function snapshot(b) {
  return {
    turn: b.turn,
    ended: b.ended,
    winner: b.winner || null,
    field: {
      weather: b.field.weather || null,
      terrain: b.field.terrain || null,
      pseudo_weather: Object.keys(b.field.pseudoWeather),
    },
    sides: {p1: sideSnapshot(b.sides[0]), p2: sideSnapshot(b.sides[1])},
    log_tail: b.log.slice(-40),
  };
}

function handle(msg) {
  if (msg.cmd === 'new') {
    const b = new Battle({formatid: msg.format || 'gen9vgc2025regi', seed: msg.seed || undefined});
    b.setPlayer('p1', {name: msg.p1.name || 'p1', team: Teams.pack(Teams.import(msg.p1.team))});
    b.setPlayer('p2', {name: msg.p2.name || 'p2', team: Teams.pack(Teams.import(msg.p2.team))});
    battles.set(msg.id, b);
    return {ok: true, id: msg.id, state: snapshot(b)};
  }
  const b = battles.get(msg.id);
  if (!b) return {ok: false, id: msg.id, error: `no battle ${msg.id}`};
  if (msg.cmd === 'choose') {
    const accepted = b.choose(msg.side, msg.choice);
    if (!accepted) {
      const side = b.getSide(msg.side);
      return {ok: false, id: msg.id, error: `choice rejected for ${msg.side}: ${msg.choice} :: ${side.choice.error || ''}`, state: snapshot(b)};
    }
    return {ok: true, id: msg.id, state: snapshot(b)};
  }
  if (msg.cmd === 'state') return {ok: true, id: msg.id, state: snapshot(b)};
  if (msg.cmd === 'close') { battles.delete(msg.id); return {ok: true, id: msg.id}; }
  return {ok: false, id: msg.id, error: `unknown cmd ${msg.cmd}`};
}

const rl = readline.createInterface({input: process.stdin});
rl.on('line', line => {
  if (!line.trim()) return;
  let msg;
  try { msg = JSON.parse(line); } catch (e) { process.stdout.write(JSON.stringify({ok: false, error: 'bad json'}) + '\n'); return; }
  let reply;
  try { reply = handle(msg); } catch (e) { reply = {ok: false, id: msg.id, error: String(e && e.stack || e)}; }
  process.stdout.write(JSON.stringify(reply) + '\n');
});
