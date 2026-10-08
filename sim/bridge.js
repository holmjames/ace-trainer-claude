// Local Pokémon Showdown battle bridge: JSON lines in, JSON lines out.
//
// The Python harness (sim/harness.py) starts this once and sends commands:
//   {"cmd":"new","id":"g1","format":"gen9vgc2025regi","seed":[1,2,3,4],
//    "p1":{"name":"A","team":"<showdown team text, all 6 drafted>"},"p2":{"name":"B","team":"..."}}
//   {"cmd":"choose","id":"g1","side":"p1","choice":"team 3412"}          (Team Preview: 4 of the 6, leads first)
//   {"cmd":"choose","id":"g1","side":"p1","choice":"move fakeout 2, move tailwind"}
//   {"cmd":"state","id":"g1"}   {"cmd":"close","id":"g1"}
// Every reply is {"ok":true,"id":...,"state":<snapshot>} or {"ok":false,"error":"..."}.
//
// The snapshot is raw engine state plus what a live player's client receives: each side's own
// Showdown request (`request`, and the Team Preview request it saw first, `preview_request`) and each
// side's own view of the protocol log (`logs`), built the way the live server's room sends it —
// |split| blocks resolved for that player (exact HP for its own Pokémon, percentages for the
// opponent's), no |t:| timestamps, a room header (|init|, |title|, |j|) and the battle timer's
// |inactive| lines whenever that side is handed a new request. sim/translate.py turns this into the
// exact observation/legal-action payload the live tournament server sends.
'use strict';
const readline = require('readline');
const {Battle, Teams, Dex} = require('pokemon-showdown');
const {extractChannelMessages} = require('pokemon-showdown/dist/sim/battle');

const battles = new Map();
let battleCounter = 0;
const SIDES = ['p1', 'p2'];

function pokemonSnapshot(side, mon) {
  const pos = side.active.indexOf(mon);
  return {
    ident: mon.fullname,
    species: mon.species.id,
    name: mon.name,
    base_species: mon.species.baseSpecies,
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

function sideSnapshot(g, side) {
  return {
    name: side.name,
    request: side.activeRequest || null,
    preview_request: g.preview[side.id] || null,
    // the six drafted Pokémon in roster order: species id, dex name, base species (the default nickname)
    roster: g.roster[side.id],
    pokemon: side.pokemon.map(mon => pokemonSnapshot(side, mon)),
    side_conditions: Object.keys(side.sideConditions),
  };
}

function snapshot(g) {
  const b = g.b;
  return {
    battle_tag: g.tag,
    turn: b.turn,
    ended: b.ended,
    winner: b.winner || null,
    field: {
      weather: b.field.weather || null,
      terrain: b.field.terrain || null,
      pseudo_weather: Object.keys(b.field.pseudoWeather),
    },
    sides: {p1: sideSnapshot(g, b.sides[0]), p2: sideSnapshot(g, b.sides[1])},
    logs: {p1: g.logs.p1.slice(), p2: g.logs.p2.slice()},
  };
}

// The live room never shows these (timestamps; the Open Team Sheets button being withdrawn).
function keepLine(line) {
  return !line.startsWith('|t:|') && !line.startsWith('|uhtmlchange|');
}

// Pull the engine's new log lines into each player's view; add the room's timer lines when a side is
// handed a new request (the live server's battle timer announces the time left at that moment).
function sync(g) {
  const b = g.b;
  const fresh = b.log.slice(g.pos);
  g.pos = b.log.length;
  if (fresh.length) {
    const channels = extractChannelMessages(fresh.join('\n'), [1, 2]);
    SIDES.forEach((sid, i) => {
      for (const line of channels[i + 1]) if (keepLine(line)) g.logs[sid].push(line);
    });
  }
  SIDES.forEach((sid, i) => {
    const req = b.sides[i].activeRequest;
    if (req && req !== g.lastRequest[sid] && !req.wait) {
      if (req.teamPreview) {
        g.logs[sid].push(`|inactive|Battle timer is ON: inactive players will automatically lose when time's up. (requested by ${g.names.p2})`);
        g.logs[sid].push('|inactive|Time left: 90 sec this turn | 420 sec total | 90 sec grace');
        g.logs[sid].push(`|inactive|${g.names.p1} also wants the timer to be on.`);
      } else {
        g.logs[sid].push('|inactive|Time left: 55 sec this turn | 420 sec total');
      }
    }
    g.lastRequest[sid] = req;
  });
}

function rosterOf(team) {
  return team.map(set => {
    const species = Dex.species.get(set.species);
    return {species: species.id, species_name: species.name, base_species: species.baseSpecies, name: set.name || species.baseSpecies};
  });
}

function handle(msg) {
  if (msg.cmd === 'new') {
    const b = new Battle({formatid: msg.format || 'gen9vgc2025regi', seed: msg.seed || undefined});
    const names = {p1: msg.p1.name || 'p1', p2: msg.p2.name || 'p2'};
    const teams = {p1: Teams.import(msg.p1.team), p2: Teams.import(msg.p2.team)};
    battleCounter += 1;
    const header = ['|init|battle', `|title|${names.p1} vs. ${names.p2}`, `|j|☆${names.p1}`, `|j|☆${names.p2}`];
    const g = {
      b, names, tag: `battle-${msg.format || 'gen9vgc2025regi'}-${battleCounter}`,
      roster: {p1: rosterOf(teams.p1), p2: rosterOf(teams.p2)},
      logs: {p1: header.slice(), p2: header.slice()}, pos: 0,
      lastRequest: {p1: null, p2: null}, preview: {p1: null, p2: null},
    };
    b.setPlayer('p1', {name: names.p1, avatar: msg.p1.avatar || '266', team: Teams.pack(teams.p1)});
    b.setPlayer('p2', {name: names.p2, avatar: msg.p2.avatar || '169', team: Teams.pack(teams.p2)});
    SIDES.forEach((sid, i) => { g.preview[sid] = b.sides[i].activeRequest || null; });
    sync(g);
    battles.set(msg.id, g);
    return {ok: true, id: msg.id, state: snapshot(g)};
  }
  const g = battles.get(msg.id);
  if (!g) return {ok: false, id: msg.id, error: `no battle ${msg.id}`};
  if (msg.cmd === 'choose') {
    const accepted = g.b.choose(msg.side, msg.choice);
    sync(g);
    if (!accepted) {
      const side = g.b.getSide(msg.side);
      return {ok: false, id: msg.id, error: `choice rejected for ${msg.side}: ${msg.choice} :: ${side.choice.error || ''}`, state: snapshot(g)};
    }
    return {ok: true, id: msg.id, state: snapshot(g)};
  }
  if (msg.cmd === 'state') return {ok: true, id: msg.id, state: snapshot(g)};
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
