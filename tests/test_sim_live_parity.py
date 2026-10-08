"""The simulator must hand the agent exactly what the live tournament server hands it.

Ground truth: the payloads captured from real matches in ``tests/fixtures/live/*/`` (draft, Team Preview and
battle states, both seats; each file is a full server state: observation + legal_actions). We lost 0-4 live while
the simulator spoke a slightly different dialect, and every bug hid behind the difference (spread moves arrive as
``targets: [0]``; Team Preview opponents arrive as blank cards; opponent HP is a percentage; ``team`` lists all six).

How this file stays honest:
* ``LIVE_*`` constants below spell the live schema out (keys, key order, value types, conventions). Every live
  fixture is checked against them, so the spec cannot drift from what the server really sends.
* Seeded sim games (bridge -> sim/translate.py -> the harness flow, random legal play, no agent code) are checked
  against the same spec, plus the live fixtures' key paths and value types directly.
* Every captured live battle log is replayed through the simulator's own tracker and must reproduce the server's
  opponent / field state exactly; a scripted battle pins the item-reveal semantics (Air Balloon, Knock Off).
A future drift on either side fails loudly here.
"""

from __future__ import annotations

import collections
import json
import random
import re
import shutil
from pathlib import Path

import pytest

from agent.pokemon import data
from sim import translate

ROOT = Path(__file__).resolve().parent.parent
LIVE = Path(__file__).parent / "fixtures" / "live"
ENGINE = ROOT / "sim" / "node_modules" / "pokemon-showdown"
CARDS = {c["card_id"]: c for c in json.loads((ROOT / "sim" / "cards.json").read_text())}

LIVE_FILES = sorted(LIVE.glob("*/*.json"))
LIVE_STATES = [json.loads(p.read_text()) for p in LIVE_FILES]
LIVE_BY_PHASE: dict[str, list[dict]] = collections.defaultdict(list)
for _raw in LIVE_STATES:
    LIVE_BY_PHASE[_raw["phase"]].append(_raw)

needs_live = pytest.mark.skipif(not all(LIVE_BY_PHASE.get(p) for p in ("draft", "team_preview", "moving")),
                                reason="needs live captures of every phase in tests/fixtures/live/")
needs_engine = pytest.mark.skipif(not shutil.which("node") or not ENGINE.exists(), reason="needs Node and `npm install` in sim/")

# -- the live schema, spelled out --------------------------------------------------------------------------

TOP_KEYS = ["session_id", "game_type", "runtime_adapter", "status", "state_version", "observation", "phase", "is_terminal",
            "is_current_actor", "current_actor", "legal_actions", "next_actions", "updated"]
OPTIONAL_TOP_KEYS = {"updated"}  # absent on the first state of a phase, present on every wait_for_update
DRAFT_OBS_KEYS = ["phase", "draft_complete", "battle_format", "current_seat", "pick_number", "picks_remaining", "first_drafter",
                  "available_card_ids", "available_cards", "rosters", "picks", "unused_card_ids", "decision_timeout_seconds",
                  "decision_deadline_at", "battle_starting"]
CARD_KEYS = ["card_id", "species", "item", "ability", "nature", "level", "evs", "ivs", "moves"]
PICK_KEYS = ["pick_number", "seat_id", "card_id", "species", "public_reason", "auto", "auto_reason"]
PREVIEW_OBS_KEYS = ["phase", "battle_format", "battle_tag", "waiting_for_action", "your_roster", "opponent_roster", "clock"]
BATTLE_OBS_KEYS = ["battle_tag", "turn", "format", "finished", "won", "lost", "weather", "fields", "side_conditions",
                   "opponent_side_conditions", "team", "opponent_team", "is_doubles", "force_switch", "can_tera", "active_pokemon",
                   "opponent_active_pokemon", "available_moves", "available_switches", "target_legend", "protocol_log",
                   "waiting_for_action", "clock"]
MON_KEYS = ["species", "name", "current_hp", "max_hp", "current_hp_fraction", "status", "ability", "item", "types", "base_stats",
            "boosts", "moves", "fainted", "active", "revealed", "is_terastallized"]
BOOST_KEYS = ["accuracy", "atk", "def", "evasion", "spa", "spd", "spe"]
BASE_STAT_KEYS = {"hp", "atk", "def", "spa", "spd", "spe"}
STATUSES = {None, "BRN", "FRZ", "PAR", "PSN", "SLP", "TOX", "FNT"}
OBS_MOVE_KEYS = ["id", "type", "category", "base_power", "accuracy", "priority", "current_pp", "max_pp", "target", "targets", "target_options"]
OPTION_MOVE_KEYS = ["type", "move_id", "base_power", "category", "move_type", "current_pp", "accuracy", "targets", "target_options"]
SLOT_KEYS = ["slot", "board_position", "active", "force_switch", "options"]
DOUBLES_ACTION_KEYS = ["type", "slots", "target_legend", "instructions"]
LINEUP_ACTION_KEYS = ["type", "roster", "bring_count", "lead_count", "instructions"]
INPUT_KEYS = ["session_id", "action", "action_id", "state_version"]
CLOCK_KEYS = ["decision_seconds_left", "bank_seconds_left"]
NO_TARGET_TYPES = {"self", "allySide", "foeSide", "all", "allAdjacent", "allAdjacentFoes", "allies", "allyTeam", "randomNormal", "scripted"}
UPPER_SNAKE = re.compile(r"^[A-Z][A-Z0-9_]*$")
FORBIDDEN_LOG_PREFIXES = ("|split|", "|t:|", "|uhtmlchange|", "|request|")


def _is(value, *types) -> bool:
    """Exact JSON type check (``True`` is not an int here, ``1`` is not a float unless allowed)."""
    return any(type(value) is t for t in types)


def mon_errors(m, where: str, *, opponent: bool, preview_opponent: bool = False) -> list[str]:
    if not isinstance(m, dict):
        return [f"{where}: not a dict ({m!r})"]
    errs = []
    if list(m) != MON_KEYS:
        errs.append(f"{where}: keys {list(m)} != live {MON_KEYS}")
        return errs
    checks = {
        "species": _is(m["species"], str) and m["species"] == data.to_id(m["species"]),
        "name": _is(m["name"], str) and bool(m["name"]),
        "current_hp": _is(m["current_hp"], int) and m["current_hp"] >= 0,
        "max_hp": _is(m["max_hp"], int),
        "current_hp_fraction": _is(m["current_hp_fraction"], float, int),
        "status": m["status"] in STATUSES,
        "ability": _is(m["ability"], str, type(None)) if opponent else _is(m["ability"], str),
        "item": _is(m["item"], str, type(None)) if opponent else _is(m["item"], str),
        "types": isinstance(m["types"], list) and bool(m["types"]) and all(_is(t, str) and t == t.upper() for t in m["types"]),
        "base_stats": isinstance(m["base_stats"], dict) and set(m["base_stats"]) == BASE_STAT_KEYS and all(_is(v, int) for v in m["base_stats"].values()),
        "boosts": isinstance(m["boosts"], dict) and list(m["boosts"]) == BOOST_KEYS and all(_is(v, int) and -6 <= v <= 6 for v in m["boosts"].values()),
        "moves": isinstance(m["moves"], list) and all(_is(x, str) and x == data.to_id(x) for x in m["moves"]),
        "fainted": _is(m["fainted"], bool) and m["fainted"] == (m["status"] == "FNT"),
        "active": _is(m["active"], bool),
        "revealed": _is(m["revealed"], bool),
        "is_terastallized": _is(m["is_terastallized"], bool),
    }
    errs += [f"{where}.{k} = {m[k]!r}" for k, ok in checks.items() if not ok]
    if m["current_hp"] == 0 and m["current_hp_fraction"] != 0:
        errs.append(f"{where}: current_hp 0 but fraction {m['current_hp_fraction']!r}")
    if m["current_hp"] and not (_is(m["current_hp_fraction"], float) and abs(m["current_hp_fraction"] - m["current_hp"] / m["max_hp"]) < 1e-9):
        errs.append(f"{where}: current_hp_fraction {m['current_hp_fraction']!r} != {m['current_hp']}/{m['max_hp']}")
    if opponent and not preview_opponent and m["max_hp"] != 100:
        errs.append(f"{where}: opponent HP must be a percentage (max_hp 100), got {m['max_hp']}")
    if preview_opponent:  # a blank card: nothing about the set is known yet
        info = data.species_info(m["species"]) or {}
        single = [data.to_id(a) for a in info.get("abilities") or []]
        want_ability = single[0] if len(single) == 1 else None
        blank = {"current_hp": 0, "max_hp": 0, "current_hp_fraction": 0, "status": None, "item": "unknown_item", "moves": [],
                 "fainted": False, "active": False, "revealed": False, "ability": want_ability}
        errs += [f"{where}.{k} = {m[k]!r}, live sends {v!r}" for k, v in blank.items() if m[k] != v or type(m[k]) is not type(v)]
    return errs


def _items_named_in(log: list[str]) -> set[str]:
    """Item ids the log names (|-item|, |-enditem|, and "[from] item: X" on damage/heal lines)."""
    out = set()
    for line in log:
        parts = line.split("|")
        if parts[1:2] in (["-item"], ["-enditem"]) and len(parts) > 3:
            out.add(data.to_id(parts[3]))
        for part in parts[3:]:
            if "item:" in part:
                out.add(data.to_id(part.split("item:", 1)[1]))
    return out


def target_errors(entry: dict, slot: int, where: str, mine: list, theirs: list) -> list[str]:
    """The live target convention: fixed by the move's target type and the slot, never by who is standing there."""
    info = data.move_info(entry.get("move_id") or entry.get("id")) or {}
    me, ally = -(slot + 1), -(2 - slot)
    target = info.get("target")
    expected = {"adjacentAlly": [ally], "adjacentAllyOrSelf": [ally, me], "adjacentFoe": [1, 2], "normal": [ally, 1, 2],
                "any": [ally, 1, 2]}.get(target, [0] if target in NO_TARGET_TYPES else None)
    errs = []
    if expected is not None and entry["targets"] != expected:
        errs.append(f"{where}: {entry.get('move_id') or entry.get('id')} ({target}) targets {entry['targets']} != live convention {expected}")
    if [t["target"] for t in entry["target_options"]] != entry["targets"]:
        errs.append(f"{where}: target_options {entry['target_options']} do not mirror targets {entry['targets']}")
    for t in entry["target_options"]:
        if list(t) != ["target", "side", "species"]:
            errs.append(f"{where}: target option keys {list(t)}")
            continue
        if t["target"] == 0:
            if t != {"target": 0, "side": "none", "species": None}:
                errs.append(f"{where}: no-target option {t} != live {{'target': 0, 'side': 'none', 'species': None}}")
            continue
        board = theirs if t["target"] > 0 else mine
        standing = board[abs(t["target"]) - 1]
        side = "opponent" if t["target"] > 0 else ("self" if t["target"] == me else "ally")
        species = standing["species"] if standing else None
        if t["side"] != side or t["species"] != species:
            errs.append(f"{where}: target option {t} != {{'side': {side!r}, 'species': {species!r}}}")
    return errs


def state_errors(raw: dict) -> list[str]:
    errs = []
    expected_top = [k for k in TOP_KEYS if k in raw or k not in OPTIONAL_TOP_KEYS]
    if list(raw) != expected_top:
        errs.append(f"top-level keys {list(raw)} != live {TOP_KEYS}")
    if raw.get("updated", True) is not True:
        errs.append(f"updated = {raw.get('updated')!r}")
    fixed = {"game_type": "pokemon_vgc_doubles_draft", "runtime_adapter": "pokemon", "status": "in_progress", "is_terminal": False,
             "next_actions": [{"tool": "play_action", "hint": "Pick one of legal_actions.actions and pass its `input`."}]}
    errs += [f"{k} = {raw.get(k)!r}" for k, v in fixed.items() if raw.get(k) != v]
    actor = raw.get("current_actor")
    if not (isinstance(actor, dict) and list(actor) == ["agent_id", "position"] and _is(actor["agent_id"], str) and actor["position"] in (0, 1)):
        errs.append(f"current_actor = {actor!r}")
    legal = raw.get("legal_actions") or {}
    if list(legal) != ["session_id", "state_version", "actions"] or legal["session_id"] != raw["session_id"] or legal["state_version"] != raw["state_version"]:
        errs.append(f"legal_actions envelope = {list(legal)}")
    for action in legal.get("actions") or []:
        if list(action) != ["action_id", "label", "input"] or list(action["input"]) != INPUT_KEYS:
            errs.append(f"action keys {list(action)} / input keys {list(action.get('input') or {})}")
        elif action["input"]["action_id"] != action["action_id"] or action["input"]["session_id"] != raw["session_id"] \
                or action["input"]["state_version"] != raw["state_version"]:
            errs.append(f"action input does not echo the envelope: {action['input']}")
    phase = raw.get("phase")
    obs = raw.get("observation") or {}
    if phase == "draft":
        errs += draft_errors(raw, obs)
    elif phase == "team_preview":
        errs += preview_errors(raw, obs)
    elif phase == "moving":
        errs += battle_errors(raw, obs)
    else:
        errs.append(f"unknown phase {phase!r}")
    return errs


def draft_errors(raw: dict, obs: dict) -> list[str]:
    errs = []
    if list(obs) != DRAFT_OBS_KEYS:
        return [f"draft observation keys {list(obs)} != live {DRAFT_OBS_KEYS}"]
    if obs["available_card_ids"] != [c["card_id"] for c in obs["available_cards"]] or obs["unused_card_ids"] != obs["available_card_ids"]:
        errs.append("available_card_ids / unused_card_ids must list every unused card (the Item Clause only filters legal_actions)")
    errs += [f"card keys {list(c)}" for c in obs["available_cards"] if list(c) != CARD_KEYS]
    errs += [f"pick keys {list(p)}" for p in obs["picks"] if list(p) != PICK_KEYS]
    if not (_is(obs["decision_timeout_seconds"], float) and _is(obs["decision_deadline_at"], str) and "T" in obs["decision_deadline_at"]):
        errs.append("decision_timeout_seconds must be a float and decision_deadline_at an ISO timestamp")
    seats = list(obs["rosters"])
    if len(seats) != 2 or obs["current_seat"] not in seats or obs["first_drafter"] not in seats:
        errs.append(f"rosters must be keyed by both seats' agent ids: {seats}")
    elif raw["current_actor"] != {"agent_id": obs["current_seat"], "position": seats.index(obs["current_seat"])}:
        errs.append(f"current_actor {raw['current_actor']} does not match current_seat")
    for seat, roster in obs["rosters"].items():
        errs += [f"roster entry keys {list(e)}" for e in roster if list(e) != ["card_id", "species"]]
    if (obs["pick_number"], obs["picks_remaining"]) != (len(obs["picks"]) + 1, 12 - len(obs["picks"])) or raw["state_version"] != len(obs["picks"]):
        errs.append("pick_number / picks_remaining / state_version out of step with picks")
    by_id = {c["card_id"]: c for c in obs["available_cards"]}
    taken = {data.to_id(CARDS.get(e["card_id"], {}).get("item")) for e in obs["rosters"].get(obs["current_seat"], [])} - {""}
    legal_ids = []
    for a in raw["legal_actions"]["actions"]:
        card_id = a["action_id"].split(":", 1)[1]
        legal_ids.append(card_id)
        if a["action_id"] != f"draft_pick:{card_id}" or a["label"] != f"Draft {card_id}" or a["input"]["action"] != {"type": "draft_pick", "card_id": card_id}:
            errs.append(f"draft action shape {a}")
    expected = [c for c in obs["available_card_ids"] if data.to_id(by_id[c]["item"]) not in taken] if raw["is_current_actor"] else []
    if CARDS and all(e["card_id"] in CARDS for e in obs["rosters"].get(obs["current_seat"], [])) and legal_ids != expected:
        errs.append(f"legal picks {legal_ids} != unused cards minus the Item Clause {expected}")
    return errs


def preview_errors(raw: dict, obs: dict) -> list[str]:
    if list(obs) != PREVIEW_OBS_KEYS:
        return [f"preview observation keys {list(obs)} != live {PREVIEW_OBS_KEYS}"]
    errs = []
    if list(obs["clock"]) != CLOCK_KEYS or not all(_is(v, int) for v in obs["clock"].values()):
        errs.append(f"clock {obs['clock']}")
    if len(obs["your_roster"]) != 6 or len(obs["opponent_roster"]) != 6:
        errs.append("both rosters must list six Pokémon")
    for i, m in enumerate(obs["your_roster"]):
        errs += mon_errors(m, f"your_roster[{i}]", opponent=False)
        if isinstance(m, dict) and len(m.get("moves") or []) != 4:
            errs.append(f"your_roster[{i}] must carry its four moves")
    if [m.get("active") for m in obs["your_roster"]] != [True, True, False, False, False, False]:
        errs.append("your_roster active flags must be the request's (first two true)")
    for i, m in enumerate(obs["opponent_roster"]):
        errs += mon_errors(m, f"opponent_roster[{i}]", opponent=True, preview_opponent=True)
    actions = raw["legal_actions"]["actions"]
    if len(actions) != 1 or actions[0]["action_id"] != "select_lineup" or actions[0]["label"] != translate.LINEUP_LABEL:
        return errs + [f"preview actions {actions}"]
    action = actions[0]["input"]["action"]
    if list(action) != LINEUP_ACTION_KEYS or action["instructions"] != translate.LINEUP_INSTRUCTIONS \
            or (action["bring_count"], action["lead_count"]) != (4, 2):
        errs.append(f"select_lineup template {action}")
    if action.get("roster") != [m["species"] for m in obs["your_roster"]]:
        errs.append("select_lineup roster must be your_roster's species ids, in order")
    if raw["state_version"] != 0:
        errs.append(f"Team Preview state_version {raw['state_version']} (live: 0)")
    return errs


def battle_errors(raw: dict, obs: dict) -> list[str]:
    if list(obs) != BATTLE_OBS_KEYS:
        return [f"battle observation keys {list(obs)} != live {BATTLE_OBS_KEYS}"]
    errs = []
    me = next(iter(obs["team"]), "p1: ?")[:2]
    them = "p2" if me == "p1" else "p1"
    fixed = {"format": "gen9vgc2025regi", "finished": False, "won": None, "lost": None, "is_doubles": True, "waiting_for_action": True,
             "target_legend": translate.TARGET_LEGEND}
    errs += [f"{k} = {obs[k]!r}" for k, v in fixed.items() if obs[k] != v]
    if not (_is(obs["turn"], int) and _is(obs["battle_tag"], str) and obs["battle_tag"].startswith("battle-gen9vgc2025regi-")):
        errs.append("turn / battle_tag")
    for key in ("weather", "fields", "side_conditions", "opponent_side_conditions"):
        d = obs[key]
        if not isinstance(d, dict) or not all(UPPER_SNAKE.match(k) and _is(v, int) for k, v in d.items()):
            errs.append(f"{key} must be a dict of UPPER_SNAKE -> int (the turn it started), got {d!r}")
    if list(obs["clock"]) != CLOCK_KEYS or not all(_is(v, int) for v in obs["clock"].values()):
        errs.append(f"clock {obs['clock']}")
    for key in ("force_switch", "can_tera"):
        if not (isinstance(obs[key], list) and len(obs[key]) == 2 and all(_is(v, bool) for v in obs[key])):
            errs.append(f"{key} = {obs[key]!r}")
    # our team: all six drafted Pokémon, full sets
    if len(obs["team"]) != 6 or not all(k.startswith(f"{me}: ") for k in obs["team"]):
        errs.append(f"team must list all 6 drafted Pokémon keyed '{me}: Name': {list(obs['team'])}")
    for key, m in obs["team"].items():
        errs += mon_errors(m, f"team[{key}]", opponent=False)
        if isinstance(m, dict) and (len(m.get("moves") or []) != 4 or key != f"{me}: {m.get('name')}"):
            errs.append(f"team[{key}] must carry its four moves under its own name")
    # opponents: only those that appeared, known only through the log
    log = obs["protocol_log"]
    appeared = {f"{them}: " + line.split("|")[2].split(": ", 1)[1] for line in log if line.startswith(("|switch|", "|drag|")) and line.split("|")[2].startswith(them)}
    if set(obs["opponent_team"]) != appeared:
        errs.append(f"opponent_team {list(obs['opponent_team'])} != opponents that appeared in the log {sorted(appeared)}")
    used = collections.defaultdict(set)
    for line in log:
        parts = line.split("|")
        if len(parts) > 3 and parts[1] == "move" and parts[2].startswith(them):
            used[f"{them}: " + parts[2].split(": ", 1)[1]].add(data.to_id(parts[3]))
    for key, m in obs["opponent_team"].items():
        errs += mon_errors(m, f"opponent_team[{key}]", opponent=True)
        if isinstance(m, dict):
            if not m["revealed"] or not set(m["moves"]) <= used[key]:
                errs.append(f"opponent_team[{key}] moves {m['moves']} must be moves it used ({sorted(used[key])}) and revealed must be true")
            if m["item"] not in (translate.UNKNOWN_ITEM, None) and m["item"] not in _items_named_in(log):
                errs.append(f"opponent_team[{key}] item {m['item']!r} was never revealed in the log")
    # actives (None for an empty or fainted position) and the slot templates
    for name, side_list, opponent in (("active_pokemon", obs["active_pokemon"], False), ("opponent_active_pokemon", obs["opponent_active_pokemon"], True)):
        if not (isinstance(side_list, list) and len(side_list) == 2):
            errs.append(f"{name} must be a list of 2 (None for an empty position)")
            continue
        for i, m in enumerate(side_list):
            if m is not None:
                errs += mon_errors(m, f"{name}[{i}]", opponent=opponent)
                if m["fainted"] or not m["active"]:
                    errs.append(f"{name}[{i}] lists a fainted or inactive Pokémon; live sends None")
    for key in ("available_moves", "available_switches"):
        if not (isinstance(obs[key], list) and len(obs[key]) == 2 and all(isinstance(x, list) for x in obs[key])):
            errs.append(f"{key} must be a list of 2 lists")
            return errs
    for slot, moves in enumerate(obs["available_moves"]):
        for m in moves:
            if list(m) != OBS_MOVE_KEYS:
                errs.append(f"available_moves[{slot}] keys {list(m)} != live {OBS_MOVE_KEYS}")
                continue
            info = data.move_info(m["id"]) or {}
            if not (m["type"] == m["type"].upper() and m["category"] in ("PHYSICAL", "SPECIAL", "STATUS") and _is(m["accuracy"], float)
                    and _is(m["base_power"], int) and _is(m["priority"], int) and _is(m["current_pp"], int) and _is(m["max_pp"], int)
                    and UPPER_SNAKE.match(m["target"])):
                errs.append(f"available_moves[{slot}] value types {m}")
            if info and m["target"] != re.sub(r"(?<!^)(?=[A-Z])", "_", info["target"]).upper():
                errs.append(f"available_moves[{slot}] {m['id']} target {m['target']} (dex {info['target']})")
            errs += target_errors(m, slot, f"available_moves[{slot}]", obs["active_pokemon"], obs["opponent_active_pokemon"])
    for slot, mons in enumerate(obs["available_switches"]):
        for m in mons:
            errs += mon_errors(m, f"available_switches[{slot}]", opponent=False)
            if isinstance(m, dict) and (m["active"] or m["fainted"]):
                errs.append(f"available_switches[{slot}] lists an active or fainted Pokémon")
    actions = raw["legal_actions"]["actions"]
    if len(actions) != 1 or actions[0]["action_id"] != "doubles_turn" or actions[0]["label"] != translate.DOUBLES_LABEL:
        return errs + [f"battle actions {[(a['action_id'], a['label']) for a in actions]}"]
    action = actions[0]["input"]["action"]
    if list(action) != DOUBLES_ACTION_KEYS or action["type"] != "doubles_turn" or action["target_legend"] != translate.TARGET_LEGEND \
            or action["instructions"] != translate.DOUBLES_INSTRUCTIONS:
        return errs + [f"doubles_turn template keys {list(action)} or text differs from live"]
    for slot_index, slot in enumerate(action["slots"]):
        if list(slot) != SLOT_KEYS or slot["slot"] != slot_index or slot["board_position"] != -(slot_index + 1):
            errs.append(f"slot keys {list(slot)}")
            continue
        if slot["active"] != obs["active_pokemon"][slot_index] or slot["force_switch"] != obs["force_switch"][slot_index]:
            errs.append(f"slot {slot_index} active/force_switch disagree with the observation")
        moves = [o for o in slot["options"] if o.get("type") == "move"]
        switches = [o for o in slot["options"] if o.get("type") == "switch"]
        passes = [o for o in slot["options"] if o.get("type") == "pass"]
        if len(moves) + len(switches) + len(passes) != len(slot["options"]) or not slot["options"]:
            errs.append(f"slot {slot_index} has unknown option types")
        if [o["type"] for o in slot["options"]] != ["move"] * len(moves) + ["switch"] * len(switches) + ["pass"] * len(passes):
            errs.append(f"slot {slot_index} options must be moves, then switches, then pass")
        for o in switches:
            if list(o) != ["type", "species"] or o["species"] != data.to_id(o["species"]):
                errs.append(f"switch option {o} (live: species id)")
        if [o["species"] for o in switches] != [m["species"] for m in obs["available_switches"][slot_index]]:
            if not (any(obs["force_switch"]) and not slot["force_switch"]):
                errs.append(f"slot {slot_index} switches {switches} != available_switches")
        if passes and passes != [{"type": "pass"}]:
            errs.append(f"pass option {passes}")
        expected_moves = [] if any(obs["force_switch"]) else obs["available_moves"][slot_index]
        if [o["move_id"] for o in moves] != [m["id"] for m in expected_moves]:
            errs.append(f"slot {slot_index} move options {[o['move_id'] for o in moves]} != available_moves {[m['id'] for m in expected_moves]}")
        for o, m in zip(moves, expected_moves):
            if list(o) != OPTION_MOVE_KEYS:
                errs.append(f"move option keys {list(o)} != live {OPTION_MOVE_KEYS}")
                continue
            same = {"move_id": "id", "base_power": "base_power", "category": "category", "move_type": "type", "current_pp": "current_pp",
                    "accuracy": "accuracy", "targets": "targets", "target_options": "target_options"}
            errs += [f"move option {o['move_id']}.{k} differs from available_moves" for k, v in same.items() if o[k] != m[v]]
    return errs


# -- the spec is the live format: every captured state satisfies it ------------------------------------------


@needs_live
@pytest.mark.parametrize("path", LIVE_FILES, ids=[f"{p.parent.name[:8]}/{p.name}" for p in LIVE_FILES])
def test_every_live_capture_satisfies_the_spec(path):
    assert state_errors(json.loads(path.read_text())) == []


@needs_live
def test_live_text_constants_are_verbatim():
    moving = LIVE_BY_PHASE["moving"][0]["legal_actions"]["actions"][0]
    assert moving["label"] == translate.DOUBLES_LABEL
    assert moving["input"]["action"]["instructions"] == translate.DOUBLES_INSTRUCTIONS
    assert moving["input"]["action"]["target_legend"] == translate.TARGET_LEGEND == LIVE_BY_PHASE["moving"][0]["observation"]["target_legend"]
    lineup = LIVE_BY_PHASE["team_preview"][0]["legal_actions"]["actions"][0]
    assert lineup["label"] == translate.LINEUP_LABEL and lineup["input"]["action"]["instructions"] == translate.LINEUP_INSTRUCTIONS
    assert LIVE_BY_PHASE["draft"][0]["next_actions"] == translate.NEXT_ACTIONS


# -- the simulator's own log tracker reproduces the live server's view ---------------------------------------


MOVING_FILES = [p for p, raw in zip(LIVE_FILES, LIVE_STATES) if raw["phase"] == "moving"]


@needs_live
@pytest.mark.parametrize("path", MOVING_FILES, ids=[f"{p.parent.name[:8]}/{p.name}" for p in MOVING_FILES])
def test_replaying_a_live_log_reproduces_the_live_opponent_and_field_state(path):
    obs = json.loads(path.read_text())["observation"]
    view = translate.BattleView(next(iter(obs["team"]))[:2]).feed_all(obs["protocol_log"])
    assert {k: m.to_obs() for k, m in view.opponent_team.items()} == obs["opponent_team"]
    assert list(view.opponent_team) == list(obs["opponent_team"])  # order of first appearance
    assert [m.to_obs() if m else None for m in view.opponent_active_pokemon] == obs["opponent_active_pokemon"]
    assert (view.turn, view.weather, view.fields, view.side_conditions, view.opponent_side_conditions) == \
        (obs["turn"], obs["weather"], obs["fields"], obs["side_conditions"], obs["opponent_side_conditions"])
    for key, mon in view.team.items():  # what the log alone decides about our side (the rest comes from requests)
        live = obs["team"][key]
        assert {k: mon.to_obs()[k] for k in ("boosts", "revealed", "fainted", "current_hp", "max_hp")} == \
            {k: live[k] for k in ("boosts", "revealed", "fainted", "current_hp", "max_hp")}, key
    assert [m.species if m else None for m in view.active_pokemon] == [m and m["species"] for m in obs["active_pokemon"]]


@needs_live
def test_live_team_preview_opponents_are_blank_cards_the_sim_reproduces():
    for raw in LIVE_BY_PHASE["team_preview"]:
        for live in raw["observation"]["opponent_roster"]:
            assert translate.Mon(live["species"], name=live["name"]).to_obs() == live


# -- simulator states, produced the way the harness produces them ----------------------------------------------


class _RandomLegal:
    """Seeded random legal play straight off the templates (no agent code, so agent changes cannot break parity)."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.states: list[dict] = []

    def choose_action(self, state, context):
        self.states.append(state.raw)
        first = state.legal_actions[0]
        if first.action_id.startswith("draft_pick:"):
            return self.rng.choice(state.legal_actions)
        template = first.input["action"]
        if template["type"] == "select_lineup":
            bring = self.rng.sample(template["roster"], 4)
            return {"type": "select_lineup", "bring": bring, "leads": bring[:2]}
        answer, taken = {"type": "doubles_turn"}, None
        for slot in template["slots"]:
            options = [o for o in slot["options"] if not (o["type"] == "switch" and o["species"] == taken)]
            option = self.rng.choice(options)
            if option["type"] == "move":
                answer[f"slot_{slot['slot']}"] = {"type": "move", "move_id": option["move_id"], "target": self.rng.choice(option["targets"])}
            elif option["type"] == "switch":
                taken = option["species"]
                answer[f"slot_{slot['slot']}"] = {"type": "switch", "species": option["species"]}
            else:
                answer[f"slot_{slot['slot']}"] = {"type": "pass"}
        return answer


@pytest.fixture(scope="module")
def sim_games():
    from sim import harness

    bridge = harness.Bridge()
    games = []
    try:
        for seed in (11, 12, 13):
            rng = random.Random(seed)
            players = {"p1": _RandomLegal(rng), "p2": _RandomLegal(rng)}
            result = harness.play_game(bridge, f"parity-{seed}", players, {"p1": "a", "p2": "b"}, rng, verbose=False)
            games.append((result, players))
    finally:
        bridge.close()
    return games


def _sim_states(sim_games, phase: str | None = None) -> list[dict]:
    states = [s for _, players in sim_games for p in players.values() for s in p.states]
    return [s for s in states if phase is None or s["phase"] == phase]


@needs_engine
def test_sim_games_cover_every_phase_without_rejected_choices(sim_games):
    for result, _ in sim_games:
        assert result["rejected"] == {"p1": 0, "p2": 0}, result
    counts = collections.Counter(s["phase"] for s in _sim_states(sim_games))
    assert counts["draft"] == 3 * 12 and counts["team_preview"] == 3 * 2 and counts["moving"] >= 20, counts
    moving = _sim_states(sim_games, "moving")
    assert any(any(s["observation"]["force_switch"]) for s in moving), "no forced switch exercised"
    assert any(None in s["observation"]["active_pokemon"] or None in s["observation"]["opponent_active_pokemon"] for s in moving)


@needs_engine
def test_every_sim_state_satisfies_the_live_spec(sim_games):
    failures = {f"{s['phase']} v{s['state_version']} ({s['current_actor']['position']})": errs
                for s in _sim_states(sim_games) if (errs := state_errors(s))}
    assert failures == {}


def _schema(value, path: str, out: dict) -> None:
    out[path].add(type(value).__name__)
    if isinstance(value, dict):
        for k, v in value.items():
            if path.endswith((".team", ".opponent_team", ".rosters")):
                k = "<key>"
            elif path.endswith(("side_conditions", ".weather", ".fields")):
                k = "<COND>"
            _schema(v, f"{path}.{k}", out)
    elif isinstance(value, list) and not path.endswith("protocol_log"):
        for v in value:
            _schema(v, f"{path}[]", out)


# Types the captures happen not to contain but the live format allows (no capture had a statused active Pokémon
# or weather, and the battle never ended in a captured state).
ALLOWED_UNCAPTURED = {
    "$.observation.active_pokemon[].status": {"str"},
    "$.observation.opponent_active_pokemon[].status": {"str"},
    "$.legal_actions.actions[].input.action.slots[].active.status": {"str"},
    "$.observation.available_switches[][].status": {"str"},
    "$.observation.weather.<COND>": {"int"},
    "$.observation.fields.<COND>": {"int"},
}


@needs_live
@needs_engine
@pytest.mark.parametrize("phase", ["draft", "team_preview", "moving"])
def test_sim_key_paths_and_value_types_match_the_live_captures(sim_games, phase):
    live: dict[str, set] = collections.defaultdict(set)
    sim: dict[str, set] = collections.defaultdict(set)
    for raw in LIVE_BY_PHASE[phase]:
        _schema(raw, "$", live)
    for raw in _sim_states(sim_games, phase):
        _schema(raw, "$", sim)
    extra = {p: sorted(t) for p, t in sim.items() if p not in live and p not in ALLOWED_UNCAPTURED}
    wrong = {p: (sorted(t), sorted(live[p])) for p, t in sim.items() if p in live and t - live[p] - ALLOWED_UNCAPTURED.get(p, set())}
    always = set.intersection(*[_paths(r) for r in LIVE_BY_PHASE[phase]])  # paths every live state of this phase has
    missing = sorted(always - set.intersection(*[_paths(r) for r in _sim_states(sim_games, phase)]))
    assert (extra, wrong, missing) == ({}, {}, [])


def _paths(raw: dict) -> set[str]:
    out: dict[str, set] = collections.defaultdict(set)
    _schema(raw, "$", out)
    return set(out)


@needs_engine
def test_sim_spread_moves_use_the_live_zero_target_convention(sim_games):
    seen = collections.Counter()
    for raw in _sim_states(sim_games, "moving"):
        for slot, moves in enumerate(raw["observation"]["available_moves"]):
            for m in moves:
                if (data.move_info(m["id"]) or {}).get("target") in NO_TARGET_TYPES:
                    assert m["targets"] == [0] and m["target_options"] == [{"target": 0, "side": "none", "species": None}], m
                    seen["none"] += 1
                elif (data.move_info(m["id"]) or {}).get("target") in ("normal", "any"):
                    assert m["targets"] == [-(2 - slot), 1, 2], m  # the ally position is always listed first
                    seen["single"] += 1
    assert seen["none"] and seen["single"]


@needs_engine
def test_sim_protocol_log_is_the_full_per_player_log(sim_games):
    for raw in _sim_states(sim_games, "moving"):
        obs = raw["observation"]
        log, me = obs["protocol_log"], next(iter(obs["team"]))[:2]
        assert log[0] == "|init|battle" and log[1].startswith("|title|") and log[2].startswith("|j|☆") and log[3].startswith("|j|☆")
        assert "|teampreview|4" in log and "|start" in log and sum(line.startswith("|poke|") for line in log) == 12
        assert not [line for line in log if line.startswith(FORBIDDEN_LOG_PREFIXES)]
        assert [line for line in log if line.startswith("|turn|")][-1] == f"|turn|{obs['turn']}"
        for line in log:  # the opponent's HP in percent, ours exact (out of the real max HP)
            parts = line.split("|")
            if parts[1:2] in (["switch"], ["drag"], ["-damage"], ["-heal"]) and len(parts) > (4 if parts[1] in ("switch", "drag") else 3):
                hp = parts[4 if parts[1] in ("switch", "drag") else 3].split()[0]
                if hp != "0":
                    max_hp = int(hp.split("/")[1])
                    assert (max_hp == 100) if not parts[2].startswith(me) else (max_hp == obs["team"][f"{me}: " + parts[2].split(": ", 1)[1]]["max_hp"]), line
    by_seat: dict[tuple, list] = collections.defaultdict(list)
    for raw in _sim_states(sim_games, "moving"):
        by_seat[(raw["session_id"], raw["current_actor"]["agent_id"])].append(raw)
    for states in by_seat.values():  # cumulative: each state's log extends the previous one
        states.sort(key=lambda r: r["state_version"])
        assert [r["state_version"] for r in states] == list(range(1, len(states) + 1))
        for earlier, later in zip(states, states[1:]):
            assert later["observation"]["protocol_log"][: len(earlier["observation"]["protocol_log"])] == earlier["observation"]["protocol_log"]


@needs_engine
def test_sim_team_lists_all_six_and_unbrought_ones_keep_their_preview_flags(sim_games):
    for result, players in sim_games:
        for side in ("p1", "p2"):
            states = players[side].states
            preview = next(s for s in states if s["phase"] == "team_preview")["observation"]["your_roster"]
            brought = {data.to_id(sp) for sp in result["teams"][side]}
            for raw in (s for s in states if s["phase"] == "moving"):
                team = raw["observation"]["team"]
                assert [m["species"] for m in team.values()] == [m["species"] for m in preview]  # roster order, all six
                for (key, mon), card in zip(team.items(), preview):
                    if mon["species"] not in brought:  # never sent out: stale Team Preview request data, as live
                        assert (mon["active"], mon["revealed"], mon["item"], mon["moves"]) == (card["active"], False, card["item"], card["moves"])


@needs_engine
def test_sim_draft_states_match_the_live_draft(sim_games):
    drafts = _sim_states(sim_games, "draft")
    for raw in drafts:
        obs = raw["observation"]
        assert raw["is_current_actor"] and raw["legal_actions"]["actions"], "live only asks the seat that is picking"
        assert re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", obs["current_seat"])
    assert any(len(r["legal_actions"]["actions"]) < len(r["observation"]["available_card_ids"]) for r in drafts), \
        "the Item Clause never filtered a legal pick while leaving the card in available_cards"


# -- a scripted battle pins the reveal semantics -----------------------------------------------------------------


def _team(*card_ids: str) -> str:
    return translate.team_text([CARDS[c] for c in card_ids])


@needs_engine
def test_scripted_battle_item_reveals_knock_off_and_the_stale_active_flag():
    """p1 leaves its first roster slot (Pelipper) at home; p2 leads Gholdengo @ Air Balloon and Tyranitar (Knock Off)."""
    from sim import harness

    bridge = harness.Bridge()
    try:
        snap = bridge.call(cmd="new", id="script", format="gen9vgc2025regi", seed=[7, 7, 7, 7],
                           p1={"name": "a-p0", "team": _team("vgc-pelipper", "vgc-incineroar", "vgc-garchomp", "vgc-whimsicott", "vgc-amoonguss", "vgc-archaludon")},
                           p2={"name": "b-p1", "team": _team("vgc-gholdengo", "vgc-tyranitar", "vgc-iron-bundle", "vgc-cresselia", "vgc-dondozo", "vgc-basculegion")})["state"]
        preview = translate.preview_state("script", snap, "p1", agent_id="seat-0")
        roster = preview.raw["observation"]["opponent_roster"]
        assert [m["species"] for m in roster] == ["gholdengo", "tyranitar", "ironbundle", "cresselia", "dondozo", "basculegion"]
        assert all(m["item"] == "unknown_item" and m["moves"] == [] and m["max_hp"] == 0 for m in roster)
        assert [m["ability"] for m in roster][:3] == ["goodasgold", None, "quarkdrive"]  # single-ability species only
        assert bridge.call(cmd="choose", id="script", side="p1", choice="team 2345")["ok"]
        snap = bridge.call(cmd="choose", id="script", side="p2", choice="team 1234")["state"]
        turn1 = translate.battle_state("script", snap, "p1", 1, agent_id="seat-0").raw["observation"]
        opp = turn1["opponent_team"]
        assert list(opp) == ["p2: Gholdengo", "p2: Tyranitar"]  # only what has appeared
        assert opp["p2: Gholdengo"]["item"] == "airballoon"  # |-item| on switch-in reveals it
        assert opp["p2: Tyranitar"]["item"] == "unknown_item" and opp["p2: Tyranitar"]["ability"] is None
        assert turn1["weather"] == {"SANDSTORM": 0}
        team = turn1["team"]
        assert list(team) == ["p1: Pelipper", "p1: Incineroar", "p1: Garchomp", "p1: Whimsicott", "p1: Amoonguss", "p1: Archaludon"]
        assert team["p1: Pelipper"]["active"] is True and team["p1: Pelipper"]["revealed"] is False  # stale preview flag
        assert team["p1: Archaludon"]["active"] is False and team["p1: Archaludon"]["item"] == "leftovers"
        eq = next(m for m in turn1["available_moves"][1] if m["id"] == "earthquake")
        assert eq["targets"] == [0] and eq["target_options"] == [{"target": 0, "side": "none", "species": None}]
        knock = next(m for m in turn1["available_moves"][0] if m["id"] == "knockoff")
        assert knock["targets"] == [-2, 1, 2] and [t["species"] for t in knock["target_options"]] == ["garchomp", "gholdengo", "tyranitar"]
        assert bridge.call(cmd="choose", id="script", side="p1", choice="move knockoff 1, move protect")["ok"]
        snap = bridge.call(cmd="choose", id="script", side="p2", choice="move nastyplot, move knockoff 1")["state"]
        log = snap["logs"]["p1"]
        assert "|-enditem|p2a: Gholdengo|Air Balloon" in log
        turn2 = translate.battle_state("script", snap, "p1", 2, agent_id="seat-0").raw["observation"]
        assert turn2["opponent_team"]["p2: Gholdengo"]["item"] is None  # consumed / popped: null
        assert turn2["opponent_team"]["p2: Tyranitar"]["moves"] == ["knockoff"]
        assert turn2["opponent_team"]["p2: Gholdengo"]["max_hp"] == 100
        assert turn2["team"]["p1: Incineroar"]["item"] == ""  # our own item, knocked off: "" (the request's spelling)
        p2_view = translate.battle_state("script", snap, "p2", 2, agent_id="seat-1").raw["observation"]
        assert p2_view["opponent_team"]["p1: Incineroar"]["moves"] == ["knockoff"]
        assert p2_view["opponent_team"]["p1: Incineroar"]["item"] is None and p2_view["opponent_team"]["p1: Incineroar"]["ability"] == "intimidate"
        assert p2_view["team"]["p2: Gholdengo"]["item"] == ""
    finally:
        bridge.close()
