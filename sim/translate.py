"""Turn local-engine snapshots (sim/bridge.js) into the GameState shapes our agent sees live.

Plain language: the real platform wraps Pokémon Showdown and hands our agent a JSON
observation plus a "template" of legal options. Our local bridge exposes raw Showdown state.
This module builds the same observation and template from it, so the agent runs the exact
code path it will run in the tournament — and the agent's answer is translated back into a
Showdown choice string ("move fakeout 2, move tailwind").

Field names follow the platform guide (api.altruagent-game.com/skill/pokemon). Where the
guide leaves a detail open we choose the most natural shape and keep the agent tolerant.
"""

from __future__ import annotations

from typing import Any

from altruagent.models import GameState

from agent.pokemon import data

SINGLE_TARGET = {"normal", "any", "adjacentFoe", "randomNormal", "scripted"}
NO_TARGET = {"self", "allySide", "foeSide", "all", "allAdjacent", "allAdjacentFoes", "allies", "allyTeam"}


# -- helpers -----------------------------------------------------------------------------------


def _mon_summary(mon: dict, *, full: bool) -> dict:
    out = {
        "species": mon["species"], "name": mon["name"],
        "current_hp": mon["hp"], "max_hp": mon["maxhp"],
        "current_hp_fraction": (mon["hp"] / mon["maxhp"]) if mon["maxhp"] else 0.0,
        "status": mon.get("status"), "types": [t.upper() for t in mon.get("types") or []],
        "base_stats": mon.get("base_stats"), "boosts": {k: v for k, v in (mon.get("boosts") or {}).items() if v},
        "fainted": mon.get("fainted", False), "active": mon.get("active", False),
    }
    if full:
        out.update({"item": mon.get("item"), "ability": mon.get("ability"), "moves": mon.get("moves") or [], "revealed": True})
    else:
        out["revealed"] = True
    return out


def _state(session_id: str, phase: str, observation: dict, legal: list[dict], version: int, *, my_turn: bool = True) -> GameState:
    return GameState.from_mcp_state({
        "session_id": session_id, "game_type": "pokemon_vgc_doubles_draft", "runtime_adapter": "pokemon",
        "status": "in_progress", "state_version": version, "phase": phase, "observation": observation,
        "is_terminal": False, "is_current_actor": my_turn,
        "legal_actions": {"session_id": session_id, "state_version": version, "actions": legal},
    })


# -- draft -----------------------------------------------------------------------------------------


def draft_state(session_id: str, *, me: str, them: str, pool: list[dict], rosters: dict[str, list[dict]], picks: list[dict],
                first_drafter: str, current_seat: str, version: int) -> GameState:
    taken_items = {data.to_id(c.get("item")) for c in rosters.get(current_seat, [])}
    available = [c for c in pool if data.to_id(c.get("item")) not in taken_items]  # Item Clause
    obs = {
        "phase": "draft", "draft_complete": False, "battle_format": "gen9vgc2025regi",
        "current_seat": current_seat, "pick_number": len(picks) + 1, "picks_remaining": 12 - len(picks),
        "first_drafter": first_drafter,
        "available_card_ids": [c["card_id"] for c in available], "available_cards": available,
        "rosters": {k: [{"card_id": c["card_id"], "species": c["species"]} for c in v] for k, v in rosters.items()},
        "picks": picks, "decision_timeout_seconds": 15, "battle_starting": False,
    }
    legal = [{"action_id": f"draft_pick:{c['card_id']}", "label": f"Draft {c['species']}",
              "input": {"session_id": session_id, "action_id": f"draft_pick:{c['card_id']}", "state_version": version,
                        "action": {"type": "draft_pick", "card_id": c["card_id"]}}} for c in available] if current_seat == me else []
    return _state(session_id, "draft", obs, legal, version, my_turn=current_seat == me)


# -- team preview -----------------------------------------------------------------------------------


def preview_state(session_id: str, *, my_cards: list[dict], opp_cards: list[dict], version: int) -> GameState:
    def entry(card: dict) -> dict:
        info = data.species_info(card["species"]) or {}
        return {"species": data.to_id(card["species"]), "name": card["species"],
                "types": [t.upper() for t in info.get("types", [])], "base_stats": info.get("base_stats")}
    obs = {"phase": "team_preview", "battle_format": "gen9vgc2025regi", "waiting_for_action": True,
           "your_roster": [entry(c) for c in my_cards], "opponent_roster": [entry(c) for c in opp_cards]}
    roster = [data.to_id(c["species"]) for c in my_cards]
    legal = [{"action_id": "select_lineup", "label": "Select lineup",
              "input": {"session_id": session_id, "action_id": "select_lineup", "state_version": version,
                        "action": {"type": "select_lineup", "roster": roster, "bring_count": 4, "lead_count": 2,
                                   "instructions": "Choose 4 of your 6 to bring (bring) and 2 of those 4 as leads (leads)."}}}]
    return _state(session_id, "team_preview", obs, legal, version)


# -- battle ------------------------------------------------------------------------------------------


def battle_state(session_id: str, snap: dict, side: str, version: int) -> GameState | None:
    """Our doubles-turn GameState from a bridge snapshot, or None when this side has no decision."""
    mine, theirs = snap["sides"][side], snap["sides"]["p2" if side == "p1" else "p1"]
    request = mine.get("request")
    if not request or request.get("wait") or request.get("teamPreview"):
        return None
    my_active = sorted([m for m in mine["pokemon"] if m["active"]], key=lambda m: m["position"])
    their_active = sorted([m for m in theirs["pokemon"] if m["active"]], key=lambda m: m["position"])
    by_pos = {m["position"]: m for m in my_active}
    their_by_pos = {m["position"] + 1: m for m in their_active if not m["fainted"]}
    force = request.get("forceSwitch") or [False, False]
    active_reqs = request.get("active") or []

    slots = []
    for slot in range(2):
        mon = by_pos.get(slot)
        options: list[dict] = []
        if force[slot]:
            options += _switch_options(mine, slot)
            if not options:
                options.append({"type": "pass"})
        elif any(force):
            options.append({"type": "pass"})  # the other slot is replacing a fainted Pokémon
        elif mon and not mon["fainted"] and slot < len(active_reqs) and active_reqs[slot]:
            req = active_reqs[slot]
            for move in req.get("moves") or []:
                if move.get("disabled"):
                    continue
                options.append(_move_option(move, slot, mon, by_pos, their_by_pos))
            if not req.get("trapped") and not mon.get("trapped"):
                options += _switch_options(mine, slot)
        else:
            options.append({"type": "pass"})
        slots.append({"slot": slot, "board_position": -(slot + 1), "active": mon["name"] if mon else None,
                      "force_switch": bool(force[slot]), "options": options})

    template = {"type": "doubles_turn", "slots": slots, "instructions": "Pick one option per slot.",
                "target_legend": {"1": "opponent position A", "2": "opponent position B", "-1": "your position A (slot 0)",
                                  "-2": "your position B (slot 1)", "0": "no target"}}
    team = {f"{side}: {m['name']}": _mon_summary(m, full=True) for m in mine["pokemon"]}
    opp = {f"{'p2' if side == 'p1' else 'p1'}: {m['name']}": _mon_summary(m, full=False) for m in theirs["pokemon"]}
    obs = {
        "phase": "moving", "is_doubles": True, "turn": snap["turn"], "format": "gen9vgc2025regi",
        "weather": snap["field"].get("weather") or None,
        "fields": [f for f in [snap["field"].get("terrain")] + list(snap["field"].get("pseudo_weather") or []) if f],
        "side_conditions": mine.get("side_conditions") or [], "opponent_side_conditions": theirs.get("side_conditions") or [],
        "active_pokemon": [_mon_summary(m, full=True) for m in my_active],
        "opponent_active_pokemon": [_mon_summary(m, full=False) for m in their_active],
        "force_switch": list(force), "team": team, "opponent_team": opp,
    }
    legal = [{"action_id": "doubles_turn", "label": "Doubles turn",
              "input": {"session_id": session_id, "action_id": "doubles_turn", "state_version": version, "action": template}}]
    return _state(session_id, "moving", obs, legal, version)


def _switch_options(side: dict, slot: int) -> list[dict]:
    return [{"type": "switch", "species": m["name"]} for m in side["pokemon"] if not m["active"] and not m["fainted"]]


def _move_option(move: dict, slot: int, mon: dict, by_pos: dict, their_by_pos: dict) -> dict:
    info = data.move_info(move["id"]) or {}
    target_type = move.get("target") or info.get("target")
    option = {"type": "move", "move_id": move["id"], "move_type": (info.get("type") or "").upper(),
              "category": (info.get("category") or "").upper(), "base_power": info.get("base_power", 0),
              "accuracy": info.get("accuracy"), "current_pp": move.get("pp"), "priority": info.get("priority", 0)}
    ally_slot = 1 - slot
    ally = by_pos.get(ally_slot)
    targets: list[int] = []
    target_options: list[dict] = []
    if target_type in SINGLE_TARGET:
        for pos, foe in sorted(their_by_pos.items()):
            targets.append(pos)
            target_options.append({"target": pos, "side": "opponent", "species": foe["name"]})
    elif target_type == "adjacentAlly":
        if ally and not ally["fainted"]:
            pos = -(ally_slot + 1)
            targets.append(pos)
            target_options.append({"target": pos, "side": "ally", "species": ally["name"]})
    elif target_type == "adjacentAllyOrSelf":
        targets.append(-(slot + 1))
        target_options.append({"target": -(slot + 1), "side": "self", "species": mon["name"]})
        if ally and not ally["fainted"]:
            targets.append(-(ally_slot + 1))
            target_options.append({"target": -(ally_slot + 1), "side": "ally", "species": ally["name"]})
    option["targets"] = targets
    option["target_options"] = target_options
    return option


# -- agent answer -> Showdown choice ------------------------------------------------------------------


def to_choice(payload: dict) -> str:
    """{"type":"doubles_turn","slot_0":{...},"slot_1":{...}} -> 'move fakeout 2, switch Amoonguss'."""
    parts = []
    for key in ("slot_0", "slot_1"):
        s = payload.get(key) or {"type": "pass"}
        if s.get("type") == "move":
            target = s.get("target")
            parts.append(f"move {s['move_id']}" + (f" {target}" if target not in (None, 0) else ""))
        elif s.get("type") == "switch":
            parts.append(f"switch {s['species']}")
        else:
            parts.append("pass")
    return ", ".join(parts)


def team_text(cards: list[dict]) -> str:
    """Showdown importable team text for a list of cards (in order)."""
    blocks = []
    for c in cards:
        evs = " / ".join(f"{v} {k.upper() if k != 'spe' else 'Spe'}" for k, v in (c.get("evs") or {}).items())
        evs = evs.replace("ATK", "Atk").replace("DEF", "Def").replace("SPA", "SpA").replace("SPD", "SpD").replace("HP", "HP")
        lines = [f"{c['species']} @ {c['item']}", f"Ability: {c['ability']}", "Level: 50"]
        if evs:
            lines.append(f"EVs: {evs}")
        lines.append(f"{c['nature']} Nature")
        lines += [f"- {m}" for m in c["moves"]]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def preview_choice(bring: list[str], leads: list[str], roster_cards: list[dict]) -> tuple[str, list[dict]]:
    """Order the 4 brought cards leads-first and return ('team 1234', ordered cards)."""
    ids = [data.to_id(c["species"]) for c in roster_cards]
    chosen = [data.to_id(b) for b in leads] + [data.to_id(b) for b in bring if data.to_id(b) not in {data.to_id(x) for x in leads}]
    ordered = [roster_cards[ids.index(s)] for s in chosen]
    return "team 1234", ordered


def legal_payload_from_choice(payload: Any) -> dict:
    return payload if isinstance(payload, dict) else {}
