"""Offline dry run: feed the agent a realistic Team Preview and one doubles turn, with the REAL
model chain, no game server needed. Costs a few cents. Prints what the model saw, what it answered,
which model answered, and the latency — the same things the decision log records.

    python scripts/dry_run.py            # Fable -> Sonnet chain from .env
    AGENT_MODEL=claude-sonnet-5-5 python scripts/dry_run.py

Use it to iterate on prompts and the turn sheet between live test matches.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from altruagent import DecisionContext  # noqa: E402
from altruagent.models import GameState  # noqa: E402

from agent.agent import PokemonAgent  # noqa: E402
from agent.llm.anthropic_provider import provider_from_env  # noqa: E402
from agent.pokemon import data  # noqa: E402
from agent.pokemon.memory import species_key  # noqa: E402

CONTEXT = DecisionContext(session_id="dry-run", tournament_id=None, game_type="pokemon_vgc_doubles_draft", agent_id="me", seat_position=0)


def card(species, moves, *, item="Leftovers", ability="Pressure", nature="Serious", evs=None):
    return {"card_id": f"vgc-{data.to_id(species)}", "species": species, "item": item, "ability": ability,
            "nature": nature, "evs": evs or {}, "moves": moves}


MINE = [
    card("Rillaboom", ["Wood Hammer", "Grassy Glide", "Fake Out", "U-turn"], ability="Grassy Surge", nature="Adamant", evs={"atk": 252, "hp": 252}, item="Assault Vest"),
    card("Incineroar", ["Flare Blitz", "Knock Off", "Fake Out", "Parting Shot"], ability="Intimidate", nature="Careful", item="Safety Goggles"),
    card("Garchomp", ["Earthquake", "Dragon Claw", "Rock Slide", "Protect"], ability="Rough Skin", nature="Jolly", evs={"spe": 252, "atk": 252}, item="Life Orb"),
    card("Flutter Mane", ["Moonblast", "Shadow Ball", "Dazzling Gleam", "Protect"], ability="Protosynthesis", nature="Timid", evs={"spa": 252, "spe": 252}, item="Choice Specs"),
    card("Amoonguss", ["Spore", "Rage Powder", "Pollen Puff", "Protect"], ability="Regenerator", nature="Calm", item="Rocky Helmet"),
    card("Whimsicott", ["Tailwind", "Moonblast", "Encore", "Protect"], ability="Prankster", nature="Timid", item="Focus Sash"),
]
THEIRS = [
    card("Gyarados", ["Waterfall", "Tera Blast", "Dragon Dance", "Protect"], ability="Intimidate", nature="Jolly", item="Sitrus Berry"),
    card("Urshifu-Rapid-Strike", ["Surging Strikes", "Close Combat", "Aqua Jet", "Detect"], ability="Unseen Fist", nature="Jolly", evs={"atk": 252, "spe": 252}, item="Choice Scarf"),
    card("Kingambit", ["Kowtow Cleave", "Sucker Punch", "Iron Head", "Protect"], ability="Defiant", nature="Adamant", item="Black Glasses"),
    card("Pelipper", ["Hurricane", "Weather Ball", "Tailwind", "Protect"], ability="Drizzle", nature="Modest", item="Covert Cloak"),
    card("Dragonite", ["Extreme Speed", "Scale Shot", "Fire Punch", "Protect"], ability="Multiscale", nature="Adamant", item="Loaded Dice"),
    card("Gholdengo", ["Make It Rain", "Shadow Ball", "Nasty Plot", "Protect"], ability="Good as Gold", nature="Modest", item="Choice Specs"),
]


def state(legal: list[dict], observation: dict, version: int) -> GameState:
    return GameState.from_mcp_state({
        "session_id": "dry-run", "status": "in_progress", "phase": "moving", "is_current_actor": True,
        "state_version": version, "observation": observation,
        "legal_actions": {"session_id": "dry-run", "state_version": version, "actions": legal},
    })


def templated(action_id: str, template: dict) -> dict:
    return {"action_id": action_id, "label": action_id, "input": {"action": template}}


def main() -> int:
    log_dir = Path("logs")
    agent = PokemonAgent(provider_from_env(log=print), log_dir=log_dir, capture_dir="", log=print)
    for c in MINE:
        agent.memory.my_cards[species_key(c["species"])] = c
    for c in THEIRS:
        agent.memory.opp_cards[species_key(c["species"])] = c

    roster = [data.to_id(c["species"]) for c in MINE]
    preview = {"phase": "team_preview", "battle_format": "gen9vgc2025regi",
               "your_roster": [{"species": data.to_id(c["species"]), "name": c["species"]} for c in MINE],
               "opponent_roster": [{"species": data.to_id(c["species"]), "name": c["species"]} for c in THEIRS]}
    lineup_action = templated("select_lineup", {"type": "select_lineup", "roster": roster, "bring_count": 4, "lead_count": 2,
                                                "instructions": "Choose 4 to bring and 2 leads."})
    started = time.monotonic()
    decision = agent.choose_action(state([lineup_action], preview, 10), CONTEXT)
    print(f"\n== TEAM PREVIEW ({int((time.monotonic() - started) * 1000)} ms)\n   {decision.action}\n   reasoning: {decision.reasoning_summary}\n")

    bring = decision.action["bring"]
    leads = decision.action["leads"]
    opp = [{"target": 1, "side": "opponent", "species": "Gyarados"}, {"target": 2, "side": "opponent", "species": "Urshifu-Rapid-Strike"}]

    def move(move_id, targets, options):
        return {"type": "move", "move_id": move_id, "targets": targets, "target_options": options}

    def options_for(species_id: str, ally: str) -> list[dict]:
        card_ = next(c for c in MINE if data.to_id(c["species"]) == species_id)
        out = []
        for m in card_["moves"]:
            info = data.move_info(m) or {}
            if info.get("target") in ("normal", "any", "adjacentFoe"):
                out.append(move(data.to_id(m), [1, 2], opp))
            elif info.get("target") == "adjacentAlly":
                out.append(move(data.to_id(m), [-1 if ally == bring[0] else -2], [{"target": -1, "side": "ally", "species": ally}]))
            else:
                out.append(move(data.to_id(m), [], []))
        for bench in bring[2:]:
            out.append({"type": "switch", "species": bench})
        return out

    template = {"type": "doubles_turn", "target_legend": {"1": "opponent A", "2": "opponent B", "-1": "your slot 0", "-2": "your slot 1", "0": "none"},
                "instructions": "Pick one option per slot.",
                "slots": [{"slot": 0, "board_position": -1, "active": leads[0], "force_switch": False, "options": options_for(leads[0], leads[1])},
                          {"slot": 1, "board_position": -2, "active": leads[1], "force_switch": False, "options": options_for(leads[1], leads[0])}]}
    battle_obs = {"phase": "moving", "turn": 1, "weather": None, "field": {},
                  "team": {f"p1: {s}": {"species": s, "active": s in leads, "current_hp_fraction": 1.0} for s in bring},
                  "opponent_team": {"p2: Gyarados": {"species": "Gyarados", "active": True, "current_hp_fraction": 1.0},
                                    "p2: Urshifu": {"species": "Urshifu-Rapid-Strike", "active": True, "current_hp_fraction": 1.0}}}
    started = time.monotonic()
    decision = agent.choose_action(state([templated("doubles_turn", template)], battle_obs, 11), CONTEXT)
    print(f"\n== TURN 1 ({int((time.monotonic() - started) * 1000)} ms)\n   {json.dumps(decision.action)}\n   reasoning: {decision.reasoning_summary}\n")

    lines = [json.loads(l) for l in (log_dir / "dry-run.jsonl").read_text().splitlines()][-2:]
    for line in lines:
        print(f"log: kind={line['kind']} model={line.get('model')} latency_ms={line.get('latency_ms')} attempts={line.get('attempts')} "
              f"fallback={line.get('fallback')} errors={line.get('errors')} usage={line.get('usage')}")
    print("memory:", agent.memory.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
