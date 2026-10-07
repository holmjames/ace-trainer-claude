"""Scenario eval: hand-built battle situations with a known good play, run through the REAL agent.

Each scenario builds a doubles-turn template and observation exactly the way the server shapes
them, hands it to ``PokemonAgent._turn`` (turn sheet + model + validation), and checks the model's
choice against an acceptance rule. Prints pass/fail, latency and which model answered, and appends
every run to logs/scenarios.jsonl so prompt changes can be compared over time.

    python scripts/scenarios.py                         # all scenarios, Fable chain from .env
    python scripts/scenarios.py --only protect_vs_ko --repeat 3
    AGENT_EFFORT=high python scripts/scenarios.py       # compare effort levels
    python scripts/scenarios.py --code-only             # what the computed fallback would do (no model)

Costs a few cents per scenario on Fable.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from altruagent import DecisionContext  # noqa: E402
from altruagent.models import GameState  # noqa: E402

from agent.agent import PokemonAgent  # noqa: E402
from agent.pokemon import data  # noqa: E402
from agent.pokemon.memory import species_key  # noqa: E402

CONTEXT = DecisionContext(session_id="scenario", tournament_id=None, game_type="pokemon_vgc_doubles_draft", agent_id="me", seat_position=0)


def card(species, moves, *, item="Leftovers", ability="Pressure", nature="Serious", evs=None):
    return {"card_id": f"vgc-{data.to_id(species)}", "species": species, "item": item, "ability": ability,
            "nature": nature, "level": 50, "evs": evs or {}, "moves": moves}


CARDS = {
    "rillaboom": card("Rillaboom", ["Wood Hammer", "Grassy Glide", "Fake Out", "U-turn"], ability="Grassy Surge", nature="Adamant", evs={"atk": 252, "hp": 252}, item="Assault Vest"),
    "incineroar": card("Incineroar", ["Flare Blitz", "Knock Off", "Fake Out", "Parting Shot"], ability="Intimidate", nature="Careful", evs={"hp": 252, "spd": 252}, item="Safety Goggles"),
    "garchomp": card("Garchomp", ["Earthquake", "Dragon Claw", "Rock Slide", "Protect"], ability="Rough Skin", nature="Jolly", evs={"spe": 252, "atk": 252}, item="Life Orb"),
    "fluttermane": card("Flutter Mane", ["Moonblast", "Shadow Ball", "Dazzling Gleam", "Protect"], ability="Protosynthesis", nature="Timid", evs={"spa": 252, "spe": 252}, item="Choice Specs"),
    "amoonguss": card("Amoonguss", ["Spore", "Rage Powder", "Pollen Puff", "Protect"], ability="Regenerator", nature="Calm", evs={"hp": 252, "spd": 252}, item="Rocky Helmet"),
    "whimsicott": card("Whimsicott", ["Tailwind", "Moonblast", "Encore", "Protect"], ability="Prankster", nature="Timid", evs={"spe": 252, "hp": 252}, item="Focus Sash"),
    "gyarados": card("Gyarados", ["Waterfall", "Tera Blast", "Dragon Dance", "Protect"], ability="Intimidate", nature="Jolly", evs={"atk": 252, "spe": 252}, item="Sitrus Berry"),
    "urshifurapidstrike": card("Urshifu-Rapid-Strike", ["Surging Strikes", "Close Combat", "Aqua Jet", "Detect"], ability="Unseen Fist", nature="Jolly", evs={"atk": 252, "spe": 252}, item="Choice Scarf"),
    "kingambit": card("Kingambit", ["Kowtow Cleave", "Sucker Punch", "Iron Head", "Protect"], ability="Defiant", nature="Adamant", evs={"atk": 252, "hp": 252}, item="Black Glasses"),
    "pelipper": card("Pelipper", ["Hurricane", "Weather Ball", "Tailwind", "Protect"], ability="Drizzle", nature="Modest", evs={"spa": 252, "hp": 252}, item="Covert Cloak"),
    "dragonite": card("Dragonite", ["Extreme Speed", "Scale Shot", "Fire Punch", "Protect"], ability="Multiscale", nature="Adamant", evs={"atk": 252, "hp": 252}, item="Loaded Dice"),
    "gholdengo": card("Gholdengo", ["Make It Rain", "Shadow Ball", "Nasty Plot", "Protect"], ability="Good as Gold", nature="Modest", evs={"spa": 252, "hp": 252}, item="Choice Specs"),
    "sneasler": card("Sneasler", ["Close Combat", "Dire Claw", "Fake Out", "Protect"], ability="Unburden", nature="Jolly", evs={"atk": 252, "spe": 252}, item="Focus Sash"),
    "zamazenta": card("Zamazenta", ["Body Press", "Heavy Slam", "Wide Guard", "Protect"], ability="Dauntless Shield", nature="Jolly", evs={"hp": 252, "spe": 252}, item="Rusted Shield"),
    "ironbundle": card("Iron Bundle", ["Freeze-Dry", "Hydro Pump", "Icy Wind", "Protect"], ability="Quark Drive", nature="Timid", evs={"spa": 252, "spe": 252}, item="Booster Energy"),
    "archaludon": card("Archaludon", ["Electro Shot", "Draco Meteor", "Flash Cannon", "Body Press"], ability="Stamina", nature="Modest", evs={"hp": 252, "spa": 252}, item="Assault Vest"),
    "basculegion": card("Basculegion", ["Wave Crash", "Last Respects", "Aqua Jet", "Protect"], ability="Swift Swim", nature="Adamant", evs={"atk": 252, "spe": 252}, item="Mystic Water"),
    "koraidon": card("Koraidon", ["Flare Blitz", "Collision Course", "Drain Punch", "Protect"], ability="Orichalcum Pulse", nature="Jolly", evs={"atk": 252, "spe": 252}, item="Clear Amulet"),
    "miraidon": card("Miraidon", ["Electro Drift", "Draco Meteor", "Dazzling Gleam", "Volt Switch"], ability="Hadron Engine", nature="Modest", evs={"hp": 252, "spa": 252}, item="Choice Specs"),
    "chienpao": card("Chien-Pao", ["Icicle Crash", "Sucker Punch", "Sacred Sword", "Protect"], ability="Sword of Ruin", nature="Jolly", evs={"atk": 252, "spe": 252}, item="Focus Sash"),
    "farigiraf": card("Farigiraf", ["Trick Room", "Psychic", "Foul Play", "Protect"], ability="Armor Tail", nature="Quiet", evs={"hp": 252, "spa": 252}, item="Sitrus Berry"),
    "hatterene": card("Hatterene", ["Trick Room", "Dazzling Gleam", "Expanding Force", "Protect"], ability="Magic Bounce", nature="Quiet", evs={"hp": 252, "spa": 252}, item="Life Orb"),
    "ironhands": card("Iron Hands", ["Fake Out", "Drain Punch", "Wild Charge", "Heavy Slam"], ability="Quark Drive", nature="Adamant", evs={"hp": 252, "atk": 252}, item="Assault Vest"),
    "maushold": card("Maushold", ["Population Bomb", "Follow Me", "Beat Up", "Protect"], ability="Technician", nature="Jolly", evs={"atk": 252, "spe": 252}, item="Wide Lens"),
    "kommoo": card("Kommo-o", ["Clanging Scales", "Aura Sphere", "Flamethrower", "Protect"], ability="Overcoat", nature="Timid", evs={"spa": 252, "spe": 252}, item="Throat Spray"),
    "chiyu": card("Chi-Yu", ["Heat Wave", "Dark Pulse", "Overheat", "Snarl"], ability="Beads of Ruin", nature="Timid", evs={"spa": 252, "spe": 252}, item="Choice Scarf"),
    "arcaninehisui": card("Arcanine-Hisui", ["Rock Slide", "Flare Blitz", "Extreme Speed", "Protect"], ability="Intimidate", nature="Jolly", evs={"atk": 252, "spe": 252}, item="Clear Amulet"),
    "urshifu": card("Urshifu", ["Wicked Blow", "Close Combat", "Sucker Punch", "Detect"], ability="Unseen Fist", nature="Jolly", evs={"atk": 252, "spe": 252}, item="Focus Sash"),
}
MINE_DEFAULT = ["rillaboom", "incineroar", "garchomp", "fluttermane", "amoonguss", "whimsicott"]
THEIRS_DEFAULT = ["gyarados", "urshifurapidstrike", "kingambit", "pelipper", "dragonite", "gholdengo"]


@dataclass
class Scenario:
    name: str
    description: str
    my_active: tuple[str, str]
    their_active: tuple[str, str]
    accept: Callable[[dict], bool]
    accept_text: str
    my_hp: dict[str, float] = field(default_factory=dict)
    their_hp: dict[str, float] = field(default_factory=dict)
    my_bench: tuple[str, ...] = ()
    turn: int = 2
    weather: str | None = None
    fields: list[str] = field(default_factory=list)
    my_side: list[str] = field(default_factory=list)
    their_side: list[str] = field(default_factory=list)
    drop_moves: dict[str, set[str]] = field(default_factory=dict)  # e.g. no Fake Out after turn 1
    mine: list[str] = field(default_factory=lambda: list(MINE_DEFAULT))
    theirs: list[str] = field(default_factory=lambda: list(THEIRS_DEFAULT))
    their_boosts: dict[str, dict] = field(default_factory=dict)  # species -> {"atk": 1}
    slot1_fainted: bool = False  # our second slot is empty (1v2 endgame): it can only pass
    hard: bool = False
    # Who just switched in this turn (Fake Out live). On turn 1 everyone is fresh; after that, nobody unless listed.
    my_fresh: tuple[str, ...] = ()
    their_fresh: tuple[str, ...] = ()


def move_of(payload: dict, slot: str) -> str | None:
    s = payload.get(slot) or {}
    return s.get("move_id") if s.get("type") == "move" else None


def slot_of(payload: dict, species_id: str, actives: tuple[str, str]) -> str:
    return "slot_0" if actives[0] == species_id else "slot_1"


SCENARIOS: list[Scenario] = [
    Scenario(
        "fake_out_turn1", "Turn 1 with Fake Out + Prankster Tailwind available against a Scarf Urshifu.",
        ("incineroar", "whimsicott"), ("gyarados", "urshifurapidstrike"), turn=1,
        accept=lambda p: move_of(p, "slot_0") == "fakeout" and move_of(p, "slot_1") in ("tailwind", "moonblast", "encore"),
        accept_text="Incineroar uses Fake Out; Whimsicott does something useful (Tailwind preferred).",
    ),
    Scenario(
        "protect_vs_ko", "Flutter Mane at 35% next to a Scarf Urshifu whose Surging Strikes KOs it; Rillaboom beside it.",
        ("fluttermane", "rillaboom"), ("urshifurapidstrike", "gyarados"), my_hp={"fluttermane": 0.35},
        drop_moves={"rillaboom": {"fakeout"}},
        accept=lambda p: (p["slot_0"].get("type") == "switch" or move_of(p, "slot_0") == "protect")
        and move_of(p, "slot_1") in ("woodhammer", "grassyglide"),
        accept_text="Flutter Mane Protects or switches; Rillaboom hits a Water type with Grass.",
    ),
    Scenario(
        "no_earthquake_into_ally", "Garchomp beside Flutter Mane (not Ground-immune); Earthquake would hit the ally.",
        ("garchomp", "fluttermane"), ("kingambit", "gholdengo"), drop_moves={},
        accept=lambda p: move_of(p, "slot_0") != "earthquake" or move_of(p, "slot_1") == "protect",
        accept_text="No Earthquake unless Flutter Mane Protects.",
    ),
    Scenario(
        "finish_low_hp_target", "Gyarados at 15% beside a healthy Urshifu; Rillaboom + Garchomp on our side.",
        ("rillaboom", "garchomp"), ("gyarados", "urshifurapidstrike"), their_hp={"gyarados": 0.15},
        drop_moves={"rillaboom": {"fakeout"}},
        accept=lambda p: any(
            (p[s].get("type") == "move" and (p[s].get("target") == 1 or move_of(p, s) == "rockslide")) for s in ("slot_0", "slot_1")
        ),
        accept_text="At least one attack reaches Gyarados (target 1 or spread Rock Slide).",
    ),
    Scenario(
        "switch_when_walled", "Incineroar in rain vs Gyarados + Pelipper (both Water); Rillaboom in the back. Turn 3, no Fake Out.",
        ("incineroar", "amoonguss"), ("gyarados", "pelipper"), turn=3, weather="RainDance", my_bench=("rillaboom", "garchomp"),
        drop_moves={"incineroar": {"fakeout"}},
        accept=lambda p: p["slot_0"].get("type") == "switch" or move_of(p, "slot_0") in ("partingshot", "knockoff"),
        accept_text="Incineroar pivots out (switch/Parting Shot) or at least uses Knock Off, not Flare Blitz into rain Waters.",
    ),
    Scenario(
        "spread_two_weak_targets", "Rock Slide hits both Gyarados and Pelipper super-effectively at 55%; Earthquake hits neither (Flying).",
        ("garchomp", "amoonguss"), ("gyarados", "pelipper"), their_hp={"gyarados": 0.55, "pelipper": 0.55},
        accept=lambda p: move_of(p, "slot_0") == "rockslide",
        accept_text="Garchomp uses Rock Slide.",
    ),
    Scenario(
        "trick_room_speed", "Trick Room is up: our slow Amoonguss + Incineroar vs Flutter Mane-speed Dragonite + Gholdengo.",
        ("amoonguss", "incineroar"), ("dragonite", "gholdengo"), fields=["Trick Room"], drop_moves={"incineroar": {"fakeout"}},
        accept=lambda p: move_of(p, "slot_0") in ("spore", "ragepowder", "pollenpuff") and move_of(p, "slot_1") in ("knockoff", "flareblitz", "partingshot"),
        accept_text="Amoonguss uses Spore/Rage Powder/Pollen Puff; Incineroar attacks (Knock Off into Gholdengo is ideal).",
    ),
    Scenario(
        "sucker_punch_respect", "Our Tailwind is up, but Kingambit's priority Sucker Punch OHKOs Flutter Mane (155 vs 130 HP). Garchomp beside it.",
        ("fluttermane", "garchomp"), ("kingambit", "urshifurapidstrike"), my_side=["Tailwind"],
        accept=lambda p: (move_of(p, "slot_0") == "protect" or p["slot_0"].get("type") == "switch") and move_of(p, "slot_1") in ("earthquake", "dragonclaw", "rockslide"),
        accept_text="Flutter Mane Protects or switches out (Sucker Punch would KO it first, and fails against a switching target); Garchomp attacks.",
    ),
    Scenario(
        "tailwind_up_go_aggressive", "Our Tailwind is up; Flutter Mane + Garchomp vs Gyarados + Pelipper, nothing with priority threatens a KO.",
        ("fluttermane", "garchomp"), ("gyarados", "pelipper"), my_side=["Tailwind"],
        accept=lambda p: move_of(p, "slot_0") in ("moonblast", "dazzlinggleam", "shadowball") and move_of(p, "slot_1") in ("rockslide", "dragonclaw"),
        accept_text="Both attack: Flutter Mane fires off a Specs move, Garchomp Rock Slides the two Flying types (not Earthquake: both immune).",
    ),
]


SCENARIOS.append(Scenario(
    "stale_fake_out", "Turn 3: Hatterene + Incineroar have been out since turn 1 vs Iron Hands and a 6% Maushold. Fake Out is still listed but fails now; Hatterene Protected last turn.",
    ("hatterene", "incineroar"), ("ironhands", "maushold"), turn=3, their_hp={"maushold": 0.06}, their_boosts={"maushold": {"atk": -1}, "ironhands": {"atk": -1}},
    mine=["hatterene", "incineroar", "basculegion", "urshifurapidstrike", "archaludon", "urshifu"], theirs=["ironhands", "maushold", "farigiraf", "kommoo", "arcaninehisui", "chiyu"],
    accept=lambda p: move_of(p, "slot_1") != "fakeout" and move_of(p, "slot_0") != "protect",
    accept_text="No Fake Out from Incineroar (not its first turn out: it fails) and no second Protect in a row from Hatterene. Attack: Maushold at 6% dies to anything, Expanding Force / Dazzling Gleam pressure Iron Hands.",
))

HARD: list[Scenario] = [
    Scenario(
        "break_sash_then_ko", "Chien-Pao at full HP behind a Focus Sash beside Kingambit; our Garchomp + Flutter Mane. One hit can't KO it.",
        ("garchomp", "fluttermane"), ("chienpao", "kingambit"), hard=True,
        mine=["garchomp", "fluttermane", "incineroar", "rillaboom", "amoonguss", "whimsicott"], theirs=["chienpao", "kingambit", "urshifurapidstrike", "pelipper", "dragonite", "gholdengo"],
        accept=lambda p: (move_of(p, "slot_0") == "protect" or p["slot_0"].get("type") == "switch") and (move_of(p, "slot_1") in ("dazzlinggleam", "moonblast", "shadowball", "protect") or p["slot_1"].get("type") == "switch"),
        accept_text="Garchomp must Protect or pivot to Incineroar (Chien-Pao speed-ties/outspeeds it and Icicle Crash is 4x). Flutter Mane either breaks the sash with Dazzling Gleam, or, since Sword of Ruin makes Kingambit's Sucker Punch and Chien-Pao's Icicle Crash both lethal to it, Protects or pivots out. Attacking with Garchomp just loses it.",
    ),
    Scenario(
        "intimidate_the_dancer", "Gyarados sits at +1 Attack after Dragon Dance beside Pelipper; our Amoonguss + Whimsicott, Incineroar (Intimidate) on the bench.",
        ("amoonguss", "whimsicott"), ("gyarados", "pelipper"), their_boosts={"gyarados": {"atk": 1, "spe": 1}}, my_bench=("incineroar", "garchomp"), hard=True,
        accept=lambda p: (move_of(p, "slot_0") == "spore" and p["slot_0"].get("target") == 1) or (p["slot_1"].get("type") == "switch" and p["slot_1"].get("species", "").lower().startswith("incineroar")),
        accept_text="Shut the boosted Gyarados down: Spore it, and/or bring Incineroar in for Intimidate.",
    ),
    Scenario(
        "redirect_to_free_specs_attacker", "Specs Flutter Mane (frail) beside Amoonguss vs Scarf Urshifu-RS (Surging Strikes KOs it) and Kingambit (Sucker Punch KOs it). Rage Powder soaks both single-target hits; Amoonguss resists Water and Sucker Punch fails on a non-attacker.",
        ("amoonguss", "fluttermane"), ("urshifurapidstrike", "kingambit"), hard=True, turn=2,
        accept=lambda p: move_of(p, "slot_0") == "ragepowder" and move_of(p, "slot_1") != "protect",
        accept_text="Amoonguss Rage Powder redirects both lethal single-target attacks; Flutter Mane then attacks freely (ideal) or pivots.",
    ),
    Scenario(
        "deny_trick_room", "Farigiraf (Armor Tail: no Fake Out) wants Trick Room beside Hatterene; our Incineroar + Garchomp outspeed it this turn only.",
        ("incineroar", "garchomp"), ("farigiraf", "hatterene"), hard=True,
        mine=["incineroar", "garchomp", "fluttermane", "rillaboom", "amoonguss", "whimsicott"], theirs=["farigiraf", "hatterene", "urshifurapidstrike", "pelipper", "dragonite", "gholdengo"],
        accept=lambda p: move_of(p, "slot_0") not in ("fakeout", None) and p["slot_0"].get("target") == 1
        and (p["slot_1"].get("type") == "switch" or move_of(p, "slot_1") == "protect" or (p["slot_1"].get("type") == "move" and p["slot_1"].get("target") == 1)),
        accept_text="Incineroar attacks Farigiraf (no Fake Out: Armor Tail). Garchomp either joins in to deny Trick Room, or Protects/switches from Hatterene's 96-113% Dazzling Gleam. Both are defensible.",
    ),
    Scenario(
        "endgame_immunities", "1v1 endgame: our Specs Flutter Mane alone vs Dragonite (Extreme Speed, Scale Shot, Fire Punch). Both immune moves are irrelevant.",
        ("fluttermane", "garchomp"), ("dragonite", "gholdengo"), slot1_fainted=True, their_hp={"gholdengo": 0.0}, hard=True,
        accept=lambda p: move_of(p, "slot_0") in ("moonblast", "dazzlinggleam") and p["slot_1"].get("type") == "pass",
        accept_text="Attack with the Fairy move: Dragonite's Extreme Speed (Normal) and Scale Shot (Dragon) cannot touch a Ghost/Fairy; Protect gains nothing.",
    ),
    Scenario(
        "sash_does_not_stop_multihit", "Our Sneasler (Focus Sash, full HP) and Zamazenta vs Scarf Urshifu-RS (Surging Strikes = 3 hits) and Iron Bundle. Urshifu outspeeds and KOs Sneasler through the sash.",
        ("sneasler", "zamazenta"), ("urshifurapidstrike", "ironbundle"), hard=True, turn=2, drop_moves={"sneasler": {"fakeout"}},
        mine=["sneasler", "zamazenta", "kingambit", "archaludon", "garchomp", "incineroar"], theirs=["urshifurapidstrike", "ironbundle", "garchomp", "gholdengo", "pelipper", "dragonite"],
        accept=lambda p: p["slot_0"].get("type") == "switch",
        accept_text="Sneasler must switch out: Surging Strikes breaks the sash, KOs it before it moves, and Unseen Fist goes through Protect.",
    ),
    Scenario(
        "trust_the_boosted_numbers", "Miraidon at -2 Special Attack still KOs our Basculegion (the ranges already include the drop). Koraidon beside it.",
        ("basculegion", "koraidon"), ("miraidon", "kingambit"), hard=True, turn=3, their_boosts={"miraidon": {"spa": -2}},
        mine=["basculegion", "koraidon", "garchomp", "fluttermane", "amoonguss", "whimsicott"], theirs=["miraidon", "kingambit", "urshifurapidstrike", "pelipper", "dragonite", "gholdengo"],
        accept=lambda p: move_of(p, "slot_0") == "protect" or p["slot_0"].get("type") == "switch",
        accept_text="Basculegion Protects or switches; the LETHAL warning already accounts for Miraidon's -2.",
    ),
    Scenario(
        "spread_breaks_sash_partner_finishes", "Whimsicott (sash, full HP) beside Gyarados at 30%; our Garchomp (faster than Incineroar) + Incineroar.",
        ("garchomp", "incineroar"), ("whimsicott", "gyarados"), their_hp={"gyarados": 0.30}, drop_moves={"incineroar": {"fakeout"}}, hard=True,
        mine=["garchomp", "incineroar", "fluttermane", "rillaboom", "amoonguss", "whimsicott"], theirs=["whimsicott", "gyarados", "urshifurapidstrike", "pelipper", "dragonite", "gholdengo"],
        accept=lambda p: move_of(p, "slot_0") == "rockslide" and move_of(p, "slot_1") in ("knockoff", "flareblitz") and p["slot_1"].get("target") == 1,
        accept_text="Rock Slide KOs the 30% Gyarados and breaks Whimsicott's sash; Incineroar's single hit then finishes Whimsicott.",
    ),
]


def build_template(sc: Scenario, mine_bring: list[str]) -> dict:
    opp = [{"target": i + 1, "side": "opponent", "species": CARDS[s]["species"]} for i, s in enumerate(sc.their_active)]

    def options_for(slot: int, species_id: str) -> list[dict]:
        ally = sc.my_active[1 - slot]
        out = []
        for m in CARDS[species_id]["moves"]:
            mid = data.to_id(m)
            if mid in sc.drop_moves.get(species_id, set()):
                continue
            info = data.move_info(m) or {}
            target = info.get("target")
            if target in ("normal", "any", "adjacentFoe", "randomNormal"):
                out.append({"type": "move", "move_id": mid, "targets": [1, 2], "target_options": opp})
            elif target == "adjacentAlly":
                pos = -2 if slot == 0 else -1
                out.append({"type": "move", "move_id": mid, "targets": [pos], "target_options": [{"target": pos, "side": "ally", "species": CARDS[ally]["species"]}]})
            elif target == "adjacentAllyOrSelf":
                out.append({"type": "move", "move_id": mid, "targets": [-1, -2], "target_options": [
                    {"target": -1 - slot, "side": "self", "species": CARDS[species_id]["species"]},
                    {"target": -2 + slot, "side": "ally", "species": CARDS[ally]["species"]}]})
            else:
                out.append({"type": "move", "move_id": mid, "targets": [], "target_options": []})
        for bench in sc.my_bench or [s for s in mine_bring if s not in sc.my_active]:
            out.append({"type": "switch", "species": bench})
        return out

    slots = []
    for i, s in enumerate(sc.my_active):
        if i == 1 and sc.slot1_fainted:
            slots.append({"slot": 1, "board_position": -2, "active": None, "force_switch": False, "options": [{"type": "pass"}]})
        else:
            slots.append({"slot": i, "board_position": -(i + 1), "active": CARDS[s]["species"], "force_switch": False, "options": options_for(i, s)})
    return {"type": "doubles_turn", "instructions": "Pick one option per slot.",
            "target_legend": {"1": "opponent A", "2": "opponent B", "-1": "your slot 0", "-2": "your slot 1", "0": "none"},
            "slots": slots}


def build_observation(sc: Scenario, mine_bring: list[str]) -> dict:
    def summary(species_id: str, active: bool, hp: float, boosts: dict | None = None) -> dict:
        c = CARDS[species_id]
        return {"species": c["species"], "active": active, "fainted": hp <= 0, "current_hp_fraction": hp,
                "status": None, "boosts": boosts or {}, "item": c["item"], "ability": c["ability"]}
    team = {}
    for s in mine_bring:
        hp = 0.0 if (sc.slot1_fainted and s == sc.my_active[1]) else sc.my_hp.get(s, 1.0)
        team[f"p1: {CARDS[s]['species']}"] = summary(s, s in sc.my_active and hp > 0, hp)
    opp = {f"p2: {CARDS[s]['species']}": summary(s, s in sc.their_active, sc.their_hp.get(s, 1.0), sc.their_boosts.get(s)) for s in sc.their_active}
    return {"phase": "moving", "is_doubles": True, "turn": sc.turn, "weather": sc.weather, "fields": sc.fields,
            "side_conditions": sc.my_side, "opponent_side_conditions": sc.their_side,
            "active_pokemon": [team[f"p1: {CARDS[s]['species']}"] for s in sc.my_active if team[f"p1: {CARDS[s]['species']}"]["active"]],
            "opponent_active_pokemon": [opp[f"p2: {CARDS[s]['species']}"] for s in sc.their_active],
            "force_switch": [False, False], "team": team, "opponent_team": opp}


def run(sc: Scenario, *, code_only: bool, log_dir: Path) -> dict:
    provider = None
    if not code_only:
        from agent.llm.anthropic_provider import provider_from_env

        provider = provider_from_env(log=lambda line: None)
    agent = PokemonAgent(provider, version=f"scenario:{sc.name}", log_dir=log_dir, capture_dir="", log=lambda line: None)
    for s in sc.mine:
        agent.memory.my_cards[species_key(CARDS[s]["species"])] = CARDS[s]
    for s in sc.theirs:
        agent.memory.opp_cards[species_key(CARDS[s]["species"])] = CARDS[s]
    mine_bring = list(sc.my_active) + [b for b in (sc.my_bench or ()) if b not in sc.my_active]
    if len(mine_bring) < 4:
        mine_bring += [s for s in sc.mine if s not in mine_bring][: 4 - len(mine_bring)]
    if sc.turn and sc.turn > 1:
        agent.memory.last_active = [species_key(CARDS[s]["species"]) for s in sc.my_active if s not in sc.my_fresh]
        agent.memory.opp_last_active = [species_key(CARDS[s]["species"]) for s in sc.their_active if s not in sc.their_fresh]
    template = build_template(sc, mine_bring)
    obs = build_observation(sc, mine_bring)
    state = GameState.from_mcp_state({
        "session_id": "scenario", "status": "in_progress", "phase": "moving", "is_current_actor": True, "state_version": 1,
        "observation": obs, "legal_actions": {"actions": [{"action_id": "doubles_turn", "label": "doubles_turn", "input": {"action": template}}]},
    })
    started = time.monotonic()
    decision = agent.choose_action(state, CONTEXT)
    elapsed = int((time.monotonic() - started) * 1000)
    payload = decision.action
    ok = bool(sc.accept(payload))
    last = agent.memory.turns[-1] if agent.memory.turns else {}
    return {"scenario": sc.name, "pass": ok, "payload": payload, "reasoning": decision.reasoning_summary, "ms": elapsed,
            "model": last.get("model"), "fallbacks": agent.memory.fallbacks}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", help="run one scenario by name")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--code-only", action="store_true", help="no model: what the computed fallback does")
    parser.add_argument("--set", choices=["basic", "hard", "all"], default="basic")
    args = parser.parse_args(argv)
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    pool = {"basic": SCENARIOS, "hard": HARD, "all": SCENARIOS + HARD}[args.set]
    chosen = [s for s in pool if not args.only or s.name == args.only]
    if not chosen:
        print("no such scenario; options:", [s.name for s in pool])
        return 2
    results = []
    for sc in chosen:
        for _ in range(args.repeat):
            r = run(sc, code_only=args.code_only, log_dir=log_dir)
            results.append(r)
            mark = "PASS" if r["pass"] else "FAIL"
            print(f"{mark} {sc.name:28s} {r['ms']:6d} ms  model={r['model']}  {json.dumps(r['payload'])}")
            print(f"     reasoning: {r['reasoning']}")
            if not r["pass"]:
                print(f"     expected: {sc.accept_text}")
            with (log_dir / "scenarios.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"ts": time.time(), "code_only": args.code_only, **r}, default=str) + "\n")
    passed = sum(1 for r in results if r["pass"])
    print(f"\n{passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
