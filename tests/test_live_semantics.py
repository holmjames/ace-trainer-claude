"""Replay whole live matches (draft -> Team Preview -> every battle turn) through ONE code-only agent and check what it
KNEW at each decision, not just that it answered.

Why this exists: on Oct 7 we lost 0-4 to a real opponent while every live fixture test passed. Each fixture was replayed
on a fresh agent and only checked for "no crash, a candidate exists". Meanwhile, live, the agent was blind: Team Preview
wiped the opponent's moves and items, their first pick was never known, our spread moves had no damage numbers, and the
model was shown two Pokémon we had left at Team Preview as if they could switch in. These tests pin the facts down."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from altruagent import DecisionContext
from altruagent.models import GameState

from agent.agent import PokemonAgent
from agent.pokemon.memory import species_key

LIVE = Path(__file__).parent / "fixtures" / "live"
PHASES = {"draft": 0, "team_preview": 1, "moving": 2}

# The four Oct 7 Testing losses vs LeCharmander (we are seat 0 in each).
M1, M2, M3, M4 = "d9405f02", "1aef9227", "7ce71025", "0a2b7c2a"


def _folder(prefix: str) -> Path:
    hits = [p for p in LIVE.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if len(hits) != 1:
        pytest.skip(f"live capture {prefix} not present")
    return hits[0]


def _sequence(folder: Path, seat: int = 0) -> list[Path]:
    files = [p for p in folder.glob(f"*-seat{seat}.json")]

    def order(p: Path) -> tuple[int, int]:
        m = re.match(r"(draft|team_preview|moving)-(\d+)-seat", p.name)
        return (PHASES[m.group(1)], int(m.group(2))) if m else (9, 0)

    return sorted(files, key=order)


def replay(prefix: str, tmp_path: Path, seat: int = 0) -> tuple[PokemonAgent, list[dict]]:
    """Feed every captured state of one seat, in order, to a single code-only agent. Returns it and its log rows."""
    folder = _folder(prefix)
    agent = PokemonAgent(None, version="live-replay", log_dir=tmp_path, capture_dir="", log=lambda *_: None)
    for path in _sequence(folder, seat):
        raw = json.loads(path.read_text())
        state = GameState.from_mcp_state(raw)
        if not state.legal_actions:
            continue
        agent_id = (raw.get("current_actor") or {}).get("agent_id") or "me"
        ctx = DecisionContext(session_id=folder.name, tournament_id=None, game_type="pokemon_vgc_doubles_draft",
                              agent_id=agent_id, seat_position=seat)
        decision = agent.choose_action(state, ctx)
        assert not str(getattr(decision, "reasoning_summary", "")).startswith("Fallback:"), path.name
    rows = [json.loads(line) for f in tmp_path.glob("*.jsonl") for line in f.read_text().splitlines() if line.strip()]
    assert not [r for r in rows if r.get("kind") in ("error", "fallback")], "decision logic raised during replay"
    return agent, rows


def turn_row(rows: list[dict], turn: int, index: int = 0) -> dict:
    found = [r for r in rows if r.get("kind") == "turn" and r.get("turn") == turn]
    assert len(found) > index, f"no decision #{index} on turn {turn}"
    return found[index]


def threat(row: dict, species: str) -> dict:
    for t in row["turn_sheet"]["opponent_threats"]:
        if species_key(t["species"]) == species:
            return t
    raise AssertionError(f"{species} is not among the threats: {[t['species'] for t in row['turn_sheet']['opponent_threats']]}")


def option(row: dict, slot: int, move: str) -> dict:
    for entry in row["turn_sheet"]["our_damage_estimates"]:
        if entry["slot"] == slot:
            for o in entry["options"]:
                if o["move"] == move:
                    return o
    raise AssertionError(f"slot {slot} has no {move} option")


@pytest.mark.parametrize("prefix", [M1, M2, M3, M4])
def test_team_preview_keeps_every_opponent_set_complete(prefix, tmp_path):
    agent, _ = replay(prefix, tmp_path)
    assert len(agent.memory.opp_cards) == 6
    for key, card in agent.memory.opp_cards.items():
        assert len(card.get("moves") or []) == 4, f"{key} lost its moves: {card.get('moves')}"
        assert card.get("item") and "unknown" not in str(card.get("item")).lower(), f"{key} item is {card.get('item')!r}"
        assert card.get("evs"), f"{key} has no EVs (a blank first pick?)"


def test_the_opponents_first_pick_is_known_from_the_catalog(tmp_path):
    # M2: they drafted Incineroar first, so it was never offered to us. Its real set (244 HP / 252 SpD Careful) means
    # Iron Hands' -1 Close Combat does ~86-101%, not the 103-122% "guaranteed KO" the blank 0-EV set gave on Oct 7.
    agent, rows = replay(M2, tmp_path)
    assert agent.memory.opp_cards["incineroar"].get("nature") == "Careful"
    cc = option(turn_row(rows, 2), 0, "closecombat")
    into_incin = next(t for t in cc["targets"] if species_key(t["species"]) == "incineroar")
    assert into_incin["damage_pct_of_current_hp"][0] < 100, into_incin


def test_unrevealed_opponent_moves_are_threats(tmp_path):
    # M3 turn 1: Iron Hands had not used Wild Charge yet; it OHKO'd our Tornadus with no warning.
    _, rows = replay(M3, tmp_path)
    hands = threat(turn_row(rows, 1), "ironhands")
    assert {"wildcharge", "closecombat", "fakeout"} <= set(hands["known_moves"])
    wc = [h for h in hands["hits"] if h["move"] == "wildcharge" and species_key(h["species"]) == "tornadus"]
    assert wc and wc[0]["ko"] in ("guaranteed", "possible"), wc
    # M3 turn 3: Incineroar had just switched in; its Fake Out (and Knock Off / Flare Blitz) were invisible.
    incin = threat(turn_row(rows, 3), "incineroar")
    assert {"fakeout", "knockoff", "flareblitz"} <= set(incin["known_moves"])
    assert "FRESH" in (incin.get("note") or "")


def test_make_it_rain_is_known_before_it_is_used(tmp_path):
    # M4 turn 2: Gholdengo had only shown Shadow Ball; Make It Rain then KO'd Baxcalibur.
    _, rows = replay(M4, tmp_path)
    gholdengo = threat(turn_row(rows, 2), "gholdengo")
    mir = [h for h in gholdengo["hits"] if h["move"] == "makeitrain" and species_key(h["species"]) == "baxcalibur"]
    assert mir and mir[0]["ko"] in ("guaranteed", "possible"), mir


def test_spread_moves_get_damage_numbers(tmp_path):
    # M3 turn 6: Hatterene alone vs Incineroar. Dazzling Gleam (STAB, neutral) was invisible; resisted Mystical Fire won.
    _, rows = replay(M3, tmp_path)
    row = turn_row(rows, 6)
    gleam = option(row, 1, "dazzlinggleam")
    fire = option(row, 1, "mysticalfire")
    assert gleam["targets"], "Dazzling Gleam has no damage rows"
    best_gleam = max(t["damage_pct_of_current_hp"][0] for t in gleam["targets"] if t["side"] == "theirs")
    best_fire = max(t["damage_pct_of_current_hp"][0] for t in fire["targets"] if t["side"] == "theirs")
    assert best_gleam > 2 * best_fire
    top = row["turn_sheet"]["candidate_turns"][0]
    assert top["slot_1"]["option"] == next(i for i, o in enumerate(_slot_options(rows, 6, 1)) if o.get("move_id") == "dazzlinggleam")


def _slot_options(rows: list[dict], turn: int, slot: int) -> list[dict]:
    # The candidate's option index refers to the server's option list; rebuild it from the logged estimates' indices.
    row = turn_row(rows, turn)
    entry = next(e for e in row["turn_sheet"]["our_damage_estimates"] if e["slot"] == slot)
    size = max([o["option"] for o in entry["options"]] + [0]) + 1
    out: list[dict] = [{} for _ in range(size)]
    for o in entry["options"]:
        out[o["option"]] = {"move_id": o["move"]}
    return out


def test_a_popped_air_balloon_lets_earthquake_hit(tmp_path):
    # M4: Bleakwind Storm popped Gholdengo's Air Balloon on turn 2; on turn 3 Landorus' Earthquake KO'd it.
    _, rows = replay(M4, tmp_path)
    eq = option(turn_row(rows, 3), 0, "earthquake")
    into = [t for t in eq["targets"] if species_key(t["species"]) == "gholdengo"]
    assert into and into[0]["damage_pct_of_current_hp"][1] > 0, eq


def test_the_model_is_told_who_is_not_in_this_battle(tmp_path):
    # M1: we brought Sneasler, Cresselia, Landorus, Garchomp. The live `team` lists all six (Baxcalibur even flagged
    # active); the model planned "Baxcalibur/Lucario can come in". The roster must say they are not here.
    _, rows = replay(M1, tmp_path)
    first = turn_row(rows, 1)["battle_roster"]["ours"]
    assert set(first["NOT_IN_THIS_BATTLE"]) == {"baxcalibur", "lucario"}
    assert {m["species"] for m in first["on_field"]} == {"sneasler", "cresselia"}
    assert {m["species"] for m in first["in_back_can_switch_in"]} == {"landorustherian", "garchomp"}
    late = turn_row(rows, 8)["battle_roster"]["ours"]
    assert late["remaining"] == 1 and late["in_back_can_switch_in"] == []


def test_protect_streak_survives_a_replacement_decision(tmp_path):
    # M1: Cresselia Protected on turn 4, Landorus fainted (a replacement decision followed), and on turn 5 a second
    # Protect was recommended as if fresh; it failed. The sheet must know it is the second in a row.
    _, rows = replay(M1, tmp_path)
    notes = " ".join(turn_row(rows, 5)["turn_sheet"]["notes"]).lower()
    assert "cresselia used protect last turn" in notes, notes


def test_no_helping_hand_without_an_ally(tmp_path):
    # M1 turns 7-10: Cresselia alone; the code's only candidate was Helping Hand into an empty slot.
    _, rows = replay(M1, tmp_path)
    for turn in (7, 8, 9, 10):
        for cand in turn_row(rows, turn)["turn_sheet"]["candidate_turns"]:
            assert "helpinghand" not in cand["why"], (turn, cand)
