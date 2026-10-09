"""Workstream A's gate: the decisive live positions replayed code-only with the reply mixture as code #1
(reply_mixture=1), asserting the played turn is the documented best play.

Positions and accept functions come from scripts/live_scenarios.py (the eval lane's set: the four Oct 7 losses L1-L4,
the Oct 8 practice win vs the search bot and the Oct 8 loss to it), replayed through one code-only agent the way
tests/test_live_semantics.py does. A ``provisional`` scenario's best play was derived from the transcript and the calc,
not documented: xfail(strict=False) until the orchestrator confirms it. A position where the sheet's own numbers
contradict the documented line is an xfail with the numbers in its reason (the "documented exceptions" of the brief).
"""

from __future__ import annotations

import importlib.util
import json
import re
import tempfile
from pathlib import Path

import pytest

from altruagent import DecisionContext
from altruagent.models import GameState

from agent.agent import PokemonAgent
from agent.pokemon import tuning

ROOT = Path(__file__).resolve().parent.parent
LIVE = ROOT / "tests" / "fixtures" / "live"
PHASES = {"draft": 0, "team_preview": 1, "moving": 2}


def _scenarios():
    import sys

    spec = importlib.util.spec_from_file_location("live_scenarios", ROOT / "scripts" / "live_scenarios.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["live_scenarios"] = module  # the dataclass decorator looks the module up by name
    spec.loader.exec_module(module)
    return module


LS = _scenarios()
BATTLE = [s for s in LS.SCENARIOS if s.turn > 0]

# Positions where the sheet's own damage numbers contradict the documented line (see each reason): the mixture's
# answer is kept and the scenario is an expected failure until the orchestrator rules on it.
EXCEPTIONS = {
    "m1_t6_dire_claw_ko": "Dire Claw into the 63% Dragonite is 86.6-101.8% of its HP, a 12% KO by the sheet (the review calls it a KO); "
                          "Fake Out + Cresselia Protect values higher against every plausible reply",
    "m2_t3_not_into_the_protecting_sylveon": "code #1 is Earthquake (KOs the 13% Incineroar) + Flash Cannon into Sylveon: the Earthquake "
                                             "line the lesson asks for, with the free Flash Cannon on the slot that may Protect (worst case 40.9 vs 51.7)",
    "m2_t4_protect_the_doubled_archaludon": "both our attacks KO the 75% Sylveon before it moves (125-151% combined; it is slower than both "
                                            "under our Tailwind), so the Moonblast + Hyper Voice double cannot happen: focus Sylveon (124.9) beats Protect",
    "m4_t2_no_telegraphed_ko": "the Landorus pivot values -54 against the plausible replies (the double into Landorus, 80% KO) vs -32 for "
                               "Baxcalibur Protect + Bleakwind; it wins only under the reply the bot made (switch to Incineroar, prior 0.03)",
    "m4_t4_out_of_the_double": "Lucario is KO'd on entry by Archaludon's Body Press (183-216%): the documented 'resists both Dragon hits' "
                               "switch is the worst row by the sheet's own numbers",
    "m4_t7_lucario_protects": "Close Combat KOs the 58% Incineroar before it moves (Lucario outspeeds it), removing the Flare Blitz that KOs "
                              "Tornadus; the -2 Def Scale Shot is priced (post-move rows) and the trade still values above Lucario Protect",
    "o8_t14_no_fake_priority_escape": "in a code-only replay Arcanine (39%) Protected on T13, so its T14 Protect works 1 in 3; Extreme Speed + Body "
                                      "Press put Dragonite at 4-19% before Arcanine is lost and Archaludon finishes next turn either way (the "
                                      "live turn's second Protect happened to work); the sheet no longer calls the Extreme Speed an escape",
}


def _sequence(folder: Path, seat: int) -> list[Path]:
    def order(p: Path) -> tuple[int, int]:
        m = re.match(r"(draft|team_preview|moving)-(\d+)-seat", p.name)
        return (PHASES[m.group(1)], int(m.group(2))) if m else (9, 0)
    return sorted(folder.glob(f"*-seat{seat}.json"), key=order)


_CACHE: dict[str, list[dict]] = {}


def replay_mix(prefix: str) -> list[dict]:
    """Every decision of a session through one code-only agent with the reply mixture on; the log rows."""
    if prefix in _CACHE:
        return _CACHE[prefix]
    folder = next(p for p in LIVE.iterdir() if p.is_dir() and p.name.startswith(prefix))
    seat = LS.seat_of(folder)
    with tempfile.TemporaryDirectory() as tmp:
        agent = PokemonAgent(None, version="replies-live", log_dir=tmp, capture_dir="", log=lambda *_: None,
                             params=tuning.merged({"reply_mixture": 1, "reply_seed": 7}))
        for path in _sequence(folder, seat):
            raw = json.loads(path.read_text())
            state = GameState.from_mcp_state(raw)
            if not state.legal_actions:
                continue
            ctx = DecisionContext(session_id=folder.name, tournament_id=None, game_type="pokemon_vgc_doubles_draft",
                                  agent_id=(raw.get("current_actor") or {}).get("agent_id") or "me", seat_position=seat)
            decision = agent.choose_action(state, ctx)
            assert not str(getattr(decision, "reasoning_summary", "")).startswith("Fallback:"), path.name
            template = state.legal_actions[0].input.get("action") or {}
            _CACHE.setdefault(prefix + ":templates", []).append((state.phase, (raw.get("observation") or {}).get("turn"), template))
        rows = [json.loads(line) for f in Path(tmp).glob("*.jsonl") for line in f.read_text().splitlines() if line.strip()]
    assert not [r for r in rows if r.get("kind") in ("error", "fallback")], "decision logic raised during replay"
    _CACHE[prefix] = rows
    return rows


def decision(prefix: str, turn: int, index: int = 0) -> tuple[dict, dict, dict]:
    """(played payload, template, log row) of the turn's decision number ``index``."""
    rows = replay_mix(prefix)
    turns = [r for r in rows if r.get("kind") == "turn" and r.get("turn") == turn]
    assert len(turns) > index, f"no decision #{index} on turn {turn}"
    templates = [t for phase, tn, t in _CACHE[prefix + ":templates"] if phase == "moving" and tn == turn]
    return turns[index]["payload"], templates[index], turns[index]


def _params(sc):
    marks = []
    if sc.provisional:
        marks.append(pytest.mark.xfail(strict=False, reason="best play derived from the transcript and the calc, awaiting confirmation"))
    if sc.name in EXCEPTIONS:
        marks.append(pytest.mark.xfail(strict=False, reason=EXCEPTIONS[sc.name]))
    return pytest.param(sc, id=sc.name, marks=marks)


@pytest.mark.parametrize("sc", [_params(s) for s in BATTLE])
def test_reply_mixture_plays_the_documented_line(sc):
    payload, template, row = decision(sc.session, sc.turn, sc.index)
    top = (row["turn_sheet"].get("candidate_turns") or [{}])[0]
    assert sc.accept(payload, template), f"{sc.name}: want {sc.accept_text}; played {json.dumps({k: v for k, v in payload.items() if k != 'type'})}; " \
                                         f"top row {top.get('name')} value {top.get('value')} worst {top.get('worst')} ({top.get('what')})"


def test_reply_table_is_logged_with_its_seed():
    rows = replay_mix("bcbaabe2")
    turn = next(r for r in rows if r.get("kind") == "turn" and r.get("turn") == 2)
    table = turn["turn_sheet"].get("reply_table") or {}
    assert table.get("seed") is not None and table.get("columns") and table.get("rows"), table
    assert all("prior" in c for c in table["columns"])
    assert all(k in table["rows"][0] for k in ("name", "value", "worst", "best_reply"))


def test_extreme_speed_then_draco_meteor_into_dragonite_is_generated(tmp_path):
    # O8 T2: Extreme Speed breaks Multiscale, then Draco Meteor at full power into Dragonite (the practice game's
    # post-mortem): the row must exist and rank first or second.
    _, _, row = decision("bcbaabe2", 2)
    cands = row["turn_sheet"]["candidate_turns"]
    line = [i for i, c in enumerate(cands) if c["slot_0"] == {"option": 1, "target": 1} and c["slot_1"] == {"option": 1, "target": 1}]
    assert line and line[0] <= 1, [(c["name"], c["slot_0"], c["slot_1"], c.get("value")) for c in cands]


def test_the_practice_game_turn_six_keeps_foul_play_on_the_other_slot():
    # O8 T6 (no regression): Foul Play on Dragonite while the partner attacks stays first; both attacks into the fresh
    # Protect of the 28% Ting-Lu (what the model did) must not be.
    payload, template, row = decision("bcbaabe2", 6)
    assert LS.move_of(payload, 1) == "foulplay" and LS.target_species(payload, 1, template) == "dragonite", payload
