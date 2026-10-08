"""Exact positions from real matches, replayed through the real agent, each with a known-good answer.

Hand-built scenarios (scripts/scenarios.py) can be transcribed wrong; these cannot. Each one replays a captured match
(tests/fixtures/live/<session>/) through one code-only agent up to the target decision, then asks for THAT decision
only — from the model chain in .env, or from the computed fallback with --code-only — and checks it.

    python scripts/live_scenarios.py --code-only          # free: what the code plays
    python scripts/live_scenarios.py                      # one model call per scenario (~$0.15 each on Opus)
    python scripts/live_scenarios.py --only m3_t6 --repeat 3

The positions come from the four Oct 7 losses to LeCharmander (see scripts/postmortem.py for the transcripts).
Results append to logs/live_scenarios.jsonl.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from altruagent import DecisionContext  # noqa: E402
from altruagent.models import GameState  # noqa: E402

from agent.agent import PokemonAgent  # noqa: E402
from agent.pokemon.memory import species_key  # noqa: E402

LIVE = ROOT / "tests" / "fixtures" / "live"
PHASES = {"draft": 0, "team_preview": 1, "moving": 2}


@dataclass
class LiveScenario:
    name: str
    session: str          # fixture folder prefix
    turn: int             # battle turn of the decision (the observation's turn number)
    what: str             # the position and the lesson, in one line
    accept: Callable[[dict, dict], bool]  # (payload, template) -> ok
    accept_text: str
    index: int = 0        # 0 = the turn's main decision; 1+ = a replacement after a faint


def move_of(payload: dict, slot: int) -> str | None:
    s = payload.get(f"slot_{slot}") or {}
    return s.get("move_id") if s.get("type") == "move" else None


def target_species(payload: dict, slot: int, template: dict) -> str | None:
    s = payload.get(f"slot_{slot}") or {}
    for t in _slot(template, slot).get("options") and [o for o in _slot(template, slot)["options"] if o.get("move_id") == s.get("move_id")] or []:
        for opt in t.get("target_options") or []:
            if opt.get("target") == s.get("target"):
                return species_key(opt.get("species"))
    return None


def _slot(template: dict, slot: int) -> dict:
    return next((x for x in template.get("slots") or [] if x.get("slot") == slot), {})


SCENARIOS = [
    LiveScenario("m3_t6_gleam", "7ce71025", 6,
                 "Hatterene, our last Pokémon, vs Incineroar 54% under Trick Room: resisted Mystical Fire (~15%) was played; "
                 "STAB Dazzling Gleam does ~3x that.",
                 lambda p, t: move_of(p, 1) == "dazzlinggleam", "slot 1 Dazzling Gleam"),
    LiveScenario("m3_t7_gleam", "7ce71025", 7,
                 "Same endgame a turn later (Incineroar 54% -> chip): Gleam again, not Mystical Fire.",
                 lambda p, t: move_of(p, 1) == "dazzlinggleam", "slot 1 Dazzling Gleam"),
    LiveScenario("m1_t8_no_stall", "d9405f02", 8,
                 "Cresselia is our LAST Pokémon vs +3 Gholdengo and Iron Hands; nothing runs out, so Protect only delays "
                 "(it Protected four turns running, believing benched teammates were coming).",
                 lambda p, t: move_of(p, 1) not in ("protect", "helpinghand"), "slot 1 does not Protect or Helping Hand"),
    LiveScenario("m1_t10_no_stall", "d9405f02", 10,
                 "Same, two turns later: still nothing to wait for.",
                 lambda p, t: move_of(p, 1) not in ("protect", "helpinghand"), "slot 1 does not Protect or Helping Hand"),
    LiveScenario("m1_t6_no_fake_out_into_ghost", "d9405f02", 6,
                 "Fresh Sneasler + Cresselia vs Dragonite and Gholdengo (a Ghost). The code's top candidate live was Fake Out "
                 "into Gholdengo (immune); Close Combat and Dire Claw do nothing to it either.",
                 lambda p, t: not (move_of(p, 0) in ("fakeout", "closecombat", "direclaw") and target_species(p, 0, t) == "gholdengo"),
                 "slot 0 does not aim Fake Out / Close Combat / Dire Claw at Gholdengo"),
    LiveScenario("m4_t1_no_earthquake_into_balloon", "0a2b7c2a", 1,
                 "Scarf Landorus + Tornadus vs Dragonite (Flying) and Gholdengo (Air Balloon): Earthquake hits neither, and the "
                 "Scarf would lock it in. Rock Slide breaks Multiscale, pops the Balloon and can flinch.",
                 lambda p, t: move_of(p, 0) != "earthquake", "slot 0 does not Earthquake"),
    LiveScenario("m3_t3_remove_whimsicott", "7ce71025", 3,
                 "Great Tusk + Hatterene vs Whimsicott 9% and a FRESH Incineroar (Fake Out). Live: Tusk 'outspeeds and KOs "
                 "Whimsicott', got Faked Out, and Moonblast KO'd it; Hatterene's Mystical Fire went into Incineroar (resisted). "
                 "Hatterene must take Whimsicott out itself (Gleam hits both).",
                 lambda p, t: move_of(p, 1) == "dazzlinggleam" or target_species(p, 1, t) == "whimsicott",
                 "slot 1 Dazzling Gleam, or slot 1 attacks Whimsicott"),
]


def _sequence(folder: Path, seat: int = 0) -> list[Path]:
    def order(p: Path) -> tuple[int, int]:
        m = re.match(r"(draft|team_preview|moving)-(\d+)-seat", p.name)
        return (PHASES[m.group(1)], int(m.group(2))) if m else (9, 0)

    return sorted(folder.glob(f"*-seat{seat}.json"), key=order)


def run(sc: LiveScenario, *, code_only: bool, provider=None) -> dict:
    folder = next(p for p in LIVE.iterdir() if p.is_dir() and p.name.startswith(sc.session))
    with tempfile.TemporaryDirectory() as tmp:
        agent = PokemonAgent(None, version="live-scenario", log_dir=tmp, capture_dir="", log=lambda *_: None)
        seen_at_turn = 0
        for path in _sequence(folder):
            raw = json.loads(path.read_text())
            state = GameState.from_mcp_state(raw)
            if not state.legal_actions:
                continue
            ctx = DecisionContext(session_id=folder.name, tournament_id=None, game_type="pokemon_vgc_doubles_draft",
                                  agent_id=(raw.get("current_actor") or {}).get("agent_id") or "me", seat_position=0)
            obs = raw.get("observation") or {}
            target = state.phase == "moving" and obs.get("turn") == sc.turn
            if target and seen_at_turn == sc.index:
                if not code_only:
                    agent._provider = provider
                started = time.monotonic()
                decision = agent.choose_action(state, ctx)
                elapsed = time.monotonic() - started
                payload = getattr(decision, "action", decision)
                template = state.legal_actions[0].input.get("action") or {}
                rows = [json.loads(line) for f in Path(tmp).glob("*.jsonl") for line in f.read_text().splitlines() if line.strip()]
                row = [r for r in rows if r.get("kind") == "turn"][-1]
                ok = bool(sc.accept(payload, template))
                return {"name": sc.name, "ok": ok, "payload": payload, "model": row.get("model"), "fallback": row.get("fallback"),
                        "reasoning": (row.get("answer") or {}).get("reasoning_summary"), "win_condition": row.get("win_condition"),
                        "top_candidate": (row.get("turn_sheet", {}).get("candidate_turns") or [{}])[0].get("why"), "seconds": round(elapsed, 1)}
            if target:
                seen_at_turn += 1
            agent.choose_action(state, ctx)
    raise RuntimeError(f"{sc.name}: decision for turn {sc.turn} #{sc.index} not found in {folder.name}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--code-only", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--repeat", type=int, default=1)
    args = ap.parse_args(argv)
    provider = None
    if not args.code_only:
        from agent.llm.anthropic_provider import provider_from_env

        provider = provider_from_env(log=lambda *_: None)
    chosen = [s for s in SCENARIOS if not args.only or s.name.startswith(args.only)]
    passed = total = 0
    log_path = ROOT / "logs" / "live_scenarios.jsonl"
    log_path.parent.mkdir(exist_ok=True)
    for sc in chosen:
        for _ in range(args.repeat):
            result = run(sc, code_only=args.code_only, provider=provider)
            total += 1
            passed += result["ok"]
            mark = "PASS" if result["ok"] else "FAIL"
            print(f"{mark} {sc.name:26} [{result['model'] or 'code'}{' FALLBACK' if result['fallback'] and not args.code_only else ''}, {result['seconds']}s] "
                  f"played {json.dumps({k: v for k, v in result['payload'].items() if k != 'type'})}")
            if not result["ok"]:
                print(f"     want: {sc.accept_text}\n     why: {result['reasoning'] or result['top_candidate']}")
            with log_path.open("a") as f:
                f.write(json.dumps({"ts": time.time(), "code_only": args.code_only, **result}, default=str) + "\n")
    print(f"\n{passed}/{total} passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
