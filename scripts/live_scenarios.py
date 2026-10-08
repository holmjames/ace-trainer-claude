"""Exact positions from real matches, replayed through the real agent, each with a known-good answer.

Hand-built scenarios (scripts/scenarios.py) can be transcribed wrong; these cannot. Each one replays a captured match
(tests/fixtures/live/<session>/) through one code-only agent up to the target decision, then asks for THAT decision
only — from the model chain in .env, or from the computed fallback with --code-only — and checks it.

    python scripts/live_scenarios.py --code-only          # free: what the code plays
    python scripts/live_scenarios.py                      # one model call per scenario (~$0.15 each on Opus)
    python scripts/live_scenarios.py --only m3_t6 --repeat 3

The positions come from the four Oct 7 losses (L1-L4 below = sessions d9405f02, 1aef9227, 7ce71025, 0a2b7c2a) and the
Oct 8 Testing win bcbaabe2 (see scripts/postmortem.py for the transcripts); a session is replayed from whichever seat we
sat in (player-2 matches are captured as ``*-seat1.json``); the damage numbers in the lessons are the
sheet's own (agent/pokemon/battle.py:damage_percent) at that position. A ``provisional`` scenario's best play was derived
from the transcript and the calc rather than documented in the review: it is run and shown but not counted.
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
    provisional: bool = False  # best play derived, not yet confirmed: reported, not counted


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


def aims_at(payload: dict, slot: int, template: dict, species: str) -> bool:
    """A single-target attack from ``slot`` aimed at ``species`` (spread moves and non-attacks never count)."""
    return target_species(payload, slot, template) == species


def covers(payload: dict, slot: int) -> bool:
    """The slot Protects or switches out."""
    s = payload.get(f"slot_{slot}") or {}
    return s.get("type") == "switch" or move_of(payload, slot) == "protect"


def switches_to(payload: dict, slot: int, species: str) -> bool:
    s = payload.get(f"slot_{slot}") or {}
    return s.get("type") == "switch" and species_key(s.get("species") or "") == species


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
    # ---- the decisive positions of the Oct 8 review (REVIEW §3, briefs/core.md "Validation"), L1-L4 in match order
    LiveScenario("m1_t3_pressure_gholdengo", "d9405f02", 3,
                 "L1 T3: Sneasler 35% + Cresselia vs Dragonite 63% (their Tailwind) and Air Balloon Gholdengo. Sneasler dies to "
                 "Extreme Speed (143-169%) or Make It Rain (119-140%) before it moves, and the Dragonite everyone aims at Protects "
                 "(it did). Take Sneasler out and pressure Gholdengo: no single-target attack into Dragonite.",
                 lambda p, t: not aims_at(p, 0, t, "dragonite") and not aims_at(p, 1, t, "dragonite"),
                 "no single-target attack aimed at Dragonite"),
    LiveScenario("m1_t4_cresselia_protects", "d9405f02", 4,
                 "L1 T4: fresh Scarf Landorus + Cresselia 68% vs Dragonite 63% and +1 Gholdengo. Shadow Ball KOs Cresselia (118-138%) "
                 "before it moves, so it Protects; Landorus (Extreme Speed 23-27% + Shadow Ball killed it) Rock Slides both (59-70% "
                 "into Dragonite, pops the Balloon) or pivots. Derived from the transcript, awaiting confirmation.",
                 lambda p, t: move_of(p, 1) == "protect", "slot 1 Protect", provisional=True),
    LiveScenario("m1_t5_no_helping_hand_into_protect", "d9405f02", 5,
                 "L1 T5: Garchomp (sash) + Cresselia vs the same two. Scale Shot threatens Garchomp (94-110%, 61% KO) and Shadow Ball "
                 "KOs Cresselia (118-138%). If Garchomp Protects, Helping Hand boosts nothing: Cresselia attacks (Icy Wind 43-51% "
                 "into Dragonite) or Protects itself.",
                 lambda p, t: not ({move_of(p, 0), move_of(p, 1)} == {"protect", "helpinghand"}),
                 "not Protect on one slot with Helping Hand on the other"),
    LiveScenario("m1_t6_dire_claw_ko", "d9405f02", 6,
                 "L1 T6: fresh Sneasler 35% + Cresselia vs Dragonite 63% at -1 Def. Dire Claw KOs it (87-102%); Fake Out only delays. "
                 "Extreme Speed (96-113%, 76% KO) may land first; the model overrode the code's Fake Out here and was right.",
                 lambda p, t: move_of(p, 0) == "direclaw" and aims_at(p, 0, t, "dragonite"), "slot 0 Dire Claw into Dragonite"),
    LiveScenario("m2_t1_tailwind_first", "1aef9227", 1,
                 "L2 T1: Iron Hands + Whimsicott (both -1) vs Incineroar + Sylveon. Incineroar's Fake Out beats Iron Hands' (it is "
                 "faster; 50% flinch) and Hyper Voice hits both (the sheet said 48-57% into Iron Hands; the live hit took 83%). The lead "
                 "is the error (review: Gyarados + Archaludon); from here Prankster Tailwind is the one sure gain. Derived, awaiting "
                 "confirmation.",
                 lambda p, t: move_of(p, 1) == "tailwind", "slot 1 Tailwind", provisional=True),
    LiveScenario("m2_t3_not_into_the_protecting_sylveon", "1aef9227", 3,
                 "L2 T3: Archaludon + Gyarados (our Tailwind) vs Incineroar 13% and Sylveon 75%, both -1. Sylveon, just hit, Protects "
                 "(it did): Flash Cannon (59-70%) and Waterfall (69-82%) into it are wasted. Earthquake KOs the 13% Incineroar "
                 "(381-448%) and hits Sylveon too; pressure the other slot.",
                 lambda p, t: not aims_at(p, 0, t, "sylveon") and (move_of(p, 1) == "earthquake" or aims_at(p, 1, t, "incineroar")
                                                                   or aims_at(p, 0, t, "incineroar")),
                 "no Flash Cannon into Sylveon; an attack reaches Incineroar (Earthquake or a single hit)"),
    LiveScenario("m2_t4_protect_the_doubled_archaludon", "1aef9227", 4,
                 "L2 T4: Archaludon + Gyarados vs a fresh Flutter Mane and Sylveon 75%. Moonblast (60-71%) + Hyper Voice (49-57%) = "
                 "109-128% into Archaludon: it is the double's target (it was, and died). It Protects; Waterfall alone KOs Flutter "
                 "Mane (91-107%).",
                 lambda p, t: move_of(p, 0) == "protect", "slot 0 Protect"),
    LiveScenario("m3_t1_cover_a_lethal_hit", "7ce71025", 1,
                 "L3 T1: Great Tusk + Tornadus vs Iron Hands + Sylveon. Wild Charge KOs Tornadus (129-152%) and Hyper Voice KOs Tusk "
                 "(100-118%, 99%); Headlong Rush KOs Iron Hands first only 45% of the time (live it left it at 2%). One of ours "
                 "Protects or leaves instead of both attacking into two KOs; the lead itself was the error (review: Indeedee + "
                 "Hatterene). Derived, awaiting confirmation.",
                 lambda p, t: covers(p, 0) or covers(p, 1), "slot 0 or slot 1 Protects or switches", provisional=True),
    LiveScenario("m3_t3_tusk_protect_gleam", "7ce71025", 3,
                 "L3 T3 again, the full answer: Whimsicott 9% moves first and its Moonblast KOs Tusk (115-135%); the fresh Incineroar "
                 "Fakes Out. Tusk Protects while Hatterene's Dazzling Gleam removes Whimsicott (609-717%) and hits Incineroar, "
                 "instead of Headlong Rush into the Fake Out.",
                 lambda p, t: move_of(p, 0) == "protect" and move_of(p, 1) == "dazzlinggleam", "slot 0 Protect, slot 1 Dazzling Gleam"),
    LiveScenario("m3_t5_helping_hand_gleam", "7ce71025", 5,
                 "L3 T5: Indeedee 22% + Hatterene 62% vs a lone Incineroar 65% (-2 SpA) under Psychic Terrain. Helping Hand + Dazzling "
                 "Gleam (47-55% unboosted, x1.5 with the hand) twice wins the 2v1; Trick Room spends the turn while Knock Off KOs "
                 "Indeedee (209-246%).",
                 lambda p, t: move_of(p, 0) == "helpinghand" and move_of(p, 1) == "dazzlinggleam", "slot 0 Helping Hand, slot 1 Dazzling Gleam"),
    LiveScenario("m4_t2_no_telegraphed_ko", "0a2b7c2a", 2,
                 "L4 T2: Baxcalibur 58% + Tornadus vs Dragonite 94% and Air Balloon Gholdengo. The 'guaranteed' Icicle Spear into "
                 "Dragonite (247-290%) is telegraphed: it switched to Incineroar and Make It Rain KO'd Baxcalibur (163-192%). "
                 "Baxcalibur goes out to Landorus; Bleakwind Storm hits both and pops the Balloon.",
                 lambda p, t: (p.get("slot_0") or {}).get("type") == "switch" and move_of(p, 1) == "bleakwindstorm",
                 "slot 0 switches out (Landorus), slot 1 Bleakwind Storm"),
    LiveScenario("m4_t4_out_of_the_double", "0a2b7c2a", 4,
                 "L4 T4: Scarf Landorus (locked into Earthquake) + Tornadus 48% vs Archaludon 35% (+2 Def) and Dragonite 94%. Draco "
                 "Meteor (66-78%) + Scale Shot (77-91%) = 143-168% into Landorus (it died to exactly that) and Draco Meteor KOs "
                 "Tornadus (119-140%): Landorus switches to Lucario, which resists both Dragon hits, and Tornadus Protects.",
                 lambda p, t: switches_to(p, 0, "lucario") and move_of(p, 1) == "protect", "slot 0 switch to Lucario, slot 1 Protect"),
    LiveScenario("m4_t5_not_into_the_17pct_protect", "0a2b7c2a", 5,
                 "L4 T5: Lucario + Tornadus 48% vs Archaludon 17% (+3 Def) and Dragonite 65%. The 17% Archaludon is the obvious target "
                 "and Protected (so did Dragonite): Close Combat belongs on Dragonite (73-86%) and Bleakwind Storm pressures both. "
                 "Derived, awaiting confirmation.",
                 lambda p, t: not aims_at(p, 0, t, "archaludon") and move_of(p, 1) == "bleakwindstorm",
                 "slot 0 not into Archaludon; slot 1 Bleakwind Storm", provisional=True),
    LiveScenario("m4_t7_lucario_protects", "0a2b7c2a", 7,
                 "L4 T7: Lucario 90% at -1 Def/-1 SpD + Tornadus vs Incineroar 58% and Dragonite 65%. Close Combat KOs Incineroar "
                 "(222-261%) but drops Lucario to -2, and Scale Shot is already 85-100% at -1: it KO'd Lucario. With no bench, "
                 "Lucario Protects (Tornadus too: a 90% Fake Out flinch).",
                 lambda p, t: move_of(p, 0) == "protect", "slot 0 Protect"),
    # ---- the Oct 8 Testing win vs the search bot (bcbaabe2): the turns its post-mortem flagged
    LiveScenario("o8_t5_no_protect_for_a_fake_double", "bcbaabe2", 5,
                 "Archaludon + Wo-Chien 62% vs Dragonite and Ting-Lu 63%. The sheet added Scale Shot (53-62%) + Ruination into a "
                 "'103-112%' double, but Ruination halves what is left and can never finish: no Protect for a double that does not "
                 "exist. Attack (Body Press 39-46% into Ting-Lu, Foul Play 32-38% into Dragonite).",
                 lambda p, t: move_of(p, 1) != "protect", "slot 1 does not Protect"),
    LiveScenario("o8_t12_no_milotic_into_spikes", "bcbaabe2", 12,
                 "Wo-Chien 43% vs Dragonite 92% and Rillaboom under Grassy Terrain, Spikes on our side. Scale Shot (111-130%) or "
                 "U-turn (143-169%) KOs it, but a fresh Milotic into Spikes + Grassy Glide + Scale Shot died on entry (it did): "
                 "never pivot into hazards and two attackers.",
                 lambda p, t: not switches_to(p, 1, "milotic"), "slot 1 does not switch to Milotic"),
    LiveScenario("o8_t13_no_fake_priority_escape", "bcbaabe2", 13,
                 "Archaludon + Arcanine 33% vs Dragonite 92% and Rillaboom 47%. Scale Shot KOs Arcanine (141-166%) before it moves "
                 "and its Extreme Speed (39-46% into Dragonite) cannot KO, so priority is no escape: Arcanine Protects (the model "
                 "did) while Body Press KOs Rillaboom (113-133%).",
                 lambda p, t: covers(p, 1), "slot 1 Protects or switches"),
    LiveScenario("o8_t14_no_fake_priority_escape", "bcbaabe2", 14,
                 "A turn later, Arcanine 39% vs the lone Dragonite: Scale Shot 119-140% again and Extreme Speed is still no escape; "
                 "Protect (or Wo-Chien) while Body Press works on Dragonite (43-51%).",
                 lambda p, t: covers(p, 1), "slot 1 Protects or switches"),
]


def _sequence(folder: Path, seat: int = 0) -> list[Path]:
    def order(p: Path) -> tuple[int, int]:
        m = re.match(r"(draft|team_preview|moving)-(\d+)-seat", p.name)
        return (PHASES[m.group(1)], int(m.group(2))) if m else (9, 0)

    return sorted(folder.glob(f"*-seat{seat}.json"), key=order)


def our_side(obs: dict) -> str | None:
    """Our protocol side from a battle observation: our ``team`` keys carry it ("p2: Archaludon" when we are player 2)."""
    team = obs.get("team") or {}
    keys = team.keys() if isinstance(team, dict) else [m.get("name", "") for m in team if isinstance(m, dict)]
    return next((k.split(":")[0] for k in keys if re.match(r"p[12]:", k or "")), None)


def seat_of(folder: Path) -> int:
    """The seat we sat in for a captured session: from the observations (the way scripts/postmortem.py reads it), else
    from the capture file names."""
    for path in _sequence(folder, 0) + _sequence(folder, 1):
        if path.name.startswith("moving-"):
            side = our_side((json.loads(path.read_text()).get("observation") or {}))
            if side:
                return 0 if side == "p1" else 1
    return 1 if _sequence(folder, 1) and not _sequence(folder, 0) else 0


def run(sc: LiveScenario, *, code_only: bool, provider=None) -> dict:
    folder = next(p for p in LIVE.iterdir() if p.is_dir() and p.name.startswith(sc.session))
    seat = seat_of(folder)
    with tempfile.TemporaryDirectory() as tmp:
        agent = PokemonAgent(None, version="live-scenario", log_dir=tmp, capture_dir="", log=lambda *_: None)
        seen_at_turn = 0
        for path in _sequence(folder, seat):
            raw = json.loads(path.read_text())
            state = GameState.from_mcp_state(raw)
            if not state.legal_actions:
                continue
            ctx = DecisionContext(session_id=folder.name, tournament_id=None, game_type="pokemon_vgc_doubles_draft",
                                  agent_id=(raw.get("current_actor") or {}).get("agent_id") or "me", seat_position=seat)
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
    passed = total = prov_passed = prov_total = 0
    log_path = ROOT / "logs" / "live_scenarios.jsonl"
    log_path.parent.mkdir(exist_ok=True)
    for sc in chosen:
        for _ in range(args.repeat):
            result = run(sc, code_only=args.code_only, provider=provider)
            result["provisional"] = sc.provisional
            if sc.provisional:
                prov_total += 1
                prov_passed += result["ok"]
            else:
                total += 1
                passed += result["ok"]
            mark = ("PASS" if result["ok"] else "FAIL") + ("?" if sc.provisional else "")
            print(f"{mark:5} {sc.name:40} [{result['model'] or 'code'}{' FALLBACK' if result['fallback'] and not args.code_only else ''}, {result['seconds']}s] "
                  f"played {json.dumps({k: v for k, v in result['payload'].items() if k != 'type'})}")
            if not result["ok"]:
                print(f"     want: {sc.accept_text}\n     why: {result['reasoning'] or result['top_candidate']}")
            with log_path.open("a") as f:
                f.write(json.dumps({"ts": time.time(), "code_only": args.code_only, **result}, default=str) + "\n")
    print(f"\n{passed}/{total} passed" + (f" ({prov_passed}/{prov_total} provisional, not counted)" if prov_total else ""))
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
