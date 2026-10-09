"""Replay every captured live Team Preview through one code-only agent and print what it brings and leads.

Each captured session (tests/fixtures/live/<session>/) is replayed from our seat: the draft decisions rebuild the match
memory, then the Team Preview decision is taken with the given tuning (the deep preview search's budget is the tuning
key ``lineup_sim_budget_s``; 0 is the stage2 preview). The match's secret tie-break seed is fixed per session (a hash of
the session id and seat) so two builds or two budgets are compared on the same seed. No simulator, no model.

    python scripts/preview_replay.py --tune '{"lineup_sim_budget_s": 0}' --out /tmp/previews-0.json
    python scripts/preview_replay.py --tune '{"lineup_sim_budget_s": 20}' --out /tmp/previews-20.json --compare /tmp/previews-0.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from altruagent import DecisionContext  # noqa: E402
from altruagent.models import GameState  # noqa: E402

from agent.agent import PokemonAgent  # noqa: E402
from agent.pokemon import draft, tuning  # noqa: E402

LIVE = ROOT / "tests" / "fixtures" / "live"
PHASES = {"draft": 0, "team_preview": 1, "moving": 2}
# The Oct 7 losses (REVIEW-2026-10-08 section 3): the lead the review names as the better one from our six. L4's row names
# no new lead: its line (Landorus-T's Earthquake after Tornadus's Bleakwind Storm pops the Balloon) is the live lead's.
REVIEW_LEADS = {"d9405f02-seat0": ("L1", {"garchomp", "cresselia"}), "1aef9227-seat0": ("L2", {"gyarados", "archaludon"}),
                "7ce71025-seat0": ("L3", {"indeedeef", "hatterene"}), "0a2b7c2a-seat0": ("L4", {"landorustherian", "tornadus"})}


def sequence(folder: Path, seat: int) -> list[Path]:
    def order(p: Path) -> tuple[int, int]:
        m = re.match(r"(draft|team_preview|moving)-(\d+)-seat", p.name)
        return (PHASES[m.group(1)], int(m.group(2))) if m else (9, 0)

    return sorted((p for p in folder.glob(f"*-seat{seat}.json") if not p.name.startswith("moving-")), key=order)


def session_seed(name: str) -> int:
    return int(hashlib.sha1(name.encode("utf-8")).hexdigest()[:16], 16)


def replay(folder: Path, seat: int, params: dict) -> dict:
    """The Team Preview decision of one captured session, replayed after its draft, with the seed fixed."""
    name = f"{folder.name[:8]}-seat{seat}"
    original = draft.secrets.randbits
    draft.secrets.randbits = lambda bits: session_seed(name) & ((1 << bits) - 1)  # the match's seed, fixed for the replay
    try:
        with tempfile.TemporaryDirectory() as tmp:
            agent = PokemonAgent(None, version="preview-replay", log_dir=tmp, capture_dir="", log=lambda *_: None, params=params)
            for path in sequence(folder, seat):
                raw = json.loads(path.read_text())
                state = GameState.from_mcp_state(raw)
                if not state.legal_actions:
                    continue
                ctx = DecisionContext(session_id=folder.name, tournament_id=None, game_type="pokemon_vgc_doubles_draft",
                                      agent_id=(raw.get("current_actor") or {}).get("agent_id") or "me", seat_position=seat)
                if state.phase != "team_preview":
                    agent.choose_action(state, ctx)
                    continue
                started = time.monotonic()
                decision = agent.choose_action(state, ctx)
                elapsed = time.monotonic() - started
                payload = getattr(decision, "action", decision)
                rows = [json.loads(line) for f in Path(tmp).glob("*.jsonl") for line in f.read_text().splitlines() if line.strip()]
                row = next((r for r in reversed(rows) if r.get("kind") == "lineup" or "candidates" in r), {})
                return {"session": name, "bring": sorted(payload.get("bring") or []), "leads": sorted(payload.get("leads") or []),
                        "seconds": round(elapsed, 2), "candidates": row.get("candidates") or [],
                        "table_chars": len(row.get("preview_table") or "")}
    finally:
        draft.secrets.randbits = original
    raise RuntimeError(f"{name}: no Team Preview decision found")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tune", default="", help="tuning overrides as a JSON object, e.g. '{\"lineup_sim_budget_s\": 20}'")
    ap.add_argument("--only", default="", help="session id prefix")
    ap.add_argument("--out", default="", help="write the results here (JSON)")
    ap.add_argument("--compare", default="", help="a results file from an earlier run: mark previews whose four or leads differ")
    args = ap.parse_args(argv)
    params = tuning.merged(json.loads(args.tune)) if args.tune else tuning.merged()
    print(tuning.label(params))
    before = {r["session"]: r for r in json.loads(Path(args.compare).read_text())} if args.compare else {}
    results = []
    for folder in sorted(p for p in LIVE.iterdir() if p.is_dir() and p.name.startswith(args.only)):
        for seat in (0, 1):
            if not list(folder.glob(f"team_preview-*-seat{seat}.json")):
                continue
            r = replay(folder, seat, params)
            results.append(r)
            old = before.get(r["session"])
            mark = ""
            if old:
                mark = " same" if (old["bring"], old["leads"]) == (r["bring"], r["leads"]) else \
                    f" CHANGED (was {old['bring']} lead {old['leads']})"
            notes = r["candidates"][0]["notes"] if r["candidates"] else []
            sim = [n for n in notes if n.startswith(("sim:", "sim search", "sim changes"))]
            print(f"{r['session']}: bring {r['bring']} lead {r['leads']} [{r['seconds']}s, table {r['table_chars']} chars]{mark}"
                  + "".join(f"\n    {n}" for n in sim), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1, default=str))
    rows = {r["session"]: r for r in results}
    print("\nOct 7 losses: our lead against the review's (REVIEW section 3)")
    for session, (label, review) in REVIEW_LEADS.items():
        if session in rows:
            old = before.get(session)
            new = set(rows[session]["leads"])
            print(f"  {label} {session}: review {sorted(review)}; this run {sorted(new)} {'MATCH' if new == review else 'differs'}"
                  + (f"; before {old['leads']} {'MATCH' if set(old['leads']) == review else 'differs'}" if old else ""))
    if results:
        times = sorted(r["seconds"] for r in results)
        print(f"\ntimes: median {times[len(times) // 2]}s, max {times[-1]}s over {len(times)} previews")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
