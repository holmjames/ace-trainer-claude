"""Turn one live match into a readable post-mortem transcript (markdown on stdout).

    python scripts/postmortem.py <session_id or prefix> [--fixtures tests/fixtures/live] [--logs logs]

Joins three sources: the captured observations (tests/fixtures/live/<session>/), our decision log
(logs/<session>.jsonl) and the battle's own protocol log (in the last captured observation). Prints the
draft (both sides, with the opponent's public reasons), both lineups, then every turn: what happened on the
board, followed by what we chose next, the code's top candidates, the warnings, and the model's reasoning.
The events after our final decision are not captured (the match ends without another observation).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

NOISE = ("|j|", "|t:|", "|inactive|", "|gametype|", "|player|", "|teamsize|", "|gen|", "|tier|", "|rule|",
         "|clearpoke", "|poke|", "|teampreview", "|upkeep", "|init|", "|title|", "|-hint|", "|raw|", "|start",
         "|c|", "|timer|", "|split|", "|inactiveoff|")


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def find_session(prefix: str, fixtures: Path) -> Path:
    hits = sorted(p for p in fixtures.iterdir() if p.is_dir() and p.name.startswith(prefix))
    if len(hits) != 1:
        sys.exit(f"{len(hits)} fixture folders match {prefix!r}")
    return hits[0]


def seq(name: str) -> int:
    m = re.search(r"-(\d+)-seat", name) or re.search(r"(\d+)", name)
    return int(m.group(1)) if m else 0


def card_line(c: dict) -> str:
    evs = " / ".join(f"{v} {k}" for k, v in (c.get("evs") or {}).items())
    return (f"{c['species']} @ {c.get('item')} | {c.get('ability')} | {c.get('nature')} | {evs} | "
            f"{', '.join(c.get('moves') or [])}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session")
    ap.add_argument("--fixtures", default="tests/fixtures/live")
    ap.add_argument("--logs", default="logs")
    args = ap.parse_args(argv)
    folder = find_session(args.session, Path(args.fixtures))
    sid = folder.name
    files = sorted(folder.glob("*.json"), key=lambda p: seq(p.name))
    drafts = [load(p) for p in files if p.name.startswith("draft-")]
    moving = sorted((p for p in files if p.name.startswith("moving-")), key=lambda p: seq(p.name))
    log_path = Path(args.logs) / f"{sid}.jsonl"
    rows = [json.loads(l) for l in log_path.read_text().splitlines()] if log_path.exists() else []

    print(f"# Post-mortem {sid}\n")
    cfg = next((r for r in rows if r.get("kind") == "config"), {})
    if cfg:
        print(f"commit {cfg.get('commit', '')[:10]} · {cfg.get('models')}\n")

    # ---- draft
    if drafts:
        first = drafts[0]["observation"]
        last = drafts[-1]["observation"]
        me = drafts[0].get("current_actor", {}).get("agent_id") or first.get("current_seat")
        cards = {c["card_id"]: c for d in drafts for c in d["observation"].get("available_cards", [])}
        picks = list(last.get("picks") or [])
        mine_ids = {r["card_id"] for r in (last.get("rosters") or {}).get(me, [])}
        print("## Draft\n")
        print(f"First drafter: {'US' if first.get('first_drafter') == me else 'THEM'}\n")
        draft_rows = {r.get("pick_number"): r for r in rows if r.get("kind") == "draft"}
        # our final pick is not in `picks` of the last observation (it is made from it); add from the log
        seen = {p["pick_number"] for p in picks}
        for n, r in draft_rows.items():
            if n not in seen:
                picks.append({"pick_number": n, "seat_id": me, "card_id": r["card_id"], "species": r["species"],
                              "public_reason": None})
        for p in sorted(picks, key=lambda p: p["pick_number"]):
            who = "US  " if p["seat_id"] == me else "THEM"
            extra = ""
            r = draft_rows.get(p["pick_number"])
            if p["seat_id"] == me and r:
                extra = f" score {r.get('score')} :: {', '.join(r.get('notes') or [])}"
            elif p.get("public_reason"):
                extra = f" :: \"{p['public_reason']}\""
            print(f"- {p['pick_number']:>2} {who} {p['species']}{extra}")
        taken = {p["card_id"] for p in picks}
        tp = next((load(p) for p in files if p.name.startswith("team_preview-")), None)
        if tp:
            opp_species = {c.get("species") for c in tp["observation"].get("opponent_roster") or []}
            for cid, c in cards.items():
                if cid not in taken and cards[cid]["species"].lower().replace("-", "").replace(" ", "") in opp_species:
                    taken.add(cid)
                    print(f"- 12 THEM {c['species']} (final pick, from team preview)")
        print("\nPool (full sets):\n")
        for cid, c in sorted(cards.items(), key=lambda kv: kv[1]["species"]):
            owner = "US  " if cid in mine_ids or any(p["card_id"] == cid and p["seat_id"] == me for p in picks) \
                else ("THEM" if cid in taken else "left")
            print(f"- [{owner}] {card_line(c)}")
        print()

    # ---- lineup
    lu = next((r for r in rows if r.get("kind") == "lineup"), None)
    if lu:
        pay = lu["payload"]
        print("## Team preview\n")
        print(f"We brought {pay['bring']}, led {pay['leads']}  ({lu.get('model')}, {lu.get('latency_ms')} ms)")
        ans = lu.get("answer") or {}
        if ans.get("reasoning_summary"):
            print(f"Model: {ans['reasoning_summary']}")
        for c in (lu.get("candidates") or [])[:3]:
            print(f"- code #{c['rank']} {c['bring']} leads {c['leads']} score {c['score']} :: {'; '.join(c['notes'])}")
        print()

    # ---- battle
    if not moving:
        return 0
    log = load(moving[-1])["observation"].get("protocol_log") or []
    opp_seen: list[str] = []
    for line in log:
        m = re.match(r"\|(switch|drag)\|p2[ab]: ([^|]+)\|", line)
        if m and m.group(2) not in opp_seen:
            opp_seen.append(m.group(2))
    print(f"Opponent brought (seen in battle): {opp_seen}\n")
    turns = [r for r in rows if r.get("kind") == "turn"]
    by_turn: dict[int, list[dict]] = {}
    for r in turns:
        by_turn.setdefault(r.get("turn"), []).append(r)

    def show_decisions(n: int, which: str = "all") -> None:
        rs = by_turn.get(n, [])
        rs = rs[:1] if which == "main" else rs[1:] if which == "forced" else rs
        for r in rs:
            sheet = r.get("turn_sheet") or {}
            ans = r.get("answer") or {}
            fb = " FALLBACK" if r.get("fallback") else ""
            print(f"\n> **WE CHOSE (turn {n})**: {json.dumps(r.get('payload'))}  [{r.get('model')}{fb}, "
                  f"{r.get('latency_ms')} ms, clock {r.get('clock')}]")
            if ans.get("reasoning_summary"):
                print(f"> model: {ans['reasoning_summary']}")
            if r.get("win_condition"):
                print(f"> win condition: {r['win_condition']}")
            for w in sheet.get("WARNINGS") or []:
                print(f"> WARN {w}")
            for c in (sheet.get("candidate_turns") or [])[:4]:
                print(f"> cand {c.get('name')} score {c.get('score')} :: {c.get('why')}")
            if r.get("errors") or r.get("provider_errors"):
                print(f"> errors: {r.get('errors')} {r.get('provider_errors')}")

    current = 0
    print("## Battle\n")
    print("```")
    for line in log:
        if not line or line == "|" or line.startswith(NOISE):
            continue
        m = re.match(r"\|turn\|(\d+)", line)
        if m:
            print("```")
            if current:
                show_decisions(current, "forced")
            current = int(m.group(1))
            print(f"\n### Turn {current}")
            show_decisions(current, "main")
            print("\n```")
            continue
        print(line)
    print("```")
    show_decisions(current, "forced")
    later = sorted(n for n in by_turn if n > current)
    for n in later:
        show_decisions(n)
    print("\n(match ended after the last decision above; final events not captured)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
