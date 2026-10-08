"""Grade the judge's overrides: every battle turn where the model's answer differed from the code's top candidate.

    python scripts/grade_overrides.py logs/live/*.jsonl logs/*.jsonl

For each session with model-answered turns it reports how often the model took the code's #1, another listed candidate,
or something of its own, and counts red flags that can be read from the decision log alone:
  FAILS      the chosen option was marked FAILS/BLOCKED on the sheet (does nothing this turn)
  LETHAL     a LETHAL warning named this slot and it still attacked (the recheck should have fired; counts those too)
  ZERO_DMG   the chosen attack's target row shows 0 damage while another target of the same move does damage
  NO_THREAT  Protect/Detect chosen for a slot with no warning or EXPOSED note mentioning it
  KO_LEFT    the code's top candidate guaranteed a KO and the model's answer contains no guaranteed-KO attack
Each flagged turn is printed with the model's one-sentence reasoning so a human can judge it. Nothing here is proof of a
mistake: the point is to find patterns worth turning into prompt rules, rechecks or scenarios.
"""

from __future__ import annotations

import glob
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.agent import _lethal_recheck  # noqa: E402  (the same test the agent applies live)


class _Sheet:
    def __init__(self, warnings):
        self.warnings = warnings


PROTECT_LIKE = {"protect", "detect", "wideguard", "quickguard", "spikyshield", "banefulbunker", "burningbulwark", "silktrap", "kingsshield", "obstruct"}


def rows_of(path: str) -> list[dict]:
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def slot_options(sheet: dict, slot: int) -> dict[int, dict]:
    for entry in sheet.get("our_damage_estimates") or []:
        if entry.get("slot") == slot:
            return {o["option"]: o for o in entry.get("options") or []}
    return {}


def grade_turn(turn: dict) -> tuple[str, list[str]]:
    sheet = turn.get("turn_sheet") or {}
    answer = turn.get("answer") or {}
    cands = sheet.get("candidate_turns") or []
    if not answer or not cands:
        return "n/a", []
    picked = {k: {"option": answer[k].get("option"), "target": answer[k].get("target", 0)} for k in ("slot_0", "slot_1") if isinstance(answer.get(k), dict)}
    def same(c):
        return all(c.get(k) and c[k].get("option") == v["option"] and (c[k].get("target", 0) or 0) == (v["target"] or 0) for k, v in picked.items())
    kind = "top" if same(cands[0]) else ("listed" if any(same(c) for c in cands[1:]) else "own")
    flags: list[str] = []
    warnings = sheet.get("WARNINGS") or []
    notes = sheet.get("notes") or []
    payload = turn.get("payload") or {}
    for n in (0, 1):
        key = f"slot_{n}"
        a = picked.get(key)
        if not a:
            continue
        opts = slot_options(sheet, n)
        row = opts.get(a["option"])
        move_id = (payload.get(key) or {}).get("move_id")
        is_attack = row is not None and any(t.get("side") == "theirs" for t in row.get("targets") or [])
        if row and ("FAILS" in (row.get("note") or "") or "BLOCKED" in (row.get("note") or "")):
            flags.append(f"FAILS({move_id})")
        if n == 0 and _lethal_recheck(payload, _Sheet(warnings)):
            flags.append(f"LETHAL(attacked into a lethal warning{' after recheck' if turn.get('recheck') else ' NO RECHECK'})")
        if is_attack and a["target"]:
            rows = {t["target"]: t for t in row.get("targets") or []}
            mine = rows.get(a["target"])
            if mine and mine.get("damage_pct_of_current_hp", [0, 0])[1] == 0 and any(t.get("damage_pct_of_current_hp", [0, 0])[1] > 0 for t in rows.values() if t.get("side") == "theirs"):
                flags.append(f"ZERO_DMG({move_id}->{a['target']})")
        if move_id in PROTECT_LIKE:
            mentioned = any(f"slot {n}" in w for w in warnings) or any(f"slot {n} " in x and "EXPOSED" in x for x in notes)
            if not mentioned:
                flags.append(f"NO_THREAT(slot {n} {move_id})")
    top = cands[0]
    if kind != "top" and "guaranteed" in json.dumps(top.get("why", "")).lower() or (kind != "top" and " KO" in str(top.get("why", "")) and "already KOs" in str(top.get("why", ""))):
        my_rows = []
        for n in (0, 1):
            a = picked.get(f"slot_{n}")
            if a:
                row = slot_options(sheet, n).get(a["option"]) or {}
                my_rows += [t for t in row.get("targets") or [] if (t.get("target") == a["target"] or a["target"] == 0) and t.get("side") == "theirs"]
        if not any(t.get("ko") == "guaranteed" for t in my_rows):
            flags.append("KO_LEFT")
    return kind, flags


def main(argv: list[str]) -> int:
    paths = [p for pattern in (argv or ["logs/live/*.jsonl", "logs/*.jsonl"]) for p in glob.glob(pattern)]
    total = Counter(); flag_total = Counter(); printed = 0
    for path in sorted(set(paths)):
        rows = rows_of(path)
        turns = [r for r in rows if r.get("kind") == "turn" and r.get("model") and not r.get("fallback")]
        if not turns:
            continue
        kinds = Counter(); flags = Counter()
        for t in turns:
            kind, fl = grade_turn(t)
            kinds[kind] += 1
            for f in fl:
                flags[f.split("(")[0]] += 1
                if printed < 40:
                    printed += 1
                    print(f"  {Path(path).name[:8]} turn {t.get('turn')}: {f} | {kind} | {(t.get('answer') or {}).get('reasoning_summary', '')[:150]}")
        total.update(kinds); flag_total.update(flags)
        print(f"{Path(path).name}: {len(turns)} model turns | top {kinds['top']} listed {kinds['listed']} own {kinds['own']} | flags {dict(flags) or 'none'}")
    n = sum(total.values()) or 1
    print(f"\nALL: {n} model turns | took code's #1 {total['top']} ({100*total['top']//n}%), another candidate {total['listed']}, own move {total['own']} | flags {dict(flag_total) or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
