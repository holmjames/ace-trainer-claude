"""Summarize test matches: wins, latency, fallbacks and cost per agent version.

Reads two kinds of files in logs/ (or the directory given with --logs):

- ``<session_id>.jsonl``   decision logs written by the agent (one line per decision)
- ``runtime-*.log``        the runtime's own output, tee'd by scripts/run_*.sh, which is the
                           only place the final score appears:
                             supervisor mode:  "[match ... seat=...] finished termination_reason=normal score=1.0"
                                               followed by "[agent] worker for session_id=<id> finished"
                             claim mode:       "Match finished (termination_reason=normal) ..." then "Your score: 1.0"
                                               (one match per log file; paired with the decision log active in that window)

    python scripts/tally.py
    python scripts/tally.py --logs logs --since 2026-10-10

Prices (per million tokens) for the cost estimate: Fable 5.1 $10 in / $50 out, Sonnet 5.5 $2 / $10,
Opus 5.5 $4 / $20; cache reads are billed at 10% of input.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PRICES = {  # $ per million tokens: (input, output)
    "claude-fable-5-1": (10.0, 50.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-haiku-4-5": (1.0, 5.0),
}
SUPERVISOR_SCORE = re.compile(r"finished termination_reason=(\S+)(?: score=(-?[\d.]+))?")
SUPERVISOR_SESSION = re.compile(r"worker for session_id=(\S+) finished")
CLAIM_FINISHED = re.compile(r"Match finished \(termination_reason=(\S+)\)")
CLAIM_SCORE = re.compile(r"Your score: (-?[\d.]+)")


def load_decisions(logs: Path, since: datetime | None) -> dict[str, list[dict]]:
    sessions: dict[str, list[dict]] = {}
    for path in sorted(logs.glob("*.jsonl")):
        lines = []
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                record = json.loads(raw)
            except ValueError:
                continue
            if since and _ts(record.get("ts")) and _ts(record.get("ts")) < since:
                continue
            lines.append(record)
        if lines:
            sessions[path.stem] = lines
    return sessions


def _ts(value) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value)).astimezone(timezone.utc) if value else None
    except ValueError:
        return None


def load_results(logs: Path, sessions: dict[str, list[dict]]) -> dict[str, dict]:
    """session_id -> {"score": float|None, "termination": str}."""
    results: dict[str, dict] = {}
    for path in sorted(logs.glob("runtime-*.log")):
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        pending: dict | None = None
        for line in lines:
            m = SUPERVISOR_SCORE.search(line)
            if m and "Match finished" not in line:
                pending = {"termination": m.group(1), "score": float(m.group(2)) if m.group(2) else None}
                continue
            s = SUPERVISOR_SESSION.search(line)
            if s and pending is not None:
                results[s.group(1)] = pending
                pending = None
        # Claim mode: one match per file; pair it with the decision log active during the file's window.
        finished = CLAIM_FINISHED.search(text)
        if finished:
            score = CLAIM_SCORE.search(text)
            window_end = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            candidates = []
            for session_id, records in sessions.items():
                if session_id in results:
                    continue
                last = _ts(records[-1].get("ts"))
                if last and last <= window_end:
                    candidates.append((window_end - last, session_id))
            if candidates:
                _, session_id = min(candidates)
                results[session_id] = {"termination": finished.group(1), "score": float(score.group(1)) if score else None}
    return results


def summarize(sessions: dict[str, list[dict]], results: dict[str, dict]) -> dict[str, dict]:
    per_version: dict[str, dict] = defaultdict(lambda: {
        "matches": set(), "wins": 0, "losses": 0, "draws": 0, "unknown": 0, "decisions": 0, "llm_latency_ms": [],
        "fallbacks": 0, "errors": 0, "models": defaultdict(int), "cost_usd": 0.0, "cache_read_tokens": 0, "input_tokens": 0,
    })
    for session_id, records in sessions.items():
        version = next((r.get("version") for r in records if r.get("version")), "unknown")
        stats = per_version[version]
        stats["matches"].add(session_id)
        for r in records:
            kind = r.get("kind")
            if kind in ("draft", "lineup", "turn"):
                stats["decisions"] += 1
            if r.get("latency_ms") is not None:
                stats["llm_latency_ms"].append(r["latency_ms"])
            if r.get("fallback") or kind == "fallback":
                stats["fallbacks"] += 1
            if kind == "error":
                stats["errors"] += 1
            model = r.get("model")
            usage = r.get("usage") or {}
            if model:
                stats["models"][model] += 1
                price_in, price_out = PRICES.get(model, (10.0, 50.0))
                uncached = (usage.get("input_tokens") or 0)
                cached = usage.get("cache_read_input_tokens") or 0
                written = usage.get("cache_creation_input_tokens") or 0
                out = usage.get("output_tokens") or 0
                stats["cost_usd"] += (uncached + 1.25 * written) * price_in / 1e6 + cached * price_in * 0.1 / 1e6 + out * price_out / 1e6
                stats["cache_read_tokens"] += cached
                stats["input_tokens"] += uncached + cached + written
        outcome = results.get(session_id)
        if outcome is None or outcome.get("score") is None:
            stats["unknown"] += 1
        elif outcome["score"] > 0.5:
            stats["wins"] += 1
        elif outcome["score"] < 0.5:
            stats["losses"] += 1
        else:
            stats["draws"] += 1
    return per_version


def render(per_version: dict[str, dict]) -> str:
    out = []
    header = f"{'version':28s} {'matches':>7s} {'W-L-D':>9s} {'unk':>4s} {'decisions':>9s} {'p50 ms':>7s} {'p95 ms':>7s} {'fallb':>5s} {'err':>4s} {'cost $':>7s}  models"
    out.append(header)
    out.append("-" * len(header))
    for version, s in sorted(per_version.items()):
        lat = sorted(s["llm_latency_ms"])
        p50 = int(statistics.median(lat)) if lat else 0
        p95 = int(lat[min(len(lat) - 1, int(len(lat) * 0.95))]) if lat else 0
        models = ", ".join(f"{m}×{n}" for m, n in sorted(s["models"].items()))
        out.append(f"{version:28s} {len(s['matches']):7d} {s['wins']:>3d}-{s['losses']:<3d}-{s['draws']:<1d} {s['unknown']:4d} {s['decisions']:9d} {p50:7d} {p95:7d} {s['fallbacks']:5d} {s['errors']:4d} {s['cost_usd']:7.2f}  {models}")
        if s["input_tokens"]:
            out.append(f"{'':28s} cache hit rate {100 * s['cache_read_tokens'] / s['input_tokens']:.0f}% of input tokens")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logs", default="logs", help="directory with *.jsonl decision logs and runtime-*.log files")
    parser.add_argument("--since", default=None, help="ISO date/time; ignore decisions before it")
    args = parser.parse_args(argv)
    logs = Path(args.logs)
    since = _ts(args.since) if args.since else None
    if since and since.tzinfo is None:
        since = since.replace(tzinfo=timezone.utc)
    sessions = load_decisions(logs, since)
    if not sessions:
        print(f"no decision logs found in {logs}/")
        return 0
    results = load_results(logs, sessions)
    print(render(summarize(sessions, results)))
    missing = [s for s in sessions if s not in results]
    if missing:
        print(f"\n{len(missing)} match(es) without a recorded result (run matches through scripts/run_*.sh so the runtime output is kept).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
