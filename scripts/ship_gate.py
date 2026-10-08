"""Ship gate: the sequential checks and paired series a build must pass before it becomes the default (REVIEW §6 G2).

Run from the integration worktree, one simulator at a time; it refuses to start while another harness or draft_eval
run is alive or while the lock file exists, and holds the lock itself while it runs.

    python scripts/ship_gate.py --quick --kind reliability --base <previous next tip> --mirror <worktree at that tip>
    python scripts/ship_gate.py --adoption --previous 48.3 --clone-latest ~/projects/opponents/<newest dated clone>
    python scripts/ship_gate.py --quick --mirror ../ace-trainer-wt-mirror --dry-run      # print the commands only
    python scripts/ship_gate.py --record-deploy 48.3                                      # log a deployed build's number

Stages (each appends one JSON line to --log, default ~/projects/ace-trainer-data/buildout/gate.jsonl, with the commit
from ``git rev-parse HEAD`` — a worktree's decision logs carry no commit of their own):
  1. checks: pytest -q -x; scripts/scenarios.py --code-only --set all; scripts/live_scenarios.py --code-only; the
     live-position test file when it exists. pytest must pass; the scenario counts must not drop below the previous
     gate run's (the first run records the baseline).
  2. greps over ``git diff <base>..HEAD`` and the commit messages for secrets and for the name patterns read from a
     file outside the repository (--name-patterns); the patterns themselves never appear here.
  3. --quick (~30 min): paired mirror vs ``checkout:<--mirror>`` (|point − 50| ≤ 4 for --kind reliability, point ≥ 50
     for --kind strength); paired vs the pinned external checkout through sim/shims/checkout_shim.py at a 2 s preview
     budget, seeded (tripwire: point ≥ the previous quick number for the same pin − 3); paired vs the Oct 7 worktree
     (point ≥ 85).
  4. --adoption (overnight): 500 pairs vs the pinned clone at a 15 s preview budget, seeded (Wilson lower bound ≥ the
     previous deployed build's point: --previous, else the log's last ``deploy`` entry); 500 pairs vs the latest
     dated clone when --clone-latest is given (tripwire vs the previous adoption number for that pin).
  5. a pass/fail table; exit status 1 on any failure, 3 when it refused to start.
Decision logs are off for every series (sim/draft_eval.py keeps them off). --dry-run prints every command and bar
without running anything or touching the lock.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
DATA = Path.home() / "projects" / "ace-trainer-data" / "buildout"
OPPONENTS = Path.home() / "projects" / "opponents"
SECRET_PATTERNS = ("eak_live_", "sk-ant-")
SIM_PATTERN = "sim/harness|draft_eval"
LIVE_POSITION_TESTS = "tests/test_replies_live.py"
QUICK_PREVIEW_BUDGET = 2.0
ADOPTION_PREVIEW_BUDGET = 15.0


def wilson(wins: float, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """(point, lower, upper) as fractions; the 95% Wilson interval (the same formula as sim/draft_eval.py)."""
    if n <= 0:
        return 0.0, 0.0, 0.0
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, centre - half, centre + half


# ---------------------------------------------------------------- small shell helpers (tests replace run_logged)
def sh(cmd: list[str], cwd: Path | None = None) -> str:
    """Output of a short read-only command ('' on failure)."""
    try:
        return subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def commit_of(path: Path | None = None) -> str:
    return sh(["git", "rev-parse", "--short", "HEAD"], cwd=path) or "unknown"


def running_simulators() -> list[str]:
    """Command lines of other harness/draft_eval processes (pgrep -f, this process excluded)."""
    out = sh(["pgrep", "-fl", SIM_PATTERN])
    me = str(os.getpid())
    return [line for line in out.splitlines() if line.strip() and line.split()[0] != me]


def run_logged(cmd: list[str], log_path: Path, env: dict | None = None) -> tuple[int, float, str]:
    """Run a stage command with its output appended to ``log_path``; (exit status, seconds, last lines)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(f"\n$ {' '.join(cmd)}\n")
        fh.flush()
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=fh, stderr=subprocess.STDOUT, text=True)
    seconds = time.monotonic() - t0
    tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-12:])
    return proc.returncode, seconds, tail


def dated_clones() -> list[Path]:
    """The external checkouts under ~/projects/opponents, oldest first (one dated directory per refresh)."""
    return sorted(p.parent.parent for p in OPPONENTS.glob("*/agent/agent.py")) if OPPONENTS.is_dir() else []


def pinned_clone() -> Path | None:
    """The clone the baselines were measured against: the first line of PINS.txt (``<sha> <date> <dir>``), else the oldest."""
    pins = OPPONENTS / "PINS.txt"
    if pins.exists():
        for line in pins.read_text(encoding="utf-8").splitlines():
            if line.strip():
                candidate = OPPONENTS / line.split()[-1]
                if (candidate / "agent" / "agent.py").exists():
                    return candidate
                break
    clones = dated_clones()
    return clones[0] if clones else None


def newest_clone() -> Path | None:
    clones = dated_clones()
    return clones[-1] if clones else None


# ---------------------------------------------------------------- the gate log
def load_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def append_log(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, default=str) + "\n")


def last_row(rows: list[dict], stage: str, **match) -> dict | None:
    for row in reversed(rows):
        if row.get("stage") == stage and all(row.get(k) == v for k, v in match.items()):
            return row
    return None


# ---------------------------------------------------------------- series and bars
@dataclass
class Series:
    stage: str
    a: str                      # the candidate's spec (sim/draft_eval.py spelling)
    b: str                      # the opponent's spec
    pairs: int
    bar: str                    # human-readable
    judge: Callable[[dict, list[dict]], tuple[bool | None, str]]  # (row, earlier log rows) -> (pass, note)
    env: dict = field(default_factory=dict)
    budgets: dict = field(default_factory=dict)
    opponent_commit: str = "unknown"


def within(points: float, target: float, tolerance: float) -> bool:
    return abs(points - target) <= tolerance


def judge_mirror(kind: str):
    def judge(row: dict, _rows: list[dict]) -> tuple[bool | None, str]:
        if kind == "reliability":
            return within(row["point"], 50.0, 4.0), f"|{row['point']:.1f} - 50| <= 4"
        return row["point"] >= 50.0, f"{row['point']:.1f} >= 50"
    return judge


def judge_floor(floor: float):
    def judge(row: dict, _rows: list[dict]) -> tuple[bool | None, str]:
        return row["point"] >= floor, f"{row['point']:.1f} >= {floor:g}"
    return judge


def judge_tripwire(stage: str, slack: float = 3.0):
    """point >= the previous number for the same stage and opponent pin, minus ``slack``; the first run is the baseline."""
    def judge(row: dict, rows: list[dict]) -> tuple[bool | None, str]:
        prev = last_row(rows, stage, clone_pin=row.get("clone_pin"))
        prev = prev if prev and prev.get("games") else None
        if prev is None:
            return None, "no earlier number for this pin: recorded as the baseline"
        return row["point"] >= prev["point"] - slack, f"{row['point']:.1f} >= {prev['point']:.1f} - {slack:g} (run of {prev.get('ts_text', '?')})"
    return judge


def judge_adoption(previous: float | None):
    def judge(row: dict, rows: list[dict]) -> tuple[bool | None, str]:
        ref = previous
        source = "--previous"
        if ref is None:
            deploy = last_row(rows, "deploy")
            if deploy is not None:
                ref, source = deploy.get("point"), f"deploy {deploy.get('commit')}"
        if ref is None:
            return None, "no deployed number to beat: recorded as the baseline"
        return row["lo"] >= ref, f"lower bound {row['lo']:.1f} >= {ref:.1f} ({source})"
    return judge


def shim_env(args, stage: str, preview_budget: float) -> dict:
    return {"SHIM_PREVIEW_BUDGET_S": f"{preview_budget:g}", "SHIM_SEED": str(args.seed),
            "SHIM_LOG_DIR": str(Path(args.work_dir) / f"shim-{stage.replace(':', '-')}")}


def clone_spec(path: Path, python: Path) -> str:
    return f"checkout:{path}:checkout_shim:{python}"


def plan(args) -> list[Series]:
    series: list[Series] = []
    if args.quick:
        series.append(Series("quick:mirror", args.candidate, f"checkout:{args.mirror}", args.pairs_mirror,
                             "|point - 50| <= 4" if args.kind == "reliability" else "point >= 50", judge_mirror(args.kind),
                             opponent_commit=commit_of(Path(args.mirror))))
        series.append(Series("quick:clone", args.candidate, clone_spec(args.clone, args.clone_python), args.pairs_clone,
                             "point >= previous quick number for this pin - 3", judge_tripwire("quick:clone"),
                             env=shim_env(args, "quick:clone", QUICK_PREVIEW_BUDGET), budgets={"preview": QUICK_PREVIEW_BUDGET},
                             opponent_commit=commit_of(args.clone)))
        series.append(Series("quick:old", args.candidate, f"checkout:{args.old}", args.pairs_old, "point >= 85", judge_floor(85.0),
                             opponent_commit=commit_of(Path(args.old))))
    if args.adoption:
        series.append(Series("adoption:clone", args.candidate, clone_spec(args.clone, args.clone_python), args.pairs_adoption,
                             "Wilson lower bound >= the previous deployed build's point", judge_adoption(args.previous),
                             env=shim_env(args, "adoption:clone", ADOPTION_PREVIEW_BUDGET), budgets={"preview": ADOPTION_PREVIEW_BUDGET},
                             opponent_commit=commit_of(args.clone)))
        if args.clone_latest:
            series.append(Series("adoption:latest", args.candidate, clone_spec(args.clone_latest, args.clone_python), args.pairs_adoption,
                                 "point >= previous adoption number for this pin - 3", judge_tripwire("adoption:latest"),
                                 env=shim_env(args, "adoption:latest", ADOPTION_PREVIEW_BUDGET), budgets={"preview": ADOPTION_PREVIEW_BUDGET},
                                 opponent_commit=commit_of(args.clone_latest)))
    return series


def series_command(s: Series, args, out_path: Path) -> list[str]:
    return [args.python, "sim/draft_eval.py", "match", s.a, s.b, "--pairs", str(s.pairs), "--seed", str(args.seed),
            "--out", str(out_path), "--progress", "25"]


def tally(out_path: Path, a: str) -> tuple[int, int, float | None]:
    """(decided games, A's wins, mean seconds per game) from a draft_eval --out file."""
    games = wins = 0
    seconds: list[float] = []
    if not out_path.exists():
        return 0, 0, None
    for line in out_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("seconds") is not None:
            seconds.append(float(rec["seconds"]))
        if rec.get("winner") is None:
            continue
        games += 1
        wins += rec["specs"][rec["winner"]] == a
    return games, wins, (round(sum(seconds) / len(seconds), 2) if seconds else None)


def run_series(s: Series, args, rows: list[dict], stamp: str) -> dict:
    work = Path(args.work_dir)
    out_path = work / f"{s.stage.replace(':', '-')}-{stamp}.jsonl"
    cmd = series_command(s, args, out_path)
    env = {**os.environ, **s.env}
    if s.env.get("SHIM_LOG_DIR"):
        shutil.rmtree(s.env["SHIM_LOG_DIR"], ignore_errors=True)
        Path(s.env["SHIM_LOG_DIR"]).mkdir(parents=True, exist_ok=True)
    rc, seconds, tail = run_logged(cmd, work / f"{s.stage.replace(':', '-')}-{stamp}.log", env=env)
    if s.env.get("SHIM_LOG_DIR"):
        shutil.rmtree(s.env["SHIM_LOG_DIR"], ignore_errors=True)
    games, wins, per_game = tally(out_path, s.a)
    p, lo, hi = wilson(wins, games)
    row = {"ts": time.time(), "ts_text": time.strftime("%Y-%m-%d %H:%M"), "stage": s.stage, "commit": args.commit, "base": args.base,
           "kind": args.kind, "spec": {"a": s.a, "b": s.b}, "opponent_commit": s.opponent_commit,
           "clone_pin": s.opponent_commit if "clone" in s.stage or "latest" in s.stage else None,
           "budgets": {"preview": s.budgets.get("preview"), "turn": s.budgets.get("turn")}, "seed": args.seed, "pairs": s.pairs,
           "games": games, "wins": wins, "point": round(100 * p, 2), "lo": round(100 * lo, 2), "hi": round(100 * hi, 2),
           "seconds_per_game": per_game, "seconds": round(seconds, 1), "exit": rc, "bar": s.bar, "out": str(out_path)}
    if games == 0:
        row["pass"], row["note"] = False, f"no decided games (exit {rc}); see the stage log"
    else:
        row["pass"], row["note"] = s.judge(row, rows)
    return row


# ---------------------------------------------------------------- stage 1 and 2
def check_commands(args) -> list[tuple[str, list[str]]]:
    py = args.python
    cmds = [("check:pytest", [py, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider"]),
            ("check:scenarios", [py, "scripts/scenarios.py", "--code-only", "--set", "all"]),
            ("check:live_scenarios", [py, "scripts/live_scenarios.py", "--code-only"])]
    if (ROOT / LIVE_POSITION_TESTS).exists():
        cmds.append(("check:live_positions", [py, "-m", "pytest", "-q", "-p", "no:cacheprovider", LIVE_POSITION_TESTS]))
    return cmds


def run_check(stage: str, cmd: list[str], args, rows: list[dict], stamp: str) -> dict:
    rc, seconds, tail = run_logged(cmd, Path(args.work_dir) / f"{stage.replace(':', '-')}-{stamp}.log")
    row = {"ts": time.time(), "ts_text": time.strftime("%Y-%m-%d %H:%M"), "stage": stage, "commit": args.commit, "base": args.base,
           "command": " ".join(cmd), "exit": rc, "seconds": round(seconds, 1), "note": tail.splitlines()[-1] if tail else ""}
    m = re.search(r"(\d+)/(\d+) passed", tail)
    if m and stage in ("check:scenarios", "check:live_scenarios"):
        row["passed"], row["total"] = int(m.group(1)), int(m.group(2))
        prev = last_row(rows, stage)
        if prev is not None and prev.get("passed") is not None:
            row["pass"] = row["passed"] >= prev["passed"]
            row["note"] = f"{row['passed']}/{row['total']} (previous {prev['passed']}/{prev.get('total')})"
        else:
            row["pass"] = True if rc == 0 else None
            row["note"] = f"{row['passed']}/{row['total']}" + ("" if rc == 0 else " (first run: recorded as the baseline)")
    else:
        row["pass"] = rc == 0
    return row


def read_patterns(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]


def grep_rows(args) -> list[dict]:
    """Secrets and opponent names in everything that is new since --base (the diff and the commit messages)."""
    texts = {"diff": sh(["git", "diff", f"{args.base}..HEAD"], cwd=ROOT),
             "log": sh(["git", "log", "--format=%h %s%n%b", f"{args.base}..HEAD"], cwd=ROOT)}
    rows = []
    for stage, patterns, flags in (("grep:secrets", list(SECRET_PATTERNS), 0),
                                   ("grep:names", read_patterns(Path(args.name_patterns)), re.IGNORECASE)):
        hits: list[str] = []
        for where, text in texts.items():
            for n, line in enumerate(text.splitlines(), 1):
                if any(re.search(pat, line, flags) for pat in patterns):
                    hits.append(f"{where}:{n}: {line.strip()[:100]}")
        rows.append({"ts": time.time(), "ts_text": time.strftime("%Y-%m-%d %H:%M"), "stage": stage, "commit": args.commit,
                     "base": args.base, "patterns": len(patterns), "hits": len(hits), "pass": not hits,
                     "note": "; ".join(hits[:5]) if hits else ("no pattern file" if not patterns else "clean")})
    return rows


# ---------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="the ~30 min tier: mirror, pinned clone at 2 s, the Oct 7 worktree")
    ap.add_argument("--adoption", action="store_true", help="the overnight tier: 500 pairs vs the pinned clone at 15 s (+ the latest clone)")
    ap.add_argument("--dry-run", action="store_true", help="print every command and bar; run nothing, touch no lock")
    ap.add_argument("--kind", choices=["reliability", "strength"], default="strength", help="what the merge claims (mirror bar)")
    ap.add_argument("--base", default="origin/pokemon-agent", help="commit the greps diff against (the previous next tip, or what is public)")
    ap.add_argument("--mirror", help="worktree checked out at the previous next tip (quick tier)")
    ap.add_argument("--old", default=str(Path.home() / "projects" / "ace-trainer-wt-old"), help="the Oct 7 worktree")
    ap.add_argument("--clone", type=Path, default=None, help="the pinned external checkout (default: the first line of PINS.txt, else the oldest dated one)")
    ap.add_argument("--clone-latest", type=Path, default=None, help="the latest dated external checkout (adoption tier; default: the newest dated one when it differs from the pin)")
    ap.add_argument("--clone-python", type=Path, default=OPPONENTS / "venv" / "bin" / "python")
    ap.add_argument("--candidate", default="agent.versions.code_only", help="the build under test, in sim/draft_eval.py spelling")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--pairs-mirror", type=int, default=300)
    ap.add_argument("--pairs-clone", type=int, default=150)
    ap.add_argument("--pairs-old", type=int, default=150)
    ap.add_argument("--pairs-adoption", type=int, default=500)
    ap.add_argument("--previous", type=float, default=None, help="the deployed build's point estimate vs the pinned clone (adoption bar)")
    ap.add_argument("--log", default=str(DATA / "gate.jsonl"))
    ap.add_argument("--lock", default=str(DATA / "SIM_LOCK"))
    ap.add_argument("--name-patterns", default=str(DATA / "name-patterns.txt"))
    ap.add_argument("--work-dir", default=str(ROOT / "logs" / "gate"), help="stage logs, --out files and shim scratch")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--skip-checks", action="store_true", help="stage 1 was run by hand already")
    ap.add_argument("--skip-greps", action="store_true")
    ap.add_argument("--mark-deployed", action="store_true", help="after a passing adoption run, log its clone number as the deployed one")
    ap.add_argument("--record-deploy", type=float, default=None, metavar="POINT", help="log HEAD as deployed with this point vs the pinned clone, then exit")
    return ap


def parse_args(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    if args.clone is None:  # both default from the opponents directory; an explicit --clone keeps --clone-latest explicit too
        args.clone = pinned_clone()
        latest = newest_clone()
        if args.adoption and args.clone_latest is None and args.clone is not None and latest is not None and latest != args.clone:
            args.clone_latest = latest
    for name in ("clone", "clone_latest", "clone_python"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, Path(value).expanduser())
    args.old = str(Path(args.old).expanduser())
    if args.mirror:
        args.mirror = str(Path(args.mirror).expanduser())
    if args.record_deploy is None and not (args.quick or args.adoption):
        raise SystemExit("choose a tier: --quick and/or --adoption (or --record-deploy POINT)")
    if args.quick and not args.mirror:
        raise SystemExit("--quick needs --mirror <worktree at the previous next tip>")
    if (args.quick or args.adoption) and args.clone is None:
        raise SystemExit("no external checkout found; pass --clone")
    return args


def print_plan(args, series: list[Series]) -> None:
    stamp = "<stamp>"
    print(f"ship gate dry run: commit {args.commit}, base {args.base}, kind {args.kind}, seed {args.seed}")
    if not args.skip_checks:
        for stage, cmd in check_commands(args):
            print(f"[{stage}] {' '.join(cmd)}")
    if not args.skip_greps:
        print(f"[grep:secrets] git diff/log {args.base}..HEAD for {len(SECRET_PATTERNS)} secret patterns")
        print(f"[grep:names] git diff/log {args.base}..HEAD for {len(read_patterns(Path(args.name_patterns)))} patterns from {args.name_patterns}")
    for s in series:
        env = " ".join(f"{k}={v}" for k, v in s.env.items())
        cmd = " ".join(series_command(s, args, Path(args.work_dir) / f"{s.stage.replace(':', '-')}-{stamp}.jsonl"))
        print(f"[{s.stage}] {env + ' ' if env else ''}{cmd}\n    bar: {s.bar}; opponent commit {s.opponent_commit}")


def print_table(rows: list[dict]) -> None:
    print("\n" + "-" * 100)
    for row in rows:
        verdict = {True: "PASS", False: "FAIL", None: "BASELINE"}[row.get("pass")]
        if "games" in row:
            detail = (f"{row['point']:.1f}% [{row['lo']:.1f}, {row['hi']:.1f}] over {row['games']} games, "
                      f"{row['seconds_per_game'] if row['seconds_per_game'] is not None else '?'} s/game; {row['note']}")
        else:
            detail = row.get("note", "")
        print(f"{row['stage']:22s} {verdict:8s} {detail}")
    print("-" * 100)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    os.chdir(ROOT)
    args.commit = commit_of(ROOT)
    log_path = Path(args.log).expanduser()
    if args.record_deploy is not None:
        row = {"ts": time.time(), "ts_text": time.strftime("%Y-%m-%d %H:%M"), "stage": "deploy", "commit": args.commit,
               "point": args.record_deploy, "clone_pin": commit_of(args.clone) if args.clone else None, "note": "recorded by hand"}
        append_log(log_path, row)
        print(f"logged deploy {args.commit}: {args.record_deploy:.1f}% vs the pinned clone")
        return 0
    series = plan(args)
    if args.dry_run:
        print_plan(args, series)
        return 0
    busy = running_simulators()
    if busy:
        print("another simulator is running; not starting:\n  " + "\n  ".join(busy))
        return 3
    lock = Path(args.lock).expanduser()
    if lock.exists():
        print(f"lock file {lock} exists ({lock.read_text(encoding='utf-8', errors='replace').strip()}); not starting")
        return 3
    lock.parent.mkdir(parents=True, exist_ok=True)
    tiers = "+".join(t for t, on in (("quick", args.quick), ("adoption", args.adoption)) if on)
    lock.write_text(f"{os.getpid()} {time.strftime('%H:%M')} ship_gate {tiers} {args.commit}\n", encoding="utf-8")
    Path(args.work_dir).mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    rows: list[dict] = []
    try:
        earlier = load_log(log_path)
        if not args.skip_checks:
            for stage, cmd in check_commands(args):
                row = run_check(stage, cmd, args, earlier, stamp)
                rows.append(row)
                append_log(log_path, row)
                print(f"{stage}: {'PASS' if row['pass'] else 'FAIL' if row['pass'] is False else 'BASELINE'} ({row['seconds']} s) {row['note']}", flush=True)
        if not args.skip_greps:
            for row in grep_rows(args):
                rows.append(row)
                append_log(log_path, row)
                print(f"{row['stage']}: {'PASS' if row['pass'] else 'FAIL'} ({row['hits']} hits)", flush=True)
        for s in series:
            print(f"{s.stage}: {s.pairs} pairs {s.a} vs {s.b} ...", flush=True)
            row = run_series(s, args, earlier, stamp)
            rows.append(row)
            append_log(log_path, row)
            earlier.append(row)
            verdict = {True: "PASS", False: "FAIL", None: "BASELINE"}[row["pass"]]
            print(f"{s.stage}: {verdict} {row['point']:.1f}% [{row['lo']:.1f}, {row['hi']:.1f}] over {row['games']} games "
                  f"in {row['seconds']} s ({row['note']})", flush=True)
            if args.mark_deployed and s.stage == "adoption:clone" and row["pass"]:
                deploy = {"ts": time.time(), "ts_text": row["ts_text"], "stage": "deploy", "commit": args.commit, "point": row["point"],
                          "lo": row["lo"], "hi": row["hi"], "clone_pin": row["clone_pin"], "note": "adoption tier passed"}
                append_log(log_path, deploy)
    finally:
        try:
            lock.unlink()
        except OSError:
            pass
    print_table(rows)
    failed = [r["stage"] for r in rows if r.get("pass") is False]
    print("GATE " + ("FAIL: " + ", ".join(failed) if failed else "PASS"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
