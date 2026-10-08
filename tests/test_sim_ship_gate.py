"""scripts/ship_gate.py: argument parsing, the log line schema, the bar arithmetic, the lock refusal and the dry run.

No series is ever started here: the stage runner (``run_logged``) is replaced by a fake that writes the ``--out`` file a
sim/draft_eval.py run would have written, and the simulator check (``running_simulators``) is replaced too."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("ship_gate", ROOT / "scripts" / "ship_gate.py")
ship_gate = importlib.util.module_from_spec(_spec)
sys.modules["ship_gate"] = ship_gate
_spec.loader.exec_module(ship_gate)

FAKE_PATTERN = "zzzfakeopponentzzz"


@pytest.fixture
def gate(tmp_path, monkeypatch):
    """Paths for one gate run: fake checkouts (plain directories), a pattern file, log, lock and work dir."""
    paths = {}
    for name in ("mirror", "old", "clone", "latest"):
        (tmp_path / name).mkdir()
        paths[name] = tmp_path / name
    python = tmp_path / "python"
    python.write_text("")
    patterns = tmp_path / "name-patterns.txt"
    patterns.write_text(f"# patterns\n{FAKE_PATTERN}\n")
    paths.update(python=python, patterns=patterns, log=tmp_path / "gate.jsonl", lock=tmp_path / "LOCK", work=tmp_path / "work")
    monkeypatch.setattr(ship_gate, "running_simulators", lambda: [])
    return paths


def base_args(g, *extra: str) -> list[str]:
    return ["--clone", str(g["clone"]), "--clone-python", str(g["python"]), "--old", str(g["old"]), "--name-patterns", str(g["patterns"]),
            "--log", str(g["log"]), "--lock", str(g["lock"]), "--work-dir", str(g["work"]), "--skip-greps", *extra]


class FakeRunner:
    """Stands in for run_logged: answers check commands with canned output and series commands with a --out file."""

    def __init__(self, series: dict[str, tuple[int, int]], checks: dict[str, tuple[int, str]] | None = None) -> None:
        self.series = series      # opponent-spec substring -> (A's wins, decided games)
        self.checks = checks or {}
        self.calls: list[tuple[list[str], dict | None]] = []

    def __call__(self, cmd: list[str], log_path: Path, env: dict | None = None) -> tuple[int, float, str]:
        self.calls.append((cmd, env))
        if "sim/draft_eval.py" in cmd:
            a, b = cmd[3], cmd[4]
            wins, games = next(v for k, v in self.series.items() if k in b)
            out = Path(cmd[cmd.index("--out") + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w") as fh:
                for i in range(games):
                    fh.write(json.dumps({"seed": i, "specs": {"p1": a, "p2": b}, "first": "p1", "winner": "p1" if i < wins else "p2",
                                         "turns": 7, "seconds": 0.5, "seat_seconds": {"p1": 0.2, "p2": 0.2}}) + "\n")
                fh.write(json.dumps({"seed": games, "specs": {"p1": a, "p2": b}, "first": "p1", "winner": None, "turns": 120, "seconds": 9.0}) + "\n")
            return 0, 12.0, f"{a} vs {b}: done"
        for key, (rc, tail) in self.checks.items():
            if any(key in part for part in cmd):
                return rc, 1.0, tail
        return 0, 1.0, "ok"


def test_parse_args_tiers_and_defaults(gate):
    g = gate
    args = ship_gate.parse_args(base_args(g, "--quick", "--mirror", str(g["mirror"]), "--kind", "reliability"))
    assert (args.quick, args.adoption, args.kind) == (True, False, "reliability")
    assert (args.pairs_mirror, args.pairs_clone, args.pairs_old, args.pairs_adoption) == (300, 150, 150, 500)
    assert args.candidate == "agent.versions.code_only" and args.seed == 1 and args.clone == g["clone"]
    series = ship_gate.plan(args)
    assert [s.stage for s in series] == ["quick:mirror", "quick:clone", "quick:old"]
    assert series[1].b == f"checkout:{g['clone']}:checkout_shim:{g['python']}"
    assert series[1].env["SHIM_PREVIEW_BUDGET_S"] == "2" and series[1].env["SHIM_SEED"] == "1"
    adoption = ship_gate.plan(ship_gate.parse_args(base_args(g, "--adoption", "--clone-latest", str(g["latest"]), "--previous", "48.3")))
    assert [s.stage for s in adoption] == ["adoption:clone", "adoption:latest"]
    assert adoption[0].env["SHIM_PREVIEW_BUDGET_S"] == "15" and adoption[0].pairs == 500
    with pytest.raises(SystemExit):
        ship_gate.parse_args(base_args(g, "--quick"))  # no --mirror
    with pytest.raises(SystemExit):
        ship_gate.parse_args(base_args(g))  # no tier


def test_wilson_and_the_bars():
    p, lo, hi = ship_gate.wilson(312, 600)
    assert p == pytest.approx(0.52) and lo == pytest.approx(0.4802, abs=1e-3) and hi == pytest.approx(0.5593, abs=1e-3)
    assert ship_gate.wilson(0, 0) == (0.0, 0.0, 0.0)
    reliability, strength = ship_gate.judge_mirror("reliability"), ship_gate.judge_mirror("strength")
    assert reliability({"point": 53.9}, [])[0] is True and reliability({"point": 45.0}, [])[0] is False
    assert strength({"point": 50.0}, [])[0] is True and strength({"point": 49.9}, [])[0] is False
    assert ship_gate.judge_floor(85.0)({"point": 85.0}, [])[0] is True and ship_gate.judge_floor(85.0)({"point": 84.9}, [])[0] is False
    trip = ship_gate.judge_tripwire("quick:clone")
    assert trip({"point": 40.0, "clone_pin": "abc"}, [])[0] is None  # nothing earlier: a baseline, not a failure
    earlier = [{"stage": "quick:clone", "clone_pin": "abc", "games": 300, "point": 60.0}, {"stage": "quick:clone", "clone_pin": "zzz", "games": 300, "point": 90.0}]
    assert trip({"point": 57.0, "clone_pin": "abc"}, earlier)[0] is True
    assert trip({"point": 56.9, "clone_pin": "abc"}, earlier)[0] is False
    adopt = ship_gate.judge_adoption(None)
    assert adopt({"lo": 52.9}, [])[0] is None
    assert adopt({"lo": 52.9}, [{"stage": "deploy", "commit": "c0", "point": 52.0}])[0] is True
    assert adopt({"lo": 51.9}, [{"stage": "deploy", "commit": "c0", "point": 52.0}])[0] is False
    assert ship_gate.judge_adoption(55.0)({"lo": 54.9}, [])[0] is False


def test_tally_counts_decided_games_only(tmp_path):
    out = tmp_path / "x.jsonl"
    out.write_text("\n".join(json.dumps(r) for r in [
        {"specs": {"p1": "A", "p2": "B"}, "winner": "p1", "seconds": 1.0},
        {"specs": {"p1": "B", "p2": "A"}, "winner": "p1", "seconds": 3.0},
        {"specs": {"p1": "A", "p2": "B"}, "winner": None, "seconds": 5.0}]) + "\n")
    assert ship_gate.tally(out, "A") == (2, 1, 3.0)
    assert ship_gate.tally(tmp_path / "missing.jsonl", "A") == (0, 0, None)


def test_dry_run_prints_every_command_and_runs_nothing(gate, monkeypatch, capsys):
    g = gate

    def boom(*_a, **_k):
        raise AssertionError("a stage ran during --dry-run")

    monkeypatch.setattr(ship_gate, "run_logged", boom)
    monkeypatch.setattr(ship_gate, "running_simulators", boom)
    rc = ship_gate.main(base_args(g, "--quick", "--adoption", "--mirror", str(g["mirror"]), "--clone-latest", str(g["latest"]), "--dry-run"))
    out = capsys.readouterr().out
    assert rc == 0 and not g["lock"].exists() and not g["log"].exists()
    assert "pytest -q -x -p no:cacheprovider" in out and "scripts/scenarios.py --code-only --set all" in out
    assert f"sim/draft_eval.py match agent.versions.code_only checkout:{g['mirror']} --pairs 300" in out
    assert f"SHIM_PREVIEW_BUDGET_S=2 SHIM_SEED=1" in out and f"checkout:{g['clone']}:checkout_shim:{g['python']} --pairs 150" in out
    assert "SHIM_PREVIEW_BUDGET_S=15" in out and f"checkout:{g['latest']}:checkout_shim:{g['python']} --pairs 500" in out
    assert "bar: |point - 50| <= 4" not in out and "bar: point >= 50" in out  # --kind defaults to strength
    assert FAKE_PATTERN not in out  # the patterns themselves are never printed


def test_refuses_while_another_simulator_runs_or_the_lock_exists(gate, monkeypatch, capsys):
    g = gate
    runner = FakeRunner({"mirror": (300, 600)})
    monkeypatch.setattr(ship_gate, "run_logged", runner)
    monkeypatch.setattr(ship_gate, "running_simulators", lambda: ["4242 python sim/draft_eval.py match v2l checkout:x"])
    assert ship_gate.main(base_args(g, "--quick", "--mirror", str(g["mirror"]), "--skip-checks")) == 3
    assert "not starting" in capsys.readouterr().out and not runner.calls and not g["lock"].exists()
    monkeypatch.setattr(ship_gate, "running_simulators", lambda: [])
    g["lock"].write_text("65307 13:54 baseline\n")
    assert ship_gate.main(base_args(g, "--quick", "--mirror", str(g["mirror"]), "--skip-checks")) == 3
    assert not runner.calls and g["lock"].read_text() == "65307 13:54 baseline\n"  # someone else's lock is left alone


def test_quick_tier_log_lines_and_verdicts(gate, monkeypatch, capsys):
    g = gate
    runner = FakeRunner({"mirror": (306, 600), "checkout_shim": (120, 300), "old": (270, 300)},
                        checks={"pytest": (0, "1733 passed, 3 skipped in 24.9s"), "scenarios.py": (1, "21/22 passed"), "live_scenarios.py": (0, "7/7 passed")})
    monkeypatch.setattr(ship_gate, "run_logged", runner)
    rc = ship_gate.main(base_args(g, "--quick", "--kind", "reliability", "--mirror", str(g["mirror"]), "--seed", "7"))
    out = capsys.readouterr().out
    assert rc == 0 and "GATE PASS" in out and not g["lock"].exists()
    rows = {r["stage"]: r for r in ship_gate.load_log(g["log"])}
    assert list(rows) == ["check:pytest", "check:scenarios", "check:live_scenarios", "quick:mirror", "quick:clone", "quick:old"]
    assert rows["check:pytest"]["pass"] is True and rows["check:scenarios"]["pass"] is None  # a failing scenario on the first run is the baseline
    assert (rows["check:scenarios"]["passed"], rows["check:scenarios"]["total"]) == (21, 22)
    mirror = rows["quick:mirror"]
    assert {"ts", "stage", "commit", "base", "kind", "spec", "budgets", "seed", "clone_pin", "pairs", "games", "wins", "point", "lo", "hi",
            "seconds_per_game", "seconds", "bar", "pass", "note"} <= set(mirror)
    assert (mirror["games"], mirror["wins"], mirror["point"], mirror["seed"], mirror["pairs"]) == (600, 306, 51.0, 7, 300)
    assert mirror["lo"] == pytest.approx(47.0, abs=0.1) and mirror["hi"] == pytest.approx(55.0, abs=0.1) and mirror["seconds_per_game"] == pytest.approx(0.51, abs=0.01)
    assert mirror["pass"] is True and mirror["kind"] == "reliability" and mirror["spec"] == {"a": "agent.versions.code_only", "b": f"checkout:{g['mirror']}"}
    clone = rows["quick:clone"]
    assert clone["budgets"] == {"preview": 2.0, "turn": None} and clone["pass"] is None and "baseline" in clone["note"]
    assert rows["quick:old"]["pass"] is True and rows["quick:old"]["point"] == 90.0
    series_calls = [(cmd, env) for cmd, env in runner.calls if "sim/draft_eval.py" in cmd]
    assert len(series_calls) == 3 and series_calls[1][1]["SHIM_PREVIEW_BUDGET_S"] == "2" and series_calls[1][1]["SHIM_SEED"] == "7"
    assert all("--seed" in cmd and cmd[cmd.index("--seed") + 1] == "7" for cmd, _ in series_calls)


def test_tripwires_and_the_adoption_bar_read_the_log(gate, monkeypatch, capsys):
    g = gate
    ship_gate.append_log(g["log"], {"stage": "quick:clone", "clone_pin": "unknown", "games": 300, "point": 60.0, "ts_text": "earlier"})
    ship_gate.append_log(g["log"], {"stage": "check:scenarios", "passed": 21, "total": 22})
    runner = FakeRunner({"mirror": (300, 600), "checkout_shim": (168, 300), "old": (270, 300)},
                        checks={"scenarios.py": (1, "20/22 passed"), "live_scenarios.py": (0, "7/7 passed")})
    monkeypatch.setattr(ship_gate, "run_logged", runner)
    rc = ship_gate.main(base_args(g, "--quick", "--mirror", str(g["mirror"])))
    out = capsys.readouterr().out
    rows = ship_gate.load_log(g["log"])
    assert rc == 1 and "GATE FAIL: check:scenarios, quick:clone" in out
    assert [r for r in rows if r["stage"] == "quick:clone"][-1]["pass"] is False  # 56 < 60 - 3
    assert [r for r in rows if r["stage"] == "check:scenarios"][-1]["pass"] is False  # 20 < 21
    # adoption: the lower bound must clear the deployed number; a pass with --mark-deployed logs the new deploy
    runner = FakeRunner({"checkout_shim": (560, 1000)})
    monkeypatch.setattr(ship_gate, "run_logged", runner)
    assert ship_gate.main(base_args(g, "--adoption", "--skip-checks", "--previous", "55")) == 1
    assert ship_gate.main(base_args(g, "--adoption", "--skip-checks", "--previous", "52", "--mark-deployed")) == 0
    capsys.readouterr()
    deploy = ship_gate.last_row(ship_gate.load_log(g["log"]), "deploy")
    assert deploy and deploy["point"] == 56.0 and deploy["lo"] == pytest.approx(52.9, abs=0.1)
    assert ship_gate.main(base_args(g, "--adoption", "--skip-checks")) == 1  # without --previous the deploy row (56.0) is the bar
    rows = ship_gate.load_log(g["log"])
    assert rows[-1]["stage"] == "adoption:clone" and rows[-1]["pass"] is False and "deploy" in rows[-1]["note"]


def test_record_deploy_writes_one_row(gate, capsys):
    g = gate
    assert ship_gate.main(base_args(g, "--record-deploy", "48.3")) == 0
    rows = ship_gate.load_log(g["log"])
    assert len(rows) == 1 and rows[0]["stage"] == "deploy" and rows[0]["point"] == 48.3 and rows[0]["commit"]


def test_greps_find_secrets_and_name_patterns_in_the_diff_and_the_messages(gate, tmp_path, monkeypatch):
    g = gate
    repo = tmp_path / "repo"
    repo.mkdir()
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@x", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@x", "HOME": str(tmp_path)}

    def git(*args):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, env={**env, "PATH": "/usr/bin:/bin:/usr/local/bin"})

    git("init", "-q")
    (repo / "a.txt").write_text("clean\n")
    git("add", "a.txt")
    git("commit", "-q", "-m", "base")
    (repo / "a.txt").write_text(f"clean\nkey = 'sk-ant-abcdef'\nvs {FAKE_PATTERN.upper()} today\n")
    git("commit", "-q", "-am", f"mention {FAKE_PATTERN} in a message")
    monkeypatch.setattr(ship_gate, "ROOT", repo)
    args = ship_gate.parse_args(base_args(g, "--adoption", "--base", "HEAD~1"))
    args.commit = "test"
    rows = {r["stage"]: r for r in ship_gate.grep_rows(args)}
    assert rows["grep:secrets"]["hits"] == 1 and rows["grep:secrets"]["pass"] is False
    assert rows["grep:names"]["hits"] == 2 and rows["grep:names"]["pass"] is False and rows["grep:names"]["patterns"] == 1
    args = ship_gate.parse_args(base_args(g, "--adoption", "--base", "HEAD"))
    args.commit = "test"
    assert all(r["pass"] for r in ship_gate.grep_rows(args))
