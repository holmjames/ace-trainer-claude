"""sim/subprocess_player.py and sim/shims/checkout_shim.py: an agent from another checkout (another starter fork, with
its own interpreter) as a harness player, and sim/draft_eval.py taking such players. The fake checkouts live in
``tmp_path`` so everything but the last test runs with this interpreter alone; the last test drives the external
checkout (the Oct 7 opponent) through the shim when its venv exists."""

from __future__ import annotations

import json
import os
import random
import shutil
import sys
from pathlib import Path

import pytest

from altruagent import DecisionContext, WithReasoning
from altruagent.models import GameState
from sim import draft_eval, harness, subprocess_player, translate
from sim.subprocess_player import CheckoutPlayer, log_dir_for, parse_spec

ROOT = Path(__file__).resolve().parent.parent
LIVE = ROOT / "tests" / "fixtures" / "live"
ENGINE = ROOT / "sim" / "node_modules" / "pokemon-showdown"
PHASES = ("draft", "team_preview", "moving")

# External checkouts live beside their shared venv; the newest dated one is the default (name them by date).
EXTERNAL_ROOT = Path.home() / "projects" / "opponents"
EXTERNAL_PYTHON = EXTERNAL_ROOT / "venv" / "bin" / "python"

# A module whose agent prints on stdout (which must never reach the one-JSON-line protocol) and answers a draft pick
# with a reason, a Team Preview with a bare LegalAction, and a battle turn with an exception.
FAKE_AGENT = '''
import sys
from altruagent import WithReasoning


class Agent:
    def choose_action(self, state, context):
        print("chatter on stdout", flush=True)
        sys.stdout.write("more chatter\\n")
        first = state.legal_actions[0]
        if first.action_id.startswith("draft_pick:"):
            return WithReasoning({"type": "draft_pick", "card_id": first.action_id.split(":", 1)[1]}, "fake pick reason")
        if first.action_id == "select_lineup":
            return first
        raise ValueError("boom")


def create_agent():
    print("created", flush=True)
    return Agent()
'''

# A checkout shaped like an external starter fork: module globals for budgets and log paths read at call time, an
# unseeded ``rng`` on the agent, logs written next to the checkout unless redirected.
FAKE_CHECKOUT_AGENT = '''
import random
from pathlib import Path
from altruagent import WithReasoning

LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
CATALOG = LOG_DIR / "cards_seen.jsonl"
PREVIEW_BUDGET_S = 15.0
TURN_BUDGET_S = 8.0


class Agent:
    def __init__(self):
        self.rng = random.Random()

    def choose_action(self, state, context):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        (LOG_DIR / "touched").write_text("x")
        seen = {"turn": TURN_BUDGET_S, "preview": PREVIEW_BUDGET_S, "log_dir": str(LOG_DIR), "catalog": str(CATALOG),
                "rng": self.rng.random(), "global": random.random()}
        return WithReasoning({"type": "draft_pick", "card_id": state.legal_actions[0].action_id.split(":", 1)[1], "seen": seen},
                             "fake checkout reason")


def create_agent():
    return Agent()
'''


def external_checkout() -> Path | None:
    given = os.environ.get("EXTERNAL_CHECKOUT")
    if given:
        return Path(given).expanduser()
    pins = EXTERNAL_ROOT / "PINS.txt"  # the pinned clone first (the yardstick), else the newest dated clone
    if pins.exists():
        for line in pins.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 3 and (EXTERNAL_ROOT / parts[2] / "agent" / "agent.py").exists():
                return EXTERNAL_ROOT / parts[2]
    found = sorted(p.parent.parent for p in EXTERNAL_ROOT.glob("*/agent/agent.py")) if EXTERNAL_ROOT.is_dir() else []
    return found[-1] if found else None


def unwrap(decision):
    return decision.action if isinstance(decision, WithReasoning) else decision


@pytest.fixture(scope="module")
def live_states() -> dict[str, tuple[GameState, DecisionContext]]:
    """The first captured state of each phase from one live session (seat 0), as the runner would hand them over."""
    session = next(d for d in sorted(LIVE.iterdir()) if all(list(d.glob(f"{phase}-*-seat0.json")) for phase in PHASES))
    states = {}
    for phase in PHASES:
        raw = json.loads(sorted(session.glob(f"{phase}-*-seat0.json"))[0].read_text())
        actor = raw["current_actor"]
        ctx = DecisionContext(session_id=raw["session_id"], tournament_id=None, game_type=raw["game_type"],
                              agent_id=actor["agent_id"], seat_position=actor["position"])
        states[phase] = (GameState.from_mcp_state(raw), ctx)
    return states


@pytest.fixture
def logs_in_tmp(tmp_path, monkeypatch):
    """Keep the children's stderr logs out of logs/sim/ for the fake checkouts."""
    monkeypatch.setattr(subprocess_player, "ROOT", tmp_path)
    return tmp_path


def test_checkout_spec_fields_and_defaults(monkeypatch):
    monkeypatch.delenv("CHECKOUT_PYTHON", raising=False)
    assert parse_spec("checkout:../wt-old") == (Path("../wt-old"), "agent.versions.code_only", sys.executable)
    assert parse_spec("checkout:/a/b:agent.agent") == (Path("/a/b"), "agent.agent", sys.executable)
    assert parse_spec("checkout:/a/b:checkout_shim:/v/bin/python") == (Path("/a/b"), "checkout_shim", "/v/bin/python")
    assert parse_spec("checkout:/a/b::/v/bin/python")[1:] == ("agent.versions.code_only", "/v/bin/python")
    assert parse_spec("checkout:~/x:m:~/v/bin/python")[::2] == (Path.home() / "x", str(Path.home() / "v/bin/python"))
    monkeypatch.setenv("CHECKOUT_PYTHON", "/env/python")
    assert parse_spec("checkout:/a/b:agent.agent")[2] == "/env/python"
    assert parse_spec("checkout:/a/b:agent.agent:/explicit")[2] == "/explicit"
    with pytest.raises(ValueError):
        parse_spec("checkout:")
    # the stderr log sits in the same directory the harness names for the player's own logs
    assert log_dir_for("checkout:/a/b:m:/v/python") == subprocess_player.ROOT / "logs" / "sim" / "checkout__a_b_m__v_python"


def test_child_stdout_goes_to_the_log_and_reasons_come_back(logs_in_tmp, live_states):
    (logs_in_tmp / "fake_agent.py").write_text(FAKE_AGENT)
    spec = f"checkout:{logs_in_tmp}:fake_agent"
    player = subprocess_player.make(spec)
    assert isinstance(player, CheckoutPlayer) and player.python == sys.executable
    state, ctx = live_states["draft"]
    pick = player.choose_action(state, ctx)
    assert isinstance(pick, WithReasoning) and pick.reasoning_summary == "fake pick reason"
    assert pick.action == {"type": "draft_pick", "card_id": state.legal_actions[0].action_id.split(":", 1)[1]}
    state, ctx = live_states["team_preview"]
    lineup = player.choose_action(state, ctx)  # a bare LegalAction travels by id and comes back as this side's object
    assert not isinstance(lineup, WithReasoning) and lineup is state.legal_actions[0]
    state, ctx = live_states["moving"]
    with pytest.raises(RuntimeError, match="ValueError: boom"):
        player.choose_action(state, ctx)
    log = (log_dir_for(spec) / "child.stderr").read_text()
    assert "created" in log and "chatter on stdout" in log and "more chatter" in log
    # the same key reuses the child; a new game means a fresh agent
    again = subprocess_player.make(spec)
    assert again.proc is player.proc and (log_dir_for(spec) / "child.stderr").read_text().count("created") == 2


def test_shim_applies_budget_log_dir_and_seed_overrides(logs_in_tmp, monkeypatch, live_states):
    checkout = logs_in_tmp / "checkout"
    (checkout / "agent").mkdir(parents=True)
    (checkout / "agent" / "__init__.py").write_text("")
    (checkout / "agent" / "agent.py").write_text(FAKE_CHECKOUT_AGENT)
    scratch = logs_in_tmp / "scratch"
    monkeypatch.setenv("SHIM_TURN_BUDGET_S", "1.5")
    monkeypatch.setenv("SHIM_PREVIEW_BUDGET_S", "2")
    monkeypatch.setenv("SHIM_LOG_DIR", str(scratch))
    monkeypatch.setenv("SHIM_SEED", "7")
    player = subprocess_player.make(f"checkout:{checkout}:checkout_shim")
    state, ctx = live_states["draft"]
    pick = player.choose_action(state, ctx)
    assert pick.reasoning_summary == "fake checkout reason"
    seen = pick.action["seen"]
    assert (seen["turn"], seen["preview"]) == (1.5, 2.0)
    assert Path(seen["log_dir"]) == scratch and Path(seen["catalog"]) == scratch / "cards_seen.jsonl"
    assert (scratch / "touched").exists() and not (checkout / "logs").exists()
    assert seen["rng"] == random.Random(7).random() and seen["global"] == random.Random(7).random()
    player._call({"cmd": "new"})  # every game's fresh agent is seeded the same way
    assert player.choose_action(state, ctx).action["seen"]["rng"] == seen["rng"]


def test_shim_is_a_no_op_for_attributes_the_module_lacks(logs_in_tmp, monkeypatch, live_states):
    checkout = logs_in_tmp / "bare"
    (checkout / "agent").mkdir(parents=True)
    (checkout / "agent" / "__init__.py").write_text("")
    (checkout / "agent" / "agent.py").write_text(FAKE_AGENT)
    monkeypatch.setenv("SHIM_TURN_BUDGET_S", "1")
    monkeypatch.setenv("SHIM_SEED", "3")
    monkeypatch.delenv("SHIM_LOG_DIR", raising=False)
    player = subprocess_player.make(f"checkout:{checkout}:checkout_shim")
    state, ctx = live_states["draft"]
    assert player.choose_action(state, ctx).reasoning_summary == "fake pick reason"


def test_draft_eval_takes_checkout_and_module_specs_beside_the_named_players(logs_in_tmp, monkeypatch):
    assert draft_eval.parse_spec('v2l{"hp_units": 0.0}') == ("v2l", {**draft_eval.PLAYERS["v2l"], "hp_units": 0.0})
    assert draft_eval.parse_spec("checkout:/x/y:checkout_shim:/p/python") == ("checkout:/x/y:checkout_shim:/p/python", None)
    assert draft_eval.parse_spec("agent.versions.code_only") == ("agent.versions.code_only", None)
    with pytest.raises(SystemExit):
        draft_eval.parse_spec("v9")
    monkeypatch.delenv("AGENT_LOG_DIR", raising=False)
    (logs_in_tmp / "fake_agent.py").write_text(FAKE_AGENT)
    player = draft_eval.make_player(f"checkout:{logs_in_tmp}:fake_agent", seed=3)
    assert isinstance(player, CheckoutPlayer) and os.environ["AGENT_LOG_DIR"] == draft_eval.NO_LOG
    agent = draft_eval.make_player("agent.versions.code_only", seed=3)
    assert agent._log_dir == draft_eval.NO_LOG  # decision logs stay off for in-process module players too
    assert draft_eval.make_player("v2l", seed=3)._log_dir == draft_eval.NO_LOG


@pytest.mark.skipif(not ENGINE.exists() or shutil.which("node") is None, reason="needs Node and npm install in sim/")
def test_draft_eval_plays_a_checkout_player_and_records_seconds(logs_in_tmp, monkeypatch):
    """One paired-style game of our own code as a checkout child vs v2l: timings land in the record."""
    monkeypatch.delenv("AGENT_LOG_DIR", raising=False)
    spec = f"checkout:{ROOT}:agent.versions.code_only"
    bridge = harness.Bridge()
    try:
        result = draft_eval.play(bridge, "eval-test-0", 5, {"p1": spec, "p2": "v2l"}, {"p1": "A", "p2": "B"})
    finally:
        bridge.close()
    assert result["winner"] in ("A", "B", None) and result["turns"] >= 1
    assert result["seconds"] > 0 and set(result["seat_seconds"]) == {"p1", "p2"} and result["seat_seconds"]["p1"] > 0
    assert result["versions"]["p1"] == spec and len(result["rosters"]["p1"]) == 6
    # the child's draft reasons reach the result the way an in-process agent's do
    assert len(result["picks"]) == 12 and {p["side"] for p in result["picks"]} == {"p1", "p2"}
    assert all(p["public_reason"] for p in result["picks"] if p["side"] == "p1")

    class _Out:
        text = ""

        def write(self, s):
            self.text += s

        def flush(self):
            pass

    out = _Out()
    draft_eval.record(out, result, 5, {"p1": spec, "p2": "v2l"})
    line = json.loads(out.text)
    assert line["seconds"] == result["seconds"] and line["seat_seconds"] == result["seat_seconds"]
    assert line["picks"] == result["picks"]


@pytest.mark.skipif(not EXTERNAL_PYTHON.exists() or external_checkout() is None,
                    reason="needs the external checkout and its venv under ~/projects/opponents")
def test_external_checkout_answers_all_three_phases_through_the_shim(logs_in_tmp, monkeypatch, live_states):
    monkeypatch.setenv("SHIM_PREVIEW_BUDGET_S", "2")
    monkeypatch.setenv("SHIM_TURN_BUDGET_S", "2")
    monkeypatch.setenv("SHIM_LOG_DIR", str(logs_in_tmp / "scratch"))
    monkeypatch.setenv("SHIM_SEED", "1")
    spec = f"checkout:{external_checkout()}:checkout_shim:{EXTERNAL_PYTHON}"
    player = subprocess_player.make(spec)
    state, ctx = live_states["draft"]
    pick = player.choose_action(state, ctx)
    assert isinstance(pick, WithReasoning) and pick.reasoning_summary, "its draft reasons must reach public_reason"
    # An external starter fork may answer with a LegalAction (newer versions do) or with a payload dict.
    action_id = getattr(pick.action, "action_id", None) or f"draft_pick:{pick.action['card_id']}"
    assert action_id.startswith("draft_pick:") and action_id in {a.action_id for a in state.legal_actions}
    state, ctx = live_states["team_preview"]
    lineup = unwrap(player.choose_action(state, ctx))
    roster = state.legal_actions[0].input["action"]["roster"]
    assert lineup["type"] == "select_lineup" and len(lineup["bring"]) == 4 and set(lineup["bring"]) <= set(roster)
    assert len(lineup["leads"]) == 2 and set(lineup["leads"]) <= set(lineup["bring"])
    assert translate.preview_choice(lineup["bring"], lineup["leads"], roster)[0].startswith("team ")
    state, ctx = live_states["moving"]
    turn = unwrap(player.choose_action(state, ctx))
    assert turn["type"] == "doubles_turn" and {"slot_0", "slot_1"} <= set(turn)
    assert translate.to_choice(turn)
    assert "Traceback" not in (log_dir_for(spec) / "child.stderr").read_text()
    assert any((logs_in_tmp / "scratch").iterdir()), "its match logs must land in the redirected scratch directory"
