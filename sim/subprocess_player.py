"""Play an agent from ANOTHER checkout (an older commit, a worktree) inside the simulator.

Two versions of the ``agent`` package can't share one Python process, so this runs the other checkout's agent in a
child process and talks to it over stdin/stdout, one JSON line per decision. The harness hands it the same live-format
states it hands everyone else (``state.raw``, rebuilt with ``GameState.from_mcp_state`` on the other side).

    git worktree add ../ace-trainer-wt-old 4f2d0cb        # the version that lost 0-4 on Oct 7
    python sim/harness.py --games 200 --p1 code --p2 checkout:../ace-trainer-wt-old

Spec: ``checkout:<path>`` (code-only agent) or ``checkout:<path>:<module>`` (any module with ``create_agent()``).
One child per checkout for the whole run; every new game asks it for a fresh agent (fresh match memory).
"""

from __future__ import annotations

import atexit
import json
import subprocess
import sys
from pathlib import Path

from altruagent import LegalAction

# The child: imports the agent from ITS working directory (cwd first on sys.path), answers one decision per line.
CHILD = r"""
import json, sys, importlib
sys.path.insert(0, '.')
from altruagent import DecisionContext, LegalAction, WithReasoning
from altruagent.models import GameState
module = importlib.import_module(sys.argv[1])
agent = None
for line in sys.stdin:
    msg = json.loads(line)
    if msg['cmd'] == 'new':
        agent = module.create_agent()
        print(json.dumps({'ok': True}), flush=True)
        continue
    state = GameState.from_mcp_state(msg['state'])
    ctx = DecisionContext(**msg['ctx'])
    try:
        decision = agent.choose_action(state, ctx)
        if isinstance(decision, WithReasoning):
            decision = decision.action
        if isinstance(decision, LegalAction):
            out = {'legal_action_id': decision.action_id}
        else:
            out = {'payload': decision}
    except Exception as exc:  # report, let the harness fall back
        out = {'error': f'{type(exc).__name__}: {exc}'}
    print(json.dumps(out, default=str), flush=True)
"""

_children: dict[tuple[str, str], subprocess.Popen] = {}


def _child(checkout: Path, module: str) -> subprocess.Popen:
    key = (str(checkout), module)
    proc = _children.get(key)
    if proc is None or proc.poll() is not None:
        proc = subprocess.Popen([sys.executable, "-u", "-c", CHILD, module], cwd=str(checkout), stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        _children[key] = proc
        atexit.register(proc.kill)
    return proc


class CheckoutPlayer:
    """An agent from another checkout, one fresh agent per game."""

    def __init__(self, checkout: str, module: str = "agent.versions.code_only") -> None:
        self.proc = _child(Path(checkout).resolve(), module)
        self._call({"cmd": "new"})

    def _call(self, msg: dict) -> dict:
        assert self.proc.stdin and self.proc.stdout
        self.proc.stdin.write(json.dumps(msg, default=str) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("checkout player exited")
        return json.loads(line)

    def choose_action(self, state, context):
        ctx = {"session_id": context.session_id, "tournament_id": context.tournament_id, "game_type": context.game_type,
               "agent_id": context.agent_id, "seat_position": context.seat_position}
        reply = self._call({"cmd": "act", "state": state.raw, "ctx": ctx})
        if "legal_action_id" in reply:
            return next(a for a in state.legal_actions if a.action_id == reply["legal_action_id"])
        if "payload" in reply:
            return reply["payload"]
        raise RuntimeError(reply.get("error") or "no decision")


def make(spec: str) -> CheckoutPlayer:
    _, _, rest = spec.partition("checkout:")
    path, _, module = rest.partition(":")
    return CheckoutPlayer(path, module or "agent.versions.code_only")


__all__ = ["CheckoutPlayer", "make", "LegalAction"]
