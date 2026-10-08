"""Play an agent from ANOTHER checkout (an older commit, a worktree, an external starter fork) inside the simulator.

Two versions of the ``agent`` package can't share one Python process, so this runs the other checkout's agent in a
child process and talks to it over stdin/stdout, one JSON line per decision. The harness hands it the same live-format
states it hands everyone else (``state.raw``, rebuilt with ``GameState.from_mcp_state`` on the other side).

    git worktree add ../ace-trainer-wt-old 4f2d0cb        # the version that lost 0-4 on Oct 7
    python sim/harness.py --games 200 --p1 code --p2 checkout:../ace-trainer-wt-old
    python sim/harness.py --games 20 --p1 code --p2 checkout:<external checkout>:checkout_shim:<its venv python>

Spec: ``checkout:<path>[:<module>[:<python>]]``. ``<module>`` is any module with ``create_agent()`` (default
``agent.versions.code_only``); ``<python>`` is the interpreter the child runs under (default ``$CHECKOUT_PYTHON``,
else this one), for checkouts whose dependencies live in their own venv. ``sim/shims/`` is on the child's
``PYTHONPATH``, so a shim there can be named as ``<module>`` (``checkout_shim`` wraps an external checkout's
``agent.agent`` with budget/log-dir/seed overrides from the environment).

In the child, ``sys.stdout`` is pointed at stderr before the module is imported, so an agent that prints cannot
corrupt the one-JSON-line protocol; everything it prints lands in ``logs/sim/<spec>/child.stderr`` (appended) next to
the harness's own logs for that player. A ``WithReasoning`` answer travels as ``{"payload"|"legal_action_id", "reason"}``
and is rebuilt on this side, so the harness records the child's ``public_reason`` like everyone else's.

One child per (checkout, module, python) for the whole run; every new game asks it for a fresh agent (fresh memory).
"""

from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
from pathlib import Path

from altruagent import LegalAction, WithReasoning

ROOT = Path(__file__).resolve().parent.parent
SHIMS = ROOT / "sim" / "shims"
DEFAULT_MODULE = "agent.versions.code_only"

# The child: imports the agent from ITS working directory (cwd first on sys.path), answers one decision per line on
# the real stdout; its own stdout is stderr from before the import, so the agent's prints never reach the protocol.
CHILD = r"""
import json, sys, importlib
out_fh = sys.stdout
sys.stdout = sys.stderr
sys.path.insert(0, '.')
from altruagent import DecisionContext, LegalAction, WithReasoning
from altruagent.models import GameState
module = importlib.import_module(sys.argv[1])
agent = None
def reply(out):
    out_fh.write(json.dumps(out, default=str) + '\n')
    out_fh.flush()
for line in sys.stdin:
    msg = json.loads(line)
    if msg['cmd'] == 'new':
        agent = module.create_agent()
        reply({'ok': True})
        continue
    state = GameState.from_mcp_state(msg['state'])
    ctx = DecisionContext(**msg['ctx'])
    try:
        decision = agent.choose_action(state, ctx)
        reason = None
        if isinstance(decision, WithReasoning):
            reason = decision.reasoning_summary
            decision = decision.action
        if isinstance(decision, LegalAction):
            out = {'legal_action_id': decision.action_id}
        else:
            out = {'payload': decision}
        if reason:
            out['reason'] = reason
    except Exception as exc:  # report, let the harness fall back
        out = {'error': f'{type(exc).__name__}: {exc}'}
    reply(out)
"""

_children: dict[tuple[str, str, str], subprocess.Popen] = {}


def log_dir_for(spec: str) -> Path:
    """Where the harness keeps this player's logs (the same spelling sim/harness.py uses for its decision logs)."""
    return ROOT / "logs" / "sim" / spec.replace(":", "_").replace("/", "_")


def parse_spec(spec: str) -> tuple[Path, str, str]:
    """``checkout:<path>[:<module>[:<python>]]`` -> (checkout path, module, interpreter). Missing fields fall back to
    ``agent.versions.code_only`` and ``$CHECKOUT_PYTHON`` (else the running interpreter)."""
    _, _, rest = spec.partition("checkout:")
    path, _, rest = rest.partition(":")
    module, _, python = rest.partition(":")
    if not path:
        raise ValueError(f"checkout spec without a path: {spec!r}")
    python = python or os.environ.get("CHECKOUT_PYTHON") or sys.executable
    return Path(path).expanduser(), module or DEFAULT_MODULE, str(Path(python).expanduser())


def _child(checkout: Path, module: str, python: str, log_dir: Path) -> subprocess.Popen:
    key = (str(checkout), module, python)
    proc = _children.get(key)
    if proc is None or proc.poll() is not None:
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(p for p in (str(SHIMS), env.get("PYTHONPATH", "")) if p)
        log_dir.mkdir(parents=True, exist_ok=True)
        with (log_dir / "child.stderr").open("a", encoding="utf-8") as err:  # the child keeps its own copy of the handle
            proc = subprocess.Popen([python, "-u", "-c", CHILD, module], cwd=str(checkout), env=env, stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE, stderr=err, text=True, bufsize=1)
        _children[key] = proc
        atexit.register(proc.kill)
    return proc


class CheckoutPlayer:
    """An agent from another checkout, one fresh agent per game."""

    def __init__(self, checkout: str | os.PathLike, module: str = DEFAULT_MODULE, python: str | None = None,
                 spec: str | None = None) -> None:
        self.checkout = Path(checkout).expanduser().resolve()
        self.module = module
        self.python = str(Path(python).expanduser()) if python else os.environ.get("CHECKOUT_PYTHON") or sys.executable
        self.spec = spec or f"checkout:{checkout}:{module}" + (f":{python}" if python else "")
        self.proc = _child(self.checkout, module, self.python, log_dir_for(self.spec))
        self._call({"cmd": "new"})

    def _call(self, msg: dict) -> dict:
        assert self.proc.stdin and self.proc.stdout
        self.proc.stdin.write(json.dumps(msg, default=str) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"checkout player exited (see {log_dir_for(self.spec) / 'child.stderr'})")
        return json.loads(line)

    def choose_action(self, state, context):
        ctx = {"session_id": context.session_id, "tournament_id": context.tournament_id, "game_type": context.game_type,
               "agent_id": context.agent_id, "seat_position": context.seat_position}
        reply = self._call({"cmd": "act", "state": state.raw, "ctx": ctx})
        if "legal_action_id" in reply:
            decision = next(a for a in state.legal_actions if a.action_id == reply["legal_action_id"])
        elif "payload" in reply:
            decision = reply["payload"]
        else:
            raise RuntimeError(reply.get("error") or "no decision")
        return WithReasoning(decision, reply["reason"]) if reply.get("reason") else decision


def make(spec: str) -> CheckoutPlayer:
    path, module, python = parse_spec(spec)
    return CheckoutPlayer(path, module, python, spec=spec)


__all__ = ["CheckoutPlayer", "make", "parse_spec", "log_dir_for", "LegalAction"]
