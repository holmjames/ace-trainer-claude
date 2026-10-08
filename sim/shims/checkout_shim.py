"""Run an external checkout's ``agent.agent`` inside the simulator's checkout child, with overrides from the environment.

Name it as the module of a ``checkout:`` spec, ``checkout:<path>:checkout_shim[:<python>]``: sim/subprocess_player.py
puts this directory on the child's PYTHONPATH and the child's cwd is the checkout, so ``agent.agent`` below is the
checkout's own package, imported untouched. Every override is optional and a no-op for an attribute the module lacks:

    SHIM_TURN_BUDGET_S, SHIM_PREVIEW_BUDGET_S   module globals ``TURN_BUDGET_S`` / ``PREVIEW_BUDGET_S`` (seconds)
    SHIM_LOG_DIR, SHIM_CATALOG                  module globals ``LOG_DIR`` / ``CATALOG``; without them the module's
                                                log directory is redirected to a fresh ``$TMPDIR/checkout-shim-<pid>``
                                                (a checkout that reloads a prior match file when a session id repeats
                                                must never see an earlier run's files) and the catalog is kept inside it
    SHIM_SEED                                   seeds ``random`` and the created agent's ``rng`` attribute, if it has
                                                one, the same way for every game of the run

``create_agent()`` is what the child calls once per game.
"""

from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path

import agent.agent as target  # the checkout's own agent (its cwd is first on sys.path in the child)

BUDGETS = {"SHIM_TURN_BUDGET_S": "TURN_BUDGET_S", "SHIM_PREVIEW_BUDGET_S": "PREVIEW_BUDGET_S"}


def scratch_dir() -> Path:
    base = os.environ.get("SHIM_LOG_DIR") or os.path.join(tempfile.gettempdir(), f"checkout-shim-{os.getpid()}")
    path = Path(base).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def apply_overrides(module=target) -> dict:
    """Set the module globals the environment asks for; returns what changed (for logs and tests)."""
    applied: dict = {}
    for env, attr in BUDGETS.items():
        value = os.environ.get(env)
        if value and hasattr(module, attr):
            setattr(module, attr, float(value))
            applied[attr] = float(value)
    if hasattr(module, "LOG_DIR"):
        module.LOG_DIR = scratch_dir()
        applied["LOG_DIR"] = str(module.LOG_DIR)
        if hasattr(module, "CATALOG"):
            module.CATALOG = Path(os.environ.get("SHIM_CATALOG") or module.LOG_DIR / "cards_seen.jsonl").expanduser()
            applied["CATALOG"] = str(module.CATALOG)
    return applied


def seed() -> int | None:
    value = os.environ.get("SHIM_SEED")
    return int(value) if value not in (None, "") else None


def create_agent():
    applied = apply_overrides()
    s = seed()
    if s is not None:
        random.seed(s)
    created = target.create_agent()
    if s is not None and hasattr(created, "rng"):
        created.rng = random.Random(s)
    if applied or s is not None:
        print(f"[checkout_shim] overrides {applied} seed {s}", flush=True)  # stdout is the child's stderr log
    return created


__all__ = ["create_agent", "apply_overrides", "scratch_dir", "seed"]
