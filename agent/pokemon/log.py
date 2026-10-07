"""Decision log: one JSON line per decision, so we can ask "why did it do that?"

Each match writes ``logs/<session_id>.jsonl`` (directory from ``AGENT_LOG_DIR``,
default ``logs/``). A line is a flat JSON object: timestamp, session, ``kind``
(``draft`` / ``lineup`` / ``turn`` / ``fallback`` / ``error`` / ...) and whatever
fields the caller adds — the compact state, the computed turn sheet, the model's
answer, which model answered, latency, validation errors, the final payload.

Writing never raises: a logging problem must never cost a move. Anything that
looks like a key is redacted as a last line of defence (keys should never reach
this code in the first place).
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SECRET_PATTERNS = [
    re.compile(r"eak_live_[A-Za-z0-9_\-]+"),
    re.compile(r"sk-ant-[A-Za-z0-9_\-]+"),
    re.compile(r"sk_agent_[A-Za-z0-9_\-]+"),
    re.compile(r"seatclaim_[A-Za-z0-9_\-]+"),
]


def redact(text: str) -> str:
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text


class DecisionLog:
    def __init__(self, session_id: str, *, directory: str | os.PathLike | None = None, version: str = "") -> None:
        self.session_id = session_id
        self.version = version
        root = Path(directory or os.environ.get("AGENT_LOG_DIR") or "logs")
        self.path = root / f"{_safe(session_id)}.jsonl"
        self._broken = False
        try:
            root.mkdir(parents=True, exist_ok=True)
        except OSError:
            self._broken = True

    def write(self, kind: str, **fields: Any) -> None:
        if self._broken:
            return
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "session_id": self.session_id,
            "version": self.version,
            "kind": kind,
            **fields,
        }
        try:
            line = json.dumps(record, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            line = json.dumps({"ts": record["ts"], "session_id": self.session_id, "kind": kind, "unserializable": True})
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(redact(line) + "\n")
        except OSError:
            self._broken = True


def git_commit(repo_root: str | os.PathLike | None = None) -> str | None:
    """The checked-out commit, read from ``.git`` directly (no git binary needed). ``None`` outside a checkout.

    The Official Rules ask for records that connect the submitted commit ID to competitive play; the first line of
    every match log carries this value.
    """
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    try:
        head = (root / ".git" / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref:"):
            return head or None
        ref = head[4:].strip()
        ref_file = root / ".git" / ref
        if ref_file.exists():
            return ref_file.read_text(encoding="utf-8").strip() or None
        packed = root / ".git" / "packed-refs"
        if packed.exists():
            for line in packed.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[1] == ref:
                    return parts[0]
    except OSError:
        pass
    return None


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]", "_", name) or "unknown"
