#!/usr/bin/env bash
# Play ONE Testing seat with the current repo's runtime and tee the output to logs/ so
# scripts/tally.py can read the result.
#
#   ./scripts/run_match.sh seatclaim_...                   # today's repo (claim codes)
#   ./scripts/run_match.sh seatclaim_... agent.arena       # a different agent factory module
#
# After upstream PR #5 lands, Testing seats are played by `python -m agent --match` instead;
# this script then just forwards to that: ./scripts/run_match.sh --match [--agent MODULE]
set -u
cd "$(dirname "$0")/.."
mkdir -p logs
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="logs/runtime-match-${STAMP}.log"
PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"

if [ "${1:-}" = "--match" ]; then
  shift
  exec caffeinate -dims "$PY" -m agent --match "$@" 2>&1 | tee -a "$LOG"
fi

CLAIM="${1:?usage: run_match.sh seatclaim_... [agent.module]}"
AGENT="${2:-agent.agent}"
echo "Logging to $LOG (claim token is NOT written to the log)"
caffeinate -dims "$PY" -m agent --claim "$CLAIM" --agent "$AGENT" 2>&1 | sed -E 's/seatclaim_[A-Za-z0-9_-]+/seatclaim_[REDACTED]/g' | tee -a "$LOG"
