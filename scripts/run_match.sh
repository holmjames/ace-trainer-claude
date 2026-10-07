#!/usr/bin/env bash
# Play your Testing matches (the dashboard's Testing page: seats set to "Mine (self-hosted)" or
# joined from Open matches) and tee the runtime output to logs/ so scripts/tally.py can read the
# results. Claim codes are retired upstream; one runtime plays every seat you hold.
#
#   ./scripts/run_match.sh                               # champion (agent/agent.py) on every seat
#   AGENT_SEAT0=agent.agent AGENT_SEAT1=agent.versions.v2 ./scripts/run_match.sh --agent agent.arena
#                                                        # self-play: a different version per seat
#   ./scripts/run_match.sh --tournament                  # Testing AND tournament games in one process
set -u
cd "$(dirname "$0")/.."
mkdir -p logs
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="logs/runtime-match-${STAMP}.log"
PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"
echo "Logging to $LOG"
exec caffeinate -dims "$PY" -m agent --match "$@" 2>&1 | tee -a "$LOG"
