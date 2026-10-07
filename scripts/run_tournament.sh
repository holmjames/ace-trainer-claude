#!/usr/bin/env bash
# Tournament-day launcher. Keeps the laptop awake, runs the official runtime, restarts it if it
# ever exits, and tees everything to a timestamped log in logs/ for the results tally.
#
#   ./scripts/run_tournament.sh              # tournament games only (what we want on Oct 16)
#   ./scripts/run_tournament.sh --match      # also play Testing matches (upstream PR #5 merged Oct 7)
#
# Stop with Ctrl+C (twice if the runtime is mid-restart).
set -u
cd "$(dirname "$0")/.."

mkdir -p logs
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="logs/runtime-tournament-${STAMP}.log"
PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"
# caffeinate keeps a Mac awake; on Linux/WSL it does not exist and Windows power settings do the job.
KEEPAWAKE=""
command -v caffeinate >/dev/null 2>&1 && KEEPAWAKE="caffeinate -dims"

echo "Logging to $LOG"
echo "Pre-flight check:"
"$PY" -m agent --check-tournament 2>&1 | tee -a "$LOG" || { echo "Pre-flight failed; fix .env before the tournament."; exit 1; }

attempt=0
trap 'echo "Stopping."; exit 0' INT TERM
while true; do
  attempt=$((attempt + 1))
  echo "[$(date +%H:%M:%S)] starting runtime (attempt $attempt)" | tee -a "$LOG"
  # caffeinate -dims: no display sleep, no idle sleep, no disk sleep, no system sleep while it runs.
  $KEEPAWAKE "$PY" -m agent --tournament "$@" 2>&1 | tee -a "$LOG"
  echo "[$(date +%H:%M:%S)] runtime exited; restarting in 5 s (Ctrl+C to stop)" | tee -a "$LOG"
  sleep 5
done
