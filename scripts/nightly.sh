#!/usr/bin/env bash
# Nightly regression (PLAN §5l item 7): tests, all scenarios code-only, and 1,000 fixed-seed games vs random and vs the smoke
# agent. Appends one summary line to logs/nightly.log so a drop shows up the next morning. Free; about 3 minutes.
#   ./scripts/nightly.sh
set -u
cd "$(dirname "$0")/.."
PY=".venv/bin/python"; [ -x "$PY" ] || PY="python3"
mkdir -p logs
STAMP="$(date +%Y-%m-%d_%H:%M)"
TESTS="$("$PY" -m pytest -q 2>&1 | tail -1 | tr -d '\n')"
SCEN="$("$PY" scripts/scenarios.py --code-only --set all 2>&1 | tail -1 | tr -d '\n')"
RANDOM_WR="$("$PY" sim/harness.py --games 1000 --p1 code --p2 random --seed 21 2>&1 | grep -E '^\s+code\s' | awk '{print $(NF)}' | tr -d '()\n')"
SMOKE_WR="$("$PY" sim/harness.py --games 1000 --p1 code --p2 examples.smoke_agent --seed 22 2>&1 | grep -E '^\s+code\s' | awk '{print $(NF)}' | tr -d '()\n')"
LINE="$STAMP | $(git rev-parse --short HEAD) | tests: $TESTS | scenarios: $SCEN | vs random (seed 21): $RANDOM_WR | vs smoke (seed 22): $SMOKE_WR"
echo "$LINE" | tee -a logs/nightly.log
