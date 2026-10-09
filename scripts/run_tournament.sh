#!/usr/bin/env bash
# Tournament-day launcher. Keeps the laptop awake, runs the official runtime DETACHED from this terminal, restarts
# it if it ever exits, and logs everything to a timestamped file in logs/ for the results tally.
#
#   ./scripts/run_tournament.sh                   # tournament games only (what we want on Oct 16)
#   ./scripts/run_tournament.sh --match           # also play Testing matches (upstream PR #5 merged Oct 7)
#   ./scripts/run_tournament.sh --stop            # stop the detached runtime (pid file: logs/runtime-tournament.pid)
#   ./scripts/run_tournament.sh --status          # is it running, and which log is it writing
#   ./scripts/run_tournament.sh --dry-run [...]   # print what would start, start nothing
#   ./scripts/run_tournament.sh --foreground [...]  # the old behaviour: attached to this terminal, Ctrl+C stops it
#
# Detached means: the restart loop runs in its own session (python os.setsid + nohup, no controlling terminal), so
# closing this Terminal window, logging out, or a SIGHUP no longer kills the runtime mid-match (Oct 8: it did). The
# launcher follows the log in the foreground; Ctrl+C there only stops following. Stop the runtime with --stop: it
# sends SIGINT to the whole process group (what Ctrl+C used to do: runtime, caffeinate and match workers), SIGTERM
# after 15 s, SIGKILL after 30 s. Anything else on the command line is passed to `python -m agent --tournament`.
#
# pipefail: the pre-flight's exit code must survive the tee, or a failed check never stops the launcher.
set -uo pipefail
# The absolute path of this script, resolved BEFORE the cd: the detached loop re-runs it (`bash "$SCRIPT" --loop`),
# and a relative $0 ("./run_tournament.sh" from scripts/) is wrong once we are in the repository root.
SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$0")/.."

LOG_DIR="${TOURNAMENT_LOG_DIR:-logs}"
PID_FILE="$LOG_DIR/runtime-tournament.pid"
PY="${TOURNAMENT_PYTHON:-.venv/bin/python}"  # TOURNAMENT_PYTHON: tests stand a stub in for the venv python
[ -x "$PY" ] || PY="python3"
# Save every raw server state the agent sees (tests/fixtures/live/<session>/<phase>-<version>-seat<n>.json):
# tests/test_live_fixtures.py replays them and scripts/pool_from_live.py learns the real card pool from the drafts.
export AGENT_CAPTURE_DIR="${AGENT_CAPTURE_DIR:-tests/fixtures/live}"
# caffeinate keeps a Mac awake; on Linux/WSL it does not exist and Windows power settings do the job.
KEEPAWAKE=""
command -v caffeinate >/dev/null 2>&1 && KEEPAWAKE="caffeinate -dims"

mode="start"
args=()
for arg in "$@"; do
  case "$arg" in
    --stop) mode="stop" ;;
    --status) mode="status" ;;
    --dry-run) mode="dry-run" ;;
    --foreground) mode="foreground" ;;
    --loop) mode="loop" ;;  # internal: the restart loop itself (started detached by `start`)
    *) args+=("$arg") ;;
  esac
done
# bash 3.2 (macOS) + set -u: an empty array must be expanded as ${args[@]+"${args[@]}"}
RUNTIME_ARGS=(--tournament ${args[@]+"${args[@]}"})

running_pid() {
  # Prints the pid of the detached loop if it is alive; nothing otherwise (a stale pid file is removed).
  [ -f "$PID_FILE" ] || return 1
  local pid
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    echo "$pid"
    return 0
  fi
  rm -f "$PID_FILE"
  return 1
}

newest_log() {
  ls -t "$LOG_DIR"/runtime-tournament-*.log 2>/dev/null | head -1
}

preflight() {
  # Must pass before anything starts: the key, the connection, the agent and (F3) a real answer from the model.
  echo "Logging to $LOG"
  local model_line
  model_line="$(grep -E '^AGENT_MODEL=' .env 2>/dev/null || true)"
  echo "Judging model from .env: ${model_line:-<unset, code default applies>}"
  echo "Pre-flight check:"
  "$PY" -m agent --check-tournament 2>&1 | tee -a "$LOG" || { echo "Pre-flight failed; fix .env before the tournament." | tee -a "$LOG"; return 1; }
}

run_loop() {
  # The restart loop: everything it prints is the log (the caller tees or redirects). SIGINT/SIGTERM stop it after
  # the runtime (which gets the same signal as a member of the process group) has exited.
  echo "$$" > "$PID_FILE" 2>/dev/null || true
  local attempt=0 rc why
  trap 'echo "[$(date +%H:%M:%S)] Stopping."; rm -f "$PID_FILE"; exit 0' INT TERM
  while true; do
    attempt=$((attempt + 1))
    echo "[$(date +%H:%M:%S)] starting runtime (attempt $attempt): $PY -m agent ${RUNTIME_ARGS[*]}"
    # caffeinate -dims: no display sleep, no idle sleep, no disk sleep, no system sleep while it runs.
    $KEEPAWAKE "$PY" -m agent "${RUNTIME_ARGS[@]}" 2>&1
    rc=$?
    # >128 means a signal killed it (e.g. 143 = SIGTERM, 137 = SIGKILL); the runtime prints nothing then.
    why="exit code $rc"; [ "$rc" -gt 128 ] && why="killed by signal $((rc - 128)) (exit code $rc)"
    echo "[$(date +%H:%M:%S)] runtime exited ($why); restarting in 5 s (--stop to stop)"
    sleep 5
  done
}

case "$mode" in
  loop)
    LOG="${TOURNAMENT_LOG_FILE:?--loop is internal: started by this script with TOURNAMENT_LOG_FILE set}"
    run_loop
    ;;

  dry-run)
    STAMP="$(date +%Y%m%d-%H%M%S)"
    echo "dry run: nothing starts"
    echo "  runtime:   $KEEPAWAKE $PY -m agent ${RUNTIME_ARGS[*]}"
    echo "  log:       $LOG_DIR/runtime-tournament-$STAMP.log"
    echo "  pid file:  $PID_FILE"
    echo "  capture:   AGENT_CAPTURE_DIR=$AGENT_CAPTURE_DIR"
    echo "  detached:  nohup + python os.setsid (own session; survives closing this terminal)"
    echo "  script:    $SCRIPT"
    echo "  stop with: $SCRIPT --stop"
    if pid="$(running_pid)"; then echo "  note:      a runtime is already running (pid $pid)"; fi
    exit 0
    ;;

  status)
    if pid="$(running_pid)"; then
      echo "running (pid $pid, process group $pid); log: $(newest_log)"
      tail -n 1 "$(newest_log)" 2>/dev/null || true
      exit 0
    fi
    echo "not running (no live pid in $PID_FILE)"
    exit 1
    ;;

  stop)
    if ! pid="$(running_pid)"; then
      echo "not running (no live pid in $PID_FILE); nothing to stop"
      exit 1
    fi
    echo "stopping runtime (pid $pid): SIGINT to its process group"
    kill -INT -- "-$pid" 2>/dev/null || kill -INT "$pid" 2>/dev/null || true
    for signal in TERM KILL; do
      i=0
      while kill -0 "$pid" 2>/dev/null && [ "$i" -lt 150 ]; do sleep 0.1; i=$((i + 1)); done
      if ! kill -0 "$pid" 2>/dev/null; then break; fi
      echo "still running after 15 s; sending SIG$signal"
      kill "-$signal" -- "-$pid" 2>/dev/null || kill "-$signal" "$pid" 2>/dev/null || true
    done
    rm -f "$PID_FILE"
    if kill -0 "$pid" 2>/dev/null; then echo "could not stop pid $pid"; exit 1; fi
    echo "stopped"
    exit 0
    ;;

  foreground)
    mkdir -p "$LOG_DIR"
    STAMP="$(date +%Y%m%d-%H%M%S)"
    LOG="$LOG_DIR/runtime-tournament-$STAMP.log"
    echo "Foreground mode: closing this terminal kills the runtime; Ctrl+C stops it (twice if mid-restart)."
    preflight || exit 1
    run_loop 2>&1 | tee -a "$LOG"
    ;;

  start)
    mkdir -p "$LOG_DIR"
    if pid="$(running_pid)"; then
      echo "a runtime is already running (pid $pid); log: $(newest_log)"
      echo "stop it first: $SCRIPT --stop"
      exit 1
    fi
    STAMP="$(date +%Y%m%d-%H%M%S)"
    LOG="$LOG_DIR/runtime-tournament-$STAMP.log"
    preflight || exit 1
    # Detach: nohup ignores SIGHUP; the python shim moves the loop into its own session (no controlling terminal)
    # and execs bash --loop in place, so $! is the loop's pid and its process-group id.
    TOURNAMENT_LOG_FILE="$LOG" nohup "$PY" -c 'import os, sys
try:
    os.setsid()
except OSError:
    pass
os.execvp(sys.argv[1], sys.argv[1:])' bash "$SCRIPT" --loop ${args[@]+"${args[@]}"} >> "$LOG" 2>&1 < /dev/null &
    loop_pid=$!
    echo "$loop_pid" > "$PID_FILE"
    # Say "started" only once the loop is alive and has written its first line (a wrong path or a broken shim dies
    # at once, and used to be announced as started anyway).
    i=0
    while [ "$i" -lt 100 ]; do
      if ! kill -0 "$loop_pid" 2>/dev/null; then break; fi
      grep -q "starting runtime" "$LOG" 2>/dev/null && break
      sleep 0.1; i=$((i + 1))
    done
    if ! kill -0 "$loop_pid" 2>/dev/null; then
      rm -f "$PID_FILE"
      echo "runtime FAILED to start: the detached loop (pid $loop_pid) exited at once. Last lines of $LOG:" | tee -a "$LOG"
      tail -n 5 "$LOG"
      exit 1
    fi
    if ! grep -q "starting runtime" "$LOG" 2>/dev/null; then
      echo "warning: the loop (pid $loop_pid) is alive but has not written to $LOG yet" | tee -a "$LOG"
    fi
    echo "[$(date +%H:%M:%S)] runtime started detached (pid $loop_pid): $PY -m agent ${RUNTIME_ARGS[*]}" | tee -a "$LOG"
    echo "Closing this window does not stop it. Stop: $SCRIPT --stop   Status: $SCRIPT --status"
    echo "Following $LOG (Ctrl+C stops following only):"
    trap 'echo; echo "Still running (pid $loop_pid). Stop with: $SCRIPT --stop"; exit 0' INT TERM
    tail -n +1 -f "$LOG"
    ;;
esac
