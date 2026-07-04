#!/usr/bin/env bash
#
# Wait for a running exp1 to finish, then run exp2.
#
# Usage (launch detached with nohup):
#     nohup bash experiments/run_exp2_after_exp1.sh > experiments/exp2_chain.log 2>&1 &
#
# Optionally pass the exp1 PID as the first argument for an exact wait:
#     nohup bash experiments/run_exp2_after_exp1.sh 12345 > experiments/exp2_chain.log 2>&1 &
#
# Override the interpreter if needed (e.g. after `conda activate cdarr`):
#     PYTHON=/path/to/python nohup bash experiments/run_exp2_after_exp1.sh > ... &
#
set -u

# ── Config ────────────────────────────────────────────────────────────────────
PYTHON="${PYTHON:-python}"                 # interpreter (activate your env first)
POLL_SECONDS="${POLL_SECONDS:-30}"         # how often to check on exp1
EXP1_PATTERN='exp1-crossing-angle\.py'     # matches exp1 in `pgrep -f`
EXP1_PID="${1:-}"                          # optional exact PID to wait on

# Project root = parent of this script's directory; run from there.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

EXP2_LOG="experiments/exp2.log"

echo "[chain] $(date '+%F %T')  waiting for exp1 to finish, then running exp2"
echo "[chain] python     : $PYTHON"
echo "[chain] cwd        : $PROJECT_ROOT"
echo "[chain] poll every : ${POLL_SECONDS}s"

# ── Wait for exp1 ─────────────────────────────────────────────────────────────
if [ -n "$EXP1_PID" ]; then
    echo "[chain] waiting on PID $EXP1_PID ..."
    while kill -0 "$EXP1_PID" 2>/dev/null; do
        sleep "$POLL_SECONDS"
    done
else
    echo "[chain] waiting on any process matching /$EXP1_PATTERN/ ..."
    # Small guard so we don't race a not-yet-started exp1.
    sleep 2
    while pgrep -f "$EXP1_PATTERN" > /dev/null 2>&1; do
        sleep "$POLL_SECONDS"
    done
fi

echo "[chain] $(date '+%F %T')  exp1 finished — starting exp2"

# ── Run exp2 ──────────────────────────────────────────────────────────────────
"$PYTHON" -u experiments/exp2-gamma.py > "$EXP2_LOG" 2>&1
STATUS=$?

echo "[chain] $(date '+%F %T')  exp2 exited with status $STATUS (output: $EXP2_LOG)"
exit "$STATUS"
