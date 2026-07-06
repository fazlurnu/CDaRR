#!/usr/bin/env bash
#
# Run exp3 to completion, then run exp4.
#
# Usage (launch detached with nohup):
#     nohup bash experiments/run_exp3_then_exp4.sh > experiments/exp3_exp4_chain.log 2>&1 &
#
# Override the interpreter if needed (e.g. after `conda activate cdarr`):
#     PYTHON=/path/to/python nohup bash experiments/run_exp3_then_exp4.sh > ... &
#
set -u

# ── Config ────────────────────────────────────────────────────────────────────
PYTHON="${PYTHON:-python}"                 # interpreter (activate your env first)

# Project root = parent of this script's directory; run from there.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

mkdir -p experiments/results
EXP3_LOG="experiments/results/exp3.log"
EXP4_LOG="experiments/results/exp4.log"

echo "[chain] $(date '+%F %T')  running exp3, then exp4"
echo "[chain] python : $PYTHON"
echo "[chain] cwd    : $PROJECT_ROOT"

# ── Run exp3 ──────────────────────────────────────────────────────────────────
echo "[chain] $(date '+%F %T')  starting exp3 (log: $EXP3_LOG)"
"$PYTHON" -u experiments/exp3-noise-model-random-angle.py > "$EXP3_LOG" 2>&1
STATUS=$?
echo "[chain] $(date '+%F %T')  exp3 exited with status $STATUS"

if [ "$STATUS" -ne 0 ]; then
    echo "[chain] exp3 failed — aborting, not running exp4"
    exit "$STATUS"
fi

# ── Run exp4 ──────────────────────────────────────────────────────────────────
echo "[chain] $(date '+%F %T')  starting exp4 (log: $EXP4_LOG)"
"$PYTHON" -u experiments/exp4-noise-model-random-angle-homogen.py > "$EXP4_LOG" 2>&1
STATUS=$?
echo "[chain] $(date '+%F %T')  exp4 exited with status $STATUS"

exit "$STATUS"
