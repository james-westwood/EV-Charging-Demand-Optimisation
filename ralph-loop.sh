#!/usr/bin/env bash
# ralph-loop.sh — Run multiple Ralph loop iterations unattended (AFK mode).
#
# Usage:
#   ./ralph-loop.sh              # run up to 10 iterations (default)
#   ./ralph-loop.sh --max 5      # run up to 5 iterations
#   ./ralph-loop.sh --max 50     # run up to 50 iterations
#
# The loop stops when:
#   - All tasks in prd.json are complete
#   - The iteration cap is reached
#   - Claude exits with a non-zero status (error or test failure)
#
# Logs are written to ralph-loop.log in addition to stdout.
#
# Requirements:
#   - claude CLI installed and authenticated
#   - Run from the project root (Energy Forecasting/)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse --max argument
MAX_ITERATIONS=10
if [[ "${1:-}" == "--max" && -n "${2:-}" ]]; then
  MAX_ITERATIONS="$2"
fi

LOG_FILE="$SCRIPT_DIR/ralph-loop.log"
CLAUDE_FLAGS="--dangerously-skip-permissions"

PROMPT='You are implementing an EV Charging Demand Optimisation ML project using a Ralph loop methodology.

Read AGENTS.md first to understand the project structure and conventions.
Read progress.txt to understand what has been done so far.
Read prd.json to find the task list.

Find the next incomplete task (lowest id where completed is false).

Then follow these steps exactly:
1. Read the task'"'"'s description and acceptance_criteria from prd.json
2. Implement the task — create or edit files in the energy-forecasting/ directory
3. Run the quality check: cd energy-forecasting && uv run pytest tests/ -v
   - If tests fail, fix the code and re-run until they pass
   - Do NOT proceed if tests are failing
4. Update prd.json: set "completed": true for the task you just finished
5. Append a one-line summary to progress.txt in the format:
   [YYYY-MM-DD] [{task_id}] {task_title}: {what was done}
6. Do a git commit from the energy-forecasting/ directory:
   git add -A && git commit -m "[{task_id}] {task_title}: {one line summary}"

Important rules:
- Implement exactly ONE task per iteration — no more
- Never skip the test run
- Never commit if tests are failing
- Use uv for all Python commands (uv run pytest, uv sync, etc.)
- All source files go under energy-forecasting/src/
- All test files go under energy-forecasting/tests/
- Follow all conventions in AGENTS.md'

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

check_complete() {
  python3 -c "
import json, sys
with open('prd.json') as f:
    prd = json.load(f)
incomplete = [t for t in prd['tasks'] if not t['completed']]
sys.exit(0 if incomplete else 1)
" 2>/dev/null
}

count_remaining() {
  python3 -c "
import json
with open('prd.json') as f:
    prd = json.load(f)
incomplete = [t for t in prd['tasks'] if not t['completed']]
print(len(incomplete))
" 2>/dev/null || echo "?"
}

show_next_task() {
  python3 -c "
import json
with open('prd.json') as f:
    prd = json.load(f)
incomplete = [t for t in prd['tasks'] if not t['completed']]
if incomplete:
    t = incomplete[0]
    print(f'[{t[\"id\"]}] {t[\"title\"]} ({t[\"epic\"]})')
" 2>/dev/null || echo "(unknown)"
}

echo "================================================================"
echo "  Ralph Loop — AFK Mode"
echo "  Max iterations: $MAX_ITERATIONS"
echo "  Log: $LOG_FILE"
echo "================================================================"
echo ""

log "Starting Ralph loop. Max iterations: $MAX_ITERATIONS"

ITERATION=0

while true; do
  # Check if all tasks done
  if ! check_complete; then
    log "All tasks complete! Stopping."
    echo ""
    echo "================================================================"
    echo "  ALL TASKS COMPLETE"
    echo "================================================================"
    break
  fi

  # Check iteration cap
  if [[ $ITERATION -ge $MAX_ITERATIONS ]]; then
    REMAINING=$(count_remaining)
    log "Reached iteration cap ($MAX_ITERATIONS). $REMAINING tasks remaining."
    echo ""
    echo "================================================================"
    echo "  ITERATION CAP REACHED ($MAX_ITERATIONS)"
    echo "  Tasks remaining: $REMAINING"
    echo "  Run ./ralph-loop.sh --max N to continue"
    echo "================================================================"
    break
  fi

  ITERATION=$((ITERATION + 1))
  NEXT=$(show_next_task)
  REMAINING=$(count_remaining)

  echo ""
  echo "--- Iteration $ITERATION / $MAX_ITERATIONS  |  $REMAINING tasks remaining ---"
  log "Iteration $ITERATION: $NEXT"

  # Run Claude for one iteration
  if ! claude $CLAUDE_FLAGS --print "$PROMPT" 2>&1 | tee -a "$LOG_FILE"; then
    log "ERROR: Claude exited with non-zero status on iteration $ITERATION. Stopping."
    echo ""
    echo "================================================================"
    echo "  STOPPED: Claude returned an error on iteration $ITERATION"
    echo "  Check ralph-loop.log for details"
    echo "  Fix the issue, then re-run to continue"
    echo "================================================================"
    exit 1
  fi

  log "Iteration $ITERATION complete."

  # Brief pause between iterations to avoid rate limits
  sleep 2
done

REMAINING=$(count_remaining)
log "Loop finished. Completed $ITERATION iterations. $REMAINING tasks remaining."
echo ""
echo "Summary: $ITERATION iterations run, $REMAINING tasks remaining."
