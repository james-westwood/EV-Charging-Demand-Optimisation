#!/usr/bin/env bash
# ralph-once.sh — Run one Ralph loop iteration using Claude Code.
#
# Usage:
#   ./ralph-once.sh               # run next task automatically
#   ./ralph-once.sh --task 1.3    # force a specific task ID
#
# Requirements:
#   - claude CLI installed and authenticated
#   - Run from the project root (Energy Forecasting/)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TASK_ID="${2:-}"  # optional --task <id> override
CLAUDE_FLAGS="--dangerously-skip-permissions"

# Build the prompt
if [[ -n "$TASK_ID" && "$1" == "--task" ]]; then
  TASK_FILTER="Focus specifically on task ID $TASK_ID."
else
  TASK_FILTER="Find the next incomplete task (lowest id where completed is false)."
fi

PROMPT="You are implementing an EV Charging Demand Optimisation ML project using a Ralph loop methodology.

Read AGENTS.md first to understand the project structure and conventions.
Read progress.txt to understand what has been done so far.
Read prd.json to find the task list.

$TASK_FILTER

Then follow these steps exactly:
1. Read the task's description and acceptance_criteria from prd.json
2. Implement the task — create or edit files in the energy-forecasting/ directory
3. Run the quality check: cd energy-forecasting && uv run pytest tests/ -v
   - If tests fail, fix the code and re-run until they pass
   - Do NOT proceed if tests are failing
4. Update prd.json: set \"completed\": true for the task you just finished
5. Append a one-line summary to progress.txt in the format:
   [YYYY-MM-DD] [{task_id}] {task_title}: {what was done}
6. Do a git commit from the energy-forecasting/ directory:
   git add -A && git commit -m \"[{task_id}] {task_title}: {one line summary}\"

Important rules:
- Implement exactly ONE task per iteration — no more
- Never skip the test run
- Never commit if tests are failing
- Use uv for all Python commands (uv run pytest, uv sync, etc.)
- All source files go under energy-forecasting/src/
- All test files go under energy-forecasting/tests/
- Follow all conventions in AGENTS.md"

echo "=== Ralph Loop — Single Iteration ==="
echo "Reading prd.json for next task..."
echo ""

# Show next incomplete task for visibility
python3 -c "
import json, sys
with open('prd.json') as f:
    prd = json.load(f)
incomplete = [t for t in prd['tasks'] if not t['completed']]
if not incomplete:
    print('ALL TASKS COMPLETE! Nothing to do.')
    sys.exit(0)
t = incomplete[0]
print(f'Next task: [{t[\"id\"]}] {t[\"title\"]}')
print(f'Epic: {t[\"epic\"]}')
print(f'Priority: {t[\"priority\"]}')
print()
" || true

echo "Launching Claude Code..."
echo ""

claude $CLAUDE_FLAGS --print "$PROMPT"

echo ""
echo "=== Iteration complete. Review changes above, then run again or check prd.json. ==="
