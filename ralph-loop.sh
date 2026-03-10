#!/usr/bin/env bash
# ralph-loop.sh — Multi-agent AFK loop.
#
# Each task:
#   1. Randomly assign Claude or Gemini as CODER, the other as REVIEWER
#   2. Create a feature branch
#   3. CODER implements in atomic commits (src → tests → tracking)
#   4. Push branch, open PR on GitHub
#   5. REVIEWER reads the diff via `gh pr diff`, posts a review comment
#   6. Auto-merge → delete branch → pull main
#
# Stops when: all ralph-owned tasks done | next task is human-owned | iteration cap
#
# Usage:
#   ./ralph-loop.sh              # up to 10 iterations
#   ./ralph-loop.sh --max 50     # up to 50 iterations
#
# Requirements:
#   - claude CLI (with --dangerously-skip-permissions support)
#   - gemini CLI  (Google Gemini CLI, uses --yolo to auto-approve tool use)
#   - gh CLI authenticated (gh auth login)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MAX_ITERATIONS=10
if [[ "${1:-}" == "--max" && -n "${2:-}" ]]; then
  MAX_ITERATIONS="$2"
fi

LOG_FILE="$SCRIPT_DIR/ralph-loop.log"
MAIN_BRANCH="main"

# ── Helpers ──────────────────────────────────────────────────────────────────

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

die() {
  log "FATAL: $*"
  exit 1
}

# Coding agent — needs full file-system tool access
run_coder() {
  local agent="$1" prompt="$2"
  if [[ "$agent" == "claude" ]]; then
    claude --dangerously-skip-permissions --print "$prompt"
  else
    # Gemini CLI: --yolo auto-approves all tool use (equivalent to --dangerously-skip-permissions)
    gemini --yolo -p "$prompt"
  fi
}

# Reviewing agent — text only, no file-system access needed
run_reviewer() {
  local agent="$1" prompt="$2"
  if [[ "$agent" == "claude" ]]; then
    claude --print "$prompt"
  else
    gemini -p "$prompt"
  fi
}

# Stop if no incomplete ralph-owned tasks remain (exit 0 = tasks remain, exit 1 = all done)
check_complete() {
  python3 -c "
import json, sys
with open('prd.json') as f: prd = json.load(f)
sys.exit(0 if [t for t in prd['tasks'] if not t['completed'] and t.get('owner') != 'human'] else 1)
" 2>/dev/null
}

# Prints task label and exits 0 if the next incomplete task is human-owned; exits 1 otherwise
check_next_is_human() {
  python3 -c "
import json, sys
with open('prd.json') as f: prd = json.load(f)
incomplete = [t for t in prd['tasks'] if not t['completed']]
if incomplete and incomplete[0].get('owner') == 'human':
    t = incomplete[0]
    print(f'[{t[\"id\"]}] {t[\"title\"]} ({t[\"epic\"]})')
    sys.exit(0)
sys.exit(1)
" 2>/dev/null
}

count_remaining() {
  python3 -c "
import json
with open('prd.json') as f: prd = json.load(f)
print(len([t for t in prd['tasks'] if not t['completed'] and t.get('owner') != 'human']))
" 2>/dev/null || echo "?"
}

# Read a single field from the next incomplete ralph-owned task
next_task_field() {
  local field="$1"
  python3 -c "
import json
with open('prd.json') as f: prd = json.load(f)
t = [x for x in prd['tasks'] if not x['completed'] and x.get('owner') != 'human'][0]
print(t['$field'])
" 2>/dev/null
}

# ── Preflight ─────────────────────────────────────────────────────────────────

command -v gh     >/dev/null 2>&1 || die "'gh' not found — install GitHub CLI: https://cli.github.com"
command -v gemini >/dev/null 2>&1 || die "'gemini' not found — install Google Gemini CLI"
gh auth status    >/dev/null 2>&1 || die "Not authenticated with gh — run: gh auth login"

echo "================================================================"
echo "  Ralph Loop — Multi-Agent AFK Mode"
echo "  Coder/Reviewer: Claude ↔ Gemini (random each task)"
echo "  Workflow: branch → atomic commits → PR → AI review → merge"
echo "  Max iterations: $MAX_ITERATIONS"
echo "  Log: $LOG_FILE"
echo "================================================================"
echo ""

log "Starting Ralph loop. Max iterations: $MAX_ITERATIONS"

ITERATION=0

while true; do

  # ── Stop conditions ────────────────────────────────────────────────────────

  if HUMAN_TASK=$(check_next_is_human 2>/dev/null); then
    log "Reached human-owned task: $HUMAN_TASK. Handing over."
    echo ""
    echo "================================================================"
    echo "  YOUR TURN"
    echo "  Next task is yours to implement: $HUMAN_TASK"
    echo "  Mark it complete in prd.json, then re-run ralph."
    echo "================================================================"
    break
  fi

  if ! check_complete; then
    log "All ralph-owned tasks complete."
    echo ""
    echo "================================================================"
    echo "  ALL RALPH TASKS COMPLETE"
    echo "================================================================"
    break
  fi

  if [[ $ITERATION -ge $MAX_ITERATIONS ]]; then
    log "Iteration cap ($MAX_ITERATIONS). $(count_remaining) tasks remaining."
    echo ""
    echo "================================================================"
    echo "  ITERATION CAP ($MAX_ITERATIONS) — run with --max N to continue"
    echo "  Tasks remaining: $(count_remaining)"
    echo "================================================================"
    break
  fi

  ITERATION=$((ITERATION + 1))

  # ── Task details ───────────────────────────────────────────────────────────

  TASK_ID=$(next_task_field id)
  TASK_TITLE=$(next_task_field title)
  TASK_DESC=$(next_task_field description)
  TASK_AC=$(next_task_field acceptance_criteria)
  TASK_EPIC=$(next_task_field epic)
  BRANCH="ralph/task-${TASK_ID}-${TASK_TITLE}"
  TODAY=$(date +%Y-%m-%d)

  # Assign agents randomly
  if (( RANDOM % 2 == 0 )); then CODER="claude"; REVIEWER="gemini"
  else                            CODER="gemini"; REVIEWER="claude"
  fi

  echo ""
  echo "--- Iteration $ITERATION / $MAX_ITERATIONS  |  $(count_remaining) remaining ---"
  echo "  Task:     [$TASK_ID] $TASK_TITLE"
  echo "  Epic:     $TASK_EPIC"
  echo "  Branch:   $BRANCH"
  echo "  Coder:    $CODER  |  Reviewer: $REVIEWER"
  log "Iteration $ITERATION: [$TASK_ID] $TASK_TITLE | coder=$CODER reviewer=$REVIEWER branch=$BRANCH"

  # ── Branch setup ───────────────────────────────────────────────────────────

  git checkout "$MAIN_BRANCH"
  git pull --ff-only origin "$MAIN_BRANCH" 2>/dev/null || true
  git checkout -b "$BRANCH"

  # ── Coding step ────────────────────────────────────────────────────────────

  log "  Running coder ($CODER)..."

  CODER_PROMPT="You are the CODER implementing task [$TASK_ID] $TASK_TITLE for the EV Charging Demand Optimisation project. Another AI will review your work — write clean, production-quality code.

Read AGENTS.md for project conventions.

Epic: $TASK_EPIC
Description: $TASK_DESC
Acceptance criteria: $TASK_AC

Implementation steps:
1. Write source files under energy-forecasting/src/
2. Run tests and fix any failures: cd energy-forecasting && uv run pytest tests/ -v
3. Make ATOMIC commits — do not squash everything into one commit:
   - Commit A (source):   git add energy-forecasting/src/ && git commit -m '[$TASK_ID] $TASK_TITLE: implement'
   - Commit B (tests):    git add energy-forecasting/tests/ && git commit -m '[$TASK_ID] $TASK_TITLE: add tests'
   - Commit C (tracking): set \"completed\": true in prd.json for task $TASK_ID,
                          append to progress.txt: [$TODAY] [$TASK_ID] $TASK_TITLE: {one-line summary}
                          git add prd.json progress.txt && git commit -m '[$TASK_ID] $TASK_TITLE: mark complete'

Do NOT push. Do NOT create a PR. The orchestrator handles that.

Rules:
- Never implement any task with \"owner\": \"human\"
- Use uv for all Python commands (uv run pytest, uv sync, etc.)
- All source under energy-forecasting/src/, all tests under energy-forecasting/tests/
- Follow all conventions in AGENTS.md"

  run_coder "$CODER" "$CODER_PROMPT" 2>&1 | tee -a "$LOG_FILE"

  # ── Push and open PR ───────────────────────────────────────────────────────

  log "  Pushing $BRANCH..."
  git push -u origin "$BRANCH"

  log "  Creating PR..."
  PR_URL=$(gh pr create \
    --title "[$TASK_ID] $TASK_TITLE" \
    --body "$(cat <<EOF
## [$TASK_ID] $TASK_TITLE

**Epic:** $TASK_EPIC
**Coder:** \`$CODER\` | **Reviewer:** \`$REVIEWER\`

### Description
$TASK_DESC

### Acceptance Criteria
$TASK_AC

---
*Ralph Loop — multi-agent AI pair programming*
EOF
)" \
    --base "$MAIN_BRANCH" \
    --head "$BRANCH")

  PR_NUMBER=$(echo "$PR_URL" | grep -oE '[0-9]+$')
  log "  PR #$PR_NUMBER: $PR_URL"

  # ── Review step ────────────────────────────────────────────────────────────

  log "  Fetching diff for review ($REVIEWER)..."
  PR_DIFF=$(gh pr diff "$PR_NUMBER")

  REVIEW_PROMPT="You are the code reviewer for a pull request in the EV Charging Demand Optimisation project.

The code was written by $CODER. You are $REVIEWER.

PR: [$TASK_ID] $TASK_TITLE
Acceptance criteria: $TASK_AC

Diff:
---
$PR_DIFF
---

Write a concise code review covering:
1. Correctness — does the implementation satisfy the acceptance criteria?
2. Code quality — readability, naming, structure
3. Test quality — are tests meaningful and sufficient?
4. Any bugs, edge cases, or concerns

Be constructive and specific. End your review with exactly one of:
- **APPROVED** — code is good to merge as-is
- **CHANGES REQUESTED: {brief reason}** — if there are real issues

Output only the review text. It will be posted as a GitHub PR comment."

  log "  Running reviewer ($REVIEWER)..."
  REVIEW_TEXT=$(run_reviewer "$REVIEWER" "$REVIEW_PROMPT" 2>&1 | tee -a "$LOG_FILE")

  log "  Posting review comment..."
  gh pr comment "$PR_NUMBER" --body "$(cat <<EOF
## Code Review by \`$REVIEWER\`

$REVIEW_TEXT

---
*Implemented by \`$CODER\` · Reviewed by \`$REVIEWER\`*
EOF
)"

  # ── Merge ──────────────────────────────────────────────────────────────────

  log "  Merging PR #$PR_NUMBER..."
  gh pr merge "$PR_NUMBER" --merge --delete-branch

  git checkout "$MAIN_BRANCH"
  git pull --ff-only origin "$MAIN_BRANCH"

  log "Iteration $ITERATION complete: [$TASK_ID] $TASK_TITLE | $PR_URL"

  sleep 2

done

log "Loop finished. $ITERATION iterations. $(count_remaining) tasks remaining."
echo ""
echo "Summary: $ITERATION iterations run, $(count_remaining) tasks remaining."
