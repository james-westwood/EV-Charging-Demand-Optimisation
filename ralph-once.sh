#!/usr/bin/env bash
# ralph-once.sh — Run one Ralph loop iteration (branch → atomic commits → PR → review → merge).
#
# Usage:
#   ./ralph-once.sh               # run next ralph-owned task
#   ./ralph-once.sh --task 1.4    # force a specific task ID
#
# Requirements:
#   - claude CLI, gemini CLI, gh CLI (authenticated)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

FORCED_TASK_ID="${2:-}"
MAIN_BRANCH="main"
TODAY=$(date +%Y-%m-%d)

# ── Helpers ───────────────────────────────────────────────────────────────────

run_coder() {
  local agent="$1" prompt="$2"
  if [[ "$agent" == "claude" ]]; then
    env -u CLAUDECODE claude --dangerously-skip-permissions --print "$prompt"
  else
    gemini --yolo -p "$prompt"
  fi
}

run_reviewer() {
  local agent="$1" prompt="$2"
  if [[ "$agent" == "claude" ]]; then
    env -u CLAUDECODE claude --print "$prompt"
  else
    gemini -p "$prompt"
  fi
}

next_task_field() {
  local field="$1" filter="${2:-}"
  python3 -c "
import json
with open('prd.json') as f: prd = json.load(f)
if '$filter':
    t = next(x for x in prd['tasks'] if x['id'] == '$filter')
else:
    t = [x for x in prd['tasks'] if not x['completed'] and x.get('owner') != 'human'][0]
print(t['$field'])
" 2>/dev/null
}

# ── Preflight ─────────────────────────────────────────────────────────────────

command -v gh     >/dev/null 2>&1 || { echo "ERROR: 'gh' not found"; exit 1; }
command -v gemini >/dev/null 2>&1 || { echo "ERROR: 'gemini' not found"; exit 1; }
gh auth status    >/dev/null 2>&1 || { echo "ERROR: not authenticated with gh — run: gh auth login"; exit 1; }

# ── Resolve task ──────────────────────────────────────────────────────────────

echo "=== Ralph — Single Iteration ==="
echo ""

TASK_ID=$(next_task_field id "$FORCED_TASK_ID")
TASK_TITLE=$(next_task_field title "$FORCED_TASK_ID")
TASK_DESC=$(next_task_field description "$FORCED_TASK_ID")
TASK_AC=$(next_task_field acceptance_criteria "$FORCED_TASK_ID")
TASK_EPIC=$(next_task_field epic "$FORCED_TASK_ID")

# Check if human-owned (only relevant when auto-picking)
if [[ -z "$FORCED_TASK_ID" ]]; then
  OWNER=$(python3 -c "
import json
with open('prd.json') as f: prd = json.load(f)
incomplete = [t for t in prd['tasks'] if not t['completed']]
if incomplete: print(incomplete[0].get('owner', ''))
" 2>/dev/null)
  if [[ "$OWNER" == "human" ]]; then
    echo "Next task [$TASK_ID] $TASK_TITLE is YOURS to implement."
    echo "Mark it complete in prd.json, then re-run ralph."
    exit 0
  fi
fi

BRANCH="ralph/task-${TASK_ID}-${TASK_TITLE}"

# Assign agents randomly
if (( RANDOM % 2 == 0 )); then CODER="claude"; REVIEWER="gemini"
else                            CODER="gemini"; REVIEWER="claude"
fi

echo "  Task:     [$TASK_ID] $TASK_TITLE"
echo "  Epic:     $TASK_EPIC"
echo "  Branch:   $BRANCH"
echo "  Coder:    $CODER  |  Reviewer: $REVIEWER"
echo ""

# ── Branch setup ──────────────────────────────────────────────────────────────

git checkout "$MAIN_BRANCH"
git pull --ff-only origin "$MAIN_BRANCH" 2>/dev/null || true
git checkout -b "$BRANCH"

# ── Coding step ───────────────────────────────────────────────────────────────

echo "--- Coding ($CODER) ---"

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
- Use uv for all Python commands
- Follow all conventions in AGENTS.md"

run_coder "$CODER" "$CODER_PROMPT"

# ── Push and open PR ──────────────────────────────────────────────────────────

echo ""
echo "--- Pushing and creating PR ---"
git push -u origin "$BRANCH"

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
echo "PR #$PR_NUMBER: $PR_URL"

# ── Review step ───────────────────────────────────────────────────────────────

echo ""
echo "--- Reviewing ($REVIEWER) ---"
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

REVIEW_TEXT=$(run_reviewer "$REVIEWER" "$REVIEW_PROMPT")

gh pr comment "$PR_NUMBER" --body "$(cat <<EOF
## Code Review by \`$REVIEWER\`

$REVIEW_TEXT

---
*Implemented by \`$CODER\` · Reviewed by \`$REVIEWER\`*
EOF
)"

# ── Merge ─────────────────────────────────────────────────────────────────────

echo ""
echo "--- Merging PR #$PR_NUMBER ---"
gh pr merge "$PR_NUMBER" --merge --delete-branch

git checkout "$MAIN_BRANCH"
git pull --ff-only origin "$MAIN_BRANCH"

echo ""
echo "=== Done: [$TASK_ID] $TASK_TITLE | $PR_URL ==="
