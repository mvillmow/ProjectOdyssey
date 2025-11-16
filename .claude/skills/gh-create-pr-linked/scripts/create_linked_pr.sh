#!/usr/bin/env bash
#
# Create a pull request linked to a GitHub issue
#
# Usage:
#   ./create_linked_pr.sh <issue-number> [description]
#
# Example:
#   ./create_linked_pr.sh 42
#   ./create_linked_pr.sh 42 "Custom description"

set -euo pipefail

ISSUE_NUMBER="${1:-}"
DESCRIPTION="${2:-}"

if [[ -z "$ISSUE_NUMBER" ]]; then
    echo "Error: Issue number required"
    echo "Usage: $0 <issue-number> [description]"
    exit 1
fi

# Check if we're in a git repo
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "Error: Not in a git repository"
    exit 1
fi

# Get current branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "Current branch: $CURRENT_BRANCH"

# Check if branch has upstream
if ! git rev-parse --abbrev-ref "@{upstream}" > /dev/null 2>&1; then
    echo "Branch has no upstream, pushing..."
    git push -u origin "$CURRENT_BRANCH"
fi

# Ensure we're up to date with remote
git fetch origin

# Create PR linked to issue
if [[ -n "$DESCRIPTION" ]]; then
    # Use custom description with issue link
    gh pr create --title "$(gh issue view "$ISSUE_NUMBER" --json title -q '.title')" --body "$(cat <<EOF
$DESCRIPTION

Closes #$ISSUE_NUMBER
EOF
)"
else
    # Use --issue flag for automatic linking
    gh pr create --issue "$ISSUE_NUMBER"
fi

# Get PR number
PR_NUMBER=$(gh pr view --json number -q '.number')

echo ""
echo "✅ PR #$PR_NUMBER created and linked to issue #$ISSUE_NUMBER"
echo ""
echo "Verify link: gh issue view $ISSUE_NUMBER"
echo "Check PR: gh pr view $PR_NUMBER"
