#!/usr/bin/env bash
#
# Verify that a PR is properly linked to an issue
#
# Usage:
#   ./verify_pr_link.sh <pr-number> <issue-number>

set -euo pipefail

PR_NUMBER="${1:-}"
ISSUE_NUMBER="${2:-}"

if [[ -z "$PR_NUMBER" ]] || [[ -z "$ISSUE_NUMBER" ]]; then
    echo "Error: PR number and issue number required"
    echo "Usage: $0 <pr-number> <issue-number>"
    exit 1
fi

# Get repository info
REPO=$(gh repo view --json nameWithOwner -q '.nameWithOwner')

echo "Checking if PR #$PR_NUMBER is linked to issue #$ISSUE_NUMBER..."
echo ""

# Check if PR references the issue
PR_BODY=$(gh pr view "$PR_NUMBER" --json body -q '.body')
if echo "$PR_BODY" | grep -q "#$ISSUE_NUMBER"; then
    echo "✅ PR body references issue #$ISSUE_NUMBER"
else
    echo "❌ PR body does not reference issue #$ISSUE_NUMBER"
fi

# Check if issue shows the PR in Development section
LINKED_PRS=$(gh issue view "$ISSUE_NUMBER" --json projectCards -q '.projectCards')
if [[ -n "$LINKED_PRS" ]]; then
    echo "✅ Issue #$ISSUE_NUMBER has linked PRs"
else
    echo "⚠️  Could not verify linked PRs via API"
fi

echo ""
echo "Manual verification:"
echo "  Issue: https://github.com/$REPO/issues/$ISSUE_NUMBER"
echo "  PR: https://github.com/$REPO/pull/$PR_NUMBER"
