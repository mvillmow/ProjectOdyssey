#!/usr/bin/env bash
#
# Get all review comments for a PR
#
# Usage:
#   ./get_feedback.sh <pr-number>

set -euo pipefail

PR_NUMBER="${1:-}"

if [[ -z "$PR_NUMBER" ]]; then
    echo "Error: PR number required"
    echo "Usage: $0 <pr-number>"
    exit 1
fi

# Get repository info
REPO=$(gh repo view --json nameWithOwner -q '.nameWithOwner')

echo "Fetching review comments for PR #$PR_NUMBER..."
echo ""

# Get all review comments
COMMENTS=$(gh api "repos/$REPO/pulls/$PR_NUMBER/comments" --jq '.')

# Check if there are any comments
if [[ "$COMMENTS" == "[]" ]]; then
    echo "No review comments found"
    exit 0
fi

# Parse and display comments
echo "$COMMENTS" | jq -r '.[] |
    "ID: \(.id)\nFile: \(.path)\nLine: \(.line // "N/A")\nReviewer: \(.user.login)\nComment:\n\(.body)\n---"'

echo ""
echo "Total comments: $(echo "$COMMENTS" | jq 'length')"
