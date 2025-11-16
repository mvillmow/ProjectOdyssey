#!/usr/bin/env bash
# Fetch PR review comments using GitHub API

set -euo pipefail

PR_NUMBER="${1:-}"
FILTER="${2:-all}"

if [[ -z "$PR_NUMBER" ]]; then
    echo "Usage: $0 <pr-number> [reviewer-username|--unresolved]"
    exit 1
fi

# Get repo info
REPO=$(gh repo view --json nameWithOwner -q '.nameWithOwner')

echo "=== Review Comments for PR #$PR_NUMBER ==="
echo

# Fetch comments
COMMENTS=$(gh api "repos/$REPO/pulls/$PR_NUMBER/comments" 2>/dev/null || echo "[]")

# Check if there are comments
if [[ "$COMMENTS" == "[]" ]]; then
    echo "No review comments found for PR #$PR_NUMBER"
    exit 0
fi

# Apply filters
case "$FILTER" in
    --unresolved)
        echo "$COMMENTS" | jq -r '.[] | select(.in_reply_to_id == null) |
            "ID: \(.id)\nFile: \(.path):\(.line // .original_line)\nReviewer: \(.user.login)\nComment: \(.body)\n---"'
        ;;
    all)
        echo "$COMMENTS" | jq -r '.[] |
            "ID: \(.id)\nFile: \(.path):\(.line // .original_line)\nReviewer: \(.user.login)\nComment: \(.body)\nResolved: \(.in_reply_to_id != null)\n---"'
        ;;
    *)
        # Filter by reviewer
        echo "$COMMENTS" | jq -r --arg user "$FILTER" '.[] | select(.user.login == $user) |
            "ID: \(.id)\nFile: \(.path):\(.line // .original_line)\nComment: \(.body)\n---"'
        ;;
esac

exit 0
