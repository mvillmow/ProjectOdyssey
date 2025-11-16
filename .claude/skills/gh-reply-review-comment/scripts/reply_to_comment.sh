#!/usr/bin/env bash
# Reply to a specific PR review comment

set -euo pipefail

PR_NUMBER="${1:-}"
COMMENT_ID="${2:-}"
REPLY_TEXT="${3:-}"

if [[ -z "$PR_NUMBER" ]] || [[ -z "$COMMENT_ID" ]] || [[ -z "$REPLY_TEXT" ]]; then
    echo "Usage: $0 <pr-number> <comment-id> <reply-text>"
    echo "Example: $0 42 123456 '✅ Fixed - updated the code as requested'"
    exit 1
fi

# Get repo info
REPO=$(gh repo view --json nameWithOwner -q '.nameWithOwner')

echo "=== Replying to Comment #$COMMENT_ID on PR #$PR_NUMBER ==="
echo

# Verify comment exists
if ! gh api "repos/$REPO/pulls/$PR_NUMBER/comments" | jq -e ".[] | select(.id == $COMMENT_ID)" &>/dev/null; then
    echo "Error: Comment #$COMMENT_ID not found on PR #$PR_NUMBER"
    exit 1
fi

# Post reply
echo "Posting reply..."
RESPONSE=$(gh api "repos/$REPO/pulls/$PR_NUMBER/comments/$COMMENT_ID/replies" \
    --method POST \
    -f body="$REPLY_TEXT" 2>&1)

if [[ $? -eq 0 ]]; then
    echo "✅ Reply posted successfully"
    echo
    echo "Reply ID: $(echo "$RESPONSE" | jq -r '.id')"
    echo "Reply text: $REPLY_TEXT"
else
    echo "❌ Failed to post reply"
    echo "$RESPONSE"
    exit 1
fi

# Verify reply is visible
echo
echo "Verifying reply..."
sleep 2
if gh api "repos/$REPO/pulls/$PR_NUMBER/comments" | jq -e ".[] | select(.in_reply_to_id == $COMMENT_ID)" &>/dev/null; then
    echo "✅ Reply verified in comment thread"
else
    echo "⚠️  Reply may not be visible yet (check GitHub UI)"
fi

exit 0
