#!/usr/bin/env bash
#
# Reply to a specific review comment
#
# Usage:
#   ./reply_to_comment.sh <pr-number> <comment-id> <reply-text>

set -euo pipefail

PR_NUMBER="${1:-}"
COMMENT_ID="${2:-}"
REPLY_TEXT="${3:-}"

if [[ -z "$PR_NUMBER" ]] || [[ -z "$COMMENT_ID" ]] || [[ -z "$REPLY_TEXT" ]]; then
    echo "Error: PR number, comment ID, and reply text required"
    echo "Usage: $0 <pr-number> <comment-id> <reply-text>"
    exit 1
fi

# Get repository info
REPO=$(gh repo view --json nameWithOwner -q '.nameWithOwner')

echo "Replying to comment #$COMMENT_ID on PR #$PR_NUMBER..."

# Post reply using GitHub API
gh api "repos/$REPO/pulls/$PR_NUMBER/comments/$COMMENT_ID/replies" \
    --method POST \
    -f body="$REPLY_TEXT"

echo "✅ Reply posted successfully"
