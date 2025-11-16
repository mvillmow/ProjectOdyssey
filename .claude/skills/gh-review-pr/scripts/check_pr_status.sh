#!/usr/bin/env bash
# Check PR CI and review status

set -euo pipefail

PR_NUMBER="${1:-}"

if [[ -z "$PR_NUMBER" ]]; then
    echo "Usage: $0 <pr-number>"
    exit 1
fi

echo "=== PR #$PR_NUMBER Status ==="
echo

# Check if PR exists
if ! gh pr view "$PR_NUMBER" &>/dev/null; then
    echo "Error: PR #$PR_NUMBER not found"
    exit 1
fi

# Get PR state
echo "📋 PR Information:"
gh pr view "$PR_NUMBER" --json number,title,state,author,isDraft,mergeable | \
    jq -r '"Number: #\(.number)\nTitle: \(.title)\nState: \(.state)\nAuthor: \(.author.login)\nDraft: \(.isDraft)\nMergeable: \(.mergeable)"'

echo
echo "✅ CI Checks:"
gh pr checks "$PR_NUMBER" || echo "No CI checks found or checks failed"

echo
echo "💬 Review Status:"
gh pr view "$PR_NUMBER" --json reviewDecision,reviews | \
    jq -r '"Decision: \(.reviewDecision // "PENDING")\nReview Count: \(.reviews | length)"'

echo
echo "🔗 Issue Linkage:"
gh pr view "$PR_NUMBER" --json body | \
    jq -r '.body' | \
    grep -E '(Closes|Fixes|Resolves) #[0-9]+' || echo "⚠️  No issue linkage found"

exit 0
