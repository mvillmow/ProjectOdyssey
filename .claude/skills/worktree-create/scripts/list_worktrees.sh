#!/usr/bin/env bash
#
# List all git worktrees with details
#
# Usage:
#   ./list_worktrees.sh

set -euo pipefail

echo "Git Worktrees:"
echo ""

# Get worktree list
git worktree list

echo ""
echo "Details:"
echo ""

# Show detailed info for each worktree
git worktree list --porcelain | while IFS= read -r line; do
    if [[ "$line" =~ ^worktree ]]; then
        WORKTREE_PATH=$(echo "$line" | cut -d' ' -f2)
        echo "📁 $WORKTREE_PATH"
    elif [[ "$line" =~ ^branch ]]; then
        BRANCH=$(echo "$line" | cut -d'/' -f3-)
        echo "   Branch: $BRANCH"
    elif [[ "$line" =~ ^HEAD ]]; then
        COMMIT=$(echo "$line" | cut -d' ' -f2)
        echo "   Commit: ${COMMIT:0:8}"
    elif [[ -z "$line" ]]; then
        echo ""
    fi
done
