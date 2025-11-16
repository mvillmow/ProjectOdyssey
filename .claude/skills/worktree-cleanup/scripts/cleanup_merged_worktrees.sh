#!/usr/bin/env bash
#
# Clean up worktrees for merged branches
#
# Usage:
#   ./cleanup_merged_worktrees.sh

set -euo pipefail

echo "Finding merged worktrees..."
echo ""

# Get main branch
MAIN_BRANCH=$(git symbolic-ref refs/remotes/origin/HEAD | sed 's@^refs/remotes/origin/@@')

# Get merged branches
MERGED_BRANCHES=$(git branch --merged "$MAIN_BRANCH" | grep -v "^\*" | grep -v "$MAIN_BRANCH" | tr -d ' ' || true)

if [[ -z "$MERGED_BRANCHES" ]]; then
    echo "No merged branches found"
    exit 0
fi

echo "Merged branches:"
echo "$MERGED_BRANCHES"
echo ""

# Find worktrees for merged branches
REMOVED=0
while IFS= read -r branch; do
    # Find worktree for this branch
    WORKTREE=$(git worktree list --porcelain | grep -B 2 "branch refs/heads/$branch" | grep "^worktree" | cut -d' ' -f2 || true)

    if [[ -n "$WORKTREE" ]] && [[ "$WORKTREE" != "$(git rev-parse --show-toplevel)" ]]; then
        echo "Found worktree for merged branch '$branch': $WORKTREE"
        read -p "Remove? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            git worktree remove "$WORKTREE"
            echo "✅ Removed: $WORKTREE"
            ((REMOVED++))
        fi
    fi
done <<< "$MERGED_BRANCHES"

echo ""
echo "Cleaned up $REMOVED worktree(s)"
