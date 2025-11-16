#!/usr/bin/env bash
#
# Remove a git worktree safely
#
# Usage:
#   ./remove_worktree.sh <issue-number>
#   ./remove_worktree.sh <worktree-path>

set -euo pipefail

INPUT="${1:-}"

if [[ -z "$INPUT" ]]; then
    echo "Error: Issue number or worktree path required"
    echo "Usage: $0 <issue-number|worktree-path>"
    exit 1
fi

# Get repository info
REPO_DIR=$(git rev-parse --show-toplevel)
REPO_NAME=$(basename "$REPO_DIR")
PARENT_DIR=$(dirname "$REPO_DIR")

# Determine worktree path
if [[ -d "$INPUT" ]]; then
    WORKTREE_DIR="$INPUT"
elif [[ "$INPUT" =~ ^[0-9]+$ ]]; then
    # Issue number provided - find matching worktree
    WORKTREE_DIR=$(git worktree list --porcelain | grep -A 2 "$REPO_NAME-$INPUT" | grep "worktree" | cut -d' ' -f2 || true)
    if [[ -z "$WORKTREE_DIR" ]]; then
        echo "Error: No worktree found for issue #$INPUT"
        echo "Available worktrees:"
        git worktree list
        exit 1
    fi
else
    echo "Error: Invalid input: $INPUT"
    echo "Provide issue number or worktree path"
    exit 1
fi

# Verify worktree exists
if [[ ! -d "$WORKTREE_DIR" ]]; then
    echo "Error: Worktree directory not found: $WORKTREE_DIR"
    exit 1
fi

# Check for uncommitted changes
cd "$WORKTREE_DIR"
if ! git diff-index --quiet HEAD --; then
    echo "Warning: Uncommitted changes in worktree"
    git status --short
    echo ""
    read -p "Remove anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted"
        exit 1
    fi
fi

# Return to original directory
cd "$REPO_DIR"

# Remove worktree
echo "Removing worktree: $WORKTREE_DIR"
git worktree remove "$WORKTREE_DIR" --force

echo "✅ Worktree removed successfully"
