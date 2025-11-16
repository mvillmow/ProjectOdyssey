#!/usr/bin/env bash
#
# Create a git worktree for an issue
#
# Usage:
#   ./create_worktree.sh <issue-number> <description> [parent-dir]
#
# Example:
#   ./create_worktree.sh 42 "implement-tensor-ops"
#   ./create_worktree.sh 42 "implement-tensor-ops" "/tmp/worktrees"

set -euo pipefail

ISSUE_NUMBER="${1:-}"
DESCRIPTION="${2:-}"
PARENT_DIR="${3:-}"

if [[ -z "$ISSUE_NUMBER" ]] || [[ -z "$DESCRIPTION" ]]; then
    echo "Error: Issue number and description required"
    echo "Usage: $0 <issue-number> <description> [parent-dir]"
    exit 1
fi

# Get repository info
REPO_DIR=$(git rev-parse --show-toplevel)
REPO_NAME=$(basename "$REPO_DIR")

# Determine parent directory for worktree
if [[ -z "$PARENT_DIR" ]]; then
    PARENT_DIR=$(dirname "$REPO_DIR")
fi

# Create branch name
BRANCH_NAME="${ISSUE_NUMBER}-${DESCRIPTION}"

# Create worktree directory name
WORKTREE_DIR="${PARENT_DIR}/${REPO_NAME}-${ISSUE_NUMBER}-${DESCRIPTION}"

# Check if worktree already exists
if [[ -d "$WORKTREE_DIR" ]]; then
    echo "Error: Worktree directory already exists: $WORKTREE_DIR"
    exit 1
fi

# Check if branch already exists
if git show-ref --verify --quiet "refs/heads/$BRANCH_NAME"; then
    echo "Warning: Branch already exists: $BRANCH_NAME"
    read -p "Use existing branch? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted"
        exit 1
    fi
    # Create worktree from existing branch
    git worktree add "$WORKTREE_DIR" "$BRANCH_NAME"
else
    # Create new branch and worktree
    git worktree add -b "$BRANCH_NAME" "$WORKTREE_DIR"
fi

echo ""
echo "✅ Worktree created successfully!"
echo ""
echo "Branch: $BRANCH_NAME"
echo "Location: $WORKTREE_DIR"
echo ""
echo "To start working:"
echo "  cd $WORKTREE_DIR"
echo ""
echo "To remove when done:"
echo "  git worktree remove $WORKTREE_DIR"
