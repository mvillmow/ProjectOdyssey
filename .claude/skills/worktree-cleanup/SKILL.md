---
name: worktree-cleanup
description: Clean up merged or stale git worktrees to free disk space and maintain organization. Use after merging PRs or when worktrees are no longer needed.
---

# Worktree Cleanup Skill

Clean up git worktrees after work is complete.

## When to Use

- After PR is merged
- Worktree no longer needed
- Free disk space
- Maintain clean worktree list

## Usage

### Remove Single Worktree

```bash
# Remove specific worktree
./scripts/remove_worktree.sh 42

# Or by path
git worktree remove ../ml-odyssey-42-feature
```

### Clean Up Merged Worktrees

```bash
# Remove all merged worktrees
./scripts/cleanup_merged_worktrees.sh

# This:
# 1. Checks which branches are merged
# 2. Finds worktrees for merged branches
# 3. Removes them (with confirmation)
```

### Force Remove

```bash
# Force remove (if has uncommitted changes)
git worktree remove --force ../ml-odyssey-42-feature
```

## Safety Checks

Before removing, verify:
- Branch is merged or no longer needed
- No uncommitted changes (or backed up)
- Not currently in worktree

```bash
# Check status before removing
cd ../ml-odyssey-42-feature
git status
cd ../ml-odyssey

# Then remove
git worktree remove ../ml-odyssey-42-feature
```

## Scripts

- `scripts/cleanup_merged_worktrees.sh` - Auto-clean merged
- `scripts/list_stale_worktrees.sh` - Find old worktrees
- `scripts/remove_worktree.sh` - Remove single worktree

See `worktree-create` skill for creating worktrees.
