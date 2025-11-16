---
name: worktree-sync
description: Sync git worktrees with remote and main branch changes. Use to keep worktrees up-to-date during long-running development.
---

# Worktree Sync Skill

Keep worktrees synchronized with remote changes.

## When to Use

- Long-running feature branches
- Main branch has updates
- Need latest changes
- Before creating PR

## Sync Workflow

### 1. Fetch Latest

```bash
# In any worktree
git fetch origin
```

### 2. Update Main Worktree

```bash
# Switch to main worktree
cd ../ml-odyssey

# Pull latest
git pull origin main
```

### 3. Update Feature Worktree

```bash
# Switch to feature worktree
cd ../ml-odyssey-42-feature

# Rebase on main
git rebase origin/main

# Or merge if preferred
git merge origin/main
```

### Sync Script

```bash
# Sync all worktrees
./scripts/sync_all_worktrees.sh

# This:
# 1. Fetches from remote
# 2. Updates main worktree
# 3. Offers to rebase feature worktrees
```

## Best Practices

- Fetch regularly
- Sync before creating PR
- Resolve conflicts promptly
- Keep feature branches short-lived

## Common Issues

### Rebase Conflicts

```bash
# If conflicts during rebase
git status  # See conflicts
# Fix conflicts
git add .
git rebase --continue
```

### Diverged Branches

```bash
# If branch has diverged
git pull --rebase origin main
```

See `worktree-create` for creating worktrees.
