---
name: worktree-cleanup
description: "Remove merged or stale git worktrees. Use after PRs are merged or when worktrees are no longer needed."
category: worktree
---

# Worktree Cleanup

Remove worktrees safely to free disk space and maintain organization.

## When to Use

- PR has been merged
- Worktree no longer needed
- Free disk space
- Maintain clean worktree list

## Quick Reference

```bash
# Remove single worktree by issue number
./scripts/remove_worktree.sh 42

# Or by path
git worktree remove ../ml-odyssey-42-feature

# Auto-clean all merged worktrees
./scripts/cleanup_merged_worktrees.sh

# Force remove (with uncommitted changes)
git worktree remove --force ../ml-odyssey-42-feature
```

## Workflow

1. **Verify state** - Check no uncommitted changes: `cd ../ml-odyssey-42 && git status`
2. **Switch away** - Don't be in the worktree you're removing
3. **Remove worktree** - Use removal script or git command
4. **Verify** - Run `git worktree list` to confirm removal
5. **Delete branch** - Optionally delete remote branch after cleanup

## Safety Checks

Before removing a worktree:

- Branch is merged to main (check GitHub PR status)
- No uncommitted changes (run `git status` in worktree)
- Not currently using the worktree (be in different directory)
- PR is actually merged (check "Development" section on issue)

## Error Handling

| Error | Solution |
|-------|----------|
| "Worktree has uncommitted changes" | Commit/stash changes or use `--force` |
| "Not a worktree" | Verify path with `git worktree list` |
| "Worktree is main" | Don't remove primary worktree |
| Directory still exists | Manually remove with `rm -rf` after `git worktree remove` |

## Scripts Available

- `scripts/remove_worktree.sh` - Remove single worktree
- `scripts/cleanup_merged_worktrees.sh` - Auto-clean merged worktrees
- `scripts/list_stale_worktrees.sh` - Find old/stale worktrees

## References

- See `worktree-create` skill for creating worktrees
- [worktree-strategy.md](../../../notes/review/worktree-strategy.md)
