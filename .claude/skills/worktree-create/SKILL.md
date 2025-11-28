---
name: worktree-create
description: "Create isolated git worktrees for parallel development. Use when working on multiple issues simultaneously."
category: worktree
---

# Worktree Create

Create separate working directories on different branches without stashing changes.

## When to Use

- Starting work on a new issue
- Need to work on multiple issues in parallel
- Want to avoid stashing/context switching overhead
- Testing changes across different branches

## Quick Reference

```bash
# Create worktree for new branch
./scripts/create_worktree.sh <issue-number> <description>

# Example
./scripts/create_worktree.sh 42 "implement-tensor-ops"
# Creates: ../ml-odyssey-42-implement-tensor-ops/

# List all worktrees
git worktree list

# Switch worktrees
cd ../ml-odyssey-42-implement-tensor-ops
```

## Workflow

1. **Create worktree** - Run create script with issue number and description
2. **Navigate** - `cd` to new worktree directory (parallel to main)
3. **Work normally** - Make changes, commit, push as usual
4. **Switch back** - `cd` to different worktree or main directory
5. **Clean up** - Remove worktree after PR merge (see `worktree-cleanup` skill)

## Error Handling

| Error | Solution |
|-------|----------|
| Branch already exists | Use different branch name or delete old branch |
| Directory exists | Choose different location or remove directory |
| Cannot switch away | Ensure all changes are committed |
| Permission denied | Check directory permissions |

## Directory Structure

```
parent-directory/
├── ml-odyssey/                    # Main worktree (main branch)
├── ml-odyssey-42-tensor-ops/      # Issue #42 worktree
├── ml-odyssey-73-bugfix/          # Issue #73 worktree
└── ml-odyssey-99-experiment/      # Experimental worktree
```

## Best Practices

- One worktree per issue (don't share branches)
- Use descriptive names: `<issue-number>-<description>`
- All worktrees share same `.git` directory
- Clean up after PR merge
- Each branch can only be checked out in ONE worktree

## References

- [worktree-strategy.md](../../../notes/review/worktree-strategy.md)
- `scripts/create_worktree.sh` implementation
- See `worktree-cleanup` skill for removing worktrees
