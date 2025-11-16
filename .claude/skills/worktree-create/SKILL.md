---
name: worktree-create
description: Create and manage git worktrees for parallel development across multiple branches. Use when working on multiple issues simultaneously or when isolating work environments.
---

# Git Worktree Creation Skill

This skill creates and manages git worktrees for parallel development on multiple branches.

## When to Use

- User asks to create a worktree (e.g., "create worktree for issue #42")
- Need to work on multiple issues simultaneously
- Want to isolate development environments
- Testing changes across different branches

## What are Worktrees?

Git worktrees allow multiple working directories from the same repository, each on a different branch.

**Benefits:**
- Work on multiple branches simultaneously
- No need to stash/commit when switching contexts
- Isolated environments for each issue
- Faster context switching

## Usage

### Create Worktree

```bash
# Create worktree for new branch
./scripts/create_worktree.sh <issue-number> <description>

# Example: Create worktree for issue #42
./scripts/create_worktree.sh 42 "implement-tensor-ops"
```

This creates:
- Branch: `42-implement-tensor-ops`
- Directory: `../ml-odyssey-42-implement-tensor-ops/`

### List Worktrees

```bash
# List all worktrees
git worktree list

# Example output:
# /home/user/ml-odyssey        abc1234 [main]
# /home/user/ml-odyssey-42     def5678 [42-implement-tensor-ops]
```

### Switch Between Worktrees

```bash
# Just cd to different worktree
cd ../ml-odyssey-42-implement-tensor-ops
```

### Remove Worktree

```bash
# Remove worktree when done
./scripts/remove_worktree.sh <issue-number>

# Or manually:
git worktree remove ../ml-odyssey-42-implement-tensor-ops
```

## Directory Structure

```text
parent-directory/
├── ml-odyssey/                    # Main worktree (main branch)
├── ml-odyssey-42-feature/         # Worktree for issue #42
├── ml-odyssey-73-bugfix/          # Worktree for issue #73
└── ml-odyssey-dev-experiment/     # Worktree for experimentation
```

## Best Practices

1. **One worktree per issue** - Keep work isolated
2. **Descriptive names** - Use issue number + description
3. **Clean up when done** - Remove merged worktrees
4. **Shared git directory** - All worktrees share same .git
5. **Independent branches** - Each worktree on different branch

## Workflow Example

```bash
# Create worktree for issue #42
./scripts/create_worktree.sh 42 "add-tensor-ops"

# Work in new worktree
cd ../ml-odyssey-42-add-tensor-ops
# Make changes, commit, push

# Meanwhile, switch to another issue
cd ../ml-odyssey-73-fix-bug
# Work on different issue without conflicts

# Return to original worktree
cd ../ml-odyssey
```

## Error Handling

- **Branch already exists**: Use different branch name or delete old branch
- **Directory exists**: Choose different location or remove directory
- **Worktree already exists**: Use `git worktree list` to find it
- **Cannot remove worktree**: Ensure no uncommitted changes

## Limitations

- **Cannot checkout same branch** in multiple worktrees
- **Shared .git directory** - Some operations affect all worktrees
- **Disk space** - Each worktree uses disk space

## Examples

**Create worktree for new feature:**
```bash
./scripts/create_worktree.sh 42 "tensor-operations"
# Creates: ../ml-odyssey-42-tensor-operations/
```

**Create worktree with custom location:**
```bash
./scripts/create_worktree.sh 42 "bugfix" "/tmp/worktrees"
# Creates: /tmp/worktrees/ml-odyssey-42-bugfix/
```

**List all worktrees:**
```bash
git worktree list
```

**Clean up merged worktrees:**
```bash
./scripts/cleanup_worktrees.sh
```

## Scripts Available

- `scripts/create_worktree.sh` - Create new worktree
- `scripts/remove_worktree.sh` - Remove worktree safely
- `scripts/list_worktrees.sh` - List all worktrees
- `scripts/cleanup_worktrees.sh` - Remove merged worktrees

See [worktree-strategy.md](/notes/review/worktree-strategy.md) for comprehensive worktree workflow documentation.
