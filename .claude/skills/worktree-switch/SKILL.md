---
name: worktree-switch
description: Switch between git worktrees for parallel development on multiple branches. Use when working on multiple issues simultaneously.
---

# Worktree Switch Skill

Switch between git worktrees efficiently.

## When to Use

- Working on multiple issues
- Need to context switch
- Testing different branches
- Comparing implementations

## Usage

### List Worktrees

```bash
# See all worktrees
git worktree list

# Example output:
# /home/user/ml-odyssey              abc1234 [main]
# /home/user/ml-odyssey-42-feature   def5678 [42-feature]
# /home/user/ml-odyssey-73-bugfix    ghi9012 [73-bugfix]
```

### Switch Worktree

```bash
# Just cd to different worktree
cd ../ml-odyssey-42-feature

# Verify current worktree
git worktree list | grep "*"
```

### Quick Switch Script

```bash
# Switch by issue number
./scripts/switch_worktree.sh 42
# Changes to worktree for issue #42
```

## Tips

### Use Shell Aliases

```bash
# Add to ~/.bashrc or ~/.zshrc
alias wt='git worktree list'
alias wtcd='cd $(git worktree list | fzf | awk "{print \$1}")'
```

### Terminal Multiplexer

Use tmux/screen for persistent sessions per worktree:
```bash
# Session per worktree
tmux new -s issue-42
cd ../ml-odyssey-42-feature

# Switch sessions
tmux attach -t issue-42
```

## Best Practices

- One worktree per issue
- Clear naming (issue-number-description)
- Keep worktrees organized
- Clean up when done

See `worktree-create` and `worktree-cleanup` skills.
