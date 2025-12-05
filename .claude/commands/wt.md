Worktree management command: $ARGUMENTS

Supported subcommands:

**list** - Show all worktrees

```bash
git worktree list
```

**create issue-number** - Create worktree for a GitHub issue

```bash
# Get issue title for branch name
gh issue view <issue> --json title
git worktree add worktree/<issue>-<desc> -b <issue>-<desc>
```

**cleanup** - Remove merged worktrees and prune stale references

```bash
# For each worktree, check if branch is merged to main
git worktree remove <path>
git worktree prune
```

**switch issue-number** - Change to the worktree directory

```bash
cd worktree/<issue>-*
```

Execute the requested operation based on the first word of $ARGUMENTS.
