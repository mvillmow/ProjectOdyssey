# Verify Issue Before Work

## Overview

| Field | Value |
|-------|-------|
| **Date** | 2025-12-29 |
| **Objective** | Continue P2 issue implementation, specifically Training Dashboard (#2672) |
| **Outcome** | ❌ Wasted effort - Issue was already closed and merged |
| **Root Cause** | Did not verify issue state before starting work |
| **Key Learning** | Always check `gh issue view <number> --json state` and existing code BEFORE planning/implementing |

## When to Use

Use this verification workflow **BEFORE** starting any implementation work:

- User says "continue with [feature]" or "work on issue #XXX"
- Starting work on an issue from a backlog or roadmap
- Resuming work after a break or context switch
- Taking over work from another session

## Verified Workflow

### Step 1: Check Issue State First

```bash
# ALWAYS run this FIRST - before any planning or implementation
gh issue view <issue-number> --json state,title,closedAt

# Example output if closed:
# {"closedAt":"2025-12-29T20:37:46Z","state":"CLOSED","title":"Add Training Dashboard"}

# If state is "CLOSED", STOP and move to next issue
```

### Step 2: Check for Existing Implementation

```bash
# Look for related commits
git log --all --oneline --grep="<issue-number>" | head -5

# Look for feature-specific commits
git log --all --oneline --grep="<feature-keyword>" | head -10

# Check if files exist
ls -la <expected-directory>
```

### Step 3: Verify with PR Search

```bash
# Check for merged PRs
gh pr list --search "<feature-keyword>" --state merged --json number,title

# Check for open PRs
gh pr list --search "<feature-keyword>" --state open --json number,title
```

### Step 4: Only Then Start Work

If all checks pass (issue open, no existing implementation), proceed with:
- Reading issue details
- Planning implementation
- Creating feature branch

## Failed Attempts

### ❌ Attempt 1: Started Work Without Verification

**What I Did**:
1. Updated todo list with #2672 as "in_progress"
2. Read issue #2672 to understand requirements
3. Used Glob to find existing infrastructure
4. Read `csv_metrics_logger.mojo` to understand format
5. Attempted to create dashboard directory

**What Went Wrong**:
- Spent 5+ tool calls on planning and exploration
- Discovered dashboard already exists at `scripts/dashboard/`
- Issue was already closed on 2025-12-29T20:37:46Z
- Commits `10f471fa` and `0260473f` already implemented it

**Why It Failed**:
- **Skipped state verification** - Did not run `gh issue view 2672 --json state` first
- **Assumed issue was open** - Based on user saying "continue with training dashboard"
- **No existence check** - Should have checked `scripts/dashboard/` before planning

**Cost**:
- Wasted 7 tool calls
- Wasted ~5 minutes of exploration
- Created duplicate effort

### ❌ Attempt 2: File Write Without Read

**What I Did**:
```python
Write(file_path="/home/mvillmow/ProjectOdyssey/scripts/dashboard/server.py", content="...")
```

**Error**:
```
File has not been read yet. Read it first before writing to it.
```

**Why It Failed**:
- Tool requires reading existing files before overwriting
- Should have checked file existence with `ls` or `Read` first

## Results & Parameters

### Correct Verification Sequence

```bash
# 1. Check issue state (< 1 second)
gh issue view 2672 --json state,title,closedAt

# Output: {"closedAt":"2025-12-29T20:37:46Z","state":"CLOSED",...}
# Result: Issue CLOSED - STOP HERE, move to next issue

# 2. Find related commits (if state was OPEN)
git log --all --oneline --grep="2672"
git log --all --oneline --grep="dashboard"

# 3. Check directory existence
ls -la scripts/dashboard/

# 4. Only if all clear: proceed with implementation
```

### Time Saved

- **Without verification**: 7 tool calls, ~5 minutes wasted
- **With verification**: 1 tool call, ~2 seconds, immediate answer

### Prevention Checklist

Before ANY implementation work:

- [ ] Run `gh issue view <N> --json state` - verify state is "OPEN"
- [ ] Run `git log --grep="<N>"` - check for existing commits
- [ ] Run `ls -la <expected-path>` - verify directory doesn't exist
- [ ] Run `gh pr list --search "<keyword>"` - check for merged PRs
- [ ] Only then: Read issue, plan, implement

## References

See `references/notes.md` for:
- Full conversation transcript
- Tool call sequence
- Error messages
- Commit hashes and timestamps

## Tags

`workflow`, `github`, `productivity`, `verification`, `issue-management`
