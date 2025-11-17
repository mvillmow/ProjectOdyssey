---
name: gh-reply-review-comment
description: Reply to PR review comments using the correct GitHub API (not gh pr comment). Use when responding to inline code review feedback or marking review comments as resolved.
---

# Reply to PR Review Comments Skill

This skill provides the **correct** way to reply to PR review comments using the GitHub API, not `gh pr comment`.

## Critical Information

**NEVER use `gh pr comment`** - that creates a general PR comment, not a reply to review comments.

**CORRECT approach:**
Use the GitHub API to reply directly to review comment threads.

## When to Use

- Responding to review feedback
- Marking review comments as addressed
- Providing status updates on fixes
- Confirming changes have been made

## Correct API Usage

### Reply to a Review Comment

```bash
# Step 1: Get comment ID
gh api repos/{owner}/{repo}/pulls/{pr}/comments --jq '.[] | {id: .id, path: .path, body: .body}'

# Step 2: Reply to the comment
gh api repos/{owner}/{repo}/pulls/{pr}/comments/{comment-id}/replies \
  --method POST \
  -f body="✅ Fixed - [brief description]"

# Step 3: Verify reply posted
gh api repos/{owner}/{repo}/pulls/{pr}/comments --jq '.[] | select(.in_reply_to_id)'
```

## Reply Format

Keep responses **SHORT and CONCISE** (1 line preferred):

**Good examples:**

- `✅ Fixed - Updated conftest.py to use real repository root`
- `✅ Fixed - Deleted test file as requested`
- `✅ Fixed - Removed markdown linting section`

**Bad examples:**

- Long explanations (unless specifically asked)
- Defensive responses
- Multiple paragraphs

## Workflow

1. **Get review comment IDs** - Use script to list all comments
2. **Apply fixes** - Make the requested changes
3. **Reply to EACH comment** - Individually respond to each piece of feedback
4. **Verify replies posted** - Check that all replies succeeded
5. **Check CI status** - Ensure changes pass CI

## Error Handling

- If comment ID invalid: Check that you're using the correct ID
- If permission denied: Check `gh auth status`
- If reply fails: Verify PR and comment exist

## Scripts

- `scripts/reply_to_comment.sh` - Reply to a specific comment
- `scripts/reply_to_all.sh` - Reply to multiple comments at once

See reference.md for detailed API documentation and examples.
