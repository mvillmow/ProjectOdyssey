---
name: gh-fix-pr-feedback
description: Automatically address PR review feedback by making requested changes and replying to all review comments. Use when a PR has review comments that need to be addressed.
---

# Fix PR Review Feedback Skill

This skill automates the process of addressing PR review feedback by making changes and replying to comments.

## When to Use

- User asks to fix PR feedback (e.g., "address review comments on PR #42")
- PR has open review comments that need responses
- Need to implement reviewer's requested changes
- Ready to push fixes and notify reviewers

## Workflow

### 1. Get Review Comments

```bash
# Fetch all review comments
./scripts/get_feedback.sh <pr-number>

# This retrieves:
# - Comment ID
# - File path
# - Line number
# - Comment text
# - Reviewer username
```

### 2. Make Changes

Address each review comment by:
- Fixing code issues
- Implementing suggestions
- Refactoring as requested
- Adding/updating tests

### 3. Reply to Comments

```bash
# Reply to all comments after fixing
./scripts/reply_to_feedback.sh <pr-number> "✅ Fixed - [description]"

# Or reply to specific comment
gh api repos/OWNER/REPO/pulls/PR/comments/COMMENT_ID/replies \
  --method POST \
  -f body="✅ Fixed - Updated logic as suggested"
```

### 4. Verify and Push

```bash
# Commit changes
git add .
git commit -m "fix: address PR review feedback"

# Push changes
git push

# Verify CI passes
./scripts/check_ci.sh <pr-number>
```

## Reply Format

Always use concise, clear replies:

**Good replies:**
- `✅ Fixed - Updated conftest.py to use real repository root`
- `✅ Fixed - Removed duplicate test file`
- `✅ Fixed - Added error handling for edge case`
- `✅ Done - Renamed variable for clarity`

**Bad replies:**
- "Done" (not specific)
- Long explanations (keep it brief)
- Defensive responses (be constructive)

## CRITICAL: Two Types of Comments

### 1. PR-level Comments
General comments in PR timeline:
```bash
gh pr comment <pr-number> --body "Response"
```

### 2. Review Comment Replies
Inline code review comment replies:
```bash
gh api repos/OWNER/REPO/pulls/PR/comments/COMMENT_ID/replies \
  --method POST \
  -f body="✅ Fixed - description"
```

**DO NOT confuse these!** Use the correct API for review comments.

## Complete Workflow Script

```bash
# Complete workflow to fix PR feedback
./scripts/fix_all_feedback.sh <pr-number>

# This script:
# 1. Gets all review comments
# 2. Displays them for review
# 3. Waits for you to make fixes
# 4. Helps you reply to each comment
# 5. Verifies CI status
```

## Error Handling

- **Comment not found**: Verify comment ID is correct
- **Auth failure**: Check `gh auth status`
- **CI fails after push**: Check logs and fix issues
- **Reply not appearing**: Verify using correct API endpoint

## Verification Checklist

After addressing feedback:

- [ ] All review comments have replies
- [ ] Changes committed and pushed
- [ ] CI checks passing
- [ ] No new issues introduced
- [ ] Replies are visible on GitHub

## Examples

**Fix all feedback:**
```bash
./scripts/fix_all_feedback.sh 42
```

**Reply to specific comment:**
```bash
./scripts/reply_to_comment.sh 42 123456 "✅ Fixed - Added validation"
```

**Check if all comments addressed:**
```bash
./scripts/check_feedback_status.sh 42
```

## Scripts Available

- `scripts/get_feedback.sh` - Get all review comments
- `scripts/reply_to_feedback.sh` - Reply to comments
- `scripts/fix_all_feedback.sh` - Complete workflow
- `scripts/check_feedback_status.sh` - Verify all addressed

## Templates

- `templates/reply_template.txt` - Standard reply format

See `/agents/guides/github-review-comments.md` for detailed guide on handling review comments.
