# GitHub Review Comment API Reference

## API Endpoints

### List Review Comments

```
GET /repos/{owner}/{repo}/pulls/{pull_number}/comments
```

Returns all review comments on a pull request.

### Create Review Comment Reply

```
POST /repos/{owner}/{repo}/pulls/{pull_number}/comments/{comment_id}/replies
```

Creates a reply to a review comment.

## Important Distinctions

### Review Comments vs PR Comments

**Review Comments** (inline code comments):

- Attached to specific lines of code
- Created during code review
- Accessed via `/pulls/{pr}/comments` endpoint
- Can have threaded replies

**PR Comments** (general timeline comments):

- Not attached to code
- Show in PR timeline
- Created via `gh pr comment`
- Not what you want for review feedback!

## Example Workflow

### Complete Review Response Process

```bash
# 1. Get all review comments
gh api repos/OWNER/REPO/pulls/PR/comments > comments.json

# 2. Parse unresolved comments
jq '.[] | select(.in_reply_to_id == null) | {id: .id, path: .path, body: .body}' comments.json

# 3. For each comment, reply
gh api repos/OWNER/REPO/pulls/PR/comments/COMMENT_ID/replies \
  --method POST \
  -f body="✅ Fixed - description of what was done"

# 4. Verify replies
gh api repos/OWNER/REPO/pulls/PR/comments | \
  jq '.[] | select(.in_reply_to_id) | {id: .id, reply_to: .in_reply_to_id, body: .body}'

# 5. Check CI status
sleep 30  # Wait for CI to start
gh pr checks PR
```

## Response Templates

### Standard Fix Confirmation

```
✅ Fixed - [concise description of change]
```

### Need Clarification

```
❓ Question - [specific question about the request]
```

### Won't Fix (with reason)

```
⚠️ Skipped - [brief reason why not applicable]
```

### In Progress

```
🔄 Working on it - [expected completion]
```

## Common Errors

### Error: 404 Not Found

- Comment ID is invalid
- PR number is wrong
- Repository path is incorrect

### Error: 403 Forbidden

- Insufficient permissions
- Need to refresh auth: `gh auth refresh`

### Error: Comment not showing in thread

- Used wrong endpoint (probably `gh pr comment`)
- Comment ID was for different PR
- Network error during POST

## Best Practices

1. **Reply to ALL comments individually** - Don't batch responses
2. **Be concise** - Keep replies to 1 line when possible
3. **Use emojis for status** - ✅ Fixed, ❓ Question, ⚠️ Skipped
4. **Verify replies posted** - Always check that replies succeeded
5. **Check CI after pushing** - Ensure fixes don't break anything
6. **Don't assume local pre-commit matches CI** - Always verify

## Verification Checklist

After replying to comments:

- [ ] All review comments have replies
- [ ] Replies are visible in GitHub UI
- [ ] Changes have been pushed to branch
- [ ] CI checks are passing
- [ ] No new review comments added

## Debugging

If replies don't appear:

1. Check the API response for errors
2. Verify comment ID is correct
3. Check that PR number is correct
4. Ensure using `/replies` endpoint
5. Check `gh auth status`
6. Try viewing comment in GitHub UI
