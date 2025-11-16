---
name: gh-get-review-comments
description: Retrieve all open review comments from a pull request using the GitHub API. Use when you need to see what feedback has been provided on a PR or when checking for unresolved review comments.
---

# Get PR Review Comments Skill

This skill retrieves all review comments from a pull request, including both resolved and unresolved comments.

## When to Use

- User asks to get review comments (e.g., "get all comments on PR #42")
- Checking for unresolved review feedback
- Analyzing reviewer feedback
- Before fixing PR issues

## Usage

```bash
# Get all review comments for a PR
gh api repos/{owner}/{repo}/pulls/{pr-number}/comments

# Get comments filtered by reviewer
gh api repos/{owner}/{repo}/pulls/{pr-number}/comments --jq '.[] | select(.user.login == "username")'

# Get only unresolved comments
gh api repos/{owner}/{repo}/pulls/{pr-number}/comments --jq '.[] | select(.in_reply_to_id == null)'
```

## Output Format

Comments are returned with:
- `id` - Comment ID
- `path` - File path
- `line` - Line number
- `body` - Comment text
- `user` - Reviewer username
- `in_reply_to_id` - Parent comment ID (null if top-level)

## Error Handling

- If PR not found: Report error with PR number
- If auth fails: Check `gh auth status`
- If no comments: Return empty result, not an error

## Examples

**Get all comments:**
```bash
./scripts/fetch_comments.sh 42
```

**Filter by reviewer:**
```bash
./scripts/fetch_comments.sh 42 reviewer-username
```

**Get unresolved only:**
```bash
./scripts/fetch_comments.sh 42 --unresolved
```

## Script

Use `scripts/fetch_comments.sh` for structured comment retrieval.
