# Pull Request Workflow

Shared PR workflow for all agents. Reference this file instead of duplicating.

## Creating a PR

```bash
git checkout -b {issue-number}-{description}
# Make changes, commit
git push -u origin {branch-name}
gh pr create --title "[Type] Description" --body "Closes #{issue}"
```

## Verification

After creating PR:

1. **Verify** the PR is linked to the issue (check issue page in GitHub)
2. **Confirm** link appears in issue's "Development" section
3. **If link missing**: Edit PR description to add "Closes #{number}"

## PR Requirements

- PR must be linked to GitHub issue
- PR title should be clear and descriptive
- PR description should summarize changes
- Do NOT create PR without linking to issue

## Responding to Review Comments

Reply to EACH review comment individually using GitHub API:

```bash
# Get review comment IDs
gh api repos/OWNER/REPO/pulls/PR/comments --jq '.[] | {id, path, body}'

# Reply to each comment
gh api repos/OWNER/REPO/pulls/PR/comments/COMMENT_ID/replies \
  --method POST -f body="Fixed - [brief description]"

# Verify replies posted
gh api repos/OWNER/REPO/pulls/PR/comments --jq '.[] | select(.in_reply_to_id)'
```

## Response Format

Keep responses SHORT and CONCISE (1 line preferred):

- `Fixed - Updated conftest.py to use real repository root`
- `Fixed - Deleted redundant test file`
- `Fixed - Removed deprecated section from README`

See [CLAUDE.md](../../CLAUDE.md#git-workflow) for complete PR creation instructions.
