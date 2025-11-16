---
name: gh-create-pr-linked
description: Create a pull request that is properly linked to a GitHub issue using gh pr create --issue flag. Use when creating a PR that implements or addresses a specific issue to ensure automatic linking.
---

# Create PR Linked to Issue Skill

This skill creates a pull request with automatic issue linking using the GitHub CLI.

## When to Use

- User asks to create a PR (e.g., "create a PR for issue #42")
- After completing implementation work
- When ready to submit changes for review
- Need to link PR to GitHub issue automatically

## Usage

### Basic PR Creation

```bash
# Create PR linked to issue (preferred method)
gh pr create --issue <issue-number>

# Create PR with title and body
gh pr create --title "Title" --body "$(cat <<'EOF'
## Summary
- Brief description

## Changes
- List of changes

Closes #<issue-number>
EOF
)"
```

### Workflow

1. **Verify changes are committed and pushed**
2. **Check current branch tracks remote** (use `git status`)
3. **Push branch if needed**: `git push -u origin branch-name`
4. **Create PR with issue link**: `gh pr create --issue <number>`
5. **Verify link appears in issue** (check issue's Development section)

## PR Requirements

- ✅ PR must be linked to GitHub issue
- ✅ All changes committed and pushed
- ✅ CI checks should be passing
- ✅ Title should be clear and descriptive
- ✅ Description should summarize changes
- ❌ Do NOT create PR without linking to issue

## Error Handling

- **No upstream branch**: Run `git push -u origin branch-name` first
- **Issue not found**: Verify issue number exists
- **Auth failure**: Check `gh auth status` and refresh if needed
- **Link not appearing**: Add "Closes #<issue-number>" to PR description

## Verification

After creating PR, verify:

```bash
# Check PR was created
gh pr view <pr-number>

# Check CI status
gh pr checks <pr-number>

# Verify issue link (check Development section in issue)
gh issue view <issue-number>
```

## Examples

**Create PR linked to issue:**
```bash
./scripts/create_linked_pr.sh 42
```

**Create PR with custom body:**
```bash
./scripts/create_linked_pr.sh 42 "Custom description of changes"
```

## Scripts Available

- `scripts/create_linked_pr.sh` - Create PR with issue linking
- `scripts/verify_pr_link.sh` - Verify PR is linked to issue

## Templates

- `templates/pr_body.md` - Standard PR description template

See CLAUDE.md for complete PR creation workflow and requirements.
