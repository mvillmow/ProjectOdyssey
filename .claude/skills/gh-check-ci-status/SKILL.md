---
name: gh-check-ci-status
description: Check the CI/CD status of a pull request including workflow runs, test results, and check statuses. Use when verifying if PR checks are passing or investigating CI failures.
---

# Check CI Status Skill

This skill checks the CI/CD status of a pull request to verify all checks are passing.

## When to Use

- User asks to check CI status (e.g., "check CI for PR #42")
- Verifying PR is ready to merge
- Investigating CI failures
- Monitoring long-running CI jobs

## Usage

### Basic CI Check

```bash
# Check CI status for a PR
gh pr checks <pr-number>

# Example output:
# ✓ build          success  2m 34s
# ✗ test           failure  1m 12s
# ○ lint           pending  0m 45s
```

### Detailed Status

```bash
# Get detailed check information
gh pr view <pr-number> --json statusCheckRollup

# View specific workflow run
gh run view <run-id>

# Get logs for failed check
gh run view <run-id> --log-failed
```

### Watch CI Progress

```bash
# Watch CI status (updates in real-time)
gh pr checks <pr-number> --watch

# Or check every 30 seconds
while true; do
  clear
  gh pr checks <pr-number>
  sleep 30
done
```

## CI Check Types

### GitHub Actions Workflows

Common workflows in ML Odyssey:

- **test-agents** - Agent configuration validation
- **pre-commit** - Code formatting and linting
- **test-mojo** - Mojo unit tests (future)
- **validate-links** - Markdown link validation

### Status Indicators

- `✓` (checkmark) - Passing
- `✗` (x) - Failed
- `○` (circle) - Pending/In progress
- `-` - Skipped

## Common CI Failures

### Pre-commit Failures

```bash
# View pre-commit logs
gh run view <run-id> --log-failed

# Common issues:
# - Trailing whitespace
# - Missing newline at EOF
# - Markdown linting errors
# - mojo format needed
```

**Fix:**

```bash
pre-commit run --all-files
git add .
git commit --amend --no-edit
git push --force-with-lease
```

### Test Failures

```bash
# View test logs
gh run view <run-id> --log-failed

# Run tests locally
pytest tests/
mojo test tests/
```

### Workflow Validation

```bash
# Check workflow syntax
gh workflow view <workflow-name>

# Re-run failed workflow
gh run rerun <run-id>
```

## Error Handling

- **No checks found**: PR may not trigger CI (check workflow conditions)
- **Pending forever**: Check workflow logs for stuck jobs
- **Auth error**: Verify `gh auth status`
- **API rate limit**: Wait or use authenticated requests

## Verification Before Merge

Checklist:

- [ ] All required checks passing
- [ ] No pending checks
- [ ] Latest commit has checks
- [ ] Branch is up-to-date with base

```bash
# Complete verification
gh pr checks <pr-number>      # All checks passing?
gh pr view <pr-number>         # Up-to-date with base?
gh pr diff <pr-number>         # Changes look correct?
```

## Examples

**Check PR status:**

```bash
gh pr checks 42
```

**Watch CI progress:**

```bash
gh pr checks 42 --watch
```

**View failed logs:**

```bash
# Get run ID
gh pr checks 42

# View logs
gh run view 123456789 --log-failed
```

**Rerun failed checks:**

```bash
gh run rerun <run-id>
```

## Integration with Other Skills

- **gh-fix-pr-feedback** - After addressing feedback, check CI
- **gh-create-pr-linked** - After creating PR, verify CI starts
- **quality-run-linters** - Run locally before pushing to avoid CI failures

## Quick Reference

```bash
# Status overview
gh pr checks <pr>

# Detailed JSON
gh pr view <pr> --json statusCheckRollup

# Watch live
gh pr checks <pr> --watch

# Failed logs
gh run view <run-id> --log-failed

# Rerun
gh run rerun <run-id>
```

See `.github/workflows/` for complete CI configuration.
