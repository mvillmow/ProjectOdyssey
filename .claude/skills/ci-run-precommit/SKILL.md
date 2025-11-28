---
name: ci-run-precommit
description: Run pre-commit hooks locally or in CI to validate code quality before committing. Use to ensure commits meet quality standards and CI will pass.
mcp_fallback: none
category: ci
---

# Run Pre-commit Hooks Skill

Validate code quality with pre-commit hooks before committing.

## When to Use

- Before committing code
- Testing if CI will pass
- After making code changes
- Troubleshooting commit failures

## Quick Reference

```bash
# Install hooks (one-time)
pre-commit install

# Run on all files
pre-commit run --all-files

# Run on staged files
pre-commit run

# Skip hooks (emergency only)
git commit --no-verify
```

## Configured Hooks

| Hook | Purpose | Auto-Fix |
|------|---------|----------|
| `mojo-format` | Format Mojo code | Yes |
| `trailing-whitespace` | Remove trailing spaces | Yes |
| `end-of-file-fixer` | Add final newline | Yes |
| `check-yaml` | Validate YAML syntax | No |
| `check-added-large-files` | Prevent large files | No |
| `mixed-line-ending` | Fix line endings | Yes |

## Workflow

```bash
# 1. Make changes
# ... edit files ...

# 2. Run hooks on staged files
pre-commit run

# 3. If hooks auto-fixed files
git add .              # Stage fixed files
git commit -m "feat: feature name"

# 4. If hooks reported errors
# Fix issues manually, then re-commit
```

## Hook Behavior

### Auto-Fix Hooks

These hooks fix issues automatically:

```bash
git commit -m "message"
# Hooks run, fix files, abort commit
# Files are fixed but not staged

git add .              # Stage fixes
git commit -m "message"  # Commit again
```

### Check-Only Hooks

These hooks report but don't fix:

```bash
git commit -m "message"
# check-yaml fails - fix manually

# Fix YAML syntax
git add .
git commit -m "message"
```

## Common Issues

| Error | Cause | Fix |
|-------|-------|-----|
| "Trailing whitespace" | Spaces at line end | Run hooks again (auto-fixed) |
| "Check YAML failed" | Invalid YAML syntax | Fix YAML manually |
| "Large file rejected" | File > 1MB | Use Git LFS or remove file |
| "Mixed line ending" | Inconsistent line endings | Run hooks again (auto-fixed) |

## Setup

```bash
# Install pre-commit (first time)
pip install pre-commit

# Install hooks (first time)
pre-commit install

# Hooks now run automatically on commit
```

## CI Integration

Pre-commit runs in GitHub Actions:

```yaml
jobs:
  pre-commit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - run: pip install pre-commit
      - run: pre-commit run --all-files
```

## Advanced Usage

```bash
# Run specific hook only
pre-commit run trailing-whitespace --all-files

# Run on specific file
pre-commit run --files src/tensor.mojo

# Update hook versions
pre-commit autoupdate

# Skip all hooks (emergency only)
git commit --no-verify -m "message"
```

## Error Handling

| Issue | Solution |
|-------|----------|
| Hooks not installed | Run `pre-commit install` |
| Hooks not running | Verify `.pre-commit-config.yaml` exists |
| All files modified after hook | Stage fixes and re-commit |
| Need to skip hook | Use `SKIP=hook-id git commit` |

## References

- Configuration: `.pre-commit-config.yaml`
- Related skill: `quality-fix-formatting` for manual fixes
- Related skill: `quality-run-linters` for all linters
