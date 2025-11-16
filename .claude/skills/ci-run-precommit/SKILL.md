---
name: ci-run-precommit
description: Run pre-commit hooks locally or in CI to validate code quality before committing. Use to ensure commits meet quality standards and CI will pass.
---

# Run Pre-commit Hooks Skill

This skill runs pre-commit hooks to validate code quality before committing.

## When to Use

- User asks to run pre-commit (e.g., "run pre-commit hooks")
- Before committing code
- Testing if CI will pass
- After making code changes
- Troubleshooting commit failures

## Pre-commit Hooks

Configured in `.pre-commit-config.yaml`:

### 1. Mojo Format

```yaml
- id: mojo-format
  name: Mojo Format
  entry: mojo format
  language: system
  files: \.(mojo|🔥)$
```

### 2. Trailing Whitespace

```yaml
- id: trailing-whitespace
  name: Trim Trailing Whitespace
```

### 3. End of File Fixer

```yaml
- id: end-of-file-fixer
  name: Fix End of Files
```

### 4. YAML Check

```yaml
- id: check-yaml
  name: Check YAML
```

### 5. Large Files Check

```yaml
- id: check-added-large-files
  name: Check for Large Files
  args: ['--maxkb=1000']
```

### 6. Mixed Line Ending

```yaml
- id: mixed-line-ending
  name: Fix Mixed Line Endings
```

### 7. Markdownlint (Disabled Currently)

```yaml
# Will enable after fixing existing files
- id: markdownlint-cli2
  name: Markdown Lint
```

## Usage

### Run All Hooks

```bash
# Run all hooks on all files
pre-commit run --all-files

# Run all hooks on staged files only
pre-commit run

# Automatically runs on commit
git commit -m "message"
```

### Run Specific Hook

```bash
# Run specific hook
pre-commit run trailing-whitespace --all-files

# Run on specific file
pre-commit run --files src/tensor.mojo
```

### Install Hooks

```bash
# Install pre-commit hooks (one-time setup)
pre-commit install

# Now hooks run automatically on git commit
```

## Hook Behavior

### Auto-Fix Hooks

These hooks fix issues automatically:
- `trailing-whitespace` - Removes trailing spaces
- `end-of-file-fixer` - Adds final newline
- `mixed-line-ending` - Fixes line endings
- `mojo-format` - Formats Mojo code

**Workflow when auto-fix runs:**
```bash
git commit -m "message"
# Hooks run, fix files, commit aborts
# Files are fixed but not staged

git add .  # Stage the fixes
git commit -m "message"  # Commit again
```

### Check-Only Hooks

These hooks check but don't fix:
- `check-yaml` - Reports YAML errors
- `check-added-large-files` - Reports large files

**If these fail:** Fix manually and re-commit

## CI Integration

Pre-commit runs in CI:

```yaml
name: Pre-commit Checks

on: [push, pull_request]

jobs:
  pre-commit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - run: pip install pre-commit
      - run: pre-commit run --all-files
```

## Common Issues

### Hooks Fail on Commit

```bash
$ git commit -m "message"
Trim Trailing Whitespace....Failed
- hook id: trailing-whitespace
- exit code: 1
- files were modified by this hook

Files were modified by this hook. Additional output:

Fixing file.md
```

**Fix:**
```bash
# Files were fixed, just stage and re-commit
git add .
git commit -m "message"
```

### YAML Validation Fails

```text
Check YAML...Failed
- hook id: check-yaml
- exit code: 1

Syntax error in .github/workflows/test.yml
```

**Fix:** Correct YAML syntax

### Large File Detected

```text
Check for Large Files...Failed
- hook id: check-added-large-files
- exit code: 1

large_file.bin (1500 KB) exceeds 1000 KB
```

**Fix:**
- Don't commit large files
- Use Git LFS if needed
- Add to `.gitignore`

## Skipping Hooks

**Only when necessary:**

```bash
# Skip all hooks (not recommended)
git commit --no-verify -m "message"

# Skip specific hook
SKIP=trailing-whitespace git commit -m "message"
```

**When to skip:**
- Emergency hotfix
- Hook has bug
- False positive

**Don't skip** to bypass quality checks!

## Examples

**Run all hooks:**
```bash
pre-commit run --all-files
```

**Run before committing:**
```bash
# Check if commit will pass
pre-commit run

# If passing, commit
git commit -m "message"
```

**Install hooks:**
```bash
pre-commit install
```

**Update hooks:**
```bash
pre-commit autoupdate
```

## Scripts Available

- `scripts/run_precommit.sh` - Run all hooks
- `scripts/install_precommit.sh` - Install hooks
- `scripts/update_precommit.sh` - Update hook versions

## Best Practices

1. **Install hooks** - Run `pre-commit install` once
2. **Let hooks fix** - Don't bypass, let them auto-fix
3. **Run before push** - Verify CI will pass
4. **Don't skip** - Only skip in emergencies
5. **Keep updated** - Run `pre-commit autoupdate` periodically

## Workflow Integration

### First Time Setup

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Test
pre-commit run --all-files
```

### Daily Workflow

```bash
# Make changes
# ...

# Commit (hooks run automatically)
git commit -m "feat: new feature"

# If hooks fix files
git add .
git commit -m "feat: new feature"
```

### Before PR

```bash
# Verify all files pass
pre-commit run --all-files

# If passing, create PR
gh pr create --issue 42
```

See `.pre-commit-config.yaml` for complete hook configuration.
