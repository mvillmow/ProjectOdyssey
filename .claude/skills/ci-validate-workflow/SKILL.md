---
name: ci-validate-workflow
description: Validate GitHub Actions workflow files for syntax, best practices, and correctness. Use before committing workflow changes or when workflows fail.
---

# CI Workflow Validation Skill

Validate GitHub Actions workflow files.

## When to Use

- Creating new workflow
- Modifying existing workflow
- Workflow syntax errors
- Before committing workflow changes

## Validation Methods

### 1. GitHub CLI

```bash
# View workflow
gh workflow view <workflow-name>

# List workflows
gh workflow list

# Check workflow syntax (via API)
gh api repos/{owner}/{repo}/actions/workflows
```

### 2. actionlint

```bash
# Install
pip install actionlint

# Validate workflows
actionlint .github/workflows/*.yml
```

### 3. yamllint

```bash
# Validate YAML syntax
yamllint .github/workflows/*.yml
```

## Common Issues

### 1. Syntax Errors

```yaml
# ❌ Wrong
on: [push pull_request]

# ✅ Correct
on: [push, pull_request]
```

### 2. Invalid Actions

```yaml
# ❌ Wrong version
- uses: actions/checkout@v99

# ✅ Correct
- uses: actions/checkout@v4
```

### 3. Missing Required Fields

```yaml
# ❌ Missing 'runs-on'
jobs:
  test:
    steps:
      - run: echo "test"

# ✅ Complete
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - run: echo "test"
```

## Best Practices

- Use specific action versions (@v4, not @main)
- Pin to commit SHA for security
- Use `if` conditions to save CI time
- Set explicit timeouts
- Cache dependencies

Example:

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v4
      - uses: actions/cache@v3
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
```

See GitHub Actions documentation for complete reference.
