# CI/CD Pipeline

Continuous integration and deployment infrastructure for ML Odyssey.

## Overview

ML Odyssey uses GitHub Actions for automated testing, code quality checks, and deployment. This guide covers
essential CI/CD pipeline setup and workflows.

## Pipeline Architecture

```text
┌─────────────┐
│   Push/PR   │
└──────┬──────┘
       │
       ├─────────────────┬──────────────────┬──────────────────┐
       │                 │                  │                  │
┌──────▼──────┐  ┌──────▼──────┐  ┌────────▼────────┐  ┌──────▼──────┐
│   Tests     │  │  Pre-commit │  │  Agent Config   │  │   Docs      │
│             │  │   Checks    │  │   Validation    │  │   Build     │
└──────┬──────┘  └──────┬──────┘  └────────┬────────┘  └──────┬──────┘
       │                 │                  │                  │
       └─────────────────┴──────────────────┴──────────────────┘
                                │
                         ┌──────▼──────┐
                         │   Merge     │
                         └──────┬──────┘
                                │
                         ┌──────▼──────┐
                         │   Deploy    │
                         └─────────────┘
```

## GitHub Actions Workflows

### Test Workflow

**File**: `.github/workflows/test.yml`

```yaml
name: Tests

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Install Pixi
        run: |
          curl -fsSL https://pixi.sh/install.sh | bash
          echo "$HOME/.pixi/bin" >> $GITHUB_PATH

      - name: Install dependencies
        run: pixi install

      - name: Run tests
        run: pixi run pytest tests/ --cov=shared --cov=papers

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
```

**Triggers**:

- Push to `main`
- Pull requests to `main`

**Steps**:

1. Checkout code
2. Install Pixi
3. Install dependencies
4. Run tests with coverage
5. Upload coverage report

### Pre-commit Checks

**File**: `.github/workflows/pre-commit.yml`

```yaml
name: Pre-commit

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  pre-commit:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install pre-commit
        run: pip install pre-commit

      - name: Run pre-commit
        run: pre-commit run --all-files
```

**Checks**:

- `mojo format` - Code formatting
- `markdownlint` - Markdown linting
- `trailing-whitespace` - Remove trailing spaces
- `end-of-file-fixer` - Ensure newline at EOF
- `check-yaml` - YAML syntax validation

### Agent Validation

**File**: `.github/workflows/test-agents.yml`

```yaml
name: Test Agents

on:
  push:
    branches: [main]
    paths:
      - '.claude/agents/**'
      - 'tests/agents/**'
  pull_request:
    branches: [main]
    paths:
      - '.claude/agents/**'
      - 'tests/agents/**'

jobs:
  validate-agents:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install pytest pyyaml

      - name: Validate agent configs
        run: python tests/agents/validate_configs.py .claude/agents/

      - name: Test agent loading
        run: python tests/agents/test_loading.py .claude/agents/

      - name: Test delegation patterns
        run: python tests/agents/test_delegation.py .claude/agents/

      - name: Test Mojo patterns
        run: python tests/agents/test_mojo_patterns.py .claude/agents/
```

**Triggers**:

- Changes to `.claude/agents/`
- Changes to `tests/agents/`

**Validation**:

- Agent YAML syntax
- Required fields present
- Delegation patterns valid
- Mojo patterns followed

### Documentation Build

**File**: `.github/workflows/docs.yml`

```yaml
name: Documentation

on:
  push:
    branches: [main]
    paths:
      - 'docs/**'
      - 'mkdocs.yml'
  pull_request:
    branches: [main]
    paths:
      - 'docs/**'
      - 'mkdocs.yml'

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install mkdocs mkdocs-material

      - name: Build documentation
        run: mkdocs build --strict

      - name: Deploy to GitHub Pages
        if: github.ref == 'refs/heads/main'
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./site
```

**Triggers**:

- Changes to `docs/`
- Changes to `mkdocs.yml`

**Steps**:

1. Build documentation
2. Check for broken links
3. Deploy to GitHub Pages (main only)

### Release Workflow

**File**: `.github/workflows/release.yml`

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Install Pixi
        run: |
          curl -fsSL https://pixi.sh/install.sh | bash
          echo "$HOME/.pixi/bin" >> $GITHUB_PATH

      - name: Install dependencies
        run: pixi install

      - name: Run full test suite
        run: pixi run pytest tests/

      - name: Run benchmarks
        run: pixi run mojo run benchmarks/run_all.mojo

      - name: Build documentation
        run: pixi run mkdocs build

      - name: Extract changelog
        id: changelog
        run: |
          VERSION=${GITHUB_REF#refs/tags/}
          CHANGELOG=$(sed -n "/## \[$VERSION\]/,/## \[/p" CHANGELOG.md | head -n -1)
          echo "changelog<<EOF" >> $GITHUB_OUTPUT
          echo "$CHANGELOG" >> $GITHUB_OUTPUT
          echo "EOF" >> $GITHUB_OUTPUT

      - name: Create Release
        uses: actions/create-release@v1
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        with:
          tag_name: ${{ github.ref }}
          release_name: Release ${{ github.ref }}
          body: ${{ steps.changelog.outputs.changelog }}
          draft: false
          prerelease: false
```

**Triggers**:

- Push tags matching `v*`

**Steps**:

1. Run full test suite
2. Run benchmarks
3. Build documentation
4. Extract changelog
5. Create GitHub release

## Quality Gates

### Required Checks

All PRs must pass:

- ✅ All tests pass
- ✅ Code coverage ≥ 90% (shared library)
- ✅ Pre-commit checks pass
- ✅ Agent validation passes
- ✅ Documentation builds successfully
- ✅ No merge conflicts
- ✅ At least one approval

### Branch Protection

**Main branch** protection rules:

```yaml
# .github/branch-protection.yml
main:
  protection:
    required_status_checks:
      strict: true
      contexts:
        - test
        - pre-commit
        - validate-agents
        - build-docs
    required_pull_request_reviews:
      required_approving_review_count: 1
      dismiss_stale_reviews: true
    enforce_admins: false
    restrictions: null
```

### Code Coverage

**Coverage requirements**:

- Shared library: ≥ 90%
- Paper implementations: ≥ 80%
- Critical paths: 100%

**Configuration** (`.coveragerc`):

```ini
[run]
source = shared, papers
omit =
    */tests/*
    */conftest.py

[report]
precision = 2
show_missing = True
skip_covered = False

fail_under = 90
```

## Local CI Testing

### Run All Checks Locally

Before pushing:

```bash
# Run all tests
pixi run pytest tests/

# Run pre-commit checks
pre-commit run --all-files

# Validate agent configs
python tests/agents/validate_configs.py .claude/agents/

# Build documentation
pixi run mkdocs build
```

### Use Act for GitHub Actions

Test workflows locally with [act](https://github.com/nektos/act):

```bash
# Install act
brew install act

# Run test workflow
act -j test

# Run pre-commit workflow
act -j pre-commit

# Run all workflows
act
```

## Performance Benchmarking

### Benchmark Workflow

**File**: `.github/workflows/benchmark.yml`

```yaml
name: Benchmarks

on:
  pull_request:
    branches: [main]
    paths:
      - 'shared/**/*.mojo'
      - 'benchmarks/**'

jobs:
  benchmark:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3
        with:
          fetch-depth: 0  # Fetch all history for comparison

      - name: Install Pixi
        run: |
          curl -fsSL https://pixi.sh/install.sh | bash
          echo "$HOME/.pixi/bin" >> $GITHUB_PATH

      - name: Install dependencies
        run: pixi install

      - name: Run benchmarks
        run: pixi run mojo run benchmarks/run_all.mojo > current.json

      - name: Checkout main
        run: |
          git checkout main
          pixi run mojo run benchmarks/run_all.mojo > baseline.json
          git checkout -

      - name: Compare results
        run: |
          python scripts/compare_benchmarks.py \
            --current current.json \
            --baseline baseline.json \
            --output comparison.md

      - name: Comment on PR
        uses: actions/github-script@v6
        with:
          script: |
            const fs = require('fs');
            const comparison = fs.readFileSync('comparison.md', 'utf8');
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: comparison
            });
```

**Output**:

```markdown
## Benchmark Comparison

| Operation      | Main    | PR      | Change   |
| -------------- | ------- | ------- | -------- |
| matmul_1024    | 12.3ms  | 11.8ms  | -4.1% ✅ |
| conv2d_forward | 45.2ms  | 44.9ms  | -0.7% ✅ |
| relu_simd      | 0.8ms   | 0.8ms   | 0.0%     |

✅ No performance regressions detected
```

## Deployment

### Documentation Deployment

Automatic deployment to GitHub Pages:

```yaml
# Triggered on main branch push
- name: Deploy to GitHub Pages
  if: github.ref == 'refs/heads/main'
  uses: peaceiris/actions-gh-pages@v3
  with:
    github_token: ${{ secrets.GITHUB_TOKEN }}
    publish_dir: ./site
```

**URL**: `https://mvillmow.github.io/ml-odyssey/`

### Package Publishing (Future)

Planned: Publish to package registry on release.

## Monitoring and Alerts

### Build Status

**Badges** in README.md:

```markdown
[![Tests](https://github.com/mvillmow/ml-odyssey/workflows/Tests/badge.svg)](https://github.com/mvillmow/ml-odyssey/actions?query=workflow%3ATests)
[![Coverage](https://codecov.io/gh/mvillmow/ml-odyssey/branch/main/graph/badge.svg)](https://codecov.io/gh/mvillmow/ml-odyssey)
[![Docs](https://github.com/mvillmow/ml-odyssey/workflows/Documentation/badge.svg)](https://mvillmow.github.io/ml-odyssey/)
```

### Failure Notifications

GitHub Actions sends notifications on:

- Workflow failures
- Scheduled check failures
- Security alerts

**Configure**: GitHub Settings → Notifications

## Caching

### Dependency Caching

Speed up CI with caching:

```yaml
- name: Cache Pixi environment
  uses: actions/cache@v3
  with:
    path: ~/.pixi
    key: ${{ runner.os }}-pixi-${{ hashFiles('**/pixi.lock') }}
    restore-keys: |
      ${{ runner.os }}-pixi-

- name: Cache test results
  uses: actions/cache@v3
  with:
    path: .pytest_cache
    key: ${{ runner.os }}-pytest-${{ hashFiles('tests/**') }}
```

## Best Practices

### DO

- ✅ Run tests locally before pushing
- ✅ Keep workflows fast (< 5 minutes)
- ✅ Use caching for dependencies
- ✅ Monitor CI status
- ✅ Fix broken builds immediately
- ✅ Keep main branch green

### DON'T

- ❌ Merge with failing CI
- ❌ Skip CI checks
- ❌ Commit directly to main
- ❌ Ignore coverage drops
- ❌ Leave flaky tests
- ❌ Disable required checks

## Troubleshooting

### CI Failing Locally Passes

**Check**:

1. Environment differences (Python version, OS)
2. Missing files (not committed)
3. Timing issues (race conditions)
4. Network dependencies

**Solution**:

```bash
# Test in clean environment
docker run -it --rm -v $(pwd):/workspace python:3.11 /bin/bash
cd /workspace
pip install pixi
pixi run pytest tests/
```

### Flaky Tests

**Identify**:

```bash
# Run test multiple times
for i in {1..10}; do
  pixi run pytest tests/test_flaky.py || echo "Failed on run $i"
done
```

**Fix**:

- Add proper setup/teardown
- Remove timing dependencies
- Use deterministic randomness
- Isolate tests

### Slow CI

**Profile**:

```bash
# Time each step
pixi run pytest tests/ --durations=10
```

**Optimize**:

- Parallelize tests
- Cache dependencies
- Skip expensive operations in CI
- Use test markers (`pytest -m "not slow"`)

## See Also

- **[Release Process](release-process.md)** - Release workflow
- **[Testing Strategy](../core/testing-strategy.md)** - Writing testable code
- **[Workflow](../core/workflow.md)** - Development process
- [GitHub Actions Docs](https://docs.github.com/en/actions) - Official documentation
- [Pre-commit Hooks](../core/workflow.md) - Local checks
