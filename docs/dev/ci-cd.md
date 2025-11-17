# CI/CD Pipeline

Continuous Integration and Continuous Deployment for ML Odyssey.

## Overview

ML Odyssey uses GitHub Actions for automated testing, linting, and deployment. Every pull request and push to main
triggers our CI/CD pipeline to ensure code quality and prevent regressions.

## CI Pipeline

### Workflow Files

Located in `.github/workflows/`:

- `test.yml` - Run test suite on every PR/push
- `pre-commit.yml` - Lint and format checks
- `test-agents.yml` - Validate agent configurations
- `docs.yml` - Build and deploy documentation

### Test Workflow

**File**: `.github/workflows/test.yml`

**Triggers**:

- Pull request creation/update
- Push to `main` branch
- Manual workflow dispatch

**Jobs**:

```yaml

name: Test Suite

on:
  pull_request:
  push:
    branches: [main]

jobs:
  test-mojo:
    runs-on: ubuntu-latest
    steps:

      - uses: actions/checkout@v4
      - name: Setup Mojo

        uses: modularml/setup-mojo@v1

      - name: Install dependencies

        run: pixi install

      - name: Run Mojo tests

        run: pixi run mojo test tests/

      - name: Upload coverage

        uses: codecov/codecov-action@v3

  test-python:
    runs-on: ubuntu-latest
    steps:

      - uses: actions/checkout@v4
      - name: Setup Python

        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies

        run: |
          pip install pytest pytest-cov
          pip install -r requirements.txt

      - name: Run Python tests

        run: pytest tests/ --cov=scripts --cov-report=xml

```text

### Pre-commit Workflow

**File**: `.github/workflows/pre-commit.yml`

**Purpose**: Enforce code quality standards

**Checks**:

- Markdown linting (`markdownlint-cli2`)
- Trailing whitespace removal
- End-of-file newline
- YAML syntax validation
- Large file prevention

```yaml

name: Pre-commit Checks

on: [pull_request, push]

jobs:
  pre-commit:
    runs-on: ubuntu-latest
    steps:

      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
      - name: Install pre-commit

        run: pip install pre-commit

      - name: Run pre-commit

        run: pre-commit run --all-files

```text

### Agent Validation Workflow

**File**: `.github/workflows/test-agents.yml`

**Purpose**: Validate agent YAML configurations

```yaml

name: Validate Agents

on:
  pull_request:
    paths:

      - '.claude/agents/**'
      - 'tests/agents/**'

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:

      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
      - name: Validate agent configs

        run: python3 tests/agents/validate_configs.py .claude/agents/

      - name: Test delegation patterns

        run: python3 tests/agents/test_delegation.py .claude/agents/

```text

## Local CI Simulation

Run CI checks locally before pushing:

```bash

# Run all pre-commit hooks
pre-commit run --all-files

# Run Mojo tests
pixi run mojo test tests/

# Run Python tests
pytest tests/

# Run agent validation
python3 tests/agents/validate_configs.py .claude/agents/

# Run specific workflow locally (requires act)
act -j test-mojo

```text

## Branch Protection

**Main branch** requires:

- All status checks passing
- At least 1 approval review
- Up-to-date with base branch
- No merge conflicts

**Configure in GitHub**:

Settings → Branches → Branch protection rules → `main`

- ✅ Require status checks to pass
- ✅ Require branches to be up to date
- ✅ Require pull request reviews (1)
- ✅ Dismiss stale reviews
- ✅ Require review from code owners

## CD Pipeline (Documentation)

### Documentation Deployment

**File**: `.github/workflows/docs.yml`

**Trigger**: Push to `main`

**Process**:

1. Build MkDocs site
2. Deploy to GitHub Pages
3. Update documentation URL

```yaml

name: Deploy Documentation

on:
  push:
    branches: [main]
    paths:

      - 'docs/**'
      - 'mkdocs.yml'

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:

      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
      - name: Install MkDocs

        run: pip install mkdocs-material

      - name: Build docs

        run: mkdocs build

      - name: Deploy to GitHub Pages

        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./site

```text

## CD Pipeline (Packages)

**Planned**: Automated package building and release

**Future workflow**: `.github/workflows/release.yml`

```yaml

name: Build and Release

on:
  release:
    types: [published]

jobs:
  build-package:
    runs-on: ubuntu-latest
    steps:

      - uses: actions/checkout@v4
      - name: Setup Mojo

        uses: modularml/setup-mojo@v1

      - name: Build .mojopkg

        run: mojo package shared/

      - name: Upload package

        uses: actions/upload-artifact@v3
        with:
          name: ml-odyssey-${{ github.ref_name }}.mojopkg
          path: shared.mojopkg

```text

## Caching Strategy

Speed up CI with caching:

```yaml

- name: Cache Pixi environment

  uses: actions/cache@v3
  with:
    path: ~/.pixi
    key: ${{ runner.os }}-pixi-${{ hashFiles('pixi.toml') }}

- name: Cache Python packages

  uses: actions/cache@v3
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}

```text

## Secrets Management

Required secrets in GitHub:

- `GITHUB_TOKEN` - Automatically provided
- `CODECOV_TOKEN` - For coverage reports (optional)

**Configure**: Settings → Secrets and variables → Actions

## Monitoring

### Status Badges

Add to README.md:

```markdown

![Tests](https://github.com/owner/ml-odyssey/workflows/Test%20Suite/badge.svg)
![Pre-commit](https://github.com/owner/ml-odyssey/workflows/Pre-commit%20Checks/badge.svg)
![Docs](https://github.com/owner/ml-odyssey/workflows/Deploy%20Documentation/badge.svg)

```text

### Workflow Run History

View at: `<https://github.com/owner/ml-odyssey/actions`>

## Troubleshooting

### Workflow Fails on Fork

**Issue**: Workflows fail with permissions error on forks

**Solution**: Allow workflows in fork settings or disable for external contributors

### Cache Invalidation

**Issue**: Cached dependencies are stale

**Solution**: Update cache key or clear cache manually in Actions UI

### Test Timeouts

**Issue**: Tests exceed GitHub's time limits

**Solution**: Split into multiple jobs or optimize slow tests

## Best Practices

1. **Test Locally First** - Run pre-commit and tests before pushing
2. **Small PRs** - Easier to review and faster CI
3. **Clear Commit Messages** - Help with debugging CI failures
4. **Monitor CI Times** - Optimize if jobs take >10 minutes
5. **Use Caching** - Speed up workflows with appropriate caching
6. **Fail Fast** - Order jobs by speed, fail on first error

## Skills for CI/CD

- `ci-run-precommit` - Run pre-commit hooks locally
- `ci-validate-workflow` - Validate GitHub Actions syntax
- `ci-fix-failures` - Diagnose and fix CI failures
- `gh-check-ci-status` - Monitor CI status for PR

See `.claude/skills/` for details.

## Related Documentation

- [Testing Strategy](../core/testing-strategy.md) - What gets tested
- [Contributing Guide](../../CONTRIBUTING.md) - Development workflow
- [Agent Validation](../../tests/agents/README.md) - Agent testing
- [Pre-commit Hooks](../core/configuration.md#pre-commit-hooks) - Local hooks

## Summary

**CI Pipeline**:

- Automated testing (Mojo + Python)
- Code quality checks (pre-commit)
- Agent configuration validation
- Documentation building

**CD Pipeline**:

- Automatic docs deployment to GitHub Pages
- (Planned) Package building and releases

**Key Points**:

1. All PRs must pass CI before merge
2. Pre-commit hooks catch issues early
3. Tests run on every commit
4. Documentation auto-deploys on main
5. Use local simulation to debug

**Next Steps**:

- Set up branch protection
- Configure required status checks
- Add status badges to README
- Monitor workflow performance
