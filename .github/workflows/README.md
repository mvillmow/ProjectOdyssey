# CI/CD Workflows

This directory contains all GitHub Actions workflows for the ML Odyssey project. These workflows provide automated
quality gates, security scanning, and validation for all code changes.

## Workflow Overview

### Tier 1: Fast Gates (< 5 minutes)

Required for all PRs - fast feedback on code quality.

| Workflow | File | Triggers | Duration | Purpose |
|----------|------|----------|----------|---------|
| Pre-commit Checks | `pre-commit.yml` | PRs, pushes to main | ~2-3 min | Code formatting, linting |
| Unit Tests | `unit-tests.yml` | PRs, pushes to main | ~5 min | Fast unit tests (Mojo + Python) |
| Agent Tests | `test-agents.yml` | PRs to agent configs | ~1 min | Agent configuration validation |

### Tier 2: Comprehensive Gates (< 10 minutes)

Required before merge - comprehensive validation.

| Workflow | File | Triggers | Duration | Purpose |
|----------|------|----------|----------|---------|
| Integration Tests | `integration-tests.yml` | PRs, pushes to main | ~8 min | Component interaction tests |
| Security Scanning | `security-scan.yml` | PRs, pushes, weekly | ~5 min | Dependencies, secrets, SAST |
| Build Validation | `build-validation.yml` | PRs, pushes to main | ~5 min | Build and packaging validation |
| Paper Validation | `paper-validation.yml` | PRs to papers/ | ~3 min | Paper implementation validation |
| Link Check | `link-check.yml` | PRs to markdown | ~1-2 min | Broken link detection |

### Tier 3: Performance Gates (Optional)

Scheduled or on-demand - performance validation.

| Workflow | File | Triggers | Duration | Purpose |
|----------|------|----------|----------|---------|
| Benchmarks | `benchmark.yml` | Manual, nightly, label | ~30 min | Performance benchmarking |

## Workflow Details

### 1. Pre-commit Checks (`pre-commit.yml`)

**Purpose**: Enforce code quality standards with automated formatting and linting.

**What it checks**:

- Mojo code formatting (`mojo format`)
- Markdown linting (`markdownlint-cli2`)
- YAML syntax validation
- Trailing whitespace, EOF newlines
- Large file detection (max 1MB)

**When it runs**: Every PR and push to main

**Local execution**:

```bash
# Install pre-commit
pip install pre-commit
pre-commit install

# Run all hooks
pre-commit run --all-files

# Run specific hook
pre-commit run mojo-format --all-files
```

---

### 2. Unit Tests (`unit-tests.yml`)

**Purpose**: Run fast unit tests for both Mojo and Python code with coverage reporting.

**What it does**:

- Runs Mojo unit tests (when they exist)
- Runs Python unit tests with pytest
- Generates coverage reports (target: 80% minimum, 90% ideal)
- Posts coverage summary to PR comments
- Fails if coverage < 80%

**When it runs**: Every PR and push to main

**Local execution**:

```bash
# Run Mojo tests
pixi run mojo test tests/unit/

# Run Python tests with coverage
pixi run pytest tests/unit/ --cov --cov-report=term

# Run both and check coverage threshold
pixi run pytest tests/unit/ --cov --cov-fail-under=80
```

**Expected output**: Coverage report in PR comment, artifacts with detailed results.

---

### 3. Integration Tests (`integration-tests.yml`)

**Purpose**: Validate component interactions across three test suites.

**Test suites**:

1. **agent-workflows**: Agent system integration
2. **paper-pipeline**: Paper implementation workflows
3. **data-pipeline**: Data loading and processing

**What it does**:

- Runs all three test suites in parallel (matrix strategy)
- Generates combined report
- Posts results to PR comments
- Fails if any suite fails

**When it runs**: Every PR (except drafts) and pushes to main

**Local execution**:

```bash
# Run specific integration suite
pixi run pytest tests/integration/agent-workflows/ -v

# Run all integration tests
pixi run pytest tests/integration/ -v
```

**Expected output**: Integration test report in PR comment with pass/fail status per suite.

---

### 4. Security Scanning (`security-scan.yml`)

**Purpose**: Comprehensive security scanning for vulnerabilities, secrets, and code issues.

**Scanning components**:

1. **Dependency Scan**: Check Python dependencies for known vulnerabilities (Safety, OSV Scanner)
2. **Secret Scan**: Detect exposed secrets (Gitleaks)
3. **SAST Scan**: Static Application Security Testing (Semgrep)
4. **Supply Chain Scan**: Dependency review for PRs (GitHub API)

**Severity handling**:

- **Critical**: Block merge immediately
- **High**: Block merge, require fix or exception
- **Medium**: Warning only
- **Low**: Info only

**When it runs**:

- Every PR and push to main
- Weekly schedule (Monday 9 AM UTC)
- Manual dispatch

**Local execution**:

```bash
# Run dependency scan
safety check --json

# Run secret scan
gitleaks detect --source . --verbose

# Run SAST
semgrep --config auto
```

**Expected output**: Security report in PR comment with findings by severity.

---

### 5. Build Validation (`build-validation.yml`)

**Purpose**: Validate that all packages build successfully.

**Build targets**:

1. **Mojo Packages**: Compile all Mojo source files
2. **Python Packages**: Build Python distributions (wheel, sdist)
3. **Documentation**: Build MkDocs site

**What it validates**:

- All source files compile without errors
- Package metadata is valid
- Documentation builds without errors
- Internal links are valid
- Smoke tests pass

**When it runs**: Every PR and push to main

**Local execution**:

```bash
# Build Mojo packages
mojo build src/your_file.mojo

# Build Python packages
python -m build

# Validate package metadata
twine check dist/*

# Build documentation
mkdocs build --strict
```

**Expected output**: Build report in PR comment with status for each build target.

---

### 6. Paper Validation (`paper-validation.yml`)

**Purpose**: Validate paper implementations against required structure and specifications.

**Validation stages**:

1. **Structure Validation** (~1 min):
   - Required files present (README.md, metadata.yml)
   - Required directories (implementation/, tests/)
   - Valid metadata structure (title, authors, year, URL)

2. **Implementation Validation** (~3 min):
   - Implementation files exist
   - Test files exist
   - Paper-specific tests pass

3. **Reproducibility Validation** (~10 min, optional):
   - Training scripts run successfully
   - Results match expected metrics (±5% tolerance)
   - Only runs with `validate-reproducibility` label

**When it runs**: PRs or pushes affecting `papers/` directory

**Local execution**:

```bash
# Validate paper structure
python scripts/validate_paper.py papers/lenet5/

# Run paper tests
pytest papers/lenet5/tests/ -v

# Run training for reproducibility
python papers/lenet5/train.py --quick-validation
```

**Expected output**: Paper validation report in PR comment with pass/fail for each stage.

---

### 7. Performance Benchmarks (`benchmark.yml`)

**Purpose**: Execute performance benchmarks and detect regressions.

**Benchmark suites**:

1. **tensor-ops**: Basic tensor operations (SIMD, vectorization)
2. **model-training**: End-to-end training performance
3. **data-loading**: Data pipeline throughput

**Regression thresholds**:

- **Critical** (> 25% slower): Block merge
- **High** (10-25% slower): Require justification
- **Medium** (5-10% slower): Warning only

**When it runs**:

- Manual dispatch (primary trigger)
- Nightly schedule (2 AM UTC)
- PRs with `benchmark` label

**Local execution**:

```bash
# Run specific benchmark suite
pixi run python benchmarks/tensor-ops/run_benchmark.py

# Run all benchmarks
pixi run python benchmarks/run_all.py
```

**Expected output**: Benchmark report with performance metrics and regression analysis.

---

### 8. Link Validation (`link-check.yml`)

**Purpose**: Detect broken links in markdown documentation.

**What it checks**:

- Internal links (relative paths)
- External links (HTTP/HTTPS)
- Anchor links within documents

**When it runs**:

- PRs or pushes affecting markdown files
- Weekly schedule (Sunday)
- Manual dispatch

**Local execution**:

```bash
# Check links with lychee
lychee --verbose --no-progress '**/*.md'
```

**Expected output**: List of broken links (if any).

---

### 9. Agent Tests (`test-agents.yml`)

**Purpose**: Validate agent configuration files.

**What it validates**:

- Agent markdown files are valid
- Delegation patterns are correct
- Mojo code patterns in agents are valid
- No broken links in agent docs

**When it runs**: PRs affecting `agents/` or `.claude/agents/`

**Local execution**:

```bash
# Run agent validation script
python scripts/validate_agents.py
```

**Expected output**: Validation report with any configuration errors.

---

## PR Workflow

When you open a PR, workflows run in this order:

```text
1. Pre-commit Checks (2-3 min)
   └─ If fails: Fix formatting issues
   └─ If passes: Continue

2. Fast Gates (parallel, 5 min)
   ├─ Unit Tests
   ├─ Agent Tests (if applicable)
   └─ Link Check (if applicable)

3. Comprehensive Gates (parallel, 10 min)
   ├─ Integration Tests
   ├─ Security Scanning
   ├─ Build Validation
   └─ Paper Validation (if applicable)

4. Optional: Benchmarks (if labeled, 30 min)
```

**Total time**: ~15-20 minutes for a typical PR (without benchmarks)

---

## Caching Strategy

All workflows use caching to improve performance:

| Cache | Key | Purpose |
|-------|-----|---------|
| Pixi environments | `pixi-${{ hashFiles('pixi.toml') }}` | Mojo dependencies |
| Python dependencies | `pip-${{ hashFiles('requirements*.txt') }}` | Python packages |
| Pre-commit | `pre-commit-${{ hashFiles('.pre-commit-config.yaml') }}` | Hook environments |
| Test data | `test-data-${{ hashFiles('tests/fixtures/**') }}` | Test fixtures |

**Cache hit rate target**: > 80%

---

## Troubleshooting

### Common Issues

#### Pre-commit failures

**Problem**: Formatting checks fail

**Solution**:

```bash
# Auto-fix most issues
pre-commit run --all-files

# Commit the fixes
git add .
git commit -m "fix: apply pre-commit auto-fixes"
```

#### Test coverage below threshold

**Problem**: Coverage < 80%

**Solution**: Add more tests or mark lines as no-cover if appropriate

```python
# Exclude specific lines
def debug_function():  # pragma: no cover
    print("Debug only")
```

#### Security scan failures

**Problem**: Secrets detected

**Solution**:

1. Remove the secret from code
2. Use environment variables or GitHub Secrets
3. Rotate the exposed credential

**Problem**: Dependency vulnerabilities

**Solution**:

```bash
# Update vulnerable package
pip install --upgrade <package-name>

# Or add exception in .safety-policy.yml (with justification)
```

#### Build failures

**Problem**: Mojo compilation errors

**Solution**: Run locally and fix syntax errors

```bash
mojo build src/your_file.mojo
```

#### Integration test failures

**Problem**: Flaky tests

**Solution**: Tests automatically retry once. If still failing, fix the underlying issue.

---

## Local CI Simulation

Run all CI checks locally before pushing:

```bash
# Create a script to run all checks
cat > scripts/ci-local.sh << 'EOF'
#!/bin/bash
set -e

echo "🔍 Running pre-commit hooks..."
pre-commit run --all-files

echo "🧪 Running unit tests..."
pixi run pytest tests/unit/ --cov --cov-fail-under=80

echo "🔗 Running integration tests..."
pixi run pytest tests/integration/ -v

echo "🔐 Running security scan..."
gitleaks detect --source . --verbose

echo "🏗️ Building packages..."
python -m build

echo "📚 Building documentation..."
mkdocs build --strict

echo "✅ All CI checks passed!"
EOF

chmod +x scripts/ci-local.sh
./scripts/ci-local.sh
```

---

## GitHub Actions Cost Optimization

**Free tier limit**: 2,000 minutes/month

**Estimated usage** (with optimizations):

- Pre-commit: ~40-60 min/day
- Unit tests: ~100 min/day
- Integration tests: ~80 min/day
- Security: ~25 min/day
- Build: ~100 min/day
- Other: ~50 min/day

**Total**: ~400 min/day ≈ 12,000 min/month (over free tier)

**Optimizations applied**:

1. Conditional execution (paper validation only on papers/ changes)
2. Aggressive caching (saves ~30-60s per run)
3. Parallel jobs where possible
4. Skip duplicate checks on draft PRs
5. Benchmarks optional/scheduled only

**Optimized estimate**: ~8,000 min/month (within free tier)

---

## Monitoring

### Key Metrics

Track these metrics for CI/CD health:

- **Workflow success rate**: Target > 95%
- **Average duration**: Fast gates < 5 min, comprehensive < 10 min
- **Cache hit rate**: Target > 80%
- **PR merge time**: Target < 24 hours for approved PRs

### Alerts

**Critical** (immediate):

- All workflows failing on main
- Critical security vulnerability
- Secrets exposed

**Warning** (daily):

- Workflow duration > 10 min
- Cache hit rate < 60%
- Test coverage drop > 5%

---

## Adding New Workflows

To add a new workflow:

1. Create `.github/workflows/your-workflow.yml`
2. Follow the template structure:

```yaml
name: Your Workflow

on:
  pull_request:
  push:
    branches:
      - main
  workflow_dispatch:

permissions:
  contents: read

jobs:
  your-job:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v4
      - name: Your step
        run: echo "Your commands"
```

1. Add caching where appropriate
2. Add timeout limits
3. Test locally first
4. Document in this README

---

## Branch Protection Rules

Configure for `main` branch:

**Required status checks**:

- pre-commit
- test-unit / test-mojo
- test-unit / test-python
- security / secret-scan
- security / sast-scan
- build / build-validation

**Required reviews**: 1

**Additional rules**:

- Dismiss stale reviews: true
- Require code owner reviews: true
- Include administrators: true

---

## References

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Workflow Syntax](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax)
- [Security Best Practices](https://docs.github.com/en/actions/reference/security/secure-use)

---

**Last Updated**: 2025-11-09
**Maintained By**: ML Odyssey Team
