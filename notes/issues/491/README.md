# Issue #491: [Package] Coverage - Integration and Packaging

## Objective

Package the complete coverage system with comprehensive documentation that ties together setup, reporting, and gates into a cohesive whole, providing developers with clear guidance on the entire coverage workflow.

## Deliverables

- Coverage system overview documentation
- Complete workflow guide
- Integration documentation linking all components
- Developer onboarding guide
- Quick reference materials

## Success Criteria

- [ ] Documentation explains complete coverage system
- [ ] Workflow guide covers end-to-end process
- [ ] Developers can easily find all coverage documentation
- [ ] Quick reference provides fast answers
- [ ] Links between component docs are clear

## References

### Parent Issues

- [Issue #488: [Plan] Coverage Master](../488/README.md) - Design and architecture
- [Issue #489: [Test] Coverage](../489/README.md) - Integration tests
- [Issue #490: [Impl] Coverage](../490/README.md) - Implementation

### Related Issues

- [Issue #492: [Cleanup] Coverage](../492/README.md) - Cleanup

### Component Documentation

- [Setup Coverage](../473/README.md) - Issues #473-477
- [Coverage Reports](../478/README.md) - Issues #478-482
- [Coverage Gates](../483/README.md) - Issues #483-487

### Comprehensive Documentation

- [5-Phase Workflow](/home/user/ml-odyssey/notes/review/README.md)
- [Agent Hierarchy](/home/user/ml-odyssey/agents/hierarchy.md)

## Implementation Notes

### Documentation Structure

**1. Coverage System Overview** (`docs/testing/coverage-overview.md`):

```markdown
# Coverage System Overview

The ML Odyssey coverage system ensures code quality through automated test coverage tracking and enforcement.

## System Components

### 1. Coverage Collection ([Setup Coverage](./coverage-setup.md))

Tracks which code is executed during tests:
- Installs and configures coverage.py
- Integrates with pytest test framework
- Collects line and branch coverage data
- Works in both local and CI environments

**Current Limitation**: Only Python code covered (scripts, tooling). Mojo ML code not yet supported.

### 2. Coverage Reports ([Coverage Reports](./coverage-reports.md))

Visualizes coverage data for analysis:
- Console reports for quick feedback
- HTML reports for detailed line-by-line analysis
- XML reports for CI integration
- Historical tracking (optional)

### 3. Coverage Gates ([Coverage Gates](./coverage-gates.md))

Enforces minimum coverage standards:
- Threshold enforcement (minimum 80% coverage)
- Regression prevention (max 2% decrease per PR)
- CI integration with automatic checks
- Clear failure messages with guidance

## Complete Workflow

```text
Write Code
    ↓
Write Tests
    ↓
Run Tests with Coverage
    ├─→ Coverage Collected
    ├─→ Reports Generated
    └─→ Gates Checked
         ↓
    Pass/Fail
```

## Quick Start

### Local Development

```bash
# Run tests with coverage
pytest --cov=scripts

# View HTML report
pytest --cov=scripts --cov-report=html
open htmlcov/index.html
```

### CI/CD

Coverage runs automatically on all PRs:
- Collects coverage data
- Generates reports
- Checks thresholds
- Blocks merge if coverage < 80% or decreases > 2%

## Coverage Requirements

- **Minimum Coverage**: 80% (line coverage)
- **Maximum Regression**: 2% (per PR)
- **Exclusions**: Generated files, vendor code, test files

## Documentation Map

- **[Coverage Setup](./coverage-setup.md)**: Installation and configuration
- **[Coverage Reports](./coverage-reports.md)**: Generating and interpreting reports
- **[Coverage Gates](./coverage-gates.md)**: Understanding and meeting requirements
- **[Coverage Troubleshooting](./coverage-troubleshooting.md)**: Common issues and solutions
- **[Writing Tests for Coverage](./writing-tests-for-coverage.md)**: Best practices

## Architecture

The coverage system has three layers:

1. **Data Collection** (pytest-cov + coverage.py)
   - Instruments Python code
   - Tracks execution during tests
   - Stores coverage data in `.coverage` file

2. **Reporting** (coverage.py reporting)
   - Reads `.coverage` data
   - Generates multiple report formats
   - Highlights covered/uncovered code

3. **Enforcement** (CI gates)
   - Checks coverage against thresholds
   - Compares to baseline (regression detection)
   - Blocks merges that fail requirements

## Mojo Coverage Status

**Current State**: Mojo code coverage not available (as of Mojo v0.25.7)

**What's Covered**:
- ✅ Python scripts in `scripts/`
- ✅ Python utilities and tooling
- ❌ Mojo source code in `src/`
- ❌ Mojo tests in `tests/`

**Future**: Monitor Mojo ecosystem for native coverage tools. See [ADR-XXX] for details.

## Next Steps

New to coverage? Start here:

1. Read [Coverage Setup](./coverage-setup.md)
2. Generate your first report
3. Review [Coverage Gates](./coverage-gates.md) requirements
4. See [Writing Tests](./writing-tests-for-coverage.md) for best practices
```

**2. Complete Workflow Guide** (`docs/testing/coverage-workflow.md`):

```markdown
# Coverage Workflow Guide

Complete walkthrough of the coverage workflow from writing code to meeting coverage requirements.

## Workflow Phases

### Phase 1: Local Development

**1.1 Write Code**

```python
# scripts/my_feature.py
def process_data(data):
    """Process input data."""
    if not data:
        raise ValueError("Data cannot be empty")
    return [x * 2 for x in data]
```

**1.2 Write Tests**

```python
# tests/test_my_feature.py
import pytest
from scripts.my_feature import process_data

def test_process_data():
    result = process_data([1, 2, 3])
    assert result == [2, 4, 6]

def test_process_empty_data():
    with pytest.raises(ValueError, match="cannot be empty"):
        process_data([])
```

**1.3 Run Tests with Coverage**

```bash
# Quick check
pytest tests/test_my_feature.py --cov=scripts.my_feature

# Expected output:
# ---------- coverage: ... ----------
# Name                      Stmts   Miss  Cover
# ---------------------------------------------
# scripts/my_feature.py         4      0   100%
# ---------------------------------------------
# TOTAL                         4      0   100%
```

**1.4 View Detailed Report (if needed)**

```bash
# Generate HTML report
pytest --cov=scripts.my_feature --cov-report=html
open htmlcov/index.html

# Identify uncovered lines (if any)
# Add tests for those lines
```

### Phase 2: Pull Request

**2.1 Push Changes**

```bash
git add scripts/my_feature.py tests/test_my_feature.py
git commit -m "feat(scripts): add data processing feature"
git push origin feature/data-processing
```

**2.2 Create PR**

```bash
gh pr create --title "Add data processing feature"
```

**2.3 CI Runs Coverage Checks**

GitHub Actions automatically:
1. Runs all tests with coverage
2. Generates reports
3. Checks coverage >= 80%
4. Compares to main branch (regression check)

**2.4 Review Coverage Results**

Check PR for:
- ✅ "Coverage: 85%" badge (pass)
- ✅ No regression (current vs main)
- 📊 Detailed HTML report in artifacts

**2.5 Fix Coverage Issues (if needed)**

If coverage fails:

```bash
# Download coverage report from CI artifacts
# Or generate locally:
pytest --cov=scripts --cov-report=html

# Open report
open htmlcov/index.html

# Find uncovered lines (red highlighting)
# Add missing tests
# Push updates
```

### Phase 3: Merge and Monitor

**3.1 Merge PR**

After approval and passing checks:

```bash
gh pr merge --squash
```

**3.2 Update Baseline**

CI automatically:
- Stores main branch coverage as new baseline
- Future PRs compare against this baseline

**3.3 Monitor Trends**

(Optional) Track coverage over time:
- Codecov integration (if configured)
- Historical tracking (if implemented)

## Common Scenarios

### Scenario: Coverage Below 80%

**Problem**: CI fails with "Coverage: 75% (required: 80%)"

**Solution**:

1. Generate HTML report:
   ```bash
   pytest --cov=scripts --cov-report=html
   open htmlcov/index.html
   ```

2. Identify low-coverage files (red/orange in report)

3. Add tests for uncovered lines:
   - Focus on critical paths
   - Test error conditions
   - Test edge cases

4. Verify locally:
   ```bash
   pytest --cov=scripts --cov-fail-under=80
   ```

5. Push updates

### Scenario: Coverage Regression

**Problem**: CI fails with "Coverage decreased by 3%"

**Solution**:

1. Check what code changed:
   ```bash
   git diff main..HEAD
   ```

2. Identify if you:
   - Added new untested code → Add tests
   - Deleted tests → Restore or justify
   - Modified code → Update tests

3. Restore coverage:
   - Add missing tests
   - Ensure new code is covered

### Scenario: Generated Files Affecting Coverage

**Problem**: Coverage includes generated files

**Solution**:

Update exclusions in `pyproject.toml`:

```toml
[tool.coverage.run]
omit = [
    "**/*_generated.py",
    "**/*_pb2.py",
]
```

## Best Practices

### During Development

1. **Write tests alongside code** (TDD)
2. **Run coverage frequently** (quick feedback)
3. **Aim for > 80% coverage** (meet requirements)
4. **Focus on meaningful tests** (not just coverage)

### Before Pushing

1. **Run full coverage check**:
   ```bash
   pytest --cov=scripts --cov-fail-under=80
   ```

2. **Review HTML report** (if time permits)

3. **Ensure critical paths tested**

### During Review

1. **Check coverage in CI**
2. **Review coverage reports** in artifacts
3. **Discuss coverage gaps** with reviewers

## Troubleshooting

See [Coverage Troubleshooting](./coverage-troubleshooting.md) for common issues.

## Tools and Commands

Quick reference:

| Task | Command |
|------|---------|
| Run with coverage | `pytest --cov=scripts` |
| Console report | `--cov-report=term-missing` |
| HTML report | `--cov-report=html` |
| Check threshold | `--cov-fail-under=80` |
| Open HTML | `open htmlcov/index.html` |
```

**3. Developer Onboarding** (`docs/testing/coverage-onboarding.md`):

```markdown
# Coverage Onboarding

Welcome! This guide helps you get started with the coverage system.

## What is Code Coverage?

Coverage measures which lines of code are executed during tests:
- **Covered** = Executed during tests (green)
- **Uncovered** = Never executed (red)
- **80% coverage** = 80% of lines executed

## Why Coverage Matters

1. **Quality assurance**: Tests actually test the code
2. **Confidence**: Changes don't break untested code
3. **Documentation**: Coverage shows what's tested

## Getting Started

### 1. First Coverage Run (5 minutes)

```bash
# Run tests with coverage
pytest --cov=scripts

# You'll see:
# ---------- coverage: ... ----------
# Name               Stmts   Miss  Cover
# --------------------------------------
# scripts/utils.py      50      5    90%
# --------------------------------------
```

### 2. View Detailed Report (10 minutes)

```bash
# Generate HTML report
pytest --cov=scripts --cov-report=html

# Open in browser
open htmlcov/index.html

# Explore:
# - File list (which files have coverage)
# - Click file (see line-by-line coverage)
# - Red lines (not covered)
# - Green lines (covered)
```

### 3. Understand Requirements (5 minutes)

Read [Coverage Gates](./coverage-gates.md) to understand:
- Minimum coverage: 80%
- Regression limit: 2%
- How gates enforce these

### 4. Write Tests for Coverage (20 minutes)

Follow [Writing Tests for Coverage](./writing-tests-for-coverage.md):
- How to write meaningful tests
- Using coverage reports to guide testing
- Best practices

## Quick Reference

**Run coverage locally**:
```bash
pytest --cov=scripts
```

**View HTML report**:
```bash
pytest --cov=scripts --cov-report=html
open htmlcov/index.html
```

**Check threshold**:
```bash
pytest --cov=scripts --cov-fail-under=80
```

## Common Questions

**Q: Do I need 100% coverage?**
A: No. 80% is the minimum. Focus on testing critical code well.

**Q: What if coverage fails in CI?**
A: Generate HTML report, find uncovered lines, add tests.

**Q: Can I exclude files from coverage?**
A: Yes, but only for generated code or vendor libraries. Edit `pyproject.toml`.

**Q: Why doesn't Mojo code show coverage?**
A: Mojo coverage tools aren't available yet. We track Python code only.

## Next Steps

1. ✅ Run coverage locally
2. ✅ Explore HTML reports
3. ✅ Read [Coverage Gates](./coverage-gates.md)
4. ✅ Review [Writing Tests](./writing-tests-for-coverage.md)
5. ✅ Create a PR and see CI coverage checks

## Need Help?

- [Coverage Troubleshooting](./coverage-troubleshooting.md)
- [Coverage Setup](./coverage-setup.md) - Configuration details
- [Coverage Reports](./coverage-reports.md) - Report interpretation
```

**4. Quick Reference Card** (`docs/testing/coverage-quick-ref.md`):

```markdown
# Coverage Quick Reference

## Common Commands

| Task | Command |
|------|---------|
| Run with coverage | `pytest --cov=scripts` |
| Console report | `pytest --cov=scripts --cov-report=term` |
| HTML report | `pytest --cov=scripts --cov-report=html` |
| Open HTML | `open htmlcov/index.html` |
| Check threshold | `pytest --cov=scripts --cov-fail-under=80` |
| Skip coverage | `pytest` (no --cov flag) |

## Requirements

- Minimum coverage: **80%**
- Max regression: **2%**
- Applies to: **Python code only** (Mojo not yet supported)

## Files

- `.coverage` - Coverage data
- `htmlcov/` - HTML reports
- `coverage.xml` - XML for CI
- `pyproject.toml` - Configuration

## Configuration

Edit `pyproject.toml`:

```toml
[tool.coverage.report]
fail_under = 80.0

[tool.coverage.run]
source = ["scripts"]
omit = ["tests/*", "**/vendor/*"]
```

## CI Workflow

1. **PR created** → CI runs coverage
2. **Coverage < 80%** → CI fails, add tests
3. **Regression > 2%** → CI fails, restore coverage
4. **All pass** → Ready to merge

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Coverage below 80% | Add tests for uncovered code |
| Coverage regression | Compare reports, add missing tests |
| Generated files counted | Add to `omit` in pyproject.toml |
| Slow coverage | Use `pytest -n auto` for parallel |

## Links

- [Full Documentation](./coverage-overview.md)
- [Setup Guide](./coverage-setup.md)
- [Reports Guide](./coverage-reports.md)
- [Gates Guide](./coverage-gates.md)
- [Troubleshooting](./coverage-troubleshooting.md)
```

### Deliverable Checklist

Documentation Files:
- [ ] `docs/testing/coverage-overview.md` - System overview
- [ ] `docs/testing/coverage-workflow.md` - Complete workflow guide
- [ ] `docs/testing/coverage-onboarding.md` - New developer guide
- [ ] `docs/testing/coverage-quick-ref.md` - Quick reference

Updates:
- [ ] `README.md` - Add coverage section with badge
- [ ] `CONTRIBUTING.md` - Add coverage requirements
- [ ] `docs/README.md` - Link to coverage documentation
- [ ] `docs/testing/README.md` - Coverage system index

### Integration with Component Docs

Link component documentation:

```markdown
# In coverage-overview.md
- [Setup Coverage](./coverage-setup.md) - From Issue #476
- [Coverage Reports](./coverage-reports.md) - From Issue #481
- [Coverage Gates](./coverage-gates.md) - From Issue #486
```

### Status

Created: 2025-11-19
Status: Pending implementation
Dependencies: Issue #490 (Impl) must be completed first
