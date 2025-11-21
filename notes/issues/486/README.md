# Issue #486: [Package] Coverage Gates - Integration and Packaging

## Objective

Package coverage gates with comprehensive documentation, configuration guides, and troubleshooting resources to help developers understand and work with coverage requirements.

## Deliverables

- Coverage gates documentation
- Configuration guide
- CI integration documentation
- Troubleshooting guide
- Developer guidelines for meeting coverage requirements

## Success Criteria

- [ ] Documentation explains coverage requirements clearly
- [ ] Configuration examples are provided
- [ ] Developers understand how to fix coverage failures
- [ ] Troubleshooting guide addresses common issues
- [ ] Integration with existing documentation

## References

### Parent Issues

- [Issue #483: [Plan] Coverage Gates](../483/README.md) - Design and architecture
- [Issue #484: [Test] Coverage Gates](../484/README.md) - Test specifications
- [Issue #485: [Impl] Coverage Gates](../485/README.md) - Implementation

### Related Issues

- [Issue #487: [Cleanup] Coverage Gates](../487/README.md) - Cleanup
- [Issue #488-492: Coverage Master](../488/README.md) - Next component

### Comprehensive Documentation

- [5-Phase Workflow](../../../../../../../home/user/ml-odyssey/notes/review/README.md)
- [Agent Hierarchy](../../../../../../../home/user/ml-odyssey/agents/hierarchy.md)

## Implementation Notes

### Documentation Structure

**1. Coverage Gates Guide** (`docs/testing/coverage-gates.md`):

```markdown
# Coverage Gates

## Overview

Coverage gates ensure code quality by enforcing minimum test coverage standards.

## Current Requirements

- **Minimum coverage**: 80% (all code)
- **Maximum regression**: 2% (per PR)
- **Exceptions**: Generated files, vendor code excluded

## How Gates Work

### Local Development

```bash

# Run tests with coverage threshold

pytest --cov=scripts --cov-fail-under=80

# If coverage < 80%, command fails with exit code 1

```text
### CI/CD Pipeline

Coverage gates run automatically on all PRs:

1. **Threshold Check**: Ensures coverage >= 80%
2. **Regression Check**: Ensures coverage doesn't drop > 2%
3. **Report Generation**: Creates detailed coverage reports

### What Happens on Failure

**Threshold Failure**:
```text

❌ Coverage: 75.5% (required: 80.0%)

Files below threshold:
  src/utils.py: 65.2%
  src/models/linear.py: 72.8%

Please add tests to improve coverage.

```text
**Regression Failure**:
```text

❌ Coverage regression detected!
   Current:  78.5%
   Baseline: 82.0%
   Delta:    -3.5% (max allowed: -2.0%)

Please add tests to restore coverage.

```text
## Meeting Coverage Requirements

### Strategy 1: Incremental Improvement

Don't try to reach 80% all at once:

1. Identify lowest-coverage files
2. Add tests for critical paths
3. Gradually improve coverage

### Strategy 2: Focus on Value

Coverage is a metric, not a goal:
- Test critical functionality thoroughly
- Document why some code has lower coverage
- Focus on meaningful tests, not coverage percentage

### Strategy 3: Use Coverage Reports

```bash

# Generate HTML report

pytest --cov=scripts --cov-report=html

# Open in browser

open htmlcov/index.html

# Find uncovered lines (highlighted in red)

# Write tests for those lines

```text
## Exceptions

Some code is excluded from coverage requirements:

- Generated files (`*_pb2.py`, `__generated__/*`)
- Third-party code (`vendor/*`)
- Test files (`tests/*`)

## Temporarily Bypassing Gates

**Not recommended**, but for emergencies:

```yaml

# In PR, add label "skip-coverage-check"

# Gates will report but not fail

# Only use for

# - Refactoring that temporarily reduces coverage

# - Generated code updates

# - Documented technical debt

```text
## Updating Thresholds

Thresholds can be adjusted in `pyproject.toml`:

```toml

[tool.coverage.report]
fail_under = 80.0  # Increase as coverage improves

```text
Changes require team discussion and ADR.
```text

**2. Configuration Reference** (`docs/testing/coverage-config.md`):

```markdown
# Coverage Gates Configuration

## Threshold Configuration

Edit `pyproject.toml`:

```toml

[tool.coverage.report]
fail_under = 80.0          # Minimum total coverage

[tool.coverage.run]
omit = [                   # Files to exclude
    "tests/*",
    "**/vendor/*",
    "**/*_pb2.py",
    "**/__generated__/*"
]

```text
## Regression Tolerance

Edit `scripts/check_coverage_regression.py` or pass flag:

```bash

python scripts/check_coverage_regression.py \
  --max-decrease 2.0      # Maximum % decrease allowed

```text
## CI Configuration

Edit `.github/workflows/test.yml`:

```yaml

- name: Run tests with coverage
  run: |
    pytest --cov=scripts --cov-fail-under=80

```text
## Per-Module Thresholds

*Note: Not currently implemented, but can be added if needed*

```toml

[tool.coverage.paths]
critical = ["src/core/*"]
experimental = ["src/experimental/*"]

[tool.coverage.report]
fail_under = 80            # Default
critical_modules = 95      # Higher for critical code
experimental_modules = 60  # Lower for experimental

```text
```text

**3. Troubleshooting Guide** (`docs/testing/coverage-troubleshooting.md`):

```markdown
# Coverage Gates Troubleshooting

## "Coverage below threshold" Error

**Symptom**: CI fails with "Coverage: 75% (required: 80%)"

**Solution**:

1. Generate coverage report:
   ```bash

   pytest --cov=scripts --cov-report=html
   open htmlcov/index.html

   ```

2. Identify low-coverage files (red in report)

3. Add tests for uncovered lines:
   - Focus on critical paths first
   - Test error conditions
   - Test edge cases

4. Re-run locally to verify:
   ```bash

   pytest --cov=scripts --cov-fail-under=80

   ```

## "Coverage regression" Error

**Symptom**: CI fails with "Coverage decreased by 3.5%"

**Solution**:

1. Check what changed in your PR:
   ```bash

   git diff main -- '*.py'

   ```

2. Identify if you:
   - Added new code without tests (add tests)
   - Deleted tests (restore or replace)
   - Modified code that broke tests (fix tests)

3. Add missing tests

4. Verify coverage restored:
   ```bash

   pytest --cov=scripts

   ```

## Gates Failing Incorrectly

**Symptom**: Coverage looks good locally, fails in CI

**Common Causes**:

1. **Different Python version**
   - CI uses Python 3.11
   - Ensure local tests use same version

2. **Missing dependencies in CI**
   - Check `requirements-dev.txt` includes all test dependencies

3. **Generated files in CI**
   - Ensure `.gitignore` excludes coverage artifacts
   - Check CI doesn't include old `.coverage` files

4. **Different file structure**
   - Verify `source` paths in pyproject.toml are correct

## Exclusions Not Working

**Symptom**: Generated files counted in coverage

**Solution**:

1. Check exclusion patterns in `pyproject.toml`:
   ```toml

   [tool.coverage.run]
   omit = [
       "**/*_pb2.py",  # Must match file pattern exactly
   ]

   ```

2. Test pattern matching:
   ```bash

   coverage debug sys  # Show coverage.py config

   ```

3. Verify files are excluded:
   ```bash

   coverage report --show-missing
   # Excluded files won't appear in report

   ```

## Performance Issues

**Symptom**: Coverage collection very slow

**Solutions**:

1. **Parallelize tests**:
   ```bash

   pytest --cov=scripts -n auto  # Requires pytest-xdist

   ```

2. **Reduce coverage scope**:
   ```toml

   [tool.coverage.run]
   source = ["scripts"]  # Only measure specific directories

   ```

3. **Skip coverage locally** (when not needed):
   ```bash

   pytest  # No --cov flag

   ```
```text

**4. Developer Guidelines** (`docs/testing/writing-tests-for-coverage.md`):

```markdown
# Writing Tests for Coverage

## Best Practices

### 1. Test Behaviors, Not Implementation

**Bad** (testing implementation):
```python

def test_function_calls_helper():
    """Test that function calls internal helper."""
    # This doesn't test actual behavior

```text
**Good** (testing behavior):
```python

def test_function_returns_correct_result():
    """Test that function produces expected output."""
    result = my_function(input_data)
    assert result == expected_output

```text
### 2. Focus on Critical Paths

Priority order for testing:
1. **Public APIs** - External interfaces
2. **Error handling** - Edge cases and failures
3. **Business logic** - Core functionality
4. **Internal helpers** - Lower priority

### 3. Use Coverage Reports to Guide Testing

```bash

# Generate HTML coverage report

pytest --cov=scripts --cov-report=html
open htmlcov/index.html

# Identify uncovered lines (red highlighting)

# Write tests specifically for those lines

```text
### 4. Don't Game Coverage

**Bad** (coverage theater):
```python

def test_function_runs():
    """Test that function runs without error."""
    my_function()  # No assertion - meaningless coverage

```text
**Good** (meaningful test):
```python

def test_function_handles_empty_input():
    """Test that function raises ValueError for empty input."""
    with pytest.raises(ValueError, match="Input cannot be empty"):
        my_function([])

```text
## Coverage-Driven Development Workflow

1. **Write test first** (TDD):
   ```python

   def test_new_feature():
       result = new_feature(input)
       assert result == expected

   ```

2. **Run with coverage**:
   ```bash

   pytest tests/test_new.py --cov=scripts.new_module

   ```

3. **Check coverage report**:
   - Should show 100% for new code
   - If not, add tests for uncovered lines

4. **Iterate** until coverage meets threshold

## Common Patterns

### Testing Error Conditions

```python

def test_handles_invalid_input():
    """Test function raises appropriate error."""
    with pytest.raises(ValueError):
        function_under_test(invalid_input)

```text
### Testing Edge Cases

```python

@pytest.mark.parametrize("input,expected", [
    (0, "zero"),           # Boundary
    (-1, "negative"),      # Below boundary
    (1, "positive"),       # Above boundary
    (sys.maxsize, "max"),  # Extreme value
])
def test_edge_cases(input, expected):
    assert function(input) == expected

```text
### Testing Branches

```python

def test_both_branches():
    """Test both branches of conditional."""
    # Test condition True
    result_true = function(condition=True)
    assert result_true == expected_when_true

    # Test condition False
    result_false = function(condition=False)
    assert result_false == expected_when_false

```text
```text

### Deliverable Checklist

Documentation Files:

- [ ] `docs/testing/coverage-gates.md` - Overview and guide
- [ ] `docs/testing/coverage-config.md` - Configuration reference
- [ ] `docs/testing/coverage-troubleshooting.md` - Common issues
- [ ] `docs/testing/writing-tests-for-coverage.md` - Developer guide

Updates:

- [ ] Main `README.md` - Add coverage badge and requirements
- [ ] `CONTRIBUTING.md` - Add coverage requirements section
- [ ] `docs/README.md` - Link to coverage documentation

### Coverage Badge

Add to `README.md`:

```markdown
[![Coverage](https://img.shields.io/badge/coverage-80%25-brightgreen.svg)](./htmlcov/index.html)

<!-- Or if using Codecov -->
[![codecov](https://codecov.io/gh/mvillmow/ml-odyssey/branch/main/graph/badge.svg)](https://codecov.io/gh/mvillmow/ml-odyssey)
```text

### Status

Created: 2025-11-19
Status: Pending implementation
Dependencies: Issue #485 (Impl) must be completed first
