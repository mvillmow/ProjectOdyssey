# Issue #484: [Test] Coverage Gates - Write Tests

## Objective

Write comprehensive tests for coverage gate enforcement, validating that CI checks correctly enforce coverage thresholds, prevent regressions, and provide clear failure messages.

## Deliverables

- Tests for threshold enforcement logic
- Tests for coverage regression detection
- Tests for exception handling
- Tests for failure message generation
- Tests for CI integration
- Test fixtures for coverage scenarios

## Success Criteria

- [ ] Threshold enforcement works correctly
- [ ] Regression detection identifies coverage drops
- [ ] Exceptions exclude correct files
- [ ] Failure messages are clear and actionable
- [ ] CI integration tests pass
- [ ] All tests are documented

## References

### Parent Issue

- [Issue #483: [Plan] Coverage Gates](../483/README.md) - Design and architecture

### Related Issues

- [Issue #485: [Impl] Coverage Gates](../485/README.md) - Implementation
- [Issue #486: [Package] Coverage Gates](../486/README.md) - Packaging
- [Issue #487: [Cleanup] Coverage Gates](../487/README.md) - Cleanup

### Dependencies

- [Issue #478-482: Coverage Reports](../478/README.md) - Must have coverage data and reports

### Comprehensive Documentation

- [5-Phase Workflow](../../../../../../../home/user/ml-odyssey/notes/review/README.md)
- [Agent Hierarchy](../../../../../../../home/user/ml-odyssey/agents/hierarchy.md)

## Implementation Notes

### Testing Strategy

Coverage gates enforce quality standards. Tests must validate:

**1. Threshold Enforcement**

Test absolute coverage minimum:

```python
def test_fails_when_coverage_below_threshold():
    """Test gate fails if coverage < configured threshold."""
    # Given: Coverage data at 75%, threshold at 80%
    # When: Check coverage gate
    # Then: Gate fails with clear message

def test_passes_when_coverage_meets_threshold():
    """Test gate passes if coverage >= threshold."""
    # Given: Coverage data at 85%, threshold at 80%
    # When: Check coverage gate
    # Then: Gate passes

def test_different_thresholds_for_metrics():
    """Test separate thresholds for line/branch coverage."""
    # Given: Line coverage 85%, branch coverage 72%, thresholds 80%/75%
    # When: Check both gates
    # Then: Both pass
```text

**2. Regression Detection**

Test coverage delta checking:

```python
def test_fails_on_coverage_regression():
    """Test gate fails when coverage decreases."""
    # Given: Main branch coverage 85%, PR coverage 82%
    # When: Check regression gate
    # Then: Gate fails (3% decrease)

def test_allows_coverage_increase():
    """Test gate passes when coverage increases."""
    # Given: Main branch coverage 80%, PR coverage 85%
    # When: Check regression gate
    # Then: Gate passes

def test_allows_small_decrease_within_tolerance():
    """Test gate allows decrease within tolerance."""
    # Given: Main branch 85%, PR 84%, tolerance 2%
    # When: Check regression gate
    # Then: Gate passes (1% decrease within 2% tolerance)
```text

**3. Exception Handling**

Test file exclusions:

```python
def test_excludes_generated_files():
    """Test gate excludes generated files from coverage."""
    # Given: Coverage including generated files
    # When: Apply exceptions
    # Then: Generated files excluded from total

def test_excludes_vendor_code():
    """Test gate excludes vendor/external code."""
    # Given: Coverage including vendor directory
    # When: Apply exceptions
    # Then: Vendor code excluded

def test_exception_patterns_match_correctly():
    """Test glob patterns match expected files."""
    # Given: Exception pattern "**/*_pb2.py"
    # When: Check file "src/models/schema_pb2.py"
    # Then: File is excluded
```text

**4. Failure Messages**

Test error messages are helpful:

```python
def test_failure_message_shows_current_coverage():
    """Test failure message includes current coverage."""
    # Given: Coverage at 75%, threshold 80%
    # When: Gate fails
    # Then: Message shows "Coverage: 75.0% (required: 80.0%)"

def test_failure_message_lists_low_coverage_files():
    """Test failure message identifies problematic files."""
    # Given: Multiple files with low coverage
    # When: Gate fails
    # Then: Message lists files below threshold

def test_regression_message_shows_delta():
    """Test regression message shows coverage change."""
    # Given: Main 85%, PR 80%
    # When: Regression gate fails
    # Then: Message shows "-5.0% vs main branch"
```text

**5. CI Integration**

Test in CI-like environment:

```python
def test_gate_returns_exit_code_on_failure():
    """Test gate script exits with non-zero on failure."""
    # Given: Coverage below threshold
    # When: Run gate check script
    # Then: Exit code is 1 (failure)

def test_gate_returns_zero_on_success():
    """Test gate script exits with zero on success."""
    # Given: Coverage meets threshold
    # When: Run gate check script
    # Then: Exit code is 0 (success)

def test_gate_reads_config_from_file():
    """Test gate reads thresholds from config."""
    # Given: pyproject.toml with thresholds
    # When: Run gate check
    # Then: Uses configured thresholds
```text

### Test Fixtures

### Coverage Data Fixtures

```python
@pytest.fixture
def coverage_below_threshold():
    """Coverage data that fails threshold check."""
    return {
        "total_coverage": 75.0,
        "line_coverage": 75.0,
        "branch_coverage": 70.0
    }

@pytest.fixture
def coverage_with_regression():
    """Coverage data showing regression from main."""
    return {
        "current": {"total": 80.0},
        "main": {"total": 85.0},
        "delta": -5.0
    }

@pytest.fixture
def coverage_with_exceptions():
    """Coverage data including files to exclude."""
    return {
        "files": {
            "src/core.py": {"coverage": 90.0},
            "src/generated/schema_pb2.py": {"coverage": 50.0},
            "vendor/lib.py": {"coverage": 60.0}
        }
    }
```text

### Key Test Scenarios

### Threshold Checks

- [ ] Fails when total coverage < threshold
- [ ] Passes when coverage >= threshold
- [ ] Handles edge case of exactly meeting threshold
- [ ] Supports different thresholds per metric
- [ ] Respects per-module threshold overrides

### Regression Detection

- [ ] Detects coverage decrease
- [ ] Allows coverage increase
- [ ] Handles missing baseline (first run)
- [ ] Respects configured tolerance
- [ ] Works with partial coverage data

### Exceptions

- [ ] Excludes files matching patterns
- [ ] Glob patterns work correctly
- [ ] Multiple patterns supported
- [ ] Exclusions don't affect gate logic
- [ ] Exclusions documented in report

### Messages

- [ ] Clear failure reasons
- [ ] Actionable guidance
- [ ] Shows coverage numbers
- [ ] Lists problematic files
- [ ] Links to detailed reports

### Integration Testing

Test complete gate workflow:

```python
def test_end_to_end_gate_workflow():
    """Test complete coverage gate workflow."""
    # Given: Real coverage data from pytest-cov
    # When: Run full gate check process
    # Then: All components work together correctly

def test_ci_pipeline_integration():
    """Test gate integrates with CI pipeline."""
    # Given: Simulated CI environment
    # When: Run gate as CI step
    # Then: CI receives correct pass/fail status
```text

### Open Questions to Address

- [ ] What threshold values should defaults be? (80%, from Issue #483)
- [ ] What tolerance for regression? (2%, from Issue #483)
- [ ] Should gates be blocking or advisory initially?
- [ ] How to handle coverage for new files?

### Status

Created: 2025-11-19
Status: Pending implementation
Dependencies: Issue #483 (Plan) and #478-482 (Reports) must be completed first
