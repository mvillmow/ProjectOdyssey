# Issue #439: [Test] Test Utilities

## Objective

Develop comprehensive tests for the test utilities themselves to ensure assertion helpers, comparison functions, data generators, and mock objects work correctly and reliably.

## Deliverables

- Tests for assertion helpers
- Tests for tensor comparison utilities
- Tests for test data generators
- Tests for mock objects
- Meta-test validation suite

## Success Criteria

- [ ] All assertion helpers are tested with valid and invalid inputs
- [ ] Tensor comparisons handle edge cases (NaN, infinity, zero)
- [ ] Data generators produce deterministic and valid outputs
- [ ] Mock objects behave correctly
- [ ] Tests verify clear error messages on failures

## Current State Analysis

### Existing Test Utilities

**conftest.mojo** provides:
1. **Assertion Functions** (8 types):
   - `assert_true(condition, message)` - Basic boolean assertion
   - `assert_false(condition, message)` - Negated boolean assertion
   - `assert_equal[T: Comparable](a, b, message)` - Exact equality
   - `assert_not_equal[T: Comparable](a, b, message)` - Inequality
   - `assert_almost_equal(a, b, tolerance, message)` - Float near-equality
   - `assert_greater(a, b, message)` - Greater than comparison
   - `assert_less(a, b, message)` - Less than comparison

2. **Test Fixtures**:
   - `TestFixtures.deterministic_seed()` - Returns fixed seed (42)
   - `TestFixtures.set_seed()` - Sets deterministic random seed

3. **Benchmark Utilities**:
   - `BenchmarkResult` struct with timing/throughput/memory metrics
   - `print_benchmark_results(results)` - Formatted output

4. **Test Helpers** (placeholders):
   - `measure_time[func]()` - TODO: Not implemented
   - `measure_throughput[func](n_iterations)` - TODO: Not implemented

5. **Test Data Generators**:
   - `create_test_vector(size, value)` - Uniform value vector
   - `create_test_matrix(rows, cols, value)` - Uniform value matrix
   - `create_sequential_vector(size, start)` - Sequential values [start, start+1, ...]

**helpers/assertions.mojo** provides:
1. **Basic Assertions**:
   - `assert_true/false(condition, message)`
   - `assert_equal_int(a, b, message)`
   - `assert_equal_float(a, b, tolerance, message)`
   - `assert_close_float(a, b, rtol, atol, message)` - NumPy-style closeness

2. **ExTensor Assertions**:
   - `assert_shape[T](tensor, expected, message)` - Shape validation
   - `assert_dtype[T](tensor, expected_dtype, message)` - Type validation
   - `assert_numel[T](tensor, expected_numel, message)` - Element count
   - `assert_dim[T](tensor, expected_dim, message)` - Dimension count
   - `assert_value_at[T](tensor, index, expected, tolerance, message)` - Single value
   - `assert_all_values[T](tensor, expected, tolerance, message)` - All values match
   - `assert_all_close[T](a, b, rtol, atol, message)` - Tensor near-equality
   - `assert_contiguous[T](tensor, message)` - Memory layout validation

### What Needs Testing

**1. Assertion Correctness**:
- Do assertions pass when they should?
- Do assertions fail when they should?
- Are error messages clear and helpful?

**2. Edge Case Handling**:
- NaN comparisons (should NaN == NaN?)
- Infinity comparisons (+inf, -inf)
- Zero handling (avoid division by zero in relative tolerances)
- Empty tensors/vectors

**3. Tolerance Behavior**:
- Absolute tolerance (atol) works correctly
- Relative tolerance (rtol) works correctly
- Combined tolerances (atol + rtol * |b|)

**4. Data Generator Reliability**:
- Generators produce expected output
- Deterministic seeding works (same seed = same output)
- Edge cases (size=0, size=1, large sizes)

## Test Implementation Strategy

### 1. Assertion Tests (`test_assertions.mojo`)

```mojo
"""Test the test utilities - meta-testing."""

from tests.shared.conftest import (
    assert_true, assert_false, assert_equal, assert_almost_equal,
    assert_greater, assert_less
)

fn test_assert_true_passes() raises:
    """Verify assert_true passes for True condition."""
    assert_true(True)  # Should not raise
    assert_true(1 == 1)  # Should not raise

fn test_assert_true_fails() raises:
    """Verify assert_true fails for False condition."""
    var raised = False
    try:
        assert_true(False, "Expected failure")
        # Should not reach here
    except e:
        raised = True
        # Verify error message contains "Expected failure"

    assert_true(raised, "assert_true should have raised")

fn test_assert_almost_equal_within_tolerance() raises:
    """Verify assert_almost_equal passes within tolerance."""
    assert_almost_equal(1.0, 1.0000001, tolerance=1e-5)  # Should pass
    assert_almost_equal(1.0, 1.001, tolerance=0.01)  # Should pass

fn test_assert_almost_equal_outside_tolerance() raises:
    """Verify assert_almost_equal fails outside tolerance."""
    var raised = False
    try:
        assert_almost_equal(1.0, 1.1, tolerance=1e-5)
    except:
        raised = True

    assert_true(raised, "Should fail outside tolerance")

fn test_assert_equal_generic_types() raises:
    """Verify assert_equal works with different types."""
    assert_equal(42, 42)  # Int
    assert_equal(3.14, 3.14)  # Float (exact equality)
    assert_equal("hello", "hello")  # String

fn test_comparisons() raises:
    """Verify comparison assertions work correctly."""
    assert_greater(5.0, 3.0)  # Should pass
    assert_less(3.0, 5.0)  # Should pass

    # Test failures
    var raised = False
    try:
        assert_greater(3.0, 5.0)  # Should fail
    except:
        raised = True
    assert_true(raised, "assert_greater should fail for 3.0 > 5.0")
```

### 2. Tolerance Tests (`test_tolerances.mojo`)

```mojo
"""Test floating-point tolerance behavior."""

from tests.helpers.assertions import assert_close_float
from math import isnan, isinf

fn test_absolute_tolerance() raises:
    """Verify absolute tolerance works correctly."""
    # |a - b| <= atol
    assert_close_float(1.0, 1.00001, rtol=0.0, atol=1e-4)  # Pass
    assert_close_float(0.0, 0.00001, rtol=0.0, atol=1e-4)  # Pass

fn test_relative_tolerance() raises:
    """Verify relative tolerance works correctly."""
    # |a - b| <= rtol * |b|
    assert_close_float(100.0, 100.1, rtol=0.01, atol=0.0)  # 1% relative
    assert_close_float(0.001, 0.0011, rtol=0.1, atol=0.0)  # 10% relative

fn test_combined_tolerance() raises:
    """Verify combined tolerance: |a - b| <= atol + rtol * |b|."""
    # Near zero: atol dominates
    assert_close_float(0.0, 0.0001, rtol=0.01, atol=1e-3)

    # Large values: rtol dominates
    assert_close_float(1000.0, 1010.0, rtol=0.01, atol=1e-3)

fn test_nan_handling() raises:
    """Verify NaN comparisons work correctly."""
    # NaN == NaN should be considered equal in testing context
    var nan_val = Float64(0.0) / Float64(0.0)
    assert_close_float(nan_val, nan_val, rtol=1e-5, atol=1e-8)

fn test_infinity_handling() raises:
    """Verify infinity comparisons work correctly."""
    var inf_val = Float64(1.0) / Float64(0.0)
    assert_close_float(inf_val, inf_val, rtol=1e-5, atol=1e-8)
```

### 3. Data Generator Tests (`test_generators.mojo`)

```mojo
"""Test data generator utilities."""

from tests.shared.conftest import (
    create_test_vector, create_test_matrix, create_sequential_vector,
    TestFixtures
)

fn test_create_test_vector() raises:
    """Verify vector creation with uniform values."""
    var vec = create_test_vector(10, 3.14)

    assert_equal(len(vec), 10)
    for i in range(len(vec)):
        assert_almost_equal(vec[i], 3.14, tolerance=1e-8)

fn test_create_test_matrix() raises:
    """Verify matrix creation with uniform values."""
    var matrix = create_test_matrix(5, 3, 2.0)

    assert_equal(len(matrix), 5)  # 5 rows
    for i in range(len(matrix)):
        assert_equal(len(matrix[i]), 3)  # 3 columns
        for j in range(len(matrix[i])):
            assert_almost_equal(matrix[i][j], 2.0, tolerance=1e-8)

fn test_create_sequential_vector() raises:
    """Verify sequential vector creation."""
    var vec = create_sequential_vector(5, start=10.0)

    assert_equal(len(vec), 5)
    assert_almost_equal(vec[0], 10.0, tolerance=1e-8)
    assert_almost_equal(vec[1], 11.0, tolerance=1e-8)
    assert_almost_equal(vec[2], 12.0, tolerance=1e-8)
    assert_almost_equal(vec[3], 13.0, tolerance=1e-8)
    assert_almost_equal(vec[4], 14.0, tolerance=1e-8)

fn test_deterministic_seed() raises:
    """Verify deterministic seeding produces consistent results."""
    TestFixtures.set_seed()
    var val1 = randn()

    TestFixtures.set_seed()  # Reset to same seed
    var val2 = randn()

    assert_almost_equal(val1, val2, tolerance=1e-10)
```

### 4. Error Message Tests (`test_error_messages.mojo`)

```mojo
"""Verify error messages are clear and helpful."""

fn test_assert_equal_error_message() raises:
    """Verify assert_equal provides clear error on failure."""
    try:
        assert_equal(42, 43, "Custom message")
        assert_true(False, "Should have raised")
    except e:
        var msg = str(e)
        # Should contain both values and custom message
        # (Implementation-specific, adjust based on actual format)
        assert_true("Custom message" in msg or "42" in msg)

fn test_assert_almost_equal_shows_diff() raises:
    """Verify assert_almost_equal shows difference in error."""
    try:
        assert_almost_equal(1.0, 1.5, tolerance=1e-5)
        assert_true(False, "Should have raised")
    except e:
        var msg = str(e)
        # Should show the difference (0.5)
        # (Implementation-specific)
```

## References

- **Source Plan**: [notes/plan/02-shared-library/04-testing/01-test-framework/02-test-utilities/plan.md](../../../plan/02-shared-library/04-testing/01-test-framework/02-test-utilities/plan.md)
- **Related Issues**:
  - Issue #438: [Plan] Test Utilities
  - Issue #440: [Impl] Test Utilities
  - Issue #441: [Package] Test Utilities
  - Issue #442: [Cleanup] Test Utilities
- **Existing Code**:
  - `/tests/shared/conftest.mojo` - Core test utilities
  - `/tests/helpers/assertions.mojo` - ExTensor assertions

## Implementation Notes

### Testing Philosophy

**Key Principle**: Test the test utilities to ensure reliability

**Why Meta-Testing Matters**:
- Test utilities are foundational - bugs here affect all tests
- Assertion failures must be trustworthy
- Error messages must be clear for debugging
- Edge cases must be handled correctly

### Test Coverage Goals

- **Assertions**: 100% coverage of all assertion functions
- **Edge Cases**: NaN, infinity, zero, empty inputs
- **Error Paths**: Verify failures occur when expected
- **Error Messages**: Verify messages are clear and helpful

### Next Steps

1. Implement meta-tests for all assertion functions
2. Test edge cases thoroughly
3. Verify error message quality
4. Test data generators for determinism
5. Document findings for Implementation phase (#440)
