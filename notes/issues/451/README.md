# Issue #451: [Package] Test Framework

## Objective

Package the complete test framework for easy adoption and reuse, creating comprehensive documentation, examples, and guides that make the framework accessible to all developers.

## Deliverables

- Complete test framework documentation
- Quick start guide for new test developers
- Comprehensive examples and tutorials
- API reference for all framework components
- Best practices and patterns guide
- Troubleshooting guide

## Success Criteria

- [ ] Framework is fully documented with clear examples
- [ ] New developers can write tests easily using documentation
- [ ] All framework components have API documentation
- [ ] Common patterns and anti-patterns are documented
- [ ] Troubleshooting guide addresses common issues
- [ ] Documentation is accurate and up-to-date

## Current State Analysis

### Framework Components

**1. Setup** (Directory structure, test running, CI):
- Tests organized in `/tests/` by component
- Test naming: `test_*.mojo`
- Execution: `mojo test <file>.mojo`
- 91+ tests passing

**2. Utilities** (Assertions, comparisons, generators):
- `/tests/shared/conftest.mojo` - Core utilities
- `/tests/helpers/assertions.mojo` - ExTensor utilities
- 7 general assertions + 8 ExTensor assertions
- 3 data generators

**3. Fixtures** (Test data, setup):
- TestFixtures struct with seeding
- Data generators for vectors/matrices
- Deterministic random data support

### Documentation Gaps

**Existing Documentation**:
- Inline docstrings in source files
- Some README files in component directories

**Missing Documentation**:
- Comprehensive quick start guide
- Complete API reference
- Tutorial with progressive examples
- Best practices guide
- Troubleshooting guide
- Migration guide (if updating from previous approach)

## Packaging Strategy

### 1. Documentation Structure

**Create**: `docs/testing/` directory with complete guides

```text
docs/
└── testing/
    ├── README.md                  # Overview and navigation
    ├── quick-start.md             # Get started quickly
    ├── tutorial.md                # Progressive tutorial
    ├── api-reference.md           # Complete API docs
    ├── best-practices.md          # Patterns and conventions
    ├── fixtures-catalog.md        # All available fixtures
    ├── troubleshooting.md         # Common issues and solutions
    └── examples/
        ├── basic-test.md          # Simple test example
        ├── fixtures-usage.md      # Using fixtures
        ├── assertions.md          # Assertion examples
        └── integration-test.md    # Integration test example
```

### 2. Quick Start Guide

**Create**: `docs/testing/quick-start.md`

```markdown
# Test Framework Quick Start

## Your First Test (2 minutes)

**Step 1**: Create a test file in appropriate directory
```bash
touch tests/myfeature/test_myfunction.mojo
```

**Step 2**: Write your test
```mojo
from tests.shared.conftest import assert_equal

fn test_addition() raises:
    """Test that addition works correctly."""
    var result = 2 + 2
    assert_equal(result, 4)
```

**Step 3**: Run your test
```bash
mojo test tests/myfeature/test_myfunction.mojo
```

**Output**:
```text
Running test_addition...
✓ test_addition passed
```

## Common Test Patterns

### Pattern 1: Testing with Assertions
```mojo
from tests.shared.conftest import assert_true, assert_equal

fn test_comparison() raises:
    assert_true(5 > 3)
    assert_equal(10 / 2, 5)
```

### Pattern 2: Testing with Fixtures
```mojo
from tests.shared.conftest import TestFixtures, create_test_vector

fn test_with_data() raises:
    var data = create_test_vector(10, 1.0)
    assert_equal(len(data), 10)
```

### Pattern 3: Testing Random Operations
```mojo
from tests.shared.conftest import TestFixtures

fn test_random_operation() raises:
    TestFixtures.set_seed()  // Make it reproducible
    var result = my_random_function()
    // Test result...
```

## Next Steps

- Read the [Tutorial](tutorial.md) for in-depth examples
- Check [API Reference](api-reference.md) for all available functions
- Review [Best Practices](best-practices.md) for patterns
```

### 3. Comprehensive Tutorial

**Create**: `docs/testing/tutorial.md`

```markdown
# Test Framework Tutorial

## Introduction

This tutorial walks through progressively complex testing scenarios,
teaching you how to use the test framework effectively.

## Lesson 1: Basic Assertions

Learn to use assertions to validate your code.

```mojo
from tests.shared.conftest import assert_equal, assert_true

fn test_basic_assertions() raises:
    """Learn basic assertions."""

    // Test exact equality
    assert_equal(42, 42)

    // Test boolean conditions
    assert_true(10 > 5)
    assert_true(is_valid(data))
```

**Exercise**: Write a test for a function you're working on.

## Lesson 2: Floating-Point Comparisons

Floating-point numbers need special handling.

```mojo
from tests.shared.conftest import assert_almost_equal

fn test_float_comparison() raises:
    """Handle floating-point precision."""

    // WRONG: This may fail due to floating-point precision
    // assert_equal(0.1 + 0.2, 0.3)

    // RIGHT: Use tolerance
    assert_almost_equal(0.1 + 0.2, 0.3, tolerance=1e-10)

    // Division example
    assert_almost_equal(10.0 / 3.0, 3.333333, tolerance=1e-6)
```

**Key Concept**: Always use `assert_almost_equal` for floating-point values.

## Lesson 3: Using Test Fixtures

Fixtures provide reusable test data.

```mojo
from tests.shared.conftest import create_test_vector, create_test_matrix

fn test_with_fixtures() raises:
    """Use fixtures for test data."""

    // Create a vector filled with ones
    var ones = create_test_vector(100, 1.0)

    // Create a matrix filled with twos
    var matrix = create_test_matrix(10, 5, 2.0)

    // Use in tests
    assert_equal(len(ones), 100)
    assert_equal(len(matrix), 10)
    assert_equal(len(matrix[0]), 5)
```

## Lesson 4: Deterministic Random Testing

Random operations need deterministic seeds for reproducibility.

```mojo
from tests.shared.conftest import TestFixtures

fn test_random_operation() raises:
    """Test random code deterministically."""

    // Set deterministic seed
    TestFixtures.set_seed()

    // Generate random values (reproducible)
    var random_val = randn()

    // Test the values
    // (You can reset seed to get same values again)
    TestFixtures.set_seed()
    var same_val = randn()
    assert_almost_equal(random_val, same_val, tolerance=1e-15)
```

**Key Concept**: Always call `TestFixtures.set_seed()` before using randomness.

## Lesson 5: Testing ExTensor Operations

Special assertions for tensor operations.

```mojo
from tests.helpers.assertions import assert_shape, assert_all_close
from extensor import ExTensor

fn test_tensor_operations() raises:
    """Test tensor operations."""

    var tensor1 = ExTensor.ones((3, 4))
    var tensor2 = ExTensor.ones((3, 4))

    // Verify shape
    var expected_shape = DynamicVector[Int](3, 4)
    assert_shape(tensor1, expected_shape)

    // Verify values are close (floating-point)
    assert_all_close(tensor1, tensor2, rtol=1e-5, atol=1e-8)
```

## Lesson 6: Integration Tests

Test multiple components working together.

```mojo
fn test_data_pipeline() raises:
    """Test complete data pipeline."""

    // Setup
    TestFixtures.set_seed()

    // Create input data
    var input = create_test_matrix(100, 10, 0.0)

    // Run pipeline
    var processed = preprocess(input)
    var features = extract_features(processed)
    var predictions = model.predict(features)

    // Validate output
    assert_equal(len(predictions), 100)
    assert_true(all_values_in_range(predictions, 0.0, 1.0))
```

## Lesson 7: Test Organization

How to organize tests effectively.

**Directory Structure**:
```text
tests/
├── mycomponent/
│   ├── test_core.mojo       # Core functionality
│   ├── test_utils.mojo      # Utility functions
│   └── test_integration.mojo  # Integration tests
```

**Naming Convention**:
- Files: `test_*.mojo`
- Functions: `test_*` prefix
- Descriptive names: `test_handles_empty_input` not `test1`

## Summary

You've learned:
- ✓ Basic assertions
- ✓ Floating-point comparisons
- ✓ Using fixtures
- ✓ Deterministic random testing
- ✓ Tensor assertions
- ✓ Integration testing
- ✓ Test organization

**Next**: Read [Best Practices](best-practices.md) for advanced patterns.
```

### 4. Complete API Reference

**Create**: `docs/testing/api-reference.md`

```markdown
# Test Framework API Reference

Complete reference for all testing utilities.

## Table of Contents

1. [Assertions](#assertions)
2. [Fixtures](#fixtures)
3. [Data Generators](#data-generators)
4. [ExTensor Assertions](#extensor-assertions)
5. [Benchmark Utilities](#benchmark-utilities)

---

## Assertions

### assert_true(condition, message="")

Assert that a boolean condition is true.

**Parameters**:
- `condition: Bool` - The condition to check
- `message: String` - Optional error message (default: "")

**Raises**:
- `Error` if condition is false

**Example**:
```mojo
assert_true(x > 0)
assert_true(is_valid(data), "Data must be valid")
```

### assert_false(condition, message="")

Assert that a boolean condition is false.

**Parameters**:
- `condition: Bool` - The condition to check
- `message: String` - Optional error message

**Raises**:
- `Error` if condition is true

**Example**:
```mojo
assert_false(is_empty(list))
assert_false(has_errors(result), "Should have no errors")
```

### assert_equal[T: Comparable](a, b, message="")

Assert exact equality of two values.

**Parameters**:
- `a: T` - First value
- `b: T` - Second value
- `message: String` - Optional error message

**Type Parameters**:
- `T: Comparable` - Any comparable type

**Raises**:
- `Error` if a != b

**Example**:
```mojo
assert_equal(42, 42)
assert_equal("hello", "hello")
assert_equal(result, expected_value)
```

### assert_not_equal[T: Comparable](a, b, message="")

Assert inequality of two values.

**Parameters**:
- `a: T` - First value
- `b: T` - Second value
- `message: String` - Optional error message

**Raises**:
- `Error` if a == b

**Example**:
```mojo
assert_not_equal(result, 0)
assert_not_equal(new_val, old_val, "Value should have changed")
```

### assert_almost_equal(a, b, tolerance=1e-6, message="")

Assert floating-point near-equality.

**Parameters**:
- `a: Float32` - First value
- `b: Float32` - Second value
- `tolerance: Float32` - Maximum allowed difference (default: 1e-6)
- `message: String` - Optional error message

**Raises**:
- `Error` if |a - b| > tolerance

**Example**:
```mojo
assert_almost_equal(0.1 + 0.2, 0.3, tolerance=1e-10)
assert_almost_equal(result, 3.14159, tolerance=1e-5)
```

**Note**: Always use this for floating-point comparisons, not `assert_equal`.

### assert_greater(a, b, message="")

Assert a > b.

**Parameters**:
- `a: Float32` - First value
- `b: Float32` - Second value
- `message: String` - Optional error message

**Raises**:
- `Error` if a <= b

**Example**:
```mojo
assert_greater(accuracy, 0.9)
assert_greater(new_loss, old_loss, "Loss should decrease")
```

### assert_less(a, b, message="")

Assert a < b.

**Parameters**:
- `a: Float32` - First value
- `b: Float32` - Second value
- `message: String` - Optional error message

**Raises**:
- `Error` if a >= b

**Example**:
```mojo
assert_less(error_rate, 0.01)
assert_less(loss, threshold)
```

---

## Fixtures

### TestFixtures.deterministic_seed()

Get deterministic random seed for reproducible tests.

**Returns**: `Int` - Fixed seed value (42)

**Example**:
```mojo
var seed = TestFixtures.deterministic_seed()
random.seed(seed)
```

### TestFixtures.set_seed()

Set global random seed to deterministic value.

**Side Effects**: Sets global random seed to 42

**Example**:
```mojo
TestFixtures.set_seed()
var random_values = generate_random_data()  // Reproducible
```

**Best Practice**: Call at the start of every test using randomness.

---

## Data Generators

[... continue with complete API documentation ...]
```

### 5. Best Practices Guide

**Create**: `docs/testing/best-practices.md`

```markdown
# Test Framework Best Practices

## General Principles

### 1. Write Clear, Focused Tests

```mojo
// GOOD: Clear test name and single responsibility
fn test_handles_empty_list() raises:
    var result = process_list(List[Int]())
    assert_equal(len(result), 0)

// BAD: Unclear name and tests multiple things
fn test_stuff() raises:
    // Tests 5 different things...
```

### 2. Make Tests Independent

```mojo
// GOOD: Each test is self-contained
fn test_addition() raises:
    assert_equal(2 + 2, 4)

fn test_subtraction() raises:
    assert_equal(5 - 3, 2)

// BAD: Tests depend on shared state
var counter = 0

fn test_increments() raises:
    counter += 1  // Modifies shared state
    assert_equal(counter, 1)

fn test_increments_again() raises:
    counter += 1  // Depends on previous test
    assert_equal(counter, 2)  // Fails if test_increments didn't run!
```

### 3. Use Deterministic Data

```mojo
// GOOD: Deterministic random data
fn test_random_operation() raises:
    TestFixtures.set_seed()  // Reproducible
    var data = generate_random_data()
    // Test data...

// BAD: Non-deterministic
fn test_random_operation() raises:
    var data = generate_random_data()  // Different every time!
    // Test may pass sometimes, fail others
```

## Assertion Selection

[... continue with comprehensive best practices ...]
```

### 6. Troubleshooting Guide

**Create**: `docs/testing/troubleshooting.md`

```markdown
# Test Framework Troubleshooting

## Common Issues and Solutions

### Issue: "Floating-Point Values Not Equal"

**Error**: Test fails with values that look equal

**Cause**: Floating-point precision issues

**Solution**: Use `assert_almost_equal` instead of `assert_equal`

```mojo
// WRONG
assert_equal(0.1 + 0.2, 0.3)  // May fail!

// RIGHT
assert_almost_equal(0.1 + 0.2, 0.3, tolerance=1e-10)
```

### Issue: "Tests Pass Sometimes, Fail Sometimes"

**Error**: Non-deterministic test failures

**Cause**: Random data without seeding

**Solution**: Always use `TestFixtures.set_seed()`

```mojo
fn test_my_feature() raises:
    TestFixtures.set_seed()  // Add this!
    var random_data = generate_data()
    // Test code...
```

[... continue with more troubleshooting ...]
```

## Deliverables Summary

### Documentation Hierarchy

**Level 1: Overview**:
- [ ] `docs/testing/README.md` - Navigation and overview

**Level 2: Getting Started**:
- [ ] `docs/testing/quick-start.md` - 2-minute start
- [ ] `docs/testing/tutorial.md` - Progressive learning

**Level 3: Reference**:
- [ ] `docs/testing/api-reference.md` - Complete API
- [ ] `docs/testing/fixtures-catalog.md` - All fixtures
- [ ] `docs/testing/best-practices.md` - Patterns and conventions

**Level 4: Support**:
- [ ] `docs/testing/troubleshooting.md` - Common issues
- [ ] `docs/testing/examples/` - Code examples

## References

- **Source Plan**: [notes/plan/02-shared-library/04-testing/01-test-framework/plan.md](../../../plan/02-shared-library/04-testing/01-test-framework/plan.md)
- **Related Issues**:
  - Issue #448: [Plan] Test Framework
  - Issue #449: [Test] Test Framework
  - Issue #450: [Impl] Test Framework
  - Issue #452: [Cleanup] Test Framework
  - All child component issues (#433-447)

## Implementation Notes

### Documentation Philosophy

**Goal**: Make it easy for anyone to write effective tests

**Principles**:
1. **Progressive Disclosure**: Start simple, reveal complexity gradually
2. **Examples First**: Show, don't just tell
3. **Searchable**: Clear structure and navigation
4. **Accurate**: Keep docs in sync with code

### Minimal Changes Principle

**Current State**: Framework exists and works
**Documentation Need**: High - framework is undocumented

**Focus**:
- Create comprehensive documentation
- Don't change working code to fit documentation
- Document what exists, recommend improvements for later
