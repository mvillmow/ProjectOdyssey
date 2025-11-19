# Issue #450: [Impl] Test Framework

## Objective

Implement any missing components or improvements to the test framework based on findings from the Test phase (#449), focusing on integration, reliability, and usability enhancements.

## Deliverables

- Framework integration improvements (if needed)
- Reliability enhancements (if gaps found)
- Performance optimizations (if bottlenecks identified)
- Usability improvements based on testing experience
- Updated documentation reflecting changes

## Success Criteria

- [ ] All gaps identified in Test phase are addressed
- [ ] Framework components integrate seamlessly
- [ ] Tests are reliably deterministic and reproducible
- [ ] Framework performance meets requirements
- [ ] Framework is easy to use for test developers
- [ ] Documentation reflects final implementation state

## Current State Analysis

### Framework Status

**Child Components**:
1. **Setup Testing** (#433-437): Test infrastructure setup
2. **Test Utilities** (#438-442): Assertions, comparisons, generators
3. **Test Fixtures** (#443-447): Seed fixtures, data generators

**Current State**:
- All child components implemented (or documented)
- 91+ tests passing across codebase
- Framework is functional and used successfully

**Potential Gaps** (to be identified in Test phase #449):
- Integration points between components
- Missing convenience functions
- Performance bottlenecks
- Usability pain points

## Implementation Strategy

### Phase 1: Review Test Phase Findings (#449)

**Questions to Answer**:
1. Do all components integrate smoothly?
2. Are there reliability issues?
3. Are there performance problems?
4. Are there usability improvements needed?

**Decision Process**:
- **If no gaps found**: Document success, minimal implementation
- **If minor gaps found**: Implement targeted fixes
- **If major gaps found**: Implement comprehensive improvements

### Phase 2: Potential Implementation Areas

**Note**: These are hypothetical - only implement if Test phase identifies actual needs.

#### A. Integration Improvements

**If Test Phase Finds**: Components don't import cleanly

**Potential Solution**: Create unified import module
```mojo
"""Unified test framework imports for convenience."""

# File: tests/framework.mojo

# Re-export all commonly used testing components
from tests.shared.conftest import (
    # Assertions
    assert_true, assert_false, assert_equal, assert_not_equal,
    assert_almost_equal, assert_greater, assert_less,

    # Fixtures
    TestFixtures,

    # Generators
    create_test_vector, create_test_matrix, create_sequential_vector,

    # Benchmarks
    BenchmarkResult, print_benchmark_results
)

from tests.helpers.assertions import (
    # ExTensor assertions
    assert_shape, assert_dtype, assert_numel, assert_dim,
    assert_value_at, assert_all_values, assert_all_close,
    assert_contiguous
)

# Then tests can do:
# from tests.framework import assert_true, TestFixtures, create_test_vector
```

**Trade-offs**:
- **Pro**: Single import for common needs
- **Con**: Adds indirection, hides original sources
- **Decision**: Only if tests show import pain

#### B. Reliability Enhancements

**If Test Phase Finds**: Non-deterministic behavior

**Potential Solutions**:

**1. Automatic Seed Management**:
```mojo
struct TestContext:
    """Automatic test setup/teardown."""

    fn __init__(inout self):
        """Setup: automatically set deterministic seed."""
        TestFixtures.set_seed()

    fn __del__(owned self):
        """Teardown: cleanup if needed."""
        pass

# Usage:
fn test_something() raises:
    var ctx = TestContext()  // Auto-seeds
    // Test code here
    // Auto-cleanup on scope exit
```

**2. Test Isolation Helpers**:
```mojo
fn with_isolated_seed[test_fn: fn () raises -> None](seed: Int = 42) raises:
    """Run test function with isolated random seed.

    Parameters:
        test_fn: Test function to run.

    Args:
        seed: Random seed to use.
    """
    var old_seed = get_current_seed()  // If available
    random.seed(seed)
    test_fn()
    random.seed(old_seed)  // Restore
```

**Decision**: Only if Test phase shows isolation problems

#### C. Performance Optimizations

**If Test Phase Finds**: Slow fixture creation

**Potential Solutions**:

**1. Fixture Caching** (for expensive fixtures):
```mojo
struct CachedFixture[T: AnyType]:
    """Cache fixture data across tests in same module."""

    var cached_data: Optional[T]

    fn __init__(inout self):
        self.cached_data = None

    fn get_or_create[create_fn: fn () -> T]() -> T:
        """Get cached data or create if not cached."""
        if self.cached_data is None:
            self.cached_data = Some(create_fn())
        return self.cached_data.value()

# Usage (module-level):
var expensive_fixture = CachedFixture[Tensor]()

fn test_1() raises:
    var data = expensive_fixture.get_or_create[create_large_tensor]()

fn test_2() raises:
    var data = expensive_fixture.get_or_create[create_large_tensor]()
    // Reuses cached data
```

**2. Lazy Fixture Initialization**:
```mojo
struct LazyFixture[T: AnyType]:
    """Defer fixture creation until first use."""

    var data: Optional[T]
    var creator: fn () -> T

    fn get(inout self) -> T:
        if self.data is None:
            self.data = Some(self.creator())
        return self.data.value()
```

**Decision**: Only if performance benchmarks show need

#### D. Usability Improvements

**If Test Phase Finds**: Common patterns are verbose

**Potential Solutions**:

**1. Assertion Shortcuts**:
```mojo
fn assert_vector_equals(
    actual: List[Float32],
    expected: List[Float32],
    tolerance: Float32 = 1e-6
) raises:
    """Assert two vectors are equal (convenience wrapper).

    Args:
        actual: Actual vector.
        expected: Expected vector.
        tolerance: Tolerance for floating-point comparison.

    Raises:
        Error if vectors differ.
    """
    assert_equal(len(actual), len(expected), "Vector lengths differ")

    for i in range(len(actual)):
        assert_almost_equal(actual[i], expected[i], tolerance,
                            "Vectors differ at index " + str(i))

fn assert_matrix_equals(
    actual: List[List[Float32]],
    expected: List[List[Float32]],
    tolerance: Float32 = 1e-6
) raises:
    """Assert two matrices are equal."""
    assert_equal(len(actual), len(expected), "Matrix row counts differ")

    for i in range(len(actual)):
        assert_vector_equals(actual[i], expected[i], tolerance)
```

**2. Test Data Builders**:
```mojo
struct VectorBuilder:
    """Fluent builder for test vectors."""

    var size: Int
    var value: Float32
    var seed: Optional[Int]

    fn __init__(inout self):
        self.size = 10  // Default
        self.value = 1.0  // Default
        self.seed = None

    fn with_size(inout self, size: Int) -> Self:
        self.size = size
        return self

    fn with_value(inout self, value: Float32) -> Self:
        self.value = value
        return self

    fn with_seed(inout self, seed: Int) -> Self:
        self.seed = Some(seed)
        return self

    fn build(self) -> List[Float32]:
        if self.seed.has_value():
            random.seed(self.seed.value())
        return create_test_vector(self.size, self.value)

# Usage:
var vec = VectorBuilder()
    .with_size(100)
    .with_value(3.14)
    .with_seed(42)
    .build()
```

**Decision**: Only if tests show verbosity pain

#### E. Documentation and Examples

**Always Implement** (regardless of Test phase):

**1. Complete Examples**:
```mojo
// File: tests/examples/example_basic_test.mojo
"""Example of basic test using framework."""

from tests.shared.conftest import assert_equal, TestFixtures

fn test_basic_example() raises:
    """Example: Basic assertion."""
    var result = 2 + 2
    assert_equal(result, 4)

fn test_with_fixture() raises:
    """Example: Using fixtures."""
    TestFixtures.set_seed()
    var random_val = randn()
    // Use random_val in test

fn test_with_data_generator() raises:
    """Example: Using data generators."""
    var vec = create_test_vector(10, 5.0)
    assert_equal(len(vec), 10)
```

**2. Best Practices Guide** (update based on experience):
```markdown
# Test Framework Best Practices

## 1. Always Use Deterministic Seeds for Random Data

```mojo
// GOOD
fn test_random_operation() raises:
    TestFixtures.set_seed()  // Ensures reproducibility
    var data = generate_random_data()
    // test code

// BAD
fn test_random_operation() raises:
    var data = generate_random_data()  // Non-deterministic!
    // test code
```

## 2. Use Appropriate Assertions

```mojo
// For exact equality (integers, strings)
assert_equal(42, 42)

// For floating-point (use tolerance)
assert_almost_equal(0.1 + 0.2, 0.3, tolerance=1e-10)

// For tensors
assert_all_close(tensor1, tensor2, rtol=1e-5, atol=1e-8)
```

## 3. Keep Tests Focused and Independent

```mojo
// GOOD: Each test is independent
fn test_addition() raises:
    assert_equal(2 + 2, 4)

fn test_subtraction() raises:
    assert_equal(5 - 3, 2)

// BAD: Tests depend on shared mutable state
var global_counter = 0

fn test_1() raises:
    global_counter += 1  // Modifies shared state
    assert_equal(global_counter, 1)

fn test_2() raises:
    global_counter += 1  // Depends on test_1
    assert_equal(global_counter, 2)
```
```

### Phase 3: Implementation Priorities

**Priority 1: Critical Issues** (implement immediately):
- Bugs in framework components
- Reliability problems
- Integration failures

**Priority 2: High-Impact Improvements** (implement if beneficial):
- Performance optimizations with measurable impact
- Usability improvements that reduce test writing effort
- Missing convenience functions used by many tests

**Priority 3: Nice-to-Have** (consider for future):
- Advanced features not needed now
- Optimizations for uncommon cases
- Features that can wait

### Phase 4: Testing and Validation

After any implementations:
1. **Run full test suite**: Ensure no regressions
2. **Validate improvements**: Verify implementations solve identified problems
3. **Update documentation**: Reflect new features/changes
4. **Get feedback**: From other test developers if available

## References

- **Source Plan**: [notes/plan/02-shared-library/04-testing/01-test-framework/plan.md](../../../plan/02-shared-library/04-testing/01-test-framework/plan.md)
- **Related Issues**:
  - Issue #448: [Plan] Test Framework
  - Issue #449: [Test] Test Framework (source of requirements)
  - Issue #451: [Package] Test Framework
  - Issue #452: [Cleanup] Test Framework
  - Issues #433-437: Setup Testing
  - Issues #438-442: Test Utilities
  - Issues #443-447: Test Fixtures
- **Existing Code**:
  - `/tests/shared/conftest.mojo`
  - `/tests/helpers/assertions.mojo`
  - All test files using the framework

## Implementation Notes

### Findings from Test Phase

(To be filled based on Issue #449 results)

**Integration Issues Found**:
- TBD

**Reliability Issues Found**:
- TBD

**Performance Issues Found**:
- TBD

**Usability Issues Found**:
- TBD

### Implementation Decisions

**Decision Log**:

| Date | Issue Type | Decision | Rationale |
|------|-----------|----------|-----------|
| TBD | Integration | Implement/Skip | TBD |
| TBD | Reliability | Implement/Skip | TBD |
| TBD | Performance | Implement/Skip | TBD |
| TBD | Usability | Implement/Skip | TBD |

### Code Changes

**Files Modified**:
- TBD based on actual needs

**Files Created**:
- TBD based on actual needs

**Examples Added**:
- `/tests/examples/` directory
- Example tests demonstrating framework usage

### Minimal Changes Principle

**Key Insight**: The framework already works (91+ tests passing). Only implement what's truly needed.

**Approach**:
1. Start with minimal implementation
2. Address only issues found in Test phase
3. Don't over-engineer
4. Keep simple and maintainable

**Avoid**:
- Implementing features "just in case"
- Adding complexity without clear benefit
- Breaking existing tests
- Over-abstracting simple patterns
