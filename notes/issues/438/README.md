# Issue #438: [Plan] Test Utilities - Design and Documentation

## Objective

Create reusable test utilities and helper functions to simplify test writing and reduce duplication, including
assertion helpers, comparison functions for tensors, test data generators, and mock objects to make tests easier to
write and maintain.

## Deliverables

- Tensor comparison utilities (approximate equality with configurable tolerance)
- Shape and dimension assertion helpers
- Test data generators (random tensors, datasets)
- Mock objects for dependencies
- Timing and profiling utilities for performance tests

## Success Criteria

- [ ] Utilities reduce test code duplication across the test suite
- [ ] Comparisons handle floating point arithmetic appropriately (epsilon tolerance)
- [ ] Generators produce valid and deterministic test data
- [ ] Mocks effectively isolate units under test from external dependencies
- [ ] All utilities have clear, comprehensive documentation
- [ ] Error messages from assertions are clear and actionable

## Design Decisions

### 1. Tensor Comparison Strategy

**Decision**: Implement approximate equality comparison with configurable tolerance for floating-point tensors.

**Rationale**: Floating-point arithmetic is inherently imprecise, and exact equality checks will fail for numerically
equivalent results. The utility should:

- Support absolute tolerance (atol) and relative tolerance (rtol) parameters
- Provide clear error messages showing actual vs expected values and where differences occur
- Handle edge cases (NaN, infinity, zero)
- Support different tensor shapes and dtypes

### Key Considerations

- Default tolerances should match common testing needs (e.g., atol=1e-5, rtol=1e-5)
- Error messages should indicate which elements differ and by how much
- Should work with Mojo's tensor types

### 2. Assertion Helper Design

**Decision**: Create focused assertion helpers for common test patterns rather than a monolithic assertion library.

**Rationale**: Following YAGNI principle - implement utilities as they're actually needed:

- Shape assertions: `assert_shape_equals(tensor, expected_shape)` for dimension validation
- Range assertions: `assert_in_range(tensor, min, max)` for value bounds checking
- Type assertions: `assert_dtype_equals(tensor, expected_dtype)` for type safety
- Zero/non-zero assertions: `assert_all_zero()`, `assert_any_nonzero()` for sparsity checks

### Key Considerations

- Each helper should have a single responsibility (SOLID principle)
- Error messages should be descriptive and include context
- Helpers should compose well with each other

### 3. Test Data Generation Approach

**Decision**: Implement deterministic random generators with explicit seed control.

**Rationale**: Test data must be reproducible for reliable testing:

- Use seeded random number generators (not time-based)
- Provide factory functions for common patterns (random normal, uniform, etc.)
- Support shape, dtype, and value range specifications
- Generate both edge cases and typical cases

### Key Patterns

```mojo
# Random tensor with specified distribution
fn random_tensor(shape: TensorShape, seed: Int, dtype: DType) -> Tensor

# Random tensor within range
fn random_range_tensor(shape: TensorShape, min: Float64, max: Float64, seed: Int) -> Tensor

# Common distributions
fn random_normal(shape: TensorShape, mean: Float64, std: Float64, seed: Int) -> Tensor
fn random_uniform(shape: TensorShape, low: Float64, high: Float64, seed: Int) -> Tensor
```text

### 4. Mock Object Strategy

**Decision**: Create minimal mock implementations for external dependencies, focusing on filesystem and I/O operations.

**Rationale**: Following KISS principle - mocks should be simple and focused:

- Mock file system for testing file operations without disk I/O
- Mock data loaders for testing training loops without real data
- Mock timers for testing performance metrics without waiting

### Key Considerations

- Mocks should have the same interface as real objects
- Mocks should be stateful to verify interactions
- Keep mock complexity minimal - if a mock is complex, the design may need refactoring

### 5. Timing and Profiling Utilities

**Decision**: Provide simple timing decorators and context managers for performance validation.

**Rationale**: Performance tests need reliable timing without complexity:

- Context manager for timing code blocks: `with Timer() as t:`
- Decorator for timing function calls: `@timed`
- Simple profiling output (mean, std, min, max over multiple runs)

### Key Considerations

- Timing should be wall-clock time for simplicity
- Should support multiple runs and statistical aggregation
- Results should be easily comparable across runs

### 6. Memory Management in Utilities

**Decision**: Follow Mojo's ownership model consistently - use `borrowed` for read-only operations, `owned` for transfers.

**Rationale**: Test utilities must not introduce memory safety issues:

- Comparison functions take `borrowed` tensors (read-only)
- Generators return `owned` tensors (transfer ownership)
- Assertion helpers take `borrowed` parameters (don't modify state)

### Pattern Example

```mojo
fn assert_tensors_equal(borrowed t1: Tensor, borrowed t2: Tensor, atol: Float64, rtol: Float64) raises:
    # Read-only comparison, no ownership transfer
    pass

fn random_tensor(shape: TensorShape, seed: Int) -> Tensor^:
    # Returns owned tensor to caller
    pass
```text

### 7. Documentation Requirements

**Decision**: Every utility function must have comprehensive docstrings following Mojo conventions.

**Rationale**: Test utilities are developer-facing tools - clear documentation is critical:

- Purpose and use cases
- Parameter descriptions with types and constraints
- Return value description
- Examples of typical usage
- Edge cases and error conditions

## References

### Source Plan

- [Test Utilities Plan](notes/plan/02-shared-library/04-testing/01-test-framework/02-test-utilities/plan.md)
- [Test Framework Parent Plan](notes/plan/02-shared-library/04-testing/01-test-framework/plan.md)

### Related Issues

- Issue #439: [Test] Test Utilities - Test the test utilities themselves
- Issue #440: [Impl] Test Utilities - Implement the utility functions
- Issue #441: [Package] Test Utilities - Package and integrate utilities
- Issue #442: [Cleanup] Test Utilities - Refactor and finalize

### Architecture Documentation

- [Mojo Language Review Patterns](.claude/agents/mojo-language-review-specialist.md)
- [Testing Strategy](notes/plan/02-shared-library/04-testing/plan.md)

## Implementation Notes

**Note**: To be filled during implementation phase

### Discovered During Implementation

- TBD

### Technical Challenges

- TBD

### Deviations from Plan

- TBD
