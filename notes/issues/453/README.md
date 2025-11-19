# Issue #453: [Plan] Test Core - Design and Documentation

## Objective

Write comprehensive unit tests for all core operations including tensor ops, activations, initializers, and metrics. These tests verify mathematical correctness, handle edge cases, and ensure numerical stability of fundamental building blocks.

## Deliverables

- Tests for all tensor operations
- Tests for all activation functions
- Tests for all initializers
- Tests for all metrics
- Edge case coverage
- Numerical stability verification

## Success Criteria

- [ ] All core operations have unit tests
- [ ] Tests verify mathematical correctness
- [ ] Edge cases are covered
- [ ] Numerical stability is tested

## Design Decisions

### Test Organization Strategy

The test suite for core operations will be organized into four major categories:

1. **Tensor Operations**: Tests for arithmetic operations (add, subtract, multiply, divide), matrix operations (matmul, transpose), and reduction operations (sum, mean, max, min)
2. **Activation Functions**: Tests for common activation functions (ReLU, sigmoid, tanh, softmax) with focus on gradient computation correctness
3. **Initializers**: Tests for weight initialization strategies (Xavier, He, uniform, normal) with statistical verification of distribution properties
4. **Metrics**: Tests for training metrics (accuracy, loss functions, confusion matrix) with manual verification on small examples

### Verification Approach

**Mathematical Correctness**: Use known mathematical results and properties to verify operations. For example:

- Tensor operations: Verify commutative, associative, and distributive properties
- Activation functions: Check derivatives against analytical solutions
- Initializers: Verify mean, variance, and distribution shape statistically
- Metrics: Hand-calculate expected values for small test cases

**Edge Case Coverage**: Comprehensive testing of boundary conditions:

- Zero tensors (all elements are zero)
- Single-element tensors
- Empty tensors (zero-dimensional)
- Infinite values (positive and negative infinity)
- NaN (Not a Number) propagation
- Very large and very small values (numerical stability)

**Data Type Testing**: Verify operations work correctly with various dtypes:

- Float32 (primary type for ML workloads)
- Float64 (higher precision for verification)
- Int32/Int64 (for indexing and discrete operations)

### Numerical Stability Testing

Test numerical stability by:

- Computing results with different input magnitudes
- Verifying precision loss is within acceptable bounds
- Testing operations near numerical limits
- Checking for catastrophic cancellation in subtraction
- Verifying gradient computation stability for activation functions

### Statistical Verification for Initializers

For weight initialization functions, verify:

- Mean is close to expected value (within statistical tolerance)
- Variance matches theoretical distribution
- Distribution shape (using Kolmogorov-Smirnov test or similar)
- Independence of samples (when applicable)
- Scaling behavior with input/output dimensions

### Test Data Strategy

Use multiple approaches for test data:

1. **Known Results**: Small hand-calculated examples with exact expected outputs
2. **Property-Based**: Verify mathematical properties hold (e.g., softmax outputs sum to 1)
3. **Random Testing**: Generate random inputs within valid ranges
4. **Reference Implementation**: Compare against NumPy/SciPy for complex operations
5. **Edge Cases**: Explicitly constructed pathological cases

### Performance Baselines

While this is primarily a correctness testing phase, establish performance baselines:

- Record execution time for standard test cases
- Document memory usage patterns
- Identify operations that may need optimization
- Create benchmarks for regression testing

### Testing Framework

Use Mojo's testing infrastructure:

- Leverage `assert_equal`, `assert_almost_equal` for floating-point comparisons
- Use appropriate tolerance levels for numerical comparisons (e.g., 1e-6 for float32, 1e-12 for float64)
- Structure tests in a modular way for easy debugging
- Provide clear error messages that indicate expected vs actual values

## References

### Source Plan

[notes/plan/02-shared-library/04-testing/02-unit-tests/01-test-core/plan.md](../../../../plan/02-shared-library/04-testing/02-unit-tests/01-test-core/plan.md)

### Related Issues

- Issue #454: [Test] Test Core - Write Tests
- Issue #455: [Impl] Test Core - Implementation
- Issue #456: [Package] Test Core - Integration and Packaging
- Issue #457: [Cleanup] Test Core - Refactor and Finalize

### Parent Component

[notes/plan/02-shared-library/04-testing/02-unit-tests/plan.md](../../../../plan/02-shared-library/04-testing/02-unit-tests/plan.md)

## Implementation Notes

### Current Test Coverage

**Existing Test Files** (in `/home/user/ml-odyssey/tests/shared/core/`):
- `test_tensors.mojo` - Empty (1 line only)
- `test_module.mojo` - TDD stubs with TODOs
- `test_layers.mojo` - TDD stubs with 16 test functions defined
- `test_initializers.mojo` - TDD stubs with TODOs
- `test_activations.mojo` - TDD stubs with TODOs

**Test Status Analysis**:
- **Total test functions**: ~50+ defined across core test files
- **Implementation status**: Mostly TDD stubs with TODO(#1538) markers
- **Working tests**: 0 (all are stubs waiting for implementation)
- **Test runner**: Not yet created for core tests

**Comparison with Data Tests** (for reference):
- Data tests have 91+ implemented tests
- Data tests include working test runner (`tests/shared/data/run_all_tests.mojo`)
- Data tests demonstrate good TDD practices with real implementations

### Gap Analysis

**What Exists**:
1. **Well-defined test structure** - Test files organized by component
2. **Clear API contracts** - Test stubs document expected interfaces
3. **Test utilities** - Shared fixtures in `tests/shared/conftest.mojo`
4. **Comprehensive coverage plan** - TODOs reference all major components

**What's Missing**:
1. **Implementations** - All core tests are stubs, need real test logic
2. **Test data** - Need reference tensors and expected outputs
3. **Mathematical verification** - Need golden values for correctness checks
4. **Statistical tests** - Initializer distribution verification not implemented
5. **Edge case tests** - Boundary conditions not yet implemented
6. **Test runner** - No unified runner like data tests have

### Recommendations for Planning Phase

1. **Prioritize Implementation Order**:
   - Start with tensor operations (most fundamental)
   - Follow with activations (depend on tensor ops)
   - Then initializers (depend on random and tensor ops)
   - Finally metrics (depend on all above)

2. **Create Test Data Sets**:
   - Define standard test tensors (shapes: 1D, 2D, 3D, 4D)
   - Generate golden values using reference implementations
   - Document edge cases to test

3. **Statistical Verification Plan**:
   - Define acceptable tolerance levels (1e-6 for float32)
   - Specify sample sizes for statistical tests (N=1000+)
   - Document expected distributions for initializers

4. **Test Runner Development**:
   - Create `tests/shared/core/run_all_tests.mojo` similar to data tests
   - Integrate with CI/CD pipeline
   - Add performance benchmarking

5. **Update TODO References**:
   - Change TODO(#1538) to TODO(#455) for implementation phase
   - Track which tests block others
