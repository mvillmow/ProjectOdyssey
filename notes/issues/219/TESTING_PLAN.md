# ExTensor Testing Plan - Comprehensive Analysis and Strategy

**Status**: 346 tests implemented out of ~680 total planned
**Coverage**: ~51% complete (tests for implemented operations: ~90%, tests for future operations: ~20%)
**Last Updated**: 2025-11-17

## Current Test Coverage Analysis

### Test Files Summary

| File | Tests | Status | Coverage |
|------|-------|--------|----------|
| test_creation.mojo | 40 | ✅ Complete | Creation operations (7/7) |
| test_arithmetic.mojo | 40 | ✅ Complete | Arithmetic operations (7/7) |
| test_comparison_ops.mojo | 19 | ✅ Complete | Comparison operations (6/6) |
| test_reductions.mojo | 21 | ✅ Complete | Reduction operations (4/4) |
| test_properties.mojo | 36 | ✅ Complete | Tensor properties |
| test_integration.mojo | 19 | ✅ Complete | Multi-op workflows |
| test_broadcasting.mojo | 19 | ⚠️ Partial | 17 impl, 2 placeholders |
| test_comparison.mojo | 24 | ❌ Duplicate | DELETE (replaced by test_comparison_ops.mojo) |
| test_reduction.mojo | 33 | ❌ Duplicate | DELETE (replaced by test_reductions.mojo) |
| test_matrix.mojo | 19 | 🚧 Placeholders | Matrix operations TODO |
| test_edge_cases.mojo | 28 | 🚧 Placeholders | Edge cases TODO |
| test_shape.mojo | 23 | 🚧 Placeholders | Shape manipulation TODO |
| test_utility.mojo | 25 | 🚧 Placeholders | Utility operations TODO |

**Total**: 346 test functions across 13 files
**Implemented**: ~210 tests (61%)
**Placeholders**: ~136 tests (39%)
**Duplicates to delete**: 57 tests (test_comparison.mojo + test_reduction.mojo)

### Coverage by Operation Category

#### ✅ Fully Tested Operations (21 ops, ~160 tests)

### Creation Operations (7 ops, 40 tests)

- zeros() - 5 tests (1D, 2D, 3D, scalar, dtype)
- ones() - 3 tests (1D, 2D, dtype)
- full() - 4 tests (1D, 2D, fill values)
- empty() - 3 tests (allocation checks)
- arange() - 5 tests (basic, step, ranges)
- eye() - 2 tests (square, rectangular)
- linspace() - 4 tests (basic, single point, inclusive)
- Plus: 7 dtype tests, 3 edge case tests

### Arithmetic Operations (7 ops, 40 tests)

- add() - 4 tests (1D, 2D, zeros, negatives)
- subtract() - 4 tests (1D, 2D, zeros, negatives)
- multiply() - 5 tests (1D, 2D, by zero, by one, negatives)
- divide() - 4 tests (NEW - basic, by one, by two, negatives)
- floor_divide() - 3 tests (NEW - positive, negative)
- modulo() - 3 tests (NEW - positive, negative, fractional)
- power() - 4 tests (NEW - integer exponents, zero, one, negative base)
- Plus: 5 operator overload tests, 3 dtype tests, 3 shape tests, 2 error tests

### Comparison Operations (6 ops, 19 tests)

- equal() - 3 tests (same values, different values, dunder)
- not_equal() - 3 tests (same, different, dunder)
- less() - 3 tests (true, false, dunder)
- less_equal() - 3 tests (less, equal, dunder)
- greater() - 3 tests (true, false, dunder)
- greater_equal() - 3 tests (greater, equal, dunder)
- Plus: 1 negative values test

### Reduction Operations (4 ops, 21 tests)

- sum() - 5 tests (ones, 2D, arange, keepdims, dtype)
- mean() - 5 tests (ones, 2D, arange, keepdims, dtype)
- max_reduce() - 5 tests (same values, arange, negative, keepdims, dtype)
- min_reduce() - 5 tests (same values, arange, negative, keepdims, dtype)
- Plus: 1 consistency test

### Properties & Integration (40 tests)

- test_properties.mojo - 36 tests (shape, dtype, numel, strides, contiguity)
- test_integration.mojo - 19 tests (chained ops, ML patterns)

#### ⚠️ Partially Tested Operations

**Broadcasting (infrastructure exists, 17/19 tests implemented)**

- broadcast_shapes() - 5 tests
- are_shapes_broadcastable() - 3 tests
- compute_broadcast_strides() - 4 tests
- BroadcastIterator - 5 tests
- **Missing**: Full integration tests with arithmetic/comparison ops

#### 🚧 Not Yet Tested (130+ operations planned)

### Matrix Operations (4 ops, 19 placeholder tests)

- matmul() - 7 placeholders
- transpose() - 5 placeholders
- dot() - 4 placeholders
- outer() - 3 placeholders

### Edge Cases (35 placeholder tests)

- NaN handling - 10 placeholders
- Infinity handling - 8 placeholders
- Overflow/underflow - 7 placeholders
- Division by zero - 5 placeholders
- Empty tensors - 5 placeholders

### Shape Manipulation (23 placeholder tests)

- reshape() - 6 placeholders
- squeeze() - 4 placeholders
- concatenate() - 5 placeholders
- stack() - 4 placeholders
- split() - 4 placeholders

### Utility Operations (25 placeholder tests)

- copy() - 3 placeholders
- Property getters - 5 placeholders
- Conversions - 7 placeholders
- Dunder methods - 5 placeholders
- diff() - 5 placeholders

## Testing Priorities

### Priority 1: Complete Testing for Implemented Operations (CRITICAL)

### Immediate Actions

1. ✅ **Delete duplicate test files**
   - Remove test_comparison.mojo (replaced by test_comparison_ops.mojo)
   - Remove test_reduction.mojo (replaced by test_reductions.mojo)

1. ✅ **Complete broadcasting tests** (Est: 1 day)
   - Implement 2 remaining placeholder tests
   - Add integration tests: broadcasting + arithmetic
   - Add integration tests: broadcasting + comparison
   - Test all broadcasting edge cases
   - **Total new tests**: ~10-15

1. ✅ **Add comprehensive edge case tests for existing ops** (Est: 1-2 days)
   - Division by zero (IEEE 754 compliance)
   - NaN propagation in arithmetic
   - Infinity handling
   - Large/small number overflow/underflow
   - Empty tensor operations
   - **Total new tests**: ~25-30

### Priority 2: Tests for Priority 1 Operations (HIGH)

These tests should be written BEFORE implementing the operations (TDD approach).

#### 2.1 Matrix Operations Tests (Est: 1 day)

**matmul() - 15 tests**

- 2D @ 2D: (m,n) @ (n,p) → (m,p) - 3 tests
- Batched: (b,m,n) @ (b,n,p) → (b,m,p) - 3 tests
- Vector @ matrix - 2 tests
- Matrix @ vector - 2 tests
- Identity matrix behavior - 2 tests
- Mismatched shapes error - 1 test
- Dtype preservation - 2 tests

**transpose() - 10 tests**

- 2D transpose: (3,4) → (4,3) - 2 tests
- 3D+ transpose (last 2 dims) - 2 tests
- Zero-copy view verification - 2 tests
- Double transpose = identity - 1 test
- Contiguity checks - 2 tests
- 1D tensor behavior - 1 test

**dot() - 8 tests**

- 1D vector dot product - 3 tests
- Orthogonal vectors - 1 test
- Zero dot product - 1 test
- Negative values - 1 test
- Dtype preservation - 2 tests

**outer() - 7 tests**

- Basic outer product - 2 tests
- Shape verification: (m,) ⊗ (n,) → (m,n) - 2 tests
- Zero vectors - 1 test
- Negative values - 1 test
- Dtype preservation - 1 test

**Total new tests**: ~40 tests

#### 2.2 Element-wise Math Functions Tests (Est: 2 days)

**exp() - 6 tests**

- Basic exponential - 1 test
- exp(0) = 1 - 1 test
- exp(negative) - 1 test
- Large inputs (overflow check) - 1 test
- Array of values - 1 test
- Dtype preservation - 1 test

**log() - 8 tests**

- Natural logarithm - 1 test
- log(1) = 0 - 1 test
- log(e) = 1 - 1 test
- Negative input → NaN - 1 test
- log(0) → -inf - 1 test
- Array of values - 1 test
- Domain errors - 1 test
- Dtype preservation - 1 test

**sqrt() - 6 tests**

- Basic square root - 1 test
- sqrt(0) = 0 - 1 test
- sqrt(1) = 1 - 1 test
- Negative input → NaN - 1 test
- Array of values - 1 test
- Dtype preservation - 1 test

**abs() - 5 tests**

- Positive values - 1 test
- Negative values - 1 test
- Zero - 1 test
- Mixed signs - 1 test
- Dtype preservation - 1 test

**sign() - 5 tests**

- Positive → 1 - 1 test
- Negative → -1 - 1 test
- Zero → 0 - 1 test
- Mixed values - 1 test
- Dtype preservation - 1 test

**tanh() - 6 tests**

- Basic hyperbolic tangent - 1 test
- tanh(0) = 0 - 1 test
- Saturation at large inputs - 1 test
- Negative values - 1 test
- Array of values - 1 test
- Dtype preservation - 1 test

**sin(), cos() - 10 tests each**

- Basic trigonometric - 2 tests
- Special angles (0, π/2, π) - 3 tests
- Negative inputs - 1 test
- Periodicity - 1 test
- Range verification [-1, 1] - 1 test
- Array of values - 1 test
- Dtype preservation - 1 test

**Total new tests**: ~62 tests

#### 2.3 Axis-Specific Reduction Tests (Est: 1 day)

**sum() with axis - 12 tests**

- Sum along axis 0 - 2 tests
- Sum along axis 1 - 2 tests
- Sum along axis -1 (last) - 1 test
- Sum along multiple axes - 2 tests
- keepdims=True - 2 tests
- 3D+ tensors - 2 tests
- Invalid axis error - 1 test

**mean() with axis - 10 tests**

- Mean along each axis - 3 tests
- keepdims - 2 tests
- Multiple axes - 2 tests
- 3D+ tensors - 2 tests
- Invalid axis error - 1 test

**max_reduce(), min_reduce() with axis - 8 tests each**

- Along each axis - 3 tests
- keepdims - 2 tests
- 3D+ tensors - 2 tests
- Invalid axis error - 1 test

**argmax(), argmin() - 10 tests each**

- Return indices - 2 tests
- Along specific axis - 3 tests
- Flat indices - 2 tests
- Ties handling - 2 tests
- Invalid axis error - 1 test

**Total new tests**: ~68 tests

### Priority 3: Shape Manipulation Tests (MEDIUM)

#### 3.1 Core Reshape Operations (Est: 1-2 days)

**reshape() - 15 tests**

- Compatible shapes - 3 tests
- -1 dimension inference - 3 tests
- Incompatible shapes error - 2 tests
- View vs copy verification - 2 tests
- Contiguity requirements - 2 tests
- Multi-dimensional reshapes - 3 tests

**squeeze() - 8 tests**

- Remove all size-1 dims - 2 tests
- Remove specific axis - 2 tests
- No size-1 dims (no-op) - 1 test
- Multi-dimensional - 2 tests
- Invalid axis error - 1 test

**unsqueeze() / expand_dims() - 8 tests**

- Add dimension at start - 2 tests
- Add dimension at end - 2 tests
- Add dimension in middle - 2 tests
- Multiple dimensions - 1 test
- Invalid axis error - 1 test

**flatten() - 5 tests**

- Multi-dimensional → 1D - 2 tests
- Copy verification - 1 test
- Order preservation - 1 test
- Already 1D (no-op) - 1 test

**ravel() - 5 tests**

- View when possible - 2 tests
- Copy when necessary - 2 tests
- Contiguity checks - 1 test

**Total new tests**: ~41 tests

#### 3.2 Concatenation and Stacking (Est: 1-2 days)

**concatenate() - 12 tests**

- Along axis 0 - 2 tests
- Along axis 1 - 2 tests
- Multiple tensors - 2 tests
- Dimension compatibility - 2 tests
- Empty list error - 1 test
- Mismatched shapes error - 2 tests
- Dtype compatibility - 1 test

**stack() - 10 tests**

- Along new axis 0 - 2 tests
- Along new axis 1 - 2 tests
- Multiple tensors - 2 tests
- Shape compatibility - 2 tests
- Mismatched shapes error - 1 test
- Dtype compatibility - 1 test

**split() - 12 tests**

- Equal splits - 3 tests
- Unequal splits - 3 tests
- Along different axes - 2 tests
- Invalid split error - 2 tests
- Edge cases (single split) - 2 tests

**tile() - 8 tests**

- Tile along single axis - 2 tests
- Tile along multiple axes - 2 tests
- Tile factors - 2 tests
- Edge cases - 2 tests

**repeat() - 8 tests**

- Repeat elements - 2 tests
- Along specific axis - 2 tests
- Repeat factors - 2 tests
- Edge cases - 2 tests

**Total new tests**: ~50 tests

### Priority 4: Indexing & Slicing Tests (MEDIUM-HIGH)

#### 4.1 Basic Indexing (Est: 2-3 days)

****getitem**() - 25 tests**

- Integer indexing - 4 tests
- Single slice - 4 tests
- Multi-dimensional slice - 4 tests
- Negative indices - 4 tests
- Stride/step - 4 tests
- Out-of-bounds error - 2 tests
- View verification - 3 tests

****setitem**() - 20 tests**

- Integer indexing - 4 tests
- Slice assignment - 4 tests
- Multi-dimensional - 4 tests
- Broadcasting RHS - 4 tests
- Type checking - 2 tests
- Out-of-bounds error - 2 tests

**Total new tests**: ~45 tests

#### 4.2 Advanced Indexing (Est: 1-2 days)

**take() - 8 tests**

- Take along axis - 3 tests
- Flat indices - 2 tests
- Out-of-bounds - 1 test
- Invalid axis - 1 test
- Negative indices - 1 test

**gather() - 10 tests**

- Gather by indices - 3 tests
- Multi-dimensional - 3 tests
- Broadcasting - 2 tests
- Invalid indices - 2 tests

**scatter() - 10 tests**

- Scatter by indices - 3 tests
- Multi-dimensional - 3 tests
- Broadcasting - 2 tests
- Invalid indices - 2 tests

**masked_select() - 8 tests**

- Boolean mask - 3 tests
- Broadcasting mask - 2 tests
- All False mask - 1 test
- All True mask - 1 test
- Shape mismatch error - 1 test

**nonzero() - 6 tests**

- Find non-zero - 2 tests
- All zero - 1 test
- All non-zero - 1 test
- Multi-dimensional - 2 tests

**Total new tests**: ~42 tests

### Priority 5: Advanced Math & Statistical Tests (MEDIUM)

#### 5.1 Statistical Functions (Est: 1-2 days)

**var() - 10 tests**

- Variance calculation - 3 tests
- Along axes - 3 tests
- Degrees of freedom - 2 tests
- keepdims - 1 test
- Dtype preservation - 1 test

**std() - 10 tests**

- Standard deviation - 3 tests
- Along axes - 3 tests
- Degrees of freedom - 2 tests
- keepdims - 1 test
- Dtype preservation - 1 test

**median() - 8 tests**

- Odd number of elements - 2 tests
- Even number of elements - 2 tests
- Along axes - 2 tests
- keepdims - 1 test
- Dtype preservation - 1 test

**percentile() / quantile() - 10 tests**

- Different percentiles - 3 tests
- Along axes - 2 tests
- Interpolation methods - 3 tests
- Edge cases (0%, 100%) - 2 tests

**cumsum() - 8 tests**

- Cumulative sum - 2 tests
- Along axes - 2 tests
- Different dtypes - 2 tests
- Edge cases - 2 tests

**cumprod() - 8 tests**

- Cumulative product - 2 tests
- Along axes - 2 tests
- Different dtypes - 2 tests
- Edge cases - 2 tests

**prod() - 8 tests**

- Product reduction - 2 tests
- Along axes - 2 tests
- keepdims - 1 test
- Edge cases - 2 tests
- Dtype preservation - 1 test

**Total new tests**: ~62 tests

#### 5.2 Logical Operations (Est: 1 day)

**logical_and(), logical_or(), logical_not() - 8 tests each**

- Boolean inputs - 2 tests
- Broadcasting - 2 tests
- Truth tables - 2 tests
- Edge cases - 2 tests

**logical_xor() - 8 tests**

- Basic XOR - 2 tests
- Broadcasting - 2 tests
- Truth table - 2 tests
- Edge cases - 2 tests

**all() - 10 tests**

- All true - 2 tests
- Some false - 2 tests
- Along axes - 3 tests
- keepdims - 1 test
- Empty tensor - 1 test
- Dtype - 1 test

**any() - 10 tests**

- Some true - 2 tests
- All false - 2 tests
- Along axes - 3 tests
- keepdims - 1 test
- Empty tensor - 1 test
- Dtype - 1 test

**where() - 15 tests** (VERY IMPORTANT)

- Conditional selection - 3 tests
- Broadcasting condition - 3 tests
- Broadcasting x/y - 3 tests
- All conditions - 2 tests
- Chaining - 2 tests
- Dtype promotion - 2 tests

**Total new tests**: ~77 tests

#### 5.3 Extended Math Functions (Est: 1 day)

**ceil(), floor(), trunc(), round() - 6 tests each**

- Positive values - 1 test
- Negative values - 1 test
- Already integer - 1 test
- Arrays - 1 test
- Edge cases - 1 test
- Dtype preservation - 1 test

**clip() / clamp() - 10 tests**

- Clip to range - 3 tests
- Below min - 2 tests
- Above max - 2 tests
- Within range - 1 test
- Broadcasting bounds - 2 tests

**maximum(), minimum() - 8 tests each**

- Element-wise max/min - 3 tests
- Broadcasting - 2 tests
- NaN handling - 2 tests
- Dtype promotion - 1 test

**reciprocal() - 5 tests**

- Basic reciprocal - 1 test
- Division by zero - 1 test
- Arrays - 1 test
- Dtype - 1 test
- Accuracy - 1 test

**square() - 4 tests**

- Positive - 1 test
- Negative - 1 test
- Arrays - 1 test
- Dtype - 1 test

**neg() - 5 tests**

- Unary negation - 2 tests
- Zero - 1 test
- Arrays - 1 test
- Dtype - 1 test

**Total new tests**: ~70 tests

### Priority 6: Specialized Operations Tests (LOW)

#### 6.1 Memory & Type Conversion (Est: 1 day)

**copy(), clone() - 5 tests each**

- Deep copy verification - 2 tests
- Independence check - 2 tests
- Dtype preservation - 1 test

**contiguous() - 6 tests**

- Already contiguous - 1 test
- Non-contiguous → contiguous - 2 tests
- View detection - 2 tests
- Stride verification - 1 test

**astype() - 12 tests**

- All dtype conversions - 6 tests
- Precision loss - 2 tests
- Overflow - 2 tests
- Underflow - 2 tests

**Total new tests**: ~28 tests

#### 6.2 Linear Algebra (Est: 2-3 days)

**tensordot() - 15 tests**
**einsum() - 20 tests** (complex!)
**diagonal() - 8 tests**
**trace() - 6 tests**
**det(), inv() - 10 tests each** (if needed)

**Total new tests**: ~59 tests

#### 6.3 Comparison & Sorting (Est: 1-2 days)

**sort() - 12 tests**
**argsort() - 10 tests**
**topk() - 10 tests**
**isnan(), isinf(), isfinite() - 6 tests each**

**Total new tests**: ~50 tests

## Test Development Strategy

### Development Workflow

1. **Before implementing an operation**:
   - Write comprehensive tests (TDD)
   - Cover happy path, edge cases, errors
   - Document expected behavior
   - Mark as placeholder or skip

1. **While implementing**:
   - Enable tests one by one
   - Fix implementation to pass tests
   - Add more tests as edge cases discovered

1. **After implementing**:
   - Verify all tests pass
   - Add integration tests
   - Add performance benchmarks
   - Document any deviations from spec

### Test Organization Principles

1. **One test file per operation category**
   - Keep related tests together
   - Easy to find and maintain
   - Clear test file naming

1. **Clear test naming convention**
   - `test_<operation>_<scenario>()`
   - Example: `test_reshape_with_inferred_dimension()`
   - Makes failures easy to understand

1. **Comprehensive coverage**
   - Happy path (basic functionality)
   - Edge cases (empty, single element, large)
   - Error cases (invalid inputs)
   - Dtype preservation
   - Shape preservation
   - Broadcasting behavior

1. **Integration tests**
   - Test operations working together
   - Realistic ML workflows
   - End-to-end scenarios

### Quality Metrics

### Target Coverage

- **Line coverage**: >90% for implemented operations
- **Branch coverage**: >85% for all code paths
- **Edge case coverage**: 100% for known edge cases

### Test Quality

- Clear, descriptive test names
- Single assertion focus per test
- Comprehensive docstrings
- Minimal code duplication
- Fast execution (<1s per test)

## Estimated Timeline

### Aggressive Schedule (by priority)

**Priority 1** (Complete existing ops): 2-3 days

- Delete duplicates: 0.5 day
- Complete broadcasting: 1 day
- Edge cases: 1.5 days

**Priority 2** (Priority 1 operations): 4-5 days

- Matrix operations: 1 day
- Element-wise math: 2 days
- Axis-specific reductions: 1.5 days

**Priority 3** (Shape manipulation): 3-4 days

- Core reshape: 1.5 days
- Concatenation/stacking: 2 days

**Priority 4** (Indexing): 4-5 days

- Basic indexing: 2.5 days
- Advanced indexing: 2 days

**Priority 5** (Advanced math/stats): 4-5 days

- Statistical functions: 2 days
- Logical operations: 1 day
- Extended math: 2 days

**Priority 6** (Specialized): 4-5 days

- Memory/type conversion: 1 day
- Linear algebra: 2.5 days
- Sorting/comparison: 1.5 days

**Total**: 21-27 days for comprehensive test suite

### Realistic Schedule

Add 30-50% buffer for:

- Test refinement and debugging
- Discovering new edge cases
- Integration testing
- Code review and refactoring

**Total**: 28-40 days for complete test suite

## Success Metrics

### Minimum Viable Test Suite (MVP)

**Goal**: Tests for implementing LeNet-5

- ✅ Creation operations (40 tests) - DONE
- ✅ Arithmetic operations (40 tests) - DONE
- ✅ Comparison operations (19 tests) - DONE
- ✅ Reductions (21 tests) - DONE
- ✅ Broadcasting (17 tests) - MOSTLY DONE
- 🔲 Matrix operations (40 tests) - NEEDED
- 🔲 Element-wise math (62 tests) - NEEDED
- 🔲 Basic reshape (20 tests) - NEEDED
- 🔲 Basic indexing (25 tests) - NEEDED

**MVP Status**: ~157/227 tests complete (69%)

### Complete Test Suite

**Goal**: 100% Array API Standard coverage

- All 150+ operations tested
- >90% line coverage
- >85% branch coverage
- All edge cases covered
- Performance benchmarks for critical ops
- Integration tests for ML workflows

**Completion Status**: ~346/~800 tests (43%)

## Next Steps

### Immediate Actions (This Week)

1. **Clean up duplicate test files** (0.5 day)

   ```bash
   rm tests/extensor/test_comparison.mojo
   rm tests/extensor/test_reduction.mojo
   ```

1. **Complete broadcasting tests** (1 day)
   - Finish 2 placeholder tests
   - Add 10-15 integration tests

1. **Add edge case tests for existing ops** (1.5 days)
   - Division by zero
   - NaN propagation
   - Overflow/underflow
   - Empty tensors

### This Sprint (Next 2 Weeks)

1. **Write tests for Priority 1 operations** (1 week)
   - Matrix operations: 40 tests
   - Element-wise math: 62 tests
   - Axis-specific reductions: 68 tests

1. **Begin implementing Priority 1 operations** (1 week)
   - Use tests to drive implementation
   - Fix bugs as tests fail
   - Add more tests as needed

### This Month

1. **Complete Priority 1 implementation and testing**
1. **Write tests for Priority 2 operations**
1. **Begin Priority 2 implementation**
1. **MVP milestone**: LeNet-5 implementation possible

## References

- [Python Array API Standard 2023.12](https://data-apis.org/array-api/latest/)
- [NumPy Testing Guidelines](https://numpy.org/doc/stable/reference/testing.html)
- [PyTorch Testing Best Practices](https://github.com/pytorch/pytorch/wiki/)
- Test files location: `tests/extensor/*.mojo`
- Test helpers: `tests/helpers/assertions.mojo`
