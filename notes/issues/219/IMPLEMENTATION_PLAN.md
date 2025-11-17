# ExTensor Implementation Plan

**Status**: In Progress (21/150+ operations implemented)
**Reference**: [Python Array API Standard 2023.12](https://data-apis.org/array-api/latest/)
**Last Updated**: 2025-11-17

## Overview

This document provides a prioritized plan for implementing the remaining ~130 operations from the Python Array API Standard 2023.12 specification.

### Current Status

**✅ Implemented (21 operations)**:
- Creation: zeros, ones, full, empty, arange, eye, linspace (7)
- Arithmetic: add, subtract, multiply, divide, floor_divide, modulo, power (7)
- Comparison: equal, not_equal, less, less_equal, greater, greater_equal (6)
- Reduction: sum, mean, max_reduce, min_reduce (4)
- Accessor methods: _get_float64, _get_int64 (2 internal)

**⚠️ Partially Implemented**:
- Broadcasting: Infrastructure exists, not integrated into operations
- Reductions: Only all-elements (axis=-1), axis-specific TODO

**🚧 In Progress**:
- Comprehensive testing (413+ tests created, need Mojo compiler to run)

## Implementation Priorities

### Priority 1: Core ML Operations (CRITICAL)

These operations are essential for basic neural network implementation and should be completed first.

#### 1.1 Complete Broadcasting (Est: 2-3 days)
**Why**: Required for all element-wise operations with different shapes
**Impact**: HIGH - Blocks matrix operations, loss functions, layer implementations

**Tasks**:
- [ ] Implement full broadcasting in add(), subtract(), multiply()
- [ ] Extend to divide(), floor_divide(), modulo(), power()
- [ ] Extend to all comparison operations
- [ ] Add comprehensive broadcasting tests
- [ ] Verify broadcast_shapes() and compute_broadcast_strides() correctness

**Validation**:
- All existing broadcast placeholder tests should pass
- New test: (3,1,5) + (4,5) → (3,4,5)
- New test: scalar + tensor → broadcast tensor

#### 1.2 Matrix Operations (Est: 3-4 days)
**Why**: Essential for layers (linear, conv), attention mechanisms
**Impact**: HIGH - Critical for any ML model

**Tasks**:
- [ ] **matmul()**: 2D matrix multiplication
  - Implement naive O(n³) algorithm first
  - Optimize with SIMD/blocking later
  - Support batched matmul: (b,m,n) @ (b,n,p) → (b,m,p)
- [ ] **transpose()**: Permute last two dims for 2D+
  - Implement as view (stride manipulation, no data copy)
- [ ] **dot()**: 1D vector dot product
- [ ] **outer()**: Outer product of two 1D vectors
- [ ] Add comprehensive tests for all cases

**Validation**:
- Enable tests in test_matrix.mojo
- Verify shapes: (m,n) @ (n,p) → (m,p)
- Verify batched: (b,m,n) @ (b,n,p) → (b,m,p)
- Verify transpose is zero-copy view

#### 1.3 Element-wise Math Functions (Est: 2-3 days)
**Why**: Activation functions, loss functions
**Impact**: HIGH - Required for forward/backward passes

**Tasks**:
- [ ] **exp()**: Exponential (for softmax, cross-entropy)
- [ ] **log()**: Natural logarithm (for log loss)
- [ ] **sqrt()**: Square root (for normalization, distance)
- [ ] **abs()**: Absolute value (for L1 loss, ReLU variants)
- [ ] **sign()**: Sign function
- [ ] **tanh()**: Hyperbolic tangent activation
- [ ] **sin(), cos()**: Trigonometric (for positional encoding)
- [ ] Optimize with SIMD where possible
- [ ] Add comprehensive tests

**Implementation Note**: Use Mojo's math library where available, implement missing functions.

**Validation**:
- Verify mathematical correctness
- Check edge cases (NaN, inf, zero)
- Verify SIMD optimizations work correctly

#### 1.4 Axis-Specific Reductions (Est: 2-3 days)
**Why**: Normalization layers, loss computation, metrics
**Impact**: HIGH - Required for batch processing

**Tasks**:
- [ ] Implement axis-specific logic for sum(), mean()
- [ ] Implement axis-specific logic for max_reduce(), min_reduce()
- [ ] Add **argmax()**, **argmin()** (return indices)
- [ ] Support multiple axes: axis=(0,2)
- [ ] Verify keepdims parameter works correctly
- [ ] Add comprehensive tests

**Validation**:
- Enable axis-specific tests in test_reduction.mojo
- Verify shape calculation: (3,4,5) sum(axis=1) → (3,5)
- Verify keepdims: (3,4,5) sum(axis=1, keepdims=True) → (3,1,5)
- Verify argmax/argmin return correct indices

### Priority 2: Shape Manipulation (HIGH)

Required for building complex architectures and preprocessing.

#### 2.1 Core Reshape Operations (Est: 2-3 days)

**Tasks**:
- [ ] **reshape()**: Change shape while preserving data order
  - Implement -1 dimension inference
  - Verify contiguity requirements
  - Return view when possible, copy when necessary
- [ ] **squeeze()**: Remove size-1 dimensions
  - Support all dims: squeeze() or specific: squeeze(axis=1)
- [ ] **unsqueeze()** / **expand_dims()**: Add size-1 dimensions
- [ ] **flatten()**: Flatten to 1D (always copy)
- [ ] **ravel()**: Flatten to 1D (return view if possible)

**Validation**:
- Enable tests in test_shape.mojo
- Verify reshape compatibility checking
- Verify squeeze/unsqueeze dimension handling
- Verify flatten creates copy, ravel creates view when possible

#### 2.2 Concatenation and Stacking (Est: 2 days)

**Tasks**:
- [ ] **concatenate()**: Join tensors along existing axis
  - Support list of tensors
  - Verify dimension compatibility
- [ ] **stack()**: Join tensors along new axis
- [ ] **split()**: Split tensor into multiple tensors
  - Support equal splits: split(tensor, 3)
  - Support unequal splits: split(tensor, [3, 5, 10])
- [ ] **tile()**: Repeat tensor along dimensions
- [ ] **repeat()**: Repeat elements

**Validation**:
- Test concatenate: [(2,3), (4,3)] concat(axis=0) → (6,3)
- Test stack: [(2,3), (2,3)] stack(axis=0) → (2,2,3)
- Test split maintains total elements
- Test tile/repeat produce correct patterns

#### 2.3 Advanced Shape Operations (Est: 1-2 days)

**Tasks**:
- [ ] **permute()** / **transpose()**: Arbitrary axis permutation
  - Implement as view (stride manipulation)
- [ ] **broadcast_to()**: Explicit broadcasting to target shape
- [ ] **moveaxis()**: Move axis from one position to another
- [ ] **swapaxes()**: Swap two axes

**Validation**:
- Verify permute creates view correctly
- Verify broadcast_to compatibility checks
- Test axis manipulation preserves data correctly

### Priority 3: Advanced Math & Statistical (MEDIUM)

Useful for training and evaluation metrics.

#### 3.1 Statistical Functions (Est: 2-3 days)

**Tasks**:
- [ ] **var()**: Variance
- [ ] **std()**: Standard deviation
- [ ] **median()**: Median value
- [ ] **percentile()** / **quantile()**: Percentile values
- [ ] **cumsum()**: Cumulative sum
- [ ] **cumprod()**: Cumulative product
- [ ] **prod()**: Product reduction

**Validation**:
- Verify statistical correctness
- Test with known distributions
- Verify axis parameter works correctly

#### 3.2 Logical Operations (Est: 1-2 days)

**Tasks**:
- [ ] **logical_and()**, **logical_or()**, **logical_not()**: Boolean logic
- [ ] **logical_xor()**: XOR operation
- [ ] **all()**: All elements true
- [ ] **any()**: Any element true
- [ ] **where()**: Conditional selection (very important!)

**Validation**:
- Test with boolean tensors
- Verify short-circuit behavior (if applicable)
- Test where() with broadcasting

#### 3.3 Extended Math Functions (Est: 2 days)

**Tasks**:
- [ ] **ceil()**, **floor()**, **trunc()**, **round()**: Rounding
- [ ] **clip()** / **clamp()**: Constrain values to range
- [ ] **maximum()**, **minimum()**: Element-wise max/min (not reduction)
- [ ] **pow()**: Improve power() with exp/log for non-integer exponents
- [ ] **reciprocal()**: 1/x
- [ ] **square()**: x²
- [ ] **neg()**: -x (unary negation)

**Validation**:
- Test rounding modes
- Test clip boundary cases
- Verify maximum/minimum broadcasting

### Priority 4: Indexing & Slicing (MEDIUM-HIGH)

Critical for flexibility but can be deferred initially.

#### 4.1 Basic Indexing (Est: 3-4 days)

**Tasks**:
- [ ] **__getitem__()**: Read access via indexing
  - Support integer indexing: tensor[0]
  - Support slicing: tensor[1:5]
  - Support multi-dimensional: tensor[1:3, 2:4]
  - Support negative indices: tensor[-1]
  - Support stride: tensor[::2]
- [ ] **__setitem__()**: Write access via indexing
  - Support same patterns as __getitem__
  - Handle broadcasting of RHS
- [ ] Return views when possible, copies when necessary

**Validation**:
- Test all index patterns
- Verify views vs copies correctly
- Test negative indices
- Test out-of-bounds error handling

#### 4.2 Advanced Indexing (Est: 2-3 days)

**Tasks**:
- [ ] **take()**: Take elements along axis
- [ ] **gather()**: Gather elements by indices
- [ ] **scatter()**: Scatter elements by indices
- [ ] **masked_select()**: Select by boolean mask
- [ ] **nonzero()**: Find non-zero indices

**Validation**:
- Test gather/scatter inverse operations
- Test boolean masking with broadcasting
- Verify index validity checking

### Priority 5: Memory & Views (LOW-MEDIUM)

Important for efficiency but not blocking.

#### 5.1 Memory Operations (Est: 2 days)

**Tasks**:
- [ ] **copy()**: Explicit deep copy
- [ ] **clone()**: Alias for copy()
- [ ] **contiguous()**: Ensure contiguous memory
- [ ] **as_strided()**: Create view with custom strides
- [ ] Improve view detection and tracking

**Validation**:
- Verify copies are independent
- Test contiguous conversion
- Verify stride calculations

#### 5.2 Type Conversion (Est: 1-2 days)

**Tasks**:
- [ ] **astype()**: Convert dtype
- [ ] **to_float()**, **to_int()**: Convenience conversions
- [ ] **view()**: Reinterpret memory as different dtype

**Validation**:
- Test all dtype conversions
- Verify precision loss handling
- Test view size compatibility

### Priority 6: Specialized Operations (LOW)

Nice-to-have operations for specific use cases.

#### 6.1 Linear Algebra (Est: 3-4 days)

**Tasks**:
- [ ] **tensordot()**: Tensor contraction over multiple axes
- [ ] **einsum()**: Einstein summation notation (complex!)
- [ ] **diagonal()**: Extract diagonal
- [ ] **trace()**: Sum of diagonal elements
- [ ] **det()**: Determinant (if needed)
- [ ] **inv()**: Matrix inverse (if needed)

**Validation**:
- Test against NumPy reference
- Verify numerical stability

#### 6.2 Comparison & Sorting (Est: 2 days)

**Tasks**:
- [ ] **sort()**: Sort along axis
- [ ] **argsort()**: Return sorting indices
- [ ] **topk()**: Top-k elements
- [ ] **isnan()**, **isinf()**, **isfinite()**: Check special values

**Validation**:
- Test sorting stability
- Test topk with ties
- Test special value detection

#### 6.3 Set Operations (Est: 1-2 days)

**Tasks**:
- [ ] **unique()**: Find unique elements
- [ ] **union()**, **intersection()**: Set operations
- [ ] **setdiff()**: Set difference

**Validation**:
- Test uniqueness detection
- Test set operation correctness

## Implementation Guidelines

### Development Workflow

1. **For each operation**:
   - Read Array API spec: https://data-apis.org/array-api/latest/API_specification/
   - Write TDD tests first (already done for most)
   - Implement simple/correct version first
   - Optimize after correctness verified
   - Document with Array API compliance notes

2. **Testing Strategy**:
   - Unit tests: Each operation in isolation
   - Integration tests: Multiple operations together
   - Property tests: Invariants (e.g., a + (-a) = 0)
   - Edge cases: NaN, inf, empty tensors, broadcasting
   - Performance benchmarks: After correctness

3. **Code Organization**:
   - Group operations by module (arithmetic.mojo, reduction.mojo, etc.)
   - Export in __init__.mojo
   - Add dunder methods to ExTensor struct where applicable
   - Keep implementation and tests in sync

### Optimization Strategy

**Phase 1: Correctness** (Current)
- Implement naive algorithms
- Focus on passing tests
- Don't optimize prematurely

**Phase 2: SIMD Optimization** (After basic implementation)
- Use simdwidthof() for vectorization
- Implement blocked algorithms for cache efficiency
- Profile to find bottlenecks

**Phase 3: Advanced Optimization** (Future)
- Multi-threading for large tensors
- GPU acceleration (if Mojo supports)
- Custom kernels for critical operations

### Quality Checklist

For each operation, verify:
- [ ] Follows Array API Standard exactly
- [ ] Comprehensive tests (happy path + edge cases)
- [ ] Documentation with examples
- [ ] DType preservation correct
- [ ] Shape calculation correct
- [ ] Broadcasting support (if applicable)
- [ ] Memory management correct (no leaks)
- [ ] Error messages clear and helpful
- [ ] Performance acceptable for ML workloads

## Estimated Timeline

### Aggressive Schedule (8-10 weeks)
- Week 1-2: Priority 1.1-1.2 (Broadcasting + Matrix ops)
- Week 3-4: Priority 1.3-1.4 (Math functions + Reductions)
- Week 5-6: Priority 2.1-2.2 (Shape manipulation)
- Week 7-8: Priority 3.1-3.2 (Statistics + Logic)
- Week 9: Priority 4.1 (Basic indexing)
- Week 10: Priority 2.3, 3.3, 5.1-5.2 (Remaining core)

### Realistic Schedule (12-16 weeks)
- Add 50% buffer for debugging, optimization, and testing
- Account for learning Mojo stdlib and best practices
- Include time for refactoring and code reviews
- Allow for integration testing and bug fixes

## Success Metrics

### Minimum Viable Product (MVP)
**Goal**: Implement LeNet-5 from scratch
- ✅ Creation operations
- ✅ Arithmetic operations (with broadcasting)
- ✅ Comparison operations
- ✅ Reductions (sum, mean for loss)
- ✅ Matrix multiplication
- ✅ Reshape operations
- ✅ Element-wise math (exp, log for softmax)
- ✅ Basic indexing

### Complete Implementation
**Goal**: Match NumPy/PyTorch feature parity
- All Array API Standard 2023.12 operations
- Comprehensive test coverage (>90%)
- Performance within 2x of NumPy for CPU
- Documentation with examples for all operations

## References

- [Python Array API Standard 2023.12](https://data-apis.org/array-api/latest/)
- [NumPy API Reference](https://numpy.org/doc/stable/reference/)
- [PyTorch Tensor API](https://pytorch.org/docs/stable/tensors.html)
- [Mojo Standard Library](https://docs.modular.com/mojo/lib.html)

## Notes

- Current implementation status: **14% complete** (21/150 operations)
- Test coverage: **413+ tests created**, awaiting Mojo compiler for validation
- Focus: Prioritize operations needed for basic neural networks
- Strategy: Correctness first, optimization second
- Philosophy: Follow Array API Standard strictly for consistency with NumPy/PyTorch
