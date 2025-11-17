# Issue #219: [Test] ExTensors - Test-Driven Development

## Objective

Develop a comprehensive test suite for ExTensor following Test-Driven Development (TDD) principles. The test suite will verify all 150+ operations, ensure correct broadcasting behavior, validate edge cases, and benchmark performance of SIMD-optimized operations.

## Deliverables

### Test Files
- `test_creation.mojo` - Creation operations (zeros, ones, full, arange, etc.)
- `test_arithmetic.mojo` - Arithmetic operations with broadcasting
- `test_bitwise.mojo` - Bitwise operations for integer/bool tensors
- `test_comparison.mojo` - Comparison operations returning bool tensors
- `test_pointwise_math.mojo` - Pointwise math operations (trig, exp, log, etc.)
- `test_matrix.mojo` - Matrix operations (matmul, transpose, dot, etc.)
- `test_reduction.mojo` - Reduction operations (sum, mean, max, etc.)
- `test_shape.mojo` - Shape manipulation operations
- `test_indexing.mojo` - Indexing and slicing operations
- `test_utility.mojo` - Utility and inspection operations
- `test_broadcasting.mojo` - Broadcasting rules and edge cases
- `test_edge_cases.mojo` - Edge cases (empty tensors, scalars, overflow, NaN, inf)
- `test_dtype.mojo` - Data type support for all dtypes
- `test_memory.mojo` - Memory safety and ownership
- `benchmark_simd.mojo` - SIMD performance benchmarks

### Documentation
- Test specification document (this file)
- Test coverage report
- Performance benchmark results

## Test Organization

### Directory Structure

```
tests/
├── extensor/
│   ├── test_creation.mojo
│   ├── test_arithmetic.mojo
│   ├── test_bitwise.mojo
│   ├── test_comparison.mojo
│   ├── test_pointwise_math.mojo
│   ├── test_matrix.mojo
│   ├── test_reduction.mojo
│   ├── test_shape.mojo
│   ├── test_indexing.mojo
│   ├── test_utility.mojo
│   ├── test_broadcasting.mojo
│   ├── test_edge_cases.mojo
│   ├── test_dtype.mojo
│   ├── test_memory.mojo
│   └── benchmark_simd.mojo
├── helpers/
│   ├── assertions.mojo
│   ├── fixtures.mojo
│   └── utils.mojo
└── conftest.mojo
```

### Test Naming Convention

- Test functions: `test_<operation>_<scenario>`
- Example: `test_add_same_shape()`, `test_add_broadcast_scalar()`, `test_add_broadcast_matrix()`

## Test Categories

### 1. Creation Operations (test_creation.mojo)

**Operations to test**: zeros, ones, full, arange, from_array, eye, linspace, empty

**Test cases per operation**:
- ✅ Create with various shapes (0D, 1D, 2D, 3D, ND)
- ✅ Create with various dtypes (float16/32/64, int8/16/32/64, uint8/16/32/64, bool)
- ✅ Verify correct values
- ✅ Verify correct shape
- ✅ Verify correct dtype
- ✅ Edge cases (empty shape, very large shapes)

**Specific tests**:
```
test_zeros_1d_float32()
test_zeros_2d_int64()
test_zeros_empty_shape()
test_ones_3d_float64()
test_full_with_negative_value()
test_arange_step_fractional()
test_arange_reverse()
test_from_array_nested_list()
test_eye_rectangular()
test_linspace_inclusive()
test_empty_uninitialized()
```

### 2. Arithmetic Operations (test_arithmetic.mojo)

**Operations to test**: add, subtract, multiply, divide, floor_divide, modulo, power, matmul

**Dunder methods to test**: `__add__`, `__radd__`, `__iadd__` (and equivalents for all ops)

**Test cases per operation**:
- ✅ Same-shape tensors
- ✅ Broadcasting with scalar
- ✅ Broadcasting with different shapes
- ✅ Reflected operations (2 + tensor)
- ✅ In-place operations (tensor += 2)
- ✅ Edge cases (division by zero, overflow)
- ✅ Multiple dtypes

**Specific tests**:
```
test_add_same_shape()
test_add_broadcast_scalar()
test_add_broadcast_vector_to_matrix()
test_add_reflected()
test_add_in_place()
test_divide_by_zero_float()
test_divide_by_zero_int()
test_power_negative_exponent()
test_modulo_negative()
test_floor_divide_rounding()
```

### 3. Bitwise Operations (test_bitwise.mojo)

**Operations to test**: bitwise_and, bitwise_or, bitwise_xor, left_shift, right_shift

**Dunder methods to test**: `__and__`, `__rand__`, `__iand__` (and equivalents)

**Test cases**:
- ✅ Integer tensors (int8/16/32/64, uint8/16/32/64)
- ✅ Boolean tensors
- ✅ Broadcasting
- ✅ Reflected and in-place variants
- ✅ Shift overflow behavior

**Specific tests**:
```
test_bitwise_and_int32()
test_bitwise_or_bool()
test_bitwise_xor_broadcast()
test_left_shift_overflow()
test_right_shift_signed()
test_right_shift_unsigned()
```

### 4. Comparison Operations (test_comparison.mojo)

**Operations to test**: equal, not_equal, less, less_equal, greater, greater_equal

**Dunder methods to test**: `__eq__`, `__ne__`, `__lt__`, `__le__`, `__gt__`, `__ge__`

**Test cases**:
- ✅ Same-shape tensors
- ✅ Broadcasting
- ✅ All dtypes
- ✅ Special values (NaN, inf, -inf)
- ✅ Return type is bool tensor

**Specific tests**:
```
test_equal_same_values()
test_equal_different_values()
test_equal_nan_behavior()
test_less_broadcast()
test_greater_equal_inf()
test_comparison_returns_bool()
```

### 5. Pointwise Math Operations (test_pointwise_math.mojo)

**Categories**:
- Trigonometric: sin, cos, tan, asin, acos, atan, atan2
- Hyperbolic: sinh, cosh, tanh, asinh, acosh, atanh
- Exponential/Logarithmic: exp, exp2, expm1, log, log2, log10, log1p
- Power/Root: sqrt, cbrt, square, rsqrt
- Rounding: ceil, floor, trunc, round
- Other: sign, copysign, fma, clip, reciprocal

**Test cases per operation**:
- ✅ Various input ranges
- ✅ Edge values (0, 1, -1, very small, very large)
- ✅ Special values (NaN, inf, -inf)
- ✅ Numerical accuracy
- ✅ Multiple dtypes

**Specific tests**:
```
test_sin_range_negative_pi_to_pi()
test_cos_accuracy()
test_tan_discontinuities()
test_exp_overflow()
test_log_negative_input()
test_log_zero()
test_sqrt_negative()
test_rsqrt_accuracy()
test_clip_min_max()
test_fma_accuracy()
test_round_ties_to_even()
```

### 6. Matrix Operations (test_matrix.mojo)

**Operations to test**: matmul, transpose, dot, outer, inner, tensordot

**Test cases**:
- ✅ 2D matrix multiplication
- ✅ Batched matrix multiplication (3D+)
- ✅ Dimension compatibility checking
- ✅ Transpose with various axes permutations
- ✅ Dot product for 1D and 2D
- ✅ Outer product of vectors
- ✅ Tensor contraction

**Specific tests**:
```
test_matmul_2d()
test_matmul_batched()
test_matmul_incompatible_shapes()
test_transpose_2d()
test_transpose_3d_permute()
test_dot_1d()
test_dot_2d()
test_outer_vectors()
test_tensordot_contraction()
```

### 7. Reduction Operations (test_reduction.mojo)

**Operations to test**: sum, prod, mean, var, std, max, min, argmax, argmin, count_nonzero, cumulative_sum, cumulative_prod, all, any

**Test cases per operation**:
- ✅ Reduce all elements (axis=None)
- ✅ Reduce along specific axis
- ✅ Reduce along multiple axes
- ✅ keepdims=True vs keepdims=False
- ✅ Empty tensors
- ✅ Single element tensors

**Specific tests**:
```
test_sum_all_elements()
test_sum_along_axis_0()
test_sum_keepdims()
test_mean_accuracy()
test_var_ddof()
test_std_numerical_stability()
test_max_nan_behavior()
test_argmax_ties()
test_cumulative_sum_axis()
test_all_short_circuit()
test_any_empty_tensor()
```

### 8. Shape Manipulation (test_shape.mojo)

**Operations to test**: reshape, squeeze, unsqueeze, expand_dims, flatten, ravel, concatenate, stack, split, tile, repeat, broadcast_to, permute

**Test cases**:
- ✅ Valid reshapes
- ✅ Invalid reshapes (incompatible sizes)
- ✅ Zero-copy operations (ravel, reshape views)
- ✅ Concatenate along various axes
- ✅ Stack along new dimensions
- ✅ Split into equal/unequal parts

**Specific tests**:
```
test_reshape_valid()
test_reshape_invalid_size()
test_reshape_infer_dimension()
test_squeeze_all_dims()
test_squeeze_specific_dim()
test_flatten_order()
test_concatenate_axis_0()
test_stack_new_axis()
test_split_equal()
test_tile_multidim()
test_broadcast_to_compatible()
```

### 9. Indexing and Slicing (test_indexing.mojo)

**Operations to test**: `__getitem__`, `__setitem__`, take, take_along_axis, put, gather, scatter, where, masked_select

**Test cases**:
- ✅ Integer indexing
- ✅ Slice indexing
- ✅ Multi-dimensional indexing
- ✅ Boolean indexing
- ✅ Advanced indexing with arrays
- ✅ Assignment via indexing

**Specific tests**:
```
test_getitem_single_int()
test_getitem_slice()
test_getitem_multidim()
test_setitem_scalar()
test_setitem_slice()
test_take_along_axis()
test_gather_dim()
test_scatter_values()
test_where_condition()
test_masked_select()
```

### 10. Utility Operations (test_utility.mojo)

**Operations to test**: copy, clone, diff, `__len__`, `__bool__`, `__int__`, `__float__`, `__str__`, `__repr__`, `__hash__`, `__contains__`, `__divmod__`, item, tolist, numel, dim, size, stride, is_contiguous, contiguous

**Test cases**:
- ✅ Deep copy independence
- ✅ Type conversions
- ✅ String representations
- ✅ Dimension queries
- ✅ Stride calculations
- ✅ Contiguity checking

**Specific tests**:
```
test_copy_independence()
test_clone_identical()
test_item_single_element()
test_tolist_nested()
test_len_first_dim()
test_bool_single_element()
test_str_readable()
test_numel_total_elements()
test_stride_row_major()
test_is_contiguous_after_transpose()
```

### 11. Broadcasting Tests (test_broadcasting.mojo)

**Focus**: Comprehensive broadcasting rule validation

**Test cases**:
- ✅ Scalar to any shape
- ✅ Vector to matrix
- ✅ Missing dimensions
- ✅ Dimension size 1 broadcasting
- ✅ Incompatible shapes (should error)
- ✅ Complex multi-dimensional broadcasting

**Specific tests**:
```
test_broadcast_scalar_to_matrix()
test_broadcast_vector_to_matrix()
test_broadcast_missing_dims()
test_broadcast_size_one_dim()
test_broadcast_incompatible_shapes()
test_broadcast_3d_complex()
test_broadcast_output_shape()
```

### 12. Edge Cases (test_edge_cases.mojo)

**Test cases**:
- ✅ Empty tensors (0 elements)
- ✅ Scalar tensors (0D)
- ✅ Very large tensors
- ✅ Overflow behavior
- ✅ Underflow behavior
- ✅ NaN propagation
- ✅ Infinity handling
- ✅ Division by zero

**Specific tests**:
```
test_empty_tensor_operations()
test_scalar_tensor_0d()
test_overflow_float32()
test_underflow_float64()
test_nan_propagation()
test_inf_arithmetic()
test_divide_by_zero_ieee754()
test_large_dimension_count()
```

### 13. DType Tests (test_dtype.mojo)

**Test all operations with all dtypes**:
- Float: float16, float32, float64
- Integer: int8, int16, int32, int64
- Unsigned: uint8, uint16, uint32, uint64
- Boolean: bool

**Test cases**:
- ✅ Creation with each dtype
- ✅ Operations preserve dtype
- ✅ Type-specific behavior (e.g., integer division)
- ✅ No implicit type coercion

**Specific tests**:
```
test_dtype_float16()
test_dtype_float32()
test_dtype_float64()
test_dtype_int8()
test_dtype_uint8()
test_dtype_bool()
test_no_implicit_coercion()
test_division_int_vs_float()
```

### 14. Memory Safety (test_memory.mojo)

**Test cases**:
- ✅ Ownership transfers
- ✅ Borrowed references
- ✅ In-place mutations (inout)
- ✅ No memory leaks
- ✅ No use-after-free
- ✅ Views vs copies

**Specific tests**:
```
test_ownership_move()
test_borrow_no_mutation()
test_inout_mutation()
test_view_shares_memory()
test_copy_independent_memory()
test_no_dangling_references()
```

### 15. Performance Benchmarks (benchmark_simd.mojo)

**Benchmark categories**:
- ✅ Element-wise operations (add, mul, div) - measure SIMD speedup
- ✅ Reduction operations (sum, mean, max) - measure SIMD speedup
- ✅ Matrix multiplication - measure performance vs naive
- ✅ Broadcasting overhead
- ✅ Memory access patterns (contiguous vs strided)

**Benchmarks**:
```
benchmark_add_simd()
benchmark_multiply_simd()
benchmark_sum_reduction()
benchmark_matmul_sizes()
benchmark_broadcast_overhead()
benchmark_memory_access_pattern()
```

## Test Implementation Guidelines

### Test Structure

Each test should follow this structure:

```mojo
fn test_operation_scenario() raises:
    # Arrange
    let input1 = ExTensor.zeros((3, 4), DType.float32)
    let input2 = ExTensor.ones((3, 4), DType.float32)

    # Act
    let result = input1 + input2

    # Assert
    assert_equal(result.shape(), (3, 4))
    assert_dtype(result, DType.float32)
    assert_all_close(result, ExTensor.ones((3, 4), DType.float32))
```

### Assertion Helpers

Create assertion helpers in `tests/helpers/assertions.mojo`:

```mojo
fn assert_equal(a: ExTensor, b: ExTensor) raises
fn assert_all_close(a: ExTensor, b: ExTensor, rtol: Float64 = 1e-5, atol: Float64 = 1e-8) raises
fn assert_shape(tensor: ExTensor, expected_shape: Tuple) raises
fn assert_dtype(tensor: ExTensor, expected_dtype: DType) raises
fn assert_raises(fn: fn() raises -> None, error_type: String) raises
```

### Fixtures

Create common fixtures in `tests/helpers/fixtures.mojo`:

```mojo
fn random_tensor(shape: Tuple, dtype: DType) -> ExTensor
fn sequential_tensor(shape: Tuple, dtype: DType) -> ExTensor
fn nan_tensor(shape: Tuple) -> ExTensor
fn inf_tensor(shape: Tuple) -> ExTensor
```

## Success Criteria

- [ ] All 150+ operations have tests
- [ ] Test coverage >95%
- [ ] All tests pass
- [ ] Broadcasting rules verified with 20+ test cases
- [ ] Edge cases comprehensively tested
- [ ] All dtypes tested for all applicable operations
- [ ] Memory safety verified
- [ ] Performance benchmarks demonstrate SIMD speedup
- [ ] Clear test names and documentation
- [ ] Tests run in CI/CD pipeline

## Test Execution

### Running Tests

```bash
# Run all tests
mojo test tests/extensor/

# Run specific test file
mojo test tests/extensor/test_arithmetic.mojo

# Run with verbose output
mojo test tests/extensor/ -v

# Run benchmarks
mojo test tests/extensor/benchmark_simd.mojo
```

### Expected Test Counts

- Creation: ~50 tests
- Arithmetic: ~80 tests
- Bitwise: ~30 tests
- Comparison: ~30 tests
- Pointwise Math: ~100 tests
- Matrix: ~40 tests
- Reduction: ~70 tests
- Shape: ~60 tests
- Indexing: ~50 tests
- Utility: ~40 tests
- Broadcasting: ~25 tests
- Edge Cases: ~30 tests
- DType: ~50 tests
- Memory: ~20 tests
- **Total: ~675 tests**

## References

- [ExTensors Implementation Prompt](/home/user/ml-odyssey/notes/issues/218/extensor-implementation-prompt.md)
- [Array API Standard 2024 Test Suite](https://data-apis.org/array-api/latest/API_specification/index.html)
- [Mojo Testing Documentation](https://docs.modular.com/mojo/)

## Implementation Notes

### Test Development Session 1

- **Date**: 2025-11-17
- **Status**: Test specification created
- **Next**: Implement test files following TDD approach
- **Priority order**:
  1. Creation operations (foundation)
  2. Arithmetic operations (core functionality)
  3. Broadcasting (critical feature)
  4. Shape manipulation
  5. All remaining categories

---

**Status**: Test specification complete, ready for implementation

**Next Steps**:
1. Set up test infrastructure (assertions, fixtures, conftest)
2. Implement creation operation tests
3. Implement arithmetic operation tests
4. Continue with remaining test categories
5. Implement performance benchmarks
