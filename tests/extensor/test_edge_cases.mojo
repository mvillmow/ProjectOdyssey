"""Tests for ExTensor edge cases.

Tests edge cases including empty tensors, 0D scalars, very large tensors,
NaN handling, infinity handling, overflow, underflow, and numerical stability.
"""

from sys import DType
from math import isnan, isinf

# Import ExTensor and operations
from extensor import ExTensor, zeros, ones, full, arange, add, multiply

# Import test helpers
from ..helpers.assertions import (
    assert_dtype,
    assert_numel,
    assert_dim,
    assert_value_at,
    assert_all_values,
    assert_equal_int,
)


# ============================================================================
# Test empty tensors (0 elements)
# ============================================================================

fn test_empty_tensor_creation() raises:
    """Test creating empty tensor with 0 elements."""
    var shape = DynamicVector[Int](1)
    shape[0] = 0
    let t = zeros(shape, DType.float32)

    assert_numel(t, 0, "Empty tensor should have 0 elements")
    assert_dim(t, 1, "Empty tensor should have 1 dimension")


fn test_empty_tensor_operations() raises:
    """Test operations on empty tensors."""
    var shape = DynamicVector[Int](1)
    shape[0] = 0
    let a = zeros(shape, DType.float32)
    let b = zeros(shape, DType.float32)
    let c = add(a, b)

    assert_numel(c, 0, "Operations on empty tensors should give empty tensor")


fn test_empty_tensor_2d() raises:
    """Test 2D empty tensor (0 rows or 0 cols)."""
    var shape = DynamicVector[Int](2)
    shape[0] = 3
    shape[1] = 0  # 3x0 matrix
    let t = zeros(shape, DType.float32)

    assert_numel(t, 0, "3x0 matrix should have 0 elements")
    assert_dim(t, 2, "Should preserve 2D structure")


# ============================================================================
# Test 0D scalar tensors
# ============================================================================

fn test_scalar_tensor_creation() raises:
    """Test creating 0D scalar tensor."""
    var shape = DynamicVector[Int](0)
    let t = full(shape, 42.0, DType.float32)

    assert_numel(t, 1, "0D tensor should have 1 element")
    assert_dim(t, 0, "0D tensor should have 0 dimensions")
    assert_value_at(t, 0, 42.0, 1e-6, "Scalar value should be 42.0")


fn test_scalar_tensor_operations() raises:
    """Test operations on 0D scalar tensors."""
    var shape = DynamicVector[Int](0)
    let a = full(shape, 3.0, DType.float32)
    let b = full(shape, 2.0, DType.float32)
    let c = add(a, b)

    assert_numel(c, 1, "Scalar + scalar should give scalar")
    assert_dim(c, 0, "Result should be 0D")
    assert_value_at(c, 0, 5.0, 1e-6, "3 + 2 should be 5")


fn test_scalar_to_vector_broadcast() raises:
    """Test broadcasting scalar to vector."""
    var shape_scalar = DynamicVector[Int](0)
    var shape_vec = DynamicVector[Int](1)
    shape_vec[0] = 5

    let a = full(shape_scalar, 10.0, DType.float32)  # scalar
    let b = full(shape_vec, 2.0, DType.float32)  # vector
    let c = multiply(a, b)

    assert_numel(c, 5, "Result should have 5 elements")
    assert_all_values(c, 20.0, 1e-6, "10 * 2 broadcast to [20, 20, 20, 20, 20]")


# ============================================================================
# Test very large tensors
# ============================================================================

fn test_large_1d_tensor() raises:
    """Test creating and operating on large 1D tensor."""
    var shape = DynamicVector[Int](1)
    shape[0] = 10000000  # 10 million elements
    let t = zeros(shape, DType.float32)

    assert_numel(t, 10000000, "Large tensor should have 10M elements")
    # Spot check a few values
    assert_value_at(t, 0, 0.0, 1e-8, "First element should be 0")
    assert_value_at(t, 9999999, 0.0, 1e-8, "Last element should be 0")


fn test_large_dimension_count() raises:
    """Test tensor with many dimensions (10D)."""
    var shape = DynamicVector[Int](10)
    for i in range(10):
        shape[i] = 2
    let t = zeros(shape, DType.float32)

    assert_dim(t, 10, "Should have 10 dimensions")
    assert_numel(t, 1024, "2^10 = 1024 elements")


# ============================================================================
# Test NaN handling
# ============================================================================

fn test_nan_propagation_add() raises:
    """Test that NaN propagates through addition."""
    var shape = DynamicVector[Int](1)
    shape[0] = 3
    # let a = full(shape, Float32.nan, DType.float32)  # TODO: Create NaN tensor
    # let b = ones(shape, DType.float32)
    # let c = add(a, b)

    # NaN + x = NaN
    # for i in range(3):
    #     let val = c._get_float64(i)
    #     assert_true(isnan(val), "NaN should propagate through addition")
    pass  # Placeholder until NaN support is added


fn test_nan_propagation_multiply() raises:
    """Test that NaN propagates through multiplication."""
    var shape = DynamicVector[Int](1)
    shape[0] = 3
    # let a = full(shape, Float32.nan, DType.float32)
    # let b = full(shape, 0.0, DType.float32)
    # let c = multiply(a, b)

    # NaN * 0 = NaN (not 0!)
    # for i in range(3):
    #     let val = c._get_float64(i)
    #     assert_true(isnan(val), "NaN * 0 should be NaN")
    pass  # Placeholder


fn test_nan_equality() raises:
    """Test NaN equality (NaN != NaN per IEEE 754)."""
    var shape = DynamicVector[Int](1)
    shape[0] = 1
    # let a = full(shape, Float32.nan, DType.float32)
    # let b = full(shape, Float32.nan, DType.float32)
    # let c = equal(a, b)  # TODO: Implement equal()

    # IEEE 754: NaN != NaN
    # assert_value_at(c, 0, 0.0, 1e-8, "NaN != NaN should be False")
    pass  # Placeholder


# ============================================================================
# Test infinity handling
# ============================================================================

fn test_inf_arithmetic() raises:
    """Test arithmetic with infinity."""
    var shape = DynamicVector[Int](1)
    shape[0] = 3
    # let a = full(shape, Float32.infinity, DType.float32)
    # let b = full(shape, 1.0, DType.float32)
    # let c = add(a, b)

    # inf + 1 = inf
    # for i in range(3):
    #     let val = c._get_float64(i)
    #     assert_true(isinf(val), "inf + 1 should be inf")
    pass  # Placeholder


fn test_inf_multiplication() raises:
    """Test infinity multiplication."""
    var shape = DynamicVector[Int](1)
    shape[0] = 3
    # let a = full(shape, Float32.infinity, DType.float32)
    # let b = full(shape, 2.0, DType.float32)
    # let c = multiply(a, b)

    # inf * 2 = inf
    # for i in range(3):
    #     let val = c._get_float64(i)
    #     assert_true(isinf(val), "inf * 2 should be inf")
    pass  # Placeholder


fn test_inf_times_zero() raises:
    """Test infinity times zero (should give NaN)."""
    var shape = DynamicVector[Int](1)
    shape[0] = 3
    # let a = full(shape, Float32.infinity, DType.float32)
    # let b = zeros(shape, DType.float32)
    # let c = multiply(a, b)

    # inf * 0 = NaN (indeterminate form)
    # for i in range(3):
    #     let val = c._get_float64(i)
    #     assert_true(isnan(val), "inf * 0 should be NaN")
    pass  # Placeholder


fn test_negative_inf() raises:
    """Test negative infinity."""
    var shape = DynamicVector[Int](1)
    shape[0] = 3
    # let a = full(shape, -Float32.infinity, DType.float32)
    # let b = full(shape, 1.0, DType.float32)
    # let c = add(a, b)

    # -inf + 1 = -inf
    # for i in range(3):
    #     let val = c._get_float64(i)
    #     assert_true(isinf(val) and val < 0, "-inf + 1 should be -inf")
    pass  # Placeholder


fn test_inf_comparison() raises:
    """Test comparison with infinity."""
    var shape = DynamicVector[Int](1)
    shape[0] = 3
    # let a = full(shape, 1000000.0, DType.float32)
    # let b = full(shape, Float32.infinity, DType.float32)
    # let c = less(a, b)  # TODO: Implement less()

    # 1e6 < inf should be True
    # assert_all_values(c, 1.0, 1e-8, "Finite < inf should be True")
    pass  # Placeholder


# ============================================================================
# Test overflow behavior
# ============================================================================

fn test_overflow_float32() raises:
    """Test overflow behavior for float32."""
    var shape = DynamicVector[Int](1)
    shape[0] = 2
    # Create values close to float32 max
    # let a = full(shape, 1e38, DType.float32)
    # let b = full(shape, 1e38, DType.float32)
    # let c = add(a, b)

    # Result should be infinity
    # for i in range(2):
    #     let val = c._get_float64(i)
    #     assert_true(isinf(val), "Overflow should give infinity")
    pass  # Placeholder


fn test_overflow_int32() raises:
    """Test overflow behavior for int32."""
    var shape = DynamicVector[Int](1)
    shape[0] = 2
    # Create values close to int32 max
    # let a = full(shape, 2147483647.0, DType.int32)  # INT32_MAX
    # let b = ones(shape, DType.int32)
    # let c = add(a, b)

    # Integer overflow behavior: wraps around or saturates depending on implementation
    # TODO: Document expected behavior
    pass  # Placeholder


# ============================================================================
# Test underflow behavior
# ============================================================================

fn test_underflow_float64() raises:
    """Test underflow behavior for float64."""
    var shape = DynamicVector[Int](1)
    shape[0] = 2
    # Create very small values
    # let a = full(shape, 1e-300, DType.float64)
    # let b = full(shape, 1e-100, DType.float64)
    # let c = multiply(a, b)

    # Result may underflow to 0 (gradual underflow)
    # assert_all_values(c, 0.0, 1e-320, "Underflow should give 0")
    pass  # Placeholder


# ============================================================================
# Test division by zero
# ============================================================================

fn test_divide_by_zero_float() raises:
    """Test division by zero for floating point."""
    var shape = DynamicVector[Int](1)
    shape[0] = 3
    let a = full(shape, 1.0, DType.float32)
    let b = zeros(shape, DType.float32)
    # let c = divide(a, b)  # TODO: Implement divide()

    # IEEE 754: 1/0 = inf, -1/0 = -inf, 0/0 = NaN
    # for i in range(3):
    #     let val = c._get_float64(i)
    #     assert_true(isinf(val), "1/0 should be inf")
    pass  # Placeholder


fn test_divide_by_zero_int() raises:
    """Test division by zero for integers."""
    var shape = DynamicVector[Int](1)
    shape[0] = 3
    let a = full(shape, 10.0, DType.int32)
    let b = zeros(shape, DType.int32)
    # let c = divide(a, b)  # TODO: Implement divide()

    # Integer division by zero: undefined behavior (should error or saturate)
    # TODO: Document expected behavior and test
    pass  # Placeholder


fn test_divide_zero_by_zero() raises:
    """Test 0/0 (should give NaN for floats)."""
    var shape = DynamicVector[Int](1)
    shape[0] = 3
    let a = zeros(shape, DType.float32)
    let b = zeros(shape, DType.float32)
    # let c = divide(a, b)  # TODO: Implement divide()

    # 0/0 = NaN (indeterminate form)
    # for i in range(3):
    #     let val = c._get_float64(i)
    #     assert_true(isnan(val), "0/0 should be NaN")
    pass  # Placeholder


# ============================================================================
# Test subnormal numbers
# ============================================================================

fn test_subnormal_numbers() raises:
    """Test handling of subnormal (denormalized) numbers."""
    var shape = DynamicVector[Int](1)
    shape[0] = 2
    # Create subnormal float32 value (~1e-40)
    # let a = full(shape, 1e-40, DType.float32)
    # let b = full(shape, 1.0, DType.float32)
    # let c = add(a, b)

    # a + 1 should be approximately 1 (subnormal is tiny)
    # assert_all_values(c, 1.0, 1e-6, "1e-40 + 1 ≈ 1")
    pass  # Placeholder


# ============================================================================
# Test numerical stability
# ============================================================================

fn test_catastrophic_cancellation() raises:
    """Test catastrophic cancellation in subtraction."""
    var shape = DynamicVector[Int](1)
    shape[0] = 2
    # Create two very close values
    # let a = full(shape, 1.0000000001, DType.float64)
    # let b = full(shape, 1.0, DType.float64)
    # let c = subtract(a, b)

    # Result should be approximately 1e-10 but may lose precision
    # TODO: Test precision loss
    pass  # Placeholder


fn test_associativity_loss() raises:
    """Test loss of associativity in floating point."""
    var shape = DynamicVector[Int](1)
    shape[0] = 3
    # Create specific values to demonstrate associativity loss
    # let a = full(shape, 1e20, DType.float32)
    # let b = full(shape, 1.0, DType.float32)
    # let c = full(shape, -1e20, DType.float32)

    # (a + b) + c != a + (b + c) in floating point
    # let result1 = add(add(a, b), c)
    # let result2 = add(a, add(b, c))

    # Results may differ due to rounding
    pass  # Placeholder


# ============================================================================
# Test special dtype behaviors
# ============================================================================

fn test_bool_dtype_operations() raises:
    """Test operations on bool dtype tensors."""
    var shape = DynamicVector[Int](1)
    shape[0] = 5
    let a = ones(shape, DType.bool)  # All True
    let b = zeros(shape, DType.bool)  # All False

    # TODO: Test bool-specific operations (and, or, xor, not)
    # let c = bitwise_and(a, b)  # Should be all False
    # assert_all_values(c, 0.0, 1e-8, "True AND False should be False")
    pass  # Placeholder


fn test_int8_range() raises:
    """Test int8 range limits."""
    var shape = DynamicVector[Int](1)
    shape[0] = 2
    # Create values at int8 boundaries
    # let a = full(shape, 127.0, DType.int8)  # INT8_MAX
    # let b = full(shape, -128.0, DType.int8)  # INT8_MIN

    # Verify values are stored correctly
    # assert_value_at(a, 0, 127.0, 1e-6, "INT8_MAX should be 127")
    # assert_value_at(b, 0, -128.0, 1e-6, "INT8_MIN should be -128")
    pass  # Placeholder


fn test_uint8_range() raises:
    """Test uint8 range limits."""
    var shape = DynamicVector[Int](1)
    shape[0] = 2
    # Create values at uint8 boundaries
    # let a = full(shape, 255.0, DType.uint8)  # UINT8_MAX
    # let b = zeros(shape, DType.uint8)  # UINT8_MIN

    # Verify values are stored correctly
    # assert_value_at(a, 0, 255.0, 1e-6, "UINT8_MAX should be 255")
    # assert_value_at(b, 0, 0.0, 1e-6, "UINT8_MIN should be 0")
    pass  # Placeholder


# ============================================================================
# Main test runner
# ============================================================================

fn main() raises:
    """Run all edge case tests."""
    print("Running ExTensor edge case tests...")

    # Empty tensors
    print("  Testing empty tensors...")
    test_empty_tensor_creation()
    test_empty_tensor_operations()
    test_empty_tensor_2d()

    # 0D scalar tensors
    print("  Testing 0D scalar tensors...")
    test_scalar_tensor_creation()
    test_scalar_tensor_operations()
    test_scalar_to_vector_broadcast()

    # Very large tensors
    print("  Testing very large tensors...")
    test_large_1d_tensor()
    test_large_dimension_count()

    # NaN handling
    print("  Testing NaN handling...")
    test_nan_propagation_add()
    test_nan_propagation_multiply()
    test_nan_equality()

    # Infinity handling
    print("  Testing infinity handling...")
    test_inf_arithmetic()
    test_inf_multiplication()
    test_inf_times_zero()
    test_negative_inf()
    test_inf_comparison()

    # Overflow
    print("  Testing overflow...")
    test_overflow_float32()
    test_overflow_int32()

    # Underflow
    print("  Testing underflow...")
    test_underflow_float64()

    # Division by zero
    print("  Testing division by zero...")
    test_divide_by_zero_float()
    test_divide_by_zero_int()
    test_divide_zero_by_zero()

    # Subnormal numbers
    print("  Testing subnormal numbers...")
    test_subnormal_numbers()

    # Numerical stability
    print("  Testing numerical stability...")
    test_catastrophic_cancellation()
    test_associativity_loss()

    # Special dtype behaviors
    print("  Testing special dtype behaviors...")
    test_bool_dtype_operations()
    test_int8_range()
    test_uint8_range()

    print("All edge case tests completed!")
