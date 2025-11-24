"""Tests for ExTensor edge cases.

Tests edge cases including empty tensors, 0D scalars, very large tensors,
NaN handling, infinity handling, overflow, underflow, and numerical stability.
"""

from math import isnan, isinf

# Import ExTensor and operations
from shared.core import (
    ExTensor, zeros, ones, full, arange,
    add, subtract, multiply, divide, floor_divide, modulo, power,
    equal, not_equal, less, less_equal, greater, greater_equal
)

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
    var shape = List[Int]()
    shape.append(0)
    vart = zeros(shape, DType.float32)

    assert_numel(t, 0, "Empty tensor should have 0 elements")
    assert_dim(t, 1, "Empty tensor should have 1 dimension")


fn test_empty_tensor_operations() raises:
    """Test operations on empty tensors."""
    var shape = List[Int]()
    shape.append(0)
    vara = zeros(shape, DType.float32)
    varb = zeros(shape, DType.float32)
    varc = add(a, b)

    assert_numel(c, 0, "Operations on empty tensors should give empty tensor")


fn test_empty_tensor_2d() raises:
    """Test 2D empty tensor (0 rows or 0 cols)."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(0  # 3x0 matrix)
    vart = zeros(shape, DType.float32)

    assert_numel(t, 0, "3x0 matrix should have 0 elements")
    assert_dim(t, 2, "Should preserve 2D structure")


# ============================================================================
# Test 0D scalar tensors
# ============================================================================

fn test_scalar_tensor_creation() raises:
    """Test creating 0D scalar tensor."""
    var shape = List[Int]()
    vart = full(shape, 42.0, DType.float32)

    assert_numel(t, 1, "0D tensor should have 1 element")
    assert_dim(t, 0, "0D tensor should have 0 dimensions")
    assert_value_at(t, 0, 42.0, 1e-6, "Scalar value should be 42.0")


fn test_scalar_tensor_operations() raises:
    """Test operations on 0D scalar tensors."""
    var shape = List[Int]()
    vara = full(shape, 3.0, DType.float32)
    varb = full(shape, 2.0, DType.float32)
    varc = add(a, b)

    assert_numel(c, 1, "Scalar + scalar should give scalar")
    assert_dim(c, 0, "Result should be 0D")
    assert_value_at(c, 0, 5.0, 1e-6, "3 + 2 should be 5")


fn test_scalar_to_vector_broadcast() raises:
    """Test broadcasting scalar to vector."""
    var shape_scalar = List[Int]()
    var shape_vec = List[Int]()
    shape_vec.append(5)

    vara = full(shape_scalar, 10.0, DType.float32)  # scalar
    varb = full(shape_vec, 2.0, DType.float32)  # vector
    varc = multiply(a, b)

    assert_numel(c, 5, "Result should have 5 elements")
    assert_all_values(c, 20.0, 1e-6, "10 * 2 broadcast to [20, 20, 20, 20, 20]")


# ============================================================================
# Test very large tensors
# ============================================================================

fn test_large_1d_tensor() raises:
    """Test creating and operating on large 1D tensor."""
    var shape = List[Int]()
    shape.append(10000000  # 10 million elements)
    vart = zeros(shape, DType.float32)

    assert_numel(t, 10000000, "Large tensor should have 10M elements")
    # Spot check a few values
    assert_value_at(t, 0, 0.0, 1e-8, "First element should be 0")
    assert_value_at(t, 9999999, 0.0, 1e-8, "Last element should be 0")


fn test_large_dimension_count() raises:
    """Test tensor with many dimensions (10D)."""
    var shape = List[Int]()
    for i in range(10):
        shape[i] = 2
    vart = zeros(shape, DType.float32)

    assert_dim(t, 10, "Should have 10 dimensions")
    assert_numel(t, 1024, "2^10 = 1024 elements")


# ============================================================================
# Test NaN handling
# ============================================================================

fn test_nan_propagation_add() raises:
    """Test that NaN propagates through addition."""
    var shape = List[Int]()
    shape.append(3)
    # vara = full(shape, Float32.nan, DType.float32)  # TODO: Create NaN tensor
    # varb = ones(shape, DType.float32)
    # varc = add(a, b)

    # NaN + x = NaN
    # for i in range(3):
    #     varval = c._get_float64(i)
    #     assert_true(isnan(val), "NaN should propagate through addition")
    pass  # Placeholder until NaN support is added


fn test_nan_propagation_multiply() raises:
    """Test that NaN propagates through multiplication."""
    var shape = List[Int]()
    shape.append(3)
    # vara = full(shape, Float32.nan, DType.float32)
    # varb = full(shape, 0.0, DType.float32)
    # varc = multiply(a, b)

    # NaN * 0 = NaN (not 0!)
    # for i in range(3):
    #     varval = c._get_float64(i)
    #     assert_true(isnan(val), "NaN * 0 should be NaN")
    pass  # Placeholder


fn test_nan_equality() raises:
    """Test NaN equality (NaN != NaN per IEEE 754)."""
    var shape = List[Int]()
    shape.append(1)
    # vara = full(shape, Float32.nan, DType.float32)
    # varb = full(shape, Float32.nan, DType.float32)
    # varc = equal(a, b)  # TODO: Implement equal()

    # IEEE 754: NaN != NaN
    # assert_value_at(c, 0, 0.0, 1e-8, "NaN != NaN should be False")
    pass  # Placeholder


# ============================================================================
# Test infinity handling
# ============================================================================

fn test_inf_arithmetic() raises:
    """Test arithmetic with infinity."""
    var shape = List[Int]()
    shape.append(3)
    # vara = full(shape, Float32.infinity, DType.float32)
    # varb = full(shape, 1.0, DType.float32)
    # varc = add(a, b)

    # inf + 1 = inf
    # for i in range(3):
    #     varval = c._get_float64(i)
    #     assert_true(isinf(val), "inf + 1 should be inf")
    pass  # Placeholder


fn test_inf_multiplication() raises:
    """Test infinity multiplication."""
    var shape = List[Int]()
    shape.append(3)
    # vara = full(shape, Float32.infinity, DType.float32)
    # varb = full(shape, 2.0, DType.float32)
    # varc = multiply(a, b)

    # inf * 2 = inf
    # for i in range(3):
    #     varval = c._get_float64(i)
    #     assert_true(isinf(val), "inf * 2 should be inf")
    pass  # Placeholder


fn test_inf_times_zero() raises:
    """Test infinity times zero (should give NaN)."""
    var shape = List[Int]()
    shape.append(3)
    # vara = full(shape, Float32.infinity, DType.float32)
    # varb = zeros(shape, DType.float32)
    # varc = multiply(a, b)

    # inf * 0 = NaN (indeterminate form)
    # for i in range(3):
    #     varval = c._get_float64(i)
    #     assert_true(isnan(val), "inf * 0 should be NaN")
    pass  # Placeholder


fn test_negative_inf() raises:
    """Test negative infinity."""
    var shape = List[Int]()
    shape.append(3)
    # vara = full(shape, -Float32.infinity, DType.float32)
    # varb = full(shape, 1.0, DType.float32)
    # varc = add(a, b)

    # -inf + 1 = -inf
    # for i in range(3):
    #     varval = c._get_float64(i)
    #     assert_true(isinf(val) and val < 0, "-inf + 1 should be -inf")
    pass  # Placeholder


fn test_inf_comparison() raises:
    """Test comparison with infinity."""
    var shape = List[Int]()
    shape.append(3)
    # vara = full(shape, 1000000.0, DType.float32)
    # varb = full(shape, Float32.infinity, DType.float32)
    # varc = less(a, b)  # TODO: Implement less()

    # 1e6 < inf should be True
    # assert_all_values(c, 1.0, 1e-8, "Finite < inf should be True")
    pass  # Placeholder


# ============================================================================
# Test overflow behavior
# ============================================================================

fn test_overflow_float32() raises:
    """Test overflow behavior for float32."""
    var shape = List[Int]()
    shape.append(2)
    # Create values close to float32 max
    # vara = full(shape, 1e38, DType.float32)
    # varb = full(shape, 1e38, DType.float32)
    # varc = add(a, b)

    # Result should be infinity
    # for i in range(2):
    #     varval = c._get_float64(i)
    #     assert_true(isinf(val), "Overflow should give infinity")
    pass  # Placeholder


fn test_overflow_int32() raises:
    """Test overflow behavior for int32."""
    var shape = List[Int]()
    shape.append(2)
    # Create values close to int32 max
    # vara = full(shape, 2147483647.0, DType.int32)  # INT32_MAX
    # varb = ones(shape, DType.int32)
    # varc = add(a, b)

    # Integer overflow behavior: wraps around or saturates depending on implementation
    # TODO: Document expected behavior
    pass  # Placeholder


# ============================================================================
# Test underflow behavior
# ============================================================================

fn test_underflow_float64() raises:
    """Test underflow behavior for float64."""
    var shape = List[Int]()
    shape.append(2)
    # Create very small values
    # vara = full(shape, 1e-300, DType.float64)
    # varb = full(shape, 1e-100, DType.float64)
    # varc = multiply(a, b)

    # Result may underflow to 0 (gradual underflow)
    # assert_all_values(c, 0.0, 1e-320, "Underflow should give 0")
    pass  # Placeholder


# ============================================================================
# Test division by zero
# ============================================================================

fn test_divide_by_zero_float() raises:
    """Test division by zero for floating point."""
    var shape = List[Int]()
    shape.append(3)
    vara = full(shape, 1.0, DType.float32)
    varb = zeros(shape, DType.float32)
    varc = divide(a, b)

    # IEEE 754: 1/0 = inf
    for i in range(3):
        varval = c._get_float64(i)
        if not isinf(val):
            raise Error("1/0 should be inf per IEEE 754")
        if val < 0:
            raise Error("1/0 should be positive infinity")


fn test_divide_by_zero_int() raises:
    """Test division by zero for integers."""
    var shape = List[Int]()
    shape.append(3)
    vara = full(shape, 10.0, DType.int32)
    varb = zeros(shape, DType.int32)
    # varc = divide(a, b)  # TODO: Implement divide()

    # Integer division by zero: undefined behavior (should error or saturate)
    # TODO: Document expected behavior and test
    pass  # Placeholder


fn test_divide_zero_by_zero() raises:
    """Test 0/0 (should give NaN for floats)."""
    var shape = List[Int]()
    shape.append(3)
    vara = zeros(shape, DType.float32)
    varb = zeros(shape, DType.float32)
    varc = divide(a, b)

    # 0/0 = NaN (indeterminate form per IEEE 754)
    for i in range(3):
        varval = c._get_float64(i)
        if not isnan(val):
            raise Error("0/0 should be NaN per IEEE 754")


fn test_divide_negative_by_zero() raises:
    """Test -1/0 (should give -inf for floats)."""
    var shape = List[Int]()
    shape.append(3)
    vara = full(shape, -1.0, DType.float32)
    varb = zeros(shape, DType.float32)
    varc = divide(a, b)

    # IEEE 754: -1/0 = -inf
    for i in range(3):
        varval = c._get_float64(i)
        if not isinf(val):
            raise Error("-1/0 should be -inf per IEEE 754")
        if val > 0:
            raise Error("-1/0 should be negative infinity")


# ============================================================================
# Test modulo edge cases
# ============================================================================

fn test_modulo_by_zero() raises:
    """Test modulo by zero (should give NaN for floats)."""
    var shape = List[Int]()
    shape.append(3)
    vara = full(shape, 5.0, DType.float32)
    varb = zeros(shape, DType.float32)
    varc = modulo(a, b)

    # Modulo by zero: undefined, should give NaN
    for i in range(3):
        varval = c._get_float64(i)
        if not isnan(val):
            raise Error("x % 0 should be NaN")


fn test_modulo_with_negative_divisor() raises:
    """Test modulo with negative divisor."""
    var shape = List[Int]()
    shape.append(1)
    vara = full(shape, 7.0, DType.float32)
    varb = full(shape, -3.0, DType.float32)
    varc = modulo(a, b)

    # Python semantics: 7 % -3 = -2 (result has sign of divisor)
    assert_value_at(c, 0, -2.0, 1e-6, "7 % -3 should be -2 (Python semantics)")


fn test_modulo_both_negative() raises:
    """Test modulo with both negative values."""
    var shape = List[Int]()
    shape.append(1)
    vara = full(shape, -7.0, DType.float32)
    varb = full(shape, -3.0, DType.float32)
    varc = modulo(a, b)

    # Python semantics: -7 % -3 = -1
    assert_value_at(c, 0, -1.0, 1e-6, "-7 % -3 should be -1 (Python semantics)")


# ============================================================================
# Test power edge cases
# ============================================================================

fn test_power_zero_to_zero() raises:
    """Test 0^0 (mathematically undefined, conventionally 1)."""
    var shape = List[Int]()
    shape.append(3)
    vara = zeros(shape, DType.float32)
    varb = zeros(shape, DType.float32)
    varc = power(a, b)

    # Convention: 0^0 = 1 (used in polynomial evaluation)
    assert_all_values(c, 1.0, 1e-6, "0^0 should be 1 by convention")


fn test_power_negative_base_even() raises:
    """Test negative base with even exponent."""
    var shape = List[Int]()
    shape.append(1)
    vara = full(shape, -2.0, DType.float32)
    varb = full(shape, 2.0, DType.float32)
    varc = power(a, b)

    # (-2)^2 = 4
    assert_value_at(c, 0, 4.0, 1e-6, "(-2)^2 should be 4")


fn test_power_negative_base_odd() raises:
    """Test negative base with odd exponent."""
    var shape = List[Int]()
    shape.append(1)
    vara = full(shape, -2.0, DType.float32)
    varb = full(shape, 3.0, DType.float32)
    varc = power(a, b)

    # (-2)^3 = -8
    assert_value_at(c, 0, -8.0, 1e-6, "(-2)^3 should be -8")


fn test_power_zero_base_positive_exp() raises:
    """Test 0^n for positive n."""
    var shape = List[Int]()
    shape.append(3)
    vara = zeros(shape, DType.float32)
    varb = full(shape, 5.0, DType.float32)
    varc = power(a, b)

    # 0^5 = 0
    assert_all_values(c, 0.0, 1e-6, "0^n should be 0 for positive n")


# ============================================================================
# Test floor_divide edge cases
# ============================================================================

fn test_floor_divide_by_zero() raises:
    """Test floor division by zero."""
    var shape = List[Int]()
    shape.append(3)
    vara = full(shape, 10.0, DType.float32)
    varb = zeros(shape, DType.float32)
    varc = floor_divide(a, b)

    # Floor division by zero should give inf (like regular division)
    for i in range(3):
        varval = c._get_float64(i)
        if not isinf(val):
            raise Error("x // 0 should be inf")


fn test_floor_divide_with_remainder() raises:
    """Test floor division with remainder."""
    var shape = List[Int]()
    shape.append(1)
    vara = full(shape, 7.0, DType.float32)
    varb = full(shape, 3.0, DType.float32)
    varc = floor_divide(a, b)

    # 7 // 3 = 2 (floor of 2.333...)
    assert_value_at(c, 0, 2.0, 1e-6, "7 // 3 should be 2")


fn test_floor_divide_negative_result() raises:
    """Test floor division with negative result."""
    var shape = List[Int]()
    shape.append(1)
    vara = full(shape, -7.0, DType.float32)
    varb = full(shape, 3.0, DType.float32)
    varc = floor_divide(a, b)

    # -7 // 3 = -3 (floor of -2.333... = -3, not -2)
    assert_value_at(c, 0, -3.0, 1e-6, "-7 // 3 should be -3 (floor toward -inf)")


# ============================================================================
# Test comparison edge cases
# ============================================================================

fn test_comparison_with_zero() raises:
    """Test comparison operations with zero."""
    var shape = List[Int]()
    shape.append(3)

    varpositive = full(shape, 1.0, DType.float32)
    varnegative = full(shape, -1.0, DType.float32)
    varzero = zeros(shape, DType.float32)

    # Test greater than zero
    varpos_gt_zero = greater(positive, zero)
    assert_all_values(pos_gt_zero, 1.0, 1e-6, "1.0 > 0 should be True")

    varneg_gt_zero = greater(negative, zero)
    assert_all_values(neg_gt_zero, 0.0, 1e-6, "-1.0 > 0 should be False")

    # Test less than zero
    varneg_lt_zero = less(negative, zero)
    assert_all_values(neg_lt_zero, 1.0, 1e-6, "-1.0 < 0 should be True")


fn test_comparison_equal_values() raises:
    """Test equality comparison with same values."""
    var shape = List[Int]()
    shape.append(5)
    vara = full(shape, 3.14159, DType.float64)
    varb = full(shape, 3.14159, DType.float64)
    varc = equal(a, b)

    assert_dtype(c, DType.bool, "Comparison should return bool")
    assert_all_values(c, 1.0, 1e-10, "Equal values should be equal")


fn test_comparison_very_close_values() raises:
    """Test comparison with very close but not equal values."""
    var shape = List[Int]()
    shape.append(1)
    vara = full(shape, 1.0, DType.float32)
    varb = full(shape, 1.0000001, DType.float32)

    # These should NOT be equal (exact comparison, no tolerance)
    vareq = equal(a, b)
    varne = not_equal(a, b)

    # Depending on float32 precision, these might be equal or not
    # For now, just verify bool dtype
    assert_dtype(eq, DType.bool, "equal() should return bool")
    assert_dtype(ne, DType.bool, "not_equal() should return bool")


# ============================================================================
# Test subnormal numbers
# ============================================================================

fn test_subnormal_numbers() raises:
    """Test handling of subnormal (denormalized) numbers."""
    var shape = List[Int]()
    shape.append(2)
    # Create subnormal float32 value (~1e-40)
    # vara = full(shape, 1e-40, DType.float32)
    # varb = full(shape, 1.0, DType.float32)
    # varc = add(a, b)

    # a + 1 should be approximately 1 (subnormal is tiny)
    # assert_all_values(c, 1.0, 1e-6, "1e-40 + 1 ≈ 1")
    pass  # Placeholder


# ============================================================================
# Test numerical stability
# ============================================================================

fn test_catastrophic_cancellation() raises:
    """Test catastrophic cancellation in subtraction."""
    var shape = List[Int]()
    shape.append(2)
    # Create two very close values
    # vara = full(shape, 1.0000000001, DType.float64)
    # varb = full(shape, 1.0, DType.float64)
    # varc = subtract(a, b)

    # Result should be approximately 1e-10 but may lose precision
    # TODO: Test precision loss
    pass  # Placeholder


fn test_associativity_loss() raises:
    """Test loss of associativity in floating point."""
    var shape = List[Int]()
    shape.append(3)
    # Create specific values to demonstrate associativity loss
    # vara = full(shape, 1e20, DType.float32)
    # varb = full(shape, 1.0, DType.float32)
    # varc = full(shape, -1e20, DType.float32)

    # (a + b) + c != a + (b + c) in floating point
    # varresult1 = add(add(a, b), c)
    # varresult2 = add(a, add(b, c))

    # Results may differ due to rounding
    pass  # Placeholder


# ============================================================================
# Test special dtype behaviors
# ============================================================================

fn test_bool_dtype_operations() raises:
    """Test operations on bool dtype tensors."""
    var shape = List[Int]()
    shape.append(5)
    vara = ones(shape, DType.bool)  # All True
    varb = zeros(shape, DType.bool)  # All False

    # TODO: Test bool-specific operations (and, or, xor, not)
    # varc = bitwise_and(a, b)  # Should be all False
    # assert_all_values(c, 0.0, 1e-8, "True AND False should be False")
    pass  # Placeholder


fn test_int8_range() raises:
    """Test int8 range limits."""
    var shape = List[Int]()
    shape.append(2)
    # Create values at int8 boundaries
    # vara = full(shape, 127.0, DType.int8)  # INT8_MAX
    # varb = full(shape, -128.0, DType.int8)  # INT8_MIN

    # Verify values are stored correctly
    # assert_value_at(a, 0, 127.0, 1e-6, "INT8_MAX should be 127")
    # assert_value_at(b, 0, -128.0, 1e-6, "INT8_MIN should be -128")
    pass  # Placeholder


fn test_uint8_range() raises:
    """Test uint8 range limits."""
    var shape = List[Int]()
    shape.append(2)
    # Create values at uint8 boundaries
    # vara = full(shape, 255.0, DType.uint8)  # UINT8_MAX
    # varb = zeros(shape, DType.uint8)  # UINT8_MIN

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
    test_divide_negative_by_zero()

    # Modulo edge cases
    print("  Testing modulo edge cases...")
    test_modulo_by_zero()
    test_modulo_with_negative_divisor()
    test_modulo_both_negative()

    # Power edge cases
    print("  Testing power edge cases...")
    test_power_zero_to_zero()
    test_power_negative_base_even()
    test_power_negative_base_odd()
    test_power_zero_base_positive_exp()

    # Floor divide edge cases
    print("  Testing floor_divide edge cases...")
    test_floor_divide_by_zero()
    test_floor_divide_with_remainder()
    test_floor_divide_negative_result()

    # Comparison edge cases
    print("  Testing comparison edge cases...")
    test_comparison_with_zero()
    test_comparison_equal_values()
    test_comparison_very_close_values()

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
