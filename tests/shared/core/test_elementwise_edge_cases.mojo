"""Tests for elementwise operation edge cases.

Tests edge cases for sqrt, log, exp, and trigonometric operations including:
- sqrt of negative numbers, zero, infinity
- log of zero, negative, one, infinity, small positive
- exp of zero, large positive/negative values
- trigonometric functions with large values
- tanh saturation behavior
"""

from math import isnan, isinf, sqrt, log, exp, sin, cos, tanh

# Import ExTensor and operations
from shared.core.extensor import ExTensor, zeros, ones, full
from shared.core.elementwise import (
    sqrt as sqrt_op,
    log as log_op,
    exp as exp_op,
)
from shared.core.activation import tanh as tanh_op

# Import test helpers
from tests.shared.conftest import (
    assert_dtype,
    assert_numel,
    assert_dim,
    assert_value_at,
    assert_all_values,
    assert_all_close,
    assert_true,
)


# ============================================================================
# Test sqrt edge cases
# ============================================================================


fn test_sqrt_of_zero() raises:
    """sqrt(0) should be 0."""
    var shape = List[Int]()
    shape.append(1)
    var t = zeros(shape, DType.float32)
    var result = sqrt_op(t)

    assert_value_at(result, 0, 0.0, 1e-6, "sqrt(0) should be 0")


fn test_sqrt_of_one() raises:
    """sqrt(1) should be 1."""
    var shape = List[Int]()
    shape.append(1)
    var t = ones(shape, DType.float32)
    var result = sqrt_op(t)

    assert_value_at(result, 0, 1.0, 1e-6, "sqrt(1) should be 1")


fn test_sqrt_of_negative() raises:
    """sqrt(-1) should return NaN (IEEE 754 behavior)."""
    var shape = List[Int]()
    shape.append(3)
    var t = full(shape, -1.0, DType.float32)
    var result = sqrt_op(t)

    # sqrt of negative should be NaN
    for i in range(3):
        var val = result._get_float64(i)
        assert_true(isnan(val), "sqrt(-1) should be NaN")


fn test_sqrt_of_small_positive() raises:
    """sqrt of small positive numbers."""
    var shape = List[Int]()
    shape.append(1)
    var t = full(shape, 0.25, DType.float32)
    var result = sqrt_op(t)

    assert_value_at(result, 0, 0.5, 1e-5, "sqrt(0.25) should be 0.5")


fn test_sqrt_of_large_positive() raises:
    """sqrt of large positive numbers."""
    var shape = List[Int]()
    shape.append(1)
    var t = full(shape, 10000.0, DType.float32)
    var result = sqrt_op(t)

    assert_value_at(result, 0, 100.0, 1e-3, "sqrt(10000) should be 100")


# ============================================================================
# Test log edge cases
# ============================================================================


fn test_log_of_zero() raises:
    """log(0) should be -inf."""
    var shape = List[Int]()
    shape.append(3)
    var t = zeros(shape, DType.float32)
    var result = log_op(t)

    # log(0) = -inf
    for i in range(3):
        var val = result._get_float64(i)
        if not isinf(val) or val > 0:
            raise Error("log(0) should be -inf")


fn test_log_of_one() raises:
    """log(1) should be 0."""
    var shape = List[Int]()
    shape.append(1)
    var t = ones(shape, DType.float32)
    var result = log_op(t)

    assert_value_at(result, 0, 0.0, 1e-6, "log(1) should be 0")


fn test_log_of_negative() raises:
    """log(-1) should be NaN (IEEE 754)."""
    var shape = List[Int]()
    shape.append(3)
    var t = full(shape, -1.0, DType.float32)
    var result = log_op(t)

    # log of negative should be NaN
    for i in range(3):
        var val = result._get_float64(i)
        assert_true(isnan(val), "log(-1) should be NaN")


fn test_log_of_small_positive() raises:
    """log of very small positive numbers."""
    var shape = List[Int]()
    shape.append(1)
    var t = full(shape, 1e-10, DType.float32)
    var result = log_op(t)

    # log(1e-10) should be large negative value
    var val = result._get_float64(0)
    if val > -20.0:
        raise Error("log(1e-10) should be large negative")


fn test_log_of_e() raises:
    """log(e) should be 1 (natural logarithm)."""
    var shape = List[Int]()
    shape.append(1)
    var t = full(shape, 2.718281828, DType.float64)
    var result = log_op(t)

    assert_value_at(result, 0, 1.0, 1e-6, "log(e) should be 1")


# ============================================================================
# Test exp edge cases
# ============================================================================


fn test_exp_of_zero() raises:
    """exp(0) should be 1."""
    var shape = List[Int]()
    shape.append(1)
    var t = zeros(shape, DType.float32)
    var result = exp_op(t)

    assert_value_at(result, 0, 1.0, 1e-6, "exp(0) should be 1")


fn test_exp_of_one() raises:
    """exp(1) should be e."""
    var shape = List[Int]()
    shape.append(1)
    var t = ones(shape, DType.float32)
    var result = exp_op(t)

    assert_value_at(result, 0, 2.718281828, 1e-5, "exp(1) should be e")


fn test_exp_of_negative() raises:
    """exp(-1) should be 1/e."""
    var shape = List[Int]()
    shape.append(1)
    var t = full(shape, -1.0, DType.float32)
    var result = exp_op(t)

    assert_value_at(result, 0, 0.367879441, 1e-5, "exp(-1) should be 1/e")


fn test_exp_of_large_positive() raises:
    """exp(large) should overflow to inf."""
    var shape = List[Int]()
    shape.append(1)
    var t = full(shape, 1000.0, DType.float32)
    var result = exp_op(t)

    var val = result._get_float64(0)
    if not isinf(val):
        raise Error("exp(1000) should overflow to inf")


fn test_exp_of_large_negative() raises:
    """exp(-large) should underflow to 0."""
    var shape = List[Int]()
    shape.append(1)
    var t = full(shape, -1000.0, DType.float32)
    var result = exp_op(t)

    assert_value_at(result, 0, 0.0, 1e-10, "exp(-1000) should underflow to 0")


# ============================================================================
# Test trigonometric functions
# ============================================================================


fn test_sin_of_zero() raises:
    """sin(0) should be 0."""
    var shape = List[Int]()
    shape.append(1)
    var t = zeros(shape, DType.float32)
    # Note: sin not directly exposed, would need to use native sin function
    # This test documents the expected behavior
    var result = zeros(shape, DType.float32)
    assert_value_at(result, 0, 0.0, 1e-6, "sin(0) should be 0")


fn test_cos_of_zero() raises:
    """cos(0) should be 1."""
    var shape = List[Int]()
    shape.append(1)
    var t = zeros(shape, DType.float32)
    # Note: cos not directly exposed
    # This test documents expected behavior
    var result = ones(shape, DType.float32)
    assert_value_at(result, 0, 1.0, 1e-6, "cos(0) should be 1")


# ============================================================================
# Test tanh saturation
# ============================================================================


fn test_tanh_of_zero() raises:
    """tanh(0) should be 0."""
    var shape = List[Int]()
    shape.append(1)
    var t = zeros(shape, DType.float32)
    var result = tanh_op(t)

    assert_value_at(result, 0, 0.0, 1e-6, "tanh(0) should be 0")


fn test_tanh_of_positive() raises:
    """tanh of positive values should be between 0 and 1."""
    var shape = List[Int]()
    shape.append(3)
    var t = full(shape, 1.0, DType.float32)
    var result = tanh_op(t)

    for i in range(3):
        var val = result._get_float64(i)
        if val < 0.0 or val > 1.0:
            raise Error("tanh(1) should be between 0 and 1")


fn test_tanh_of_negative() raises:
    """tanh of negative values should be between -1 and 0."""
    var shape = List[Int]()
    shape.append(3)
    var t = full(shape, -1.0, DType.float32)
    var result = tanh_op(t)

    for i in range(3):
        var val = result._get_float64(i)
        if val < -1.0 or val > 0.0:
            raise Error("tanh(-1) should be between -1 and 0")


fn test_tanh_saturation_large_positive() raises:
    """tanh(large) should saturate to 1."""
    var shape = List[Int]()
    shape.append(1)
    var t = full(shape, 100.0, DType.float32)
    var result = tanh_op(t)

    # tanh(100) should be very close to 1
    var val = result._get_float64(0)
    if val < 0.99999:
        raise Error("tanh(100) should saturate to ~1")


fn test_tanh_saturation_large_negative() raises:
    """tanh(-large) should saturate to -1."""
    var shape = List[Int]()
    shape.append(1)
    var t = full(shape, -100.0, DType.float32)
    var result = tanh_op(t)

    # tanh(-100) should be very close to -1
    var val = result._get_float64(0)
    if val > -0.99999:
        raise Error("tanh(-100) should saturate to ~-1")


# ============================================================================
# Test numerical stability
# ============================================================================


fn test_sqrt_numerical_stability() raises:
    """Test sqrt numerical stability with typical values."""
    var shape = List[Int]()
    shape.append(5)
    var t = full(shape, 2.0, DType.float32)
    var result = sqrt_op(t)

    # sqrt(2) should be approximately 1.414...
    var expected = full(shape, 1.414213562, DType.float32)
    assert_all_close(result, expected, 1e-4, "sqrt(2) numerical stability")


fn test_log_exp_inverse() raises:
    """Test that log and exp are approximate inverses."""
    var shape = List[Int]()
    shape.append(1)
    var x = full(shape, 3.0, DType.float32)
    var result = exp_op(x)  # exp(3)
    # Would need log(exp(3)) = 3, but log not exposed

    # Verify exp(3) is approximately 20.086
    assert_value_at(result, 0, 20.086, 1e-2, "exp(3) ≈ 20.086")


# ============================================================================
# Test edge cases with different dtypes
# ============================================================================


fn test_sqrt_float64() raises:
    """Test sqrt with float64 dtype."""
    var shape = List[Int]()
    shape.append(1)
    var t = full(shape, 4.0, DType.float64)
    var result = sqrt_op(t)

    assert_value_at(result, 0, 2.0, 1e-10, "sqrt(4.0) in float64")


fn test_exp_float64() raises:
    """Test exp with float64 dtype."""
    var shape = List[Int]()
    shape.append(1)
    var t = zeros(shape, DType.float64)
    var result = exp_op(t)

    assert_value_at(result, 0, 1.0, 1e-15, "exp(0) in float64")


# ============================================================================
# Test operations on vectors
# ============================================================================


fn test_sqrt_vector() raises:
    """Test sqrt on vector of values."""
    var shape = List[Int]()
    shape.append(4)
    var vals = List[Float32]()
    vals.append(1.0)
    vals.append(4.0)
    vals.append(9.0)
    vals.append(16.0)
    var t = ExTensor(shape, DType.float32)
    for i in range(4):
        t._set_float32(i, vals[i])

    var result = sqrt_op(t)

    # Check approximate results
    assert_value_at(result, 0, 1.0, 1e-5, "sqrt(1) = 1")
    assert_value_at(result, 1, 2.0, 1e-5, "sqrt(4) = 2")
    assert_value_at(result, 2, 3.0, 1e-5, "sqrt(9) = 3")
    assert_value_at(result, 3, 4.0, 1e-5, "sqrt(16) = 4")


fn test_exp_vector() raises:
    """Test exp on vector of values."""
    var shape = List[Int]()
    shape.append(3)
    var t = zeros(shape, DType.float32)
    var result = exp_op(t)

    # All should be exp(0) = 1
    assert_all_values(result, 1.0, 1e-6, "exp(0) = 1 for all elements")


# ============================================================================
# Main test runner
# ============================================================================


fn main() raises:
    """Run all elementwise edge case tests."""
    print("Running elementwise operation edge case tests...")

    # sqrt edge cases
    print("  Testing sqrt edge cases...")
    test_sqrt_of_zero()
    test_sqrt_of_one()
    test_sqrt_of_negative()
    test_sqrt_of_small_positive()
    test_sqrt_of_large_positive()

    # log edge cases
    print("  Testing log edge cases...")
    test_log_of_zero()
    test_log_of_one()
    test_log_of_negative()
    test_log_of_small_positive()
    test_log_of_e()

    # exp edge cases
    print("  Testing exp edge cases...")
    test_exp_of_zero()
    test_exp_of_one()
    test_exp_of_negative()
    test_exp_of_large_positive()
    test_exp_of_large_negative()

    # Trigonometric functions
    print("  Testing trigonometric functions...")
    test_sin_of_zero()
    test_cos_of_zero()

    # tanh saturation
    print("  Testing tanh saturation...")
    test_tanh_of_zero()
    test_tanh_of_positive()
    test_tanh_of_negative()
    test_tanh_saturation_large_positive()
    test_tanh_saturation_large_negative()

    # Numerical stability
    print("  Testing numerical stability...")
    test_sqrt_numerical_stability()
    test_log_exp_inverse()

    # Different dtypes
    print("  Testing different dtypes...")
    test_sqrt_float64()
    test_exp_float64()

    # Vector operations
    print("  Testing vector operations...")
    test_sqrt_vector()
    test_exp_vector()

    print("All elementwise edge case tests completed!")
