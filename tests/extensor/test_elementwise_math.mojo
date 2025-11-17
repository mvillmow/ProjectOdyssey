"""Tests for ExTensor element-wise mathematical operations.

Tests transcendental functions (exp, log, sqrt), trigonometric functions (sin, cos, tanh),
and utility functions (abs, sign, clip) following Array API Standard.
"""

from sys import DType
from math import abs as math_abs

# Import ExTensor and operations
from extensor import ExTensor, zeros, ones, full, arange, abs, sign, exp, log, sqrt, sin, cos, tanh, clip

# Import test helpers
from ..helpers.assertions import (
    assert_dtype,
    assert_numel,
    assert_dim,
    assert_value_at,
    assert_all_values,
    assert_all_close,
)


# ============================================================================
# Test abs() - Absolute value
# ============================================================================

fn test_abs_positive() raises:
    """Test abs with positive values."""
    var shape = DynamicVector[Int](1)
    shape[0] = 5
    let a = arange(1.0, 6.0, 1.0, DType.float32)  # [1, 2, 3, 4, 5]
    let b = abs(a)

    # Positive values should remain unchanged
    assert_value_at(b, 0, 1.0, 1e-6, "abs(1) = 1")
    assert_value_at(b, 1, 2.0, 1e-6, "abs(2) = 2")
    assert_value_at(b, 2, 3.0, 1e-6, "abs(3) = 3")
    assert_value_at(b, 3, 4.0, 1e-6, "abs(4) = 4")
    assert_value_at(b, 4, 5.0, 1e-6, "abs(5) = 5")


fn test_abs_negative() raises:
    """Test abs with negative values."""
    var shape = DynamicVector[Int](1)
    shape[0] = 5
    let a = full(shape, -3.0, DType.float32)
    let b = abs(a)

    # Should convert to positive
    assert_all_values(b, 3.0, 1e-6, "abs(negative) = positive")


fn test_abs_mixed() raises:
    """Test abs with mixed positive/negative values."""
    # Create array: [-2, -1, 0, 1, 2]
    let a = arange(-2.0, 3.0, 1.0, DType.float32)
    let b = abs(a)

    # Expected: [2, 1, 0, 1, 2]
    assert_value_at(b, 0, 2.0, 1e-6, "abs(-2) = 2")
    assert_value_at(b, 1, 1.0, 1e-6, "abs(-1) = 1")
    assert_value_at(b, 2, 0.0, 1e-6, "abs(0) = 0")
    assert_value_at(b, 3, 1.0, 1e-6, "abs(1) = 1")
    assert_value_at(b, 4, 2.0, 1e-6, "abs(2) = 2")


fn test_abs_preserves_dtype() raises:
    """Test that abs preserves dtype."""
    var shape = DynamicVector[Int](1)
    shape[0] = 5
    let a = ones(shape, DType.float64)
    let b = abs(a)

    assert_dtype(b, DType.float64, "abs should preserve float64")


# ============================================================================
# Test sign() - Sign of values
# ============================================================================

fn test_sign_positive() raises:
    """Test sign with positive values."""
    var shape = DynamicVector[Int](1)
    shape[0] = 3
    let a = full(shape, 5.0, DType.float32)
    let b = sign(a)

    # Positive values should give +1
    assert_all_values(b, 1.0, 1e-6, "sign(positive) = 1")


fn test_sign_negative() raises:
    """Test sign with negative values."""
    var shape = DynamicVector[Int](1)
    shape[0] = 3
    let a = full(shape, -5.0, DType.float32)
    let b = sign(a)

    # Negative values should give -1
    assert_all_values(b, -1.0, 1e-6, "sign(negative) = -1")


fn test_sign_zero() raises:
    """Test sign with zero values."""
    var shape = DynamicVector[Int](1)
    shape[0] = 3
    let a = zeros(shape, DType.float32)
    let b = sign(a)

    # Zero should give 0
    assert_all_values(b, 0.0, 1e-6, "sign(0) = 0")


fn test_sign_mixed() raises:
    """Test sign with mixed values."""
    # Create array: [-2, -1, 0, 1, 2]
    let a = arange(-2.0, 3.0, 1.0, DType.float32)
    let b = sign(a)

    # Expected: [-1, -1, 0, 1, 1]
    assert_value_at(b, 0, -1.0, 1e-6, "sign(-2) = -1")
    assert_value_at(b, 1, -1.0, 1e-6, "sign(-1) = -1")
    assert_value_at(b, 2, 0.0, 1e-6, "sign(0) = 0")
    assert_value_at(b, 3, 1.0, 1e-6, "sign(1) = 1")
    assert_value_at(b, 4, 1.0, 1e-6, "sign(2) = 1")


# ============================================================================
# Test exp() - Exponential
# ============================================================================

fn test_exp_zeros() raises:
    """Test exp(0) = 1."""
    var shape = DynamicVector[Int](1)
    shape[0] = 5
    let a = zeros(shape, DType.float32)
    let b = exp(a)

    # exp(0) = 1
    assert_all_values(b, 1.0, 1e-6, "exp(0) should be 1")


fn test_exp_ones() raises:
    """Test exp(1) ≈ 2.71828."""
    var shape = DynamicVector[Int](1)
    shape[0] = 3
    let a = ones(shape, DType.float32)
    let b = exp(a)

    # exp(1) ≈ e ≈ 2.71828
    assert_all_values(b, 2.71828, 1e-5, "exp(1) should be approximately e")


fn test_exp_small_values() raises:
    """Test exp with small values."""
    var shape = DynamicVector[Int](1)
    shape[0] = 1
    let a = full(shape, 0.5, DType.float32)
    let b = exp(a)

    # exp(0.5) ≈ 1.64872
    assert_value_at(b, 0, 1.64872, 1e-4, "exp(0.5) should be ~1.649")


fn test_exp_negative() raises:
    """Test exp with negative values."""
    var shape = DynamicVector[Int](1)
    shape[0] = 1
    let a = full(shape, -1.0, DType.float32)
    let b = exp(a)

    # exp(-1) ≈ 0.36788 (1/e)
    assert_value_at(b, 0, 0.36788, 1e-4, "exp(-1) should be ~0.368")


# ============================================================================
# Test log() - Natural logarithm
# ============================================================================

fn test_log_one() raises:
    """Test log(1) = 0."""
    var shape = DynamicVector[Int](1)
    shape[0] = 5
    let a = ones(shape, DType.float32)
    let b = log(a)

    # log(1) = 0
    assert_all_values(b, 0.0, 1e-6, "log(1) should be 0")


fn test_log_e() raises:
    """Test log(e) = 1."""
    var shape = DynamicVector[Int](1)
    shape[0] = 1
    let a = full(shape, 2.71828, DType.float32)  # e
    let b = log(a)

    # log(e) = 1
    assert_value_at(b, 0, 1.0, 1e-4, "log(e) should be 1")


fn test_log_powers_of_2() raises:
    """Test log with powers of 2."""
    var shape = DynamicVector[Int](1)
    shape[0] = 1
    let a = full(shape, 2.0, DType.float32)
    let b = log(a)

    # log(2) ≈ 0.69315
    assert_value_at(b, 0, 0.69315, 1e-4, "log(2) should be ~0.693")


# ============================================================================
# Test sqrt() - Square root
# ============================================================================

fn test_sqrt_perfect_squares() raises:
    """Test sqrt with perfect squares."""
    # Create array: [1, 4, 9, 16, 25]
    var shape = DynamicVector[Int](1)
    shape[0] = 5
    var a = ExTensor(shape, DType.float32)
    a._set_float64(0, 1.0)
    a._set_float64(1, 4.0)
    a._set_float64(2, 9.0)
    a._set_float64(3, 16.0)
    a._set_float64(4, 25.0)
    let b = sqrt(a)

    # Expected: [1, 2, 3, 4, 5]
    assert_value_at(b, 0, 1.0, 1e-6, "sqrt(1) = 1")
    assert_value_at(b, 1, 2.0, 1e-6, "sqrt(4) = 2")
    assert_value_at(b, 2, 3.0, 1e-6, "sqrt(9) = 3")
    assert_value_at(b, 3, 4.0, 1e-6, "sqrt(16) = 4")
    assert_value_at(b, 4, 5.0, 1e-6, "sqrt(25) = 5")


fn test_sqrt_zero() raises:
    """Test sqrt(0) = 0."""
    var shape = DynamicVector[Int](1)
    shape[0] = 3
    let a = zeros(shape, DType.float32)
    let b = sqrt(a)

    # sqrt(0) = 0
    assert_all_values(b, 0.0, 1e-6, "sqrt(0) should be 0")


fn test_sqrt_one() raises:
    """Test sqrt(1) = 1."""
    var shape = DynamicVector[Int](1)
    shape[0] = 3
    let a = ones(shape, DType.float32)
    let b = sqrt(a)

    # sqrt(1) = 1
    assert_all_values(b, 1.0, 1e-6, "sqrt(1) should be 1")


fn test_sqrt_two() raises:
    """Test sqrt(2) ≈ 1.41421."""
    var shape = DynamicVector[Int](1)
    shape[0] = 1
    let a = full(shape, 2.0, DType.float32)
    let b = sqrt(a)

    # sqrt(2) ≈ 1.41421
    assert_value_at(b, 0, 1.41421, 1e-4, "sqrt(2) should be ~1.414")


# ============================================================================
# Test sin() - Sine
# ============================================================================

fn test_sin_zero() raises:
    """Test sin(0) = 0."""
    var shape = DynamicVector[Int](1)
    shape[0] = 3
    let a = zeros(shape, DType.float32)
    let b = sin(a)

    # sin(0) = 0
    assert_all_values(b, 0.0, 1e-6, "sin(0) should be 0")


fn test_sin_pi_over_2() raises:
    """Test sin(π/2) = 1."""
    var shape = DynamicVector[Int](1)
    shape[0] = 1
    let a = full(shape, 1.5708, DType.float32)  # π/2 ≈ 1.5708
    let b = sin(a)

    # sin(π/2) = 1
    assert_value_at(b, 0, 1.0, 1e-4, "sin(π/2) should be 1")


fn test_sin_pi() raises:
    """Test sin(π) ≈ 0."""
    var shape = DynamicVector[Int](1)
    shape[0] = 1
    let a = full(shape, 3.14159, DType.float32)  # π
    let b = sin(a)

    # sin(π) ≈ 0
    assert_value_at(b, 0, 0.0, 1e-5, "sin(π) should be ~0")


# ============================================================================
# Test cos() - Cosine
# ============================================================================

fn test_cos_zero() raises:
    """Test cos(0) = 1."""
    var shape = DynamicVector[Int](1)
    shape[0] = 3
    let a = zeros(shape, DType.float32)
    let b = cos(a)

    # cos(0) = 1
    assert_all_values(b, 1.0, 1e-6, "cos(0) should be 1")


fn test_cos_pi_over_2() raises:
    """Test cos(π/2) ≈ 0."""
    var shape = DynamicVector[Int](1)
    shape[0] = 1
    let a = full(shape, 1.5708, DType.float32)  # π/2
    let b = cos(a)

    # cos(π/2) ≈ 0
    assert_value_at(b, 0, 0.0, 1e-4, "cos(π/2) should be ~0")


fn test_cos_pi() raises:
    """Test cos(π) = -1."""
    var shape = DynamicVector[Int](1)
    shape[0] = 1
    let a = full(shape, 3.14159, DType.float32)  # π
    let b = cos(a)

    # cos(π) = -1
    assert_value_at(b, 0, -1.0, 1e-4, "cos(π) should be ~-1")


# ============================================================================
# Test tanh() - Hyperbolic tangent
# ============================================================================

fn test_tanh_zero() raises:
    """Test tanh(0) = 0."""
    var shape = DynamicVector[Int](1)
    shape[0] = 3
    let a = zeros(shape, DType.float32)
    let b = tanh(a)

    # tanh(0) = 0
    assert_all_values(b, 0.0, 1e-6, "tanh(0) should be 0")


fn test_tanh_large_positive() raises:
    """Test tanh(large) ≈ 1."""
    var shape = DynamicVector[Int](1)
    shape[0] = 1
    let a = full(shape, 10.0, DType.float32)
    let b = tanh(a)

    # tanh(10) ≈ 1 (saturates)
    assert_value_at(b, 0, 1.0, 1e-5, "tanh(large) should be ~1")


fn test_tanh_large_negative() raises:
    """Test tanh(-large) ≈ -1."""
    var shape = DynamicVector[Int](1)
    shape[0] = 1
    let a = full(shape, -10.0, DType.float32)
    let b = tanh(a)

    # tanh(-10) ≈ -1 (saturates)
    assert_value_at(b, 0, -1.0, 1e-5, "tanh(-large) should be ~-1")


fn test_tanh_small_values() raises:
    """Test tanh with small values."""
    var shape = DynamicVector[Int](1)
    shape[0] = 1
    let a = full(shape, 0.5, DType.float32)
    let b = tanh(a)

    # tanh(0.5) ≈ 0.46212
    assert_value_at(b, 0, 0.46212, 1e-4, "tanh(0.5) should be ~0.462")


# ============================================================================
# Test clip() - Clip values to range
# ============================================================================

fn test_clip_basic() raises:
    """Test clip with basic range."""
    # Create array: [1, 2, 3, 4, 5]
    let a = arange(1.0, 6.0, 1.0, DType.float32)
    let b = clip(a, 2.0, 4.0)

    # Expected: [2, 2, 3, 4, 4]
    assert_value_at(b, 0, 2.0, 1e-6, "clip(1, 2, 4) = 2")
    assert_value_at(b, 1, 2.0, 1e-6, "clip(2, 2, 4) = 2")
    assert_value_at(b, 2, 3.0, 1e-6, "clip(3, 2, 4) = 3")
    assert_value_at(b, 3, 4.0, 1e-6, "clip(4, 2, 4) = 4")
    assert_value_at(b, 4, 4.0, 1e-6, "clip(5, 2, 4) = 4")


fn test_clip_all_below() raises:
    """Test clip when all values below min."""
    var shape = DynamicVector[Int](1)
    shape[0] = 3
    let a = ones(shape, DType.float32)
    let b = clip(a, 5.0, 10.0)

    # All values should be clipped to min
    assert_all_values(b, 5.0, 1e-6, "clip(1, 5, 10) should be 5")


fn test_clip_all_above() raises:
    """Test clip when all values above max."""
    var shape = DynamicVector[Int](1)
    shape[0] = 3
    let a = full(shape, 20.0, DType.float32)
    let b = clip(a, 5.0, 10.0)

    # All values should be clipped to max
    assert_all_values(b, 10.0, 1e-6, "clip(20, 5, 10) should be 10")


# ============================================================================
# Test dtype preservation for all operations
# ============================================================================

fn test_operations_preserve_dtype() raises:
    """Test that all operations preserve dtype."""
    var shape = DynamicVector[Int](1)
    shape[0] = 3
    let a = ones(shape, DType.float64)

    # All operations should preserve float64
    let b_abs = abs(a)
    assert_dtype(b_abs, DType.float64, "abs should preserve dtype")

    let b_sign = sign(a)
    assert_dtype(b_sign, DType.float64, "sign should preserve dtype")

    let b_exp = exp(a)
    assert_dtype(b_exp, DType.float64, "exp should preserve dtype")

    let b_log = log(a)
    assert_dtype(b_log, DType.float64, "log should preserve dtype")

    let b_sqrt = sqrt(a)
    assert_dtype(b_sqrt, DType.float64, "sqrt should preserve dtype")


# ============================================================================
# Main test runner
# ============================================================================

fn main() raises:
    """Run all element-wise math tests."""
    print("Running ExTensor element-wise math tests...")

    # abs() tests
    print("  Testing abs()...")
    test_abs_positive()
    test_abs_negative()
    test_abs_mixed()
    test_abs_preserves_dtype()

    # sign() tests
    print("  Testing sign()...")
    test_sign_positive()
    test_sign_negative()
    test_sign_zero()
    test_sign_mixed()

    # exp() tests
    print("  Testing exp()...")
    test_exp_zeros()
    test_exp_ones()
    test_exp_small_values()
    test_exp_negative()

    # log() tests
    print("  Testing log()...")
    test_log_one()
    test_log_e()
    test_log_powers_of_2()

    # sqrt() tests
    print("  Testing sqrt()...")
    test_sqrt_perfect_squares()
    test_sqrt_zero()
    test_sqrt_one()
    test_sqrt_two()

    # sin() tests
    print("  Testing sin()...")
    test_sin_zero()
    test_sin_pi_over_2()
    test_sin_pi()

    # cos() tests
    print("  Testing cos()...")
    test_cos_zero()
    test_cos_pi_over_2()
    test_cos_pi()

    # tanh() tests
    print("  Testing tanh()...")
    test_tanh_zero()
    test_tanh_large_positive()
    test_tanh_large_negative()
    test_tanh_small_values()

    # clip() tests
    print("  Testing clip()...")
    test_clip_basic()
    test_clip_all_below()
    test_clip_all_above()

    # dtype preservation
    print("  Testing dtype preservation...")
    test_operations_preserve_dtype()

    print("All element-wise math tests completed!")
