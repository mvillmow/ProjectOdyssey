"""Tests for gradient checking utilities in shared.testing.gradient_checker.

Tests all gradient checking functions including:
- numerical_gradient() using central differences
- check_gradient() comprehensive validation
- compute_numerical_gradient() standalone numerical computation
- assert_gradients_close() assertion helper
- relative_error() error computation

These tests verify the gradient checking infrastructure works correctly
for validating backward passes throughout the codebase.
"""

from testing import assert_true, assert_equal
from shared.testing import (
    check_gradients,
    check_gradients_verbose,
    compute_numerical_gradient,
    assert_gradients_close,
    relative_error,
    check_gradient,
)
from shared.core import ExTensor, zeros, ones, full, zeros_like
from shared.core.arithmetic import add, multiply, add_backward, multiply_backward


# ============================================================================
# Test Setup: Simple Functions for Gradient Checking
# ============================================================================


fn square_forward(x: ExTensor) raises escaping -> ExTensor:
    """Forward: y = x²"""
    return multiply(x, x)


fn square_backward(grad_out: ExTensor, x: ExTensor) raises escaping -> ExTensor:
    """Backward: dy/dx = 2x"""
    var two = full(zeros_like(x).shape(), 2.0, x._dtype)
    return multiply_backward(grad_out, x, zeros_like(x), two)


fn cube_forward(x: ExTensor) raises escaping -> ExTensor:
    """Forward: y = x³ = x · x · x"""
    var x_squared = multiply(x, x)
    return multiply(x_squared, x)


fn cube_backward(grad_out: ExTensor, x: ExTensor) raises escaping -> ExTensor:
    """Backward: dy/dx = 3x²"""
    var x_squared = multiply(x, x)
    var three = full(zeros_like(x).shape(), 3.0, x._dtype)
    return multiply_backward(grad_out, x_squared, zeros_like(x), three)


fn linear_forward(x: ExTensor) raises escaping -> ExTensor:
    """Forward: y = 2x + 1"""
    var two = full(x.shape(), 2.0, x._dtype)
    var result = multiply(x, two)
    var one = full(x.shape(), 1.0, x._dtype)
    return add(result, one)


fn linear_backward(grad_out: ExTensor, x: ExTensor) raises escaping -> ExTensor:
    """Backward: dy/dx = 2"""
    var two = full(x.shape(), 2.0, x._dtype)
    return multiply(grad_out, two)


# ============================================================================
# Basic Gradient Checking Tests
# ============================================================================


fn test_square_function_gradient() raises:
    """Test gradient checking on y = x² function.

    Expected: Numerical gradient should match analytical gradient.
    Analytical: dy/dx = 2x
    For x=2: dy/dx = 4
    """
    print("Testing square function gradient...")
    var shape = List[Int](2, 3)
    var x = full(shape, 2.0, DType.float32)

    var passed = check_gradients(square_forward, square_backward, x)
    assert_true(passed, "Square function gradient check failed")
    print("  ✓ Square function gradient correct")


fn test_cube_function_gradient() raises:
    """Test gradient checking on y = x³ function.

    Expected: Numerical gradient should match analytical gradient.
    Analytical: dy/dx = 3x²
    For x=2: dy/dx = 12
    """
    print("Testing cube function gradient...")
    var shape = List[Int](2, 3)
    var x = full(shape, 2.0, DType.float32)

    var passed = check_gradients(cube_forward, cube_backward, x)
    assert_true(passed, "Cube function gradient check failed")
    print("  ✓ Cube function gradient correct")


fn test_linear_function_gradient() raises:
    """Test gradient checking on y = 2x + 1 function.

    Expected: Numerical gradient should match analytical gradient.
    Analytical: dy/dx = 2 (constant)
    """
    print("Testing linear function gradient...")
    var shape = List[Int](3, 4)
    var x = full(shape, 1.5, DType.float32)

    var passed = check_gradients(linear_forward, linear_backward, x)
    assert_true(passed, "Linear function gradient check failed")
    print("  ✓ Linear function gradient correct")


# ============================================================================
# Numerical Gradient Computation Tests
# ============================================================================


fn test_compute_numerical_gradient_square() raises:
    """Test compute_numerical_gradient on y = x² function."""
    print("Testing compute_numerical_gradient on square function...")
    var shape = List[Int](2, 3)
    var x = full(shape, 2.0, DType.float32)

    # Compute numerical gradient
    var numerical_grad = compute_numerical_gradient(square_forward, x, 1e-4)

    # Expected: dy/dx = 2x = 4.0
    var tolerance = 1e-3
    for i in range(numerical_grad.numel()):
        var val = numerical_grad._get_float64(i)
        var expected = 4.0
        var diff = abs(val - expected)
        assert_true(
            diff < tolerance,
            "Numerical gradient mismatch: " + String(val) + " vs " + String(expected)
        )

    print("  ✓ Numerical gradient computation correct")


fn test_compute_numerical_gradient_linear() raises:
    """Test compute_numerical_gradient on y = 2x + 1 function."""
    print("Testing compute_numerical_gradient on linear function...")
    var shape = List[Int](3, 4)
    var x = full(shape, 1.0, DType.float32)

    # Compute numerical gradient
    var numerical_grad = compute_numerical_gradient(linear_forward, x, 1e-4)

    # Expected: dy/dx = 2.0
    var tolerance = 1e-3
    for i in range(numerical_grad.numel()):
        var val = numerical_grad._get_float64(i)
        var expected = 2.0
        var diff = abs(val - expected)
        assert_true(
            diff < tolerance,
            "Numerical gradient mismatch at index " + String(i)
        )

    print("  ✓ Numerical gradient linear function correct")


# ============================================================================
# Relative Error Tests
# ============================================================================


fn test_relative_error_identical() raises:
    """Test relative_error with identical values.

    Expected: relative_error(x, x) = 0.0
    """
    print("Testing relative_error with identical values...")
    var err = relative_error(1.0, 1.0)
    assert_true(err < 1e-10, "Relative error should be ~0")
    print("  ✓ Relative error identical values correct")


fn test_relative_error_small_difference() raises:
    """Test relative_error with small difference.

    Expected: relative_error(1.0, 1.001) ≈ 0.001
    """
    print("Testing relative_error with small difference...")
    var err = relative_error(1.0, 1.001)
    assert_true(err > 0.0009 and err < 0.0011, "Relative error mismatch")
    print("  ✓ Relative error small difference correct")


fn test_relative_error_near_zero() raises:
    """Test relative_error with values near zero.

    Formula: |a - n| / max(|a|, |n|, 1e-8)
    Should not divide by zero when both values are near zero.
    """
    print("Testing relative_error near zero values...")
    var err = relative_error(1e-9, 1e-9)
    assert_true(err < 1e-10, "Relative error should handle near-zero gracefully")
    print("  ✓ Relative error near-zero handling correct")


# ============================================================================
# Gradient Assertion Tests
# ============================================================================


fn test_assert_gradients_close_identical() raises:
    """Test assert_gradients_close with identical gradients."""
    print("Testing assert_gradients_close with identical gradients...")
    var shape = List[Int](3, 4)
    var grad1 = full(shape, 1.5, DType.float32)
    var grad2 = full(shape, 1.5, DType.float32)

    # Should not raise
    assert_gradients_close(grad1, grad2)
    print("  ✓ assert_gradients_close identical gradients passes")


fn test_assert_gradients_close_within_tolerance() raises:
    """Test assert_gradients_close with close gradients."""
    print("Testing assert_gradients_close within tolerance...")
    var shape = List[Int](3, 4)
    var grad1 = full(shape, 1.0, DType.float32)
    var grad2 = full(shape, 1.001, DType.float32)

    # Should not raise with default tolerances
    assert_gradients_close(grad1, grad2, rtol=1e-2, atol=1e-2)
    print("  ✓ assert_gradients_close within tolerance passes")


fn test_assert_gradients_close_mismatched_shape() raises:
    """Test assert_gradients_close with mismatched shapes."""
    print("Testing assert_gradients_close with mismatched shapes...")
    var shape1 = List[Int](3, 4)
    var shape2 = List[Int](4, 5)
    var grad1 = full(shape1, 1.0, DType.float32)
    var grad2 = full(shape2, 1.0, DType.float32)

    var failed = False
    try:
        assert_gradients_close(grad1, grad2)
    except:
        failed = True

    assert_true(failed, "Should raise error on shape mismatch")
    print("  ✓ assert_gradients_close shape mismatch detection works")


fn test_assert_gradients_close_mismatched_dtype() raises:
    """Test assert_gradients_close with mismatched dtypes."""
    print("Testing assert_gradients_close with mismatched dtypes...")
    var shape = List[Int](3, 4)
    var grad1 = full(shape, 1.0, DType.float32)
    var grad2 = full(shape, 1.0, DType.float64)

    var failed = False
    try:
        assert_gradients_close(grad1, grad2)
    except:
        failed = True

    assert_true(failed, "Should raise error on dtype mismatch")
    print("  ✓ assert_gradients_close dtype mismatch detection works")


# ============================================================================
# Comprehensive Gradient Check Tests
# ============================================================================


fn test_check_gradient_simple() raises:
    """Test check_gradient comprehensive function."""
    print("Testing check_gradient function...")
    var shape = List[Int](2, 3)
    var x = full(shape, 1.5, DType.float32)
    var grad_output = ones(shape, DType.float32)

    # Should not raise if gradients match
    check_gradient(square_forward, square_backward, x, grad_output)
    print("  ✓ check_gradient function works")


fn test_check_gradient_with_auto_epsilon() raises:
    """Test check_gradient with auto-selected epsilon."""
    print("Testing check_gradient with auto epsilon...")
    var shape = List[Int](2, 3)
    var x = full(shape, 2.0, DType.float32)
    var grad_output = ones(shape, DType.float32)

    # epsilon=0.0 triggers auto-selection
    check_gradient(
        square_forward,
        square_backward,
        x,
        grad_output,
        epsilon=0.0  # Auto-select based on dtype
    )
    print("  ✓ check_gradient auto-epsilon selection works")


fn test_check_gradient_float64() raises:
    """Test check_gradient with Float64 dtype."""
    print("Testing check_gradient with Float64...")
    var shape = List[Int](2, 3)
    var x = full(shape, 1.5, DType.float64)
    var grad_output = ones(shape, DType.float64)

    check_gradient(
        square_forward,
        square_backward,
        x,
        grad_output,
        epsilon=0.0  # Auto-select based on Float64
    )
    print("  ✓ check_gradient Float64 works")


# ============================================================================
# Verbose Gradient Checking Tests
# ============================================================================


fn test_check_gradients_verbose_passes() raises:
    """Test check_gradients_verbose with passing gradients."""
    print("Testing check_gradients_verbose with passing gradients...")
    var shape = List[Int](2, 3)
    var x = full(shape, 2.0, DType.float32)

    var passed = check_gradients_verbose(
        square_forward,
        square_backward,
        x,
        print_all=False  # Don't print unless failing
    )
    assert_true(passed, "check_gradients_verbose should pass")
    print("  ✓ check_gradients_verbose passes")


fn test_check_gradients_verbose_with_printing() raises:
    """Test check_gradients_verbose with print_all=True."""
    print("Testing check_gradients_verbose with detailed printing...")
    var shape = List[Int](2, 2)
    var x = full(shape, 1.5, DType.float32)

    var passed = check_gradients_verbose(
        square_forward,
        square_backward,
        x,
        print_all=True  # Print all comparisons
    )
    assert_true(passed, "Gradient check should pass")
    print("  ✓ check_gradients_verbose detailed printing works")


# ============================================================================
# Edge Cases and Robustness Tests
# ============================================================================


fn test_gradient_check_small_values() raises:
    """Test gradient checking with small input values.

    Verifies numerical stability near zero.
    """
    print("Testing gradient check with small values...")
    var shape = List[Int](3, 4)
    var x = full(shape, 0.01, DType.float32)

    var passed = check_gradients(square_forward, square_backward, x)
    assert_true(passed, "Gradient check should handle small values")
    print("  ✓ Gradient check handles small values")


fn test_gradient_check_large_values() raises:
    """Test gradient checking with large input values."""
    print("Testing gradient check with large values...")
    var shape = List[Int](3, 4)
    var x = full(shape, 100.0, DType.float32)

    var passed = check_gradients(square_forward, square_backward, x)
    assert_true(passed, "Gradient check should handle large values")
    print("  ✓ Gradient check handles large values")


fn test_gradient_check_negative_values() raises:
    """Test gradient checking with negative input values."""
    print("Testing gradient check with negative values...")
    var shape = List[Int](3, 4)
    var x = full(shape, -2.5, DType.float32)

    var passed = check_gradients(square_forward, square_backward, x)
    assert_true(passed, "Gradient check should handle negative values")
    print("  ✓ Gradient check handles negative values")


fn test_gradient_check_mixed_values() raises:
    """Test gradient checking with mixed positive/negative values."""
    print("Testing gradient check with mixed values...")
    var shape = List[Int](2, 3)
    var x = zeros(shape, DType.float32)
    x._set_float64(0, 1.5)
    x._set_float64(1, -1.5)
    x._set_float64(2, 0.5)
    x._set_float64(3, -0.5)
    x._set_float64(4, 2.0)
    x._set_float64(5, -2.0)

    var passed = check_gradients(square_forward, square_backward, x)
    assert_true(passed, "Gradient check should handle mixed values")
    print("  ✓ Gradient check handles mixed values")


# ============================================================================
# Different Tensor Shapes
# ============================================================================


fn test_gradient_check_1d_tensor() raises:
    """Test gradient checking on 1D tensors."""
    print("Testing gradient check on 1D tensor...")
    var shape = List[Int](10)
    var x = full(shape, 2.0, DType.float32)

    var passed = check_gradients(square_forward, square_backward, x)
    assert_true(passed, "Gradient check should work on 1D tensors")
    print("  ✓ Gradient check works on 1D tensors")


fn test_gradient_check_2d_tensor() raises:
    """Test gradient checking on 2D tensors."""
    print("Testing gradient check on 2D tensor...")
    var shape = List[Int](4, 5)
    var x = full(shape, 1.5, DType.float32)

    var passed = check_gradients(square_forward, square_backward, x)
    assert_true(passed, "Gradient check should work on 2D tensors")
    print("  ✓ Gradient check works on 2D tensors")


fn test_gradient_check_3d_tensor() raises:
    """Test gradient checking on 3D tensors."""
    print("Testing gradient check on 3D tensor...")
    var shape = List[Int](2, 3, 4)
    var x = full(shape, 1.0, DType.float32)

    var passed = check_gradients(square_forward, square_backward, x)
    assert_true(passed, "Gradient check should work on 3D tensors")
    print("  ✓ Gradient check works on 3D tensors")


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run all gradient checking tests."""
    print("=" * 60)
    print("GRADIENT CHECKING UTILITY TESTS")
    print("=" * 60)

    # Basic gradient checking
    test_square_function_gradient()
    test_cube_function_gradient()
    test_linear_function_gradient()

    # Numerical gradient computation
    test_compute_numerical_gradient_square()
    test_compute_numerical_gradient_linear()

    # Relative error
    test_relative_error_identical()
    test_relative_error_small_difference()
    test_relative_error_near_zero()

    # Gradient assertions
    test_assert_gradients_close_identical()
    test_assert_gradients_close_within_tolerance()
    test_assert_gradients_close_mismatched_shape()
    test_assert_gradients_close_mismatched_dtype()

    # Comprehensive checks
    test_check_gradient_simple()
    test_check_gradient_with_auto_epsilon()
    test_check_gradient_float64()

    # Verbose checking
    test_check_gradients_verbose_passes()
    test_check_gradients_verbose_with_printing()

    # Edge cases
    test_gradient_check_small_values()
    test_gradient_check_large_values()
    test_gradient_check_negative_values()
    test_gradient_check_mixed_values()

    # Different shapes
    test_gradient_check_1d_tensor()
    test_gradient_check_2d_tensor()
    test_gradient_check_3d_tensor()

    print("=" * 60)
    print("All gradient checking tests passed!")
    print("=" * 60)
