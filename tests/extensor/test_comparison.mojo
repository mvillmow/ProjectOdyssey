"""Tests for ExTensor comparison operations.

Tests comparison operations including equal, not_equal, less, less_equal,
greater, greater_equal, returning bool tensors with same shape as inputs.
"""

from sys import DType

# Import ExTensor and operations
from extensor import ExTensor, zeros, ones, full, arange

# Import test helpers
from ..helpers.assertions import (
    assert_dtype,
    assert_numel,
    assert_dim,
    assert_value_at,
    assert_all_values,
)


# ============================================================================
# Test equal()
# ============================================================================

fn test_equal_same_values() raises:
    """Test equal with tensors containing same values."""
    var shape = DynamicVector[Int](1)
    shape[0] = 5
    let a = full(shape, 3.0, DType.float32)
    let b = full(shape, 3.0, DType.float32)
    # let c = equal(a, b)  # TODO: Implement equal()

    # assert_dtype(c, DType.bool, "Comparison should return bool tensor")
    # assert_all_values(c, 1.0, 1e-8, "All elements should be True (1.0)")
    pass  # Placeholder until equal() is implemented


fn test_equal_different_values() raises:
    """Test equal with tensors containing different values."""
    var shape = DynamicVector[Int](1)
    shape[0] = 5
    let a = full(shape, 3.0, DType.float32)
    let b = full(shape, 5.0, DType.float32)
    # let c = equal(a, b)  # TODO: Implement equal()

    # assert_dtype(c, DType.bool, "Comparison should return bool tensor")
    # assert_all_values(c, 0.0, 1e-8, "All elements should be False (0.0)")
    pass  # Placeholder


fn test_equal_partial_match() raises:
    """Test equal with partially matching values."""
    let a = arange(0.0, 5.0, 1.0, DType.float32)  # [0, 1, 2, 3, 4]
    let b = arange(0.0, 5.0, 1.0, DType.float32)  # [0, 1, 2, 3, 4]
    # let c = equal(a, b)  # TODO: Implement equal()

    # All should be True since arrays are identical
    # assert_all_values(c, 1.0, 1e-8, "Identical arrays should be all True")
    pass  # Placeholder


fn test_equal_nan_behavior() raises:
    """Test equal with NaN values (NaN != NaN in IEEE 754)."""
    # TODO: Test NaN comparison behavior
    # var shape = DynamicVector[Int](1)
    # shape[0] = 3
    # let a = full(shape, Float32.nan, DType.float32)
    # let b = full(shape, Float32.nan, DType.float32)
    # let c = equal(a, b)

    # IEEE 754: NaN != NaN, so result should be False
    # assert_all_values(c, 0.0, 1e-8, "NaN != NaN per IEEE 754")
    pass  # Placeholder


# ============================================================================
# Test not_equal()
# ============================================================================

fn test_not_equal_same_values() raises:
    """Test not_equal with tensors containing same values."""
    var shape = DynamicVector[Int](1)
    shape[0] = 5
    let a = full(shape, 3.0, DType.float32)
    let b = full(shape, 3.0, DType.float32)
    # let c = not_equal(a, b)  # TODO: Implement not_equal()

    # assert_all_values(c, 0.0, 1e-8, "Same values should give False")
    pass  # Placeholder


fn test_not_equal_different_values() raises:
    """Test not_equal with tensors containing different values."""
    var shape = DynamicVector[Int](1)
    shape[0] = 5
    let a = full(shape, 3.0, DType.float32)
    let b = full(shape, 5.0, DType.float32)
    # let c = not_equal(a, b)  # TODO: Implement not_equal()

    # assert_all_values(c, 1.0, 1e-8, "Different values should give True")
    pass  # Placeholder


# ============================================================================
# Test less()
# ============================================================================

fn test_less_all_true() raises:
    """Test less with a < b for all elements."""
    var shape = DynamicVector[Int](1)
    shape[0] = 5
    let a = full(shape, 2.0, DType.float32)
    let b = full(shape, 5.0, DType.float32)
    # let c = less(a, b)  # TODO: Implement less()

    # assert_all_values(c, 1.0, 1e-8, "2 < 5 should be True")
    pass  # Placeholder


fn test_less_all_false() raises:
    """Test less with a >= b for all elements."""
    var shape = DynamicVector[Int](1)
    shape[0] = 5
    let a = full(shape, 5.0, DType.float32)
    let b = full(shape, 2.0, DType.float32)
    # let c = less(a, b)  # TODO: Implement less()

    # assert_all_values(c, 0.0, 1e-8, "5 < 2 should be False")
    pass  # Placeholder


fn test_less_mixed() raises:
    """Test less with mixed comparisons."""
    let a = arange(0.0, 5.0, 1.0, DType.float32)  # [0, 1, 2, 3, 4]
    var shape = DynamicVector[Int](1)
    shape[0] = 5
    let b = full(shape, 2.5, DType.float32)  # [2.5, 2.5, 2.5, 2.5, 2.5]
    # let c = less(a, b)  # TODO: Implement less()

    # Expected: [True, True, True, False, False] = [1, 1, 1, 0, 0]
    # assert_value_at(c, 0, 1.0, 1e-8, "0 < 2.5 is True")
    # assert_value_at(c, 3, 0.0, 1e-8, "3 < 2.5 is False")
    pass  # Placeholder


# ============================================================================
# Test less_equal()
# ============================================================================

fn test_less_equal_with_equal_values() raises:
    """Test less_equal with equal values."""
    var shape = DynamicVector[Int](1)
    shape[0] = 5
    let a = full(shape, 3.0, DType.float32)
    let b = full(shape, 3.0, DType.float32)
    # let c = less_equal(a, b)  # TODO: Implement less_equal()

    # assert_all_values(c, 1.0, 1e-8, "3 <= 3 should be True")
    pass  # Placeholder


fn test_less_equal_mixed() raises:
    """Test less_equal with mixed comparisons."""
    let a = arange(0.0, 5.0, 1.0, DType.float32)  # [0, 1, 2, 3, 4]
    var shape = DynamicVector[Int](1)
    shape[0] = 5
    let b = full(shape, 2.0, DType.float32)  # [2, 2, 2, 2, 2]
    # let c = less_equal(a, b)  # TODO: Implement less_equal()

    # Expected: [True, True, True, False, False]
    # assert_value_at(c, 0, 1.0, 1e-8, "0 <= 2 is True")
    # assert_value_at(c, 2, 1.0, 1e-8, "2 <= 2 is True")
    # assert_value_at(c, 3, 0.0, 1e-8, "3 <= 2 is False")
    pass  # Placeholder


# ============================================================================
# Test greater()
# ============================================================================

fn test_greater_all_true() raises:
    """Test greater with a > b for all elements."""
    var shape = DynamicVector[Int](1)
    shape[0] = 5
    let a = full(shape, 5.0, DType.float32)
    let b = full(shape, 2.0, DType.float32)
    # let c = greater(a, b)  # TODO: Implement greater()

    # assert_all_values(c, 1.0, 1e-8, "5 > 2 should be True")
    pass  # Placeholder


fn test_greater_all_false() raises:
    """Test greater with a <= b for all elements."""
    var shape = DynamicVector[Int](1)
    shape[0] = 5
    let a = full(shape, 2.0, DType.float32)
    let b = full(shape, 5.0, DType.float32)
    # let c = greater(a, b)  # TODO: Implement greater()

    # assert_all_values(c, 0.0, 1e-8, "2 > 5 should be False")
    pass  # Placeholder


# ============================================================================
# Test greater_equal()
# ============================================================================

fn test_greater_equal_with_equal_values() raises:
    """Test greater_equal with equal values."""
    var shape = DynamicVector[Int](1)
    shape[0] = 5
    let a = full(shape, 3.0, DType.float32)
    let b = full(shape, 3.0, DType.float32)
    # let c = greater_equal(a, b)  # TODO: Implement greater_equal()

    # assert_all_values(c, 1.0, 1e-8, "3 >= 3 should be True")
    pass  # Placeholder


# ============================================================================
# Test dunder methods
# ============================================================================

fn test_dunder_eq() raises:
    """Test __eq__ operator overloading (a == b)."""
    var shape = DynamicVector[Int](1)
    shape[0] = 5
    let a = full(shape, 3.0, DType.float32)
    let b = full(shape, 3.0, DType.float32)
    # let c = a == b  # TODO: Implement __eq__

    # assert_all_values(c, 1.0, 1e-8, "a == b should work via __eq__")
    pass  # Placeholder


fn test_dunder_ne() raises:
    """Test __ne__ operator overloading (a != b)."""
    var shape = DynamicVector[Int](1)
    shape[0] = 5
    let a = full(shape, 3.0, DType.float32)
    let b = full(shape, 5.0, DType.float32)
    # let c = a != b  # TODO: Implement __ne__

    # assert_all_values(c, 1.0, 1e-8, "a != b should work via __ne__")
    pass  # Placeholder


fn test_dunder_lt() raises:
    """Test __lt__ operator overloading (a < b)."""
    var shape = DynamicVector[Int](1)
    shape[0] = 5
    let a = full(shape, 2.0, DType.float32)
    let b = full(shape, 5.0, DType.float32)
    # let c = a < b  # TODO: Implement __lt__

    # assert_all_values(c, 1.0, 1e-8, "a < b should work via __lt__")
    pass  # Placeholder


fn test_dunder_le() raises:
    """Test __le__ operator overloading (a <= b)."""
    var shape = DynamicVector[Int](1)
    shape[0] = 5
    let a = full(shape, 3.0, DType.float32)
    let b = full(shape, 3.0, DType.float32)
    # let c = a <= b  # TODO: Implement __le__

    # assert_all_values(c, 1.0, 1e-8, "a <= b should work via __le__")
    pass  # Placeholder


fn test_dunder_gt() raises:
    """Test __gt__ operator overloading (a > b)."""
    var shape = DynamicVector[Int](1)
    shape[0] = 5
    let a = full(shape, 5.0, DType.float32)
    let b = full(shape, 2.0, DType.float32)
    # let c = a > b  # TODO: Implement __gt__

    # assert_all_values(c, 1.0, 1e-8, "a > b should work via __gt__")
    pass  # Placeholder


fn test_dunder_ge() raises:
    """Test __ge__ operator overloading (a >= b)."""
    var shape = DynamicVector[Int](1)
    shape[0] = 5
    let a = full(shape, 3.0, DType.float32)
    let b = full(shape, 3.0, DType.float32)
    # let c = a >= b  # TODO: Implement __ge__

    # assert_all_values(c, 1.0, 1e-8, "a >= b should work via __ge__")
    pass  # Placeholder


# ============================================================================
# Test special values
# ============================================================================

fn test_comparison_with_inf() raises:
    """Test comparison with infinity values."""
    # TODO: Test inf comparisons
    # var shape = DynamicVector[Int](1)
    # shape[0] = 3
    # let a = full(shape, Float32.infinity, DType.float32)
    # let b = full(shape, 100.0, DType.float32)
    # let c = greater(a, b)

    # assert_all_values(c, 1.0, 1e-8, "inf > 100 should be True")
    pass  # Placeholder


fn test_comparison_with_negative_inf() raises:
    """Test comparison with negative infinity."""
    # TODO: Test -inf comparisons
    pass  # Placeholder


# ============================================================================
# Test broadcasting in comparisons
# ============================================================================

fn test_comparison_broadcast_scalar() raises:
    """Test comparison with scalar broadcasting."""
    var shape_vec = DynamicVector[Int](1)
    shape_vec[0] = 5
    var shape_scalar = DynamicVector[Int](0)

    let a = arange(0.0, 5.0, 1.0, DType.float32)  # [0, 1, 2, 3, 4]
    let b = full(shape_scalar, 2.0, DType.float32)  # scalar 2
    # let c = less(a, b)  # TODO: Implement less() with broadcasting

    # Expected: [True, True, False, False, False]
    # assert_value_at(c, 0, 1.0, 1e-8, "0 < 2 is True")
    # assert_value_at(c, 2, 0.0, 1e-8, "2 < 2 is False")
    pass  # Placeholder


fn test_comparison_broadcast_1d_to_2d() raises:
    """Test comparison with 1D to 2D broadcasting."""
    var shape_2d = DynamicVector[Int](2)
    shape_2d[0] = 3
    shape_2d[1] = 4
    var shape_1d = DynamicVector[Int](1)
    shape_1d[0] = 4

    let a = ones(shape_2d, DType.float32)  # 3x4 matrix of ones
    let b = full(shape_1d, 2.0, DType.float32)  # 4-element vector
    # let c = less(a, b)  # TODO: Implement less() with broadcasting

    # All should be True (1 < 2)
    # assert_all_values(c, 1.0, 1e-8, "1 < 2 should broadcast to all True")
    pass  # Placeholder


# ============================================================================
# Main test runner
# ============================================================================

fn main() raises:
    """Run all comparison operation tests."""
    print("Running ExTensor comparison operation tests...")

    # equal() tests
    print("  Testing equal()...")
    test_equal_same_values()
    test_equal_different_values()
    test_equal_partial_match()
    test_equal_nan_behavior()

    # not_equal() tests
    print("  Testing not_equal()...")
    test_not_equal_same_values()
    test_not_equal_different_values()

    # less() tests
    print("  Testing less()...")
    test_less_all_true()
    test_less_all_false()
    test_less_mixed()

    # less_equal() tests
    print("  Testing less_equal()...")
    test_less_equal_with_equal_values()
    test_less_equal_mixed()

    # greater() tests
    print("  Testing greater()...")
    test_greater_all_true()
    test_greater_all_false()

    # greater_equal() tests
    print("  Testing greater_equal()...")
    test_greater_equal_with_equal_values()

    # Dunder methods
    print("  Testing comparison dunders...")
    test_dunder_eq()
    test_dunder_ne()
    test_dunder_lt()
    test_dunder_le()
    test_dunder_gt()
    test_dunder_ge()

    # Special values
    print("  Testing special values...")
    test_comparison_with_inf()
    test_comparison_with_negative_inf()

    # Broadcasting
    print("  Testing comparison with broadcasting...")
    test_comparison_broadcast_scalar()
    test_comparison_broadcast_1d_to_2d()

    print("All comparison operation tests completed!")
