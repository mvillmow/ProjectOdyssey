"""Tests for ExTensor reduction operations.

Tests reduction operations including sum, prod, mean, var, std, max, min,
argmax, argmin with axis reduction, keepdims, and edge cases.
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
    assert_equal_float,
)


# ============================================================================
# Test sum() - Reduce all elements
# ============================================================================

fn test_sum_all_elements_1d() raises:
    """Test summing all elements in 1D tensor."""
    let t = arange(1.0, 6.0, 1.0, DType.float32)  # [1, 2, 3, 4, 5]
    # let result = sum(t)  # TODO: Implement sum()

    # Expected: 1 + 2 + 3 + 4 + 5 = 15
    # assert_numel(result, 1, "Sum all should return scalar")
    # assert_value_at(result, 0, 15.0, 1e-6, "Sum of [1,2,3,4,5] should be 15")
    pass  # Placeholder


fn test_sum_all_elements_2d() raises:
    """Test summing all elements in 2D tensor."""
    var shape = DynamicVector[Int](2)
    shape[0] = 3
    shape[1] = 4
    let t = full(shape, 2.0, DType.float32)  # 3x4 matrix of 2s
    # let result = sum(t)  # TODO: Implement sum()

    # Expected: 2 * 12 = 24
    # assert_value_at(result, 0, 24.0, 1e-6, "Sum of 12 twos should be 24")
    pass  # Placeholder


fn test_sum_zeros() raises:
    """Test summing zeros."""
    var shape = DynamicVector[Int](1)
    shape[0] = 10
    let t = zeros(shape, DType.float32)
    # let result = sum(t)  # TODO: Implement sum()

    # assert_value_at(result, 0, 0.0, 1e-8, "Sum of zeros should be 0")
    pass  # Placeholder


# ============================================================================
# Test sum() - Axis reduction
# ============================================================================

fn test_sum_along_axis_0() raises:
    """Test sum along axis 0 (reduce rows)."""
    var shape = DynamicVector[Int](2)
    shape[0] = 3
    shape[1] = 4
    let t = ones(shape, DType.float32)  # 3x4 matrix of ones
    # let result = sum(t, axis=0)  # TODO: Implement axis parameter

    # Expected: sum along rows -> [3, 3, 3, 3] (4 elements)
    # assert_numel(result, 4, "Sum along axis 0 should give 4 elements")
    # assert_all_values(result, 3.0, 1e-6, "Each column sum should be 3")
    pass  # Placeholder


fn test_sum_along_axis_1() raises:
    """Test sum along axis 1 (reduce columns)."""
    var shape = DynamicVector[Int](2)
    shape[0] = 3
    shape[1] = 4
    let t = ones(shape, DType.float32)  # 3x4 matrix of ones
    # let result = sum(t, axis=1)  # TODO: Implement axis parameter

    # Expected: sum along columns -> [4, 4, 4] (3 elements)
    # assert_numel(result, 3, "Sum along axis 1 should give 3 elements")
    # assert_all_values(result, 4.0, 1e-6, "Each row sum should be 4")
    pass  # Placeholder


fn test_sum_along_axis_keepdims() raises:
    """Test sum with keepdims=True."""
    var shape = DynamicVector[Int](2)
    shape[0] = 3
    shape[1] = 4
    let t = ones(shape, DType.float32)
    # let result = sum(t, axis=0, keepdims=True)  # TODO: Implement keepdims

    # Expected: shape (1, 4) instead of (4,)
    # assert_dim(result, 2, "keepdims should preserve number of dimensions")
    # assert_numel(result, 4, "Should still have 4 elements")
    pass  # Placeholder


# ============================================================================
# Test mean()
# ============================================================================

fn test_mean_all_elements() raises:
    """Test computing mean of all elements."""
    let t = arange(1.0, 6.0, 1.0, DType.float32)  # [1, 2, 3, 4, 5]
    # let result = mean(t)  # TODO: Implement mean()

    # Expected: (1 + 2 + 3 + 4 + 5) / 5 = 3
    # assert_value_at(result, 0, 3.0, 1e-6, "Mean of [1,2,3,4,5] should be 3")
    pass  # Placeholder


fn test_mean_along_axis() raises:
    """Test computing mean along specific axis."""
    var shape = DynamicVector[Int](2)
    shape[0] = 2
    shape[1] = 4
    let t = full(shape, 6.0, DType.float32)  # 2x4 matrix of 6s
    # let result = mean(t, axis=1)  # TODO: Implement axis parameter

    # Expected: mean of each row -> [6, 6]
    # assert_all_values(result, 6.0, 1e-6, "Mean of constant row should be constant")
    pass  # Placeholder


fn test_mean_accuracy() raises:
    """Test mean numerical accuracy."""
    var shape = DynamicVector[Int](1)
    shape[0] = 1000
    let t = ones(shape, DType.float64)
    # let result = mean(t)  # TODO: Implement mean()

    # Mean of 1000 ones should be exactly 1.0
    # assert_value_at(result, 0, 1.0, 1e-12, "Mean should be numerically accurate")
    pass  # Placeholder


# ============================================================================
# Test var() and std()
# ============================================================================

fn test_var_all_elements() raises:
    """Test computing variance of all elements."""
    let t = arange(0.0, 5.0, 1.0, DType.float64)  # [0, 1, 2, 3, 4]
    # let result = var(t)  # TODO: Implement var()

    # Variance of [0,1,2,3,4] = 2.0 (population variance)
    # assert_value_at(result, 0, 2.0, 1e-8, "Variance calculation")
    pass  # Placeholder


fn test_var_ddof() raises:
    """Test variance with degrees of freedom parameter."""
    let t = arange(0.0, 5.0, 1.0, DType.float64)  # [0, 1, 2, 3, 4]
    # let result = var(t, ddof=1)  # TODO: Implement ddof parameter

    # Sample variance with ddof=1: 2.5
    # assert_value_at(result, 0, 2.5, 1e-8, "Sample variance (ddof=1)")
    pass  # Placeholder


fn test_std_numerical_stability() raises:
    """Test standard deviation with numerically stable algorithm."""
    var shape = DynamicVector[Int](1)
    shape[0] = 1000
    let t = full(shape, 1e6, DType.float64)  # Large values
    # let result = std(t)  # TODO: Implement std()

    # Std of constant should be 0, even for large values
    # assert_value_at(result, 0, 0.0, 1e-6, "Std of constant should be 0")
    pass  # Placeholder


# ============================================================================
# Test max() and min()
# ============================================================================

fn test_max_all_elements() raises:
    """Test finding maximum of all elements."""
    let t = arange(0.0, 10.0, 1.0, DType.float32)  # [0, 1, ..., 9]
    # let result = max_reduce(t)  # TODO: Implement max_reduce()

    # assert_value_at(result, 0, 9.0, 1e-6, "Max of [0..9] should be 9")
    pass  # Placeholder


fn test_min_all_elements() raises:
    """Test finding minimum of all elements."""
    let t = arange(1.0, 11.0, 1.0, DType.float32)  # [1, 2, ..., 10]
    # let result = min_reduce(t)  # TODO: Implement min_reduce()

    # assert_value_at(result, 0, 1.0, 1e-6, "Min of [1..10] should be 1")
    pass  # Placeholder


fn test_max_along_axis() raises:
    """Test max along specific axis."""
    var shape = DynamicVector[Int](2)
    shape[0] = 3
    shape[1] = 4
    # Create matrix: [[1,2,3,4], [5,6,7,8], [9,10,11,12]]
    # let t = ...  # TODO: Create appropriate test tensor
    # let result = max_reduce(t, axis=0)  # TODO: Implement axis parameter

    # Expected: max along rows -> [9, 10, 11, 12]
    pass  # Placeholder


fn test_max_nan_behavior() raises:
    """Test max behavior with NaN values."""
    # TODO: Test NaN handling in max
    # Different behaviors possible:
    # - Propagate NaN (return NaN if any input is NaN)
    # - Ignore NaN (return max of non-NaN values)
    pass  # Placeholder


# ============================================================================
# Test argmax() and argmin()
# ============================================================================

fn test_argmax_1d() raises:
    """Test argmax on 1D tensor."""
    let t = arange(0.0, 5.0, 1.0, DType.float32)  # [0, 1, 2, 3, 4]
    # let result = argmax(t)  # TODO: Implement argmax()

    # assert_value_at(result, 0, 4.0, 1e-6, "Index of max (4) should be 4")
    pass  # Placeholder


fn test_argmin_1d() raises:
    """Test argmin on 1D tensor."""
    let t = arange(5.0, 0.0, -1.0, DType.float32)  # [5, 4, 3, 2, 1]
    # let result = argmin(t)  # TODO: Implement argmin()

    # assert_value_at(result, 0, 4.0, 1e-6, "Index of min (1) should be 4")
    pass  # Placeholder


fn test_argmax_ties() raises:
    """Test argmax behavior with tied values (should return first index)."""
    var shape = DynamicVector[Int](1)
    shape[0] = 5
    let t = full(shape, 5.0, DType.float32)  # All same value
    # let result = argmax(t)  # TODO: Implement argmax()

    # Convention: return first index when tied
    # assert_value_at(result, 0, 0.0, 1e-6, "Tied max should return first index")
    pass  # Placeholder


# ============================================================================
# Test cumulative operations
# ============================================================================

fn test_cumulative_sum_1d() raises:
    """Test cumulative sum on 1D tensor."""
    let t = arange(1.0, 6.0, 1.0, DType.float32)  # [1, 2, 3, 4, 5]
    # let result = cumulative_sum(t)  # TODO: Implement cumulative_sum()

    # Expected: [1, 3, 6, 10, 15]
    # assert_value_at(result, 0, 1.0, 1e-6, "cumsum[0]")
    # assert_value_at(result, 2, 6.0, 1e-6, "cumsum[2]")
    # assert_value_at(result, 4, 15.0, 1e-6, "cumsum[4]")
    pass  # Placeholder


fn test_cumulative_sum_axis() raises:
    """Test cumulative sum along specific axis."""
    var shape = DynamicVector[Int](2)
    shape[0] = 2
    shape[1] = 3
    let t = ones(shape, DType.float32)  # 2x3 matrix of ones
    # let result = cumulative_sum(t, axis=1)  # TODO: Implement axis parameter

    # Expected along axis 1: [[1, 2, 3], [1, 2, 3]]
    pass  # Placeholder


fn test_cumulative_prod() raises:
    """Test cumulative product."""
    let t = arange(1.0, 6.0, 1.0, DType.float32)  # [1, 2, 3, 4, 5]
    # let result = cumulative_prod(t)  # TODO: Implement cumulative_prod()

    # Expected: [1, 2, 6, 24, 120]
    # assert_value_at(result, 0, 1.0, 1e-6, "cumprod[0]")
    # assert_value_at(result, 3, 24.0, 1e-4, "cumprod[3]")
    # assert_value_at(result, 4, 120.0, 1e-3, "cumprod[4]")
    pass  # Placeholder


# ============================================================================
# Test prod()
# ============================================================================

fn test_prod_all_elements() raises:
    """Test product of all elements."""
    let t = arange(1.0, 6.0, 1.0, DType.float32)  # [1, 2, 3, 4, 5]
    # let result = prod(t)  # TODO: Implement prod()

    # Expected: 1 * 2 * 3 * 4 * 5 = 120
    # assert_value_at(result, 0, 120.0, 1e-4, "Product of [1,2,3,4,5] should be 120")
    pass  # Placeholder


fn test_prod_with_zero() raises:
    """Test product with zero element."""
    var shape = DynamicVector[Int](1)
    shape[0] = 5
    let t = full(shape, 5.0, DType.float32)
    # Set one element to 0
    # t[2] = 0.0  # TODO: Implement __setitem__
    # let result = prod(t)  # TODO: Implement prod()

    # assert_value_at(result, 0, 0.0, 1e-8, "Product with zero should be 0")
    pass  # Placeholder


# ============================================================================
# Test all() and any()
# ============================================================================

fn test_all_true() raises:
    """Test all() with all True values."""
    var shape = DynamicVector[Int](1)
    shape[0] = 5
    let t = ones(shape, DType.bool)
    # let result = all(t)  # TODO: Implement all()

    # assert_value_at(result, 0, 1.0, 1e-8, "all() of all True should be True")
    pass  # Placeholder


fn test_all_with_false() raises:
    """Test all() with at least one False."""
    var shape = DynamicVector[Int](1)
    shape[0] = 5
    let t = ones(shape, DType.bool)
    # t[2] = False  # TODO: Implement __setitem__
    # let result = all(t)  # TODO: Implement all()

    # assert_value_at(result, 0, 0.0, 1e-8, "all() with one False should be False")
    pass  # Placeholder


fn test_all_short_circuit() raises:
    """Test that all() can short-circuit on first False."""
    # TODO: Test short-circuit behavior (performance test)
    pass  # Placeholder


fn test_any_with_true() raises:
    """Test any() with at least one True."""
    var shape = DynamicVector[Int](1)
    shape[0] = 5
    let t = zeros(shape, DType.bool)
    # t[2] = True  # TODO: Implement __setitem__
    # let result = any(t)  # TODO: Implement any()

    # assert_value_at(result, 0, 1.0, 1e-8, "any() with one True should be True")
    pass  # Placeholder


fn test_any_all_false() raises:
    """Test any() with all False values."""
    var shape = DynamicVector[Int](1)
    shape[0] = 5
    let t = zeros(shape, DType.bool)
    # let result = any(t)  # TODO: Implement any()

    # assert_value_at(result, 0, 0.0, 1e-8, "any() of all False should be False")
    pass  # Placeholder


fn test_any_empty_tensor() raises:
    """Test any() on empty tensor."""
    var shape = DynamicVector[Int](1)
    shape[0] = 0
    let t = zeros(shape, DType.bool)
    # let result = any(t)  # TODO: Implement any()

    # Convention: any() of empty is False
    # assert_value_at(result, 0, 0.0, 1e-8, "any() of empty should be False")
    pass  # Placeholder


# ============================================================================
# Test count_nonzero()
# ============================================================================

fn test_count_nonzero() raises:
    """Test counting non-zero elements."""
    let t = arange(0.0, 5.0, 1.0, DType.float32)  # [0, 1, 2, 3, 4]
    # let result = count_nonzero(t)  # TODO: Implement count_nonzero()

    # Expected: 4 (all except 0)
    # assert_value_at(result, 0, 4.0, 1e-6, "Should count 4 non-zero elements")
    pass  # Placeholder


# ============================================================================
# Test edge cases
# ============================================================================

fn test_sum_empty_tensor() raises:
    """Test sum of empty tensor."""
    var shape = DynamicVector[Int](1)
    shape[0] = 0
    let t = zeros(shape, DType.float32)
    # let result = sum(t)  # TODO: Implement sum()

    # Convention: sum of empty is 0
    # assert_value_at(result, 0, 0.0, 1e-8, "Sum of empty should be 0")
    pass  # Placeholder


fn test_mean_single_element() raises:
    """Test mean of single element."""
    var shape = DynamicVector[Int](1)
    shape[0] = 1
    let t = full(shape, 42.0, DType.float32)
    # let result = mean(t)  # TODO: Implement mean()

    # assert_value_at(result, 0, 42.0, 1e-6, "Mean of single element is itself")
    pass  # Placeholder


# ============================================================================
# Main test runner
# ============================================================================

fn main() raises:
    """Run all reduction operation tests."""
    print("Running ExTensor reduction operation tests...")

    # sum() tests
    print("  Testing sum()...")
    test_sum_all_elements_1d()
    test_sum_all_elements_2d()
    test_sum_zeros()
    test_sum_along_axis_0()
    test_sum_along_axis_1()
    test_sum_along_axis_keepdims()

    # mean() tests
    print("  Testing mean()...")
    test_mean_all_elements()
    test_mean_along_axis()
    test_mean_accuracy()

    # var() and std() tests
    print("  Testing var() and std()...")
    test_var_all_elements()
    test_var_ddof()
    test_std_numerical_stability()

    # max() and min() tests
    print("  Testing max() and min()...")
    test_max_all_elements()
    test_min_all_elements()
    test_max_along_axis()
    test_max_nan_behavior()

    # argmax() and argmin() tests
    print("  Testing argmax() and argmin()...")
    test_argmax_1d()
    test_argmin_1d()
    test_argmax_ties()

    # Cumulative operations
    print("  Testing cumulative operations...")
    test_cumulative_sum_1d()
    test_cumulative_sum_axis()
    test_cumulative_prod()

    # prod() tests
    print("  Testing prod()...")
    test_prod_all_elements()
    test_prod_with_zero()

    # all() and any() tests
    print("  Testing all() and any()...")
    test_all_true()
    test_all_with_false()
    test_all_short_circuit()
    test_any_with_true()
    test_any_all_false()
    test_any_empty_tensor()

    # count_nonzero() tests
    print("  Testing count_nonzero()...")
    test_count_nonzero()

    # Edge cases
    print("  Testing edge cases...")
    test_sum_empty_tensor()
    test_mean_single_element()

    print("All reduction operation tests completed!")
