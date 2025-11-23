"""Tests for ExTensor reduction operations.

Tests reduction operations following the Array API Standard:
sum, mean, max_reduce, min_reduce with all-elements reduction (axis=-1).
"""

from memory import DType

# Import ExTensor and reduction operations
from shared.core import ExTensor, full, ones, zeros, arange, sum, mean, max_reduce, min_reduce

# Import test helpers
from ..helpers.assertions import (
    assert_dtype,
    assert_numel,
    assert_dim,
    assert_value_at,
)


# ============================================================================
# Test sum()
# ============================================================================

fn test_sum_all_ones() raises:
    """Test sum of all ones."""
    var shape = List[Int]()
    shape.append(10)
    vara = ones(shape, DType.float32)
    varb = sum(a)  # Sum all elements

    assert_dim(b, 0, "Sum should return scalar (0D)")
    assert_numel(b, 1, "Scalar should have 1 element")
    assert_value_at(b, 0, 10.0, 1e-6, "Sum of 10 ones should be 10.0")


fn test_sum_2d_tensor() raises:
    """Test sum of 2D tensor."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    vara = full(shape, 2.0, DType.float32)
    varb = sum(a)  # Sum all 12 elements

    assert_dim(b, 0, "Sum should return scalar")
    assert_value_at(b, 0, 24.0, 1e-6, "Sum of 12 twos should be 24.0")


fn test_sum_arange() raises:
    """Test sum of range [0, 1, 2, 3, 4]."""
    vara = arange(0.0, 5.0, 1.0, DType.float32)
    varb = sum(a)

    # 0 + 1 + 2 + 3 + 4 = 10
    assert_value_at(b, 0, 10.0, 1e-6, "Sum of [0,1,2,3,4] should be 10.0")


fn test_sum_with_keepdims() raises:
    """Test sum with keepdims=True."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    vara = ones(shape, DType.float32)
    varb = sum(a, keepdims=True)

    # Should be (1, 1) shape instead of scalar
    assert_dim(b, 2, "keepdims should preserve dimensions")
    assert_value_at(b, 0, 12.0, 1e-6, "Sum should still be 12.0")


fn test_sum_preserves_dtype() raises:
    """Test that sum preserves dtype."""
    var shape = List[Int]()
    shape.append(5)
    vara = ones(shape, DType.float64)
    varb = sum(a)

    assert_dtype(b, DType.float64, "Sum should preserve float64 dtype")
    assert_value_at(b, 0, 5.0, 1e-10, "Sum of 5 ones should be 5.0")


# ============================================================================
# Test mean()
# ============================================================================

fn test_mean_all_ones() raises:
    """Test mean of all ones."""
    var shape = List[Int]()
    shape.append(10)
    vara = ones(shape, DType.float32)
    varb = mean(a)

    assert_dim(b, 0, "Mean should return scalar")
    assert_value_at(b, 0, 1.0, 1e-6, "Mean of ones should be 1.0")


fn test_mean_2d_tensor() raises:
    """Test mean of 2D tensor."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    vara = full(shape, 5.0, DType.float32)
    varb = mean(a)

    assert_dim(b, 0, "Mean should return scalar")
    assert_value_at(b, 0, 5.0, 1e-6, "Mean of all 5s should be 5.0")


fn test_mean_arange() raises:
    """Test mean of range [0, 1, 2, 3, 4]."""
    vara = arange(0.0, 5.0, 1.0, DType.float32)
    varb = mean(a)

    # (0 + 1 + 2 + 3 + 4) / 5 = 10 / 5 = 2.0
    assert_value_at(b, 0, 2.0, 1e-6, "Mean of [0,1,2,3,4] should be 2.0")


fn test_mean_with_keepdims() raises:
    """Test mean with keepdims=True."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    vara = full(shape, 6.0, DType.float32)
    varb = mean(a, keepdims=True)

    # Should be (1, 1) shape instead of scalar
    assert_dim(b, 2, "keepdims should preserve dimensions")
    assert_value_at(b, 0, 6.0, 1e-6, "Mean should be 6.0")


fn test_mean_preserves_dtype() raises:
    """Test that mean preserves dtype."""
    var shape = List[Int]()
    shape.append(4)
    vara = full(shape, 8.0, DType.float64)
    varb = mean(a)

    assert_dtype(b, DType.float64, "Mean should preserve float64 dtype")
    assert_value_at(b, 0, 8.0, 1e-10, "Mean of all 8s should be 8.0")


# ============================================================================
# Test max_reduce()
# ============================================================================

fn test_max_all_same() raises:
    """Test max of all same values."""
    var shape = List[Int]()
    shape.append(10)
    vara = full(shape, 7.0, DType.float32)
    varb = max_reduce(a)

    assert_dim(b, 0, "Max should return scalar")
    assert_value_at(b, 0, 7.0, 1e-6, "Max of all 7s should be 7.0")


fn test_max_arange() raises:
    """Test max of range [0, 1, 2, 3, 4]."""
    vara = arange(0.0, 5.0, 1.0, DType.float32)
    varb = max_reduce(a)

    assert_value_at(b, 0, 4.0, 1e-6, "Max of [0,1,2,3,4] should be 4.0")


fn test_max_negative_values() raises:
    """Test max with negative values."""
    var shape = List[Int]()
    shape.append(5)
    vara = full(shape, -3.0, DType.float32)
    varb = max_reduce(a)

    assert_value_at(b, 0, -3.0, 1e-6, "Max of all -3s should be -3.0")


fn test_max_with_keepdims() raises:
    """Test max with keepdims=True."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    vara = arange(0.0, 12.0, 1.0, DType.float32)
    # Note: arange creates 1D, would need reshape for 2D, but keepdims test still valid
    var shape2d = List[Int]()
    shape2d.append(3)
    shape2d.append(4)
    vara2d = full(shape2d, 9.0, DType.float32)
    varb = max_reduce(a2d, keepdims=True)

    assert_dim(b, 2, "keepdims should preserve dimensions")
    assert_value_at(b, 0, 9.0, 1e-6, "Max should be 9.0")


fn test_max_preserves_dtype() raises:
    """Test that max preserves dtype."""
    var shape = List[Int]()
    shape.append(5)
    vara = arange(0.0, 5.0, 1.0, DType.float64)
    varb = max_reduce(a)

    assert_dtype(b, DType.float64, "Max should preserve float64 dtype")
    assert_value_at(b, 0, 4.0, 1e-10, "Max should be 4.0")


# ============================================================================
# Test min_reduce()
# ============================================================================

fn test_min_all_same() raises:
    """Test min of all same values."""
    var shape = List[Int]()
    shape.append(10)
    vara = full(shape, 3.0, DType.float32)
    varb = min_reduce(a)

    assert_dim(b, 0, "Min should return scalar")
    assert_value_at(b, 0, 3.0, 1e-6, "Min of all 3s should be 3.0")


fn test_min_arange() raises:
    """Test min of range [0, 1, 2, 3, 4]."""
    vara = arange(0.0, 5.0, 1.0, DType.float32)
    varb = min_reduce(a)

    assert_value_at(b, 0, 0.0, 1e-6, "Min of [0,1,2,3,4] should be 0.0")


fn test_min_negative_values() raises:
    """Test min with negative values."""
    var shape = List[Int]()
    shape.append(5)
    vara = full(shape, -7.0, DType.float32)
    varb = min_reduce(a)

    assert_value_at(b, 0, -7.0, 1e-6, "Min of all -7s should be -7.0")


fn test_min_with_keepdims() raises:
    """Test min with keepdims=True."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    vara = full(shape, 2.5, DType.float32)
    varb = min_reduce(a, keepdims=True)

    assert_dim(b, 2, "keepdims should preserve dimensions")
    assert_value_at(b, 0, 2.5, 1e-6, "Min should be 2.5")


fn test_min_preserves_dtype() raises:
    """Test that min preserves dtype."""
    var shape = List[Int]()
    shape.append(5)
    vara = arange(1.0, 6.0, 1.0, DType.float64)
    varb = min_reduce(a)

    assert_dtype(b, DType.float64, "Min should preserve float64 dtype")
    assert_value_at(b, 0, 1.0, 1e-10, "Min should be 1.0")


# ============================================================================
# Test axis-specific reductions
# ============================================================================

fn test_sum_axis_0() raises:
    """Test sum along axis 0."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    vara = full(shape, 2.0, DType.float32)  # 3x4 matrix of 2s
    varb = sum(a, axis=0)  # Sum along rows -> shape (4,)

    # Should sum 3 values (each 2.0) per column
    assert_dim(b, 1, "Sum along axis 0 should be 1D")
    assert_numel(b, 4, "Sum along axis 0 should have 4 elements")
    assert_value_at(b, 0, 6.0, 1e-6, "Each column sum should be 6.0")
    assert_value_at(b, 1, 6.0, 1e-6, "Each column sum should be 6.0")
    assert_value_at(b, 2, 6.0, 1e-6, "Each column sum should be 6.0")
    assert_value_at(b, 3, 6.0, 1e-6, "Each column sum should be 6.0")


fn test_sum_axis_1() raises:
    """Test sum along axis 1."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    vara = full(shape, 2.0, DType.float32)  # 3x4 matrix of 2s
    varb = sum(a, axis=1)  # Sum along columns -> shape (3,)

    # Should sum 4 values (each 2.0) per row
    assert_dim(b, 1, "Sum along axis 1 should be 1D")
    assert_numel(b, 3, "Sum along axis 1 should have 3 elements")
    assert_value_at(b, 0, 8.0, 1e-6, "Each row sum should be 8.0")
    assert_value_at(b, 1, 8.0, 1e-6, "Each row sum should be 8.0")
    assert_value_at(b, 2, 8.0, 1e-6, "Each row sum should be 8.0")


fn test_sum_axis_keepdims() raises:
    """Test sum with axis and keepdims=True."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    vara = ones(shape, DType.float32)
    varb = sum(a, axis=0, keepdims=True)

    # Should be shape (1, 4) instead of (4,)
    assert_dim(b, 2, "keepdims should preserve dimensions")
    assert_numel(b, 4, "Should have 4 elements")


fn test_mean_axis_0() raises:
    """Test mean along axis 0."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    vara = full(shape, 6.0, DType.float32)  # 3x4 matrix of 6s
    varb = mean(a, axis=0)  # Mean along rows -> shape (4,)

    # Should average 3 values (each 6.0) per column
    assert_dim(b, 1, "Mean along axis 0 should be 1D")
    assert_numel(b, 4, "Mean along axis 0 should have 4 elements")
    assert_value_at(b, 0, 6.0, 1e-6, "Each column mean should be 6.0")
    assert_value_at(b, 1, 6.0, 1e-6, "Each column mean should be 6.0")


fn test_mean_axis_1() raises:
    """Test mean along axis 1."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(5)
    vara = full(shape, 10.0, DType.float32)  # 2x5 matrix of 10s
    varb = mean(a, axis=1)  # Mean along columns -> shape (2,)

    # Should average 5 values (each 10.0) per row
    assert_dim(b, 1, "Mean along axis 1 should be 1D")
    assert_numel(b, 2, "Mean along axis 1 should have 2 elements")
    assert_value_at(b, 0, 10.0, 1e-6, "Each row mean should be 10.0")
    assert_value_at(b, 1, 10.0, 1e-6, "Each row mean should be 10.0")


fn test_max_axis_0() raises:
    """Test max along axis 0."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    vara = full(shape, 7.0, DType.float32)
    varb = max_reduce(a, axis=0)

    # Should find max of 3 values per column
    assert_dim(b, 1, "Max along axis 0 should be 1D")
    assert_numel(b, 4, "Max along axis 0 should have 4 elements")
    assert_value_at(b, 0, 7.0, 1e-6, "Max should be 7.0")


fn test_max_axis_1() raises:
    """Test max along axis 1."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    vara = full(shape, 9.0, DType.float32)
    varb = max_reduce(a, axis=1)

    # Should find max of 3 values per row
    assert_dim(b, 1, "Max along axis 1 should be 1D")
    assert_numel(b, 2, "Max along axis 1 should have 2 elements")
    assert_value_at(b, 0, 9.0, 1e-6, "Max should be 9.0")
    assert_value_at(b, 1, 9.0, 1e-6, "Max should be 9.0")


fn test_min_axis_0() raises:
    """Test min along axis 0."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    vara = full(shape, 3.5, DType.float32)
    varb = min_reduce(a, axis=0)

    # Should find min of 3 values per column
    assert_dim(b, 1, "Min along axis 0 should be 1D")
    assert_numel(b, 4, "Min along axis 0 should have 4 elements")
    assert_value_at(b, 0, 3.5, 1e-6, "Min should be 3.5")


fn test_min_axis_1() raises:
    """Test min along axis 1."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    vara = full(shape, 2.5, DType.float32)
    varb = min_reduce(a, axis=1)

    # Should find min of 3 values per row
    assert_dim(b, 1, "Min along axis 1 should be 1D")
    assert_numel(b, 2, "Min along axis 1 should have 2 elements")
    assert_value_at(b, 0, 2.5, 1e-6, "Min should be 2.5")
    assert_value_at(b, 1, 2.5, 1e-6, "Min should be 2.5")


# ============================================================================
# Test reduction combinations
# ============================================================================

fn test_reductions_consistent() raises:
    """Test that reductions are consistent with each other."""
    var shape = List[Int]()
    shape.append(10)
    vara = full(shape, 5.0, DType.float32)

    varsum_result = sum(a)
    varmean_result = mean(a)
    varmax_result = max_reduce(a)
    varmin_result = min_reduce(a)

    # For all same values:
    # sum = n * value
    # mean = value
    # max = value
    # min = value
    assert_value_at(sum_result, 0, 50.0, 1e-6, "Sum should be 10 * 5 = 50")
    assert_value_at(mean_result, 0, 5.0, 1e-6, "Mean should be 5")
    assert_value_at(max_result, 0, 5.0, 1e-6, "Max should be 5")
    assert_value_at(min_result, 0, 5.0, 1e-6, "Min should be 5")


# ============================================================================
# Main test runner
# ============================================================================

fn main() raises:
    """Run all reduction operation tests."""
    print("Running ExTensor reduction operation tests...")

    # sum() tests
    print("  Testing sum()...")
    test_sum_all_ones()
    test_sum_2d_tensor()
    test_sum_arange()
    test_sum_with_keepdims()
    test_sum_preserves_dtype()

    # mean() tests
    print("  Testing mean()...")
    test_mean_all_ones()
    test_mean_2d_tensor()
    test_mean_arange()
    test_mean_with_keepdims()
    test_mean_preserves_dtype()

    # max_reduce() tests
    print("  Testing max_reduce()...")
    test_max_all_same()
    test_max_arange()
    test_max_negative_values()
    test_max_with_keepdims()
    test_max_preserves_dtype()

    # min_reduce() tests
    print("  Testing min_reduce()...")
    test_min_all_same()
    test_min_arange()
    test_min_negative_values()
    test_min_with_keepdims()
    test_min_preserves_dtype()

    # Axis-specific reduction tests
    print("  Testing axis-specific sum()...")
    test_sum_axis_0()
    test_sum_axis_1()
    test_sum_axis_keepdims()

    print("  Testing axis-specific mean()...")
    test_mean_axis_0()
    test_mean_axis_1()

    print("  Testing axis-specific max_reduce()...")
    test_max_axis_0()
    test_max_axis_1()

    print("  Testing axis-specific min_reduce()...")
    test_min_axis_0()
    test_min_axis_1()

    # Combination tests
    print("  Testing reduction consistency...")
    test_reductions_consistent()

    print("All reduction operation tests completed!")
