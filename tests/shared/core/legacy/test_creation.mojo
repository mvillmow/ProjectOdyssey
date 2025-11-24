"""Tests for ExTensor creation operations.

Tests all tensor creation functions including zeros, ones, full, arange,
from_array, eye, linspace, and empty with various shapes and dtypes.
"""

from memory import DType

# Import ExTensor and creation operations
from shared.core import ExTensor, zeros, ones, full, empty, arange, eye, linspace

# Import test helpers
from ..helpers.assertions import (
    assert_true,
    assert_false,
    assert_equal_int,
    assert_equal_float,
    assert_close_float,
    assert_shape,
    assert_dtype,
    assert_numel,
    assert_dim,
    assert_value_at,
    assert_all_values,
    assert_all_close,
)


# ============================================================================
# Test zeros()
# ============================================================================

fn test_zeros_1d_float32() raises:
    """Test creating 1D tensor of zeros with float32."""
    var shape = List[Int]()
    shape.append(5)
    vart = zeros(shape, DType.float32)

    assert_dim(t, 1, "zeros 1D should have 1 dimension")
    assert_numel(t, 5, "zeros 1D should have 5 elements")
    assert_dtype(t, DType.float32, "zeros should have float32 dtype")
    assert_all_values(t, 0.0, 1e-8, "zeros should contain all 0.0 values")


fn test_zeros_2d_int64() raises:
    """Test creating 2D tensor of zeros with int64."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    vart = zeros(shape, DType.int64)

    assert_dim(t, 2, "zeros 2D should have 2 dimensions")
    assert_numel(t, 12, "zeros 2D(3,4) should have 12 elements")
    assert_dtype(t, DType.int64, "zeros should have int64 dtype")
    assert_all_values(t, 0.0, 1e-8, "zeros should contain all 0.0 values")


fn test_zeros_3d_float64() raises:
    """Test creating 3D tensor of zeros with float64."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)
    vart = zeros(shape, DType.float64)

    assert_dim(t, 3, "zeros 3D should have 3 dimensions")
    assert_numel(t, 24, "zeros 3D(2,3,4) should have 24 elements")
    assert_dtype(t, DType.float64, "zeros should have float64 dtype")
    assert_all_values(t, 0.0, 1e-8, "zeros should contain all 0.0 values")


fn test_zeros_empty_shape() raises:
    """Test creating 0D scalar tensor of zeros."""
    var shape = List[Int]()
    vart = zeros(shape, DType.float32)

    assert_dim(t, 0, "zeros 0D should have 0 dimensions")
    assert_numel(t, 1, "zeros 0D should have 1 element")
    assert_dtype(t, DType.float32, "zeros should have float32 dtype")
    assert_value_at(t, 0, 0.0, 1e-8, "zeros 0D should be 0.0")


fn test_zeros_large_shape() raises:
    """Test creating zeros with very large shape."""
    var shape = List[Int]()
    shape.append(10000)
    vart = zeros(shape, DType.float32)

    assert_numel(t, 10000, "zeros large should have 10000 elements")
    # Spot-check a few values
    assert_value_at(t, 0, 0.0, 1e-8, "zeros first element should be 0.0")
    assert_value_at(t, 5000, 0.0, 1e-8, "zeros middle element should be 0.0")
    assert_value_at(t, 9999, 0.0, 1e-8, "zeros last element should be 0.0")


# ============================================================================
# Test ones()
# ============================================================================

fn test_ones_1d_float32() raises:
    """Test creating 1D tensor of ones with float32."""
    var shape = List[Int]()
    shape.append(5)
    vart = ones(shape, DType.float32)

    assert_dim(t, 1, "ones 1D should have 1 dimension")
    assert_numel(t, 5, "ones 1D should have 5 elements")
    assert_dtype(t, DType.float32, "ones should have float32 dtype")
    assert_all_values(t, 1.0, 1e-8, "ones should contain all 1.0 values")


fn test_ones_2d_int32() raises:
    """Test creating 2D tensor of ones with int32."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    vart = ones(shape, DType.int32)

    assert_dim(t, 2, "ones 2D should have 2 dimensions")
    assert_numel(t, 12, "ones 2D(3,4) should have 12 elements")
    assert_dtype(t, DType.int32, "ones should have int32 dtype")
    assert_all_values(t, 1.0, 1e-8, "ones should contain all 1.0 values")


fn test_ones_3d_float64() raises:
    """Test creating 3D tensor of ones with float64."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)
    vart = ones(shape, DType.float64)

    assert_dim(t, 3, "ones 3D should have 3 dimensions")
    assert_numel(t, 24, "ones 3D(2,3,4) should have 24 elements")
    assert_dtype(t, DType.float64, "ones should have float64 dtype")
    assert_all_values(t, 1.0, 1e-8, "ones should contain all 1.0 values")


# ============================================================================
# Test full()
# ============================================================================

fn test_full_positive_value() raises:
    """Test creating tensor filled with positive value."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    vart = full(shape, 5.5, DType.float32)

    assert_numel(t, 12, "full should have 12 elements")
    assert_dtype(t, DType.float32, "full should have float32 dtype")
    assert_all_values(t, 5.5, 1e-6, "full should contain all 5.5 values")


fn test_full_negative_value() raises:
    """Test creating tensor filled with negative value."""
    var shape = List[Int]()
    shape.append(10)
    vart = full(shape, -3.14, DType.float64)

    assert_numel(t, 10, "full should have 10 elements")
    assert_dtype(t, DType.float64, "full should have float64 dtype")
    assert_all_values(t, -3.14, 1e-8, "full should contain all -3.14 values")


fn test_full_zero_value() raises:
    """Test creating tensor filled with zero (should match zeros)."""
    var shape = List[Int]()
    shape.append(5)
    shape.append(5)
    vart = full(shape, 0.0, DType.float32)

    assert_numel(t, 25, "full with 0.0 should have 25 elements")
    assert_all_values(t, 0.0, 1e-8, "full with 0.0 should match zeros")


fn test_full_large_value() raises:
    """Test creating tensor filled with large value."""
    var shape = List[Int]()
    shape.append(100)
    vart = full(shape, 999999.0, DType.float32)

    assert_numel(t, 100, "full should have 100 elements")
    assert_all_values(t, 999999.0, 1e-2, "full should contain large value")


# ============================================================================
# Test empty()
# ============================================================================

fn test_empty_allocates_memory() raises:
    """Test that empty() allocates memory without initialization."""
    var shape = List[Int]()
    shape.append(5)
    shape.append(10)
    vart = empty(shape, DType.float32)

    # Verify tensor is created with correct shape and dtype
    # Don't check values (they are undefined/uninitialized)
    assert_numel(t, 50, "empty should allocate correct size")
    assert_dtype(t, DType.float32, "empty should have correct dtype")
    assert_dim(t, 2, "empty should have correct dimensions")


fn test_empty_1d() raises:
    """Test creating empty 1D tensor."""
    var shape = List[Int]()
    shape.append(100)
    vart = empty(shape, DType.float64)

    assert_numel(t, 100, "empty 1D should have 100 elements")
    assert_dim(t, 1, "empty 1D should have 1 dimension")
    assert_dtype(t, DType.float64, "empty should have float64 dtype")


fn test_empty_2d() raises:
    """Test creating empty 2D tensor."""
    var shape = List[Int]()
    shape.append(8)
    shape.append(8)
    vart = empty(shape, DType.int32)

    assert_numel(t, 64, "empty 2D should have 64 elements")
    assert_dim(t, 2, "empty 2D should have 2 dimensions")
    assert_dtype(t, DType.int32, "empty should have int32 dtype")


# ============================================================================
# Test arange()
# ============================================================================

fn test_arange_basic() raises:
    """Test arange with start, stop, step=1."""
    vart = arange(0.0, 10.0, 1.0, DType.float32)

    assert_numel(t, 10, "arange(0, 10, 1) should have 10 elements")
    assert_dim(t, 1, "arange should be 1D")
    assert_dtype(t, DType.float32, "arange should have float32 dtype")

    # Check values: [0, 1, 2, ..., 9]
    for i in range(10):
        assert_value_at(t, i, Float64(i), 1e-6, "arange value at index " + String(i))


fn test_arange_step_2() raises:
    """Test arange with step > 1."""
    vart = arange(0.0, 10.0, 2.0, DType.float32)

    assert_numel(t, 5, "arange(0, 10, 2) should have 5 elements")
    assert_value_at(t, 0, 0.0, 1e-6, "arange[0]")
    assert_value_at(t, 1, 2.0, 1e-6, "arange[1]")
    assert_value_at(t, 2, 4.0, 1e-6, "arange[2]")
    assert_value_at(t, 3, 6.0, 1e-6, "arange[3]")
    assert_value_at(t, 4, 8.0, 1e-6, "arange[4]")


fn test_arange_step_fractional() raises:
    """Test arange with fractional step."""
    vart = arange(0.0, 1.0, 0.2, DType.float64)

    assert_numel(t, 5, "arange(0, 1, 0.2) should have 5 elements")
    assert_value_at(t, 0, 0.0, 1e-8, "arange fractional [0]")
    assert_value_at(t, 1, 0.2, 1e-8, "arange fractional [1]")
    assert_value_at(t, 2, 0.4, 1e-8, "arange fractional [2]")
    assert_value_at(t, 3, 0.6, 1e-8, "arange fractional [3]")
    assert_value_at(t, 4, 0.8, 1e-8, "arange fractional [4]")


fn test_arange_reverse() raises:
    """Test arange with negative step (reverse order)."""
    vart = arange(10.0, 0.0, -1.0, DType.float32)

    assert_numel(t, 10, "arange(10, 0, -1) should have 10 elements")
    # Check values: [10, 9, 8, ..., 1]
    for i in range(10):
        assert_value_at(t, i, Float64(10 - i), 1e-6, "arange reverse value")


fn test_arange_float() raises:
    """Test arange with float dtype."""
    vart = arange(1.5, 5.5, 1.0, DType.float64)

    assert_numel(t, 4, "arange(1.5, 5.5, 1.0) should have 4 elements")
    assert_dtype(t, DType.float64, "arange should have float64 dtype")
    assert_value_at(t, 0, 1.5, 1e-8, "arange float [0]")
    assert_value_at(t, 1, 2.5, 1e-8, "arange float [1]")
    assert_value_at(t, 2, 3.5, 1e-8, "arange float [2]")
    assert_value_at(t, 3, 4.5, 1e-8, "arange float [3]")


# ============================================================================
# Test from_array()
# ============================================================================

fn test_from_array_1d() raises:
    """Test creating tensor from 1D array."""
    # TODO: Implement once from_array() is available
    # Not yet implemented
    pass


fn test_from_array_2d() raises:
    """Test creating tensor from 2D nested array."""
    # TODO: Implement once from_array() is available
    # Not yet implemented
    pass


fn test_from_array_3d() raises:
    """Test creating tensor from 3D nested array."""
    # TODO: Implement once from_array() is available
    # Not yet implemented
    pass


# ============================================================================
# Test eye()
# ============================================================================

fn test_eye_square() raises:
    """Test creating square identity matrix."""
    vart = eye(5, 5, 0, DType.float32)

    assert_dim(t, 2, "eye should be 2D")
    assert_numel(t, 25, "eye(5,5) should have 25 elements")
    assert_dtype(t, DType.float32, "eye should have float32 dtype")

    # Check diagonal is 1, off-diagonal is 0
    for i in range(5):
        for j in range(5):
            varflat_idx = i * 5 + j
            if i == j:
                assert_value_at(t, flat_idx, 1.0, 1e-8, "eye diagonal should be 1.0")
            else:
                assert_value_at(t, flat_idx, 0.0, 1e-8, "eye off-diagonal should be 0.0")


fn test_eye_rectangular() raises:
    """Test creating rectangular identity matrix."""
    vart = eye(3, 5, 0, DType.float64)

    assert_dim(t, 2, "eye should be 2D")
    assert_numel(t, 15, "eye(3,5) should have 15 elements")
    assert_dtype(t, DType.float64, "eye should have float64 dtype")

    # Check diagonal is 1 where i==j, rest is 0
    for i in range(3):
        for j in range(5):
            varflat_idx = i * 5 + j
            if i == j:
                assert_value_at(t, flat_idx, 1.0, 1e-8, "eye diagonal should be 1.0")
            else:
                assert_value_at(t, flat_idx, 0.0, 1e-8, "eye off-diagonal should be 0.0")


fn test_eye_offset_diagonal() raises:
    """Test creating identity matrix with offset diagonal (k parameter)."""
    # TODO: Implement once eye() supports k parameter for offset diagonal
    # This would create identity with 1s on a different diagonal
    pass


# ============================================================================
# Test linspace()
# ============================================================================

fn test_linspace_basic() raises:
    """Test linspace with basic range."""
    vart = linspace(0.0, 10.0, 11, DType.float32)

    assert_numel(t, 11, "linspace(0, 10, 11) should have 11 elements")
    assert_dim(t, 1, "linspace should be 1D")
    assert_dtype(t, DType.float32, "linspace should have float32 dtype")

    # Check values: [0, 1, 2, ..., 10]
    for i in range(11):
        assert_value_at(t, i, Float64(i), 1e-6, "linspace value at index " + String(i))


fn test_linspace_negative_range() raises:
    """Test linspace with negative start/stop."""
    vart = linspace(-5.0, 5.0, 11, DType.float64)

    assert_numel(t, 11, "linspace(-5, 5, 11) should have 11 elements")
    assert_dtype(t, DType.float64, "linspace should have float64 dtype")

    # Check values: [-5, -4, -3, ..., 5]
    for i in range(11):
        assert_value_at(t, i, Float64(-5 + i), 1e-8, "linspace negative value")


fn test_linspace_small_num() raises:
    """Test linspace with small number of points."""
    vart = linspace(0.0, 1.0, 2, DType.float32)

    assert_numel(t, 2, "linspace(0, 1, 2) should have 2 elements")
    assert_value_at(t, 0, 0.0, 1e-6, "linspace start should be 0.0")
    assert_value_at(t, 1, 1.0, 1e-6, "linspace end should be 1.0")


fn test_linspace_large_num() raises:
    """Test linspace with large number of points."""
    vart = linspace(0.0, 100.0, 101, DType.float64)

    assert_numel(t, 101, "linspace(0, 100, 101) should have 101 elements")
    # Spot-check a few values
    assert_value_at(t, 0, 0.0, 1e-8, "linspace start")
    assert_value_at(t, 50, 50.0, 1e-6, "linspace middle")
    assert_value_at(t, 100, 100.0, 1e-8, "linspace end")


# ============================================================================
# Test dtype support
# ============================================================================

fn test_creation_float16() raises:
    """Test creation operations with float16 dtype."""
    var shape = List[Int]()
    shape.append(5)
    vart = zeros(shape, DType.float16)
    assert_dtype(t, DType.float16, "zeros should support float16")


fn test_creation_float32() raises:
    """Test creation operations with float32 dtype."""
    var shape = List[Int]()
    shape.append(5)
    vart = ones(shape, DType.float32)
    assert_dtype(t, DType.float32, "ones should support float32")


fn test_creation_float64() raises:
    """Test creation operations with float64 dtype."""
    var shape = List[Int]()
    shape.append(5)
    vart = full(shape, 3.14, DType.float64)
    assert_dtype(t, DType.float64, "full should support float64")


fn test_creation_int8() raises:
    """Test creation operations with int8 dtype."""
    var shape = List[Int]()
    shape.append(5)
    vart = zeros(shape, DType.int8)
    assert_dtype(t, DType.int8, "zeros should support int8")


fn test_creation_int32() raises:
    """Test creation operations with int32 dtype."""
    var shape = List[Int]()
    shape.append(5)
    vart = ones(shape, DType.int32)
    assert_dtype(t, DType.int32, "ones should support int32")


fn test_creation_uint8() raises:
    """Test creation operations with uint8 dtype."""
    var shape = List[Int]()
    shape.append(5)
    vart = full(shape, 255.0, DType.uint8)
    assert_dtype(t, DType.uint8, "full should support uint8")


fn test_creation_bool() raises:
    """Test creation operations with bool dtype."""
    var shape = List[Int]()
    shape.append(5)
    vart = zeros(shape, DType.bool)
    assert_dtype(t, DType.bool, "zeros should support bool")


# ============================================================================
# Test edge cases
# ============================================================================

fn test_creation_0d_scalar() raises:
    """Test creating 0D scalar tensor."""
    var shape = List[Int]()
    vart = zeros(shape, DType.float32)

    assert_dim(t, 0, "0D tensor should have 0 dimensions")
    assert_numel(t, 1, "0D tensor should have 1 element")
    assert_value_at(t, 0, 0.0, 1e-8, "0D tensor value")


fn test_creation_very_large_1d() raises:
    """Test creating very large 1D tensor."""
    var shape = List[Int]()
    shape.append(1000000)
    vart = zeros(shape, DType.float32)

    assert_numel(t, 1000000, "Large 1D tensor should have 1000000 elements")
    # Spot-check a few values
    assert_value_at(t, 0, 0.0, 1e-8, "Large tensor first element")
    assert_value_at(t, 999999, 0.0, 1e-8, "Large tensor last element")


fn test_creation_high_dimensional() raises:
    """Test creating tensor with many dimensions (e.g., 8D)."""
    var shape = List[Int]()
    for i in range(8):
        shape[i] = 2
    vart = zeros(shape, DType.float32)

    assert_dim(t, 8, "8D tensor should have 8 dimensions")
    assert_numel(t, 256, "8D tensor (2x2x2x2x2x2x2x2) should have 256 elements")


# ============================================================================
# Main test runner
# ============================================================================

fn main() raises:
    """Run all creation operation tests."""
    print("Running ExTensor creation operation tests...")

    # zeros() tests
    test_zeros_1d_float32()
    test_zeros_2d_int64()
    test_zeros_3d_float64()
    test_zeros_empty_shape()
    test_zeros_large_shape()

    # ones() tests
    test_ones_1d_float32()
    test_ones_2d_int32()
    test_ones_3d_float64()

    # full() tests
    test_full_positive_value()
    test_full_negative_value()
    test_full_zero_value()
    test_full_large_value()

    # empty() tests
    test_empty_allocates_memory()
    test_empty_1d()
    test_empty_2d()

    # arange() tests
    test_arange_basic()
    test_arange_step_2()
    test_arange_step_fractional()
    test_arange_reverse()
    test_arange_float()

    # from_array() tests
    test_from_array_1d()
    test_from_array_2d()
    test_from_array_3d()

    # eye() tests
    test_eye_square()
    test_eye_rectangular()
    test_eye_offset_diagonal()

    # linspace() tests
    test_linspace_basic()
    test_linspace_negative_range()
    test_linspace_small_num()
    test_linspace_large_num()

    # dtype tests
    test_creation_float16()
    test_creation_float32()
    test_creation_float64()
    test_creation_int8()
    test_creation_int32()
    test_creation_uint8()
    test_creation_bool()

    # Edge case tests
    test_creation_0d_scalar()
    test_creation_very_large_1d()
    test_creation_high_dimensional()

    print("All creation operation tests completed!")
