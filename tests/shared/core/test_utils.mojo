"""Tests for utility functions in shared.core.utils.

This module tests all utility functions including:
- argmax (scalar and axis-based)
- top_k_indices
- top_k
- argsort
"""

from tests.shared.conftest import (
    assert_equal,
    assert_equal_int,
    assert_shape,
    assert_true,
    assert_close_float,
)
from shared.core.extensor import ExTensor, zeros, ones, arange
from shared.core.utils import (
    argmax,
    top_k_indices,
    top_k,
    argsort,
)


# ============================================================================
# Argmax Tests (Scalar)
# ============================================================================


fn test_argmax_scalar_simple() raises:
    """Test argmax on a simple 1D tensor."""
    var t = arange(0.0, 10.0, 1.0, DType.float32)
    var idx = argmax(t)
    assert_equal_int(idx, 9)


fn test_argmax_scalar_negative_values() raises:
    """Test argmax with negative values."""
    var shape = List[Int]()
    shape.append(5)
    var t = zeros(shape, DType.float32)
    t._data.bitcast[Float32]()[0] = -5.0
    t._data.bitcast[Float32]()[1] = -2.0
    t._data.bitcast[Float32]()[2] = -10.0
    t._data.bitcast[Float32]()[3] = -1.0
    t._data.bitcast[Float32]()[4] = -3.0
    var idx = argmax(t)
    assert_equal_int(idx, 3)


fn test_argmax_scalar_multi_dimensional() raises:
    """Test argmax on multi-dimensional tensor flattens correctly."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var t = zeros(shape, DType.float32)
    # Set max at linear index 5 (row 1, col 1)
    t._data.bitcast[Float32]()[5] = 42.0
    var idx = argmax(t)
    assert_equal_int(idx, 5)


# ============================================================================
# Argmax Tests (Axis)
# ============================================================================


fn test_argmax_axis_1d() raises:
    """Test argmax along axis on 1D tensor."""
    var t = arange(0.0, 5.0, 1.0, DType.float32)
    var result = argmax(t, axis=0)
    assert_equal_int(result.numel(), 1)
    assert_equal_int(Int(result._get_int64(0)), 4)


fn test_argmax_axis_2d_axis0() raises:
    """Test argmax along axis 0 on 2D tensor."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var t = zeros(shape, DType.float32)
    # Set values such that row 2 has the max in each column
    # Column 0: indices 0, 4, 8 -> max at 8
    t._data.bitcast[Float32]()[8] = 100.0
    t._data.bitcast[Float32]()[9] = 100.0
    t._data.bitcast[Float32]()[10] = 100.0
    t._data.bitcast[Float32]()[11] = 100.0

    var result = argmax(t, axis=0)
    assert_shape(result, List[Int](4))

    # All should be 2 (row index 2, which is linear indices 8-11)
    for i in range(4):
        assert_equal_int(Int(result._get_int64(i)), 2)


fn test_argmax_axis_2d_axis1() raises:
    """Test argmax along axis 1 on 2D tensor."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var t = zeros(shape, DType.float32)
    # Set max in last column for each row
    # Row 0, col 3: linear index 3
    # Row 1, col 3: linear index 7
    # Row 2, col 3: linear index 11
    t._data.bitcast[Float32]()[3] = 10.0
    t._data.bitcast[Float32]()[7] = 20.0
    t._data.bitcast[Float32]()[11] = 30.0

    var result = argmax(t, axis=1)
    assert_shape(result, List[Int](3))
    assert_equal_int(Int(result._get_int64(0)), 3)
    assert_equal_int(Int(result._get_int64(1)), 3)
    assert_equal_int(Int(result._get_int64(2)), 3)


fn test_argmax_axis_3d() raises:
    """Test argmax on 3D tensor along axis."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)
    var t = zeros(shape, DType.float32)
    # Set max values
    t._data.bitcast[Float32]()[5] = 50.0

    var result = argmax(t, axis=2)
    assert_shape(result, List[Int](2, 3))


# ============================================================================
# Top K Tests
# ============================================================================


fn test_top_k_indices_simple() raises:
    """Test top_k_indices on a simple tensor."""
    var t = arange(0.0, 10.0, 1.0, DType.float32)
    var indices = top_k_indices(t, 3)
    assert_equal_int(indices[0], 9)
    assert_equal_int(indices[1], 8)
    assert_equal_int(indices[2], 7)


fn test_top_k_indices_single_element() raises:
    """Test top_k_indices with k=1."""
    var t = arange(0.0, 5.0, 1.0, DType.float32)
    var indices = top_k_indices(t, 1)
    assert_equal_int(indices[0], 4)


fn test_top_k_indices_all_elements() raises:
    """Test top_k_indices with k=numel."""
    var t = arange(0.0, 5.0, 1.0, DType.float32)
    var indices = top_k_indices(t, 5)
    assert_equal_int(indices[0], 4)
    assert_equal_int(indices[1], 3)
    assert_equal_int(indices[2], 2)
    assert_equal_int(indices[3], 1)
    assert_equal_int(indices[4], 0)


fn test_top_k_indices_with_duplicates() raises:
    """Test top_k_indices with duplicate values."""
    var shape = List[Int]()
    shape.append(6)
    var t = zeros(shape, DType.float32)
    t._data.bitcast[Float32]()[0] = 5.0
    t._data.bitcast[Float32]()[1] = 5.0
    t._data.bitcast[Float32]()[2] = 3.0
    t._data.bitcast[Float32]()[3] = 3.0
    t._data.bitcast[Float32]()[4] = 1.0
    t._data.bitcast[Float32]()[5] = 1.0

    var indices = top_k_indices(t, 3)
    # First two should be indices 0 and 1 (both value 5)
    # Third should be index 2 or 3 (value 3)
    assert_equal_int(len(indices), 3)


fn test_top_k_values_and_indices() raises:
    """Test top_k function returns both values and indices."""
    var t = arange(0.0, 10.0, 1.0, DType.float32)
    var (values, indices) = top_k(t, 3)

    # Check shape of values
    assert_shape(values, List[Int](3))

    # Check values are in correct order (descending)
    assert_close_float(values._get_float64(0), 9.0)
    assert_close_float(values._get_float64(1), 8.0)
    assert_close_float(values._get_float64(2), 7.0)

    # Check indices
    assert_equal_int(indices[0], 9)
    assert_equal_int(indices[1], 8)
    assert_equal_int(indices[2], 7)


fn test_top_k_multidimensional() raises:
    """Test top_k on multi-dimensional tensor."""
    var t = arange(0.0, 12.0, 1.0, DType.float32)
    var (values, indices) = top_k(t, 2)

    assert_shape(values, List[Int](2))
    assert_close_float(values._get_float64(0), 11.0)
    assert_close_float(values._get_float64(1), 10.0)


# ============================================================================
# Argsort Tests
# ============================================================================


fn test_argsort_ascending() raises:
    """Test argsort in ascending order."""
    var shape = List[Int]()
    shape.append(5)
    var t = zeros(shape, DType.float32)
    t._data.bitcast[Float32]()[0] = 5.0
    t._data.bitcast[Float32]()[1] = 2.0
    t._data.bitcast[Float32]()[2] = 8.0
    t._data.bitcast[Float32]()[3] = 1.0
    t._data.bitcast[Float32]()[4] = 9.0

    var indices = argsort(t, descending=False)
    assert_equal_int(indices[0], 3)  # value 1
    assert_equal_int(indices[1], 1)  # value 2
    assert_equal_int(indices[2], 0)  # value 5
    assert_equal_int(indices[3], 2)  # value 8
    assert_equal_int(indices[4], 4)  # value 9


fn test_argsort_descending() raises:
    """Test argsort in descending order."""
    var shape = List[Int]()
    shape.append(5)
    var t = zeros(shape, DType.float32)
    t._data.bitcast[Float32]()[0] = 5.0
    t._data.bitcast[Float32]()[1] = 2.0
    t._data.bitcast[Float32]()[2] = 8.0
    t._data.bitcast[Float32]()[3] = 1.0
    t._data.bitcast[Float32]()[4] = 9.0

    var indices = argsort(t, descending=True)
    assert_equal_int(indices[0], 4)  # value 9
    assert_equal_int(indices[1], 2)  # value 8
    assert_equal_int(indices[2], 0)  # value 5
    assert_equal_int(indices[3], 1)  # value 2
    assert_equal_int(indices[4], 3)  # value 1


fn test_argsort_single_element() raises:
    """Test argsort with single element."""
    var t = arange(0.0, 1.0, 1.0, DType.float32)
    var indices = argsort(t, descending=False)
    assert_equal_int(indices[0], 0)


fn test_argsort_sorted_array() raises:
    """Test argsort on already sorted array."""
    var t = arange(0.0, 5.0, 1.0, DType.float32)
    var indices = argsort(t, descending=False)
    assert_equal_int(indices[0], 0)
    assert_equal_int(indices[1], 1)
    assert_equal_int(indices[2], 2)
    assert_equal_int(indices[3], 3)
    assert_equal_int(indices[4], 4)


fn test_argsort_reverse_sorted() raises:
    """Test argsort on reverse sorted array."""
    var shape = List[Int]()
    shape.append(5)
    var t = zeros(shape, DType.float32)
    t._data.bitcast[Float32]()[0] = 5.0
    t._data.bitcast[Float32]()[1] = 4.0
    t._data.bitcast[Float32]()[2] = 3.0
    t._data.bitcast[Float32]()[3] = 2.0
    t._data.bitcast[Float32]()[4] = 1.0

    var indices = argsort(t, descending=False)
    assert_equal_int(indices[0], 4)
    assert_equal_int(indices[1], 3)
    assert_equal_int(indices[2], 2)
    assert_equal_int(indices[3], 1)
    assert_equal_int(indices[4], 0)


fn test_argsort_negative_values() raises:
    """Test argsort with negative values."""
    var shape = List[Int]()
    shape.append(5)
    var t = zeros(shape, DType.float32)
    t._data.bitcast[Float32]()[0] = -5.0
    t._data.bitcast[Float32]()[1] = 2.0
    t._data.bitcast[Float32]()[2] = -1.0
    t._data.bitcast[Float32]()[3] = 0.0
    t._data.bitcast[Float32]()[4] = 3.0

    var indices = argsort(t, descending=False)
    assert_equal_int(indices[0], 0)  # -5
    assert_equal_int(indices[1], 2)  # -1
    assert_equal_int(indices[2], 3)  # 0
    assert_equal_int(indices[3], 1)  # 2
    assert_equal_int(indices[4], 4)  # 3


fn test_argsort_multidimensional() raises:
    """Test argsort on multi-dimensional tensor (flattens)."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    var t = zeros(shape, DType.float32)
    t._data.bitcast[Float32]()[0] = 5.0
    t._data.bitcast[Float32]()[1] = 1.0
    t._data.bitcast[Float32]()[2] = 3.0
    t._data.bitcast[Float32]()[3] = 2.0
    t._data.bitcast[Float32]()[4] = 4.0
    t._data.bitcast[Float32]()[5] = 0.0

    var indices = argsort(t, descending=False)
    assert_equal_int(indices[0], 5)  # 0
    assert_equal_int(indices[1], 1)  # 1
    assert_equal_int(indices[2], 3)  # 2
    assert_equal_int(indices[3], 2)  # 3
    assert_equal_int(indices[4], 4)  # 4
    assert_equal_int(indices[5], 0)  # 5
