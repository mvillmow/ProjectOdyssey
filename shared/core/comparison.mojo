"""Comparison operations for ExTensor with broadcasting support.

Implements element-wise comparison operations following NumPy-style broadcasting.
"""

from collections import List
from .extensor import ExTensor
from .broadcasting import broadcast_shapes, compute_broadcast_strides


fn equal(a: ExTensor, b: ExTensor) raises -> ExTensor:
    """Element-wise equality comparison with broadcasting.

    Args:
        a: First tensor.
        b: Second tensor.

    Returns:
        A new boolean tensor containing a == b.

    Raises:
        Error if shapes are not broadcast-compatible or dtypes don't match.

    Examples:
        var a = full(List[Int](3, 4), 2.0, DType.float32)
        var b = full(List[Int](3, 4), 2.0, DType.float32)
        var c = equal(a, b)  # Shape (3, 4), all True.
    """
    if a.dtype() != b.dtype():
        raise Error("Cannot compare tensors with different dtypes")

    var result_shape = broadcast_shapes(a.shape(), b.shape())
    var result = ExTensor(result_shape, DType.bool)

    # Compute broadcast strides
    var strides_a = compute_broadcast_strides(a.shape(), result_shape)
    var strides_b = compute_broadcast_strides(b.shape(), result_shape)

    # Calculate total elements in result
    var total_elems = 1
    for i in range(len(result_shape)):
        total_elems *= result_shape[i]

    # Precompute row-major strides for result shape
    var result_strides= List[Int]()
    var stride = 1
    for i in range(len(result_shape) - 1, -1, -1):
        result_strides.append(stride)
        stride *= result_shape[i]

    # Reverse to get correct order (left-to-right)
    var result_strides_final= List[Int]()
    for i in range(len(result_strides) - 1, -1, -1):
        result_strides_final.append(result_strides[i])

    # Iterate over all result elements
    for result_idx in range(total_elems):
        var idx_a = 0
        var idx_b = 0
        var temp_idx = result_idx

        # Convert flat index to multi-dimensional coordinates, then compute source indices
        for dim in range(len(result_shape)):
            var coord = temp_idx // result_strides_final[dim]
            temp_idx = temp_idx % result_strides_final[dim]

            idx_a += coord * strides_a[dim]
            idx_b += coord * strides_b[dim]

        # Perform comparison
        var a_val = a._get_float64(idx_a)
        var b_val = b._get_float64(idx_b)
        result._set_int64(result_idx, 1 if a_val == b_val else 0)

    return result^


fn not_equal(a: ExTensor, b: ExTensor) raises -> ExTensor:
    """Element-wise inequality comparison with broadcasting.

    Args:
        a: First tensor.
        b: Second tensor.

    Returns:
        A new boolean tensor containing a != b.

    Raises:
        Error if shapes are not broadcast-compatible or dtypes don't match.

    Examples:
        var a = full(List[Int](3, 4), 2.0, DType.float32)
        var b = full(List[Int](3, 4), 3.0, DType.float32)
        var c = not_equal(a, b)  # Shape (3, 4), all True.
    """
    if a.dtype() != b.dtype():
        raise Error("Cannot compare tensors with different dtypes")

    var result_shape = broadcast_shapes(a.shape(), b.shape())
    var result = ExTensor(result_shape, DType.bool)

    # Compute broadcast strides
    var strides_a = compute_broadcast_strides(a.shape(), result_shape)
    var strides_b = compute_broadcast_strides(b.shape(), result_shape)

    # Calculate total elements in result
    var total_elems = 1
    for i in range(len(result_shape)):
        total_elems *= result_shape[i]

    # Precompute row-major strides for result shape
    var result_strides= List[Int]()
    var stride = 1
    for i in range(len(result_shape) - 1, -1, -1):
        result_strides.append(stride)
        stride *= result_shape[i]

    # Reverse to get correct order (left-to-right)
    var result_strides_final= List[Int]()
    for i in range(len(result_strides) - 1, -1, -1):
        result_strides_final.append(result_strides[i])

    # Iterate over all result elements
    for result_idx in range(total_elems):
        var idx_a = 0
        var idx_b = 0
        var temp_idx = result_idx

        # Convert flat index to multi-dimensional coordinates, then compute source indices
        for dim in range(len(result_shape)):
            var coord = temp_idx // result_strides_final[dim]
            temp_idx = temp_idx % result_strides_final[dim]

            idx_a += coord * strides_a[dim]
            idx_b += coord * strides_b[dim]

        # Perform comparison
        var a_val = a._get_float64(idx_a)
        var b_val = b._get_float64(idx_b)
        result._set_int64(result_idx, 1 if a_val != b_val else 0)

    return result^


fn less(a: ExTensor, b: ExTensor) raises -> ExTensor:
    """Element-wise less-than comparison with broadcasting.

    Args:
        a: First tensor.
        b: Second tensor.

    Returns:
        A new boolean tensor containing a < b.

    Raises:
        Error if shapes are not broadcast-compatible or dtypes don't match.

    Examples:
        var a = full(List[Int](3, 4), 2.0, DType.float32)
        var b = full(List[Int](3, 4), 3.0, DType.float32)
        var c = less(a, b)  # Shape (3, 4), all True.
    """
    if a.dtype() != b.dtype():
        raise Error("Cannot compare tensors with different dtypes")

    var result_shape = broadcast_shapes(a.shape(), b.shape())
    var result = ExTensor(result_shape, DType.bool)

    # Compute broadcast strides
    var strides_a = compute_broadcast_strides(a.shape(), result_shape)
    var strides_b = compute_broadcast_strides(b.shape(), result_shape)

    # Calculate total elements in result
    var total_elems = 1
    for i in range(len(result_shape)):
        total_elems *= result_shape[i]

    # Precompute row-major strides for result shape
    var result_strides= List[Int]()
    var stride = 1
    for i in range(len(result_shape) - 1, -1, -1):
        result_strides.append(stride)
        stride *= result_shape[i]

    # Reverse to get correct order (left-to-right)
    var result_strides_final= List[Int]()
    for i in range(len(result_strides) - 1, -1, -1):
        result_strides_final.append(result_strides[i])

    # Iterate over all result elements
    for result_idx in range(total_elems):
        var idx_a = 0
        var idx_b = 0
        var temp_idx = result_idx

        # Convert flat index to multi-dimensional coordinates, then compute source indices
        for dim in range(len(result_shape)):
            var coord = temp_idx // result_strides_final[dim]
            temp_idx = temp_idx % result_strides_final[dim]

            idx_a += coord * strides_a[dim]
            idx_b += coord * strides_b[dim]

        # Perform comparison
        var a_val = a._get_float64(idx_a)
        var b_val = b._get_float64(idx_b)
        result._set_int64(result_idx, 1 if a_val < b_val else 0)

    return result^


fn less_equal(a: ExTensor, b: ExTensor) raises -> ExTensor:
    """Element-wise less-than-or-equal comparison with broadcasting.

    Args:
        a: First tensor.
        b: Second tensor.

    Returns:
        A new boolean tensor containing a <= b.

    Raises:
        Error if shapes are not broadcast-compatible or dtypes don't match.

    Examples:
        var a = full(List[Int](3, 4), 2.0, DType.float32)
        var b = full(List[Int](3, 4), 2.0, DType.float32)
        var c = less_equal(a, b)  # Shape (3, 4), all True.
    """
    if a.dtype() != b.dtype():
        raise Error("Cannot compare tensors with different dtypes")

    var result_shape = broadcast_shapes(a.shape(), b.shape())
    var result = ExTensor(result_shape, DType.bool)

    # Compute broadcast strides
    var strides_a = compute_broadcast_strides(a.shape(), result_shape)
    var strides_b = compute_broadcast_strides(b.shape(), result_shape)

    # Calculate total elements in result
    var total_elems = 1
    for i in range(len(result_shape)):
        total_elems *= result_shape[i]

    # Precompute row-major strides for result shape
    var result_strides= List[Int]()
    var stride = 1
    for i in range(len(result_shape) - 1, -1, -1):
        result_strides.append(stride)
        stride *= result_shape[i]

    # Reverse to get correct order (left-to-right)
    var result_strides_final= List[Int]()
    for i in range(len(result_strides) - 1, -1, -1):
        result_strides_final.append(result_strides[i])

    # Iterate over all result elements
    for result_idx in range(total_elems):
        var idx_a = 0
        var idx_b = 0
        var temp_idx = result_idx

        # Convert flat index to multi-dimensional coordinates, then compute source indices
        for dim in range(len(result_shape)):
            var coord = temp_idx // result_strides_final[dim]
            temp_idx = temp_idx % result_strides_final[dim]

            idx_a += coord * strides_a[dim]
            idx_b += coord * strides_b[dim]

        # Perform comparison
        var a_val = a._get_float64(idx_a)
        var b_val = b._get_float64(idx_b)
        result._set_int64(result_idx, 1 if a_val <= b_val else 0)

    return result^


fn greater(a: ExTensor, b: ExTensor) raises -> ExTensor:
    """Element-wise greater-than comparison with broadcasting.

    Args:
        a: First tensor.
        b: Second tensor.

    Returns:
        A new boolean tensor containing a > b.

    Raises:
        Error if shapes are not broadcast-compatible or dtypes don't match.

    Examples:
        var a = full(List[Int](3, 4), 3.0, DType.float32)
        var b = full(List[Int](3, 4), 2.0, DType.float32)
        var c = greater(a, b)  # Shape (3, 4), all True.
    """
    if a.dtype() != b.dtype():
        raise Error("Cannot compare tensors with different dtypes")

    var result_shape = broadcast_shapes(a.shape(), b.shape())
    var result = ExTensor(result_shape, DType.bool)

    # Compute broadcast strides
    var strides_a = compute_broadcast_strides(a.shape(), result_shape)
    var strides_b = compute_broadcast_strides(b.shape(), result_shape)

    # Calculate total elements in result
    var total_elems = 1
    for i in range(len(result_shape)):
        total_elems *= result_shape[i]

    # Precompute row-major strides for result shape
    var result_strides= List[Int]()
    var stride = 1
    for i in range(len(result_shape) - 1, -1, -1):
        result_strides.append(stride)
        stride *= result_shape[i]

    # Reverse to get correct order (left-to-right)
    var result_strides_final= List[Int]()
    for i in range(len(result_strides) - 1, -1, -1):
        result_strides_final.append(result_strides[i])

    # Iterate over all result elements
    for result_idx in range(total_elems):
        var idx_a = 0
        var idx_b = 0
        var temp_idx = result_idx

        # Convert flat index to multi-dimensional coordinates, then compute source indices
        for dim in range(len(result_shape)):
            var coord = temp_idx // result_strides_final[dim]
            temp_idx = temp_idx % result_strides_final[dim]

            idx_a += coord * strides_a[dim]
            idx_b += coord * strides_b[dim]

        # Perform comparison
        var a_val = a._get_float64(idx_a)
        var b_val = b._get_float64(idx_b)
        result._set_int64(result_idx, 1 if a_val > b_val else 0)

    return result^


fn greater_equal(a: ExTensor, b: ExTensor) raises -> ExTensor:
    """Element-wise greater-than-or-equal comparison with broadcasting.

    Args:
        a: First tensor.
        b: Second tensor.

    Returns:
        A new boolean tensor containing a >= b.

    Raises:
        Error if shapes are not broadcast-compatible or dtypes don't match.

    Examples:
        var a = full(List[Int](3, 4), 3.0, DType.float32)
        var b = full(List[Int](3, 4), 3.0, DType.float32)
        var c = greater_equal(a, b)  # Shape (3, 4), all True.
    """
    if a.dtype() != b.dtype():
        raise Error("Cannot compare tensors with different dtypes")

    var result_shape = broadcast_shapes(a.shape(), b.shape())
    var result = ExTensor(result_shape, DType.bool)

    # Compute broadcast strides
    var strides_a = compute_broadcast_strides(a.shape(), result_shape)
    var strides_b = compute_broadcast_strides(b.shape(), result_shape)

    # Calculate total elements in result
    var total_elems = 1
    for i in range(len(result_shape)):
        total_elems *= result_shape[i]

    # Precompute row-major strides for result shape
    var result_strides= List[Int]()
    var stride = 1
    for i in range(len(result_shape) - 1, -1, -1):
        result_strides.append(stride)
        stride *= result_shape[i]

    # Reverse to get correct order (left-to-right)
    var result_strides_final= List[Int]()
    for i in range(len(result_strides) - 1, -1, -1):
        result_strides_final.append(result_strides[i])

    # Iterate over all result elements
    for result_idx in range(total_elems):
        var idx_a = 0
        var idx_b = 0
        var temp_idx = result_idx

        # Convert flat index to multi-dimensional coordinates, then compute source indices
        for dim in range(len(result_shape)):
            var coord = temp_idx // result_strides_final[dim]
            temp_idx = temp_idx % result_strides_final[dim]

            idx_a += coord * strides_a[dim]
            idx_b += coord * strides_b[dim]

        # Perform comparison
        var a_val = a._get_float64(idx_a)
        var b_val = b._get_float64(idx_b)
        result._set_int64(result_idx, 1 if a_val >= b_val else 0)

    return result^
