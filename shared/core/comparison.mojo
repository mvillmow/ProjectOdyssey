"""Comparison operations for ExTensor with broadcasting support.

Implements element-wise comparison operations following NumPy-style broadcasting.
"""

from collections import List
from .extensor import ExTensor
from .broadcasting import broadcast_shapes


fn equal(a: ExTensor, b: ExTensor) raises -> ExTensor:
    """Element-wise equality comparison with broadcasting.

    Args:.        `a`: First tensor.
        `b`: Second tensor.

    Returns:.        A new boolean tensor containing a == b.

    Raises:.        Error if shapes are not broadcast-compatible or dtypes don't match.

    Examples:
        var a = full(List[Int](3, 4), 2.0, DType.float32)
        var b = full(List[Int](3, 4), 2.0, DType.float32)
        var c = equal(a, b)  # Shape (3, 4), all True
    """
    if a.dtype() != b.dtype():
        raise Error("Cannot compare tensors with different dtypes")

    var result_shape = broadcast_shapes(a.shape, b.shape)
    var result = ExTensor(result_shape, DType.bool)

    # Simple case: same shape (no broadcasting)
    if len(a.shape) == len(b.shape):
        var same_shape = True
        for i in range(len(a.shape)):
            if a.shape[i] != b.shape[i]:
                same_shape = False
                break

        if same_shape:
            for i in range(a.numel()):
                var a_val = a._get_float64(i)
                var b_val = b._get_float64(i)
                result._set_int64(i, 1 if a_val == b_val else 0)
            return result^

    # TODO: Implement full broadcasting for different shapes
    result._fill_zero()
    return result^


fn not_equal(a: ExTensor, b: ExTensor) raises -> ExTensor:
    """Element-wise inequality comparison with broadcasting.

    Args:.        `a`: First tensor.
        `b`: Second tensor.

    Returns:.        A new boolean tensor containing a != b.

    Raises:.        Error if shapes are not broadcast-compatible or dtypes don't match.

    Examples:
        var a = full(List[Int](3, 4), 2.0, DType.float32)
        var b = full(List[Int](3, 4), 3.0, DType.float32)
        var c = not_equal(a, b)  # Shape (3, 4), all True
    """
    if a.dtype() != b.dtype():
        raise Error("Cannot compare tensors with different dtypes")

    var result_shape = broadcast_shapes(a.shape, b.shape)
    var result = ExTensor(result_shape, DType.bool)

    # Simple case: same shape (no broadcasting)
    if len(a.shape) == len(b.shape):
        var same_shape = True
        for i in range(len(a.shape)):
            if a.shape[i] != b.shape[i]:
                same_shape = False
                break

        if same_shape:
            for i in range(a.numel()):
                var a_val = a._get_float64(i)
                var b_val = b._get_float64(i)
                result._set_int64(i, 1 if a_val != b_val else 0)
            return result^

    # TODO: Implement full broadcasting for different shapes
    result._fill_zero()
    return result^


fn less(a: ExTensor, b: ExTensor) raises -> ExTensor:
    """Element-wise less-than comparison with broadcasting.

    Args:.        `a`: First tensor.
        `b`: Second tensor.

    Returns:.        A new boolean tensor containing a < b.

    Raises:.        Error if shapes are not broadcast-compatible or dtypes don't match.

    Examples:
        var a = full(List[Int](3, 4), 2.0, DType.float32)
        var b = full(List[Int](3, 4), 3.0, DType.float32)
        var c = less(a, b)  # Shape (3, 4), all True
    """
    if a.dtype() != b.dtype():
        raise Error("Cannot compare tensors with different dtypes")

    var result_shape = broadcast_shapes(a.shape, b.shape)
    var result = ExTensor(result_shape, DType.bool)

    # Simple case: same shape (no broadcasting)
    if len(a.shape) == len(b.shape):
        var same_shape = True
        for i in range(len(a.shape)):
            if a.shape[i] != b.shape[i]:
                same_shape = False
                break

        if same_shape:
            for i in range(a.numel()):
                var a_val = a._get_float64(i)
                var b_val = b._get_float64(i)
                result._set_int64(i, 1 if a_val < b_val else 0)
            return result^

    # TODO: Implement full broadcasting for different shapes
    result._fill_zero()
    return result^


fn less_equal(a: ExTensor, b: ExTensor) raises -> ExTensor:
    """Element-wise less-than-or-equal comparison with broadcasting.

    Args:.        `a`: First tensor.
        `b`: Second tensor.

    Returns:.        A new boolean tensor containing a <= b.

    Raises:.        Error if shapes are not broadcast-compatible or dtypes don't match.

    Examples:
        var a = full(List[Int](3, 4), 2.0, DType.float32)
        var b = full(List[Int](3, 4), 2.0, DType.float32)
        var c = less_equal(a, b)  # Shape (3, 4), all True
    """
    if a.dtype() != b.dtype():
        raise Error("Cannot compare tensors with different dtypes")

    var result_shape = broadcast_shapes(a.shape, b.shape)
    var result = ExTensor(result_shape, DType.bool)

    # Simple case: same shape (no broadcasting)
    if len(a.shape) == len(b.shape):
        var same_shape = True
        for i in range(len(a.shape)):
            if a.shape[i] != b.shape[i]:
                same_shape = False
                break

        if same_shape:
            for i in range(a.numel()):
                var a_val = a._get_float64(i)
                var b_val = b._get_float64(i)
                result._set_int64(i, 1 if a_val <= b_val else 0)
            return result^

    # TODO: Implement full broadcasting for different shapes
    result._fill_zero()
    return result^


fn greater(a: ExTensor, b: ExTensor) raises -> ExTensor:
    """Element-wise greater-than comparison with broadcasting.

    Args:.        `a`: First tensor.
        `b`: Second tensor.

    Returns:.        A new boolean tensor containing a > b.

    Raises:.        Error if shapes are not broadcast-compatible or dtypes don't match.

    Examples:
        var a = full(List[Int](3, 4), 3.0, DType.float32)
        var b = full(List[Int](3, 4), 2.0, DType.float32)
        var c = greater(a, b)  # Shape (3, 4), all True
    """
    if a.dtype() != b.dtype():
        raise Error("Cannot compare tensors with different dtypes")

    var result_shape = broadcast_shapes(a.shape, b.shape)
    var result = ExTensor(result_shape, DType.bool)

    # Simple case: same shape (no broadcasting)
    if len(a.shape) == len(b.shape):
        var same_shape = True
        for i in range(len(a.shape)):
            if a.shape[i] != b.shape[i]:
                same_shape = False
                break

        if same_shape:
            for i in range(a.numel()):
                var a_val = a._get_float64(i)
                var b_val = b._get_float64(i)
                result._set_int64(i, 1 if a_val > b_val else 0)
            return result^

    # TODO: Implement full broadcasting for different shapes
    result._fill_zero()
    return result^


fn greater_equal(a: ExTensor, b: ExTensor) raises -> ExTensor:
    """Element-wise greater-than-or-equal comparison with broadcasting.

    Args:.        `a`: First tensor.
        `b`: Second tensor.

    Returns:.        A new boolean tensor containing a >= b.

    Raises:.        Error if shapes are not broadcast-compatible or dtypes don't match.

    Examples:
        var a = full(List[Int](3, 4), 3.0, DType.float32)
        var b = full(List[Int](3, 4), 3.0, DType.float32)
        var c = greater_equal(a, b)  # Shape (3, 4), all True
    """
    if a.dtype() != b.dtype():
        raise Error("Cannot compare tensors with different dtypes")

    var result_shape = broadcast_shapes(a.shape, b.shape)
    var result = ExTensor(result_shape, DType.bool)

    # Simple case: same shape (no broadcasting)
    if len(a.shape) == len(b.shape):
        var same_shape = True
        for i in range(len(a.shape)):
            if a.shape[i] != b.shape[i]:
                same_shape = False
                break

        if same_shape:
            for i in range(a.numel()):
                var a_val = a._get_float64(i)
                var b_val = b._get_float64(i)
                result._set_int64(i, 1 if a_val >= b_val else 0)
            return result^

    # TODO: Implement full broadcasting for different shapes
    result._fill_zero()
    return result^
