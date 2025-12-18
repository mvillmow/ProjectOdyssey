"""Reduction operations for ExTensor.

Implements operations that reduce tensors along specified axes
"""

from collections import List
from .extensor import ExTensor
from .reduction_utils import (
    compute_strides,
    linear_to_coords,
    coords_to_linear,
    map_result_to_input_coords,
    create_result_coords,
    compute_axis_strides,
    build_reduced_shape,
)
from .reduction_ops import (
    ReduceOp,
    ReduceBackwardOp,
    SumOp,
    MeanOp,
    MaxOp,
    MinOp,
    SumBackwardOp,
    MeanBackwardOp,
    MaxBackwardOp,
    MinBackwardOp,
)


# ============================================================================
# Generic Reduction Templates
# ============================================================================


fn _reduce_all_impl[dtype: DType, Op: ReduceOp](
    result: ExTensor, tensor: ExTensor, numel: Int
):
    """Generic dtype-specialized reduction over all elements."""
    var in_ptr = tensor._data.bitcast[Scalar[dtype]]()
    var out_ptr = result._data.bitcast[Scalar[dtype]]()
    var op = Op()
    var acc: Scalar[dtype] = Scalar[dtype](op.init_value())
    for i in range(numel):
        var val = Float64(in_ptr[i])
        var acc_float = Float64(acc)
        var result_float = op.apply(acc_float, val)
        acc = Scalar[dtype](result_float)
    var final_val = op.finalize(Float64(acc), numel)
    out_ptr[0] = Scalar[dtype](final_val)


fn _dispatch_reduce_all[Op: ReduceOp](
    result: ExTensor, tensor: ExTensor, numel: Int
) raises:
    """Generic runtime dispatch for reduction over all elements."""
    var dt = tensor.dtype()
    if dt == DType.float16:
        _reduce_all_impl[DType.float16, Op](result, tensor, numel)
    elif dt == DType.float32:
        _reduce_all_impl[DType.float32, Op](result, tensor, numel)
    elif dt == DType.float64:
        _reduce_all_impl[DType.float64, Op](result, tensor, numel)
    elif dt == DType.int32:
        _reduce_all_impl[DType.int32, Op](result, tensor, numel)
    elif dt == DType.int64:
        _reduce_all_impl[DType.int64, Op](result, tensor, numel)
    else:
        raise Error("reduce_all: unsupported dtype")


fn _reduce_axis_impl[dtype: DType, Op: ReduceOp](
    result: ExTensor,
    tensor: ExTensor,
    outer_size: Int,
    axis_size: Int,
    inner_size: Int,
):
    """Generic dtype-specialized reduction along axis."""
    var in_ptr = tensor._data.bitcast[Scalar[dtype]]()
    var out_ptr = result._data.bitcast[Scalar[dtype]]()
    var op = Op()

    for outer in range(outer_size):
        for inner in range(inner_size):
            var acc: Scalar[dtype] = Scalar[dtype](op.init_value())
            for k in range(axis_size):
                var input_idx = (
                    outer * axis_size * inner_size + k * inner_size + inner
                )
                var val = Float64(in_ptr[input_idx])
                var acc_float = Float64(acc)
                var result_float = op.apply(acc_float, val)
                acc = Scalar[dtype](result_float)
            var result_idx = outer * inner_size + inner
            var final_val = op.finalize(Float64(acc), axis_size)
            out_ptr[result_idx] = Scalar[dtype](final_val)


fn _dispatch_reduce_axis[Op: ReduceOp](
    result: ExTensor,
    tensor: ExTensor,
    outer_size: Int,
    axis_size: Int,
    inner_size: Int,
) raises:
    """Generic runtime dispatch for reduction along axis."""
    var dt = tensor.dtype()
    if dt == DType.float16:
        _reduce_axis_impl[DType.float16, Op](
            result, tensor, outer_size, axis_size, inner_size
        )
    elif dt == DType.float32:
        _reduce_axis_impl[DType.float32, Op](
            result, tensor, outer_size, axis_size, inner_size
        )
    elif dt == DType.float64:
        _reduce_axis_impl[DType.float64, Op](
            result, tensor, outer_size, axis_size, inner_size
        )
    elif dt == DType.int32:
        _reduce_axis_impl[DType.int32, Op](
            result, tensor, outer_size, axis_size, inner_size
        )
    elif dt == DType.int64:
        _reduce_axis_impl[DType.int64, Op](
            result, tensor, outer_size, axis_size, inner_size
        )
    else:
        raise Error("reduce_axis: unsupported dtype")


# ============================================================================
# Dtype-specialized reduction helpers
# ============================================================================


fn _sum_all_impl[dtype: DType](result: ExTensor, tensor: ExTensor, numel: Int):
    """Dtype-specialized sum over all elements."""
    var in_ptr = tensor._data.bitcast[Scalar[dtype]]()
    var out_ptr = result._data.bitcast[Scalar[dtype]]()
    var sum_val: Scalar[dtype] = 0
    for i in range(numel):
        sum_val += in_ptr[i]
    out_ptr[0] = sum_val


fn _dispatch_sum_all(result: ExTensor, tensor: ExTensor, numel: Int) raises:
    """Runtime dispatch for sum over all elements."""
    var dt = tensor.dtype()
    if dt == DType.float16:
        _sum_all_impl[DType.float16](result, tensor, numel)
    elif dt == DType.float32:
        _sum_all_impl[DType.float32](result, tensor, numel)
    elif dt == DType.float64:
        _sum_all_impl[DType.float64](result, tensor, numel)
    elif dt == DType.int32:
        _sum_all_impl[DType.int32](result, tensor, numel)
    elif dt == DType.int64:
        _sum_all_impl[DType.int64](result, tensor, numel)
    else:
        raise Error("sum: unsupported dtype")


fn _sum_axis_impl[
    dtype: DType
](
    result: ExTensor,
    tensor: ExTensor,
    outer_size: Int,
    axis_size: Int,
    inner_size: Int,
):
    """Dtype-specialized sum along axis."""
    var in_ptr = tensor._data.bitcast[Scalar[dtype]]()
    var out_ptr = result._data.bitcast[Scalar[dtype]]()

    for outer in range(outer_size):
        for inner in range(inner_size):
            var sum_val: Scalar[dtype] = 0
            for k in range(axis_size):
                var input_idx = (
                    outer * axis_size * inner_size + k * inner_size + inner
                )
                sum_val += in_ptr[input_idx]
            var result_idx = outer * inner_size + inner
            out_ptr[result_idx] = sum_val


fn _dispatch_sum_axis(
    result: ExTensor,
    tensor: ExTensor,
    outer_size: Int,
    axis_size: Int,
    inner_size: Int,
) raises:
    """Runtime dispatch for sum along axis."""
    var dt = tensor.dtype()
    if dt == DType.float16:
        _sum_axis_impl[DType.float16](
            result, tensor, outer_size, axis_size, inner_size
        )
    elif dt == DType.float32:
        _sum_axis_impl[DType.float32](
            result, tensor, outer_size, axis_size, inner_size
        )
    elif dt == DType.float64:
        _sum_axis_impl[DType.float64](
            result, tensor, outer_size, axis_size, inner_size
        )
    elif dt == DType.int32:
        _sum_axis_impl[DType.int32](
            result, tensor, outer_size, axis_size, inner_size
        )
    elif dt == DType.int64:
        _sum_axis_impl[DType.int64](
            result, tensor, outer_size, axis_size, inner_size
        )
    else:
        raise Error("sum: unsupported dtype")


fn _mean_divide_impl[dtype: DType](result: ExTensor, numel: Int, count: Int):
    """Dtype-specialized mean division."""
    var ptr = result._data.bitcast[Scalar[dtype]]()
    var scale = Scalar[dtype](1) / Scalar[dtype](count)
    for i in range(numel):
        ptr[i] = ptr[i] * scale


fn _dispatch_mean_divide(result: ExTensor, numel: Int, count: Int) raises:
    """Runtime dispatch for mean division."""
    var dt = result.dtype()
    if dt == DType.float16:
        _mean_divide_impl[DType.float16](result, numel, count)
    elif dt == DType.float32:
        _mean_divide_impl[DType.float32](result, numel, count)
    elif dt == DType.float64:
        _mean_divide_impl[DType.float64](result, numel, count)
    elif dt == DType.int32:
        _mean_divide_impl[DType.int32](result, numel, count)
    elif dt == DType.int64:
        _mean_divide_impl[DType.int64](result, numel, count)
    else:
        raise Error("mean: unsupported dtype")


fn _max_all_impl[dtype: DType](result: ExTensor, tensor: ExTensor, numel: Int):
    """Dtype-specialized max over all elements."""
    var in_ptr = tensor._data.bitcast[Scalar[dtype]]()
    var out_ptr = result._data.bitcast[Scalar[dtype]]()
    var max_val = in_ptr[0]
    for i in range(1, numel):
        var val = in_ptr[i]
        if val > max_val:
            max_val = val
    out_ptr[0] = max_val


fn _dispatch_max_all(result: ExTensor, tensor: ExTensor, numel: Int) raises:
    """Runtime dispatch for max over all elements."""
    var dt = tensor.dtype()
    if dt == DType.float16:
        _max_all_impl[DType.float16](result, tensor, numel)
    elif dt == DType.float32:
        _max_all_impl[DType.float32](result, tensor, numel)
    elif dt == DType.float64:
        _max_all_impl[DType.float64](result, tensor, numel)
    elif dt == DType.int32:
        _max_all_impl[DType.int32](result, tensor, numel)
    elif dt == DType.int64:
        _max_all_impl[DType.int64](result, tensor, numel)
    else:
        raise Error("max_reduce: unsupported dtype")


fn _max_axis_impl[
    dtype: DType
](
    result: ExTensor,
    tensor: ExTensor,
    outer_size: Int,
    axis_size: Int,
    inner_size: Int,
):
    """Dtype-specialized max along axis."""
    var in_ptr = tensor._data.bitcast[Scalar[dtype]]()
    var out_ptr = result._data.bitcast[Scalar[dtype]]()

    for outer in range(outer_size):
        for inner in range(inner_size):
            var first_idx = outer * axis_size * inner_size + inner
            var max_val = in_ptr[first_idx]
            for k in range(1, axis_size):
                var input_idx = (
                    outer * axis_size * inner_size + k * inner_size + inner
                )
                var val = in_ptr[input_idx]
                if val > max_val:
                    max_val = val
            var result_idx = outer * inner_size + inner
            out_ptr[result_idx] = max_val


fn _dispatch_max_axis(
    result: ExTensor,
    tensor: ExTensor,
    outer_size: Int,
    axis_size: Int,
    inner_size: Int,
) raises:
    """Runtime dispatch for max along axis."""
    var dt = tensor.dtype()
    if dt == DType.float16:
        _max_axis_impl[DType.float16](
            result, tensor, outer_size, axis_size, inner_size
        )
    elif dt == DType.float32:
        _max_axis_impl[DType.float32](
            result, tensor, outer_size, axis_size, inner_size
        )
    elif dt == DType.float64:
        _max_axis_impl[DType.float64](
            result, tensor, outer_size, axis_size, inner_size
        )
    elif dt == DType.int32:
        _max_axis_impl[DType.int32](
            result, tensor, outer_size, axis_size, inner_size
        )
    elif dt == DType.int64:
        _max_axis_impl[DType.int64](
            result, tensor, outer_size, axis_size, inner_size
        )
    else:
        raise Error("max_reduce: unsupported dtype")


fn _min_all_impl[dtype: DType](result: ExTensor, tensor: ExTensor, numel: Int):
    """Dtype-specialized min over all elements."""
    var in_ptr = tensor._data.bitcast[Scalar[dtype]]()
    var out_ptr = result._data.bitcast[Scalar[dtype]]()
    var min_val = in_ptr[0]
    for i in range(1, numel):
        var val = in_ptr[i]
        if val < min_val:
            min_val = val
    out_ptr[0] = min_val


fn _dispatch_min_all(result: ExTensor, tensor: ExTensor, numel: Int) raises:
    """Runtime dispatch for min over all elements."""
    var dt = tensor.dtype()
    if dt == DType.float16:
        _min_all_impl[DType.float16](result, tensor, numel)
    elif dt == DType.float32:
        _min_all_impl[DType.float32](result, tensor, numel)
    elif dt == DType.float64:
        _min_all_impl[DType.float64](result, tensor, numel)
    elif dt == DType.int32:
        _min_all_impl[DType.int32](result, tensor, numel)
    elif dt == DType.int64:
        _min_all_impl[DType.int64](result, tensor, numel)
    else:
        raise Error("min_reduce: unsupported dtype")


fn _min_axis_impl[
    dtype: DType
](
    result: ExTensor,
    tensor: ExTensor,
    outer_size: Int,
    axis_size: Int,
    inner_size: Int,
):
    """Dtype-specialized min along axis."""
    var in_ptr = tensor._data.bitcast[Scalar[dtype]]()
    var out_ptr = result._data.bitcast[Scalar[dtype]]()

    for outer in range(outer_size):
        for inner in range(inner_size):
            var first_idx = outer * axis_size * inner_size + inner
            var min_val = in_ptr[first_idx]
            for k in range(1, axis_size):
                var input_idx = (
                    outer * axis_size * inner_size + k * inner_size + inner
                )
                var val = in_ptr[input_idx]
                if val < min_val:
                    min_val = val
            var result_idx = outer * inner_size + inner
            out_ptr[result_idx] = min_val


fn _dispatch_min_axis(
    result: ExTensor,
    tensor: ExTensor,
    outer_size: Int,
    axis_size: Int,
    inner_size: Int,
) raises:
    """Runtime dispatch for min along axis."""
    var dt = tensor.dtype()
    if dt == DType.float16:
        _min_axis_impl[DType.float16](
            result, tensor, outer_size, axis_size, inner_size
        )
    elif dt == DType.float32:
        _min_axis_impl[DType.float32](
            result, tensor, outer_size, axis_size, inner_size
        )
    elif dt == DType.float64:
        _min_axis_impl[DType.float64](
            result, tensor, outer_size, axis_size, inner_size
        )
    elif dt == DType.int32:
        _min_axis_impl[DType.int32](
            result, tensor, outer_size, axis_size, inner_size
        )
    elif dt == DType.int64:
        _min_axis_impl[DType.int64](
            result, tensor, outer_size, axis_size, inner_size
        )
    else:
        raise Error("min_reduce: unsupported dtype")


fn sum(
    tensor: ExTensor, axis: Int = -1, keepdims: Bool = False
) raises -> ExTensor:
    """Sum tensor elements along an axis.

    Args:
            tensor: Input tensor.
            axis: Axis to reduce (-1 for all axes).
            keepdims: Whether to keep reduced dimensions as size 1.

    Returns:
            A new tensor with sum along specified axis.

    Examples:
        ```
            var t = ones([3, 4], DType.float32)
            var s = sum(t, axis=-1)  # Sum all elements -> scalar 12.0
            var row_sums = sum(t, axis=1)  # Sum along rows -> shape (3,)
        ```
    """
    if axis == -1:
        # Sum all elements
        var result_shape = List[Int](capacity=tensor.dim() if keepdims else 0)
        if keepdims:
            for _ in range(tensor.dim()):
                result_shape.append(1)
        var result = ExTensor(result_shape, tensor.dtype())

        _dispatch_reduce_all[SumOp](result, tensor, tensor.numel())
        return result^
    else:
        # Sum along specific axis
        if axis < 0 or axis >= tensor.dim():
            raise Error(
                "Axis "
                + String(axis)
                + " is out of bounds for tensor with "
                + String(tensor.dim())
                + " dimensions"
            )

        # Use direct index computation (no coordinate allocations)
        var result_shape = build_reduced_shape(tensor.shape(), axis, keepdims)
        var result = ExTensor(result_shape, tensor.dtype())
        result._fill_zero()

        var sizes = compute_axis_strides(tensor.shape(), axis)
        var outer_size = sizes[0]
        var axis_size = sizes[1]
        var inner_size = sizes[2]

        _dispatch_reduce_axis[SumOp](result, tensor, outer_size, axis_size, inner_size)
        return result^


fn mean(
    tensor: ExTensor, axis: Int = -1, keepdims: Bool = False
) raises -> ExTensor:
    """Compute mean of tensor elements along an axis.

    Args:
            tensor: Input tensor.
            axis: Axis to reduce (-1 for all axes).
            keepdims: Whether to keep reduced dimensions as size 1.

    Returns:
            A new tensor with mean along specified axis.

    Examples:
        ```
            var t = ones([3, 4], DType.float32)
            var m = mean(t)  # Mean of all elements -> scalar 1.0
        ```
    """
    if axis == -1:
        # Mean of all elements
        var sum_result = sum(tensor, axis, keepdims)
        _dispatch_mean_divide(sum_result, 1, tensor.numel())
        return sum_result^
    else:
        # Mean along specific axis
        if axis < 0 or axis >= tensor.dim():
            raise Error(
                "Axis "
                + String(axis)
                + " is out of bounds for tensor with "
                + String(tensor.dim())
                + " dimensions"
            )

        # Compute sum along axis
        var sum_result = sum(tensor, axis, keepdims)

        # Divide by count along the reduction axis
        _dispatch_mean_divide(
            sum_result, sum_result.numel(), tensor.shape()[axis]
        )

        return sum_result^


fn max_reduce(
    tensor: ExTensor, axis: Int = -1, keepdims: Bool = False
) raises -> ExTensor:
    """Find maximum of tensor elements along an axis.

    Args:
            tensor: Input tensor.
            axis: Axis to reduce (-1 for all axes).
            keepdims: Whether to keep reduced dimensions as size 1.

    Returns:
            A new tensor with maximum along specified axis.

    Examples:
        ```
            var t = arange(0.0, 12.0, 1.0, DType.float32)
            var m = max_reduce(t)  # Maximum element -> scalar 11.0
        ```
    """
    if axis == -1:
        # Max of all elements
        var result_shape = List[Int](capacity=tensor.dim() if keepdims else 0)
        if keepdims:
            for _ in range(tensor.dim()):
                result_shape.append(1)
        var result = ExTensor(result_shape, tensor.dtype())

        _dispatch_reduce_all[MaxOp](result, tensor, tensor.numel())
        return result^
    else:
        # Max along specific axis
        if axis < 0 or axis >= tensor.dim():
            raise Error(
                "Axis "
                + String(axis)
                + " is out of bounds for tensor with "
                + String(tensor.dim())
                + " dimensions"
            )

        var result_shape = build_reduced_shape(tensor.shape(), axis, keepdims)
        var result = ExTensor(result_shape, tensor.dtype())

        var sizes = compute_axis_strides(tensor.shape(), axis)
        var outer_size = sizes[0]
        var axis_size = sizes[1]
        var inner_size = sizes[2]

        _dispatch_reduce_axis[MaxOp](result, tensor, outer_size, axis_size, inner_size)
        return result^


fn min_reduce(
    tensor: ExTensor, axis: Int = -1, keepdims: Bool = False
) raises -> ExTensor:
    """Find minimum of tensor elements along an axis.

    Args:
            tensor: Input tensor.
            axis: Axis to reduce (-1 for all axes).
            keepdims: Whether to keep reduced dimensions as size 1.

    Returns:
            A new tensor with minimum along specified axis.

    Examples:
        ```
            var t = arange(0.0, 12.0, 1.0, DType.float32)
            var m = min_reduce(t)  # Minimum element -> scalar 0.0
        ```
    """
    if axis == -1:
        # Min of all elements
        var result_shape = List[Int](capacity=tensor.dim() if keepdims else 0)
        if keepdims:
            for _ in range(tensor.dim()):
                result_shape.append(1)
        var result = ExTensor(result_shape, tensor.dtype())

        _dispatch_reduce_all[MinOp](result, tensor, tensor.numel())
        return result^
    else:
        # Min along specific axis
        if axis < 0 or axis >= tensor.dim():
            raise Error(
                "Axis "
                + String(axis)
                + " is out of bounds for tensor with "
                + String(tensor.dim())
                + " dimensions"
            )

        var result_shape = build_reduced_shape(tensor.shape(), axis, keepdims)
        var result = ExTensor(result_shape, tensor.dtype())

        var sizes = compute_axis_strides(tensor.shape(), axis)
        var outer_size = sizes[0]
        var axis_size = sizes[1]
        var inner_size = sizes[2]

        _dispatch_reduce_axis[MinOp](result, tensor, outer_size, axis_size, inner_size)
        return result^


# ============================================================================
# Generic Backward Pass (Gradient Computation)
# ============================================================================


fn reduce_backward[Op: ReduceBackwardOp](
    grad_output: ExTensor, x: ExTensor, axis: Int = -1
) raises -> ExTensor:
    """Generic backward pass for reduction operations.

    Consolidates the coordinate transformation logic used by all backward passes.
    The operation is determined by the Op template parameter.

    Args:
        grad_output: Gradient from upstream (∂L/∂Y) - reduced tensor.
        x: Original input tensor before reduction.
        axis: Axis along which reduction was computed (-1 for all axes).

    Returns:
        Gradient w.r.t. input (∂L/∂X) - broadcast back to input_shape.
    """
    var input_shape = x.shape()
    var result = ExTensor(input_shape, grad_output.dtype())
    var op = Op()

    if axis == -1:
        # Reduction over all elements - collect all values for extremum ops
        var grad_val = grad_output._get_float64(0)
        var all_values = List[Float64]()
        for i in range(x.numel()):
            all_values.append(x._get_float64(i))

        for i in range(result.numel()):
            var input_val = x._get_float64(i)
            var grad = op.compute_gradient(
                grad_val, input_val, all_values, x.numel()
            )
            result._set_float64(i, grad)
    else:
        # Reduction along specific axis - shared coordinate transformation
        var ndim = len(input_shape)
        var normalized_axis = axis if axis >= 0 else ndim + axis

        # Validate axis
        if normalized_axis < 0 or normalized_axis >= ndim:
            raise Error(
                "Axis "
                + String(axis)
                + " is out of bounds for tensor with "
                + String(ndim)
                + " dimensions"
            )

        var strides = compute_strides(input_shape)
        var axis_size = input_shape[normalized_axis]

        for result_idx in range(x.numel()):
            # Convert linear index to coordinates
            var coords = linear_to_coords(result_idx, input_shape)

            # Map to grad_output coordinates (remove axis dimension)
            var grad_dim = grad_output.dim()
            var grad_coords = List[Int](capacity=grad_dim)
            for _ in range(grad_dim):
                grad_coords.append(0)
            var coord_idx = 0
            for i in range(ndim):
                if i != normalized_axis:
                    grad_coords[coord_idx] = coords[i]
                    coord_idx += 1

            # Convert to linear index in grad_output
            var grad_strides = compute_strides(grad_output.shape())
            var grad_idx = coords_to_linear(grad_coords, grad_strides)
            var grad_val = grad_output._get_float64(grad_idx)

            # Collect values along the reduction axis
            var axis_values = List[Float64]()
            for k in range(axis_size):
                var test_coords = List[Int](coords)
                test_coords[normalized_axis] = k
                var test_idx = coords_to_linear(test_coords, strides)
                axis_values.append(x._get_float64(test_idx))

            # Use Op-specific gradient computation
            var input_val = x._get_float64(result_idx)
            var grad = op.compute_gradient(grad_val, input_val, axis_values, axis_size)
            result._set_float64(result_idx, grad)

    return result^


# ============================================================================
# Backward Pass (Gradient Computation)
# ============================================================================


fn sum_backward(
    grad_output: ExTensor, x: ExTensor, axis: Int = -1
) raises -> ExTensor:
    """Compute gradient for sum reduction.

        For Y = sum(X, axis), given ∂L/∂Y, computes:
            ∂L/∂X = broadcast(∂L/∂Y, input_shape)

        The gradient broadcasts the reduced gradient back to the original input shape
        Each element of the input contributes equally to the sum, so gradient is 1

    Args:
            grad_output: Gradient from upstream (∂L/∂Y) - reduced tensor.
            x: Original input tensor before reduction.
            axis: Axis along which sum was computed (-1 for all axes).

    Returns:
            Gradient w.r.t. input (∂L/∂X) - broadcast back to input_shape.

    Examples:
        ```
            # Sum all elements
            var x = ones([3, 4], DType.float32)
            var y = sum(x, axis=-1)  # Scalar
            var grad_y = ones(List[Int](), DType.float32)  # Scalar gradient
            var grad_x = sum_backward(grad_y, x, axis=-1)  # Shape (3, 4)

            # Sum along specific axis
            var x2 = ones([3, 4], DType.float32)
            var y2 = sum(x2, axis=1)  # Shape (3,)
            var grad_y2 = ones(List[Int](), DType.float32)
            var grad_x2 = sum_backward(grad_y2, x2, axis=1)  # Shape (3, 4)
        ```
    """
    return reduce_backward[SumBackwardOp](grad_output, x, axis)


fn mean_backward(
    grad_output: ExTensor, x: ExTensor, axis: Int = -1
) raises -> ExTensor:
    """Compute gradient for mean reduction.

        For Y = mean(X, axis), given ∂L/∂Y, computes:
            ∂L/∂X = broadcast(∂L/∂Y, input_shape) / N

        where N is the number of elements that were averaged

        Similar to sum_backward, but scaled by 1/N since each input element
        contributes 1/N to the mean

    Args:
            grad_output: Gradient from upstream (∂L/∂Y) - reduced tensor.
            x: Original input tensor before reduction.
            axis: Axis along which mean was computed (-1 for all axes).

    Returns:
            Gradient w.r.t. input (∂L/∂X) - broadcast and scaled.

    Examples:
        ```
            var x = ones([3, 4], DType.float32)
            var y = mean(x, axis=-1)  # Scalar mean
            var grad_y = ones(List[Int](), DType.float32)
            var grad_x = mean_backward(grad_y, x, axis=-1)
            # Each element gets gradient / 12
        ```
    """
    return reduce_backward[MeanBackwardOp](grad_output, x, axis)


fn max_reduce_backward(
    grad_output: ExTensor, x: ExTensor, axis: Int = -1
) raises -> ExTensor:
    """Compute gradient for max reduction.

        For Y = max_reduce(X, axis), given ∂L/∂Y, computes:
            ∂L/∂X - Gradient flows only to maximum element(s)

        If multiple elements are maximum, gradient is split equally among them
        This is the standard behavior for max pooling backward pass

    Args:
            grad_output: Gradient from upstream (∂L/∂Y) - reduced tensor.
            x: Input from forward pass (before reduction).
            axis: Axis along which max was computed (-1 for all axes).

    Returns:
            Gradient w.r.t. input (∂L/∂X).

    Examples:
        ```
            # Max over all elements
            var x = tensor([1.0, 3.0, 2.0, 3.0])  # Two max values at indices 1, 3
            var y = max_reduce(x, axis=-1)  # Scalar: 3.0
            var grad_y = ones([])  # Gradient: 1.0
            var grad_x = max_reduce_backward(grad_y, x, axis=-1)
            # grad_x = [0.0, 0.5, 0.0, 0.5]  # Split equally between the two 3.0s

            # Max along axis
            var x2 = tensor([[1.0, 3.0], [2.0, 1.0]])
            var y2 = max_reduce(x2, axis=1)  # [3.0, 2.0]
            var grad_y2 = ones([2])
            var grad_x2 = max_reduce_backward(grad_y2, x2, axis=1)
            # grad_x2 = [[0.0, 1.0], [1.0, 0.0]]
        ```
    """
    return reduce_backward[MaxBackwardOp](grad_output, x, axis)


fn min_reduce_backward(
    grad_output: ExTensor, x: ExTensor, axis: Int = -1
) raises -> ExTensor:
    """Compute gradient for min reduction.

        For Y = min_reduce(X, axis), given ∂L/∂Y, computes:
            ∂L/∂X - Gradient flows only to minimum element(s)

        If multiple elements are minimum, gradient is split equally among them
        This is analogous to max pooling but for minimum values

    Args:
            grad_output: Gradient from upstream (∂L/∂Y) - reduced tensor.
            x: Input from forward pass (before reduction).
            axis: Axis along which min was computed (-1 for all axes).

    Returns:
            Gradient w.r.t. input (∂L/∂X).

    Examples:
        ```
            var x = tensor([3.0, 1.0, 2.0, 1.0])  # Two min values at indices 1, 3
            var y = min_reduce(x, axis=-1)  # Scalar: 1.0
            var grad_y = ones([])  # Gradient: 1.0
            var grad_x = min_reduce_backward(grad_y, x, axis=-1)
            # grad_x = [0.0, 0.5, 0.0, 0.5]  # Split equally between the two 1.0s
        ```
    """
    return reduce_backward[MinBackwardOp](grad_output, x, axis)
