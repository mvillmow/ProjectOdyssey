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


# ============================================================================
# Dtype-Specialized Reduction Helpers (Eliminates Float64 Conversions)
# ============================================================================


fn _sum_all_impl[dtype: DType](result: ExTensor, tensor: ExTensor, numel: Int):
    """Compile-time specialized sum over all elements.

    Uses Float64 accumulation for float16/float32 to maintain precision.

    Parameters:
        dtype: Compile-time dtype parameter.

    Args:
        result: Pre-allocated result tensor (scalar or 1-element).
        tensor: Input tensor.
        numel: Number of elements to sum.
    """
    var in_ptr = tensor._data.bitcast[Scalar[dtype]]()
    var out_ptr = result._data.bitcast[Scalar[dtype]]()

    @parameter
    if dtype == DType.float16 or dtype == DType.float32:
        var sum_val: Float64 = 0.0
        for i in range(numel):
            sum_val += Float64(in_ptr[i])
        out_ptr[0] = Scalar[dtype](sum_val)
    else:
        var sum_val = Scalar[dtype](0)
        for i in range(numel):
            sum_val += in_ptr[i]
        out_ptr[0] = sum_val


fn _dispatch_sum_all(result: ExTensor, tensor: ExTensor, numel: Int) raises:
    """Runtime dispatch for sum over all elements."""
    if tensor._dtype == DType.float16:
        _sum_all_impl[DType.float16](result, tensor, numel)
    elif tensor._dtype == DType.float32:
        _sum_all_impl[DType.float32](result, tensor, numel)
    elif tensor._dtype == DType.float64:
        _sum_all_impl[DType.float64](result, tensor, numel)
    elif tensor._dtype == DType.int8:
        _sum_all_impl[DType.int8](result, tensor, numel)
    elif tensor._dtype == DType.int16:
        _sum_all_impl[DType.int16](result, tensor, numel)
    elif tensor._dtype == DType.int32:
        _sum_all_impl[DType.int32](result, tensor, numel)
    elif tensor._dtype == DType.int64:
        _sum_all_impl[DType.int64](result, tensor, numel)
    else:
        raise Error("sum: unsupported dtype")


fn _sum_axis_impl[dtype: DType](
    result: ExTensor,
    tensor: ExTensor,
    outer_size: Int,
    axis_size: Int,
    inner_size: Int,
):
    """Compile-time specialized sum along axis.

    Uses Float64 accumulation for float16/float32 to maintain precision.

    Parameters:
        dtype: Compile-time dtype parameter.

    Args:
        result: Pre-allocated result tensor.
        tensor: Input tensor.
        outer_size: Product of dimensions before axis.
        axis_size: Size of axis dimension.
        inner_size: Product of dimensions after axis.
    """
    var in_ptr = tensor._data.bitcast[Scalar[dtype]]()
    var out_ptr = result._data.bitcast[Scalar[dtype]]()

    @parameter
    if dtype == DType.float16 or dtype == DType.float32:
        for outer in range(outer_size):
            for inner in range(inner_size):
                var sum_val: Float64 = 0.0
                for k in range(axis_size):
                    var input_idx = outer * axis_size * inner_size + k * inner_size + inner
                    sum_val += Float64(in_ptr[input_idx])
                var result_idx = outer * inner_size + inner
                out_ptr[result_idx] = Scalar[dtype](sum_val)
    else:
        for outer in range(outer_size):
            for inner in range(inner_size):
                var sum_val = Scalar[dtype](0)
                for k in range(axis_size):
                    var input_idx = outer * axis_size * inner_size + k * inner_size + inner
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
    if tensor._dtype == DType.float16:
        _sum_axis_impl[DType.float16](result, tensor, outer_size, axis_size, inner_size)
    elif tensor._dtype == DType.float32:
        _sum_axis_impl[DType.float32](result, tensor, outer_size, axis_size, inner_size)
    elif tensor._dtype == DType.float64:
        _sum_axis_impl[DType.float64](result, tensor, outer_size, axis_size, inner_size)
    elif tensor._dtype == DType.int8:
        _sum_axis_impl[DType.int8](result, tensor, outer_size, axis_size, inner_size)
    elif tensor._dtype == DType.int16:
        _sum_axis_impl[DType.int16](result, tensor, outer_size, axis_size, inner_size)
    elif tensor._dtype == DType.int32:
        _sum_axis_impl[DType.int32](result, tensor, outer_size, axis_size, inner_size)
    elif tensor._dtype == DType.int64:
        _sum_axis_impl[DType.int64](result, tensor, outer_size, axis_size, inner_size)
    else:
        raise Error("sum: unsupported dtype")


fn _mean_divide_impl[dtype: DType](result: ExTensor, count: Int):
    """Divide result tensor by count for mean computation.

    Parameters:
        dtype: Compile-time dtype parameter.

    Args:
        result: Result tensor to modify in-place.
        count: Divisor for mean computation.
    """
    var ptr = result._data.bitcast[Scalar[dtype]]()
    var numel = result.numel()

    @parameter
    if dtype == DType.float16 or dtype == DType.float32:
        var scale = 1.0 / Float64(count)
        for i in range(numel):
            ptr[i] = Scalar[dtype](Float64(ptr[i]) * scale)
    else:
        # For float64 and integers, divide directly
        for i in range(numel):
            ptr[i] = ptr[i] / Scalar[dtype](count)


fn _dispatch_mean_divide(result: ExTensor, count: Int) raises:
    """Runtime dispatch for mean division."""
    if result._dtype == DType.float16:
        _mean_divide_impl[DType.float16](result, count)
    elif result._dtype == DType.float32:
        _mean_divide_impl[DType.float32](result, count)
    elif result._dtype == DType.float64:
        _mean_divide_impl[DType.float64](result, count)
    elif result._dtype == DType.int8:
        _mean_divide_impl[DType.int8](result, count)
    elif result._dtype == DType.int16:
        _mean_divide_impl[DType.int16](result, count)
    elif result._dtype == DType.int32:
        _mean_divide_impl[DType.int32](result, count)
    elif result._dtype == DType.int64:
        _mean_divide_impl[DType.int64](result, count)
    else:
        raise Error("mean: unsupported dtype")


fn _max_all_impl[dtype: DType](result: ExTensor, tensor: ExTensor, numel: Int):
    """Compile-time specialized max over all elements.

    Parameters:
        dtype: Compile-time dtype parameter.

    Args:
        result: Pre-allocated result tensor (scalar or 1-element).
        tensor: Input tensor.
        numel: Number of elements.
    """
    var in_ptr = tensor._data.bitcast[Scalar[dtype]]()
    var out_ptr = result._data.bitcast[Scalar[dtype]]()

    var max_val = in_ptr[0]
    for i in range(1, numel):
        if in_ptr[i] > max_val:
            max_val = in_ptr[i]
    out_ptr[0] = max_val


fn _dispatch_max_all(result: ExTensor, tensor: ExTensor, numel: Int) raises:
    """Runtime dispatch for max over all elements."""
    if tensor._dtype == DType.float16:
        _max_all_impl[DType.float16](result, tensor, numel)
    elif tensor._dtype == DType.float32:
        _max_all_impl[DType.float32](result, tensor, numel)
    elif tensor._dtype == DType.float64:
        _max_all_impl[DType.float64](result, tensor, numel)
    elif tensor._dtype == DType.int8:
        _max_all_impl[DType.int8](result, tensor, numel)
    elif tensor._dtype == DType.int16:
        _max_all_impl[DType.int16](result, tensor, numel)
    elif tensor._dtype == DType.int32:
        _max_all_impl[DType.int32](result, tensor, numel)
    elif tensor._dtype == DType.int64:
        _max_all_impl[DType.int64](result, tensor, numel)
    else:
        raise Error("max: unsupported dtype")


fn _max_axis_impl[dtype: DType](
    result: ExTensor,
    tensor: ExTensor,
    outer_size: Int,
    axis_size: Int,
    inner_size: Int,
):
    """Compile-time specialized max along axis.

    Parameters:
        dtype: Compile-time dtype parameter.

    Args:
        result: Pre-allocated result tensor.
        tensor: Input tensor.
        outer_size: Product of dimensions before axis.
        axis_size: Size of axis dimension.
        inner_size: Product of dimensions after axis.
    """
    var in_ptr = tensor._data.bitcast[Scalar[dtype]]()
    var out_ptr = result._data.bitcast[Scalar[dtype]]()

    for outer in range(outer_size):
        for inner in range(inner_size):
            # Initialize with first value along axis
            var first_idx = outer * axis_size * inner_size + inner
            var max_val = in_ptr[first_idx]
            # Compare with remaining values
            for k in range(1, axis_size):
                var input_idx = outer * axis_size * inner_size + k * inner_size + inner
                if in_ptr[input_idx] > max_val:
                    max_val = in_ptr[input_idx]
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
    if tensor._dtype == DType.float16:
        _max_axis_impl[DType.float16](result, tensor, outer_size, axis_size, inner_size)
    elif tensor._dtype == DType.float32:
        _max_axis_impl[DType.float32](result, tensor, outer_size, axis_size, inner_size)
    elif tensor._dtype == DType.float64:
        _max_axis_impl[DType.float64](result, tensor, outer_size, axis_size, inner_size)
    elif tensor._dtype == DType.int8:
        _max_axis_impl[DType.int8](result, tensor, outer_size, axis_size, inner_size)
    elif tensor._dtype == DType.int16:
        _max_axis_impl[DType.int16](result, tensor, outer_size, axis_size, inner_size)
    elif tensor._dtype == DType.int32:
        _max_axis_impl[DType.int32](result, tensor, outer_size, axis_size, inner_size)
    elif tensor._dtype == DType.int64:
        _max_axis_impl[DType.int64](result, tensor, outer_size, axis_size, inner_size)
    else:
        raise Error("max: unsupported dtype")


fn _min_all_impl[dtype: DType](result: ExTensor, tensor: ExTensor, numel: Int):
    """Compile-time specialized min over all elements.

    Parameters:
        dtype: Compile-time dtype parameter.

    Args:
        result: Pre-allocated result tensor (scalar or 1-element).
        tensor: Input tensor.
        numel: Number of elements.
    """
    var in_ptr = tensor._data.bitcast[Scalar[dtype]]()
    var out_ptr = result._data.bitcast[Scalar[dtype]]()

    var min_val = in_ptr[0]
    for i in range(1, numel):
        if in_ptr[i] < min_val:
            min_val = in_ptr[i]
    out_ptr[0] = min_val


fn _dispatch_min_all(result: ExTensor, tensor: ExTensor, numel: Int) raises:
    """Runtime dispatch for min over all elements."""
    if tensor._dtype == DType.float16:
        _min_all_impl[DType.float16](result, tensor, numel)
    elif tensor._dtype == DType.float32:
        _min_all_impl[DType.float32](result, tensor, numel)
    elif tensor._dtype == DType.float64:
        _min_all_impl[DType.float64](result, tensor, numel)
    elif tensor._dtype == DType.int8:
        _min_all_impl[DType.int8](result, tensor, numel)
    elif tensor._dtype == DType.int16:
        _min_all_impl[DType.int16](result, tensor, numel)
    elif tensor._dtype == DType.int32:
        _min_all_impl[DType.int32](result, tensor, numel)
    elif tensor._dtype == DType.int64:
        _min_all_impl[DType.int64](result, tensor, numel)
    else:
        raise Error("min: unsupported dtype")


fn _min_axis_impl[dtype: DType](
    result: ExTensor,
    tensor: ExTensor,
    outer_size: Int,
    axis_size: Int,
    inner_size: Int,
):
    """Compile-time specialized min along axis.

    Parameters:
        dtype: Compile-time dtype parameter.

    Args:
        result: Pre-allocated result tensor.
        tensor: Input tensor.
        outer_size: Product of dimensions before axis.
        axis_size: Size of axis dimension.
        inner_size: Product of dimensions after axis.
    """
    var in_ptr = tensor._data.bitcast[Scalar[dtype]]()
    var out_ptr = result._data.bitcast[Scalar[dtype]]()

    for outer in range(outer_size):
        for inner in range(inner_size):
            # Initialize with first value along axis
            var first_idx = outer * axis_size * inner_size + inner
            var min_val = in_ptr[first_idx]
            # Compare with remaining values
            for k in range(1, axis_size):
                var input_idx = outer * axis_size * inner_size + k * inner_size + inner
                if in_ptr[input_idx] < min_val:
                    min_val = in_ptr[input_idx]
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
    if tensor._dtype == DType.float16:
        _min_axis_impl[DType.float16](result, tensor, outer_size, axis_size, inner_size)
    elif tensor._dtype == DType.float32:
        _min_axis_impl[DType.float32](result, tensor, outer_size, axis_size, inner_size)
    elif tensor._dtype == DType.float64:
        _min_axis_impl[DType.float64](result, tensor, outer_size, axis_size, inner_size)
    elif tensor._dtype == DType.int8:
        _min_axis_impl[DType.int8](result, tensor, outer_size, axis_size, inner_size)
    elif tensor._dtype == DType.int16:
        _min_axis_impl[DType.int16](result, tensor, outer_size, axis_size, inner_size)
    elif tensor._dtype == DType.int32:
        _min_axis_impl[DType.int32](result, tensor, outer_size, axis_size, inner_size)
    elif tensor._dtype == DType.int64:
        _min_axis_impl[DType.int64](result, tensor, outer_size, axis_size, inner_size)
    else:
        raise Error("min: unsupported dtype")


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

        # Compute sum using dtype-specialized implementation
        _dispatch_sum_all(result, tensor, tensor.numel())
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

        # Compute outer/inner sizes for direct indexing
        var sizes = compute_axis_strides(tensor.shape(), axis)
        var outer_size = sizes[0]
        var axis_size = sizes[1]
        var inner_size = sizes[2]

        # Use dtype-specialized implementation
        _dispatch_sum_axis(result, tensor, outer_size, axis_size, inner_size)

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
        # Divide by count using dtype-specialized implementation
        _dispatch_mean_divide(sum_result, tensor.numel())
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

        # Divide by count using dtype-specialized implementation
        _dispatch_mean_divide(sum_result, tensor.shape()[axis])

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

        # Find maximum using dtype-specialized implementation
        _dispatch_max_all(result, tensor, tensor.numel())
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

        # Use direct index computation (no coordinate allocations)
        var result_shape = build_reduced_shape(tensor.shape(), axis, keepdims)
        var result = ExTensor(result_shape, tensor.dtype())

        # Compute outer/inner sizes for direct indexing
        var sizes = compute_axis_strides(tensor.shape(), axis)
        var outer_size = sizes[0]
        var axis_size = sizes[1]
        var inner_size = sizes[2]

        # Use dtype-specialized implementation
        _dispatch_max_axis(result, tensor, outer_size, axis_size, inner_size)

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

        # Find minimum using dtype-specialized implementation
        _dispatch_min_all(result, tensor, tensor.numel())
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

        # Use direct index computation (no coordinate allocations)
        var result_shape = build_reduced_shape(tensor.shape(), axis, keepdims)
        var result = ExTensor(result_shape, tensor.dtype())

        # Compute outer/inner sizes for direct indexing
        var sizes = compute_axis_strides(tensor.shape(), axis)
        var outer_size = sizes[0]
        var axis_size = sizes[1]
        var inner_size = sizes[2]

        # Use dtype-specialized implementation
        _dispatch_min_axis(result, tensor, outer_size, axis_size, inner_size)

        return result^


# ============================================================================
# Backward Pass Dtype-Specialized Helpers
# ============================================================================


fn _sum_backward_all_impl[dtype: DType](
    result: ExTensor, grad_output: ExTensor, numel: Int
):
    """Broadcast scalar gradient to all elements.

    Parameters:
        dtype: Compile-time dtype parameter.

    Args:
        result: Pre-allocated result tensor.
        grad_output: Scalar gradient tensor.
        numel: Number of result elements.
    """
    var grad_ptr = grad_output._data.bitcast[Scalar[dtype]]()
    var out_ptr = result._data.bitcast[Scalar[dtype]]()
    var grad_val = grad_ptr[0]
    for i in range(numel):
        out_ptr[i] = grad_val


fn _dispatch_sum_backward_all(result: ExTensor, grad_output: ExTensor, numel: Int) raises:
    """Runtime dispatch for sum_backward all elements."""
    if result._dtype == DType.float16:
        _sum_backward_all_impl[DType.float16](result, grad_output, numel)
    elif result._dtype == DType.float32:
        _sum_backward_all_impl[DType.float32](result, grad_output, numel)
    elif result._dtype == DType.float64:
        _sum_backward_all_impl[DType.float64](result, grad_output, numel)
    elif result._dtype == DType.int8:
        _sum_backward_all_impl[DType.int8](result, grad_output, numel)
    elif result._dtype == DType.int16:
        _sum_backward_all_impl[DType.int16](result, grad_output, numel)
    elif result._dtype == DType.int32:
        _sum_backward_all_impl[DType.int32](result, grad_output, numel)
    elif result._dtype == DType.int64:
        _sum_backward_all_impl[DType.int64](result, grad_output, numel)
    else:
        raise Error("sum_backward: unsupported dtype")


fn _sum_backward_axis_element_impl[dtype: DType](
    result: ExTensor, grad_output: ExTensor, result_idx: Int, grad_idx: Int
):
    """Copy single gradient element for sum_backward axis case.

    Parameters:
        dtype: Compile-time dtype parameter.

    Args:
        result: Result tensor.
        grad_output: Gradient tensor.
        result_idx: Index in result tensor.
        grad_idx: Index in gradient tensor.
    """
    var grad_ptr = grad_output._data.bitcast[Scalar[dtype]]()
    var out_ptr = result._data.bitcast[Scalar[dtype]]()
    out_ptr[result_idx] = grad_ptr[grad_idx]


fn _dispatch_sum_backward_axis_element(
    result: ExTensor, grad_output: ExTensor, result_idx: Int, grad_idx: Int
) raises:
    """Runtime dispatch for sum_backward axis element copy."""
    if result._dtype == DType.float16:
        _sum_backward_axis_element_impl[DType.float16](result, grad_output, result_idx, grad_idx)
    elif result._dtype == DType.float32:
        _sum_backward_axis_element_impl[DType.float32](result, grad_output, result_idx, grad_idx)
    elif result._dtype == DType.float64:
        _sum_backward_axis_element_impl[DType.float64](result, grad_output, result_idx, grad_idx)
    elif result._dtype == DType.int8:
        _sum_backward_axis_element_impl[DType.int8](result, grad_output, result_idx, grad_idx)
    elif result._dtype == DType.int16:
        _sum_backward_axis_element_impl[DType.int16](result, grad_output, result_idx, grad_idx)
    elif result._dtype == DType.int32:
        _sum_backward_axis_element_impl[DType.int32](result, grad_output, result_idx, grad_idx)
    elif result._dtype == DType.int64:
        _sum_backward_axis_element_impl[DType.int64](result, grad_output, result_idx, grad_idx)
    else:
        raise Error("sum_backward: unsupported dtype")


fn _fill_zero_impl[dtype: DType](result: ExTensor, numel: Int):
    """Fill tensor with zeros using typed pointer.

    Parameters:
        dtype: Compile-time dtype parameter.

    Args:
        result: Result tensor to fill.
        numel: Number of elements.
    """
    var out_ptr = result._data.bitcast[Scalar[dtype]]()
    for i in range(numel):
        out_ptr[i] = Scalar[dtype](0)


fn _dispatch_fill_zero(result: ExTensor, numel: Int) raises:
    """Runtime dispatch for fill zero."""
    if result._dtype == DType.float16:
        _fill_zero_impl[DType.float16](result, numel)
    elif result._dtype == DType.float32:
        _fill_zero_impl[DType.float32](result, numel)
    elif result._dtype == DType.float64:
        _fill_zero_impl[DType.float64](result, numel)
    elif result._dtype == DType.int8:
        _fill_zero_impl[DType.int8](result, numel)
    elif result._dtype == DType.int16:
        _fill_zero_impl[DType.int16](result, numel)
    elif result._dtype == DType.int32:
        _fill_zero_impl[DType.int32](result, numel)
    elif result._dtype == DType.int64:
        _fill_zero_impl[DType.int64](result, numel)
    else:
        raise Error("fill_zero: unsupported dtype")


fn _max_backward_all_impl[dtype: DType](
    result: ExTensor, x: ExTensor, grad_output: ExTensor, numel: Int
):
    """Compute max_reduce_backward for all elements.

    Parameters:
        dtype: Compile-time dtype parameter.

    Args:
        result: Pre-allocated result tensor (zeros).
        x: Input tensor from forward pass.
        grad_output: Gradient tensor (scalar).
        numel: Number of elements.
    """
    var x_ptr = x._data.bitcast[Scalar[dtype]]()
    var grad_ptr = grad_output._data.bitcast[Scalar[dtype]]()
    var out_ptr = result._data.bitcast[Scalar[dtype]]()

    # Find max value
    var max_val = x_ptr[0]
    for i in range(1, numel):
        if x_ptr[i] > max_val:
            max_val = x_ptr[i]

    # Count max elements
    var count: Int = 0
    for i in range(numel):
        if x_ptr[i] == max_val:
            count += 1

    # Split gradient among max elements
    @parameter
    if dtype == DType.float16 or dtype == DType.float32:
        var grad_val = Float64(grad_ptr[0])
        var grad_per_max = grad_val / Float64(count)
        for i in range(numel):
            if x_ptr[i] == max_val:
                out_ptr[i] = Scalar[dtype](grad_per_max)
    else:
        var grad_val = grad_ptr[0]
        var grad_per_max = grad_val / Scalar[dtype](count)
        for i in range(numel):
            if x_ptr[i] == max_val:
                out_ptr[i] = grad_per_max


fn _dispatch_max_backward_all(
    result: ExTensor, x: ExTensor, grad_output: ExTensor, numel: Int
) raises:
    """Runtime dispatch for max_reduce_backward all elements."""
    if result._dtype == DType.float16:
        _max_backward_all_impl[DType.float16](result, x, grad_output, numel)
    elif result._dtype == DType.float32:
        _max_backward_all_impl[DType.float32](result, x, grad_output, numel)
    elif result._dtype == DType.float64:
        _max_backward_all_impl[DType.float64](result, x, grad_output, numel)
    elif result._dtype == DType.int8:
        _max_backward_all_impl[DType.int8](result, x, grad_output, numel)
    elif result._dtype == DType.int16:
        _max_backward_all_impl[DType.int16](result, x, grad_output, numel)
    elif result._dtype == DType.int32:
        _max_backward_all_impl[DType.int32](result, x, grad_output, numel)
    elif result._dtype == DType.int64:
        _max_backward_all_impl[DType.int64](result, x, grad_output, numel)
    else:
        raise Error("max_backward: unsupported dtype")


fn _min_backward_all_impl[dtype: DType](
    result: ExTensor, x: ExTensor, grad_output: ExTensor, numel: Int
):
    """Compute min_reduce_backward for all elements.

    Parameters:
        dtype: Compile-time dtype parameter.

    Args:
        result: Pre-allocated result tensor (zeros).
        x: Input tensor from forward pass.
        grad_output: Gradient tensor (scalar).
        numel: Number of elements.
    """
    var x_ptr = x._data.bitcast[Scalar[dtype]]()
    var grad_ptr = grad_output._data.bitcast[Scalar[dtype]]()
    var out_ptr = result._data.bitcast[Scalar[dtype]]()

    # Find min value
    var min_val = x_ptr[0]
    for i in range(1, numel):
        if x_ptr[i] < min_val:
            min_val = x_ptr[i]

    # Count min elements
    var count: Int = 0
    for i in range(numel):
        if x_ptr[i] == min_val:
            count += 1

    # Split gradient among min elements
    @parameter
    if dtype == DType.float16 or dtype == DType.float32:
        var grad_val = Float64(grad_ptr[0])
        var grad_per_min = grad_val / Float64(count)
        for i in range(numel):
            if x_ptr[i] == min_val:
                out_ptr[i] = Scalar[dtype](grad_per_min)
    else:
        var grad_val = grad_ptr[0]
        var grad_per_min = grad_val / Scalar[dtype](count)
        for i in range(numel):
            if x_ptr[i] == min_val:
                out_ptr[i] = grad_per_min


fn _dispatch_min_backward_all(
    result: ExTensor, x: ExTensor, grad_output: ExTensor, numel: Int
) raises:
    """Runtime dispatch for min_reduce_backward all elements."""
    if result._dtype == DType.float16:
        _min_backward_all_impl[DType.float16](result, x, grad_output, numel)
    elif result._dtype == DType.float32:
        _min_backward_all_impl[DType.float32](result, x, grad_output, numel)
    elif result._dtype == DType.float64:
        _min_backward_all_impl[DType.float64](result, x, grad_output, numel)
    elif result._dtype == DType.int8:
        _min_backward_all_impl[DType.int8](result, x, grad_output, numel)
    elif result._dtype == DType.int16:
        _min_backward_all_impl[DType.int16](result, x, grad_output, numel)
    elif result._dtype == DType.int32:
        _min_backward_all_impl[DType.int32](result, x, grad_output, numel)
    elif result._dtype == DType.int64:
        _min_backward_all_impl[DType.int64](result, x, grad_output, numel)
    else:
        raise Error("min_backward: unsupported dtype")


fn _get_value_impl[dtype: DType](tensor: ExTensor, idx: Int) -> Float64:
    """Get tensor value at index as Float64 for comparison.

    Parameters:
        dtype: Compile-time dtype parameter.

    Args:
        tensor: Input tensor.
        idx: Index to read.

    Returns:
        Value as Float64.
    """
    var ptr = tensor._data.bitcast[Scalar[dtype]]()
    return Float64(ptr[idx])


fn _dispatch_get_value(tensor: ExTensor, idx: Int) raises -> Float64:
    """Runtime dispatch for getting tensor value."""
    if tensor._dtype == DType.float16:
        return _get_value_impl[DType.float16](tensor, idx)
    elif tensor._dtype == DType.float32:
        return _get_value_impl[DType.float32](tensor, idx)
    elif tensor._dtype == DType.float64:
        return _get_value_impl[DType.float64](tensor, idx)
    elif tensor._dtype == DType.int8:
        return _get_value_impl[DType.int8](tensor, idx)
    elif tensor._dtype == DType.int16:
        return _get_value_impl[DType.int16](tensor, idx)
    elif tensor._dtype == DType.int32:
        return _get_value_impl[DType.int32](tensor, idx)
    elif tensor._dtype == DType.int64:
        return _get_value_impl[DType.int64](tensor, idx)
    else:
        raise Error("get_value: unsupported dtype")


fn _set_value_impl[dtype: DType](tensor: ExTensor, idx: Int, val: Float64):
    """Set tensor value at index from Float64.

    Parameters:
        dtype: Compile-time dtype parameter.

    Args:
        tensor: Output tensor.
        idx: Index to write.
        val: Value to write.
    """
    var ptr = tensor._data.bitcast[Scalar[dtype]]()
    ptr[idx] = Scalar[dtype](val)


fn _dispatch_set_value(tensor: ExTensor, idx: Int, val: Float64) raises:
    """Runtime dispatch for setting tensor value."""
    if tensor._dtype == DType.float16:
        _set_value_impl[DType.float16](tensor, idx, val)
    elif tensor._dtype == DType.float32:
        _set_value_impl[DType.float32](tensor, idx, val)
    elif tensor._dtype == DType.float64:
        _set_value_impl[DType.float64](tensor, idx, val)
    elif tensor._dtype == DType.int8:
        _set_value_impl[DType.int8](tensor, idx, val)
    elif tensor._dtype == DType.int16:
        _set_value_impl[DType.int16](tensor, idx, val)
    elif tensor._dtype == DType.int32:
        _set_value_impl[DType.int32](tensor, idx, val)
    elif tensor._dtype == DType.int64:
        _set_value_impl[DType.int64](tensor, idx, val)
    else:
        raise Error("set_value: unsupported dtype")


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
    # Create result tensor with input shape
    var input_shape = x.shape()
    var result = ExTensor(input_shape, grad_output.dtype())

    if axis == -1:
        # Sum over all elements - broadcast scalar gradient using dtype-specialized impl
        _dispatch_sum_backward_all(result, grad_output, result.numel())
    else:
        # Sum along specific axis - broadcast gradient along that axis
        # For each position in grad_output, broadcast it to all positions along axis
        for result_idx in range(result.numel()):
            # Convert result index to coordinates
            var coords = linear_to_coords(result_idx, input_shape)

            # Map to grad_output coordinates (remove axis dimension)
            var grad_dim = grad_output.dim()
            var grad_coords = List[Int](capacity=grad_dim)
            for _ in range(grad_dim):
                grad_coords.append(0)
            var coord_idx = 0
            for i in range(len(input_shape)):
                if i != axis:
                    grad_coords[coord_idx] = coords[i]
                    coord_idx += 1

            # Convert grad_coords to linear index in grad_output
            var grad_strides = compute_strides(grad_output.shape())
            var grad_idx = coords_to_linear(grad_coords, grad_strides)

            # Set result value using dtype-specialized impl
            _dispatch_sum_backward_axis_element(result, grad_output, result_idx, grad_idx)

    return result^


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
    # First get the sum backward (broadcasts gradient)
    var grad_sum = sum_backward(grad_output, x, axis)

    # Compute number of elements that were averaged
    var input_shape = x.shape()
    var n: Int
    if axis == -1:
        # Mean over all elements
        n = 1
        for i in range(len(input_shape)):
            n *= input_shape[i]
    else:
        # Mean along specific axis
        n = input_shape[axis]

    # Scale by 1/N using dtype-specialized implementation
    _dispatch_mean_divide(grad_sum, n)

    return grad_sum^


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
    var result = ExTensor(x.shape(), x.dtype())
    # Initialize to zero using dtype-specialized implementation
    _dispatch_fill_zero(result, result.numel())

    if axis == -1:
        # Max over all elements using dtype-specialized implementation
        _dispatch_max_backward_all(result, x, grad_output, x.numel())

    else:
        # Max along specific axis
        var input_shape = x.shape()
        var ndim = len(input_shape)

        # Normalize axis
        var normalized_axis = axis if axis >= 0 else ndim + axis

        # Compute strides
        var strides = compute_strides(input_shape)

        var axis_size = input_shape[normalized_axis]

        # For each position in grad_output
        for result_idx in range(x.numel()):
            # Convert to coordinates
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

            # Find max value along this slice using typed access
            var max_val: Float64 = _dispatch_get_value(x, 0)  # Placeholder
            var count = 0

            # First pass: find max
            for k in range(axis_size):
                var test_coords: List[Int] = List[Int](coords)
                test_coords[normalized_axis] = k
                var test_idx = coords_to_linear(test_coords, strides)
                var val = _dispatch_get_value(x, test_idx)
                if k == 0 or val > max_val:
                    max_val = val

            # Second pass: count max elements
            for k in range(axis_size):
                var test_coords: List[Int] = List[Int](coords)
                test_coords[normalized_axis] = k
                var test_idx = coords_to_linear(test_coords, strides)
                var val = _dispatch_get_value(x, test_idx)
                if val == max_val:
                    count += 1

            # Third pass: set gradients for max elements
            var current_val = _dispatch_get_value(x, result_idx)
            if current_val == max_val:
                var grad_val = _dispatch_get_value(grad_output, grad_idx)
                _dispatch_set_value(result, result_idx, grad_val / Float64(count))

    return result^


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
    var result = ExTensor(x.shape(), x.dtype())
    # Initialize to zero using dtype-specialized implementation
    _dispatch_fill_zero(result, result.numel())

    if axis == -1:
        # Min over all elements using dtype-specialized implementation
        _dispatch_min_backward_all(result, x, grad_output, x.numel())

    else:
        # Min along specific axis (similar logic to max_reduce_backward)
        var input_shape = x.shape()
        var ndim = len(input_shape)

        # Normalize axis
        var normalized_axis = axis if axis >= 0 else ndim + axis

        # Compute strides
        var strides = compute_strides(input_shape)

        var axis_size = input_shape[normalized_axis]

        # For each position in result
        for result_idx in range(x.numel()):
            # Convert to coordinates
            var coords = linear_to_coords(result_idx, input_shape)

            # Map to grad_output coordinates
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

            # Find min value along this slice using typed access
            var min_val: Float64 = _dispatch_get_value(x, 0)  # Placeholder
            var count = 0

            # First pass: find min
            for k in range(axis_size):
                var test_coords = List[Int](coords)
                test_coords[normalized_axis] = k
                var test_idx = coords_to_linear(test_coords, strides)
                var val = _dispatch_get_value(x, test_idx)
                if k == 0 or val < min_val:
                    min_val = val

            # Second pass: count min elements
            for k in range(axis_size):
                var test_coords = List[Int](coords)
                test_coords[normalized_axis] = k
                var test_idx = coords_to_linear(test_coords, strides)
                var val = _dispatch_get_value(x, test_idx)
                if val == min_val:
                    count += 1

            # Third pass: set gradients for min elements
            var current_val = _dispatch_get_value(x, result_idx)
            if current_val == min_val:
                var grad_val = _dispatch_get_value(grad_output, grad_idx)
                _dispatch_set_value(result, result_idx, grad_val / Float64(count))

    return result^
