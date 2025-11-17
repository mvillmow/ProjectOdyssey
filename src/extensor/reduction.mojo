"""Reduction operations for ExTensor.

Implements operations that reduce tensors along specified axes.
"""

from .extensor import ExTensor


fn sum(tensor: ExTensor, axis: Int = -1, keepdims: Bool = False) raises -> ExTensor:
    """Sum tensor elements along an axis.

    Args:
        tensor: Input tensor
        axis: Axis to reduce (-1 for all axes)
        keepdims: Whether to keep reduced dimensions as size 1

    Returns:
        A new tensor with sum along specified axis

    Examples:
        var t = ones(DynamicVector[Int](3, 4), DType.float32)
        var s = sum(t, axis=-1)  # Sum all elements -> scalar 12.0
        var row_sums = sum(t, axis=1)  # Sum along rows -> shape (3,)
    """
    if axis == -1:
        # Sum all elements
        var result_shape = DynamicVector[Int]()
        if keepdims:
            for i in range(tensor.dim()):
                result_shape.push_back(1)
        var result = ExTensor(result_shape, tensor.dtype())

        # Compute sum of all elements
        var sum_val: Float64 = 0.0
        for i in range(tensor.numel()):
            sum_val += tensor._get_float64(i)

        # Set result value (scalar or 1-element tensor)
        result._set_float64(0, sum_val)
        return result^
    else:
        # Sum along specific axis
        # TODO: Implement axis-specific reduction
        var result_shape = DynamicVector[Int]()
        for i in range(tensor.dim()):
            if i != axis:
                result_shape.push_back(tensor.shape()[i])
            elif keepdims:
                result_shape.push_back(1)

        var result = ExTensor(result_shape, tensor.dtype())
        result._fill_zero()
        return result^


fn mean(tensor: ExTensor, axis: Int = -1, keepdims: Bool = False) raises -> ExTensor:
    """Compute mean of tensor elements along an axis.

    Args:
        tensor: Input tensor
        axis: Axis to reduce (-1 for all axes)
        keepdims: Whether to keep reduced dimensions as size 1

    Returns:
        A new tensor with mean along specified axis

    Examples:
        var t = ones(DynamicVector[Int](3, 4), DType.float32)
        var m = mean(t)  # Mean of all elements -> scalar 1.0
    """
    if axis == -1:
        # Mean of all elements
        var sum_result = sum(tensor, axis, keepdims)
        let count = Float64(tensor.numel())
        let mean_val = sum_result._get_float64(0) / count
        sum_result._set_float64(0, mean_val)
        return sum_result^
    else:
        # Mean along specific axis
        # TODO: Implement axis-specific reduction
        var sum_result = sum(tensor, axis, keepdims)
        return sum_result^


fn max_reduce(tensor: ExTensor, axis: Int = -1, keepdims: Bool = False) raises -> ExTensor:
    """Find maximum of tensor elements along an axis.

    Args:
        tensor: Input tensor
        axis: Axis to reduce (-1 for all axes)
        keepdims: Whether to keep reduced dimensions as size 1

    Returns:
        A new tensor with maximum along specified axis

    Examples:
        var t = arange(0.0, 12.0, 1.0, DType.float32)
        var m = max_reduce(t)  # Maximum element -> scalar 11.0
    """
    if axis == -1:
        # Max of all elements
        var result_shape = DynamicVector[Int]()
        if keepdims:
            for i in range(tensor.dim()):
                result_shape.push_back(1)
        var result = ExTensor(result_shape, tensor.dtype())

        # Find maximum value
        var max_val = tensor._get_float64(0)
        for i in range(1, tensor.numel()):
            let val = tensor._get_float64(i)
            if val > max_val:
                max_val = val

        result._set_float64(0, max_val)
        return result^
    else:
        # Max along specific axis
        # TODO: Implement axis-specific reduction
        var result_shape = DynamicVector[Int]()
        for i in range(tensor.dim()):
            if i != axis:
                result_shape.push_back(tensor.shape()[i])
            elif keepdims:
                result_shape.push_back(1)

        var result = ExTensor(result_shape, tensor.dtype())
        result._fill_zero()
        return result^


fn min_reduce(tensor: ExTensor, axis: Int = -1, keepdims: Bool = False) raises -> ExTensor:
    """Find minimum of tensor elements along an axis.

    Args:
        tensor: Input tensor
        axis: Axis to reduce (-1 for all axes)
        keepdims: Whether to keep reduced dimensions as size 1

    Returns:
        A new tensor with minimum along specified axis

    Examples:
        var t = arange(0.0, 12.0, 1.0, DType.float32)
        var m = min_reduce(t)  # Minimum element -> scalar 0.0
    """
    if axis == -1:
        # Min of all elements
        var result_shape = DynamicVector[Int]()
        if keepdims:
            for i in range(tensor.dim()):
                result_shape.push_back(1)
        var result = ExTensor(result_shape, tensor.dtype())

        # Find minimum value
        var min_val = tensor._get_float64(0)
        for i in range(1, tensor.numel()):
            let val = tensor._get_float64(i)
            if val < min_val:
                min_val = val

        result._set_float64(0, min_val)
        return result^
    else:
        # Min along specific axis
        # TODO: Implement axis-specific reduction
        var result_shape = DynamicVector[Int]()
        for i in range(tensor.dim()):
            if i != axis:
                result_shape.push_back(tensor.shape()[i])
            elif keepdims:
                result_shape.push_back(1)

        var result = ExTensor(result_shape, tensor.dtype())
        result._fill_zero()
        return result^
