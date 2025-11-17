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
    # TODO: Implement reduction logic
    if axis == -1:
        # Sum all elements
        var result_shape = DynamicVector[Int]()
        if keepdims:
            for i in range(tensor.dim()):
                result_shape.push_back(1)
        var result = ExTensor(result_shape, tensor.dtype())
        result._fill_zero()
        return result^
    else:
        # Sum along specific axis
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
    # TODO: Implement
    # Mean = sum / count
    let sum_result = sum(tensor, axis, keepdims)
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
    # TODO: Implement
    if axis == -1:
        var result_shape = DynamicVector[Int]()
        if keepdims:
            for i in range(tensor.dim()):
                result_shape.push_back(1)
        var result = ExTensor(result_shape, tensor.dtype())
        result._fill_zero()
        return result^
    else:
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
    # TODO: Implement
    if axis == -1:
        var result_shape = DynamicVector[Int]()
        if keepdims:
            for i in range(tensor.dim()):
                result_shape.push_back(1)
        var result = ExTensor(result_shape, tensor.dtype())
        result._fill_zero()
        return result^
    else:
        var result_shape = DynamicVector[Int]()
        for i in range(tensor.dim()):
            if i != axis:
                result_shape.push_back(tensor.shape()[i])
            elif keepdims:
                result_shape.push_back(1)

        var result = ExTensor(result_shape, tensor.dtype())
        result._fill_zero()
        return result^
