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
)


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
        ```mojo
        var t = ones([3, 4], DType.float32)
        var s = sum(t, axis=-1)  # Sum all elements -> scalar 12.0
        var row_sums = sum(t, axis=1)  # Sum along rows -> shape (3,)
        ```
    """
    if axis == -1:
        # Sum all elements
        var result_shape = List[Int]()
        if keepdims:
            for _ in range(tensor.dim()):
                result_shape.append(1)
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
        if axis < 0 or axis >= tensor.dim():
            raise Error(
                "Axis "
                + String(axis)
                + " is out of bounds for tensor with "
                + String(tensor.dim())
                + " dimensions"
            )

        # Build result shape
        var result_shape = List[Int]()
        for i in range(tensor.dim()):
            if i != axis:
                result_shape.append(tensor.shape()[i])
            elif keepdims:
                result_shape.append(1)

        var result = ExTensor(result_shape, tensor.dtype())
        result._fill_zero()

        # Compute strides for indexing
        var input_shape = tensor.shape()
        var strides = compute_strides(input_shape)

        # Iterate over all elements and accumulate
        var axis_size = input_shape[axis]

        # For each position in the result
        for result_idx in range(result.numel()):
            var sum_val: Float64 = 0.0

            # Convert result index to coordinates
            var result_coords = create_result_coords(result_idx, result.shape())

            # Map result coordinates to input coordinates (accounting for reduced axis)
            var input_coords = map_result_to_input_coords(
                result_coords, axis, tensor.dim()
            )

            # Sum along the reduction axis
            for k in range(axis_size):
                input_coords[axis] = k

                # Convert coordinates to linear index
                var linear_idx = coords_to_linear(input_coords, strides)

                sum_val += tensor._get_float64(linear_idx)

            result._set_float64(result_idx, sum_val)

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
        ```mojo
        var t = ones([3, 4], DType.float32)
        var m = mean(t)  # Mean of all elements -> scalar 1.0
        ```
    """
    if axis == -1:
        # Mean of all elements
        var sum_result = sum(tensor, axis, keepdims)
        var count = Float64(tensor.numel())
        var mean_val = sum_result._get_float64(0) / count
        sum_result._set_float64(0, mean_val)
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
        var count = Float64(tensor.shape()[axis])

        # Divide each element by count
        for i in range(sum_result.numel()):
            var mean_val = sum_result._get_float64(i) / count
            sum_result._set_float64(i, mean_val)

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
        ```mojo
        var t = arange(0.0, 12.0, 1.0, DType.float32)
        var m = max_reduce(t)  # Maximum element -> scalar 11.0
        ```
    """
    if axis == -1:
        # Max of all elements
        var result_shape = List[Int]()
        if keepdims:
            for _ in range(tensor.dim()):
                result_shape.append(1)
        var result = ExTensor(result_shape, tensor.dtype())

        # Find maximum value
        var max_val = tensor._get_float64(0)
        for i in range(1, tensor.numel()):
            var val = tensor._get_float64(i)
            if val > max_val:
                max_val = val

        result._set_float64(0, max_val)
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

        # Build result shape
        var result_shape = List[Int]()
        for i in range(tensor.dim()):
            if i != axis:
                result_shape.append(tensor.shape()[i])
            elif keepdims:
                result_shape.append(1)

        var result = ExTensor(result_shape, tensor.dtype())

        # Compute strides for indexing
        var input_shape = tensor.shape()
        var strides = compute_strides(input_shape)

        # Iterate over all elements and find maximum
        var axis_size = input_shape[axis]

        # For each position in the result
        for result_idx in range(result.numel()):
            # Convert result index to coordinates
            var result_coords = create_result_coords(result_idx, result.shape())

            # Map result coordinates to input coordinates (accounting for reduced axis)
            var input_coords = map_result_to_input_coords(
                result_coords, axis, tensor.dim()
            )

            # Find max along the reduction axis
            # Initialize with first value
            input_coords[axis] = 0
            var linear_idx = coords_to_linear(input_coords, strides)
            var max_val = tensor._get_float64(linear_idx)

            # Compare with remaining values
            for k in range(1, axis_size):
                input_coords[axis] = k

                # Convert coordinates to linear index
                linear_idx = coords_to_linear(input_coords, strides)

                var val = tensor._get_float64(linear_idx)
                if val > max_val:
                    max_val = val

            result._set_float64(result_idx, max_val)

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
        ```mojo
        var t = arange(0.0, 12.0, 1.0, DType.float32)
        var m = min_reduce(t)  # Minimum element -> scalar 0.0
        ```
    """
    if axis == -1:
        # Min of all elements
        var result_shape = List[Int]()
        if keepdims:
            for i in range(tensor.dim()):
                result_shape.append(1)
        var result = ExTensor(result_shape, tensor.dtype())

        # Find minimum value
        var min_val = tensor._get_float64(0)
        for i in range(1, tensor.numel()):
            var val = tensor._get_float64(i)
            if val < min_val:
                min_val = val

        result._set_float64(0, min_val)
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

        # Build result shape
        var result_shape = List[Int]()
        for i in range(tensor.dim()):
            if i != axis:
                result_shape.append(tensor.shape()[i])
            elif keepdims:
                result_shape.append(1)

        var result = ExTensor(result_shape, tensor.dtype())

        # Compute strides for indexing
        var input_shape = tensor.shape()
        var strides = compute_strides(input_shape)

        # Iterate over all elements and find minimum
        var axis_size = input_shape[axis]

        # For each position in the result
        for result_idx in range(result.numel()):
            # Convert result index to coordinates
            var result_coords = create_result_coords(result_idx, result.shape())

            # Map result coordinates to input coordinates (accounting for reduced axis)
            var input_coords = map_result_to_input_coords(
                result_coords, axis, tensor.dim()
            )

            # Find min along the reduction axis
            # Initialize with first value
            input_coords[axis] = 0
            var linear_idx = coords_to_linear(input_coords, strides)
            var min_val = tensor._get_float64(linear_idx)

            # Compare with remaining values
            for k in range(1, axis_size):
                input_coords[axis] = k

                # Convert coordinates to linear index
                linear_idx = coords_to_linear(input_coords, strides)

                var val = tensor._get_float64(linear_idx)
                if val < min_val:
                    min_val = val

            result._set_float64(result_idx, min_val)

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

    The gradient broadcasts the reduced gradient back to the original input shape.
    Each element of the input contributes equally to the sum, so gradient is 1.

    Args:
        grad_output: Gradient from upstream (∂L/∂Y) - reduced tensor.
        x: Original input tensor before reduction.
        axis: Axis along which sum was computed (-1 for all axes).

    Returns:
        Gradient w.r.t. input (∂L/∂X) - broadcast back to input_shape.

    Examples:
        ```mojo
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
        # Sum over all elements - broadcast scalar gradient to all elements
        var grad_val = grad_output._get_float64(0)
        for i in range(result.numel()):
            result._set_float64(i, grad_val)
    else:
        # Sum along specific axis - broadcast gradient along that axis
        # The gradient value is replicated axis_size times.

        # Compute strides for input tensor
        var ndim = len(input_shape)
        var strides = compute_strides(input_shape)

        var axis_size = input_shape[axis]

        # For each position in grad_output, broadcast it to all positions along axis
        for result_idx in range(result.numel()):
            # Convert result index to coordinates
            var coords = linear_to_coords(result_idx, input_shape)

            # Map to grad_output coordinates (remove axis dimension)
            var grad_dim = grad_output.dim()
            var grad_coords = List[Int]()
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

            # Set result value
            var grad_val = grad_output._get_float64(grad_idx)
            result._set_float64(result_idx, grad_val)

    return result^


fn mean_backward(
    grad_output: ExTensor, x: ExTensor, axis: Int = -1
) raises -> ExTensor:
    """Compute gradient for mean reduction.

    For Y = mean(X, axis), given ∂L/∂Y, computes:
        ∂L/∂X = broadcast(∂L/∂Y, input_shape) / N

    Where N is the number of elements that were averaged.

    Similar to sum_backward, but scaled by 1/N since each input element
    contributes 1/N to the mean.

    Args:
        grad_output: Gradient from upstream (∂L/∂Y) - reduced tensor.
        x: Original input tensor before reduction.
        axis: Axis along which mean was computed (-1 for all axes).

    Returns:
        Gradient w.r.t. input (∂L/∂X) - broadcast and scaled.

    Examples:
        ```mojo
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

    # Scale by 1/N
    var scale = 1.0 / Float64(n)
    for i in range(grad_sum.numel()):
        var val = grad_sum._get_float64(i)
        grad_sum._set_float64(i, val * scale)

    return grad_sum^


fn max_reduce_backward(
    grad_output: ExTensor, x: ExTensor, axis: Int = -1
) raises -> ExTensor:
    """Compute gradient for max reduction.

    For Y = max_reduce(X, axis), given ∂L/∂Y, computes:
        ∂L/∂X - Gradient flows only to maximum element(s)

    If multiple elements are maximum, gradient is split equally among them.
    This is the standard behavior for max pooling backward pass.

    Args:
        grad_output: Gradient from upstream (∂L/∂Y) - reduced tensor.
        x: Input from forward pass (before reduction).
        axis: Axis along which max was computed (-1 for all axes).

    Returns:
        Gradient w.r.t. input (∂L/∂X).

    Examples:
        ```mojo
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
    # Initialize to zero
    for i in range(result.numel()):
        result._set_float64(i, 0.0)

    if axis == -1:
        # Max over all elements - find all elements equal to max
        var max_val: Float64 = x._get_float64(0)
        for i in range(1, x.numel()):
            var val = x._get_float64(i)
            if val > max_val:
                max_val = val

        # Count how many elements are maximum
        var count: Int = 0
        for i in range(x.numel()):
            var val = x._get_float64(i)
            if val == max_val:
                count += 1

        # Split gradient equally among max elements
        var grad_val = grad_output._get_float64(0)
        var grad_per_max = grad_val / Float64(count)

        for i in range(x.numel()):
            var val = x._get_float64(i)
            if val == max_val:
                result._set_float64(i, grad_per_max)

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
            var grad_coords = List[Int]()
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

            # Find max value along this slice
            var max_val: Float64 = x._get_float64(0)  # Placeholder
            var count = 0

            # First pass: find max
            for k in range(axis_size):
                var test_coords: List[Int] = List[Int](coords)
                test_coords[normalized_axis] = k
                var test_idx = coords_to_linear(test_coords, strides)
                var val = x._get_float64(test_idx)
                if k == 0 or val > max_val:
                    max_val = val

            # Second pass: count max elements
            for k in range(axis_size):
                var test_coords: List[Int] = List[Int](coords)
                test_coords[normalized_axis] = k
                var test_idx = coords_to_linear(test_coords, strides)
                var val = x._get_float64(test_idx)
                if val == max_val:
                    count += 1

            # Third pass: set gradients for max elements
            var current_val = x._get_float64(result_idx)
            if current_val == max_val:
                var grad_val = grad_output._get_float64(grad_idx)
                result._set_float64(result_idx, grad_val / Float64(count))

    return result^


fn min_reduce_backward(
    grad_output: ExTensor, x: ExTensor, axis: Int = -1
) raises -> ExTensor:
    """Compute gradient for min reduction.

    For Y = min_reduce(X, axis), given ∂L/∂Y, computes:
        ∂L/∂X - Gradient flows only to minimum element(s)

    If multiple elements are minimum, gradient is split equally among them.
    This is analogous to max pooling but for minimum values.

    Args:
        grad_output: Gradient from upstream (∂L/∂Y) - reduced tensor.
        x: Input from forward pass (before reduction).
        axis: Axis along which min was computed (-1 for all axes).

    Returns:
        Gradient w.r.t. input (∂L/∂X).

    Examples:
        ```mojo
        var x = tensor([3.0, 1.0, 2.0, 1.0])  # Two min values at indices 1, 3
        var y = min_reduce(x, axis=-1)  # Scalar: 1.0
        var grad_y = ones([])  # Gradient: 1.0
        var grad_x = min_reduce_backward(grad_y, x, axis=-1)
        # grad_x = [0.0, 0.5, 0.0, 0.5]  # Split equally between the two 1.0s
        ```
    """
    var result = ExTensor(x.shape(), x.dtype())
    # Initialize to zero
    for i in range(result.numel()):
        result._set_float64(i, 0.0)

    if axis == -1:
        # Min over all elements - find all elements equal to min
        var min_val: Float64 = x._get_float64(0)
        for i in range(1, x.numel()):
            var val = x._get_float64(i)
            if val < min_val:
                min_val = val

        # Count how many elements are minimum
        var count: Int = 0
        for i in range(x.numel()):
            var val = x._get_float64(i)
            if val == min_val:
                count += 1

        # Split gradient equally among min elements
        var grad_val = grad_output._get_float64(0)
        var grad_per_min = grad_val / Float64(count)

        for i in range(x.numel()):
            var val = x._get_float64(i)
            if val == min_val:
                result._set_float64(i, grad_per_min)

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
            var grad_coords = List[Int]()
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

            # Find min value along this slice
            var min_val: Float64 = x._get_float64(0)  # Placeholder
            var count = 0

            # First pass: find min
            for k in range(axis_size):
                var test_coords = List[Int](coords)
                test_coords[normalized_axis] = k
                var test_idx = coords_to_linear(test_coords, strides)
                var val = x._get_float64(test_idx)
                if k == 0 or val < min_val:
                    min_val = val

            # Second pass: count min elements
            for k in range(axis_size):
                var test_coords = List[Int](coords)
                test_coords[normalized_axis] = k
                var test_idx = coords_to_linear(test_coords, strides)
                var val = x._get_float64(test_idx)
                if val == min_val:
                    count += 1

            # Third pass: set gradients for min elements
            var current_val = x._get_float64(result_idx)
            if current_val == min_val:
                var grad_val = grad_output._get_float64(grad_idx)
                result._set_float64(result_idx, grad_val / Float64(count))

    return result^
