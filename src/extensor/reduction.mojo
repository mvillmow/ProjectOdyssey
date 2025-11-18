"""Reduction operations for ExTensor.

Implements operations that reduce tensors along specified axes.
"""

from collections.vector import DynamicVector
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
        if axis < 0 or axis >= tensor.dim():
            raise Error("Axis " + str(axis) + " is out of bounds for tensor with " + str(tensor.dim()) + " dimensions")

        # Build result shape
        var result_shape = DynamicVector[Int]()
        for i in range(tensor.dim()):
            if i != axis:
                result_shape.push_back(tensor.shape()[i])
            elif keepdims:
                result_shape.push_back(1)

        var result = ExTensor(result_shape, tensor.dtype())
        result._fill_zero()

        # Compute strides for indexing
        let input_shape = tensor.shape()
        var strides = DynamicVector[Int](tensor.dim())
        var stride = 1
        for i in range(tensor.dim() - 1, -1, -1):
            strides[i] = stride
            stride *= input_shape[i]

        # Iterate over all elements and accumulate
        let axis_size = input_shape[axis]
        let total_elements = tensor.numel()

        # For each position in the result
        for result_idx in range(result.numel()):
            var sum_val: Float64 = 0.0

            # Convert result index to coordinates
            var result_coords = DynamicVector[Int](result.dim())
            var temp_idx = result_idx
            for i in range(result.dim() - 1, -1, -1):
                result_coords[i] = temp_idx % result.shape()[i]
                temp_idx //= result.shape()[i]

            # Map result coordinates to input coordinates (accounting for reduced axis)
            var input_coords = DynamicVector[Int](tensor.dim())
            var result_coord_idx = 0
            for i in range(tensor.dim()):
                if i != axis:
                    input_coords[i] = result_coords[result_coord_idx]
                    result_coord_idx += 1
                else:
                    input_coords[i] = 0  # Will iterate over this

            # Sum along the reduction axis
            for k in range(axis_size):
                input_coords[axis] = k

                # Convert coordinates to linear index
                var linear_idx = 0
                for i in range(tensor.dim()):
                    linear_idx += input_coords[i] * strides[i]

                sum_val += tensor._get_float64(linear_idx)

            result._set_float64(result_idx, sum_val)

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
        if axis < 0 or axis >= tensor.dim():
            raise Error("Axis " + str(axis) + " is out of bounds for tensor with " + str(tensor.dim()) + " dimensions")

        # Compute sum along axis
        var sum_result = sum(tensor, axis, keepdims)

        # Divide by count along the reduction axis
        let count = Float64(tensor.shape()[axis])

        # Divide each element by count
        for i in range(sum_result.numel()):
            let mean_val = sum_result._get_float64(i) / count
            sum_result._set_float64(i, mean_val)

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
        if axis < 0 or axis >= tensor.dim():
            raise Error("Axis " + str(axis) + " is out of bounds for tensor with " + str(tensor.dim()) + " dimensions")

        # Build result shape
        var result_shape = DynamicVector[Int]()
        for i in range(tensor.dim()):
            if i != axis:
                result_shape.push_back(tensor.shape()[i])
            elif keepdims:
                result_shape.push_back(1)

        var result = ExTensor(result_shape, tensor.dtype())

        # Compute strides for indexing
        let input_shape = tensor.shape()
        var strides = DynamicVector[Int](tensor.dim())
        var stride = 1
        for i in range(tensor.dim() - 1, -1, -1):
            strides[i] = stride
            stride *= input_shape[i]

        # Iterate over all elements and find maximum
        let axis_size = input_shape[axis]

        # For each position in the result
        for result_idx in range(result.numel()):
            # Convert result index to coordinates
            var result_coords = DynamicVector[Int](result.dim())
            var temp_idx = result_idx
            for i in range(result.dim() - 1, -1, -1):
                result_coords[i] = temp_idx % result.shape()[i]
                temp_idx //= result.shape()[i]

            # Map result coordinates to input coordinates (accounting for reduced axis)
            var input_coords = DynamicVector[Int](tensor.dim())
            var result_coord_idx = 0
            for i in range(tensor.dim()):
                if i != axis:
                    input_coords[i] = result_coords[result_coord_idx]
                    result_coord_idx += 1
                else:
                    input_coords[i] = 0  # Will iterate over this

            # Find max along the reduction axis
            # Initialize with first value
            input_coords[axis] = 0
            var linear_idx = 0
            for i in range(tensor.dim()):
                linear_idx += input_coords[i] * strides[i]
            var max_val = tensor._get_float64(linear_idx)

            # Compare with remaining values
            for k in range(1, axis_size):
                input_coords[axis] = k

                # Convert coordinates to linear index
                linear_idx = 0
                for i in range(tensor.dim()):
                    linear_idx += input_coords[i] * strides[i]

                let val = tensor._get_float64(linear_idx)
                if val > max_val:
                    max_val = val

            result._set_float64(result_idx, max_val)

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
        if axis < 0 or axis >= tensor.dim():
            raise Error("Axis " + str(axis) + " is out of bounds for tensor with " + str(tensor.dim()) + " dimensions")

        # Build result shape
        var result_shape = DynamicVector[Int]()
        for i in range(tensor.dim()):
            if i != axis:
                result_shape.push_back(tensor.shape()[i])
            elif keepdims:
                result_shape.push_back(1)

        var result = ExTensor(result_shape, tensor.dtype())

        # Compute strides for indexing
        let input_shape = tensor.shape()
        var strides = DynamicVector[Int](tensor.dim())
        var stride = 1
        for i in range(tensor.dim() - 1, -1, -1):
            strides[i] = stride
            stride *= input_shape[i]

        # Iterate over all elements and find minimum
        let axis_size = input_shape[axis]

        # For each position in the result
        for result_idx in range(result.numel()):
            # Convert result index to coordinates
            var result_coords = DynamicVector[Int](result.dim())
            var temp_idx = result_idx
            for i in range(result.dim() - 1, -1, -1):
                result_coords[i] = temp_idx % result.shape()[i]
                temp_idx //= result.shape()[i]

            # Map result coordinates to input coordinates (accounting for reduced axis)
            var input_coords = DynamicVector[Int](tensor.dim())
            var result_coord_idx = 0
            for i in range(tensor.dim()):
                if i != axis:
                    input_coords[i] = result_coords[result_coord_idx]
                    result_coord_idx += 1
                else:
                    input_coords[i] = 0  # Will iterate over this

            # Find min along the reduction axis
            # Initialize with first value
            input_coords[axis] = 0
            var linear_idx = 0
            for i in range(tensor.dim()):
                linear_idx += input_coords[i] * strides[i]
            var min_val = tensor._get_float64(linear_idx)

            # Compare with remaining values
            for k in range(1, axis_size):
                input_coords[axis] = k

                # Convert coordinates to linear index
                linear_idx = 0
                for i in range(tensor.dim()):
                    linear_idx += input_coords[i] * strides[i]

                let val = tensor._get_float64(linear_idx)
                if val < min_val:
                    min_val = val

            result._set_float64(result_idx, min_val)

        return result^


# ============================================================================
# Backward Pass (Gradient Computation)
# ============================================================================


fn sum_backward(grad_output: ExTensor, input_shape: DynamicVector[Int], axis: Int = -1) raises -> ExTensor:
    """Compute gradient for sum reduction.

    For Y = sum(X, axis), given ∂L/∂Y, computes:
        ∂L/∂X = broadcast(∂L/∂Y, input_shape)

    The gradient broadcasts the reduced gradient back to the original input shape.
    Each element of the input contributes equally to the sum, so gradient is 1.

    Args:
        grad_output: Gradient from upstream (∂L/∂Y) - reduced tensor
        input_shape: Original shape of input before reduction
        axis: Axis along which sum was computed (-1 for all axes)

    Returns:
        Gradient w.r.t. input (∂L/∂X) - broadcast back to input_shape

    Examples:
        # Sum all elements
        var x = ones(DynamicVector[Int](3, 4), DType.float32)
        var y = sum(x, axis=-1)  # Scalar
        var grad_y = ones(DynamicVector[Int](), DType.float32)  # Scalar gradient
        var grad_x = sum_backward(grad_y, x.shape(), axis=-1)  # Shape (3, 4)

        # Sum along specific axis
        var x2 = ones(DynamicVector[Int](3, 4), DType.float32)
        var y2 = sum(x2, axis=1)  # Shape (3,)
        var grad_y2 = ones(DynamicVector[Int](3), DType.float32)
        var grad_x2 = sum_backward(grad_y2, x2.shape(), axis=1)  # Shape (3, 4)
    """
    # Create result tensor with input shape
    var result = ExTensor(input_shape, grad_output.dtype())

    if axis == -1:
        # Sum over all elements - broadcast scalar gradient to all elements
        let grad_val = grad_output._get_float64(0)
        for i in range(result.numel()):
            result._set_float64(i, grad_val)
    else:
        # Sum along specific axis - broadcast gradient along that axis
        # The gradient value is replicated axis_size times

        # Compute strides for input tensor
        var strides = DynamicVector[Int](len(input_shape))
        var stride = 1
        for i in range(len(input_shape) - 1, -1, -1):
            strides[i] = stride
            stride *= input_shape[i]

        let axis_size = input_shape[axis]

        # For each position in grad_output, broadcast it to all positions along axis
        for result_idx in range(result.numel()):
            # Convert result index to coordinates
            var coords = DynamicVector[Int](len(input_shape))
            var temp_idx = result_idx
            for i in range(len(input_shape) - 1, -1, -1):
                coords[i] = temp_idx % input_shape[i]
                temp_idx //= input_shape[i]

            # Map to grad_output coordinates (remove axis dimension)
            var grad_coords = DynamicVector[Int](grad_output.dim())
            var coord_idx = 0
            for i in range(len(input_shape)):
                if i != axis:
                    grad_coords[coord_idx] = coords[i]
                    coord_idx += 1

            # Convert grad_coords to linear index in grad_output
            var grad_idx = 0
            var grad_stride = 1
            for i in range(grad_output.dim() - 1, -1, -1):
                grad_idx += grad_coords[i] * grad_stride
                grad_stride *= grad_output.shape()[i]

            # Set result value
            let grad_val = grad_output._get_float64(grad_idx)
            result._set_float64(result_idx, grad_val)

    return result


fn mean_backward(grad_output: ExTensor, input_shape: DynamicVector[Int], axis: Int = -1) raises -> ExTensor:
    """Compute gradient for mean reduction.

    For Y = mean(X, axis), given ∂L/∂Y, computes:
        ∂L/∂X = broadcast(∂L/∂Y, input_shape) / N

    where N is the number of elements that were averaged.

    Similar to sum_backward, but scaled by 1/N since each input element
    contributes 1/N to the mean.

    Args:
        grad_output: Gradient from upstream (∂L/∂Y) - reduced tensor
        input_shape: Original shape of input before reduction
        axis: Axis along which mean was computed (-1 for all axes)

    Returns:
        Gradient w.r.t. input (∂L/∂X) - broadcast and scaled

    Examples:
        var x = ones(DynamicVector[Int](3, 4), DType.float32)
        var y = mean(x, axis=-1)  # Scalar mean
        var grad_y = ones(DynamicVector[Int](), DType.float32)
        var grad_x = mean_backward(grad_y, x.shape(), axis=-1)
        # Each element gets gradient / 12
    """
    # First get the sum backward (broadcasts gradient)
    var grad_sum = sum_backward(grad_output, input_shape, axis)

    # Compute number of elements that were averaged
    var n: Int = 0
    if axis == -1:
        # Mean over all elements
        n = 1
        for i in range(len(input_shape)):
            n *= input_shape[i]
    else:
        # Mean along specific axis
        n = input_shape[axis]

    # Scale by 1/N
    let scale = 1.0 / Float64(n)
    for i in range(grad_sum.numel()):
        let val = grad_sum._get_float64(i)
        grad_sum._set_float64(i, val * scale)

    return grad_sum
