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
