"""Shape manipulation operations for ExTensor.

Implements shape operations like reshape, squeeze, unsqueeze, flatten, concatenate, stack, split.
Following the Python Array API Standard 2023.12.
"""

from collections import List
from .extensor import ExTensor


fn reshape(tensor: ExTensor, new_shape: List[Int]) raises -> ExTensor:
    """Reshape tensor to new shape.

    Args:
        `tensor`: Input tensor.
        `new_shape`: Target shape (must have same total number of elements)

    Returns:
        A new tensor with the specified shape.

    Raises:
        Error if new shape has different number of elements.

    Examples:
        # Reshape 1D to 2D
        var a = arange(0.0, 12.0, 1.0, DType.float32)  # Shape (12,)
        var b = reshape(a, [3, 4])  # Shape (3, 4)

        # With -1 for inferred dimension
        var c = reshape(a, [3, -1])  # Shape (3, 4) - infers 4
    """
    # Calculate total elements in new shape (handling -1 for inference)
    var inferred_dim = -1
    var known_product: Int = 1
    var new_len = len(new_shape)

    for i in range(new_len):
        if new_shape[i] == -1:
            if inferred_dim != -1:
                raise Error("reshape: can only specify one unknown dimension (-1)")
            inferred_dim = i
        elif new_shape[i] <= 0:
            raise Error("reshape: shape dimensions must be positive or -1")
        else:
            known_product *= new_shape[i]

    # If we have -1, infer that dimension
    var final_shape = List[Int]()
    var total_elements = tensor.numel()

    if inferred_dim != -1:
        # Infer the -1 dimension
        if total_elements % known_product != 0:
            raise Error("reshape: cannot infer dimension, incompatible size")
        var inferred_size = total_elements // known_product

        for i in range(new_len):
            if i == inferred_dim:
                final_shape.append(inferred_size)
            else:
                final_shape.append(new_shape[i])
    else:
        # No -1, just copy
        for i in range(new_len):
            final_shape.append(new_shape[i])

    # Verify total elements match
    var new_total: Int = 1
    for i in range(new_len):
        new_total *= final_shape[i]

    if new_total != total_elements:
        raise Error("reshape: new shape must have same number of elements")

    # Create new tensor with new shape (data copy)
    var result = ExTensor(final_shape, tensor.dtype())

    # Copy data element by element (row-major order preserved)
    for i in range(total_elements):
        var val = tensor._get_float64(i)
        result._set_float64(i, val)

    return result^


fn squeeze(tensor: ExTensor, dim: Int = -999) raises -> ExTensor:
    """Remove size-1 dimensions.

    Args:
        `tensor`: Input tensor.
        `dim`: Specific dimension to squeeze (optional, default squeezes all size-1 dims)

    Returns:
        Tensor with size-1 dimensions removed.

    Raises:
        Error if specified dim is not size 1.

    Examples:
        # Squeeze all size-1 dims
        var a = ones([1, 3, 1, 4], DType.float32)  # Shape (1, 3, 1, 4)
        var b = squeeze(a)  # Shape (3, 4)

        # Squeeze specific dim
        var c = squeeze(a, 0)  # Shape (3, 1, 4)
    """
    var old_shape = tensor.shape()
    var ndim = len(old_shape)

    if dim != -999:
        # Squeeze specific dimension
        var actual_dim = dim if dim >= 0 else ndim + dim

        if actual_dim < 0 or actual_dim >= ndim:
            raise Error("squeeze: dimension out of range")

        if old_shape[actual_dim] != 1:
            raise Error("squeeze: cannot squeeze dimension that is not size 1")

        # Create new shape without this dimension
        var new_shape = List[Int]()
        for i in range(ndim):
            if i != actual_dim:
                new_shape.append(old_shape[i])

        return reshape(tensor, new_shape)
    else:
        # Squeeze all size-1 dimensions
        var new_dims = 0
        for i in range(ndim):
            if old_shape[i] != 1:
                new_dims += 1

        if new_dims == ndim:
            # No size-1 dims, return copy
            return reshape(tensor, old_shape)

        # Build new shape
        var new_shape = List[Int]()
        for i in range(ndim):
            if old_shape[i] != 1:
                new_shape.append(old_shape[i])

        return reshape(tensor, new_shape)


fn unsqueeze(tensor: ExTensor, dim: Int) raises -> ExTensor:
    """Add a size-1 dimension at specified position.

    Args:
        `tensor`: Input tensor.
        `dim`: Position to insert new dimension (supports negative indexing)

    Returns:
        Tensor with additional size-1 dimension.

    Examples:
        var a = ones([3, 4], DType.float32)  # Shape (3, 4)
        var b = unsqueeze(a, 0)  # Shape (1, 3, 4)
        var c = unsqueeze(a, -1)  # Shape (3, 4, 1)
    """
    var old_shape = tensor.shape()
    var ndim = len(old_shape)
    var new_ndim = ndim + 1

    # Handle negative indexing (allow dim in range [-ndim-1, ndim])
    var actual_dim = dim if dim >= 0 else new_ndim + dim

    if actual_dim < 0 or actual_dim > ndim:
        raise Error("unsqueeze: dimension out of range")

    # Create new shape with size-1 dimension inserted
    var new_shape = List[Int]()
    var j = 0
    for i in range(new_ndim):
        if i == actual_dim:
            new_shape.append(1)
        else:
            new_shape.append(old_shape[j])
            j += 1

    return reshape(tensor, new_shape)


@always_inline
fn expand_dims(tensor: ExTensor, dim: Int) raises -> ExTensor:
    """Alias for unsqueeze(). Add a size-1 dimension at specified position.

    Args:
        `tensor`: Input tensor.
        `dim`: Position to insert new dimension.

    Returns:
        Tensor with additional size-1 dimension.
    """
    return unsqueeze(tensor, dim)


fn flatten(tensor: ExTensor) raises -> ExTensor:
    """Flatten tensor to 1D.

    Args:
        `tensor`: Input tensor.

    Returns:
        1D tensor with all elements in row-major (C) order.

    Examples:
        var a = ones([3, 4], DType.float32)  # Shape (3, 4)
        var b = flatten(a)  # Shape (12,)
    """
    var numel = tensor.numel()
    var shape_1d = List[Int]()
    shape_1d.append(numel)

    return reshape(tensor, shape_1d)


@always_inline
fn ravel(tensor: ExTensor) raises -> ExTensor:
    """Flatten tensor to 1D (alias for flatten).

    Note: In NumPy, ravel can return a view. Our implementation always copies.
    TODO: Implement zero-copy views with strides.

    Args:
        `tensor`: Input tensor.

    Returns:
        1D tensor with all elements.
    """
    return flatten(tensor)


fn concatenate(tensors: List[ExTensor], axis: Int = 0) raises -> ExTensor:
    """Concatenate tensors along an existing axis.

    Args:
        `tensors`: Vector of tensors to concatenate.
        `axis`: Axis along which to concatenate (default 0)

    Returns:
        Concatenated tensor.

    Raises:
        Error if tensors have incompatible shapes.

    Examples:
        var a = ones([2, 3], DType.float32)  # 2x3
        var b = ones([3, 3], DType.float32)  # 3x3
        var tensors = List[ExTensor]()
        tensors.append(a)
        tensors.append(b)
        var c = concatenate(tensors, axis=0)  # Shape (5, 3)
    """
    var num_tensors = len(tensors)
    if num_tensors == 0:
        raise Error("concatenate: need at least one tensor")

    if num_tensors == 1:
        # Single tensor, just return copy
        return reshape(tensors[0], tensors[0].shape())

    # Get reference shape and dtype from first tensor
    var ref_shape = tensors[0].shape()
    var ndim = len(ref_shape)
    var dtype = tensors[0].dtype()

    # Handle negative axis
    var actual_axis = axis if axis >= 0 else ndim + axis
    if actual_axis < 0 or actual_axis >= ndim:
        raise Error("concatenate: axis out of range")

    # Validate all tensors have same shape except along concat axis
    var concat_size = 0
    for i in range(num_tensors):
        var shape = tensors[i].shape()

        if len(shape) != ndim:
            raise Error("concatenate: all tensors must have same number of dimensions")

        if tensors[i].dtype() != dtype:
            raise Error("concatenate: all tensors must have same dtype")

        for j in range(ndim):
            if j != actual_axis and shape[j] != ref_shape[j]:
                raise Error("concatenate: incompatible shapes")

        concat_size += shape[actual_axis]

    # Create result shape
    var result_shape = List[Int]()
    for i in range(ndim):
        if i == actual_axis:
            result_shape.append(concat_size)
        else:
            result_shape.append(ref_shape[i])

    # Create result tensor
    var result = ExTensor(result_shape, dtype)

    # Copy data from each tensor
    # This is simplified - copies element by element
    # TODO: Optimize with memcpy for contiguous blocks

    var offset = 0
    for tensor_idx in range(num_tensors):
        var t = tensors[tensor_idx]
        var t_numel = t.numel()

        for i in range(t_numel):
            var val = t._get_float64(i)
            result._set_float64(offset + i, val)

        offset += t_numel

    return result^


fn stack(tensors: List[ExTensor], axis: Int = 0) raises -> ExTensor:
    """Stack tensors along a new axis.

    Args:
        `tensors`: Vector of tensors to stack (must have identical shapes)
        `axis`: Position of new axis (default 0)

    Returns:
        Stacked tensor with one additional dimension.

    Raises:
        Error if tensors have different shapes.

    Examples:
        var a = ones([2, 3], DType.float32)  # 2x3
        var b = ones([2, 3], DType.float32)  # 2x3
        var tensors = List[ExTensor]()
        tensors.append(a)
        tensors.append(b)
        var c = stack(tensors, axis=0)  # Shape (2, 2, 3)
    """
    var num_tensors = len(tensors)
    if num_tensors == 0:
        raise Error("stack: need at least one tensor")

    # All tensors must have identical shapes
    var ref_shape = tensors[0].shape()
    var ndim = len(ref_shape)
    var dtype = tensors[0].dtype()

    for i in range(1, num_tensors):
        var shape = tensors[i].shape()

        if len(shape) != ndim:
            raise Error("stack: all tensors must have same number of dimensions")

        if tensors[i].dtype() != dtype:
            raise Error("stack: all tensors must have same dtype")

        for j in range(ndim):
            if shape[j] != ref_shape[j]:
                raise Error("stack: all tensors must have identical shapes")

    # Add unsqueeze dimension to each tensor
    var new_ndim = ndim + 1
    var actual_axis = axis if axis >= 0 else new_ndim + axis

    if actual_axis < 0 or actual_axis > ndim:
        raise Error("stack: axis out of range")

    # Unsqueeze each tensor and concatenate
    var unsqueezed = List[ExTensor]()
    for i in range(num_tensors):
        unsqueezed.append(unsqueeze(tensors[i], actual_axis))

    return concatenate(unsqueezed, actual_axis)


# ============================================================================
# Shape Computation Functions for Neural Network Layers
# ============================================================================

fn conv2d_output_shape(
    input_h: Int, input_w: Int,
    kernel_h: Int, kernel_w: Int,
    stride: Int, padding: Int,
    dilation: Int = 1
) -> Tuple[Int, Int]:
    """Compute output dimensions for 2D convolution.

    Calculates the spatial output dimensions (height, width) of a 2D convolution
    operation given input dimensions, kernel size, stride, padding, and dilation.

    Args:
        `input_h`: Input height in pixels
        `input_w`: Input width in pixels
        `kernel_h`: Kernel height in pixels
        `kernel_w`: Kernel width in pixels
        `stride`: Convolution stride (same for both dimensions)
        `padding`: Zero-padding added to input (same for all sides)
        `dilation`: Dilation factor for kernel (default: 1 for standard convolution)

    Returns:
        Tuple of (output_height, output_width)

    Formula:
        output_h = (input_h + 2*padding - dilation*(kernel_h - 1) - 1) // stride + 1
        output_w = (input_w + 2*padding - dilation*(kernel_w - 1) - 1) // stride + 1

    Examples:
        # Standard 3x3 convolution with stride=1, padding=1
        var out_h, out_w = conv2d_output_shape(224, 224, 3, 3, 1, 1)  # (224, 224)

        # 5x5 convolution with stride=2, padding=2
        var out_h, out_w = conv2d_output_shape(224, 224, 5, 5, 2, 2)  # (112, 112)

        # Dilated convolution (dilation=2)
        var out_h, out_w = conv2d_output_shape(224, 224, 3, 3, 1, 1, dilation=2)  # (222, 222)
    """
    var out_h = (input_h + 2 * padding - dilation * (kernel_h - 1) - 1) // stride + 1
    var out_w = (input_w + 2 * padding - dilation * (kernel_w - 1) - 1) // stride + 1
    return Tuple[Int, Int](out_h, out_w)


fn pool_output_shape(
    input_h: Int, input_w: Int,
    kernel_size: Int,
    stride: Int, padding: Int
) -> Tuple[Int, Int]:
    """Compute output dimensions for 2D pooling.

    Calculates the spatial output dimensions (height, width) of a 2D pooling
    operation given input dimensions, kernel size, stride, and padding.

    Args:
        `input_h`: Input height in pixels
        `input_w`: Input width in pixels
        `kernel_size`: Pooling window size (square, same for both dimensions)
        `stride`: Pooling stride (same for both dimensions)
        `padding`: Zero-padding added to input (same for all sides)

    Returns:
        Tuple of (output_height, output_width)

    Formula:
        output_h = (input_h + 2*padding - kernel_size) // stride + 1
        output_w = (input_w + 2*padding - kernel_size) // stride + 1

    Examples:
        # 2x2 max pooling with stride=2, no padding
        var out_h, out_w = pool_output_shape(224, 224, 2, 2, 0)  # (112, 112)

        # 3x3 pooling with stride=1, padding=1 (same spatial dims)
        var out_h, out_w = pool_output_shape(224, 224, 3, 1, 1)  # (224, 224)
    """
    var out_h = (input_h + 2 * padding - kernel_size) // stride + 1
    var out_w = (input_w + 2 * padding - kernel_size) // stride + 1
    return Tuple[Int, Int](out_h, out_w)


fn flatten_size(height: Int, width: Int, channels: Int) -> Int:
    """Compute flattened size for fully connected layer input.

    Calculates the total number of elements in a flattened tensor from
    4D spatial dimensions. Used to determine input size for dense/linear layers
    following convolutional or pooling layers.

    Args:
        `height`: Spatial height dimension
        `width`: Spatial width dimension
        `channels`: Number of channels

    Returns:
        Total number of elements: height * width * channels

    Examples:
        # After final pooling layer in CNN
        var fc_input_size = flatten_size(7, 7, 512)  # 25088 for 7x7x512 feature map
        var fc_weight_shape = [4096, 25088]  # Common dense layer size

        # After initial conv layer
        var fc_input_size = flatten_size(112, 112, 64)  # 802816 elements
    """
    return height * width * channels
