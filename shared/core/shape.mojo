"""Shape manipulation operations for ExTensor.

Implements shape operations like reshape, squeeze, unsqueeze, flatten, concatenate, stack, split
Following the Python Array API Standard 2023.12.

Optimizations:
- Zero-copy views with stride-based indexing
- memcpy for bulk copying of contiguous memory blocks
- Automatic contiguity detection and conversion
"""

from collections import List
from memory import memcpy, UnsafePointer
from .extensor import ExTensor


# ============================================================================
# Zero-Copy Views and Memory Optimization Helpers
# ============================================================================


fn is_contiguous(tensor: ExTensor) -> Bool:
    """Check if tensor data is contiguous in memory (row-major C order).

        A tensor is contiguous if elements are laid out sequentially in memory
        with no gaps. This is true when strides match C-order (row-major) layout.

    Args:
            tensor: The tensor to check.

    Returns:
            True if tensor is contiguous in memory, False otherwise.

    Note:
            Contiguous tensors can be efficiently copied with memcpy instead of
            element-by-element copying.
    """
    var shape = tensor.shape()
    var ndim = len(shape)

    if ndim == 0:
        return True  # Scalar is trivially contiguous

    if ndim == 1:
        # 1D tensor is contiguous if stride is 1
        return tensor._strides[0] == 1

    # For multi-dimensional tensors, check if strides match C-order
    # In C-order (row-major), stride[i] = product(shape[i+1:])
    var expected_stride = 1
    for i in range(ndim - 1, -1, -1):
        if tensor._strides[i] != expected_stride:
            return False
        expected_stride *= shape[i]

    return True


fn as_contiguous(tensor: ExTensor) raises -> ExTensor:
    """Convert tensor to contiguous memory layout if needed.

        If the tensor is already contiguous, returns a copy. If it's a view with
        non-contiguous strides, creates a new contiguous copy.

    Args:
            tensor: The tensor to make contiguous.

    Returns:
            A new contiguous tensor with the same data.

    Note:
            This function always copies data. For zero-copy operations, check
            is_contiguous() first.
    """
    if is_contiguous(tensor):
        # Already contiguous - just copy
        var shape = tensor.shape()
        var result = ExTensor(shape, tensor.dtype())
        var numel = tensor.numel()

        # Use memcpy for efficient bulk copy
        var dtype_size = tensor._get_dtype_size()
        var total_bytes = numel * dtype_size
        memcpy(dest=result._data, src=tensor._data, count=total_bytes)

        return result^
    else:
        # Non-contiguous - copy element by element to create contiguous layout
        var shape = tensor.shape()
        var result = ExTensor(shape, tensor.dtype())
        var numel = tensor.numel()

        for i in range(numel):
            var val = tensor._get_float64(i)
            result._set_float64(i, val)

        return result^


fn view(tensor: ExTensor, new_shape: List[Int]) raises -> ExTensor:
    """Create a zero-copy view of tensor with new shape (if compatible).

        Attempts to create a view with different shape while preserving the
        underlying data and strides. Returns a view if possible, otherwise raises
        an error.

        This is more strict than reshape() which always copies. view() only succeeds
        if the new shape is compatible with the current stride pattern.

    Args:
            tensor: Input tensor.
            new_shape: Target shape.

    Returns:
            A new ExTensor sharing the same data with different shape/strides.

    Raises:
            Error: If reshape cannot be done as a view (would require data movement).

    Note:
            This is an advanced function. Most code should use reshape() which
            handles all cases by copying if necessary.

    Examples:
            # View works for compatible reshapes
            var a = ones([2, 3], DType.float32)  # Contiguous (2, 3)
            var b = view(a, [6])  # Creates view with shape (6,)

            # View fails for non-trivial reshapes
            # var c = view(a, [3, 2])  # Would fail - need to transpose memory layout
    """
    var old_shape = tensor.shape()
    var old_numel = tensor.numel()
    var new_numel = 1
    var new_len = len(new_shape)

    # Validate new shape has same total elements
    for i in range(new_len):
        new_numel *= new_shape[i]

    if new_numel != old_numel:
        raise Error("view: new shape must have same number of elements")

    # Use ExTensor's built-in reshape which creates views via __copyinit__
    # This leverages reference counting for safe shared ownership
    return tensor.reshape(new_shape)


fn reshape(tensor: ExTensor, new_shape: List[Int]) raises -> ExTensor:
    """Reshape tensor to new shape.

    Args:
            tensor: Input tensor.
            new_shape: Target shape (must have same total number of elements).

    Returns:
            A new tensor with the specified shape.

    Raises:
            Error: If new shape has different number of elements.

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
                raise Error(
                    "reshape: can only specify one unknown dimension (-1)"
                )
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
            tensor: Input tensor.
            dim: Specific dimension to squeeze (optional, default squeezes all size-1 dims).

    Returns:
            Tensor with size-1 dimensions removed.

    Raises:
            Error: If specified dim is not size 1.

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
            tensor: Input tensor.
            dim: Position to insert new dimension (supports negative indexing).

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
            tensor: Input tensor.
            dim: Position to insert new dimension.

    Returns:
            Tensor with additional size-1 dimension.
    """
    return unsqueeze(tensor, dim)


fn flatten(tensor: ExTensor) raises -> ExTensor:
    """Flatten tensor to 1D.

    Args:
            tensor: Input tensor.

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

        Note: Our implementation now uses zero-copy views for contiguous tensors.
        If the tensor is contiguous, ravel() returns a view. Otherwise, it copies.

    Args:
            tensor: Input tensor.

    Returns:
            1D tensor with all elements (may be a view if contiguous).

    Examples:
            var a = ones([2, 3], DType.float32)  # Contiguous
            var b = ravel(a)  # Returns view of shape (6,)
    """
    # For contiguous tensors, we can safely flatten as a view
    # For non-contiguous tensors, we need to copy
    if is_contiguous(tensor):
        var new_shape = List[Int]()
        new_shape.append(tensor.numel())
        return view(tensor, new_shape)
    else:
        return flatten(tensor)


fn concatenate(tensors: List[ExTensor], axis: Int = 0) raises -> ExTensor:
    """Concatenate tensors along an existing axis.

    Args:
            tensors: Vector of tensors to concatenate.
            axis: Axis along which to concatenate (default 0).

    Returns:
            Concatenated tensor.

    Raises:
            Error: If tensors have incompatible shapes.

    Examples:
            var a = ones([2, 3], DType.float32)  # 2x3
            var b = ones([3, 3], DType.float32)  # 3x3
            var tensors : List[ExTensor] = []
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
            raise Error(
                "concatenate: all tensors must have same number of dimensions"
            )

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

    # Copy data from each tensor with memcpy optimization for contiguous tensors
    var dtype_size = result._get_dtype_size()
    var offset_bytes = 0

    for tensor_idx in range(num_tensors):
        var t = tensors[tensor_idx]
        var t_numel = t.numel()
        var t_bytes = t_numel * dtype_size

        # Use memcpy for efficient bulk copy if source is contiguous
        if is_contiguous(t):
            memcpy(
                dest=(result._data + offset_bytes).bitcast[UInt8](),
                src=t._data,
                count=t_bytes,
            )
        else:
            # Fall back to element-by-element copy for non-contiguous tensors
            for i in range(t_numel):
                var val = t._get_float64(i)
                result._set_float64(offset_bytes // dtype_size + i, val)

        offset_bytes += t_bytes

    return result^


fn stack(tensors: List[ExTensor], axis: Int = 0) raises -> ExTensor:
    """Stack tensors along a new axis.

    Args:
            tensors: Vector of tensors to stack (must have identical shapes).
            axis: Position of new axis (default 0).

    Returns:
            Stacked tensor with one additional dimension.

    Raises:
            Error: If tensors have different shapes.

    Examples:
            var a = ones([2, 3], DType.float32)  # 2x3
            var b = ones([2, 3], DType.float32)  # 2x3
            var tensors : List[ExTensor] = []
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
            raise Error(
                "stack: all tensors must have same number of dimensions"
            )

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
    var unsqueezed: List[ExTensor] = []
    for i in range(num_tensors):
        unsqueezed.append(unsqueeze(tensors[i], actual_axis))

    return concatenate(unsqueezed, actual_axis)


# ============================================================================
# Shape Computation Functions for Neural Network Layers
# ============================================================================


fn conv2d_output_shape(
    input_h: Int,
    input_w: Int,
    kernel_h: Int,
    kernel_w: Int,
    stride: Int,
    padding: Int,
    dilation: Int = 1,
) -> Tuple[Int, Int]:
    """Compute output dimensions for 2D convolution.

        Calculates the spatial output dimensions (height, width) of a 2D convolution
        operation given input dimensions, kernel size, stride, padding, and dilation.

    Args:
            input_h: Input height in pixels.
            input_w: Input width in pixels.
            kernel_h: Kernel height in pixels.
            kernel_w: Kernel width in pixels.
            stride: Convolution stride (same for both dimensions).
            padding: Zero-padding added to input (same for all sides).
            dilation: Dilation factor for kernel (default: 1 for standard convolution).

    Returns:
            Tuple of (output_height, output_width).

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
    var out_h = (
        input_h + 2 * padding - dilation * (kernel_h - 1) - 1
    ) // stride + 1
    var out_w = (
        input_w + 2 * padding - dilation * (kernel_w - 1) - 1
    ) // stride + 1
    return Tuple[Int, Int](out_h, out_w)


fn pool_output_shape(
    input_h: Int, input_w: Int, kernel_size: Int, stride: Int, padding: Int
) -> Tuple[Int, Int]:
    """Compute output dimensions for 2D pooling.

        Calculates the spatial output dimensions (height, width) of a 2D pooling
        operation given input dimensions, kernel size, stride, and padding.

    Args:
            input_h: Input height in pixels.
            input_w: Input width in pixels.
            kernel_size: Pooling window size (square, same for both dimensions).
            stride: Pooling stride (same for both dimensions).
            padding: Zero-padding added to input (same for all sides).

    Returns:
            Tuple of (output_height, output_width).

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
            height: Spatial height dimension.
            width: Spatial width dimension.
            channels: Number of channels.

    Returns:
            Total number of elements: height * width * channels.

    Examples:
            # After final pooling layer in CNN
            var fc_input_size = flatten_size(7, 7, 512)  # 25088 for 7x7x512 feature map
            var fc_weight_shape = [4096, 25088]  # Common dense layer size

            # After initial conv layer
            var fc_input_size = flatten_size(112, 112, 64)  # 802816 elements
    """
    return height * width * channels


fn flatten_to_2d(tensor: ExTensor) raises -> ExTensor:
    """Flatten a 4D tensor to 2D, preserving the batch dimension.

        Commonly used before fully connected layers in CNNs to reshape
        (batch, channels, height, width) to (batch, channels * height * width).

    Args:
            tensor: Input tensor of shape (batch, channels, height, width).

    Returns:
            Tensor of shape (batch, channels * height * width).

    Raises:
            Error: If input tensor is not 4D.

    Examples:
            # After pooling layer, flatten before FC layer
            var pool_out = maxpool2d(x, kernel_size=2, stride=2)  # (32, 64, 7, 7)
            var flattened = flatten_to_2d(pool_out)  # (32, 3136)

            # Use in forward pass
            var fc_input = flatten_to_2d(conv_output)
            var fc_output = linear(fc_input, weights, bias)
    """
    var shape = tensor.shape()

    if len(shape) != 4:
        raise Error(
            "flatten_to_2d requires 4D input (batch, channels, height, width),"
            " got "
            + String(len(shape))
            + "D"
        )

    var batch_size = shape[0]
    var channels = shape[1]
    var height = shape[2]
    var width = shape[3]
    var flattened_size = channels * height * width

    var new_shape = List[Int]()
    new_shape.append(batch_size)
    new_shape.append(flattened_size)
    return reshape(tensor, new_shape)


fn transposed_conv2d_output_shape(
    input_h: Int,
    input_w: Int,
    kernel_h: Int,
    kernel_w: Int,
    stride: Int,
    padding: Int,
    output_padding: Int = 0,
) -> Tuple[Int, Int]:
    """Compute output dimensions for 2D transposed convolution.

        Calculates the spatial output dimensions (height, width) of a 2D transposed
        convolution (deconvolution) operation. Transposed convolution upsamples the
        input and is commonly used in decoder networks and generative models.

    Args:
            input_h: Input height in pixels.
            input_w: Input width in pixels.
            kernel_h: Kernel height in pixels.
            kernel_w: Kernel width in pixels.
            stride: Convolution stride (same for both dimensions).
            padding: Padding applied to input (same for all sides).
            output_padding: Additional padding added to output (default: 0).

    Returns:
            Tuple of (output_height, output_width).

        Formula:
            output_h = (input_h - 1) * stride - 2 * padding + kernel_h + output_padding
            output_w = (input_w - 1) * stride - 2 * padding + kernel_w + output_padding

    Examples:
            # Upsample 7x7 to 14x14 with stride=2
            var out_h, out_w = transposed_conv2d_output_shape(7, 7, 4, 4, 2, 1)  # (14, 14)

            # Upsample 14x14 to 28x28 with stride=2
            var out_h, out_w = transposed_conv2d_output_shape(14, 14, 4, 4, 2, 1)  # (28, 28)
    """
    var out_h = (input_h - 1) * stride - 2 * padding + kernel_h + output_padding
    var out_w = (input_w - 1) * stride - 2 * padding + kernel_w + output_padding
    return Tuple[Int, Int](out_h, out_w)


fn global_avgpool_output_shape(
    batch: Int, channels: Int
) -> Tuple[Int, Int, Int, Int]:
    """Compute output shape for global average pooling.

        Global average pooling reduces each channel to a single value by averaging
        all spatial dimensions. The output has shape (batch, channels, 1, 1).

    Args:
            batch: Batch size.
            channels: Number of channels.

    Returns:
            Tuple of (batch, channels, 1, 1).

    Examples:
            # Global average pooling on feature map
            var shape = global_avgpool_output_shape(32, 512)  # (32, 512, 1, 1)

            # Common in classification networks (replaces flatten + FC)
            var shape = global_avgpool_output_shape(16, 2048)  # (16, 2048, 1, 1)
    """
    return Tuple[Int, Int, Int, Int](batch, channels, 1, 1)


fn linear_output_shape(batch_size: Int, out_features: Int) -> Tuple[Int, Int]:
    """Compute output shape for linear/dense layer.

        Linear layers transform input features to output features. The output
        shape is (batch_size, out_features).

    Args:
            batch_size: Number of samples in the batch.
            out_features: Number of output features (neurons).

    Returns:
            Tuple of (batch_size, out_features).

    Examples:
            # Classification head: 512 features -> 10 classes
            var shape = linear_output_shape(32, 10)  # (32, 10)

            # Hidden layer: 784 features -> 256 hidden units
            var shape = linear_output_shape(64, 256)  # (64, 256)
    """
    return Tuple[Int, Int](batch_size, out_features)
