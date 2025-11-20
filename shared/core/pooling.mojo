"""Functional pooling operations for 2D inputs.

This module provides pure functional implementations of pooling operations.
All operations are stateless - caller provides all inputs.
"""

from .extensor import ExTensor, zeros
from collections.vector import DynamicVector
from math import max as math_max, min as math_min


fn maxpool2d(
    x: ExTensor,
    kernel_size: Int,
    stride: Int = 0,  # 0 means use kernel_size
    padding: Int = 0,
    method: String = "direct"
) raises -> ExTensor:
    """Functional 2D max pooling with selectable implementation.

    Pure function - no internal state. Downsamples spatial dimensions by
    taking maximum value in each kernel_size x kernel_size window.

    Args:
        x: Input tensor of shape (batch, channels, height, width)
        kernel_size: Size of the pooling window
        stride: Stride for pooling (default: kernel_size if 0)
        padding: Zero-padding added to input (default: 0)
        method: Implementation method - "direct" (default), "optimized" (future)

    Returns:
        Output tensor of shape (batch, channels, out_height, out_width)
        where:
            stride_actual = kernel_size if stride == 0 else stride
            out_height = (height + 2*padding - kernel_size) / stride_actual + 1
            out_width = (width + 2*padding - kernel_size) / stride_actual + 1

    Example:
        ```mojo
        from shared.core import ExTensor, maxpool2d

        # Pure function call - no state to manage
        var pooled = maxpool2d(input, kernel_size=2, stride=2)

        # Or select implementation method
        var pooled = maxpool2d(input, kernel_size=2, stride=2, method="direct")
        ```

    Raises:
        Error: If tensor shapes are incompatible or method is unsupported.
    """
    if method != "direct":
        raise Error("Only 'direct' method is currently supported for maxpool2d")

    # Get input dimensions
    var x_shape = x.shape()
    if x_shape.size != 4:
        raise Error("Input must be 4D tensor (batch, channels, height, width)")

    var batch = x_shape[0]
    var channels = x_shape[1]
    var in_height = x_shape[2]
    var in_width = x_shape[3]

    # Determine actual stride
    var actual_stride = stride if stride > 0 else kernel_size

    # Compute output dimensions
    var out_height = (in_height + 2 * padding - kernel_size) // actual_stride + 1
    var out_width = (in_width + 2 * padding - kernel_size) // actual_stride + 1

    # Create output tensor
    var out_shape = DynamicVector[Int](4)
    out_shape[0] = batch
    out_shape[1] = channels
    out_shape[2] = out_height
    out_shape[3] = out_width
    var output = zeros(out_shape, x.dtype())

    # Direct max pooling algorithm
    for b in range(batch):
        for c in range(channels):
            for oh in range(out_height):
                for ow in range(out_width):
                    # Compute input window bounds
                    var in_h_start = oh * actual_stride - padding
                    var in_w_start = ow * actual_stride - padding
                    var in_h_end = in_h_start + kernel_size
                    var in_w_end = in_w_start + kernel_size

                    # Find maximum in window
                    var max_val = Float32(-1e9)  # Very small initial value

                    for kh in range(kernel_size):
                        for kw in range(kernel_size):
                            var in_h = in_h_start + kh
                            var in_w = in_w_start + kw

                            # Check bounds (zero padding treated as -inf for max)
                            if in_h >= 0 and in_h < in_height and in_w >= 0 and in_w < in_width:
                                var in_idx = b * (channels * in_height * in_width) + c * (in_height * in_width) + in_h * in_width + in_w
                                var val = x._data.bitcast[Float32]()[in_idx]
                                if val > max_val:
                                    max_val = val

                    # Write maximum to output
                    var out_idx = b * (channels * out_height * out_width) + c * (out_height * out_width) + oh * out_width + ow
                    output._data.bitcast[Float32]()[out_idx] = max_val

    return output


fn avgpool2d(
    x: ExTensor,
    kernel_size: Int,
    stride: Int = 0,  # 0 means use kernel_size
    padding: Int = 0,
    method: String = "direct"
) raises -> ExTensor:
    """Functional 2D average pooling with selectable implementation.

    Pure function - no internal state. Downsamples spatial dimensions by
    taking average value in each kernel_size x kernel_size window.

    Args:
        x: Input tensor of shape (batch, channels, height, width)
        kernel_size: Size of the pooling window
        stride: Stride for pooling (default: kernel_size if 0)
        padding: Zero-padding added to input (default: 0)
        method: Implementation method - "direct" (default), "optimized" (future)

    Returns:
        Output tensor of shape (batch, channels, out_height, out_width)

    Example:
        ```mojo
        from shared.core import ExTensor, avgpool2d

        # Pure function call - no state to manage
        var pooled = avgpool2d(input, kernel_size=2, stride=2)

        # Or select implementation method
        var pooled = avgpool2d(input, kernel_size=2, stride=2, method="direct")
        ```

    Raises:
        Error: If tensor shapes are incompatible or method is unsupported.
    """
    if method != "direct":
        raise Error("Only 'direct' method is currently supported for avgpool2d")

    # Get input dimensions
    var x_shape = x.shape()
    if x_shape.size != 4:
        raise Error("Input must be 4D tensor (batch, channels, height, width)")

    var batch = x_shape[0]
    var channels = x_shape[1]
    var in_height = x_shape[2]
    var in_width = x_shape[3]

    # Determine actual stride
    var actual_stride = stride if stride > 0 else kernel_size

    # Compute output dimensions
    var out_height = (in_height + 2 * padding - kernel_size) // actual_stride + 1
    var out_width = (in_width + 2 * padding - kernel_size) // actual_stride + 1

    # Create output tensor
    var out_shape = DynamicVector[Int](4)
    out_shape[0] = batch
    out_shape[1] = channels
    out_shape[2] = out_height
    out_shape[3] = out_width
    var output = zeros(out_shape, x.dtype())

    # Direct average pooling algorithm
    for b in range(batch):
        for c in range(channels):
            for oh in range(out_height):
                for ow in range(out_width):
                    # Compute input window bounds
                    var in_h_start = oh * actual_stride - padding
                    var in_w_start = ow * actual_stride - padding

                    # Compute sum and count in window
                    var sum_val = Float32(0.0)
                    var count = 0

                    for kh in range(kernel_size):
                        for kw in range(kernel_size):
                            var in_h = in_h_start + kh
                            var in_w = in_w_start + kw

                            # Check bounds (zero padding counted in divisor)
                            if in_h >= 0 and in_h < in_height and in_w >= 0 and in_w < in_width:
                                var in_idx = b * (channels * in_height * in_width) + c * (in_height * in_width) + in_h * in_width + in_w
                                var val = x._data.bitcast[Float32]()[in_idx]
                                sum_val += val
                                count += 1

                    # Compute average
                    var avg_val = sum_val / Float32(count) if count > 0 else Float32(0.0)

                    # Write average to output
                    var out_idx = b * (channels * out_height * out_width) + c * (out_height * out_width) + oh * out_width + ow
                    output._data.bitcast[Float32]()[out_idx] = avg_val

    return output


fn global_avgpool2d(x: ExTensor, method: String = "direct") raises -> ExTensor:
    """Functional global average pooling with selectable implementation.

    Pure function that reduces spatial dimensions (H, W) to (1, 1) by
    averaging all values in each channel.

    Args:
        x: Input tensor of shape (batch, channels, height, width)
        method: Implementation method - "direct" (default), "optimized" (future)

    Returns:
        Output tensor of shape (batch, channels, 1, 1)

    Example:
        ```mojo
        from shared.core import ExTensor, global_avgpool2d

        # Pure function call
        var pooled = global_avgpool2d(input)  # (B, C, H, W) -> (B, C, 1, 1)

        # Or select implementation method
        var pooled = global_avgpool2d(input, method="direct")
        ```

    Raises:
        Error: If tensor shapes are incompatible or method is unsupported.
    """
    if method != "direct":
        raise Error("Only 'direct' method is currently supported for global_avgpool2d")

    # Get input dimensions
    var x_shape = x.shape()
    if x_shape.size != 4:
        raise Error("Input must be 4D tensor (batch, channels, height, width)")

    var batch = x_shape[0]
    var channels = x_shape[1]
    var height = x_shape[2]
    var width = x_shape[3]

    # Create output tensor (B, C, 1, 1)
    var out_shape = DynamicVector[Int](4)
    out_shape[0] = batch
    out_shape[1] = channels
    out_shape[2] = 1
    out_shape[3] = 1
    var output = zeros(out_shape, x.dtype())

    # Compute global average for each channel
    for b in range(batch):
        for c in range(channels):
            # Sum all spatial values
            var sum_val = Float32(0.0)

            for h in range(height):
                for w in range(width):
                    var in_idx = b * (channels * height * width) + c * (height * width) + h * width + w
                    var val = x._data.bitcast[Float32]()[in_idx]
                    sum_val += val

            # Compute average
            var avg_val = sum_val / Float32(height * width)

            # Write to output
            var out_idx = b * channels + c
            output._data.bitcast[Float32]()[out_idx] = avg_val

    return output
