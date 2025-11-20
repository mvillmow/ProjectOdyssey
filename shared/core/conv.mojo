"""Functional multi-dimensional convolution operations.

This module provides pure functional implementations of multi-dimensional
convolution operations using direct convolution (not im2col).
The caller manages all state (kernels, biases).
"""

from .extensor import ExTensor, zeros
from .arithmetic import add
from collections.vector import DynamicVector
from math import max as math_max


fn conv2d(
    x: ExTensor,
    kernel: ExTensor,
    bias: ExTensor,
    stride: Int = 1,
    padding: Int = 0
) raises -> ExTensor:
    """Functional 2D convolution using direct convolution: y = conv2d(x, kernel) + bias

    Pure function - caller manages kernel and bias. No internal state.
    Uses direct convolution algorithm (not im2col).

    Args:
        x: Input tensor of shape (batch, in_channels, height, width)
        kernel: Convolution kernels of shape (out_channels, in_channels, kH, kW)
        bias: Bias vector of shape (out_channels,)
        stride: Stride for convolution (default: 1)
        padding: Zero-padding added to input (default: 0)

    Returns:
        Output tensor of shape (batch, out_channels, out_height, out_width)
        where:
            out_height = (height + 2*padding - kH) // stride + 1
            out_width = (width + 2*padding - kW) // stride + 1

    Example:
        ```mojo
        from shared.core import ExTensor, conv2d, zeros, he_uniform

        # Caller manages state
        var kernel = he_uniform((16, 3, 3, 3), DType.float32)  # 16 filters, 3x3
        var bias = zeros(16, DType.float32)

        # Pure function call
        var output = conv2d(input, kernel, bias, stride=1, padding=1)
        ```

    Raises:
        Error: If tensor shapes are incompatible.
    """
    # Get input dimensions
    var x_shape = x.shape()
    if x_shape.size != 4:
        raise Error("Input must be 4D tensor (batch, channels, height, width)")

    var batch = x_shape[0]
    var in_channels = x_shape[1]
    var in_height = x_shape[2]
    var in_width = x_shape[3]

    # Get kernel dimensions
    var k_shape = kernel.shape()
    if k_shape.size != 4:
        raise Error("Kernel must be 4D tensor (out_channels, in_channels, kH, kW)")

    var out_channels = k_shape[0]
    var kernel_in_channels = k_shape[1]
    var kH = k_shape[2]
    var kW = k_shape[3]

    if kernel_in_channels != in_channels:
        raise Error("Kernel in_channels must match input in_channels")

    # Compute output dimensions
    var out_height = (in_height + 2 * padding - kH) // stride + 1
    var out_width = (in_width + 2 * padding - kW) // stride + 1

    # Create output tensor
    var out_shape = DynamicVector[Int](4)
    out_shape[0] = batch
    out_shape[1] = out_channels
    out_shape[2] = out_height
    out_shape[3] = out_width
    var output = zeros(out_shape, x.dtype())

    # Direct convolution algorithm
    # For each batch element
    for b in range(batch):
        # For each output channel
        for oc in range(out_channels):
            # For each output position
            for oh in range(out_height):
                for ow in range(out_width):
                    var sum_val = Float32(0.0)

                    # Compute input position
                    var in_h_start = oh * stride - padding
                    var in_w_start = ow * stride - padding

                    # Convolve over input channels and kernel
                    for ic in range(in_channels):
                        for kh in range(kH):
                            for kw in range(kW):
                                # Input position with padding
                                var in_h = in_h_start + kh
                                var in_w = in_w_start + kw

                                # Check bounds (zero padding)
                                if in_h >= 0 and in_h < in_height and in_w >= 0 and in_w < in_width:
                                    # Get input and kernel values
                                    var in_idx = b * (in_channels * in_height * in_width) + ic * (in_height * in_width) + in_h * in_width + in_w
                                    var k_idx = oc * (in_channels * kH * kW) + ic * (kH * kW) + kh * kW + kw

                                    var in_val = x._data.bitcast[Float32]()[in_idx]
                                    var k_val = kernel._data.bitcast[Float32]()[k_idx]

                                    sum_val += in_val * k_val

                    # Add bias
                    var b_val = bias._data.bitcast[Float32]()[oc]
                    sum_val += b_val

                    # Write to output
                    var out_idx = b * (out_channels * out_height * out_width) + oc * (out_height * out_width) + oh * out_width + ow
                    output._data.bitcast[Float32]()[out_idx] = sum_val

    return output


fn conv2d_no_bias(
    x: ExTensor,
    kernel: ExTensor,
    stride: Int = 1,
    padding: Int = 0
) raises -> ExTensor:
    """Functional 2D convolution without bias: y = conv2d(x, kernel)

    Pure function for convolution with no bias term.
    Uses direct convolution algorithm.

    Args:
        x: Input tensor of shape (batch, in_channels, height, width)
        kernel: Convolution kernels of shape (out_channels, in_channels, kH, kW)
        stride: Stride for convolution (default: 1)
        padding: Zero-padding added to input (default: 0)

    Returns:
        Output tensor of shape (batch, out_channels, out_height, out_width)

    Raises:
        Error: If tensor shapes are incompatible.
    """
    # Create zero bias
    var k_shape = kernel.shape()
    var out_channels = k_shape[0]
    var bias_shape = DynamicVector[Int](1)
    bias_shape[0] = out_channels
    var bias = zeros(bias_shape, x.dtype())

    return conv2d(x, kernel, bias, stride, padding)

