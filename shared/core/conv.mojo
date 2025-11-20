"""Functional multi-dimensional convolution operations.

This module provides pure functional implementations of multi-dimensional
convolution operations using direct convolution (not im2col).
The caller manages all state (kernels, biases).
"""

from .extensor import ExTensor, zeros
from .arithmetic import add
from .reduction import sum as reduce_sum
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


fn conv2d_backward(
    grad_output: ExTensor,
    x: ExTensor,
    kernel: ExTensor,
    stride: Int = 1,
    padding: Int = 0
) raises -> (ExTensor, ExTensor, ExTensor):
    """Backward pass for 2D convolution.

    Computes gradients with respect to input, kernel, and bias.

    Math:
        Given: y = conv2d(x, kernel, bias, stride, padding)
        This function computes:
        - grad_input: Gradient w.r.t. input
        - grad_kernel: Gradient w.r.t. kernel
        - grad_bias: Gradient w.r.t. bias

    Args:
        grad_output: Gradient w.r.t. output, shape (batch, out_channels, out_H, out_W)
        x: Input from forward pass, shape (batch, in_channels, in_H, in_W)
        kernel: Kernel from forward pass, shape (out_channels, in_channels, kH, kW)
        stride: Stride used in forward pass
        padding: Padding used in forward pass

    Returns:
        Tuple of (grad_input, grad_kernel, grad_bias):
            - grad_input: Gradient w.r.t. input, shape (batch, in_channels, in_H, in_W)
            - grad_kernel: Gradient w.r.t. kernel, shape (out_channels, in_channels, kH, kW)
            - grad_bias: Gradient w.r.t. bias, shape (out_channels,)

    Example:
        ```mojo
        from shared.core import conv2d, conv2d_backward

        # Forward pass
        var output = conv2d(x, kernel, bias, stride, padding)
        # ... compute loss and grad_output ...

        # Backward pass
        var (grad_x, grad_k, grad_b) = conv2d_backward(grad_output, x, kernel, stride, padding)
        ```

    Raises:
        Error if tensor shapes are incompatible.
    """
    # Get dimensions
    var x_shape = x.shape()
    var k_shape = kernel.shape()
    var grad_out_shape = grad_output.shape()

    var batch = x_shape[0]
    var in_channels = x_shape[1]
    var in_height = x_shape[2]
    var in_width = x_shape[3]

    var out_channels = k_shape[0]
    var kH = k_shape[2]
    var kW = k_shape[3]

    var out_height = grad_out_shape[2]
    var out_width = grad_out_shape[3]

    # Initialize gradients
    var grad_input = zeros(x_shape, x.dtype())
    var grad_kernel = zeros(k_shape, kernel.dtype())

    # Compute grad_input
    # For each input position, sum contributions from all output positions it affected
    for b in range(batch):
        for ic in range(in_channels):
            for ih in range(in_height):
                for iw in range(in_width):
                    var grad_sum = Float32(0.0)

                    # This input position (ih, iw) contributed to output positions (oh, ow)
                    # where: ih = oh * stride - padding + kh
                    # So: kh = ih - oh * stride + padding
                    # For each output position
                    for oh in range(out_height):
                        for ow in range(out_width):
                            # Compute kernel offsets
                            var kh = ih - oh * stride + padding
                            var kw = iw - ow * stride + padding

                            # Check if kernel offsets are valid
                            if kh >= 0 and kh < kH and kw >= 0 and kw < kW:
                                # Sum over output channels
                                for oc in range(out_channels):
                                    # Get grad_output value
                                    var grad_out_idx = b * (out_channels * out_height * out_width) + oc * (out_height * out_width) + oh * out_width + ow
                                    var grad_out_val = grad_output._data.bitcast[Float32]()[grad_out_idx]

                                    # Get kernel value
                                    var k_idx = oc * (in_channels * kH * kW) + ic * (kH * kW) + kh * kW + kw
                                    var k_val = kernel._data.bitcast[Float32]()[k_idx]

                                    grad_sum += grad_out_val * k_val

                    # Write to grad_input
                    var grad_in_idx = b * (in_channels * in_height * in_width) + ic * (in_height * in_width) + ih * in_width + iw
                    grad_input._data.bitcast[Float32]()[grad_in_idx] = grad_sum

    # Compute grad_kernel
    # For each kernel position, sum input * grad_output over all valid positions
    for oc in range(out_channels):
        for ic in range(in_channels):
            for kh in range(kH):
                for kw in range(kW):
                    var grad_sum = Float32(0.0)

                    # For each batch element
                    for b in range(batch):
                        # For each output position
                        for oh in range(out_height):
                            for ow in range(out_width):
                                # Compute input position
                                var in_h = oh * stride - padding + kh
                                var in_w = ow * stride - padding + kw

                                # Check bounds
                                if in_h >= 0 and in_h < in_height and in_w >= 0 and in_w < in_width:
                                    # Get input value
                                    var in_idx = b * (in_channels * in_height * in_width) + ic * (in_height * in_width) + in_h * in_width + in_w
                                    var in_val = x._data.bitcast[Float32]()[in_idx]

                                    # Get grad_output value
                                    var grad_out_idx = b * (out_channels * out_height * out_width) + oc * (out_height * out_width) + oh * out_width + ow
                                    var grad_out_val = grad_output._data.bitcast[Float32]()[grad_out_idx]

                                    grad_sum += in_val * grad_out_val

                    # Write to grad_kernel
                    var grad_k_idx = oc * (in_channels * kH * kW) + ic * (kH * kW) + kh * kW + kw
                    grad_kernel._data.bitcast[Float32]()[grad_k_idx] = grad_sum

    # Compute grad_bias: sum over batch, height, width
    var grad_bias_shape = DynamicVector[Int](1)
    grad_bias_shape[0] = out_channels
    var grad_bias = zeros(grad_bias_shape, grad_output.dtype())

    for oc in range(out_channels):
        var bias_grad_sum = Float32(0.0)

        for b in range(batch):
            for oh in range(out_height):
                for ow in range(out_width):
                    var grad_out_idx = b * (out_channels * out_height * out_width) + oc * (out_height * out_width) + oh * out_width + ow
                    var grad_out_val = grad_output._data.bitcast[Float32]()[grad_out_idx]
                    bias_grad_sum += grad_out_val

        grad_bias._data.bitcast[Float32]()[oc] = bias_grad_sum

    return (grad_input, grad_kernel, grad_bias)


fn conv2d_no_bias_backward(
    grad_output: ExTensor,
    x: ExTensor,
    kernel: ExTensor,
    stride: Int = 1,
    padding: Int = 0
) raises -> (ExTensor, ExTensor):
    """Backward pass for 2D convolution without bias.

    Args:
        grad_output: Gradient w.r.t. output
        x: Input from forward pass
        kernel: Kernel from forward pass
        stride: Stride used in forward pass
        padding: Padding used in forward pass

    Returns:
        Tuple of (grad_input, grad_kernel)

    Raises:
        Error if tensor shapes are incompatible.
    """
    var (grad_input, grad_kernel, _) = conv2d_backward(grad_output, x, kernel, stride, padding)
    return (grad_input, grad_kernel)

