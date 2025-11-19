"""Functional 2D convolution operations.

This module provides pure functional implementations of 2D convolution operations.
The caller manages all state (kernels, biases).

Note: Full convolution implementation requires im2col or direct convolution loops
with SIMD optimization. This is a placeholder for the functional API.
"""

from .extensor import ExTensor


fn conv2d(
    x: ExTensor,
    kernel: ExTensor,
    bias: ExTensor,
    stride: Int = 1,
    padding: Int = 0
) raises -> ExTensor:
    """Functional 2D convolution: y = conv2d(x, kernel) + bias

    Pure function - caller manages kernel and bias. No internal state.

    Args:
        x: Input tensor of shape (batch, in_channels, height, width)
        kernel: Convolution kernels of shape (out_channels, in_channels, kH, kW)
        bias: Bias vector of shape (out_channels,)
        stride: Stride for convolution (default: 1)
        padding: Zero-padding added to input (default: 0)

    Returns:
        Output tensor of shape (batch, out_channels, out_height, out_width)
        where:
            out_height = (height + 2*padding - kH) / stride + 1
            out_width = (width + 2*padding - kW) / stride + 1

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
        Error: Not yet implemented. Placeholder for functional API.

    TODO: Implement using im2col + matmul or direct convolution with SIMD.
    """
    raise Error("conv2d not yet implemented - placeholder for functional API")


fn conv2d_no_bias(
    x: ExTensor,
    kernel: ExTensor,
    stride: Int = 1,
    padding: Int = 0
) raises -> ExTensor:
    """Functional 2D convolution without bias: y = conv2d(x, kernel)

    Pure function for convolution with no bias term.

    Args:
        x: Input tensor of shape (batch, in_channels, height, width)
        kernel: Convolution kernels of shape (out_channels, in_channels, kH, kW)
        stride: Stride for convolution (default: 1)
        padding: Zero-padding added to input (default: 0)

    Returns:
        Output tensor of shape (batch, out_channels, out_height, out_width)

    Raises:
        Error: Not yet implemented.

    TODO: Implement using im2col + matmul or direct convolution with SIMD.
    """
    raise Error("conv2d_no_bias not yet implemented - placeholder for functional API")
