"""Functional linear (fully connected) transformation.

This module provides a pure functional implementation of linear transformations,
following the pattern y = xW^T + b. The caller manages all state (weights, bias).
"""

from .extensor import ExTensor
from .matrix import matmul, transpose
from .arithmetic import add


fn linear(x: ExTensor, weights: ExTensor, bias: ExTensor) raises -> ExTensor:
    """Functional linear transformation: y = xW^T + b

    Pure function - caller manages weights and bias. No internal state.

    Args:
        x: Input tensor of shape (batch_size, in_features)
        weights: Weight matrix of shape (out_features, in_features)
        bias: Bias vector of shape (out_features,)

    Returns:
        Output tensor of shape (batch_size, out_features)

    Example:
        ```mojo
        from shared.core import ExTensor, linear, zeros, xavier_uniform

        # Caller manages state
        var w = xavier_uniform(10, 784, DType.float32)
        var b = zeros(10, DType.float32)

        # Pure function call
        var output = linear(input, w, b)
        ```

    Raises:
        Error if shapes are incompatible for matrix multiplication.
    """
    # Compute xW^T
    var out = matmul(x, transpose(weights))

    # Add bias
    return add(out, bias)


fn linear_no_bias(x: ExTensor, weights: ExTensor) raises -> ExTensor:
    """Functional linear transformation without bias: y = xW^T

    Pure function for linear transformation with no bias term.

    Args:
        x: Input tensor of shape (batch_size, in_features)
        weights: Weight matrix of shape (out_features, in_features)

    Returns:
        Output tensor of shape (batch_size, out_features)

    Raises:
        Error if shapes are incompatible for matrix multiplication.
    """
    return matmul(x, transpose(weights))
