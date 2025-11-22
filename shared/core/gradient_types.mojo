"""Gradient container types for backward pass functions.

Provides type-safe containers for multiple gradient returns, replacing tuple
return types which are not fully supported in the current Mojo version.

This module defines:
- GradientPair: For binary operations returning 2 gradients
- GradientTriple: For ternary operations returning 3 gradients
"""

from .extensor import ExTensor


struct GradientPair:
    """Container for gradients from binary operations.

    Used for backward functions that compute gradients with respect to
    two inputs (e.g., add_backward, multiply_backward).

    Attributes:
        grad_a: Gradient with respect to first input
        grad_b: Gradient with respect to second input

    Examples:
        var grads = add_backward(grad_output, a_shape, b_shape)
        var grad_a = grads.grad_a
        var grad_b = grads.grad_b
    """

    var grad_a: ExTensor
    var grad_b: ExTensor

    fn __init__(out self, grad_a: ExTensor, grad_b: ExTensor):
        """Initialize gradient pair.

        Args:
            grad_a: Gradient tensor for first input
            grad_b: Gradient tensor for second input
        """
        self.grad_a = grad_a
        self.grad_b = grad_b


struct GradientTriple:
    """Container for gradients from ternary operations.

    Used for backward functions that compute gradients with respect to
    three inputs (e.g., linear_backward, conv2d_backward).

    Attributes:
        grad_input: Gradient with respect to input activation
        grad_weights: Gradient with respect to weights
        grad_bias: Gradient with respect to bias

    Examples:
        var grads = linear_backward(grad_output, x, weights)
        var grad_input = grads.grad_input
        var grad_weights = grads.grad_weights
        var grad_bias = grads.grad_bias
    """

    var grad_input: ExTensor
    var grad_weights: ExTensor
    var grad_bias: ExTensor

    fn __init__(
        out self,
        grad_input: ExTensor,
        grad_weights: ExTensor,
        grad_bias: ExTensor,
    ):
        """Initialize gradient triple.

        Args:
            grad_input: Gradient tensor for input
            grad_weights: Gradient tensor for weights
            grad_bias: Gradient tensor for bias
        """
        self.grad_input = grad_input
        self.grad_weights = grad_weights
        self.grad_bias = grad_bias
