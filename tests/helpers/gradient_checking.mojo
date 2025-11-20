"""Numerical gradient checking for validating backward passes.

Provides gold-standard gradient validation using finite differences.
Critical for ensuring mathematical correctness of backpropagation.

Example usage:
    fn forward(x: ExTensor) raises -> ExTensor:
        return relu(x)

    var x = create_test_tensor()
    var analytical_grad = relu_backward(grad_output, x)
    var numerical_grad = compute_numerical_gradient(forward, x)
    assert_gradients_close(analytical_grad, numerical_grad)
"""

from shared.core.extensor import ExTensor, zeros_like, ones_like
from math import abs as math_abs
from collections.vector import DynamicVector


fn compute_numerical_gradient(
    forward_fn: fn(ExTensor) raises -> ExTensor,
    x: ExTensor,
    epsilon: Float64 = 1e-5
) raises -> ExTensor:
    """Compute numerical gradient using finite differences.

    Uses central difference formula: ∇f(x) ≈ (f(x + ε) - f(x - ε)) / 2ε

    This is the gold standard for validating analytical gradients. The central
    difference method has O(ε²) error compared to O(ε) for forward/backward
    differences, making it much more accurate.

    Args:
        forward_fn: Function that computes forward pass (takes ExTensor, returns ExTensor)
        x: Input tensor at which to compute gradient
        epsilon: Small perturbation for finite differences (default: 1e-5)

    Returns:
        ExTensor containing numerical gradient, same shape as x

    Raises:
        Error: If forward function fails or dtypes are incompatible

    Notes:
        - For scalar outputs, gradient shape matches input shape
        - For vector outputs, this computes Jacobian row-by-row (expensive!)
        - Epsilon of 1e-5 is a good compromise between roundoff and truncation error
        - Use smaller epsilon (1e-7) for Float64, larger (1e-4) for Float16

    Mathematical basis:
        Taylor expansion: f(x+ε) = f(x) + ε·f'(x) + O(ε²)
        Taylor expansion: f(x-ε) = f(x) - ε·f'(x) + O(ε²)
        Subtracting: f(x+ε) - f(x-ε) = 2ε·f'(x) + O(ε³)
        Therefore: f'(x) ≈ (f(x+ε) - f(x-ε)) / 2ε  [O(ε²) error]

    Example:
        # Validate ReLU gradient
        fn relu_forward(x: ExTensor) raises -> ExTensor:
            return relu(x)

        var x = ExTensor(DynamicVector[Int](10), DType.float32)
        var numerical_grad = compute_numerical_gradient(relu_forward, x)
        var analytical_grad = relu_backward(ones_like(x), x)
        assert_gradients_close(analytical_grad, numerical_grad, rtol=1e-4)
    """
    # Create gradient tensor (same shape as input)
    var grad = zeros_like(x)

    # Compute gradient for each element using finite differences
    for i in range(x._numel):
        # Save original value
        var original_val = x._get_float64(i)

        # Compute f(x + ε)
        x._set_float64(i, original_val + epsilon)
        var f_plus = forward_fn(x)

        # Compute f(x - ε)
        x._set_float64(i, original_val - epsilon)
        var f_minus = forward_fn(x)

        # Restore original value
        x._set_float64(i, original_val)

        # Central difference: (f(x+ε) - f(x-ε)) / 2ε
        # Handle both scalar and vector outputs
        var grad_val: Float64
        if f_plus._numel == 1:
            # Scalar output: gradient is simply the finite difference
            grad_val = (f_plus._get_float64(0) - f_minus._get_float64(0)) / (2.0 * epsilon)
        else:
            # Vector output: sum of gradients (for loss functions)
            # This assumes we're computing gradient of sum(output) w.r.t input
            grad_val = 0.0
            for j in range(f_plus._numel):
                grad_val += (f_plus._get_float64(j) - f_minus._get_float64(j)) / (2.0 * epsilon)

        grad._set_float64(i, grad_val)

    return grad


fn assert_gradients_close(
    analytical: ExTensor,
    numerical: ExTensor,
    rtol: Float64 = 1e-4,
    atol: Float64 = 1e-7,
    message: String = "Gradients do not match"
) raises:
    """Assert analytical and numerical gradients are close.

    Uses relative and absolute tolerance to handle both small and large gradients:
        |analytical - numerical| <= atol + rtol * |numerical|

    Args:
        analytical: Gradient computed by backward pass
        numerical: Gradient computed by finite differences
        rtol: Relative tolerance (default: 1e-4, suitable for float32)
        atol: Absolute tolerance (default: 1e-7)
        message: Error message prefix

    Raises:
        Error: If gradients differ beyond tolerance

    Tolerance Guidelines:
        Float16: rtol=1e-2, atol=1e-4
        Float32: rtol=1e-4, atol=1e-7
        Float64: rtol=1e-7, atol=1e-10

    Example:
        var analytical = relu_backward(grad_output, x)
        var numerical = compute_numerical_gradient(relu, x)
        assert_gradients_close(analytical, numerical)  # Uses default tolerances
    """
    # Check shapes match
    if analytical._numel != numerical._numel:
        raise Error(message + ": shape mismatch - analytical has " +
                   str(analytical._numel) + " elements, numerical has " +
                   str(numerical._numel) + " elements")

    # Check dtypes match
    if analytical._dtype != numerical._dtype:
        raise Error(message + ": dtype mismatch")

    # Compare element-wise
    var max_diff: Float64 = 0.0
    var max_rel_diff: Float64 = 0.0
    var worst_idx: Int = -1

    for i in range(analytical._numel):
        var a = analytical._get_float64(i)
        var n = numerical._get_float64(i)
        var abs_diff = math_abs(a - n)
        var tolerance = atol + rtol * math_abs(n)

        if abs_diff > max_diff:
            max_diff = abs_diff
            worst_idx = i

        # Compute relative difference for reporting
        if math_abs(n) > 1e-10:
            var rel_diff = abs_diff / math_abs(n)
            if rel_diff > max_rel_diff:
                max_rel_diff = rel_diff

        if abs_diff > tolerance:
            raise Error(message + ":\n" +
                       "  Index: " + str(i) + "\n" +
                       "  Analytical: " + str(a) + "\n" +
                       "  Numerical: " + str(n) + "\n" +
                       "  Absolute difference: " + str(abs_diff) + "\n" +
                       "  Tolerance: " + str(tolerance) + "\n" +
                       "  (atol=" + str(atol) + ", rtol=" + str(rtol) + ")")


fn check_gradient(
    forward_fn: fn(ExTensor) raises -> ExTensor,
    backward_fn: fn(ExTensor, ExTensor) raises -> ExTensor,
    x: ExTensor,
    grad_output: ExTensor,
    epsilon: Float64 = 1e-5,
    rtol: Float64 = 1e-4,
    atol: Float64 = 1e-7
) raises:
    """Comprehensive gradient check helper.

    Combines numerical gradient computation and comparison in one call.
    This is the recommended way to validate backward passes in tests.

    Args:
        forward_fn: Forward pass function
        backward_fn: Backward pass function (takes grad_output and x)
        x: Input tensor
        grad_output: Gradient from upstream (typically ones_like(output))
        epsilon: Perturbation size for finite differences
        rtol: Relative tolerance
        atol: Absolute tolerance

    Raises:
        Error: If gradients don't match within tolerance

    Example:
        fn test_relu_gradient() raises:
            var x = ExTensor(DynamicVector[Int](10), DType.float32)
            # ... initialize x with test values ...

            fn forward(inp: ExTensor) raises -> ExTensor:
                return relu(inp)

            var grad_out = ones_like(relu(x))
            check_gradient(forward, relu_backward, x, grad_out)
    """
    # Compute analytical gradient
    var analytical = backward_fn(grad_output, x)

    # Compute numerical gradient
    # Note: For functions with grad_output, we need to wrap the forward function
    # to include the grad_output multiplication
    fn scaled_forward(inp: ExTensor) raises -> ExTensor:
        var out = forward_fn(inp)
        # For simplicity, assume scalar output or sum of outputs
        var result: Float64 = 0.0
        for i in range(out._numel):
            result += out._get_float64(i) * grad_output._get_float64(i)
        var scalar_out = ExTensor(DynamicVector[Int](1), out._dtype)
        scalar_out._set_float64(0, result)
        return scalar_out

    var numerical = compute_numerical_gradient(scaled_forward, x, epsilon)

    # Compare
    assert_gradients_close(analytical, numerical, rtol, atol,
                          "Gradient check failed for " + str(x._dtype))
