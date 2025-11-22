"""Gradient checking utilities for validating backward passes.

Implements numerical gradient computation using finite differences to
verify analytical gradients computed by backward passes.

Theory:
    Numerical gradient: f'(x) ≈ [f(x + ε) - f(x - ε)] / (2ε)
    Analytical gradient: Computed by backward pass
    If correct: |numerical - analytical| < tolerance

Benefits:
- Catches gradient bugs early in development
- Validates complex backward pass implementations
- Essential for debugging custom layers
- Provides confidence in optimization

Usage:
    from shared.testing.gradient_checker import check_gradients

    fn forward(x: ExTensor) -> ExTensor:
        return relu(x)

    fn backward(grad_out: ExTensor, x: ExTensor) -> ExTensor:
        return relu_backward(grad_out, x)

    var input = randn([3, 4], DType.float32)
    var passed = check_gradients(forward, backward, input)
    assert_true(passed, "ReLU gradient check failed")

Performance:
    - O(n) function evaluations (n = input size)
    - Use sparingly (expensive for large tensors)
    - Typically run only in test suite, not production

References:
    - CS231n: http://cs231n.github.io/neural-networks-3/#gradcheck
    - Goodfellow et al., Deep Learning, Chapter 4.3
"""

from shared.core import ExTensor, zeros_like
from math import abs as math_abs


fn check_gradients(
    forward_fn: fn(ExTensor) -> ExTensor,
    backward_fn: fn(ExTensor, ExTensor) -> ExTensor,
    input: ExTensor,
    epsilon: Float64 = 1e-5,
    tolerance: Float64 = 1e-3
) raises -> Bool:
    """Verify gradients using finite differences.

    Compares analytical gradients from backward_fn against numerical
    gradients computed using finite differences. Returns True if all
    gradients match within tolerance.

    Args:
        forward_fn: Forward pass function: input -> output
        backward_fn: Backward pass function: (grad_output, input) -> grad_input
        input: Input tensor for testing
        epsilon: Step size for finite differences (default: 1e-5)
        tolerance: Maximum allowed difference (default: 1e-3)

    Returns:
        True if gradients are correct, False otherwise

    Raises:
        Error: If forward/backward functions fail

    Algorithm:
        1. Run forward pass to get output
        2. Compute analytical gradient using backward pass
        3. For each input element:
            a. Perturb by +ε, compute f(x+ε)
            b. Perturb by -ε, compute f(x-ε)
            c. Numerical gradient = [f(x+ε) - f(x-ε)] / (2ε)
            d. Compare with analytical gradient
        4. Return True if max difference < tolerance

    Example:
        fn my_forward(x: ExTensor) -> ExTensor:
            return x * x  # f(x) = x²

        fn my_backward(grad_out: ExTensor, x: ExTensor) -> ExTensor:
            return multiply(grad_out, multiply(x, full_like(x, 2.0)))  # f'(x) = 2x

        var x = full([3, 4], 2.0, DType.float32)
        var passed = check_gradients(my_forward, my_backward, x)
        # Should return True (2x is correct derivative of x²)

    Notes:
        - Expensive: O(n) forward passes where n = input.numel()
        - Use small tensors for testing (e.g., 3x4 instead of 1024x1024)
        - Typical tolerance: 1e-3 to 1e-5
        - Lower epsilon = more accurate but numerically unstable
        - Higher epsilon = more stable but less accurate
    """
    # Step 1: Compute analytical gradient
    var output = forward_fn(input)
    var grad_output = zeros_like(output)

    # Set grad_output to ones (∂L/∂output = 1 for all elements)
    for i in range(output.numel()):
        grad_output._set_float64(i, 1.0)

    var analytical_grad = backward_fn(grad_output, input)

    # Step 2: Compute numerical gradient using finite differences
    var numerical_grad = zeros_like(input)
    var input_copy_plus = input.copy()
    var input_copy_minus = input.copy()

    for i in range(input.numel()):
        # Save original value
        var original_val = input._get_float64(i)

        # f(x + ε)
        input_copy_plus._set_float64(i, original_val + epsilon)
        var output_plus = forward_fn(input_copy_plus)
        var sum_plus = output_plus.sum()

        # f(x - ε)
        input_copy_minus._set_float64(i, original_val - epsilon)
        var output_minus = forward_fn(input_copy_minus)
        var sum_minus = output_minus.sum()

        # Numerical gradient: [f(x+ε) - f(x-ε)] / (2ε)
        var numerical = (sum_plus - sum_minus) / (2.0 * epsilon)
        numerical_grad._set_float64(i, numerical)

        # Restore original value for next iteration
        input_copy_plus._set_float64(i, original_val)
        input_copy_minus._set_float64(i, original_val)

    # Step 3: Compare analytical vs numerical gradients
    var max_diff = 0.0
    var max_diff_idx = 0

    for i in range(input.numel()):
        var analytical = analytical_grad._get_float64(i)
        var numerical = numerical_grad._get_float64(i)
        var diff = math_abs(analytical - numerical)

        if diff > max_diff:
            max_diff = diff
            max_diff_idx = i

    # Print diagnostics if gradients don't match
    if max_diff >= tolerance:
        print("Gradient check FAILED:")
        print("  Max difference:", max_diff)
        print("  At index:", max_diff_idx)
        print("  Analytical:", analytical_grad._get_float64(max_diff_idx))
        print("  Numerical:", numerical_grad._get_float64(max_diff_idx))
        print("  Tolerance:", tolerance)
        return False

    return True


fn check_gradients_verbose(
    forward_fn: fn(ExTensor) -> ExTensor,
    backward_fn: fn(ExTensor, ExTensor) -> ExTensor,
    input: ExTensor,
    epsilon: Float64 = 1e-5,
    tolerance: Float64 = 1e-3,
    print_all: Bool = False
) raises -> Bool:
    """Gradient checking with detailed output.

    Same as check_gradients but prints all differences, not just maximum.
    Useful for debugging specific gradient issues.

    Args:
        forward_fn: Forward pass function
        backward_fn: Backward pass function
        input: Input tensor
        epsilon: Finite difference step size
        tolerance: Maximum allowed difference
        print_all: If True, print all gradients (even passing ones)

    Returns:
        True if gradients correct, False otherwise

    Example:
        var passed = check_gradients_verbose(
            forward, backward, input,
            print_all=True  # Print all gradient comparisons
        )
    """
    # Run standard gradient check
    var passed = check_gradients(forward_fn, backward_fn, input, epsilon, tolerance)

    if print_all or not passed:
        print("\n=== Gradient Check Details ===")
        print("Input shape:", input.shape())
        print("Epsilon:", epsilon)
        print("Tolerance:", tolerance)

        # Recompute for printing
        var output = forward_fn(input)
        var grad_output = zeros_like(output)
        for i in range(output.numel()):
            grad_output._set_float64(i, 1.0)
        var analytical_grad = backward_fn(grad_output, input)

        var numerical_grad = zeros_like(input)
        var input_copy_plus = input.copy()
        var input_copy_minus = input.copy()

        for i in range(input.numel()):
            var original_val = input._get_float64(i)

            input_copy_plus._set_float64(i, original_val + epsilon)
            var output_plus = forward_fn(input_copy_plus)
            var sum_plus = output_plus.sum()

            input_copy_minus._set_float64(i, original_val - epsilon)
            var output_minus = forward_fn(input_copy_minus)
            var sum_minus = output_minus.sum()

            var numerical = (sum_plus - sum_minus) / (2.0 * epsilon)
            numerical_grad._set_float64(i, numerical)

            input_copy_plus._set_float64(i, original_val)
            input_copy_minus._set_float64(i, original_val)

        print("\nGradient Comparisons:")
        print("Index | Analytical | Numerical | Diff | Status")
        print("-" * 60)

        for i in range(min(input.numel(), 20)):  # Print first 20
            var analytical = analytical_grad._get_float64(i)
            var numerical = numerical_grad._get_float64(i)
            var diff = math_abs(analytical - numerical)
            var status = "PASS" if diff < tolerance else "FAIL"

            if print_all or diff >= tolerance:
                print(
                    i, " | ",
                    analytical, " | ",
                    numerical, " | ",
                    diff, " | ",
                    status
                )

        if input.numel() > 20:
            print(f"... ({input.numel() - 20} more elements)")

        print("=" * 60)

    return passed


fn relative_error(analytical: Float64, numerical: Float64) -> Float64:
    """Compute relative error between analytical and numerical gradients.

    Uses formula: |a - n| / max(|a|, |n|, 1e-8)
    Handles edge cases where gradients are near zero.

    Args:
        analytical: Analytical gradient value
        numerical: Numerical gradient value

    Returns:
        Relative error (typically 0-1, < 0.01 is good)

    Example:
        var err = relative_error(0.5, 0.501)  # Returns ~0.002 (0.2%)
    """
    var numerator = math_abs(analytical - numerical)
    var denominator = max(
        math_abs(analytical),
        max(math_abs(numerical), 1e-8)
    )
    return numerator / denominator
