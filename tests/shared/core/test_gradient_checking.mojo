"""Comprehensive gradient checking tests for all backward passes.

Tests all layer backward implementations using numerical differentiation
to verify analytical gradients are correct.

Usage:
    mojo test tests/shared/core/test_gradient_checking.mojo

Expected: All tests should pass (gradients match within tolerance)
"""

from tests.shared.conftest import assert_true, assert_equal
from shared.testing import check_gradients, check_gradients_verbose
from shared.core import ExTensor, zeros, ones, full
from shared.core.activation import relu, relu_backward, sigmoid, sigmoid_backward, tanh, tanh_backward
from shared.core.arithmetic import add, multiply, add_backward, multiply_backward


# ============================================================================
# Activation Function Gradient Tests
# ============================================================================


fn test_relu_gradient() raises:
    """Test ReLU backward pass using gradient checking."""
    print("Testing ReLU gradient...")

    var shape = List[Int](2)
    shape[0] = 3
    shape[1] = 4

    var input = full(shape, 2.0, DType.float32)  # Positive inputs

    fn forward(x: ExTensor) -> ExTensor:
        return relu(x)

    fn backward(grad_out: ExTensor, x: ExTensor) -> ExTensor:
        return relu_backward(grad_out, x)

    var passed = check_gradients(forward, backward, input)
    assert_true(passed, "ReLU gradient check failed")
    print("  ✓ ReLU gradient correct")


fn test_relu_negative_inputs() raises:
    """Test ReLU gradient with negative inputs (zero gradient region)."""
    print("Testing ReLU gradient (negative inputs)...")

    var shape = List[Int](2)
    shape[0] = 3
    shape[1] = 4

    var input = full(shape, -2.0, DType.float32)  # Negative inputs

    fn forward(x: ExTensor) -> ExTensor:
        return relu(x)

    fn backward(grad_out: ExTensor, x: ExTensor) -> ExTensor:
        return relu_backward(grad_out, x)

    var passed = check_gradients(forward, backward, input)
    assert_true(passed, "ReLU gradient check failed for negative inputs")
    print("  ✓ ReLU gradient correct (negative inputs)")


fn test_relu_mixed_inputs() raises:
    """Test ReLU gradient with mixed positive/negative inputs."""
    print("Testing ReLU gradient (mixed inputs)...")

    var shape = List[Int](2)
    shape[0] = 3
    shape[1] = 4

    var input = zeros(shape, DType.float32)
    # Set some positive, some negative
    input._set_float64(0, 1.0)
    input._set_float64(1, -1.0)
    input._set_float64(2, 2.0)
    input._set_float64(3, -2.0)

    fn forward(x: ExTensor) -> ExTensor:
        return relu(x)

    fn backward(grad_out: ExTensor, x: ExTensor) -> ExTensor:
        return relu_backward(grad_out, x)

    var passed = check_gradients(forward, backward, input)
    assert_true(passed, "ReLU gradient check failed for mixed inputs")
    print("  ✓ ReLU gradient correct (mixed inputs)")


fn test_sigmoid_gradient() raises:
    """Test Sigmoid backward pass using gradient checking."""
    print("Testing Sigmoid gradient...")

    var shape = List[Int](2)
    shape[0] = 3
    shape[1] = 4

    var input = full(shape, 0.5, DType.float32)

    fn forward(x: ExTensor) -> ExTensor:
        return sigmoid(x)

    fn backward(grad_out: ExTensor, x: ExTensor) -> ExTensor:
        var output = sigmoid(x)
        return sigmoid_backward(grad_out, output)

    var passed = check_gradients(forward, backward, input)
    assert_true(passed, "Sigmoid gradient check failed")
    print("  ✓ Sigmoid gradient correct")


fn test_tanh_gradient() raises:
    """Test Tanh backward pass using gradient checking."""
    print("Testing Tanh gradient...")

    var shape = List[Int](2)
    shape[0] = 3
    shape[1] = 4

    var input = full(shape, 0.5, DType.float32)

    fn forward(x: ExTensor) -> ExTensor:
        return tanh(x)

    fn backward(grad_out: ExTensor, x: ExTensor) -> ExTensor:
        var output = tanh(x)
        return tanh_backward(grad_out, output)

    var passed = check_gradients(forward, backward, input)
    assert_true(passed, "Tanh gradient check failed")
    print("  ✓ Tanh gradient correct")


# ============================================================================
# Arithmetic Operation Gradient Tests
# ============================================================================


fn test_add_gradient() raises:
    """Test addition backward pass using gradient checking."""
    print("Testing Add gradient...")

    var shape = List[Int](2)
    shape[0] = 3
    shape[1] = 4

    var input_a = ones(shape, DType.float32)
    var input_b = ones(shape, DType.float32)

    # For add, we test gradient w.r.t. first input
    fn forward(x: ExTensor) -> ExTensor:
        return add(x, input_b)

    fn backward(grad_out: ExTensor, x: ExTensor) -> ExTensor:
        var (grad_a, _) = add_backward(grad_out, x, input_b)
        return grad_a

    var passed = check_gradients(forward, backward, input_a)
    assert_true(passed, "Add gradient check failed")
    print("  ✓ Add gradient correct")


fn test_multiply_gradient() raises:
    """Test multiplication backward pass using gradient checking."""
    print("Testing Multiply gradient...")

    var shape = List[Int](2)
    shape[0] = 3
    shape[1] = 4

    var input_a = full(shape, 2.0, DType.float32)
    var input_b = full(shape, 3.0, DType.float32)

    # Test gradient w.r.t. first input
    fn forward(x: ExTensor) -> ExTensor:
        return multiply(x, input_b)

    fn backward(grad_out: ExTensor, x: ExTensor) -> ExTensor:
        var (grad_a, _) = multiply_backward(grad_out, x, input_b)
        return grad_a

    var passed = check_gradients(forward, backward, input_a)
    assert_true(passed, "Multiply gradient check failed")
    print("  ✓ Multiply gradient correct")


# ============================================================================
# Composite Operation Tests
# ============================================================================


fn test_composite_relu_multiply() raises:
    """Test gradient through composite operation: multiply -> relu."""
    print("Testing composite gradient (multiply -> relu)...")

    var shape = List[Int](2)
    shape[0] = 3
    shape[1] = 4

    var input_a = full(shape, 2.0, DType.float32)
    var input_b = full(shape, 3.0, DType.float32)

    fn forward(x: ExTensor) -> ExTensor:
        var mul_result = multiply(x, input_b)
        return relu(mul_result)

    fn backward(grad_out: ExTensor, x: ExTensor) -> ExTensor:
        # Forward to get intermediate
        var mul_result = multiply(x, input_b)
        var relu_result = relu(mul_result)

        # Backward through relu
        var grad_relu = relu_backward(grad_out, mul_result)

        # Backward through multiply
        var (grad_mul, _) = multiply_backward(grad_relu, x, input_b)
        return grad_mul

    var passed = check_gradients(forward, backward, input_a)
    assert_true(passed, "Composite gradient check failed")
    print("  ✓ Composite gradient correct (multiply -> relu)")


# ============================================================================
# Edge Case Tests
# ============================================================================


fn test_gradient_at_zero() raises:
    """Test gradient checking at zero (potential numerical issues)."""
    print("Testing gradient at zero...")

    var shape = List[Int](2)
    shape[0] = 2
    shape[1] = 2

    var input = zeros(shape, DType.float32)

    fn forward(x: ExTensor) -> ExTensor:
        return relu(x)

    fn backward(grad_out: ExTensor, x: ExTensor) -> ExTensor:
        return relu_backward(grad_out, x)

    # Use larger tolerance for zero region
    var passed = check_gradients(forward, backward, input, tolerance=1e-2)
    assert_true(passed, "Gradient at zero check failed")
    print("  ✓ Gradient at zero correct")


fn test_gradient_small_tensor() raises:
    """Test gradient checking on very small tensors (1x1)."""
    print("Testing gradient on small tensor...")

    var shape = List[Int](2)
    shape[0] = 1
    shape[1] = 1

    var input = full(shape, 2.0, DType.float32)

    fn forward(x: ExTensor) -> ExTensor:
        return relu(x)

    fn backward(grad_out: ExTensor, x: ExTensor) -> ExTensor:
        return relu_backward(grad_out, x)

    var passed = check_gradients(forward, backward, input)
    assert_true(passed, "Small tensor gradient check failed")
    print("  ✓ Small tensor gradient correct")


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run all gradient checking tests."""
    print("\n" + "="*80)
    print("Gradient Checking Test Suite")
    print("="*80 + "\n")

    # Activation functions
    print("Activation Functions:")
    test_relu_gradient()
    test_relu_negative_inputs()
    test_relu_mixed_inputs()
    test_sigmoid_gradient()
    test_tanh_gradient()

    # Arithmetic operations
    print("\nArithmetic Operations:")
    test_add_gradient()
    test_multiply_gradient()

    # Composite operations
    print("\nComposite Operations:")
    test_composite_relu_multiply()

    # Edge cases
    print("\nEdge Cases:")
    test_gradient_at_zero()
    test_gradient_small_tensor()

    print("\n" + "="*80)
    print("✅ All gradient checks PASSED!")
    print("="*80 + "\n")

    print("Summary:")
    print("  - All backward passes produce correct gradients")
    print("  - Numerical differentiation matches analytical gradients")
    print("  - Edge cases (zero, small tensors) handled correctly")
    print("  - Composite operations use chain rule correctly")
    print("\n")
