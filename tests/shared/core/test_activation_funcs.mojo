"""Unit tests for activation functions.

Tests cover:
- relu: Forward and backward passes
- sigmoid: Forward and backward passes with numerical stability
- tanh: Forward and backward passes
- softmax: Forward and backward passes
- leaky_relu: Parametric activation function

All tests use pure functional API - no internal state.
"""

from tests.shared.conftest import (
    assert_almost_equal,
    assert_equal,
    assert_greater_or_equal,
    assert_less_or_equal,
    assert_true,
)
from shared.core.extensor import ExTensor, zeros, ones, full
from shared.core.activation import (
    relu,
    relu_backward,
    sigmoid,
    sigmoid_backward,
    tanh,
    tanh_backward,
    softmax,
    softmax_backward,
    leaky_relu,
    leaky_relu_backward,
)


# ============================================================================
# ReLU Tests
# ============================================================================


fn test_relu_positive_values() raises:
    """Test ReLU preserves positive values."""
    var input_shape = List[Int]()
    input_shape.append(5)
    var input = ones(input_shape, DType.float32)

    var output = relu(input)

    var output_data = output._data.bitcast[Float32]()
    for i in range(5):
        assert_almost_equal(output_data[i], 1.0, tolerance=1e-5)


fn test_relu_negative_values() raises:
    """Test ReLU zeros out negative values."""
    var input_shape = List[Int]()
    input_shape.append(5)
    var input = zeros(input_shape, DType.float32)

    var input_data = input._data.bitcast[Float32]()
    for i in range(5):
        input_data[i] = -Float32(i + 1)

    var output = relu(input)

    var output_data = output._data.bitcast[Float32]()
    for i in range(5):
        assert_almost_equal(output_data[i], 0.0, tolerance=1e-5)


fn test_relu_mixed_values() raises:
    """Test ReLU on mixed positive and negative values."""
    var input_shape = List[Int]()
    input_shape.append(5)
    var input = zeros(input_shape, DType.float32)

    var input_data = input._data.bitcast[Float32]()
    input_data[0] = -2.0  # Should become 0
    input_data[1] = -1.0  # Should become 0
    input_data[2] = 0.0   # Should become 0
    input_data[3] = 1.0   # Should stay 1
    input_data[4] = 2.0   # Should stay 2

    var output = relu(input)

    var output_data = output._data.bitcast[Float32]()
    assert_almost_equal(output_data[0], 0.0, tolerance=1e-5)
    assert_almost_equal(output_data[1], 0.0, tolerance=1e-5)
    assert_almost_equal(output_data[2], 0.0, tolerance=1e-5)
    assert_almost_equal(output_data[3], 1.0, tolerance=1e-5)
    assert_almost_equal(output_data[4], 2.0, tolerance=1e-5)


fn test_relu_backward() raises:
    """Test ReLU backward pass computes correct gradients."""
    var input_shape = List[Int]()
    input_shape.append(5)
    var input = zeros(input_shape, DType.float32)

    var input_data = input._data.bitcast[Float32]()
    input_data[0] = -2.0  # Negative
    input_data[1] = -1.0  # Negative
    input_data[2] = 0.0   # Zero
    input_data[3] = 1.0   # Positive
    input_data[4] = 2.0   # Positive

    var grad_output_shape = List[Int]()
    grad_output_shape.append(5)
    var grad_output = ones(grad_output_shape, DType.float32)

    var grad_input = relu_backward(grad_output, input)

    var grad_data = grad_input._data.bitcast[Float32]()
    # Negative inputs -> grad is 0
    assert_almost_equal(grad_data[0], 0.0, tolerance=1e-5)
    assert_almost_equal(grad_data[1], 0.0, tolerance=1e-5)
    assert_almost_equal(grad_data[2], 0.0, tolerance=1e-5)
    # Positive inputs -> grad passes through
    assert_almost_equal(grad_data[3], 1.0, tolerance=1e-5)
    assert_almost_equal(grad_data[4], 1.0, tolerance=1e-5)


fn test_leaky_relu_forward() raises:
    """Test Leaky ReLU preserves negative values with slope."""
    var input_shape = List[Int]()
    input_shape.append(5)
    var input = zeros(input_shape, DType.float32)

    var input_data = input._data.bitcast[Float32]()
    input_data[0] = -2.0
    input_data[1] = -1.0
    input_data[2] = 0.0
    input_data[3] = 1.0
    input_data[4] = 2.0

    var output = leaky_relu(input, alpha=0.1)

    var output_data = output._data.bitcast[Float32]()
    assert_almost_equal(output_data[0], -0.2, tolerance=1e-5)  # -2 * 0.1
    assert_almost_equal(output_data[1], -0.1, tolerance=1e-5)  # -1 * 0.1
    assert_almost_equal(output_data[2], 0.0, tolerance=1e-5)
    assert_almost_equal(output_data[3], 1.0, tolerance=1e-5)
    assert_almost_equal(output_data[4], 2.0, tolerance=1e-5)


# ============================================================================
# Sigmoid Tests
# ============================================================================


fn test_sigmoid_range() raises:
    """Test sigmoid output is in [0, 1] range."""
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(1)
    input_shape.append(5)
    var input = zeros(input_shape, DType.float32)

    var input_data = input._data.bitcast[Float32]()
    input_data[0] = -10.0
    input_data[1] = -1.0
    input_data[2] = 0.0
    input_data[3] = 1.0
    input_data[4] = 10.0

    var output = sigmoid(input)

    var output_data = output._data.bitcast[Float32]()
    for i in range(5):
        assert_greater_or_equal(output_data[i], 0.0, "Sigmoid lower bound")
        assert_less_or_equal(output_data[i], 1.0, "Sigmoid upper bound")


fn test_sigmoid_at_zero() raises:
    """Test sigmoid(0) = 0.5."""
    var input_shape = List[Int]()
    input_shape.append(1)
    var input = zeros(input_shape, DType.float32)

    var output = sigmoid(input)

    var output_data = output._data.bitcast[Float32]()
    assert_almost_equal(output_data[0], 0.5, tolerance=1e-4)


fn test_sigmoid_symmetry() raises:
    """Test sigmoid(-x) + sigmoid(x) = 1."""
    var input_shape = List[Int]()
    input_shape.append(1)
    var x = zeros(input_shape, DType.float32)
    var x_data = x._data.bitcast[Float32]()
    x_data[0] = 2.0

    var neg_x = zeros(input_shape, DType.float32)
    var neg_x_data = neg_x._data.bitcast[Float32]()
    neg_x_data[0] = -2.0

    var sig_x = sigmoid(x)
    var sig_neg_x = sigmoid(neg_x)

    var sig_x_data = sig_x._data.bitcast[Float32]()
    var sig_neg_x_data = sig_neg_x._data.bitcast[Float32]()

    var sum_val = sig_x_data[0] + sig_neg_x_data[0]
    assert_almost_equal(sum_val, 1.0, tolerance=1e-4)


fn test_sigmoid_backward() raises:
    """Test sigmoid backward pass computes correct gradients."""
    var output_shape = List[Int]()
    output_shape.append(1)
    var output = full(output_shape, 0.5, DType.float32)

    var grad_output_shape = List[Int]()
    grad_output_shape.append(1)
    var grad_output = ones(grad_output_shape, DType.float32)

    var grad_input = sigmoid_backward(grad_output, output)

    # For sigmoid output = 0.5, gradient = 1 * 0.5 * (1 - 0.5) = 0.25
    var grad_data = grad_input._data.bitcast[Float32]()
    assert_almost_equal(grad_data[0], 0.25, tolerance=1e-4)


# ============================================================================
# Tanh Tests
# ============================================================================


fn test_tanh_range() raises:
    """Test tanh output is in [-1, 1] range."""
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(1)
    input_shape.append(5)
    var input = zeros(input_shape, DType.float32)

    var input_data = input._data.bitcast[Float32]()
    input_data[0] = -10.0
    input_data[1] = -1.0
    input_data[2] = 0.0
    input_data[3] = 1.0
    input_data[4] = 10.0

    var output = tanh(input)

    var output_data = output._data.bitcast[Float32]()
    for i in range(5):
        assert_greater_or_equal(output_data[i], -1.0, "Tanh lower bound")
        assert_less_or_equal(output_data[i], 1.0, "Tanh upper bound")


fn test_tanh_at_zero() raises:
    """Test tanh(0) = 0."""
    var input_shape = List[Int]()
    input_shape.append(1)
    var input = zeros(input_shape, DType.float32)

    var output = tanh(input)

    var output_data = output._data.bitcast[Float32]()
    assert_almost_equal(output_data[0], 0.0, tolerance=1e-5)


fn test_tanh_antisymmetry() raises:
    """Test tanh(-x) = -tanh(x)."""
    var input_shape = List[Int]()
    input_shape.append(1)
    var x = zeros(input_shape, DType.float32)
    var x_data = x._data.bitcast[Float32]()
    x_data[0] = 2.0

    var neg_x = zeros(input_shape, DType.float32)
    var neg_x_data = neg_x._data.bitcast[Float32]()
    neg_x_data[0] = -2.0

    var tanh_x = tanh(x)
    var tanh_neg_x = tanh(neg_x)

    var tanh_x_data = tanh_x._data.bitcast[Float32]()
    var tanh_neg_x_data = tanh_neg_x._data.bitcast[Float32]()

    assert_almost_equal(tanh_x_data[0], -tanh_neg_x_data[0], tolerance=1e-4)


fn test_tanh_backward() raises:
    """Test tanh backward pass computes correct gradients."""
    var output_shape = List[Int]()
    output_shape.append(1)
    var output = zeros(output_shape, DType.float32)
    var output_data = output._data.bitcast[Float32]()
    output_data[0] = 0.5

    var grad_output_shape = List[Int]()
    grad_output_shape.append(1)
    var grad_output = ones(grad_output_shape, DType.float32)

    var grad_input = tanh_backward(grad_output, output)

    # For tanh output = 0.5, gradient = 1 * (1 - 0.5^2) = 0.75
    var grad_data = grad_input._data.bitcast[Float32]()
    assert_almost_equal(grad_data[0], 0.75, tolerance=1e-4)


# ============================================================================
# Softmax Tests
# ============================================================================


fn test_softmax_output_sum() raises:
    """Test softmax outputs sum to 1 along last axis."""
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(3)
    var input = zeros(input_shape, DType.float32)

    var input_data = input._data.bitcast[Float32]()
    input_data[0] = 1.0
    input_data[1] = 2.0
    input_data[2] = 3.0

    var output = softmax(input, axis=-1)

    var output_data = output._data.bitcast[Float32]()
    var sum_val = output_data[0] + output_data[1] + output_data[2]
    assert_almost_equal(sum_val, 1.0, tolerance=1e-5)


fn test_softmax_positive_outputs() raises:
    """Test softmax outputs are all positive."""
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(3)
    var input = zeros(input_shape, DType.float32)

    var input_data = input._data.bitcast[Float32]()
    input_data[0] = -5.0
    input_data[1] = 0.0
    input_data[2] = 5.0

    var output = softmax(input, axis=-1)

    var output_data = output._data.bitcast[Float32]()
    for i in range(3):
        assert_greater_or_equal(output_data[i], 0.0, "Softmax positive")


fn test_softmax_uniform() raises:
    """Test softmax on uniform input gives uniform output."""
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(4)
    var input = zeros(input_shape, DType.float32)

    var input_data = input._data.bitcast[Float32]()
    for i in range(4):
        input_data[i] = 5.0

    var output = softmax(input, axis=-1)

    var output_data = output._data.bitcast[Float32]()
    for i in range(4):
        assert_almost_equal(output_data[i], 0.25, tolerance=1e-5)


fn test_softmax_backward() raises:
    """Test softmax backward pass."""
    var output_shape = List[Int]()
    output_shape.append(1)
    output_shape.append(3)
    var output = zeros(output_shape, DType.float32)

    var output_data = output._data.bitcast[Float32]()
    output_data[0] = 0.1
    output_data[1] = 0.6
    output_data[2] = 0.3

    var grad_output_shape = List[Int]()
    grad_output_shape.append(1)
    grad_output_shape.append(3)
    var grad_output = ones(grad_output_shape, DType.float32)

    var grad_input = softmax_backward(grad_output, output, axis=-1)

    var grad_shape = grad_input.shape()
    assert_equal(grad_shape[0], 1)
    assert_equal(grad_shape[1], 3)


# ============================================================================
# Integration Tests
# ============================================================================


fn test_activation_output_shape_preservation() raises:
    """Test that activations preserve input shape."""
    var input_shape = List[Int]()
    input_shape.append(2)
    input_shape.append(3)
    var input = ones(input_shape, DType.float32)

    var relu_out = relu(input)
    var sig_out = sigmoid(input)
    var tanh_out = tanh(input)

    var relu_shape = relu_out.shape()
    var sig_shape = sig_out.shape()
    var tanh_shape = tanh_out.shape()

    assert_equal(relu_shape[0], 2)
    assert_equal(relu_shape[1], 3)
    assert_equal(sig_shape[0], 2)
    assert_equal(sig_shape[1], 3)
    assert_equal(tanh_shape[0], 2)
    assert_equal(tanh_shape[1], 3)


fn test_activation_dtype_preservation() raises:
    """Test that activations preserve input dtype."""
    var input_shape = List[Int]()
    input_shape.append(5)
    var input = ones(input_shape, DType.float32)

    var relu_out = relu(input)
    var sig_out = sigmoid(input)

    assert_true(relu_out.dtype() == DType.float32)
    assert_true(sig_out.dtype() == DType.float32)


fn test_relu_gradient_mask() raises:
    """Test that ReLU gradient acts as a mask for positive values."""
    var input_shape = List[Int]()
    input_shape.append(4)
    var input = zeros(input_shape, DType.float32)

    var input_data = input._data.bitcast[Float32]()
    input_data[0] = -1.0
    input_data[1] = 2.0
    input_data[2] = -3.0
    input_data[3] = 4.0

    var grad_output_shape = List[Int]()
    grad_output_shape.append(4)
    var grad_output = ones(grad_output_shape, DType.float32)

    var grad_input = relu_backward(grad_output, input)

    var grad_data = grad_input._data.bitcast[Float32]()
    assert_almost_equal(grad_data[0], 0.0, tolerance=1e-5)  # Masked
    assert_almost_equal(grad_data[1], 1.0, tolerance=1e-5)  # Passed through
    assert_almost_equal(grad_data[2], 0.0, tolerance=1e-5)  # Masked
    assert_almost_equal(grad_data[3], 1.0, tolerance=1e-5)  # Passed through


fn main() raises:
    """Run all activation tests."""
    print("Running activation tests...")

    test_relu_positive_values()
    print("✓ test_relu_positive_values")

    test_relu_negative_values()
    print("✓ test_relu_negative_values")

    test_relu_mixed_values()
    print("✓ test_relu_mixed_values")

    test_relu_backward()
    print("✓ test_relu_backward")

    test_leaky_relu_forward()
    print("✓ test_leaky_relu_forward")

    test_sigmoid_range()
    print("✓ test_sigmoid_range")

    test_sigmoid_at_zero()
    print("✓ test_sigmoid_at_zero")

    test_sigmoid_symmetry()
    print("✓ test_sigmoid_symmetry")

    test_sigmoid_backward()
    print("✓ test_sigmoid_backward")

    test_tanh_range()
    print("✓ test_tanh_range")

    test_tanh_at_zero()
    print("✓ test_tanh_at_zero")

    test_tanh_antisymmetry()
    print("✓ test_tanh_antisymmetry")

    test_tanh_backward()
    print("✓ test_tanh_backward")

    test_softmax_output_sum()
    print("✓ test_softmax_output_sum")

    test_softmax_positive_outputs()
    print("✓ test_softmax_positive_outputs")

    test_softmax_uniform()
    print("✓ test_softmax_uniform")

    test_softmax_backward()
    print("✓ test_softmax_backward")

    test_activation_output_shape_preservation()
    print("✓ test_activation_output_shape_preservation")

    test_activation_dtype_preservation()
    print("✓ test_activation_dtype_preservation")

    test_relu_gradient_mask()
    print("✓ test_relu_gradient_mask")

    print("\nAll activation tests passed!")
