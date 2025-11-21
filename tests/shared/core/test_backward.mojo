"""Tests for backward passes - gradient checking validation.

This module tests all backward pass implementations using numerical gradient checking
with finite differences. This is the gold standard for validating gradient computation.

Gradient checking formula:
    numerical_grad ≈ (f(x + ε) - f(x - ε)) / (2ε)

All tests use pure functional API.
"""

from tests.shared.conftest import (
    assert_true,
    assert_equal,
    assert_almost_equal,
    assert_shape_equal,
    TestFixtures,
)
from shared.core.extensor import ExTensor, zeros, ones, zeros_like, ones_like
from shared.core.linear import linear, linear_backward
from shared.core.conv import conv2d, conv2d_backward
from shared.core.pooling import maxpool2d, maxpool2d_backward, avgpool2d, avgpool2d_backward
from shared.core.loss import cross_entropy, cross_entropy_backward
from tests.helpers.gradient_checking import check_gradient, compute_numerical_gradient, assert_gradients_close
from collections.vector import DynamicVector


# ============================================================================
# Linear Backward Tests
# ============================================================================


fn test_linear_backward_shapes() raises:
    """Test that linear_backward returns correct gradient shapes."""
    var batch = 2
    var in_features = 10
    var out_features = 5

    # Create tensors
    var input_shape = DynamicVector[Int](2)
    input_shape[0] = batch
    input_shape[1] = in_features
    var x = ones(input_shape, DType.float32)

    var weight_shape = DynamicVector[Int](2)
    weight_shape[0] = out_features
    weight_shape[1] = in_features
    var weights = ones(weight_shape, DType.float32)

    var grad_out_shape = DynamicVector[Int](2)
    grad_out_shape[0] = batch
    grad_out_shape[1] = out_features
    var grad_output = ones(grad_out_shape, DType.float32)

    # Backward pass
    var grads = linear_backward(grad_output, x, weights)

    # Check shapes
    var gi_shape = grads.grad_input.shape()
    assert_equal(gi_shape[0], batch)
    assert_equal(gi_shape[1], in_features)

    var gw_shape = grads.grad_weights.shape()
    assert_equal(gw_shape[0], out_features)
    assert_equal(gw_shape[1], in_features)

    var gb_shape = grads.grad_bias.shape()
    assert_equal(gb_shape[0], out_features)


fn test_linear_backward_numerical() raises:
    """Test linear_backward with numerical gradient checking.

    This is a simple sanity check with known values.
    """
    # Simple case: 1 sample, 2 inputs, 2 outputs
    var input_shape = DynamicVector[Int](2)
    input_shape[0] = 1
    input_shape[1] = 2
    var x = ones(input_shape, DType.float32)
    x._data.bitcast[Float32]()[0] = 1.0
    x._data.bitcast[Float32]()[1] = 2.0

    var weight_shape = DynamicVector[Int](2)
    weight_shape[0] = 2
    weight_shape[1] = 2
    var weights = zeros(weight_shape, DType.float32)
    weights._data.bitcast[Float32]()[0] = 0.5  # w[0,0]
    weights._data.bitcast[Float32]()[1] = 0.3  # w[0,1]
    weights._data.bitcast[Float32]()[2] = 0.2  # w[1,0]
    weights._data.bitcast[Float32]()[3] = 0.4  # w[1,1]

    # Gradient from upstream (pretend loss gradient)
    var grad_out_shape = DynamicVector[Int](2)
    grad_out_shape[0] = 1
    grad_out_shape[1] = 2
    var grad_output = ones(grad_out_shape, DType.float32)

    # Backward pass
    var grads = linear_backward(grad_output, x, weights)

    # Check grad_input shape
    var gi_shape = grads.grad_input.shape()
    assert_equal(gi_shape[0], 1)
    assert_equal(gi_shape[1], 2)

    # grad_input = grad_output @ weights
    # grad_output = [1, 1], weights = [[0.5, 0.3], [0.2, 0.4]]
    # result should be [1*0.5 + 1*0.2, 1*0.3 + 1*0.4] = [0.7, 0.7]
    assert_almost_equal(grads.grad_input._data.bitcast[Float32]()[0], Float32(0.7), tolerance=1e-5)
    assert_almost_equal(grads.grad_input._data.bitcast[Float32]()[1], Float32(0.7), tolerance=1e-5)


fn test_linear_backward_batch() raises:
    """Test linear_backward with batch size > 1."""
    var batch = 3
    var in_features = 4
    var out_features = 2

    var input_shape = DynamicVector[Int](2)
    input_shape[0] = batch
    input_shape[1] = in_features
    var x = ones(input_shape, DType.float32)

    var weight_shape = DynamicVector[Int](2)
    weight_shape[0] = out_features
    weight_shape[1] = in_features
    var weights = ones(weight_shape, DType.float32)

    var grad_out_shape = DynamicVector[Int](2)
    grad_out_shape[0] = batch
    grad_out_shape[1] = out_features
    var grad_output = ones(grad_out_shape, DType.float32)

    # Backward pass
    var grads = linear_backward(grad_output, x, weights)

    # Verify shapes
    assert_equal(grads.grad_input.shape()[0], batch)
    assert_equal(grads.grad_weights.shape()[0], out_features)
    assert_equal(grads.grad_bias.shape()[0], out_features)


fn test_linear_backward_gradient() raises:
    """Test linear backward with numerical gradient checking."""
    var batch = 2
    var in_features = 3
    var out_features = 2

    var input_shape = DynamicVector[Int](2)
    input_shape[0] = batch
    input_shape[1] = in_features
    var x = zeros(input_shape, DType.float32)

    # Initialize with non-uniform values
    x._data.bitcast[Float32]()[0] = 0.5
    x._data.bitcast[Float32]()[1] = -0.3
    x._data.bitcast[Float32]()[2] = 1.2
    x._data.bitcast[Float32]()[3] = -0.8
    x._data.bitcast[Float32]()[4] = 0.1
    x._data.bitcast[Float32]()[5] = 0.7

    var weight_shape = DynamicVector[Int](2)
    weight_shape[0] = out_features
    weight_shape[1] = in_features
    var weights = zeros(weight_shape, DType.float32)
    weights._data.bitcast[Float32]()[0] = 0.4
    weights._data.bitcast[Float32]()[1] = 0.2
    weights._data.bitcast[Float32]()[2] = -0.3
    weights._data.bitcast[Float32]()[3] = 0.6
    weights._data.bitcast[Float32]()[4] = -0.2
    weights._data.bitcast[Float32]()[5] = 0.5

    # Forward function wrapper (linear without bias for simplicity)
    fn forward(inp: ExTensor) raises -> ExTensor:
        var bias_shape = DynamicVector[Int](1)
        bias_shape[0] = out_features
        var bias = zeros(bias_shape, DType.float32)
        return linear(inp, weights, bias)

    # Backward function wrapper (only return grad_input)
    fn backward(grad_out: ExTensor, inp: ExTensor) raises -> ExTensor:
        var bias_shape = DynamicVector[Int](1)
        bias_shape[0] = out_features
        var bias = zeros(bias_shape, DType.float32)
        var grads = linear_backward(grad_out, inp, weights)
        return grads.grad_input

    var output = forward(x)
    var grad_output = ones_like(output)

    # Numerical gradient checking
    check_gradient(forward, backward, x, grad_output, rtol=1e-3, atol=1e-6)


# ============================================================================
# Conv2D Backward Tests
# ============================================================================


fn test_conv2d_backward_shapes() raises:
    """Test that conv2d_backward returns correct gradient shapes."""
    var batch = 2
    var in_channels = 3
    var out_channels = 4
    var in_h = 8
    var in_w = 8
    var kh = 3
    var kw = 3

    # Create input
    var input_shape = DynamicVector[Int](4)
    input_shape[0] = batch
    input_shape[1] = in_channels
    input_shape[2] = in_h
    input_shape[3] = in_w
    var x = ones(input_shape, DType.float32)

    # Create kernel
    var kernel_shape = DynamicVector[Int](4)
    kernel_shape[0] = out_channels
    kernel_shape[1] = in_channels
    kernel_shape[2] = kh
    kernel_shape[3] = kw
    var kernel = ones(kernel_shape, DType.float32)

    # Create bias
    var bias_shape = DynamicVector[Int](1)
    bias_shape[0] = out_channels
    var bias = zeros(bias_shape, DType.float32)

    # Forward pass to get output shape
    var output = conv2d(x, kernel, bias, stride=1, padding=0)
    var out_shape = output.shape()

    # Create grad_output with same shape as output
    var grad_output = ones_like(output)

    # Backward pass
    var grads = conv2d_backward(grad_output, x, kernel, stride=1, padding=0)

    # Check shapes match original inputs
    var gi_shape = grads.grad_input.shape()
    assert_equal(gi_shape[0], batch)
    assert_equal(gi_shape[1], in_channels)
    assert_equal(gi_shape[2], in_h)
    assert_equal(gi_shape[3], in_w)

    var gk_shape = grads.grad_weights.shape()
    assert_equal(gk_shape[0], out_channels)
    assert_equal(gk_shape[1], in_channels)
    assert_equal(gk_shape[2], kh)
    assert_equal(gk_shape[3], kw)

    var gb_shape = grads.grad_bias.shape()
    assert_equal(gb_shape[0], out_channels)


fn test_conv2d_backward_with_stride() raises:
    """Test conv2d_backward with stride > 1."""
    var batch = 1
    var in_channels = 1
    var out_channels = 1

    var input_shape = DynamicVector[Int](4)
    input_shape[0] = batch
    input_shape[1] = in_channels
    input_shape[2] = 8
    input_shape[3] = 8
    var x = ones(input_shape, DType.float32)

    var kernel_shape = DynamicVector[Int](4)
    kernel_shape[0] = out_channels
    kernel_shape[1] = in_channels
    kernel_shape[2] = 3
    kernel_shape[3] = 3
    var kernel = ones(kernel_shape, DType.float32)

    var bias_shape = DynamicVector[Int](1)
    bias_shape[0] = out_channels
    var bias = zeros(bias_shape, DType.float32)

    # Forward pass with stride=2
    var output = conv2d(x, kernel, bias, stride=2, padding=0)
    var grad_output = ones_like(output)

    # Backward pass
    var grads = conv2d_backward(grad_output, x, kernel, stride=2, padding=0)

    # Check grad_input shape matches input
    var gi_shape = grads.grad_input.shape()
    assert_equal(gi_shape[0], 1)
    assert_equal(gi_shape[1], 1)
    assert_equal(gi_shape[2], 8)
    assert_equal(gi_shape[3], 8)


fn test_conv2d_backward_gradient() raises:
    """Test conv2d backward with numerical gradient checking."""
    var batch = 1
    var in_channels = 2
    var out_channels = 2
    var in_h = 5
    var in_w = 5
    var kh = 3
    var kw = 3

    var input_shape = DynamicVector[Int](4)
    input_shape[0] = batch
    input_shape[1] = in_channels
    input_shape[2] = in_h
    input_shape[3] = in_w
    var x = zeros(input_shape, DType.float32)

    # Initialize with non-uniform values
    for i in range(batch * in_channels * in_h * in_w):
        x._data.bitcast[Float32]()[i] = Float32(i) * 0.1 - 2.5

    var kernel_shape = DynamicVector[Int](4)
    kernel_shape[0] = out_channels
    kernel_shape[1] = in_channels
    kernel_shape[2] = kh
    kernel_shape[3] = kw
    var kernel = zeros(kernel_shape, DType.float32)

    # Initialize kernel with non-uniform values
    for i in range(out_channels * in_channels * kh * kw):
        kernel._data.bitcast[Float32]()[i] = Float32(i) * 0.05 - 0.5

    var bias_shape = DynamicVector[Int](1)
    bias_shape[0] = out_channels
    var bias = zeros(bias_shape, DType.float32)

    # Forward function wrapper
    fn forward(inp: ExTensor) raises -> ExTensor:
        return conv2d(inp, kernel, bias, stride=1, padding=0)

    # Backward function wrapper (only return grad_input)
    fn backward(grad_out: ExTensor, inp: ExTensor) raises -> ExTensor:
        var grads = conv2d_backward(grad_out, inp, kernel, stride=1, padding=0)
        return grads.grad_input

    var output = forward(x)
    var grad_output = ones_like(output)

    # Numerical gradient checking (looser tolerance for conv2d due to accumulation)
    check_gradient(forward, backward, x, grad_output, rtol=1e-2, atol=1e-5)


# ============================================================================
# Pooling Backward Tests
# ============================================================================


fn test_maxpool2d_backward_shapes() raises:
    """Test that maxpool2d_backward returns correct gradient shape."""
    var batch = 2
    var channels = 3
    var height = 8
    var width = 8

    var input_shape = DynamicVector[Int](4)
    input_shape[0] = batch
    input_shape[1] = channels
    input_shape[2] = height
    input_shape[3] = width
    var x = ones(input_shape, DType.float32)

    # Forward pass
    var output = maxpool2d(x, kernel_size=2, stride=2, padding=0)
    var grad_output = ones_like(output)

    # Backward pass
    var grad_input = maxpool2d_backward(grad_output, x, kernel_size=2, stride=2, padding=0)

    # Check shape matches input
    var gi_shape = grad_input.shape()
    assert_equal(gi_shape[0], batch)
    assert_equal(gi_shape[1], channels)
    assert_equal(gi_shape[2], height)
    assert_equal(gi_shape[3], width)


fn test_maxpool2d_backward_gradient_routing() raises:
    """Test that maxpool2d_backward routes gradients only to max positions."""
    # Create a simple 2x2 input with known max
    var input_shape = DynamicVector[Int](4)
    input_shape[0] = 1
    input_shape[1] = 1
    input_shape[2] = 2
    input_shape[3] = 2
    var x = zeros(input_shape, DType.float32)

    # Set values: [1, 2]
    #              [3, 4]  <- max is at position (1, 1) = 4
    x._data.bitcast[Float32]()[0] = 1.0
    x._data.bitcast[Float32]()[1] = 2.0
    x._data.bitcast[Float32]()[2] = 3.0
    x._data.bitcast[Float32]()[3] = 4.0

    # Forward pass - should select 4.0
    var output = maxpool2d(x, kernel_size=2, stride=2, padding=0)

    # Gradient of 1.0 from upstream
    var grad_output = ones_like(output)

    # Backward pass
    var grad_input = maxpool2d_backward(grad_output, x, kernel_size=2, stride=2, padding=0)

    # Only position (1, 1) should have gradient
    assert_almost_equal(grad_input._data.bitcast[Float32]()[0], Float32(0.0), tolerance=1e-5)
    assert_almost_equal(grad_input._data.bitcast[Float32]()[1], Float32(0.0), tolerance=1e-5)
    assert_almost_equal(grad_input._data.bitcast[Float32]()[2], Float32(0.0), tolerance=1e-5)
    assert_almost_equal(grad_input._data.bitcast[Float32]()[3], Float32(1.0), tolerance=1e-5)


fn test_avgpool2d_backward_shapes() raises:
    """Test that avgpool2d_backward returns correct gradient shape."""
    var batch = 2
    var channels = 3
    var height = 8
    var width = 8

    var input_shape = DynamicVector[Int](4)
    input_shape[0] = batch
    input_shape[1] = channels
    input_shape[2] = height
    input_shape[3] = width
    var x = ones(input_shape, DType.float32)

    # Forward pass
    var output = avgpool2d(x, kernel_size=2, stride=2, padding=0)
    var grad_output = ones_like(output)

    # Backward pass
    var grad_input = avgpool2d_backward(grad_output, x, kernel_size=2, stride=2, padding=0)

    # Check shape matches input
    var gi_shape = grad_input.shape()
    assert_equal(gi_shape[0], batch)
    assert_equal(gi_shape[1], channels)
    assert_equal(gi_shape[2], height)
    assert_equal(gi_shape[3], width)


fn test_avgpool2d_backward_gradient_distribution() raises:
    """Test that avgpool2d_backward distributes gradients equally."""
    # Create a simple 2x2 input
    var input_shape = DynamicVector[Int](4)
    input_shape[0] = 1
    input_shape[1] = 1
    input_shape[2] = 2
    input_shape[3] = 2
    var x = ones(input_shape, DType.float32)

    # Forward pass
    var output = avgpool2d(x, kernel_size=2, stride=2, padding=0)

    # Gradient of 1.0 from upstream
    var grad_output = ones_like(output)

    # Backward pass
    var grad_input = avgpool2d_backward(grad_output, x, kernel_size=2, stride=2, padding=0)

    # All 4 positions should have equal gradient (1.0 / 4 = 0.25)
    assert_almost_equal(grad_input._data.bitcast[Float32]()[0], Float32(0.25), tolerance=1e-5)
    assert_almost_equal(grad_input._data.bitcast[Float32]()[1], Float32(0.25), tolerance=1e-5)
    assert_almost_equal(grad_input._data.bitcast[Float32]()[2], Float32(0.25), tolerance=1e-5)
    assert_almost_equal(grad_input._data.bitcast[Float32]()[3], Float32(0.25), tolerance=1e-5)


fn test_maxpool2d_backward_gradient() raises:
    """Test maxpool2d backward with numerical gradient checking."""
    # Create input with non-uniform values
    var input_shape = DynamicVector[Int](4)
    input_shape[0] = 1  # batch
    input_shape[1] = 2  # channels
    input_shape[2] = 4  # height
    input_shape[3] = 4  # width
    var x = zeros(input_shape, DType.float32)

    # Initialize with non-uniform values
    for i in range(1 * 2 * 4 * 4):
        x._data.bitcast[Float32]()[i] = Float32(i) * 0.1 - 1.6

    # Forward function wrapper
    fn forward(inp: ExTensor) raises -> ExTensor:
        return maxpool2d(inp, kernel_size=2, stride=2, padding=0)

    # Backward function wrapper (only return grad_input)
    fn backward(grad_out: ExTensor, inp: ExTensor) raises -> ExTensor:
        return maxpool2d_backward(grad_out, inp, kernel_size=2, stride=2, padding=0)

    var output = forward(x)
    var grad_output = ones_like(output)

    # Numerical gradient checking
    check_gradient(forward, backward, x, grad_output, rtol=1e-3, atol=1e-6)


fn test_avgpool2d_backward_gradient() raises:
    """Test avgpool2d backward with numerical gradient checking."""
    # Create input with non-uniform values
    var input_shape = DynamicVector[Int](4)
    input_shape[0] = 1  # batch
    input_shape[1] = 2  # channels
    input_shape[2] = 4  # height
    input_shape[3] = 4  # width
    var x = zeros(input_shape, DType.float32)

    # Initialize with non-uniform values
    for i in range(1 * 2 * 4 * 4):
        x._data.bitcast[Float32]()[i] = Float32(i) * 0.1 - 1.6

    # Forward function wrapper
    fn forward(inp: ExTensor) raises -> ExTensor:
        return avgpool2d(inp, kernel_size=2, stride=2, padding=0)

    # Backward function wrapper (only return grad_input)
    fn backward(grad_out: ExTensor, inp: ExTensor) raises -> ExTensor:
        return avgpool2d_backward(grad_out, inp, kernel_size=2, stride=2, padding=0)

    var output = forward(x)
    var grad_output = ones_like(output)

    # Numerical gradient checking
    check_gradient(forward, backward, x, grad_output, rtol=1e-3, atol=1e-6)


# ============================================================================
# Cross-Entropy Backward Tests
# ============================================================================


fn test_cross_entropy_backward_shapes() raises:
    """Test that cross_entropy_backward returns correct gradient shape."""
    var batch = 4
    var num_classes = 10

    var logits_shape = DynamicVector[Int](2)
    logits_shape[0] = batch
    logits_shape[1] = num_classes
    var logits = ones(logits_shape, DType.float32)

    var targets_shape = DynamicVector[Int](2)
    targets_shape[0] = batch
    targets_shape[1] = num_classes
    var targets = zeros(targets_shape, DType.float32)
    # Set one-hot targets (first class)
    for i in range(batch):
        targets._data.bitcast[Float32]()[i * num_classes] = 1.0

    # Forward pass
    var loss = cross_entropy(logits, targets)

    # Backward pass (grad_output is scalar 1.0 typically)
    var grad_output = ones_like(loss)
    var grad_logits = cross_entropy_backward(grad_output, logits, targets)

    # Check shape matches logits
    var gl_shape = grad_logits.shape()
    assert_equal(gl_shape[0], batch)
    assert_equal(gl_shape[1], num_classes)


fn test_cross_entropy_backward_gradient() raises:
    """Test cross-entropy backward with numerical gradient checking."""
    var batch = 2
    var num_classes = 3

    var logits_shape = DynamicVector[Int](2)
    logits_shape[0] = batch
    logits_shape[1] = num_classes
    var logits = zeros(logits_shape, DType.float32)

    # Initialize with non-uniform values
    logits._data.bitcast[Float32]()[0] = 0.5
    logits._data.bitcast[Float32]()[1] = -0.3
    logits._data.bitcast[Float32]()[2] = 1.2
    logits._data.bitcast[Float32]()[3] = -0.8
    logits._data.bitcast[Float32]()[4] = 0.1
    logits._data.bitcast[Float32]()[5] = 0.7

    var targets_shape = DynamicVector[Int](2)
    targets_shape[0] = batch
    targets_shape[1] = num_classes
    var targets = zeros(targets_shape, DType.float32)

    # Set one-hot targets: [1, 0, 0] and [0, 1, 0]
    targets._data.bitcast[Float32]()[0] = 1.0  # Batch 0, class 0
    targets._data.bitcast[Float32]()[4] = 1.0  # Batch 1, class 1

    # Forward function wrapper
    fn forward(inp: ExTensor) raises -> ExTensor:
        return cross_entropy(inp, targets)

    # Backward function wrapper
    fn backward(grad_out: ExTensor, inp: ExTensor) raises -> ExTensor:
        return cross_entropy_backward(grad_out, inp, targets)

    var loss = forward(logits)
    var grad_output = ones_like(loss)

    # Numerical gradient checking
    check_gradient(forward, backward, logits, grad_output, rtol=1e-3, atol=1e-6)


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run all backward pass tests."""
    print("Running backward pass tests...")

    # Linear backward tests
    test_linear_backward_shapes()
    print("✓ test_linear_backward_shapes")

    test_linear_backward_numerical()
    print("✓ test_linear_backward_numerical")

    test_linear_backward_batch()
    print("✓ test_linear_backward_batch")

    test_linear_backward_gradient()
    print("✓ test_linear_backward_gradient")

    # Conv2D backward tests
    test_conv2d_backward_shapes()
    print("✓ test_conv2d_backward_shapes")

    test_conv2d_backward_with_stride()
    print("✓ test_conv2d_backward_with_stride")

    test_conv2d_backward_gradient()
    print("✓ test_conv2d_backward_gradient")

    # MaxPool2D backward tests
    test_maxpool2d_backward_shapes()
    print("✓ test_maxpool2d_backward_shapes")

    test_maxpool2d_backward_gradient_routing()
    print("✓ test_maxpool2d_backward_gradient_routing")

    test_maxpool2d_backward_gradient()
    print("✓ test_maxpool2d_backward_gradient")

    # AvgPool2D backward tests
    test_avgpool2d_backward_shapes()
    print("✓ test_avgpool2d_backward_shapes")

    test_avgpool2d_backward_gradient_distribution()
    print("✓ test_avgpool2d_backward_gradient_distribution")

    test_avgpool2d_backward_gradient()
    print("✓ test_avgpool2d_backward_gradient")

    # Cross-entropy backward tests
    test_cross_entropy_backward_shapes()
    print("✓ test_cross_entropy_backward_shapes")

    test_cross_entropy_backward_gradient()
    print("✓ test_cross_entropy_backward_gradient")

    print("\nAll backward pass tests passed!")
