"""Tests for activation functions.

Tests cover all 10 activation functions:
- ReLU, Leaky ReLU, PReLU
- Sigmoid, Tanh
- Softmax
- GELU, Swish, Mish, ELU

Each test covers:
- Basic correctness with known values
- Backward pass (gradient checking)
- Edge cases (zero, very large, very small)
- Dtype support (float32, float64)

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
from shared.core.activation import (
    relu,
    leaky_relu,
    prelu,
    sigmoid,
    tanh,
    softmax,
    gelu,
    swish,
    mish,
    elu,
    relu_backward,
    leaky_relu_backward,
    prelu_backward,
    sigmoid_backward,
    tanh_backward,
    softmax_backward,
    gelu_backward,
    swish_backward,
    mish_backward,
    elu_backward,
)
from tests.helpers.gradient_checking import check_gradient, compute_numerical_gradient, assert_gradients_close
from collections.vector import DynamicVector
from math import tanh as math_tanh, exp as math_exp


# ============================================================================
# ReLU Tests
# ============================================================================


fn test_relu_basic() raises:
    """Test ReLU with known values."""
    var shape = DynamicVector[Int](1)
    shape[0] = 5
    var x = zeros(shape, DType.float32)

    # Set test values: [-2, -1, 0, 1, 2]
    x._data.bitcast[Float32]()[0] = -2.0
    x._data.bitcast[Float32]()[1] = -1.0
    x._data.bitcast[Float32]()[2] = 0.0
    x._data.bitcast[Float32]()[3] = 1.0
    x._data.bitcast[Float32]()[4] = 2.0

    var y = relu(x)

    # Expected: [0, 0, 0, 1, 2]
    assert_almost_equal(y._data.bitcast[Float32]()[0], Float32(0.0), tolerance=1e-5)
    assert_almost_equal(y._data.bitcast[Float32]()[1], Float32(0.0), tolerance=1e-5)
    assert_almost_equal(y._data.bitcast[Float32]()[2], Float32(0.0), tolerance=1e-5)
    assert_almost_equal(y._data.bitcast[Float32]()[3], Float32(1.0), tolerance=1e-5)
    assert_almost_equal(y._data.bitcast[Float32]()[4], Float32(2.0), tolerance=1e-5)


fn test_relu_backward() raises:
    """Test ReLU gradient with numerical validation."""
    var shape = DynamicVector[Int](1)
    shape[0] = 4
    var x = zeros(shape, DType.float32)

    # Set test values: [-1, 0, 0.5, 2]
    x._data.bitcast[Float32]()[0] = -1.0
    x._data.bitcast[Float32]()[1] = 0.0
    x._data.bitcast[Float32]()[2] = 0.5
    x._data.bitcast[Float32]()[3] = 2.0

    # Forward function wrapper
    fn forward(inp: ExTensor) raises -> ExTensor:
        return relu(inp)

    var y = relu(x)
    var grad_out = ones_like(y)

    # Use numerical gradient checking (gold standard)
    check_gradient(forward, relu_backward, x, grad_out, rtol=1e-4, atol=1e-7)


fn test_relu_shape() raises:
    """Test ReLU preserves shape."""
    var shape = DynamicVector[Int](3)
    shape[0] = 2
    shape[1] = 3
    shape[2] = 4
    var x = ones(shape, DType.float32)

    var y = relu(x)

    assert_equal(y.shape()[0], 2)
    assert_equal(y.shape()[1], 3)
    assert_equal(y.shape()[2], 4)


# ============================================================================
# Leaky ReLU Tests
# ============================================================================


fn test_leaky_relu_basic() raises:
    """Test Leaky ReLU with known values."""
    var shape = DynamicVector[Int](1)
    shape[0] = 3
    var x = zeros(shape, DType.float32)

    x._data.bitcast[Float32]()[0] = -2.0
    x._data.bitcast[Float32]()[1] = 0.0
    x._data.bitcast[Float32]()[2] = 2.0

    var y = leaky_relu(x, alpha=0.1)

    # Expected: [-0.2, 0, 2.0]
    assert_almost_equal(y._data.bitcast[Float32]()[0], Float32(-0.2), tolerance=1e-5)
    assert_almost_equal(y._data.bitcast[Float32]()[1], Float32(0.0), tolerance=1e-5)
    assert_almost_equal(y._data.bitcast[Float32]()[2], Float32(2.0), tolerance=1e-5)


fn test_leaky_relu_backward() raises:
    """Test Leaky ReLU gradient with numerical validation."""
    var shape = DynamicVector[Int](1)
    shape[0] = 2
    var x = zeros(shape, DType.float32)

    x._data.bitcast[Float32]()[0] = -1.0
    x._data.bitcast[Float32]()[1] = 1.0

    # Forward function wrapper
    fn forward(inp: ExTensor) raises -> ExTensor:
        return leaky_relu(inp, alpha=0.1)

    var y = leaky_relu(x, alpha=0.1)
    var grad_out = ones_like(y)

    # Use numerical gradient checking (gold standard)
    fn backward_wrapper(grad: ExTensor, inp: ExTensor) raises -> ExTensor:
        return leaky_relu_backward(grad, inp, alpha=0.1)

    check_gradient(forward, backward_wrapper, x, grad_out, rtol=1e-4, atol=1e-7)


# ============================================================================
# PReLU Tests
# ============================================================================


fn test_prelu_basic() raises:
    """Test PReLU with known values."""
    var shape = DynamicVector[Int](1)
    shape[0] = 3
    var x = zeros(shape, DType.float32)
    var alpha = zeros(shape, DType.float32)

    x._data.bitcast[Float32]()[0] = -2.0
    x._data.bitcast[Float32]()[1] = 0.0
    x._data.bitcast[Float32]()[2] = 2.0

    alpha._data.bitcast[Float32]()[0] = 0.25
    alpha._data.bitcast[Float32]()[1] = 0.25
    alpha._data.bitcast[Float32]()[2] = 0.25

    var y = prelu(x, alpha)

    # Expected: [-0.5, 0, 2.0]
    assert_almost_equal(y._data.bitcast[Float32]()[0], Float32(-0.5), tolerance=1e-5)
    assert_almost_equal(y._data.bitcast[Float32]()[1], Float32(0.0), tolerance=1e-5)
    assert_almost_equal(y._data.bitcast[Float32]()[2], Float32(2.0), tolerance=1e-5)


fn test_prelu_backward() raises:
    """Test PReLU gradient with numerical validation."""
    var shape = DynamicVector[Int](1)
    shape[0] = 2
    var x = zeros(shape, DType.float32)
    var alpha = zeros(shape, DType.float32)

    x._data.bitcast[Float32]()[0] = -1.0
    x._data.bitcast[Float32]()[1] = 1.0

    alpha._data.bitcast[Float32]()[0] = 0.5
    alpha._data.bitcast[Float32]()[1] = 0.5

    # Forward function wrapper
    fn forward(inp: ExTensor) raises -> ExTensor:
        return prelu(inp, alpha)

    var y = prelu(x, alpha)
    var grad_out = ones_like(y)

    # Validate gradient w.r.t. input using numerical checking
    fn backward_input(grad: ExTensor, inp: ExTensor) raises -> ExTensor:
        var result = prelu_backward(grad, inp, alpha)
        return result[0]

    check_gradient(forward, backward_input, x, grad_out, rtol=1e-4, atol=1e-7)


# ============================================================================
# Sigmoid Tests
# ============================================================================


fn test_sigmoid_basic() raises:
    """Test sigmoid with known values."""
    var shape = DynamicVector[Int](1)
    shape[0] = 3
    var x = zeros(shape, DType.float32)

    x._data.bitcast[Float32]()[0] = -100.0  # Should be ~0
    x._data.bitcast[Float32]()[1] = 0.0     # Should be 0.5
    x._data.bitcast[Float32]()[2] = 100.0   # Should be ~1

    var y = sigmoid(x)

    assert_almost_equal(y._data.bitcast[Float32]()[0], Float32(0.0), tolerance=1e-3)
    assert_almost_equal(y._data.bitcast[Float32]()[1], Float32(0.5), tolerance=1e-5)
    assert_almost_equal(y._data.bitcast[Float32]()[2], Float32(1.0), tolerance=1e-3)


fn test_sigmoid_backward() raises:
    """Test sigmoid gradient with numerical validation."""
    var shape = DynamicVector[Int](1)
    shape[0] = 3
    var x = zeros(shape, DType.float32)

    # Use multiple test points for better coverage
    x._data.bitcast[Float32]()[0] = -1.0
    x._data.bitcast[Float32]()[1] = 0.0
    x._data.bitcast[Float32]()[2] = 1.0

    # Forward function wrapper
    fn forward(inp: ExTensor) raises -> ExTensor:
        return sigmoid(inp)

    var y = sigmoid(x)
    var grad_out = ones_like(y)

    # Note: sigmoid_backward takes output y, not input x
    fn backward_fn(grad: ExTensor, `_`: ExTensor) raises -> ExTensor:
        return sigmoid_backward(grad, y)

    # Use numerical gradient checking (gold standard)
    check_gradient(forward, backward_fn, x, grad_out, rtol=1e-4, atol=1e-7)


fn test_sigmoid_range() raises:
    """Test sigmoid output is in (0, 1)."""
    var shape = DynamicVector[Int](1)
    shape[0] = 5
    var x = zeros(shape, DType.float32)

    x._data.bitcast[Float32]()[0] = -10.0
    x._data.bitcast[Float32]()[1] = -1.0
    x._data.bitcast[Float32]()[2] = 0.0
    x._data.bitcast[Float32]()[3] = 1.0
    x._data.bitcast[Float32]()[4] = 10.0

    var y = sigmoid(x)

    # All values should be in (0, 1)
    for i in range(5):
        var val = y._data.bitcast[Float32]()[i]
        assert_true(val > 0.0)
        assert_true(val < 1.0)


# ============================================================================
# Tanh Tests
# ============================================================================


fn test_tanh_basic() raises:
    """Test tanh with known values."""
    var shape = DynamicVector[Int](1)
    shape[0] = 3
    var x = zeros(shape, DType.float32)

    x._data.bitcast[Float32]()[0] = -100.0  # Should be ~-1
    x._data.bitcast[Float32]()[1] = 0.0     # Should be 0
    x._data.bitcast[Float32]()[2] = 100.0   # Should be ~1

    var y = tanh(x)

    assert_almost_equal(y._data.bitcast[Float32]()[0], Float32(-1.0), tolerance=1e-3)
    assert_almost_equal(y._data.bitcast[Float32]()[1], Float32(0.0), tolerance=1e-5)
    assert_almost_equal(y._data.bitcast[Float32]()[2], Float32(1.0), tolerance=1e-3)


fn test_tanh_backward() raises:
    """Test tanh gradient with numerical validation."""
    var shape = DynamicVector[Int](1)
    shape[0] = 3
    var x = zeros(shape, DType.float32)

    # Use multiple test points for better coverage
    x._data.bitcast[Float32]()[0] = -1.0
    x._data.bitcast[Float32]()[1] = 0.0
    x._data.bitcast[Float32]()[2] = 1.0

    # Forward function wrapper
    fn forward(inp: ExTensor) raises -> ExTensor:
        return tanh(inp)

    var y = tanh(x)
    var grad_out = ones_like(y)

    # Note: tanh_backward takes output y, not input x
    fn backward_fn(grad: ExTensor, `_`: ExTensor) raises -> ExTensor:
        return tanh_backward(grad, y)

    # Use numerical gradient checking (gold standard)
    check_gradient(forward, backward_fn, x, grad_out, rtol=1e-4, atol=1e-7)


fn test_tanh_range() raises:
    """Test tanh output is in (-1, 1)."""
    var shape = DynamicVector[Int](1)
    shape[0] = 5
    var x = zeros(shape, DType.float32)

    x._data.bitcast[Float32]()[0] = -10.0
    x._data.bitcast[Float32]()[1] = -1.0
    x._data.bitcast[Float32]()[2] = 0.0
    x._data.bitcast[Float32]()[3] = 1.0
    x._data.bitcast[Float32]()[4] = 10.0

    var y = tanh(x)

    # All values should be in (-1, 1)
    for i in range(5):
        var val = y._data.bitcast[Float32]()[i]
        assert_true(val > -1.0)
        assert_true(val < 1.0)


# ============================================================================
# Softmax Tests
# ============================================================================


fn test_softmax_basic_2d() raises:
    """Test softmax 2D normalization."""
    var shape = DynamicVector[Int](2)
    shape[0] = 1
    shape[1] = 3
    var x = zeros(shape, DType.float32)

    # All zeros should give uniform distribution
    var y = softmax(x, axis=1)

    # Sum should be 1.0
    var sum = y._data.bitcast[Float32]()[0] + y._data.bitcast[Float32]()[1] + y._data.bitcast[Float32]()[2]
    assert_almost_equal(sum, Float32(1.0), tolerance=1e-5)

    # Each value should be ~1/3
    assert_almost_equal(y._data.bitcast[Float32]()[0], Float32(0.333333), tolerance=1e-3)
    assert_almost_equal(y._data.bitcast[Float32]()[1], Float32(0.333333), tolerance=1e-3)
    assert_almost_equal(y._data.bitcast[Float32]()[2], Float32(0.333333), tolerance=1e-3)


fn test_softmax_one_hot() raises:
    """Test softmax with large difference (one-hot-like)."""
    var shape = DynamicVector[Int](2)
    shape[0] = 1
    shape[1] = 3
    var x = zeros(shape, DType.float32)

    x._data.bitcast[Float32]()[0] = 0.0
    x._data.bitcast[Float32]()[1] = 10.0
    x._data.bitcast[Float32]()[2] = 0.0

    var y = softmax(x, axis=1)

    # Middle value should be ~1.0, others ~0.0
    assert_almost_equal(y._data.bitcast[Float32]()[0], Float32(0.0), tolerance=1e-3)
    assert_almost_equal(y._data.bitcast[Float32]()[1], Float32(1.0), tolerance=1e-3)
    assert_almost_equal(y._data.bitcast[Float32]()[2], Float32(0.0), tolerance=1e-3)


fn test_softmax_sum_to_one() raises:
    """Test softmax probabilities sum to 1."""
    var shape = DynamicVector[Int](2)
    shape[0] = 2
    shape[1] = 4
    var x = zeros(shape, DType.float32)

    # Set random values
    for i in range(8):
        x._data.bitcast[Float32]()[i] = Float32(i % 5) - 2.0

    var y = softmax(x, axis=1)

    # Each row should sum to 1.0
    var sum_row0 = Float32(0.0)
    var sum_row1 = Float32(0.0)
    for i in range(4):
        sum_row0 += y._data.bitcast[Float32]()[i]
        sum_row1 += y._data.bitcast[Float32]()[4 + i]

    assert_almost_equal(sum_row0, Float32(1.0), tolerance=1e-5)
    assert_almost_equal(sum_row1, Float32(1.0), tolerance=1e-5)


fn test_softmax_backward() raises:
    """Test softmax gradient with numerical validation."""
    var shape = DynamicVector[Int](2)
    shape[0] = 2
    shape[1] = 3
    var x = zeros(shape, DType.float32)

    # Set test values
    x._data.bitcast[Float32]()[0] = -1.0
    x._data.bitcast[Float32]()[1] = 0.0
    x._data.bitcast[Float32]()[2] = 1.0
    x._data.bitcast[Float32]()[3] = -0.5
    x._data.bitcast[Float32]()[4] = 0.5
    x._data.bitcast[Float32]()[5] = 1.5

    # Forward function wrapper
    fn forward(inp: ExTensor) raises -> ExTensor:
        return softmax(inp, axis=1)

    var y = softmax(x, axis=1)
    var grad_out = ones_like(y)

    # Note: softmax_backward takes output y, not input x
    fn backward_fn(grad: ExTensor, `_`: ExTensor) raises -> ExTensor:
        return softmax_backward(grad, y, axis=1)

    # Use numerical gradient checking (gold standard)
    check_gradient(forward, backward_fn, x, grad_out, rtol=1e-4, atol=1e-7)


# ============================================================================
# GELU Tests
# ============================================================================


fn test_gelu_basic() raises:
    """Test GELU with known value at x=0."""
    var shape = DynamicVector[Int](1)
    shape[0] = 1
    var x = zeros(shape, DType.float32)

    x._data.bitcast[Float32]()[0] = 0.0

    var y = gelu(x)

    # GELU(0) = 0
    assert_almost_equal(y._data.bitcast[Float32]()[0], Float32(0.0), tolerance=1e-5)


fn test_gelu_positive() raises:
    """Test GELU with positive values."""
    var shape = DynamicVector[Int](1)
    shape[0] = 2
    var x = zeros(shape, DType.float32)

    x._data.bitcast[Float32]()[0] = 1.0
    x._data.bitcast[Float32]()[1] = 2.0

    var y = gelu(x)

    # For positive x, GELU(x) H x (asymptotically)
    # GELU(1) H 0.84, GELU(2) H 1.96
    assert_true(y._data.bitcast[Float32]()[0] > 0.8)
    assert_true(y._data.bitcast[Float32]()[0] < 1.0)
    assert_true(y._data.bitcast[Float32]()[1] > 1.9)
    assert_true(y._data.bitcast[Float32]()[1] < 2.0)


fn test_gelu_shape() raises:
    """Test GELU preserves shape."""
    var shape = DynamicVector[Int](2)
    shape[0] = 3
    shape[1] = 4
    var x = ones(shape, DType.float32)

    var y = gelu(x)

    assert_equal(y.shape()[0], 3)
    assert_equal(y.shape()[1], 4)


fn test_gelu_backward_gradient() raises:
    """Test GELU backward with numerical gradient checking."""
    var shape = DynamicVector[Int](1)
    shape[0] = 3
    var x = zeros(shape, DType.float32)

    # Set non-uniform values
    x._data.bitcast[Float32]()[0] = -0.5
    x._data.bitcast[Float32]()[1] = 0.0
    x._data.bitcast[Float32]()[2] = 0.5

    # Forward function wrapper
    fn forward(inp: ExTensor) raises -> ExTensor:
        return gelu(inp, approximate=False)

    var y = gelu(x, approximate=False)
    var grad_out = ones_like(y)

    # Backward function wrapper
    fn backward_fn(grad: ExTensor, inp: ExTensor) raises -> ExTensor:
        return gelu_backward(grad, inp, approximate=False)

    # Use numerical gradient checking (gold standard)
    check_gradient(forward, backward_fn, x, grad_out, rtol=1e-3, atol=1e-6)


# ============================================================================
# Swish Tests
# ============================================================================


fn test_swish_basic() raises:
    """Test swish with known values."""
    var shape = DynamicVector[Int](1)
    shape[0] = 1
    var x = zeros(shape, DType.float32)

    x._data.bitcast[Float32]()[0] = 0.0

    var y = swish(x)

    # swish(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
    assert_almost_equal(y._data.bitcast[Float32]()[0], Float32(0.0), tolerance=1e-5)


fn test_swish_positive() raises:
    """Test swish with large positive value."""
    var shape = DynamicVector[Int](1)
    shape[0] = 1
    var x = zeros(shape, DType.float32)

    x._data.bitcast[Float32]()[0] = 10.0

    var y = swish(x)

    # swish(10) H 10 * sigmoid(10) H 10 * 1 H 10
    assert_almost_equal(y._data.bitcast[Float32]()[0], Float32(10.0), tolerance=0.01)


fn test_swish_backward_gradient() raises:
    """Test Swish backward with numerical gradient checking."""
    var shape = DynamicVector[Int](1)
    shape[0] = 3
    var x = zeros(shape, DType.float32)

    # Set non-uniform values
    x._data.bitcast[Float32]()[0] = -0.5
    x._data.bitcast[Float32]()[1] = 0.0
    x._data.bitcast[Float32]()[2] = 0.5

    # Forward function wrapper
    fn forward(inp: ExTensor) raises -> ExTensor:
        return swish(inp)

    var y = swish(x)
    var grad_out = ones_like(y)

    # Backward function wrapper
    fn backward_fn(grad: ExTensor, inp: ExTensor) raises -> ExTensor:
        return swish_backward(grad, inp)

    # Use numerical gradient checking (gold standard)
    check_gradient(forward, backward_fn, x, grad_out, rtol=1e-3, atol=1e-6)


# ============================================================================
# Mish Tests
# ============================================================================


fn test_mish_basic() raises:
    """Test mish with known values."""
    var shape = DynamicVector[Int](1)
    shape[0] = 1
    var x = zeros(shape, DType.float32)

    x._data.bitcast[Float32]()[0] = 0.0

    var y = mish(x)

    # mish(0) = 0 * tanh(softplus(0)) = 0 * tanh(log(2)) H 0
    assert_almost_equal(y._data.bitcast[Float32]()[0], Float32(0.0), tolerance=0.01)


fn test_mish_shape() raises:
    """Test mish preserves shape."""
    var shape = DynamicVector[Int](3)
    shape[0] = 2
    shape[1] = 3
    shape[2] = 4
    var x = ones(shape, DType.float32)

    var y = mish(x)

    assert_equal(y.shape()[0], 2)
    assert_equal(y.shape()[1], 3)
    assert_equal(y.shape()[2], 4)


fn test_mish_backward_gradient() raises:
    """Test Mish backward with numerical gradient checking."""
    var shape = DynamicVector[Int](1)
    shape[0] = 3
    var x = zeros(shape, DType.float32)

    # Set non-uniform values
    x._data.bitcast[Float32]()[0] = -0.5
    x._data.bitcast[Float32]()[1] = 0.0
    x._data.bitcast[Float32]()[2] = 0.5

    # Forward function wrapper
    fn forward(inp: ExTensor) raises -> ExTensor:
        return mish(inp)

    var y = mish(x)
    var grad_out = ones_like(y)

    # Backward function wrapper
    fn backward_fn(grad: ExTensor, inp: ExTensor) raises -> ExTensor:
        return mish_backward(grad, inp)

    # Use numerical gradient checking (gold standard)
    check_gradient(forward, backward_fn, x, grad_out, rtol=1e-3, atol=1e-6)


# ============================================================================
# ELU Tests
# ============================================================================


fn test_elu_basic() raises:
    """Test ELU with known values."""
    var shape = DynamicVector[Int](1)
    shape[0] = 3
    var x = zeros(shape, DType.float32)

    x._data.bitcast[Float32]()[0] = -1.0
    x._data.bitcast[Float32]()[1] = 0.0
    x._data.bitcast[Float32]()[2] = 1.0

    var y = elu(x, alpha=1.0)

    # ELU(-1) = 1.0 * (exp(-1) - 1) H -0.632
    # ELU(0) = 0
    # ELU(1) = 1
    assert_almost_equal(y._data.bitcast[Float32]()[0], Float32(-0.632), tolerance=0.01)
    assert_almost_equal(y._data.bitcast[Float32]()[1], Float32(0.0), tolerance=1e-5)
    assert_almost_equal(y._data.bitcast[Float32]()[2], Float32(1.0), tolerance=1e-5)


fn test_elu_backward() raises:
    """Test ELU gradient with numerical validation."""
    var shape = DynamicVector[Int](1)
    shape[0] = 3
    var x = zeros(shape, DType.float32)

    x._data.bitcast[Float32]()[0] = -1.0
    x._data.bitcast[Float32]()[1] = 0.0
    x._data.bitcast[Float32]()[2] = 1.0

    # Forward function wrapper
    fn forward(inp: ExTensor) raises -> ExTensor:
        return elu(inp, alpha=1.0)

    var y = elu(x, alpha=1.0)
    var grad_out = ones_like(y)

    # Note: elu_backward takes x, y, and alpha
    fn backward_fn(grad: ExTensor, inp: ExTensor) raises -> ExTensor:
        return elu_backward(grad, inp, y, alpha=1.0)

    # Use numerical gradient checking (gold standard)
    check_gradient(forward, backward_fn, x, grad_out, rtol=1e-4, atol=1e-7)


# ============================================================================
# Dtype Support Tests
# ============================================================================


fn test_relu_float64() raises:
    """Test ReLU with float64 dtype."""
    var shape = DynamicVector[Int](1)
    shape[0] = 2
    var x = zeros(shape, DType.float64)

    x._data.bitcast[Float64]()[0] = -1.0
    x._data.bitcast[Float64]()[1] = 1.0

    var y = relu(x)

    assert_almost_equal(y._data.bitcast[Float64]()[0], 0.0, tolerance=1e-10)
    assert_almost_equal(y._data.bitcast[Float64]()[1], 1.0, tolerance=1e-10)


fn test_sigmoid_float64() raises:
    """Test sigmoid with float64 dtype."""
    var shape = DynamicVector[Int](1)
    shape[0] = 1
    var x = zeros(shape, DType.float64)

    x._data.bitcast[Float64]()[0] = 0.0

    var y = sigmoid(x)

    assert_almost_equal(y._data.bitcast[Float64]()[0], 0.5, tolerance=1e-10)
