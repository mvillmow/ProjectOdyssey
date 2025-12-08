"""Tests for activation functions.

Comprehensive test suite covering all 10 activation functions:
- ReLU, Leaky ReLU, PReLU
- Sigmoid, Tanh
- Softmax
- GELU, Swish, Mish, ELU

Test coverage:
- Basic correctness with known values
- Backward pass (gradient checking with numerical validation)
- Edge cases (zero, very large, very small, integer types, float16)
- Numerical stability (overflow/underflow handling)
- Dtype support (float16, float32, float64, int32, uint8)
- Integration tests (forward + backward pass)

All tests use pure functional API.
"""

from tests.shared.conftest import (
    assert_almost_equal,
    assert_close_float,
    assert_equal,
    assert_equal_int,
    assert_false,
    assert_shape,
    assert_true,
)
from tests.shared.conftest import TestFixtures
from shared.core.extensor import (
    ExTensor,
    zeros,
    ones,
    full,
    zeros_like,
    ones_like,
)
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
    GradientPair,
)
from shared.testing import (
    check_gradient,
    compute_numerical_gradient,
    assert_gradients_close,
)
from math import tanh as math_tanh, exp as math_exp


# ============================================================================
# ReLU Tests
# ============================================================================


fn test_relu_basic() raises:
    """Test ReLU with known values."""
    var shape = List[Int]()
    shape.append(5)
    var x = zeros(shape, DType.float32)

    # Set test values: [-2, -1, 0, 1, 2]
    x._data.bitcast[Float32]()[0] = -2.0
    x._data.bitcast[Float32]()[1] = -1.0
    x._data.bitcast[Float32]()[2] = 0.0
    x._data.bitcast[Float32]()[3] = 1.0
    x._data.bitcast[Float32]()[4] = 2.0

    var y = relu(x)

    # Expected: [0, 0, 0, 1, 2]
    assert_almost_equal(
        y._data.bitcast[Float32]()[0], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        y._data.bitcast[Float32]()[1], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        y._data.bitcast[Float32]()[2], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        y._data.bitcast[Float32]()[3], Float32(1.0), tolerance=1e-5
    )
    assert_almost_equal(
        y._data.bitcast[Float32]()[4], Float32(2.0), tolerance=1e-5
    )


fn test_relu_non_negativity() raises:
    """Test ReLU always produces non-negative outputs."""
    var shape = List[Int]()
    shape.append(100)
    var x = zeros(shape, DType.float32)

    # Fill with values from -50 to 50
    for i in range(100):
        x._data.bitcast[Float32]()[i] = Float32(i - 50)

    var y = relu(x)

    # All outputs should be >= 0
    for i in range(100):
        var val = y._data.bitcast[Float32]()[i]
        assert_true(val >= 0.0)


fn test_relu_backward() raises:
    """Test ReLU gradient with numerical validation."""
    var shape = List[Int]()
    shape.append(4)
    var x = zeros(shape, DType.float32)

    # Set test values: [-1, 1e-4, 0.5, 2]
    x._data.bitcast[Float32]()[0] = -1.0
    x._data.bitcast[Float32]()[1] = 1e-4
    x._data.bitcast[Float32]()[2] = 0.5
    x._data.bitcast[Float32]()[3] = 2.0

    # Forward function wrapper
    fn forward(x: ExTensor) raises escaping -> ExTensor:
        return relu(x)

    var y = relu(x)
    var grad_out = ones_like(y)

    # Backward function wrapper
    fn backward_wrapper(
        grad: ExTensor, x: ExTensor
    ) raises escaping -> ExTensor:
        return relu_backward(grad, x)

    # Use numerical gradient checking (gold standard)
    # Note: rtol=1e-3 is appropriate for float32 finite differences
    check_gradient(forward, backward_wrapper, x, grad_out, rtol=1e-3, atol=1e-6)


fn test_relu_shape() raises:
    """Test ReLU preserves shape."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)
    var x = ones(shape, DType.float32)

    var y = relu(x)

    assert_equal(y.shape()[0], 2)
    assert_equal(y.shape()[1], 3)
    assert_equal(y.shape()[2], 4)


fn test_relu_integer_types() raises:
    """Test ReLU with integer types."""
    # Test int32
    var shape = List[Int]()
    shape.append(5)
    var x_int32 = zeros(shape, DType.int32)
    x_int32._data.bitcast[Int32]()[0] = -2
    x_int32._data.bitcast[Int32]()[1] = -1
    x_int32._data.bitcast[Int32]()[2] = 0
    x_int32._data.bitcast[Int32]()[3] = 1
    x_int32._data.bitcast[Int32]()[4] = 2

    var y_int32 = relu(x_int32)

    # Expected: [0, 0, 0, 1, 2]
    assert_equal(y_int32._data.bitcast[Int32]()[0], 0)
    assert_equal(y_int32._data.bitcast[Int32]()[1], 0)
    assert_equal(y_int32._data.bitcast[Int32]()[2], 0)
    assert_equal(y_int32._data.bitcast[Int32]()[3], 1)
    assert_equal(y_int32._data.bitcast[Int32]()[4], 2)

    # Test uint8 (already non-negative)
    var x_uint8 = zeros(shape, DType.uint8)
    x_uint8._data.bitcast[UInt8]()[0] = 0
    x_uint8._data.bitcast[UInt8]()[1] = 1
    x_uint8._data.bitcast[UInt8]()[2] = 128
    x_uint8._data.bitcast[UInt8]()[3] = 255
    x_uint8._data.bitcast[UInt8]()[4] = 100

    var y_uint8 = relu(x_uint8)

    # Should be unchanged
    assert_equal(y_uint8._data.bitcast[UInt8]()[0], 0)
    assert_equal(y_uint8._data.bitcast[UInt8]()[1], 1)
    assert_equal(y_uint8._data.bitcast[UInt8]()[2], 128)
    assert_equal(y_uint8._data.bitcast[UInt8]()[3], 255)


fn test_relu_float64() raises:
    """Test ReLU with float64 dtype."""
    var shape = List[Int]()
    shape.append(2)
    var x = zeros(shape, DType.float64)

    x._data.bitcast[Float64]()[0] = -1.0
    x._data.bitcast[Float64]()[1] = 1.0

    var y = relu(x)

    assert_almost_equal(y._data.bitcast[Float64]()[0], 0.0, tolerance=1e-10)
    assert_almost_equal(y._data.bitcast[Float64]()[1], 1.0, tolerance=1e-10)


# ============================================================================
# Leaky ReLU Tests
# ============================================================================


fn test_leaky_relu_basic() raises:
    """Test Leaky ReLU with known values."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)

    x._data.bitcast[Float32]()[0] = -2.0
    x._data.bitcast[Float32]()[1] = 0.0
    x._data.bitcast[Float32]()[2] = 2.0

    var y = leaky_relu(x, alpha=0.1)

    # Expected: [-0.2, 0, 2.0]
    assert_almost_equal(
        y._data.bitcast[Float32]()[0], Float32(-0.2), tolerance=1e-5
    )
    assert_almost_equal(
        y._data.bitcast[Float32]()[1], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        y._data.bitcast[Float32]()[2], Float32(2.0), tolerance=1e-5
    )


fn test_leaky_relu_custom_alpha() raises:
    """Test Leaky ReLU with custom alpha value."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)
    x._data.bitcast[Float32]()[0] = -4.0
    x._data.bitcast[Float32]()[1] = 0.0
    x._data.bitcast[Float32]()[2] = 4.0

    var y = leaky_relu(x, alpha=0.25)

    # Expected with alpha=0.25: [-1.0, 0, 4.0]
    assert_almost_equal(
        y._data.bitcast[Float32]()[0], Float32(-1.0), tolerance=1e-6
    )
    assert_almost_equal(
        y._data.bitcast[Float32]()[1], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        y._data.bitcast[Float32]()[2], Float32(4.0), tolerance=1e-5
    )


fn test_leaky_relu_backward() raises:
    """Test Leaky ReLU gradient with numerical validation."""
    var shape = List[Int]()
    shape.append(2)
    var x = zeros(shape, DType.float32)

    x._data.bitcast[Float32]()[0] = -1.0
    x._data.bitcast[Float32]()[1] = 1.0

    # Forward function wrapper
    fn forward(x: ExTensor) raises escaping -> ExTensor:
        return leaky_relu(x, alpha=0.1)

    var y = leaky_relu(x, alpha=0.1)
    var grad_out = ones_like(y)

    # Use numerical gradient checking (gold standard)
    fn backward_wrapper(
        grad: ExTensor, x: ExTensor
    ) raises escaping -> ExTensor:
        return leaky_relu_backward(grad, x, alpha=0.1)

    # Note: rtol=1e-3 is appropriate for float32 finite differences
    check_gradient(forward, backward_wrapper, x, grad_out, rtol=1e-3, atol=1e-6)


# ============================================================================
# PReLU Tests
# ============================================================================


fn test_prelu_basic() raises:
    """Test PReLU with known values."""
    var shape = List[Int]()
    shape.append(3)
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
    assert_almost_equal(
        y._data.bitcast[Float32]()[0], Float32(-0.5), tolerance=1e-5
    )
    assert_almost_equal(
        y._data.bitcast[Float32]()[1], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        y._data.bitcast[Float32]()[2], Float32(2.0), tolerance=1e-5
    )


fn test_prelu_scalar_alpha() raises:
    """Test PReLU with scalar alpha parameter."""
    var shape = List[Int]()
    shape.append(5)
    var x = zeros(shape, DType.float32)
    x._data.bitcast[Float32]()[0] = -2.0
    x._data.bitcast[Float32]()[1] = -1.0
    x._data.bitcast[Float32]()[2] = 0.0
    x._data.bitcast[Float32]()[3] = 1.0
    x._data.bitcast[Float32]()[4] = 2.0

    # Scalar alpha = 0.2
    var alpha_shape = List[Int]()
    alpha_shape.append(1)
    var alpha = full(alpha_shape, 0.2, DType.float32)

    var y = prelu(x, alpha)

    # Expected with alpha=0.2: [-0.4, -0.2, 0, 1, 2]
    assert_almost_equal(
        y._data.bitcast[Float32]()[0], Float32(-0.4), tolerance=1e-6
    )
    assert_almost_equal(
        y._data.bitcast[Float32]()[1], Float32(-0.2), tolerance=1e-6
    )
    assert_almost_equal(
        y._data.bitcast[Float32]()[2], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        y._data.bitcast[Float32]()[3], Float32(1.0), tolerance=1e-5
    )
    assert_almost_equal(
        y._data.bitcast[Float32]()[4], Float32(2.0), tolerance=1e-5
    )


fn test_prelu_elementwise_alpha() raises:
    """Test PReLU with element-wise alpha parameters."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)
    x._data.bitcast[Float32]()[0] = -2.0
    x._data.bitcast[Float32]()[1] = -1.0
    x._data.bitcast[Float32]()[2] = 2.0

    # Element-wise alpha = [0.1, 0.2, 0.3]
    var alpha = zeros(shape, DType.float32)
    alpha._data.bitcast[Float32]()[0] = 0.1
    alpha._data.bitcast[Float32]()[1] = 0.2
    alpha._data.bitcast[Float32]()[2] = 0.3

    var y = prelu(x, alpha)

    # Expected: [-0.2, -0.2, 2.0]
    assert_almost_equal(
        y._data.bitcast[Float32]()[0], Float32(-0.2), tolerance=1e-6
    )
    assert_almost_equal(
        y._data.bitcast[Float32]()[1], Float32(-0.2), tolerance=1e-6
    )
    assert_almost_equal(
        y._data.bitcast[Float32]()[2], Float32(2.0), tolerance=1e-5
    )


fn test_prelu_backward() raises:
    """Test PReLU gradient with numerical validation."""
    var shape = List[Int]()
    shape.append(2)
    var x = zeros(shape, DType.float32)
    var alpha = zeros(shape, DType.float32)

    x._data.bitcast[Float32]()[0] = -1.0
    x._data.bitcast[Float32]()[1] = 1.0

    alpha._data.bitcast[Float32]()[0] = 0.5
    alpha._data.bitcast[Float32]()[1] = 0.5

    # Forward function wrapper
    fn forward(x: ExTensor) raises escaping -> ExTensor:
        return prelu(x, alpha)

    var y = prelu(x, alpha)
    var grad_out = ones_like(y)

    # Validate gradient w.r.t. input using numerical checking
    fn backward_input(grad: ExTensor, x: ExTensor) raises escaping -> ExTensor:
        var result = prelu_backward(grad, x, alpha)
        return result.grad_a

    # Note: rtol=1e-3 is appropriate for float32 finite differences
    check_gradient(forward, backward_input, x, grad_out, rtol=1e-3, atol=1e-6)


# ============================================================================
# Sigmoid Tests
# ============================================================================


fn test_sigmoid_basic() raises:
    """Test sigmoid with known values."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)

    x._data.bitcast[Float32]()[0] = -100.0  # Should be ~0
    x._data.bitcast[Float32]()[1] = 0.0  # Should be 0.5
    x._data.bitcast[Float32]()[2] = 100.0  # Should be ~1

    var y = sigmoid(x)

    assert_almost_equal(
        y._data.bitcast[Float32]()[0], Float32(0.0), tolerance=1e-3
    )
    assert_almost_equal(
        y._data.bitcast[Float32]()[1], Float32(0.5), tolerance=1e-5
    )
    assert_almost_equal(
        y._data.bitcast[Float32]()[2], Float32(1.0), tolerance=1e-3
    )


fn test_sigmoid_backward() raises:
    """Test sigmoid gradient with numerical validation."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)

    # Use multiple test points for better coverage
    x._data.bitcast[Float32]()[0] = -1.0
    x._data.bitcast[Float32]()[1] = 0.0
    x._data.bitcast[Float32]()[2] = 1.0

    # Forward function wrapper
    fn forward(x: ExTensor) raises escaping -> ExTensor:
        return sigmoid(x)

    var y = sigmoid(x)
    var grad_out = ones_like(y)

    # Note: sigmoid_backward takes output y, not input x
    fn backward_fn(grad: ExTensor, x: ExTensor) raises escaping -> ExTensor:
        var out = sigmoid(x)  # Recompute output inside wrapper
        return sigmoid_backward(grad, out)

    # Use numerical gradient checking (gold standard)
    # Note: rtol=1e-3 is appropriate for float32 finite differences
    check_gradient(forward, backward_fn, x, grad_out, rtol=1e-3, atol=1e-6)


fn test_sigmoid_range() raises:
    """Test sigmoid output is in (0, 1)."""
    var shape = List[Int]()
    shape.append(5)
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


fn test_sigmoid_numerical_stability() raises:
    """Test sigmoid with extreme values."""
    var shape = List[Int]()
    shape.append(4)
    var x = zeros(shape, DType.float32)
    x._data.bitcast[Float32]()[0] = -100.0
    x._data.bitcast[Float32]()[1] = -20.0
    x._data.bitcast[Float32]()[2] = 20.0
    x._data.bitcast[Float32]()[3] = 100.0

    var y = sigmoid(x)

    # Large negative values should be close to 0
    assert_true(y._data.bitcast[Float32]()[0] < 1e-6)
    assert_true(y._data.bitcast[Float32]()[1] < 1e-6)

    # Large positive values should be close to 1
    assert_true(y._data.bitcast[Float32]()[2] > 0.999999)
    assert_true(y._data.bitcast[Float32]()[3] > 0.999999)


fn test_sigmoid_float16() raises:
    """Test sigmoid with float16."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float16)
    x._data.bitcast[Float16]()[0] = Float16(-1.0)
    x._data.bitcast[Float16]()[1] = Float16(0.0)
    x._data.bitcast[Float16]()[2] = Float16(1.0)

    var y = sigmoid(x)

    # Check sigmoid(0) = 0.5
    var val_0 = Float32(y._data.bitcast[Float16]()[1])
    assert_almost_equal(val_0, Float32(0.5), tolerance=0.01)

    # Check range (0, 1)
    for i in range(3):
        var val = Float32(y._data.bitcast[Float16]()[i])
        assert_true(val > 0.0 and val < 1.0)


fn test_sigmoid_float64() raises:
    """Test sigmoid with float64 dtype."""
    var shape = List[Int]()
    shape.append(1)
    var x = zeros(shape, DType.float64)

    x._data.bitcast[Float64]()[0] = 0.0

    var y = sigmoid(x)

    assert_almost_equal(y._data.bitcast[Float64]()[0], 0.5, tolerance=1e-10)


# ============================================================================
# Tanh Tests
# ============================================================================


fn test_tanh_basic() raises:
    """Test tanh with known values."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)

    x._data.bitcast[Float32]()[0] = -100.0  # Should be ~-1
    x._data.bitcast[Float32]()[1] = 0.0  # Should be 0
    x._data.bitcast[Float32]()[2] = 100.0  # Should be ~1

    var y = tanh(x)

    assert_almost_equal(
        y._data.bitcast[Float32]()[0], Float32(-1.0), tolerance=1e-3
    )
    assert_almost_equal(
        y._data.bitcast[Float32]()[1], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        y._data.bitcast[Float32]()[2], Float32(1.0), tolerance=1e-3
    )


fn test_tanh_values() raises:
    """Test tanh against known values."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float64)
    x._data.bitcast[Float64]()[0] = 0.0
    x._data.bitcast[Float64]()[1] = 1.0
    x._data.bitcast[Float64]()[2] = -1.0

    var y = tanh(x)

    # tanh(0) = 0
    assert_almost_equal(y._data.bitcast[Float64]()[0], 0.0, tolerance=1e-10)

    # tanh(1) ≈ 0.7616
    var expected_tanh_1 = math_tanh(1.0)
    assert_almost_equal(
        y._data.bitcast[Float64]()[1], expected_tanh_1, tolerance=1e-10
    )

    # tanh(-1) ≈ -0.7616
    assert_almost_equal(
        y._data.bitcast[Float64]()[2], -expected_tanh_1, tolerance=1e-10
    )


fn test_tanh_backward() raises:
    """Test tanh gradient with numerical validation."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)

    # Use multiple test points for better coverage
    x._data.bitcast[Float32]()[0] = -1.0
    x._data.bitcast[Float32]()[1] = 0.0
    x._data.bitcast[Float32]()[2] = 1.0

    # Forward function wrapper
    fn forward(x: ExTensor) raises escaping -> ExTensor:
        return tanh(x)

    var y = tanh(x)
    var grad_out = ones_like(y)

    # Note: tanh_backward takes output y, not input x
    fn backward_fn(grad: ExTensor, x: ExTensor) raises escaping -> ExTensor:
        var out = tanh(x)  # Recompute output inside wrapper
        return tanh_backward(grad, out)

    # Use numerical gradient checking (gold standard)
    # Note: rtol=1e-3 is appropriate for float32 finite differences
    check_gradient(forward, backward_fn, x, grad_out, rtol=1e-3, atol=1e-6)


fn test_tanh_range() raises:
    """Test tanh output is in (-1, 1)."""
    var shape = List[Int]()
    shape.append(5)
    var x = zeros(shape, DType.float32)

    x._data.bitcast[Float32]()[0] = -10.0
    x._data.bitcast[Float32]()[1] = -1.0
    x._data.bitcast[Float32]()[2] = 0.0
    x._data.bitcast[Float32]()[3] = 1.0
    x._data.bitcast[Float32]()[4] = 10.0

    var y = tanh(x)

    # All values should be in [-1, 1] (inclusive due to floating point precision)
    for i in range(5):
        var val = y._data.bitcast[Float32]()[i]
        assert_true(val >= -1.0, "tanh output should be >= -1.0")
        assert_true(val <= 1.0, "tanh output should be <= 1.0")


# ============================================================================
# Softmax Tests
# ============================================================================


fn test_softmax_basic_2d() raises:
    """Test softmax 2D normalization."""
    var shape = List[Int]()
    shape.append(1)
    shape.append(3)
    var x = zeros(shape, DType.float32)

    # All zeros should give uniform distribution
    var y = softmax(x, axis=1)

    # Sum should be 1.0
    var sum = (
        y._data.bitcast[Float32]()[0]
        + y._data.bitcast[Float32]()[1]
        + y._data.bitcast[Float32]()[2]
    )
    assert_almost_equal(sum, Float32(1.0), tolerance=1e-5)

    # Each value should be ~1/3
    assert_almost_equal(
        y._data.bitcast[Float32]()[0], Float32(0.333333), tolerance=1e-3
    )
    assert_almost_equal(
        y._data.bitcast[Float32]()[1], Float32(0.333333), tolerance=1e-3
    )
    assert_almost_equal(
        y._data.bitcast[Float32]()[2], Float32(0.333333), tolerance=1e-3
    )


fn test_softmax_one_hot() raises:
    """Test softmax with large difference (one-hot-like)."""
    var shape = List[Int]()
    shape.append(1)
    shape.append(3)
    var x = zeros(shape, DType.float32)

    x._data.bitcast[Float32]()[0] = 0.0
    x._data.bitcast[Float32]()[1] = 10.0
    x._data.bitcast[Float32]()[2] = 0.0

    var y = softmax(x, axis=1)

    # Middle value should be ~1.0, others ~0.0
    assert_almost_equal(
        y._data.bitcast[Float32]()[0], Float32(0.0), tolerance=1e-3
    )
    assert_almost_equal(
        y._data.bitcast[Float32]()[1], Float32(1.0), tolerance=1e-3
    )
    assert_almost_equal(
        y._data.bitcast[Float32]()[2], Float32(0.0), tolerance=1e-3
    )


fn test_softmax_sum_to_one() raises:
    """Test softmax probabilities sum to 1."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(4)
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


fn test_softmax_numerical_stability() raises:
    """Test softmax with large values (numerical stability)."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)
    x._data.bitcast[Float32]()[0] = 1000.0
    x._data.bitcast[Float32]()[1] = 1001.0
    x._data.bitcast[Float32]()[2] = 1002.0

    var y = softmax(x, axis=-1)

    # Should still sum to 1 (no overflow)
    var sum: Float32 = 0.0
    for i in range(3):
        sum += y._data.bitcast[Float32]()[i]

    assert_almost_equal(sum, Float32(1.0), tolerance=1e-5)

    # Largest value should have largest probability
    assert_true(y._data.bitcast[Float32]()[2] > y._data.bitcast[Float32]()[1])
    assert_true(y._data.bitcast[Float32]()[1] > y._data.bitcast[Float32]()[0])


fn test_softmax_backward() raises:
    """Test softmax gradient with numerical validation."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    var x = zeros(shape, DType.float32)

    # Set test values
    x._data.bitcast[Float32]()[0] = -1.0
    x._data.bitcast[Float32]()[1] = 0.0
    x._data.bitcast[Float32]()[2] = 1.0
    x._data.bitcast[Float32]()[3] = -0.5
    x._data.bitcast[Float32]()[4] = 0.5
    x._data.bitcast[Float32]()[5] = 1.5

    # Forward function wrapper
    fn forward(x: ExTensor) raises escaping -> ExTensor:
        return softmax(x, axis=1)

    var y = softmax(x, axis=1)
    var grad_out = ones_like(y)

    # Note: softmax_backward takes output y, not input x
    fn backward_fn(grad: ExTensor, x: ExTensor) raises escaping -> ExTensor:
        var out = softmax(x, axis=1)  # Recompute output inside wrapper
        return softmax_backward(grad, out, axis=1)

    # Use numerical gradient checking (gold standard)
    # Note: rtol=1e-3, atol=5e-4 is needed for float32 softmax gradients
    # Softmax involves exp() and division which amplify numerical errors,
    # especially at the edges of the distribution
    check_gradient(forward, backward_fn, x, grad_out, rtol=1e-3, atol=5e-4)


# ============================================================================
# GELU Tests
# ============================================================================


fn test_gelu_basic() raises:
    """Test GELU with known value at x=0."""
    var shape = List[Int]()
    shape.append(1)
    var x = zeros(shape, DType.float32)

    x._data.bitcast[Float32]()[0] = 0.0

    var y = gelu(x)

    # GELU(0) = 0
    assert_almost_equal(
        y._data.bitcast[Float32]()[0], Float32(0.0), tolerance=1e-5
    )


fn test_gelu_positive() raises:
    """Test GELU with positive values."""
    var shape = List[Int]()
    shape.append(2)
    var x = zeros(shape, DType.float32)

    x._data.bitcast[Float32]()[0] = 1.0
    x._data.bitcast[Float32]()[1] = 2.0

    var y = gelu(x)

    # For positive x, GELU(x) ≈ x (asymptotically)
    # GELU(1) ≈ 0.84, GELU(2) ≈ 1.96
    assert_true(y._data.bitcast[Float32]()[0] > 0.8)
    assert_true(y._data.bitcast[Float32]()[0] < 1.0)
    assert_true(y._data.bitcast[Float32]()[1] > 1.9)
    assert_true(y._data.bitcast[Float32]()[1] < 2.0)


fn test_gelu_shape() raises:
    """Test GELU preserves shape."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var x = ones(shape, DType.float32)

    var y = gelu(x)

    assert_equal(y.shape()[0], 3)
    assert_equal(y.shape()[1], 4)


fn test_gelu_approximate() raises:
    """Test GELU with tanh approximation."""
    var shape = List[Int]()
    shape.append(5)
    var x = zeros(shape, DType.float32)
    x._data.bitcast[Float32]()[0] = -2.0
    x._data.bitcast[Float32]()[1] = -1.0
    x._data.bitcast[Float32]()[2] = 0.0
    x._data.bitcast[Float32]()[3] = 1.0
    x._data.bitcast[Float32]()[4] = 2.0

    var y = gelu(x, approximate=True)

    # GELU(0) should be 0
    assert_almost_equal(
        y._data.bitcast[Float32]()[2], Float32(0.0), tolerance=1e-5
    )

    # GELU is NOT symmetric (unlike relu). For x < 0, GELU(x) is close to 0.
    # For x > 0, GELU(x) is close to x.
    var val_neg2 = y._data.bitcast[Float32]()[0]  # GELU(-2.0) ≈ -0.045
    var val_pos2 = y._data.bitcast[Float32]()[4]  # GELU(2.0) ≈ 1.954

    # For large positive x, GELU(x) ≈ x
    assert_true(val_pos2 > 1.9, "GELU(2.0) should be close to 2.0")

    # For large negative x, GELU(x) ≈ 0 (small negative value)
    assert_true(abs(val_neg2) < 0.1, "GELU(-2.0) should be close to 0")


fn test_gelu_exact() raises:
    """Test GELU with exact erf implementation."""
    var shape = List[Int]()
    shape.append(5)
    var x = zeros(shape, DType.float32)
    x._data.bitcast[Float32]()[0] = -2.0
    x._data.bitcast[Float32]()[1] = -1.0
    x._data.bitcast[Float32]()[2] = 0.0
    x._data.bitcast[Float32]()[3] = 1.0
    x._data.bitcast[Float32]()[4] = 2.0

    var y = gelu(x, approximate=False)

    # GELU(0) = 0
    assert_almost_equal(
        y._data.bitcast[Float32]()[2], Float32(0.0), tolerance=1e-5
    )

    # For large positive x, GELU(x) ≈ x
    assert_true(y._data.bitcast[Float32]()[4] > 1.9)

    # For large negative x, GELU(x) ≈ 0
    assert_true(abs(y._data.bitcast[Float32]()[0]) < 0.1)


fn test_gelu_comparison() raises:
    """Compare approximate and exact GELU implementations."""
    var shape = List[Int]()
    shape.append(5)
    var x = zeros(shape, DType.float32)
    x._data.bitcast[Float32]()[0] = -2.0
    x._data.bitcast[Float32]()[1] = -1.0
    x._data.bitcast[Float32]()[2] = 0.0
    x._data.bitcast[Float32]()[3] = 1.0
    x._data.bitcast[Float32]()[4] = 2.0

    var y_approx = gelu(x, approximate=True)
    var y_exact = gelu(x, approximate=False)

    # Approximate and exact should be close
    for i in range(5):
        var approx_val = y_approx._data.bitcast[Float32]()[i]
        var exact_val = y_exact._data.bitcast[Float32]()[i]
        var diff = abs(approx_val - exact_val)

        # Approximation error should be small (< 1%)
        if abs(exact_val) > 0.01:
            var rel_error = diff / abs(exact_val)
            assert_true(rel_error < 0.01)


fn test_gelu_float16() raises:
    """Test GELU with float16."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float16)
    x._data.bitcast[Float16]()[0] = Float16(-1.0)
    x._data.bitcast[Float16]()[1] = Float16(0.0)
    x._data.bitcast[Float16]()[2] = Float16(1.0)

    var y = gelu(x, approximate=True)

    # GELU(0) should be 0
    var val_0 = Float32(y._data.bitcast[Float16]()[1])
    assert_almost_equal(val_0, Float32(0.0), tolerance=0.01)


fn test_gelu_backward_gradient() raises:
    """Test GELU backward with numerical gradient checking."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)

    # Set non-uniform values
    x._data.bitcast[Float32]()[0] = -0.5
    x._data.bitcast[Float32]()[1] = 0.0
    x._data.bitcast[Float32]()[2] = 0.5

    # Forward function wrapper
    fn forward(x: ExTensor) raises escaping -> ExTensor:
        return gelu(x, approximate=False)

    var y = gelu(x, approximate=False)
    var grad_out = ones_like(y)

    # Backward function wrapper
    fn backward_fn(grad: ExTensor, x: ExTensor) raises escaping -> ExTensor:
        return gelu_backward(grad, x, approximate=False)

    # Use numerical gradient checking (gold standard)
    check_gradient(forward, backward_fn, x, grad_out, rtol=1e-3, atol=1e-6)


# ============================================================================
# Swish Tests
# ============================================================================


fn test_swish_basic() raises:
    """Test swish with known values."""
    var shape = List[Int]()
    shape.append(1)
    var x = zeros(shape, DType.float32)

    x._data.bitcast[Float32]()[0] = 0.0

    var y = swish(x)

    # swish(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
    assert_almost_equal(
        y._data.bitcast[Float32]()[0], Float32(0.0), tolerance=1e-5
    )


fn test_swish_positive() raises:
    """Test swish with large positive value."""
    var shape = List[Int]()
    shape.append(1)
    var x = zeros(shape, DType.float32)

    x._data.bitcast[Float32]()[0] = 10.0

    var y = swish(x)

    # swish(10) ≈ 10 * sigmoid(10) ≈ 10 * 1 ≈ 10
    assert_almost_equal(
        y._data.bitcast[Float32]()[0], Float32(10.0), tolerance=0.01
    )


fn test_swish_backward_gradient() raises:
    """Test Swish backward with numerical gradient checking."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)

    # Set non-uniform values
    x._data.bitcast[Float32]()[0] = -0.5
    x._data.bitcast[Float32]()[1] = 0.0
    x._data.bitcast[Float32]()[2] = 0.5

    # Forward function wrapper
    fn forward(x: ExTensor) raises escaping -> ExTensor:
        return swish(x)

    var y = swish(x)
    var grad_out = ones_like(y)

    # Backward function wrapper
    fn backward_fn(grad: ExTensor, x: ExTensor) raises escaping -> ExTensor:
        return swish_backward(grad, x)

    # Use numerical gradient checking (gold standard)
    check_gradient(forward, backward_fn, x, grad_out, rtol=1e-3, atol=1e-6)


# ============================================================================
# Mish Tests
# ============================================================================


fn test_mish_basic() raises:
    """Test mish with known values."""
    var shape = List[Int]()
    shape.append(1)
    var x = zeros(shape, DType.float32)

    x._data.bitcast[Float32]()[0] = 0.0

    var y = mish(x)

    # mish(0) = 0 * tanh(softplus(0)) = 0 * tanh(log(2)) ≈ 0
    assert_almost_equal(
        y._data.bitcast[Float32]()[0], Float32(0.0), tolerance=0.01
    )


fn test_mish_shape() raises:
    """Test mish preserves shape."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)
    var x = ones(shape, DType.float32)

    var y = mish(x)

    assert_equal(y.shape()[0], 2)
    assert_equal(y.shape()[1], 3)
    assert_equal(y.shape()[2], 4)


fn test_mish_backward_gradient() raises:
    """Test Mish backward with numerical gradient checking."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)

    # Set non-uniform values
    x._data.bitcast[Float32]()[0] = -0.5
    x._data.bitcast[Float32]()[1] = 0.0
    x._data.bitcast[Float32]()[2] = 0.5

    # Forward function wrapper
    fn forward(x: ExTensor) raises escaping -> ExTensor:
        return mish(x)

    var y = mish(x)
    var grad_out = ones_like(y)

    # Backward function wrapper
    fn backward_fn(grad: ExTensor, x: ExTensor) raises escaping -> ExTensor:
        return mish_backward(grad, x)

    # Use numerical gradient checking (gold standard)
    check_gradient(forward, backward_fn, x, grad_out, rtol=1e-3, atol=1e-6)


# ============================================================================
# ELU Tests
# ============================================================================


fn test_elu_basic() raises:
    """Test ELU with known values."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)

    x._data.bitcast[Float32]()[0] = -1.0
    x._data.bitcast[Float32]()[1] = 0.0
    x._data.bitcast[Float32]()[2] = 1.0

    var y = elu(x, alpha=1.0)

    # ELU(-1) = 1.0 * (exp(-1) - 1) ≈ -0.632
    # ELU(0) = 0
    # ELU(1) = 1
    assert_almost_equal(
        y._data.bitcast[Float32]()[0], Float32(-0.632), tolerance=0.01
    )
    assert_almost_equal(
        y._data.bitcast[Float32]()[1], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        y._data.bitcast[Float32]()[2], Float32(1.0), tolerance=1e-5
    )


fn test_elu_backward() raises:
    """Test ELU gradient with numerical validation."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)

    x._data.bitcast[Float32]()[0] = -1.0
    x._data.bitcast[Float32]()[1] = 0.0
    x._data.bitcast[Float32]()[2] = 1.0

    # Forward function wrapper
    fn forward(x: ExTensor) raises escaping -> ExTensor:
        return elu(x, alpha=1.0)

    var y = elu(x, alpha=1.0)
    var grad_out = ones_like(y)

    # Note: elu_backward takes x, y, and alpha
    fn backward_fn(grad: ExTensor, x: ExTensor) raises escaping -> ExTensor:
        return elu_backward(grad, x, alpha=1.0)

    # Use numerical gradient checking (gold standard)
    # Note: rtol=1e-3 is appropriate for float32 finite differences
    check_gradient(forward, backward_fn, x, grad_out, rtol=1e-3, atol=1e-6)


# ============================================================================
# Integration Tests
# ============================================================================


fn test_integration_forward_backward() raises:
    """Integration test: Complete forward and backward pass through activations.

    Simulates a simple neural network layer with:
    - Input -> ReLU -> Sigmoid -> Output
    - Loss gradient flows back through the network.
    """
    # Input data
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)
    x._data.bitcast[Float32]()[0] = -1.0
    x._data.bitcast[Float32]()[1] = 0.5
    x._data.bitcast[Float32]()[2] = 2.0

    # Forward pass: x -> ReLU -> Sigmoid
    var relu_out = relu(x)
    var sigmoid_out = sigmoid(relu_out)

    # Check forward pass values
    # After ReLU: [0, 0.5, 2.0]
    assert_almost_equal(
        relu_out._data.bitcast[Float32]()[0], Float32(0.0), tolerance=0.001
    )
    assert_almost_equal(
        relu_out._data.bitcast[Float32]()[1], Float32(0.5), tolerance=0.001
    )
    assert_almost_equal(
        relu_out._data.bitcast[Float32]()[2], Float32(2.0), tolerance=0.001
    )

    # After Sigmoid: [0.5, sigmoid(0.5), sigmoid(2.0)]
    var sig_0_5 = sigmoid_out._data.bitcast[Float32]()[1]
    var sig_2_0 = sigmoid_out._data.bitcast[Float32]()[2]
    assert_true(sig_0_5 > 0.6 and sig_0_5 < 0.7)
    assert_true(sig_2_0 > 0.8 and sig_2_0 < 0.9)

    # Simulate loss gradient (all ones)
    var grad_loss = ones(shape, DType.float32)

    # Backward pass: Sigmoid <- ReLU <- x
    var grad_sigmoid = sigmoid_backward(grad_loss, sigmoid_out)
    var grad_x = relu_backward(grad_sigmoid, x)

    # Check backward pass values
    # Gradient through ReLU should be 0 at x=-1 (negative input)
    assert_almost_equal(
        grad_x._data.bitcast[Float32]()[0], Float32(0.0), tolerance=0.001
    )

    # Gradients at positive inputs should be non-zero
    assert_true(grad_x._data.bitcast[Float32]()[1] > 0.0)
    assert_true(grad_x._data.bitcast[Float32]()[2] > 0.0)


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run all tests in this file."""
    var total = 0
    var passed = 0
    var failed = 0

    print("\n" + "=" * 70)
    print("Running tests from: test_activations.mojo")
    print("=" * 70 + "\n")

    # test_relu_basic
    total += 1
    try:
        test_relu_basic()
        passed += 1
        print("  ✓ test_relu_basic")
    except e:
        failed += 1
        print("  ✗ test_relu_basic:", e)

    # test_relu_non_negativity
    total += 1
    try:
        test_relu_non_negativity()
        passed += 1
        print("  ✓ test_relu_non_negativity")
    except e:
        failed += 1
        print("  ✗ test_relu_non_negativity:", e)

    # test_relu_backward
    total += 1
    try:
        test_relu_backward()
        passed += 1
        print("  ✓ test_relu_backward")
    except e:
        failed += 1
        print("  ✗ test_relu_backward:", e)

    # test_relu_shape
    total += 1
    try:
        test_relu_shape()
        passed += 1
        print("  ✓ test_relu_shape")
    except e:
        failed += 1
        print("  ✗ test_relu_shape:", e)

    # test_relu_integer_types
    total += 1
    try:
        test_relu_integer_types()
        passed += 1
        print("  ✓ test_relu_integer_types")
    except e:
        failed += 1
        print("  ✗ test_relu_integer_types:", e)

    # test_relu_float64
    total += 1
    try:
        test_relu_float64()
        passed += 1
        print("  ✓ test_relu_float64")
    except e:
        failed += 1
        print("  ✗ test_relu_float64:", e)

    # test_leaky_relu_basic
    total += 1
    try:
        test_leaky_relu_basic()
        passed += 1
        print("  ✓ test_leaky_relu_basic")
    except e:
        failed += 1
        print("  ✗ test_leaky_relu_basic:", e)

    # test_leaky_relu_custom_alpha
    total += 1
    try:
        test_leaky_relu_custom_alpha()
        passed += 1
        print("  ✓ test_leaky_relu_custom_alpha")
    except e:
        failed += 1
        print("  ✗ test_leaky_relu_custom_alpha:", e)

    # test_leaky_relu_backward
    total += 1
    try:
        test_leaky_relu_backward()
        passed += 1
        print("  ✓ test_leaky_relu_backward")
    except e:
        failed += 1
        print("  ✗ test_leaky_relu_backward:", e)

    # test_prelu_basic
    total += 1
    try:
        test_prelu_basic()
        passed += 1
        print("  ✓ test_prelu_basic")
    except e:
        failed += 1
        print("  ✗ test_prelu_basic:", e)

    # test_prelu_scalar_alpha
    total += 1
    try:
        test_prelu_scalar_alpha()
        passed += 1
        print("  ✓ test_prelu_scalar_alpha")
    except e:
        failed += 1
        print("  ✗ test_prelu_scalar_alpha:", e)

    # test_prelu_elementwise_alpha
    total += 1
    try:
        test_prelu_elementwise_alpha()
        passed += 1
        print("  ✓ test_prelu_elementwise_alpha")
    except e:
        failed += 1
        print("  ✗ test_prelu_elementwise_alpha:", e)

    # test_prelu_backward
    total += 1
    try:
        test_prelu_backward()
        passed += 1
        print("  ✓ test_prelu_backward")
    except e:
        failed += 1
        print("  ✗ test_prelu_backward:", e)

    # test_sigmoid_basic
    total += 1
    try:
        test_sigmoid_basic()
        passed += 1
        print("  ✓ test_sigmoid_basic")
    except e:
        failed += 1
        print("  ✗ test_sigmoid_basic:", e)

    # test_sigmoid_backward
    total += 1
    try:
        test_sigmoid_backward()
        passed += 1
        print("  ✓ test_sigmoid_backward")
    except e:
        failed += 1
        print("  ✗ test_sigmoid_backward:", e)

    # test_sigmoid_range
    total += 1
    try:
        test_sigmoid_range()
        passed += 1
        print("  ✓ test_sigmoid_range")
    except e:
        failed += 1
        print("  ✗ test_sigmoid_range:", e)

    # test_sigmoid_numerical_stability
    total += 1
    try:
        test_sigmoid_numerical_stability()
        passed += 1
        print("  ✓ test_sigmoid_numerical_stability")
    except e:
        failed += 1
        print("  ✗ test_sigmoid_numerical_stability:", e)

    # test_sigmoid_float16
    total += 1
    try:
        test_sigmoid_float16()
        passed += 1
        print("  ✓ test_sigmoid_float16")
    except e:
        failed += 1
        print("  ✗ test_sigmoid_float16:", e)

    # test_sigmoid_float64
    total += 1
    try:
        test_sigmoid_float64()
        passed += 1
        print("  ✓ test_sigmoid_float64")
    except e:
        failed += 1
        print("  ✗ test_sigmoid_float64:", e)

    # test_tanh_basic
    total += 1
    try:
        test_tanh_basic()
        passed += 1
        print("  ✓ test_tanh_basic")
    except e:
        failed += 1
        print("  ✗ test_tanh_basic:", e)

    # test_tanh_values
    total += 1
    try:
        test_tanh_values()
        passed += 1
        print("  ✓ test_tanh_values")
    except e:
        failed += 1
        print("  ✗ test_tanh_values:", e)

    # test_tanh_backward
    total += 1
    try:
        test_tanh_backward()
        passed += 1
        print("  ✓ test_tanh_backward")
    except e:
        failed += 1
        print("  ✗ test_tanh_backward:", e)

    # test_tanh_range
    total += 1
    try:
        test_tanh_range()
        passed += 1
        print("  ✓ test_tanh_range")
    except e:
        failed += 1
        print("  ✗ test_tanh_range:", e)

    # test_softmax_basic_2d
    total += 1
    try:
        test_softmax_basic_2d()
        passed += 1
        print("  ✓ test_softmax_basic_2d")
    except e:
        failed += 1
        print("  ✗ test_softmax_basic_2d:", e)

    # test_softmax_one_hot
    total += 1
    try:
        test_softmax_one_hot()
        passed += 1
        print("  ✓ test_softmax_one_hot")
    except e:
        failed += 1
        print("  ✗ test_softmax_one_hot:", e)

    # test_softmax_sum_to_one
    total += 1
    try:
        test_softmax_sum_to_one()
        passed += 1
        print("  ✓ test_softmax_sum_to_one")
    except e:
        failed += 1
        print("  ✗ test_softmax_sum_to_one:", e)

    # test_softmax_numerical_stability
    total += 1
    try:
        test_softmax_numerical_stability()
        passed += 1
        print("  ✓ test_softmax_numerical_stability")
    except e:
        failed += 1
        print("  ✗ test_softmax_numerical_stability:", e)

    # test_softmax_backward
    total += 1
    try:
        test_softmax_backward()
        passed += 1
        print("  ✓ test_softmax_backward")
    except e:
        failed += 1
        print("  ✗ test_softmax_backward:", e)

    # test_gelu_basic
    total += 1
    try:
        test_gelu_basic()
        passed += 1
        print("  ✓ test_gelu_basic")
    except e:
        failed += 1
        print("  ✗ test_gelu_basic:", e)

    # test_gelu_positive
    total += 1
    try:
        test_gelu_positive()
        passed += 1
        print("  ✓ test_gelu_positive")
    except e:
        failed += 1
        print("  ✗ test_gelu_positive:", e)

    # test_gelu_shape
    total += 1
    try:
        test_gelu_shape()
        passed += 1
        print("  ✓ test_gelu_shape")
    except e:
        failed += 1
        print("  ✗ test_gelu_shape:", e)

    # test_gelu_approximate
    total += 1
    try:
        test_gelu_approximate()
        passed += 1
        print("  ✓ test_gelu_approximate")
    except e:
        failed += 1
        print("  ✗ test_gelu_approximate:", e)

    # test_gelu_exact
    total += 1
    try:
        test_gelu_exact()
        passed += 1
        print("  ✓ test_gelu_exact")
    except e:
        failed += 1
        print("  ✗ test_gelu_exact:", e)

    # test_gelu_comparison
    total += 1
    try:
        test_gelu_comparison()
        passed += 1
        print("  ✓ test_gelu_comparison")
    except e:
        failed += 1
        print("  ✗ test_gelu_comparison:", e)

    # test_gelu_float16
    total += 1
    try:
        test_gelu_float16()
        passed += 1
        print("  ✓ test_gelu_float16")
    except e:
        failed += 1
        print("  ✗ test_gelu_float16:", e)

    # test_gelu_backward_gradient
    total += 1
    try:
        test_gelu_backward_gradient()
        passed += 1
        print("  ✓ test_gelu_backward_gradient")
    except e:
        failed += 1
        print("  ✗ test_gelu_backward_gradient:", e)

    # test_swish_basic
    total += 1
    try:
        test_swish_basic()
        passed += 1
        print("  ✓ test_swish_basic")
    except e:
        failed += 1
        print("  ✗ test_swish_basic:", e)

    # test_swish_positive
    total += 1
    try:
        test_swish_positive()
        passed += 1
        print("  ✓ test_swish_positive")
    except e:
        failed += 1
        print("  ✗ test_swish_positive:", e)

    # test_swish_backward_gradient
    total += 1
    try:
        test_swish_backward_gradient()
        passed += 1
        print("  ✓ test_swish_backward_gradient")
    except e:
        failed += 1
        print("  ✗ test_swish_backward_gradient:", e)

    # test_mish_basic
    total += 1
    try:
        test_mish_basic()
        passed += 1
        print("  ✓ test_mish_basic")
    except e:
        failed += 1
        print("  ✗ test_mish_basic:", e)

    # test_mish_shape
    total += 1
    try:
        test_mish_shape()
        passed += 1
        print("  ✓ test_mish_shape")
    except e:
        failed += 1
        print("  ✗ test_mish_shape:", e)

    # test_mish_backward_gradient
    total += 1
    try:
        test_mish_backward_gradient()
        passed += 1
        print("  ✓ test_mish_backward_gradient")
    except e:
        failed += 1
        print("  ✗ test_mish_backward_gradient:", e)

    # test_elu_basic
    total += 1
    try:
        test_elu_basic()
        passed += 1
        print("  ✓ test_elu_basic")
    except e:
        failed += 1
        print("  ✗ test_elu_basic:", e)

    # test_elu_backward
    total += 1
    try:
        test_elu_backward()
        passed += 1
        print("  ✓ test_elu_backward")
    except e:
        failed += 1
        print("  ✗ test_elu_backward:", e)

    # test_integration_forward_backward
    total += 1
    try:
        test_integration_forward_backward()
        passed += 1
        print("  ✓ test_integration_forward_backward")
    except e:
        failed += 1
        print("  ✗ test_integration_forward_backward:", e)

    # Summary
    print("\n" + "=" * 70)
    print("Results:", passed, "/", total, "passed,", failed, "failed")
    print("=" * 70)

    if failed > 0:
        raise Error("Tests failed")
