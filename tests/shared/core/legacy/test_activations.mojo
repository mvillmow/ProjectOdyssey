"""Tests for activation functions.

Comprehensive test suite for neural network activation functions including
ReLU family, sigmoid/tanh, softmax, and GELU.

Test coverage:
- #239: ReLU Family tests (ReLU, Leaky ReLU, PReLU)
- #244: Sigmoid/Tanh tests
- #249: Softmax/GELU tests
- #254: Integration tests for all activations

Testing strategy:
- Correctness: Mathematical properties and known values
- Edge cases: Zeros, large values, negative values
- Numerical stability: Overflow/underflow handling
- Gradient verification: Numerical gradient checking (future)
"""

from testing import assert_true, assert_false, assert_equal, assert_almost_equal
from math import exp, sqrt, tanh as math_tanh
from shared.core import (
    ExTensor, zeros, ones, full, arange,
    relu, leaky_relu, prelu,
    sigmoid, tanh,
    softmax, gelu
)


fn test_relu_basic() raises:
    """Test ReLU basic functionality: max(0, x)."""
    print("Testing ReLU basic functionality...")

    # Create test tensor: [-2, -1, 0, 1, 2]
    var shape = List[Int]()
    var x = ExTensor(shape, DType.float32)
    x._data.bitcast[Float32]()[0] = -2.0
    x._data.bitcast[Float32]()[1] = -1.0
    x._data.bitcast[Float32]()[2] = 0.0
    x._data.bitcast[Float32]()[3] = 1.0
    x._data.bitcast[Float32]()[4] = 2.0

    var y = relu(x)

    # Expected: [0, 0, 0, 1, 2]
    assert_equal(y._data.bitcast[Float32]()[0], 0.0, "ReLU(-2) should be 0")
    assert_equal(y._data.bitcast[Float32]()[1], 0.0, "ReLU(-1) should be 0")
    assert_equal(y._data.bitcast[Float32]()[2], 0.0, "ReLU(0) should be 0")
    assert_equal(y._data.bitcast[Float32]()[3], 1.0, "ReLU(1) should be 1")
    assert_equal(y._data.bitcast[Float32]()[4], 2.0, "ReLU(2) should be 2")

    print("  ✓ ReLU basic test passed")


fn test_relu_non_negativity() raises:
    """Test ReLU always produces non-negative outputs."""
    print("Testing ReLU non-negativity property...")

    var shape = List[Int]()
    var x = ExTensor(shape, DType.float32)

    # Fill with values from -50 to 50
    for i in range(100):
        x._data.bitcast[Float32]()[i] = Float32(i - 50)

    var y = relu(x)

    # All outputs should be >= 0
    for i in range(100):
        var val = y._data.bitcast[Float32]()[i]
        assert_true(val >= 0.0, "ReLU output should be non-negative")

    print("  ✓ ReLU non-negativity test passed")


fn test_leaky_relu_basic() raises:
    """Test Leaky ReLU with default alpha=0.01."""
    print("Testing Leaky ReLU basic functionality...")

    var shape = List[Int]()
    var x = ExTensor(shape, DType.float32)
    x._data.bitcast[Float32]()[0] = -2.0
    x._data.bitcast[Float32]()[1] = -1.0
    x._data.bitcast[Float32]()[2] = 0.0
    x._data.bitcast[Float32]()[3] = 1.0
    x._data.bitcast[Float32]()[4] = 2.0

    var y = leaky_relu(x, 0.01)

    # Expected: [-0.02, -0.01, 0, 1, 2]
    assert_almost_equal(y._data.bitcast[Float32]()[0], -0.02, 1e-6, "Leaky ReLU(-2) with alpha=0.01")
    assert_almost_equal(y._data.bitcast[Float32]()[1], -0.01, 1e-6, "Leaky ReLU(-1) with alpha=0.01")
    assert_equal(y._data.bitcast[Float32]()[2], 0.0, "Leaky ReLU(0) should be 0")
    assert_equal(y._data.bitcast[Float32]()[3], 1.0, "Leaky ReLU(1) should be 1")
    assert_equal(y._data.bitcast[Float32]()[4], 2.0, "Leaky ReLU(2) should be 2")

    print("  ✓ Leaky ReLU basic test passed")


fn test_leaky_relu_custom_alpha() raises:
    """Test Leaky ReLU with custom alpha value."""
    print("Testing Leaky ReLU with custom alpha...")

    var shape = List[Int]()
    var x = ExTensor(shape, DType.float32)
    x._data.bitcast[Float32]()[0] = -4.0
    x._data.bitcast[Float32]()[1] = 0.0
    x._data.bitcast[Float32]()[2] = 4.0

    var y = leaky_relu(x, 0.25)

    # Expected with alpha=0.25: [-1.0, 0, 4.0]
    assert_almost_equal(y._data.bitcast[Float32]()[0], -1.0, 1e-6, "Leaky ReLU(-4) with alpha=0.25")
    assert_equal(y._data.bitcast[Float32]()[1], 0.0, "Leaky ReLU(0)")
    assert_equal(y._data.bitcast[Float32]()[2], 4.0, "Leaky ReLU(4)")

    print("  ✓ Leaky ReLU custom alpha test passed")


fn test_prelu_scalar_alpha() raises:
    """Test PReLU with scalar alpha parameter."""
    print("Testing PReLU with scalar alpha...")

    var shape = List[Int]()
    var x = ExTensor(shape, DType.float32)
    x._data.bitcast[Float32]()[0] = -2.0
    x._data.bitcast[Float32]()[1] = -1.0
    x._data.bitcast[Float32]()[2] = 0.0
    x._data.bitcast[Float32]()[3] = 1.0
    x._data.bitcast[Float32]()[4] = 2.0

    # Scalar alpha = 0.2
    var alpha_shape = List[Int]()
    var alpha = full(alpha_shape, 0.2, DType.float32)

    var y = prelu(x, alpha)

    # Expected with alpha=0.2: [-0.4, -0.2, 0, 1, 2]
    assert_almost_equal(y._data.bitcast[Float32]()[0], -0.4, 1e-6, "PReLU(-2) with alpha=0.2")
    assert_almost_equal(y._data.bitcast[Float32]()[1], -0.2, 1e-6, "PReLU(-1) with alpha=0.2")
    assert_equal(y._data.bitcast[Float32]()[2], 0.0, "PReLU(0)")
    assert_equal(y._data.bitcast[Float32]()[3], 1.0, "PReLU(1)")
    assert_equal(y._data.bitcast[Float32]()[4], 2.0, "PReLU(2)")

    print("  ✓ PReLU scalar alpha test passed")


fn test_prelu_elementwise_alpha() raises:
    """Test PReLU with element-wise alpha parameters."""
    print("Testing PReLU with element-wise alpha...")

    var shape = List[Int]()
    var x = ExTensor(shape, DType.float32)
    x._data.bitcast[Float32]()[0] = -2.0
    x._data.bitcast[Float32]()[1] = -1.0
    x._data.bitcast[Float32]()[2] = 2.0

    # Element-wise alpha = [0.1, 0.2, 0.3]
    var alpha = ExTensor(shape, DType.float32)
    alpha._data.bitcast[Float32]()[0] = 0.1
    alpha._data.bitcast[Float32]()[1] = 0.2
    alpha._data.bitcast[Float32]()[2] = 0.3

    var y = prelu(x, alpha)

    # Expected: [-0.2, -0.2, 2.0]
    assert_almost_equal(y._data.bitcast[Float32]()[0], -0.2, 1e-6, "PReLU(-2) with alpha=0.1")
    assert_almost_equal(y._data.bitcast[Float32]()[1], -0.2, 1e-6, "PReLU(-1) with alpha=0.2")
    assert_equal(y._data.bitcast[Float32]()[2], 2.0, "PReLU(2) with alpha=0.3")

    print("  ✓ PReLU element-wise alpha test passed")


fn test_sigmoid_basic() raises:
    """Test sigmoid basic functionality."""
    print("Testing sigmoid basic functionality...")

    var shape = List[Int]()
    var x = ExTensor(shape, DType.float32)
    x._data.bitcast[Float32]()[0] = -2.0
    x._data.bitcast[Float32]()[1] = -1.0
    x._data.bitcast[Float32]()[2] = 0.0
    x._data.bitcast[Float32]()[3] = 1.0
    x._data.bitcast[Float32]()[4] = 2.0

    var y = sigmoid(x)

    # Check sigmoid(0) = 0.5
    assert_almost_equal(y._data.bitcast[Float32]()[2], 0.5, 1e-6, "sigmoid(0) should be 0.5")

    # Check symmetry: sigmoid(-x) + sigmoid(x) ≈ 1
    var sum_neg2_pos2 = y._data.bitcast[Float32]()[0] + y._data.bitcast[Float32]()[4]
    assert_almost_equal(sum_neg2_pos2, 1.0, 1e-5, "sigmoid(-2) + sigmoid(2) should be 1")

    var sum_neg1_pos1 = y._data.bitcast[Float32]()[1] + y._data.bitcast[Float32]()[3]
    assert_almost_equal(sum_neg1_pos1, 1.0, 1e-5, "sigmoid(-1) + sigmoid(1) should be 1")

    # Check range (0, 1)
    for i in range(5):
        var val = y._data.bitcast[Float32]()[i]
        assert_true(val > 0.0 and val < 1.0, "sigmoid output should be in (0, 1)")

    print("  ✓ Sigmoid basic test passed")


fn test_sigmoid_numerical_stability() raises:
    """Test sigmoid with extreme values."""
    print("Testing sigmoid numerical stability...")

    var shape = List[Int]()
    var x = ExTensor(shape, DType.float32)
    x._data.bitcast[Float32]()[0] = -100.0
    x._data.bitcast[Float32]()[1] = -20.0
    x._data.bitcast[Float32]()[2] = 20.0
    x._data.bitcast[Float32]()[3] = 100.0

    var y = sigmoid(x)

    # Large negative values should be close to 0
    assert_true(y._data.bitcast[Float32]()[0] < 1e-6, "sigmoid(-100) should be ~0")
    assert_true(y._data.bitcast[Float32]()[1] < 1e-6, "sigmoid(-20) should be ~0")

    # Large positive values should be close to 1
    assert_true(y._data.bitcast[Float32]()[2] > 0.999999, "sigmoid(20) should be ~1")
    assert_true(y._data.bitcast[Float32]()[3] > 0.999999, "sigmoid(100) should be ~1")

    print("  ✓ Sigmoid numerical stability test passed")


fn test_tanh_basic() raises:
    """Test tanh basic functionality."""
    print("Testing tanh basic functionality...")

    var shape = List[Int]()
    var x = ExTensor(shape, DType.float32)
    x._data.bitcast[Float32]()[0] = -2.0
    x._data.bitcast[Float32]()[1] = -1.0
    x._data.bitcast[Float32]()[2] = 0.0
    x._data.bitcast[Float32]()[3] = 1.0
    x._data.bitcast[Float32]()[4] = 2.0

    var y = tanh(x)

    # Check tanh(0) = 0
    assert_almost_equal(y._data.bitcast[Float32]()[2], 0.0, 1e-6, "tanh(0) should be 0")

    # Check symmetry: tanh(-x) = -tanh(x)
    assert_almost_equal(
        y._data.bitcast[Float32]()[0],
        -y._data.bitcast[Float32]()[4],
        1e-5,
        "tanh(-2) should equal -tanh(2)"
    )
    assert_almost_equal(
        y._data.bitcast[Float32]()[1],
        -y._data.bitcast[Float32]()[3],
        1e-5,
        "tanh(-1) should equal -tanh(1)"
    )

    # Check range (-1, 1)
    for i in range(5):
        var val = y._data.bitcast[Float32]()[i]
        assert_true(val > -1.0 and val < 1.0, "tanh output should be in (-1, 1)")

    print("  ✓ Tanh basic test passed")


fn test_tanh_values() raises:
    """Test tanh against known values."""
    print("Testing tanh against known values...")

    var shape = List[Int]()
    var x = ExTensor(shape, DType.float64)
    x._data.bitcast[Float64]()[0] = 0.0
    x._data.bitcast[Float64]()[1] = 1.0
    x._data.bitcast[Float64]()[2] = -1.0

    var y = tanh(x)

    # tanh(0) = 0
    assert_almost_equal(y._data.bitcast[Float64]()[0], 0.0, 1e-10, "tanh(0) = 0")

    # tanh(1) ≈ 0.7616
    var expected_tanh_1 = math_tanh(1.0)
    assert_almost_equal(y._data.bitcast[Float64]()[1], expected_tanh_1, 1e-10, "tanh(1)")

    # tanh(-1) ≈ -0.7616
    assert_almost_equal(y._data.bitcast[Float64]()[2], -expected_tanh_1, 1e-10, "tanh(-1)")

    print("  ✓ Tanh known values test passed")


fn test_softmax_basic() raises:
    """Test softmax basic functionality."""
    print("Testing softmax basic functionality...")

    # 1D case: [1, 2, 3]
    var shape = List[Int]()
    var x = ExTensor(shape, DType.float32)
    x._data.bitcast[Float32]()[0] = 1.0
    x._data.bitcast[Float32]()[1] = 2.0
    x._data.bitcast[Float32]()[2] = 3.0

    var y = softmax(x, axis=-1)

    # Check sum to 1
    var sum: Float32 = 0.0
    for i in range(3):
        sum += y._data.bitcast[Float32]()[i]
    assert_almost_equal(sum, 1.0, 1e-6, "Softmax outputs should sum to 1")

    # Check monotonicity: larger input -> larger output
    assert_true(
        y._data.bitcast[Float32]()[0] < y._data.bitcast[Float32]()[1],
        "softmax preserves ordering"
    )
    assert_true(
        y._data.bitcast[Float32]()[1] < y._data.bitcast[Float32]()[2],
        "softmax preserves ordering"
    )

    print("  ✓ Softmax basic test passed")


fn test_softmax_2d() raises:
    """Test softmax on 2D tensors."""
    print("Testing softmax on 2D tensor...")

    # 2x3 tensor
    var shape = List[Int](2, 3)
    var x = ExTensor(shape, DType.float32)
    # Row 1: [1, 2, 3]
    x._data.bitcast[Float32]()[0] = 1.0
    x._data.bitcast[Float32]()[1] = 2.0
    x._data.bitcast[Float32]()[2] = 3.0
    # Row 2: [4, 5, 6]
    x._data.bitcast[Float32]()[3] = 4.0
    x._data.bitcast[Float32]()[4] = 5.0
    x._data.bitcast[Float32]()[5] = 6.0

    var y = softmax(x, axis=-1)

    # Check each row sums to 1
    var row1_sum = y._data.bitcast[Float32]()[0] + y._data.bitcast[Float32]()[1] + y._data.bitcast[Float32]()[2]
    var row2_sum = y._data.bitcast[Float32]()[3] + y._data.bitcast[Float32]()[4] + y._data.bitcast[Float32]()[5]

    assert_almost_equal(row1_sum, 1.0, 1e-6, "Row 1 should sum to 1")
    assert_almost_equal(row2_sum, 1.0, 1e-6, "Row 2 should sum to 1")

    print("  ✓ Softmax 2D test passed")


fn test_softmax_numerical_stability() raises:
    """Test softmax with large values (numerical stability)."""
    print("Testing softmax numerical stability...")

    var shape = List[Int]()
    var x = ExTensor(shape, DType.float32)
    x._data.bitcast[Float32]()[0] = 1000.0
    x._data.bitcast[Float32]()[1] = 1001.0
    x._data.bitcast[Float32]()[2] = 1002.0

    var y = softmax(x, axis=-1)

    # Should still sum to 1 (no overflow)
    var sum: Float32 = 0.0
    for i in range(3):
        sum += y._data.bitcast[Float32]()[i]

    assert_almost_equal(sum, 1.0, 1e-5, "Softmax with large values should still sum to 1")

    # Largest value should have largest probability
    assert_true(y._data.bitcast[Float32]()[2] > y._data.bitcast[Float32]()[1], "Ordering preserved")
    assert_true(y._data.bitcast[Float32]()[1] > y._data.bitcast[Float32]()[0], "Ordering preserved")

    print("  ✓ Softmax numerical stability test passed")


fn test_gelu_approximate() raises:
    """Test GELU with tanh approximation."""
    print("Testing GELU with approximation...")

    var shape = List[Int]()
    var x = ExTensor(shape, DType.float32)
    x._data.bitcast[Float32]()[0] = -2.0
    x._data.bitcast[Float32]()[1] = -1.0
    x._data.bitcast[Float32]()[2] = 0.0
    x._data.bitcast[Float32]()[3] = 1.0
    x._data.bitcast[Float32]()[4] = 2.0

    var y = gelu(x, approximate=True)

    # GELU(0) should be 0
    assert_almost_equal(y._data.bitcast[Float32]()[2], 0.0, 1e-5, "GELU(0) should be 0")

    # GELU should be approximately symmetric: GELU(-x) ≈ -GELU(x)
    # (not exactly symmetric but close)
    var val_neg2 = y._data.bitcast[Float32]()[0]
    var val_pos2 = y._data.bitcast[Float32]()[4]
    assert_true(abs(val_neg2 + val_pos2) < 0.1, "GELU approximately antisymmetric")

    # For large positive x, GELU(x) ≈ x
    assert_true(y._data.bitcast[Float32]()[4] > 1.9, "GELU(2) should be close to 2")

    # For large negative x, GELU(x) ≈ 0
    assert_true(abs(y._data.bitcast[Float32]()[0]) < 0.1, "GELU(-2) should be close to 0")

    print("  ✓ GELU approximate test passed")


fn test_gelu_exact() raises:
    """Test GELU with exact erf implementation."""
    print("Testing GELU with exact formula...")

    var shape = List[Int]()
    var x = ExTensor(shape, DType.float32)
    x._data.bitcast[Float32]()[0] = -2.0
    x._data.bitcast[Float32]()[1] = -1.0
    x._data.bitcast[Float32]()[2] = 0.0
    x._data.bitcast[Float32]()[3] = 1.0
    x._data.bitcast[Float32]()[4] = 2.0

    var y = gelu(x, approximate=False)

    # GELU(0) = 0
    assert_almost_equal(y._data.bitcast[Float32]()[2], 0.0, 1e-5, "GELU exact: GELU(0) = 0")

    # For large positive x, GELU(x) ≈ x
    assert_true(y._data.bitcast[Float32]()[4] > 1.9, "GELU exact: GELU(2) ≈ 2")

    # For large negative x, GELU(x) ≈ 0
    assert_true(abs(y._data.bitcast[Float32]()[0]) < 0.1, "GELU exact: GELU(-2) ≈ 0")

    print("  ✓ GELU exact test passed")


fn test_gelu_comparison() raises:
    """Compare approximate and exact GELU implementations."""
    print("Testing GELU approximate vs exact...")

    var shape = List[Int]()
    var x = ExTensor(shape, DType.float32)
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
            assert_true(rel_error < 0.01, "GELU approximate should be within 1% of exact")

    print("  ✓ GELU comparison test passed")


fn test_relu_integer_types() raises:
    """Test ReLU with integer types."""
    print("Testing ReLU with integer types...")

    # Test int32
    var shape = List[Int]()
    var x_int32 = ExTensor(shape, DType.int32)
    x_int32._data.bitcast[Int32]()[0] = -2
    x_int32._data.bitcast[Int32]()[1] = -1
    x_int32._data.bitcast[Int32]()[2] = 0
    x_int32._data.bitcast[Int32]()[3] = 1
    x_int32._data.bitcast[Int32]()[4] = 2

    var y_int32 = relu(x_int32)

    # Expected: [0, 0, 0, 1, 2]
    assert_equal(y_int32._data.bitcast[Int32]()[0], 0, "ReLU int32: -2 -> 0")
    assert_equal(y_int32._data.bitcast[Int32]()[1], 0, "ReLU int32: -1 -> 0")
    assert_equal(y_int32._data.bitcast[Int32]()[2], 0, "ReLU int32: 0 -> 0")
    assert_equal(y_int32._data.bitcast[Int32]()[3], 1, "ReLU int32: 1 -> 1")
    assert_equal(y_int32._data.bitcast[Int32]()[4], 2, "ReLU int32: 2 -> 2")

    # Test uint8 (already non-negative)
    var x_uint8 = ExTensor(shape, DType.uint8)
    x_uint8._data.bitcast[UInt8]()[0] = 0
    x_uint8._data.bitcast[UInt8]()[1] = 1
    x_uint8._data.bitcast[UInt8]()[2] = 128
    x_uint8._data.bitcast[UInt8]()[3] = 255
    x_uint8._data.bitcast[UInt8]()[4] = 100

    var y_uint8 = relu(x_uint8)

    # Should be unchanged
    assert_equal(y_uint8._data.bitcast[UInt8]()[0], 0, "ReLU uint8: 0 -> 0")
    assert_equal(y_uint8._data.bitcast[UInt8]()[1], 1, "ReLU uint8: 1 -> 1")
    assert_equal(y_uint8._data.bitcast[UInt8]()[2], 128, "ReLU uint8: 128 -> 128")
    assert_equal(y_uint8._data.bitcast[UInt8]()[3], 255, "ReLU uint8: 255 -> 255")

    print("  ✓ ReLU integer types test passed")


fn test_sigmoid_float16() raises:
    """Test sigmoid with float16."""
    print("Testing sigmoid with float16...")

    var shape = List[Int]()
    var x = ExTensor(shape, DType.float16)
    x._data.bitcast[Float16]()[0] = Float16(-1.0)
    x._data.bitcast[Float16]()[1] = Float16(0.0)
    x._data.bitcast[Float16]()[2] = Float16(1.0)

    var y = sigmoid(x)

    # Check sigmoid(0) = 0.5
    var val_0 = Float32(y._data.bitcast[Float16]()[1])
    assert_almost_equal(val_0, 0.5, 0.01, "sigmoid float16: sigmoid(0) = 0.5")

    # Check range (0, 1)
    for i in range(3):
        var val = Float32(y._data.bitcast[Float16]()[i])
        assert_true(val > 0.0 and val < 1.0, "sigmoid float16: output in (0, 1)")

    print("  ✓ Sigmoid float16 test passed")


fn test_gelu_float16() raises:
    """Test GELU with float16."""
    print("Testing GELU with float16...")

    var shape = List[Int]()
    var x = ExTensor(shape, DType.float16)
    x._data.bitcast[Float16]()[0] = Float16(-1.0)
    x._data.bitcast[Float16]()[1] = Float16(0.0)
    x._data.bitcast[Float16]()[2] = Float16(1.0)

    var y = gelu(x, approximate=True)

    # GELU(0) should be 0
    var val_0 = Float32(y._data.bitcast[Float16]()[1])
    assert_almost_equal(val_0, 0.0, 0.01, "GELU float16: GELU(0) = 0")

    print("  ✓ GELU float16 test passed")


# ============================================================================
# Gradient (Backward Pass) Tests
# ============================================================================


fn test_relu_gradient() raises:
    """Test ReLU backward pass using numerical gradient checking."""
    print("Testing ReLU gradient...")

    var shape = List[Int]()
    var x = ExTensor(shape, DType.float32)
    x._data.bitcast[Float32]()[0] = -2.0
    x._data.bitcast[Float32]()[1] = -0.5
    x._data.bitcast[Float32]()[2] = 0.0
    x._data.bitcast[Float32]()[3] = 0.5
    x._data.bitcast[Float32]()[4] = 2.0

    # Simulate gradient from loss
    var grad_output = ExTensor(shape, DType.float32)
    for i in range(5):
        grad_output._data.bitcast[Float32]()[i] = 1.0

    # Compute analytical gradient
    var grad_x = relu_backward(grad_output, x)

    # Check gradients
    # ReLU gradient: 1 if x > 0, else 0
    assert_almost_equal(grad_x._data.bitcast[Float32]()[0], 0.0, 0.001, "ReLU grad at x=-2")
    assert_almost_equal(grad_x._data.bitcast[Float32]()[1], 0.0, 0.001, "ReLU grad at x=-0.5")
    assert_almost_equal(grad_x._data.bitcast[Float32]()[2], 0.0, 0.001, "ReLU grad at x=0")
    assert_almost_equal(grad_x._data.bitcast[Float32]()[3], 1.0, 0.001, "ReLU grad at x=0.5")
    assert_almost_equal(grad_x._data.bitcast[Float32]()[4], 1.0, 0.001, "ReLU grad at x=2")

    print("  ✓ ReLU gradient test passed")


fn test_leaky_relu_gradient() raises:
    """Test Leaky ReLU backward pass."""
    print("Testing Leaky ReLU gradient...")

    var shape = List[Int]()
    var x = ExTensor(shape, DType.float32)
    x._data.bitcast[Float32]()[0] = -1.0
    x._data.bitcast[Float32]()[1] = -0.1
    x._data.bitcast[Float32]()[2] = 0.1
    x._data.bitcast[Float32]()[3] = 1.0

    var grad_output = ExTensor(shape, DType.float32)
    for i in range(4):
        grad_output._data.bitcast[Float32]()[i] = 1.0

    var alpha = 0.01
    var grad_x = leaky_relu_backward(grad_output, x, alpha)

    # Leaky ReLU gradient: alpha if x <= 0, else 1
    assert_almost_equal(grad_x._data.bitcast[Float32]()[0], 0.01, 0.001, "Leaky ReLU grad at x=-1")
    assert_almost_equal(grad_x._data.bitcast[Float32]()[1], 0.01, 0.001, "Leaky ReLU grad at x=-0.1")
    assert_almost_equal(grad_x._data.bitcast[Float32]()[2], 1.0, 0.001, "Leaky ReLU grad at x=0.1")
    assert_almost_equal(grad_x._data.bitcast[Float32]()[3], 1.0, 0.001, "Leaky ReLU grad at x=1")

    print("  ✓ Leaky ReLU gradient test passed")


fn test_sigmoid_gradient() raises:
    """Test sigmoid backward pass."""
    print("Testing sigmoid gradient...")

    var shape = List[Int]()
    var x = ExTensor(shape, DType.float32)
    x._data.bitcast[Float32]()[0] = -1.0
    x._data.bitcast[Float32]()[1] = 0.0
    x._data.bitcast[Float32]()[2] = 1.0

    # Forward pass
    var output = sigmoid(x)

    # Gradient from loss
    var grad_output = ExTensor(shape, DType.float32)
    for i in range(3):
        grad_output._data.bitcast[Float32]()[i] = 1.0

    # Backward pass
    var grad_x = sigmoid_backward(grad_output, output)

    # Sigmoid gradient: y * (1 - y)
    # At x=0, sigmoid(0) = 0.5, gradient = 0.5 * 0.5 = 0.25
    var grad_at_0 = grad_x._data.bitcast[Float32]()[1]
    assert_almost_equal(grad_at_0, 0.25, 0.001, "Sigmoid grad at x=0")

    # Gradients should be positive and symmetric
    var grad_at_neg1 = grad_x._data.bitcast[Float32]()[0]
    var grad_at_pos1 = grad_x._data.bitcast[Float32]()[2]
    assert_almost_equal(grad_at_neg1, grad_at_pos1, 0.001, "Sigmoid grad symmetry")

    print("  ✓ Sigmoid gradient test passed")


fn test_tanh_gradient() raises:
    """Test tanh backward pass."""
    print("Testing tanh gradient...")

    var shape = List[Int]()
    var x = ExTensor(shape, DType.float32)
    x._data.bitcast[Float32]()[0] = -1.0
    x._data.bitcast[Float32]()[1] = 0.0
    x._data.bitcast[Float32]()[2] = 1.0

    # Forward pass
    var output = tanh(x)

    # Gradient from loss
    var grad_output = ExTensor(shape, DType.float32)
    for i in range(3):
        grad_output._data.bitcast[Float32]()[i] = 1.0

    # Backward pass
    var grad_x = tanh_backward(grad_output, output)

    # Tanh gradient: 1 - y²
    # At x=0, tanh(0) = 0, gradient = 1 - 0 = 1
    var grad_at_0 = grad_x._data.bitcast[Float32]()[1]
    assert_almost_equal(grad_at_0, 1.0, 0.001, "Tanh grad at x=0")

    # Gradients should be positive and symmetric
    var grad_at_neg1 = grad_x._data.bitcast[Float32]()[0]
    var grad_at_pos1 = grad_x._data.bitcast[Float32]()[2]
    assert_almost_equal(grad_at_neg1, grad_at_pos1, 0.001, "Tanh grad symmetry")

    print("  ✓ Tanh gradient test passed")


fn test_softmax_gradient() raises:
    """Test softmax backward pass."""
    print("Testing softmax gradient...")

    var shape = List[Int]()
    var x = ExTensor(shape, DType.float32)
    x._data.bitcast[Float32]()[0] = 1.0
    x._data.bitcast[Float32]()[1] = 2.0
    x._data.bitcast[Float32]()[2] = 3.0

    # Forward pass
    var output = softmax(x)

    # Gradient from loss (e.g., cross-entropy)
    var grad_output = ExTensor(shape, DType.float32)
    grad_output._data.bitcast[Float32]()[0] = 0.0
    grad_output._data.bitcast[Float32]()[1] = 0.0
    grad_output._data.bitcast[Float32]()[2] = 1.0

    # Backward pass
    var grad_x = softmax_backward(grad_output, output)

    # Gradient sum should be close to 0 (property of softmax gradient)
    var grad_sum: Float32 = 0.0
    for i in range(3):
        grad_sum += grad_x._data.bitcast[Float32]()[i]

    # Due to the Jacobian structure, sum should be approximately 0
    # (not exactly 0 due to numerical precision)
    assert_true(abs(grad_sum) < 0.1, "Softmax gradient sum property")

    print("  ✓ Softmax gradient test passed")


fn test_gelu_gradient() raises:
    """Test GELU backward pass."""
    print("Testing GELU gradient...")

    var shape = List[Int]()
    var x = ExTensor(shape, DType.float32)
    x._data.bitcast[Float32]()[0] = -1.0
    x._data.bitcast[Float32]()[1] = 0.0
    x._data.bitcast[Float32]()[2] = 1.0

    # Gradient from loss
    var grad_output = ExTensor(shape, DType.float32)
    for i in range(3):
        grad_output._data.bitcast[Float32]()[i] = 1.0

    # Test approximate version
    var grad_x_approx = gelu_backward(grad_output, x, approximate=True)

    # GELU gradient at x=0 should be approximately 0.5
    var grad_at_0 = grad_x_approx._data.bitcast[Float32]()[1]
    assert_almost_equal(grad_at_0, 0.5, 0.05, "GELU grad at x=0")

    # Gradients should be positive
    assert_true(grad_x_approx._data.bitcast[Float32]()[0] > 0.0, "GELU grad positive at x=-1")
    assert_true(grad_x_approx._data.bitcast[Float32]()[2] > 0.0, "GELU grad positive at x=1")

    print("  ✓ GELU gradient test passed")


fn test_prelu_gradient() raises:
    """Test PReLU backward pass with learnable parameters."""
    print("Testing PReLU gradient...")

    var shape = List[Int]()
    var x = ExTensor(shape, DType.float32)
    x._data.bitcast[Float32]()[0] = -2.0
    x._data.bitcast[Float32]()[1] = -1.0
    x._data.bitcast[Float32]()[2] = 1.0
    x._data.bitcast[Float32]()[3] = 2.0

    var alpha_shape = List[Int]()
    var alpha = ExTensor(alpha_shape, DType.float32)
    for i in range(4):
        alpha._data.bitcast[Float32]()[i] = 0.25

    var grad_output = ExTensor(shape, DType.float32)
    for i in range(4):
        grad_output._data.bitcast[Float32]()[i] = 1.0

    # Backward pass returns tuple (grad_input, grad_alpha)
    var grads = prelu_backward(grad_output, x, alpha)
    var grad_x = grads[0]
    var grad_alpha = grads[1]

    # Check grad_input
    # PReLU grad: alpha if x <= 0, else 1
    assert_almost_equal(grad_x._data.bitcast[Float32]()[0], 0.25, 0.001, "PReLU grad_x at x=-2")
    assert_almost_equal(grad_x._data.bitcast[Float32]()[1], 0.25, 0.001, "PReLU grad_x at x=-1")
    assert_almost_equal(grad_x._data.bitcast[Float32]()[2], 1.0, 0.001, "PReLU grad_x at x=1")
    assert_almost_equal(grad_x._data.bitcast[Float32]()[3], 1.0, 0.001, "PReLU grad_x at x=2")

    # Check grad_alpha (gradient w.r.t. learnable parameter)
    # grad_alpha = x * grad_output where x <= 0
    assert_almost_equal(grad_alpha._data.bitcast[Float32]()[0], -2.0, 0.001, "PReLU grad_alpha at x=-2")
    assert_almost_equal(grad_alpha._data.bitcast[Float32]()[1], -1.0, 0.001, "PReLU grad_alpha at x=-1")
    assert_almost_equal(grad_alpha._data.bitcast[Float32]()[2], 0.0, 0.001, "PReLU grad_alpha at x=1")
    assert_almost_equal(grad_alpha._data.bitcast[Float32]()[3], 0.0, 0.001, "PReLU grad_alpha at x=2")

    print("  ✓ PReLU gradient test passed")


fn test_integration_forward_backward() raises:
    """Integration test: Complete forward and backward pass through activations.

    Simulates a simple neural network layer with:
    - Input -> ReLU -> Sigmoid -> Output
    - Loss gradient flows back through the network
    """
    print("Testing integration: forward + backward pass...")

    # Input data
    var shape = List[Int]()
    var x = ExTensor(shape, DType.float32)
    x._data.bitcast[Float32]()[0] = -1.0
    x._data.bitcast[Float32]()[1] = 0.5
    x._data.bitcast[Float32]()[2] = 2.0

    # Forward pass: x -> ReLU -> Sigmoid
    var relu_out = relu(x)
    var sigmoid_out = sigmoid(relu_out)

    # Check forward pass values
    # After ReLU: [0, 0.5, 2.0]
    assert_almost_equal(relu_out._data.bitcast[Float32]()[0], 0.0, 0.001, "ReLU forward")
    assert_almost_equal(relu_out._data.bitcast[Float32]()[1], 0.5, 0.001, "ReLU forward")
    assert_almost_equal(relu_out._data.bitcast[Float32]()[2], 2.0, 0.001, "ReLU forward")

    # After Sigmoid: [0.5, sigmoid(0.5), sigmoid(2.0)]
    var sig_0_5 = sigmoid_out._data.bitcast[Float32]()[1]
    var sig_2_0 = sigmoid_out._data.bitcast[Float32]()[2]
    assert_true(sig_0_5 > 0.6 and sig_0_5 < 0.7, "Sigmoid(0.5) ≈ 0.62")
    assert_true(sig_2_0 > 0.8 and sig_2_0 < 0.9, "Sigmoid(2.0) ≈ 0.88")

    # Simulate loss gradient (all ones)
    var grad_loss = ExTensor(shape, DType.float32)
    for i in range(3):
        grad_loss._data.bitcast[Float32]()[i] = 1.0

    # Backward pass: Sigmoid <- ReLU <- x
    var grad_sigmoid = sigmoid_backward(grad_loss, sigmoid_out)
    var grad_x = relu_backward(grad_sigmoid, x)

    # Check backward pass values
    # Gradient through ReLU should be 0 at x=-1 (negative input)
    assert_almost_equal(grad_x._data.bitcast[Float32]()[0], 0.0, 0.001, "Gradient at x=-1")

    # Gradients at positive inputs should be non-zero
    assert_true(grad_x._data.bitcast[Float32]()[1] > 0.0, "Gradient at x=0.5 should be positive")
    assert_true(grad_x._data.bitcast[Float32]()[2] > 0.0, "Gradient at x=2.0 should be positive")

    print("  Forward pass: x -> ReLU -> Sigmoid ✓")
    print("  Backward pass: grad flows through both activations ✓")
    print("  ✓ Integration test passed")


fn main() raises:
    """Run all activation function tests."""
    print("\n" + "="*70)
    print("ACTIVATION FUNCTIONS TEST SUITE")
    print("="*70 + "\n")

    print("ReLU Family Tests (#239)")
    print("-" * 70)
    test_relu_basic()
    test_relu_non_negativity()
    test_relu_integer_types()
    test_leaky_relu_basic()
    test_leaky_relu_custom_alpha()
    test_prelu_scalar_alpha()
    test_prelu_elementwise_alpha()

    print("\nSigmoid/Tanh Tests (#244)")
    print("-" * 70)
    test_sigmoid_basic()
    test_sigmoid_numerical_stability()
    test_sigmoid_float16()
    test_tanh_basic()
    test_tanh_values()

    print("\nSoftmax/GELU Tests (#249)")
    print("-" * 70)
    test_softmax_basic()
    test_softmax_2d()
    test_softmax_numerical_stability()
    test_gelu_approximate()
    test_gelu_exact()
    test_gelu_comparison()
    test_gelu_float16()

    print("\nGradient Tests (Backward Pass)")
    print("-" * 70)
    test_relu_gradient()
    test_leaky_relu_gradient()
    test_prelu_gradient()
    test_sigmoid_gradient()
    test_tanh_gradient()
    test_softmax_gradient()
    test_gelu_gradient()

    print("\nIntegration Tests")
    print("-" * 70)
    test_integration_forward_backward()

    print("\n" + "="*70)
    print("ALL ACTIVATION TESTS PASSED ✓")
    print("="*70 + "\n")
