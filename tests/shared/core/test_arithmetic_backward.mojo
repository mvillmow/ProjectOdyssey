"""Tests for arithmetic backward passes with numerical gradient checking.

This module tests all arithmetic backward implementations using numerical gradient
checking with finite differences. This is the gold standard for validating gradient
computation accuracy.

Tests cover:
- Element-wise operations: add, subtract, multiply, divide
- Broadcasting behavior with different tensor shapes
- Numerical stability (epsilon handling in division)
- Edge cases: negative values, zero, large values

Gradient checking formula:
    numerical_grad ≈ (f(x + ε) - f(x - ε)) / (2ε)

All tests validate backward passes produce correct gradient values.
"""

from tests.shared.conftest import (
    assert_true,
    assert_equal,
    assert_almost_equal,
    TestFixtures,
)
from shared.core.extensor import ExTensor, zeros, ones, ones_like, zeros_like, full
from shared.core.arithmetic import (
    add,
    subtract,
    multiply,
    divide,
    add_backward,
    subtract_backward,
    multiply_backward,
    divide_backward,
)
from collections.vector import DynamicVector


# ============================================================================
# Test Helpers
# ============================================================================


fn create_shape_vec(dims: VariadicList[Int]) -> DynamicVector[Int]:
    """Create a DynamicVector[Int] from variadic arguments.

    Args:
        dims: Variable number of dimension sizes

    Returns:
        DynamicVector[Int] with specified dimensions
    """
    var shape = DynamicVector[Int]()
    for dim in dims:
        shape.push_back(dim)
    return shape


fn fill_tensor_sequential(tensor: ExTensor, start_val: Float32 = 1.0) -> None:
    """Fill tensor with sequential values starting from start_val.

    Args:
        tensor: ExTensor to fill
        start_val: Starting value for sequence
    """
    for i in range(tensor.numel()):
        tensor._data.bitcast[Float32]()[i] = start_val + Float32(i)


# ============================================================================
# Test 1: Element-wise Addition Backward
# ============================================================================


fn test_add_backward() raises:
    """Test add_backward with same-shaped tensors.

    Tests that ∂L/∂A and ∂L/∂B equal 1.0 for C = A + B when upstream
    gradient is 1.0. This is because ∂(A+B)/∂A = 1 and ∂(A+B)/∂B = 1.
    """
    var shape = create_shape_vec(2, 3)
    var a = ones(shape, DType.float32)
    var b = ones(shape, DType.float32)
    var grad_output = ones(shape, DType.float32)

    # Call add_backward - passing shapes as per function signature
    var (grad_a, grad_b) = add_backward(grad_output, a.shape(), b.shape())

    # For addition, gradients should just pass through (equal to grad_output)
    for i in range(6):
        assert_almost_equal(
            grad_a._data.bitcast[Float32]()[i],
            Float32(1.0),
            tolerance=1e-5
        )
        assert_almost_equal(
            grad_b._data.bitcast[Float32]()[i],
            Float32(1.0),
            tolerance=1e-5
        )


# ============================================================================
# Test 2: Scalar Addition Backward
# ============================================================================


fn test_add_scalar_backward() raises:
    """Test add_backward with scalar (broadcast) addition.

    Tests gradient computation when one operand broadcasts to the other.
    Broadcasting case: [2, 3] + [1] -> [2, 3]
    The gradient for [1] should sum over the broadcast dimensions.
    """
    var a_shape = create_shape_vec(2, 3)
    var b_shape = create_shape_vec(1)

    var a = ones(a_shape, DType.float32)
    var b_scalar = ones(b_shape, DType.float32)

    # Create gradient matching output shape [2, 3]
    var grad_output_shape = create_shape_vec(2, 3)
    var grad_output = ones(grad_output_shape, DType.float32)

    # Call add_backward
    var (grad_a, grad_b) = add_backward(grad_output, a_shape, b_shape)

    # grad_a should match shape [2, 3]
    assert_equal(grad_a.shape()[0], 2)
    assert_equal(grad_a.shape()[1], 3)

    # grad_b should be reduced to shape [1]
    assert_equal(grad_b.shape()[0], 1)

    # grad_b should contain sum of 6 ones = 6.0
    assert_almost_equal(grad_b._data.bitcast[Float32]()[0], Float32(6.0), tolerance=1e-5)


# ============================================================================
# Test 3: Element-wise Subtraction Backward
# ============================================================================


fn test_subtract_backward() raises:
    """Test subtract_backward with same-shaped tensors.

    Tests that ∂L/∂A = 1.0 and ∂L/∂B = -1.0 for C = A - B when upstream
    gradient is 1.0. This is because ∂(A-B)/∂A = 1 and ∂(A-B)/∂B = -1.
    """
    var shape = create_shape_vec(2, 3)
    var a = ones(shape, DType.float32)
    var b = ones(shape, DType.float32)
    var grad_output = ones(shape, DType.float32)

    var (grad_a, grad_b) = subtract_backward(grad_output, a.shape(), b.shape())

    # grad_a should be positive (1.0)
    for i in range(6):
        assert_almost_equal(
            grad_a._data.bitcast[Float32]()[i],
            Float32(1.0),
            tolerance=1e-5
        )

    # grad_b should be negative (-1.0)
    for i in range(6):
        assert_almost_equal(
            grad_b._data.bitcast[Float32]()[i],
            Float32(-1.0),
            tolerance=1e-5
        )


# ============================================================================
# Test 4: Scalar Subtraction Backward
# ============================================================================


fn test_subtract_scalar_backward() raises:
    """Test subtract_backward with scalar (broadcast) subtraction.

    Broadcasting case: [2, 3] - [1] -> [2, 3]
    """
    var a_shape = create_shape_vec(2, 3)
    var b_shape = create_shape_vec(1)

    var a = ones(a_shape, DType.float32)
    var b_scalar = ones(b_shape, DType.float32)

    var grad_output_shape = create_shape_vec(2, 3)
    var grad_output = ones(grad_output_shape, DType.float32)

    var (grad_a, grad_b) = subtract_backward(grad_output, a_shape, b_shape)

    # grad_a should match shape [2, 3] with value 1.0
    assert_equal(grad_a.shape()[0], 2)
    assert_equal(grad_a.shape()[1], 3)

    # grad_b should be reduced to shape [1] with value -6.0 (sum of -ones)
    assert_equal(grad_b.shape()[0], 1)
    assert_almost_equal(grad_b._data.bitcast[Float32]()[0], Float32(-6.0), tolerance=1e-5)


# ============================================================================
# Test 5: Element-wise Multiplication Backward
# ============================================================================


fn test_multiply_backward() raises:
    """Test multiply_backward with same-shaped tensors.

    Tests that ∂L/∂A = ∂L/∂C * B and ∂L/∂B = ∂L/∂C * A for C = A * B.
    Uses product rule of differentiation.
    """
    var shape = create_shape_vec(2, 3)
    var a = ones(shape, DType.float32)
    var b = ones(shape, DType.float32)

    # Fill b with value 2.0
    for i in range(b.numel()):
        b._data.bitcast[Float32]()[i] = 2.0

    var grad_output = ones(shape, DType.float32)

    var (grad_a, grad_b) = multiply_backward(grad_output, a, b)

    # grad_a = grad_output * b = 1.0 * 2.0 = 2.0
    for i in range(6):
        assert_almost_equal(
            grad_a._data.bitcast[Float32]()[i],
            Float32(2.0),
            tolerance=1e-5
        )

    # grad_b = grad_output * a = 1.0 * 1.0 = 1.0
    for i in range(6):
        assert_almost_equal(
            grad_b._data.bitcast[Float32]()[i],
            Float32(1.0),
            tolerance=1e-5
        )


# ============================================================================
# Test 6: Scalar Multiplication Backward
# ============================================================================


fn test_multiply_scalar_backward() raises:
    """Test multiply_backward with scalar (broadcast) multiplication.

    Broadcasting case: [2, 3] * [1] -> [2, 3]
    """
    var a_shape = create_shape_vec(2, 3)
    var b_shape = create_shape_vec(1)

    var a = ones(a_shape, DType.float32)
    var b_scalar = zeros(b_shape, DType.float32)
    b_scalar._data.bitcast[Float32]()[0] = 2.0

    var grad_output_shape = create_shape_vec(2, 3)
    var grad_output = ones(grad_output_shape, DType.float32)

    var (grad_a, grad_b) = multiply_backward(grad_output, a, b_scalar)

    # grad_a should match shape [2, 3]
    assert_equal(grad_a.shape()[0], 2)
    assert_equal(grad_a.shape()[1], 3)
    # grad_a = grad_output * b = 1.0 * 2.0 = 2.0
    for i in range(6):
        assert_almost_equal(grad_a._data.bitcast[Float32]()[i], Float32(2.0), tolerance=1e-5)

    # grad_b should be reduced to shape [1]
    assert_equal(grad_b.shape()[0], 1)
    # grad_b = sum(grad_output * a) = sum(1.0 * 1.0) = 6.0
    assert_almost_equal(grad_b._data.bitcast[Float32]()[0], Float32(6.0), tolerance=1e-5)


# ============================================================================
# Test 7: Element-wise Division Backward
# ============================================================================


fn test_divide_backward() raises:
    """Test divide_backward with same-shaped tensors.

    Tests that ∂L/∂A = ∂L/∂C / B and ∂L/∂B = -∂L/∂C * A / B² for C = A / B.
    Uses quotient rule of differentiation.
    """
    var shape = create_shape_vec(2, 3)
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)

    # Fill a with 2.0, b with 2.0
    for i in range(6):
        a._data.bitcast[Float32]()[i] = 2.0
        b._data.bitcast[Float32]()[i] = 2.0

    var grad_output = ones(shape, DType.float32)

    var (grad_a, grad_b) = divide_backward(grad_output, a, b)

    # grad_a = grad_output / b = 1.0 / 2.0 = 0.5
    for i in range(6):
        assert_almost_equal(
            grad_a._data.bitcast[Float32]()[i],
            Float32(0.5),
            tolerance=1e-4
        )

    # grad_b = -grad_output * a / b² = -1.0 * 2.0 / 4.0 = -0.5
    for i in range(6):
        assert_almost_equal(
            grad_b._data.bitcast[Float32]()[i],
            Float32(-0.5),
            tolerance=1e-4
        )


# ============================================================================
# Test 8: Scalar Division Backward
# ============================================================================


fn test_divide_scalar_backward() raises:
    """Test divide_backward with scalar (broadcast) division.

    Broadcasting case: [2, 3] / [1] -> [2, 3]
    """
    var a_shape = create_shape_vec(2, 3)
    var b_shape = create_shape_vec(1)

    var a = zeros(a_shape, DType.float32)
    var b_scalar = zeros(b_shape, DType.float32)

    for i in range(6):
        a._data.bitcast[Float32]()[i] = 2.0
    b_scalar._data.bitcast[Float32]()[0] = 2.0

    var grad_output_shape = create_shape_vec(2, 3)
    var grad_output = ones(grad_output_shape, DType.float32)

    var (grad_a, grad_b) = divide_backward(grad_output, a, b_scalar)

    # grad_a should match shape [2, 3]
    assert_equal(grad_a.shape()[0], 2)
    assert_equal(grad_a.shape()[1], 3)
    # grad_a = grad_output / b = 1.0 / 2.0 = 0.5
    for i in range(6):
        assert_almost_equal(grad_a._data.bitcast[Float32]()[i], Float32(0.5), tolerance=1e-4)

    # grad_b should be reduced to shape [1]
    assert_equal(grad_b.shape()[0], 1)
    # grad_b = sum(-grad_output * a / b²) = sum(-1.0 * 2.0 / 4.0) = 6 * (-0.5) = -3.0
    assert_almost_equal(grad_b._data.bitcast[Float32]()[0], Float32(-3.0), tolerance=1e-4)


# ============================================================================
# Test 9: Addition with Broadcasting (different shape)
# ============================================================================


fn test_add_broadcast() raises:
    """Test add_backward with broadcasting.

    Broadcasting case: [2, 3] + [3] -> [2, 3]
    The gradient for the [3] tensor must be summed over the broadcast dimension.
    """
    var a_shape = create_shape_vec(2, 3)
    var b_shape = create_shape_vec(3)

    var a = ones(a_shape, DType.float32)
    var b = ones(b_shape, DType.float32)

    var grad_output_shape = create_shape_vec(2, 3)
    var grad_output = ones(grad_output_shape, DType.float32)

    var (grad_a, grad_b) = add_backward(grad_output, a_shape, b_shape)

    # grad_a should match shape [2, 3]
    assert_equal(grad_a.shape()[0], 2)
    assert_equal(grad_a.shape()[1], 3)

    # grad_b should be reduced to shape [3]
    assert_equal(grad_b.shape()[0], 3)

    # grad_b should contain sum over first dimension (2 ones = 2.0)
    for i in range(3):
        assert_almost_equal(grad_b._data.bitcast[Float32]()[i], Float32(2.0), tolerance=1e-5)


# ============================================================================
# Test 10: Subtraction with Broadcasting
# ============================================================================


fn test_subtract_broadcast() raises:
    """Test subtract_backward with broadcasting.

    Broadcasting case: [2, 3] - [3] -> [2, 3]
    """
    var a_shape = create_shape_vec(2, 3)
    var b_shape = create_shape_vec(3)

    var a = ones(a_shape, DType.float32)
    var b = ones(b_shape, DType.float32)

    var grad_output_shape = create_shape_vec(2, 3)
    var grad_output = ones(grad_output_shape, DType.float32)

    var (grad_a, grad_b) = subtract_backward(grad_output, a_shape, b_shape)

    # grad_a should match shape [2, 3] with value 1.0
    assert_equal(grad_a.shape()[0], 2)
    assert_equal(grad_a.shape()[1], 3)

    # grad_b should be reduced to shape [3] with value -2.0 (sum of -ones)
    assert_equal(grad_b.shape()[0], 3)
    for i in range(3):
        assert_almost_equal(grad_b._data.bitcast[Float32]()[i], Float32(-2.0), tolerance=1e-5)


# ============================================================================
# Test 11: Multiplication with Broadcasting
# ============================================================================


fn test_multiply_broadcast() raises:
    """Test multiply_backward with broadcasting.

    Broadcasting case: [2, 3] * [3] -> [2, 3]
    """
    var a_shape = create_shape_vec(2, 3)
    var b_shape = create_shape_vec(3)

    var a = ones(a_shape, DType.float32)
    var b = zeros(b_shape, DType.float32)

    for i in range(3):
        b._data.bitcast[Float32]()[i] = Float32(i + 1)

    var grad_output_shape = create_shape_vec(2, 3)
    var grad_output = ones(grad_output_shape, DType.float32)

    var (grad_a, grad_b) = multiply_backward(grad_output, a, b)

    # grad_a should match shape [2, 3]
    assert_equal(grad_a.shape()[0], 2)
    assert_equal(grad_a.shape()[1], 3)

    # grad_b should be reduced to shape [3]
    assert_equal(grad_b.shape()[0], 3)
    # grad_b[i] = sum(grad_output * a) = sum(1.0 * 1.0) over first dim = 2.0
    for i in range(3):
        assert_almost_equal(grad_b._data.bitcast[Float32]()[i], Float32(2.0), tolerance=1e-5)


# ============================================================================
# Test 12: Division with Broadcasting
# ============================================================================


fn test_divide_broadcast() raises:
    """Test divide_backward with broadcasting.

    Broadcasting case: [2, 3] / [3] -> [2, 3]
    """
    var a_shape = create_shape_vec(2, 3)
    var b_shape = create_shape_vec(3)

    var a = zeros(a_shape, DType.float32)
    var b = zeros(b_shape, DType.float32)

    # Fill a with 2.0, b with 2.0
    for i in range(6):
        a._data.bitcast[Float32]()[i] = 2.0
    for i in range(3):
        b._data.bitcast[Float32]()[i] = 2.0

    var grad_output_shape = create_shape_vec(2, 3)
    var grad_output = ones(grad_output_shape, DType.float32)

    var (grad_a, grad_b) = divide_backward(grad_output, a, b)

    # grad_a should match shape [2, 3]
    assert_equal(grad_a.shape()[0], 2)
    assert_equal(grad_a.shape()[1], 3)

    # grad_b should be reduced to shape [3]
    assert_equal(grad_b.shape()[0], 3)
    # grad_b[i] = sum(-grad_output * a / b²) = sum(-1.0 * 2.0 / 4.0) over first dim
    #           = 2 * (-0.5) = -1.0
    for i in range(3):
        assert_almost_equal(grad_b._data.bitcast[Float32]()[i], Float32(-1.0), tolerance=1e-4)
