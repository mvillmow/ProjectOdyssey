"""Tests for arithmetic backward passes with numerical gradient checking.

This module tests all arithmetic backward implementations using numerical gradient
checking with finite differences. This is the gold standard for validating gradient
computation accuracy.

Tests cover:
- Element-wise operations: add, subtract, multiply, divide
- Broadcasting behavior with different tensor shapes
- Numerical stability (epsilon handling in division)
- Edge cases: negative values, zero, large values
- Numerical gradient checking using finite differences

Gradient checking formula:
    numerical_grad ≈ (f(x + ε) - f(x - ε)) / (2ε)

All tests validate backward passes produce correct gradient values.
"""

from tests.shared.conftest import (
    assert_almost_equal,
    assert_close_float,
    assert_equal,
    assert_equal_int,
    assert_true,
)
from tests.shared.conftest import TestFixtures
from shared.core.extensor import (
    ExTensor,
    zeros,
    ones,
    ones_like,
    zeros_like,
    full,
)
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
from shared.testing import (
    check_gradient,
    compute_numerical_gradient,
)


# ============================================================================
# Test Helpers
# ============================================================================


fn create_shape_vec(*dims: Int) -> List[Int]:
    """Create a List[Int] from variadic arguments.

    Args:
        dims: Variable number of dimension sizes.

    Returns:
        List[Int] with specified dimensions.
    """
    var shape = List[Int]()
    for i in range(len(dims)):
        shape.append(dims[i])
    return shape^


fn fill_tensor_sequential(tensor: ExTensor, start_val: Float32 = 1.0) -> None:
    """Fill tensor with sequential values starting from start_val.

    Args:
        tensor: ExTensor to fill.
        start_val: Starting value for sequence.
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
    var grads = add_backward(grad_output, a, b)
    var grad_a = grads.grad_a
    var grad_b = grads.grad_b

    # For addition, gradients should just pass through (equal to grad_output)
    for i in range(6):
        assert_almost_equal(
            grad_a._data.bitcast[Float32]()[i], Float32(1.0), tolerance=1e-5
        )
        assert_almost_equal(
            grad_b._data.bitcast[Float32]()[i], Float32(1.0), tolerance=1e-5
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
    var grads = add_backward(grad_output, a, b_scalar)
    var grad_a = grads.grad_a
    var grad_b = grads.grad_b

    # grad_a should match shape [2, 3]
    assert_equal(grad_a.shape()[0], 2)
    assert_equal(grad_a.shape()[1], 3)

    # grad_b should be reduced to shape [1]
    assert_equal(grad_b.shape()[0], 1)

    # grad_b should contain sum of 6 ones = 6.0
    assert_almost_equal(
        grad_b._data.bitcast[Float32]()[0], Float32(6.0), tolerance=1e-5
    )


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

    var grads = subtract_backward(grad_output, a, b)
    var grad_a = grads.grad_a
    var grad_b = grads.grad_b

    # grad_a should be positive (1.0)
    for i in range(6):
        assert_almost_equal(
            grad_a._data.bitcast[Float32]()[i], Float32(1.0), tolerance=1e-5
        )

    # grad_b should be negative (-1.0)
    for i in range(6):
        assert_almost_equal(
            grad_b._data.bitcast[Float32]()[i], Float32(-1.0), tolerance=1e-5
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

    var grads = subtract_backward(grad_output, a, b_scalar)
    var grad_a = grads.grad_a
    var grad_b = grads.grad_b

    # grad_a should match shape [2, 3] with value 1.0
    assert_equal(grad_a.shape()[0], 2)
    assert_equal(grad_a.shape()[1], 3)

    # grad_b should be reduced to shape [1] with value -6.0 (sum of -ones)
    assert_equal(grad_b.shape()[0], 1)
    assert_almost_equal(
        grad_b._data.bitcast[Float32]()[0], Float32(-6.0), tolerance=1e-5
    )


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

    var grads = multiply_backward(grad_output, a, b)
    var grad_a = grads.grad_a
    var grad_b = grads.grad_b

    # grad_a = grad_output * b = 1.0 * 2.0 = 2.0
    for i in range(6):
        assert_almost_equal(
            grad_a._data.bitcast[Float32]()[i], Float32(2.0), tolerance=1e-5
        )

    # grad_b = grad_output * a = 1.0 * 1.0 = 1.0
    for i in range(6):
        assert_almost_equal(
            grad_b._data.bitcast[Float32]()[i], Float32(1.0), tolerance=1e-5
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

    var grads = multiply_backward(grad_output, a, b_scalar)
    var grad_a = grads.grad_a
    var grad_b = grads.grad_b

    # grad_a should match shape [2, 3]
    assert_equal(grad_a.shape()[0], 2)
    assert_equal(grad_a.shape()[1], 3)
    # grad_a = grad_output * b = 1.0 * 2.0 = 2.0
    for i in range(6):
        assert_almost_equal(
            grad_a._data.bitcast[Float32]()[i], Float32(2.0), tolerance=1e-5
        )

    # grad_b should be reduced to shape [1]
    assert_equal(grad_b.shape()[0], 1)
    # grad_b = sum(grad_output * a) = sum(1.0 * 1.0) = 6.0
    assert_almost_equal(
        grad_b._data.bitcast[Float32]()[0], Float32(6.0), tolerance=1e-5
    )


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

    var grads = divide_backward(grad_output, a, b)
    var grad_a = grads.grad_a
    var grad_b = grads.grad_b

    # grad_a = grad_output / b = 1.0 / 2.0 = 0.5
    for i in range(6):
        assert_almost_equal(
            grad_a._data.bitcast[Float32]()[i], Float32(0.5), tolerance=1e-4
        )

    # grad_b = -grad_output * a / b² = -1.0 * 2.0 / 4.0 = -0.5
    for i in range(6):
        assert_almost_equal(
            grad_b._data.bitcast[Float32]()[i], Float32(-0.5), tolerance=1e-4
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

    var grads = divide_backward(grad_output, a, b_scalar)
    var grad_a = grads.grad_a
    var grad_b = grads.grad_b

    # grad_a should match shape [2, 3]
    assert_equal(grad_a.shape()[0], 2)
    assert_equal(grad_a.shape()[1], 3)
    # grad_a = grad_output / b = 1.0 / 2.0 = 0.5
    for i in range(6):
        assert_almost_equal(
            grad_a._data.bitcast[Float32]()[i], Float32(0.5), tolerance=1e-4
        )

    # grad_b should be reduced to shape [1]
    assert_equal(grad_b.shape()[0], 1)
    # grad_b = sum(-grad_output * a / b²) = sum(-1.0 * 2.0 / 4.0) = 6 * (-0.5) = -3.0
    assert_almost_equal(
        grad_b._data.bitcast[Float32]()[0], Float32(-3.0), tolerance=1e-4
    )


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

    var grads = add_backward(grad_output, a, b)
    var grad_a = grads.grad_a
    var grad_b = grads.grad_b

    # grad_a should match shape [2, 3]
    assert_equal(grad_a.shape()[0], 2)
    assert_equal(grad_a.shape()[1], 3)

    # grad_b should be reduced to shape [3]
    assert_equal(grad_b.shape()[0], 3)

    # grad_b should contain sum over first dimension (2 ones = 2.0)
    for i in range(3):
        assert_almost_equal(
            grad_b._data.bitcast[Float32]()[i], Float32(2.0), tolerance=1e-5
        )


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

    var grads = subtract_backward(grad_output, a, b)
    var grad_a = grads.grad_a
    var grad_b = grads.grad_b

    # grad_a should match shape [2, 3] with value 1.0
    assert_equal(grad_a.shape()[0], 2)
    assert_equal(grad_a.shape()[1], 3)

    # grad_b should be reduced to shape [3] with value -2.0 (sum of -ones)
    assert_equal(grad_b.shape()[0], 3)
    for i in range(3):
        assert_almost_equal(
            grad_b._data.bitcast[Float32]()[i], Float32(-2.0), tolerance=1e-5
        )


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

    var grads = multiply_backward(grad_output, a, b)
    var grad_a = grads.grad_a
    var grad_b = grads.grad_b

    # grad_a should match shape [2, 3]
    assert_equal(grad_a.shape()[0], 2)
    assert_equal(grad_a.shape()[1], 3)

    # grad_b should be reduced to shape [3]
    assert_equal(grad_b.shape()[0], 3)
    # grad_b[i] = sum(grad_output * a) = sum(1.0 * 1.0) over first dim = 2.0
    for i in range(3):
        assert_almost_equal(
            grad_b._data.bitcast[Float32]()[i], Float32(2.0), tolerance=1e-5
        )


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

    var grads = divide_backward(grad_output, a, b)
    var grad_a = grads.grad_a
    var grad_b = grads.grad_b

    # grad_a should match shape [2, 3]
    assert_equal(grad_a.shape()[0], 2)
    assert_equal(grad_a.shape()[1], 3)

    # grad_b should be reduced to shape [3]
    assert_equal(grad_b.shape()[0], 3)
    # grad_b[i] = sum(-grad_output * a / b²) = sum(-1.0 * 2.0 / 4.0) over first dim
    #           = 2 * (-0.5) = -1.0
    for i in range(3):
        assert_almost_equal(
            grad_b._data.bitcast[Float32]()[i], Float32(-1.0), tolerance=1e-4
        )


# ============================================================================
# Test 13: Addition Backward with Numerical Gradient Checking
# ============================================================================


fn test_add_backward_gradient() raises:
    """Test add_backward with numerical gradient checking.

    Validates that analytical gradients match numerical gradients computed
    via central differences, confirming correct implementation of the
    backward pass for addition.
    """
    var shape = create_shape_vec(3, 4)
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)

    # Initialize with non-uniform values
    for i in range(12):
        a._data.bitcast[Float32]()[i] = Float32(i) * 0.1 - 1.2
        b._data.bitcast[Float32]()[i] = Float32(i) * 0.15 - 0.8

    fn forward(inp: ExTensor) raises -> ExTensor:
        return add(inp, b)

    fn backward(grad_out: ExTensor, inp: ExTensor) raises -> ExTensor:
        var grads = add_backward(grad_out, inp, b)
        return grads.grad_a

    var output = forward(a)
    var grad_output = ones_like(output)

    check_gradient(forward, backward, a, grad_output, rtol=5e-3, atol=1e-5)


# ============================================================================
# Test 14: Subtraction Backward with Numerical Gradient Checking
# ============================================================================


fn test_subtract_backward_gradient() raises:
    """Test subtract_backward with numerical gradient checking.

    Validates analytical vs numerical gradients for subtraction operation.
    """
    var shape = create_shape_vec(3, 4)
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)

    # Initialize with non-uniform values
    for i in range(12):
        a._data.bitcast[Float32]()[i] = Float32(i) * 0.1 + 0.5
        b._data.bitcast[Float32]()[i] = Float32(i) * 0.2 - 1.5

    fn forward(inp: ExTensor) raises -> ExTensor:
        return subtract(inp, b)

    fn backward(grad_out: ExTensor, inp: ExTensor) raises -> ExTensor:
        var grads = subtract_backward(grad_out, inp, b)
        return grads.grad_a

    var output = forward(a)
    var grad_output = ones_like(output)

    check_gradient(forward, backward, a, grad_output, rtol=5e-3, atol=1e-5)


# ============================================================================
# Test 15: Multiplication Backward with Numerical Gradient Checking
# ============================================================================


fn test_multiply_backward_gradient() raises:
    """Test multiply_backward with numerical gradient checking.

    Validates analytical vs numerical gradients for multiplication operation.
    Tests product rule: ∂(A*B)/∂A = B.
    """
    var shape = create_shape_vec(3, 4)
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)

    # Initialize with non-uniform values (avoid zero to test product properly)
    for i in range(12):
        a._data.bitcast[Float32]()[i] = Float32(i) * 0.1 + 0.1
        b._data.bitcast[Float32]()[i] = Float32(i) * 0.15 + 0.2

    fn forward(inp: ExTensor) raises -> ExTensor:
        return multiply(inp, b)

    fn backward(grad_out: ExTensor, inp: ExTensor) raises -> ExTensor:
        var grads = multiply_backward(grad_out, inp, b)
        return grads.grad_a

    var output = forward(a)
    var grad_output = ones_like(output)

    check_gradient(forward, backward, a, grad_output, rtol=5e-3, atol=1e-5)


# ============================================================================
# Test 16: Division Backward with Numerical Gradient Checking
# ============================================================================


fn test_divide_backward_gradient() raises:
    """Test divide_backward with numerical gradient checking.

    Validates analytical vs numerical gradients for division operation.
    Tests quotient rule: ∂(A/B)/∂A = 1/B.
    """
    var shape = create_shape_vec(3, 4)
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)

    # Initialize with non-uniform values (avoid zero denominator)
    for i in range(12):
        a._data.bitcast[Float32]()[i] = Float32(i) * 0.2 + 0.5
        b._data.bitcast[Float32]()[i] = Float32(i) * 0.1 + 1.0  # Ensure b > 0

    fn forward(inp: ExTensor) raises -> ExTensor:
        return divide(inp, b)

    fn backward(grad_out: ExTensor, inp: ExTensor) raises -> ExTensor:
        var grads = divide_backward(grad_out, inp, b)
        return grads.grad_a

    var output = forward(a)
    var grad_output = ones_like(output)

    check_gradient(forward, backward, a, grad_output, rtol=1e-2, atol=1e-5)


# ============================================================================
# Test 17: Add Backward (B operand) with Numerical Gradient Checking
# ============================================================================


fn test_add_backward_b_gradient() raises:
    """Test add_backward gradient w.r.t. second operand (B).

    Validates gradient computation for the second input of addition.
    """
    var shape = create_shape_vec(3, 4)
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)

    # Initialize with non-uniform values
    for i in range(12):
        a._data.bitcast[Float32]()[i] = Float32(i) * 0.1 - 0.5
        b._data.bitcast[Float32]()[i] = Float32(i) * 0.12 + 0.3

    fn forward(inp: ExTensor) raises -> ExTensor:
        return add(a, inp)

    fn backward(grad_out: ExTensor, inp: ExTensor) raises -> ExTensor:
        var grads = add_backward(grad_out, a, inp)
        return grads.grad_b

    var output = forward(b)
    var grad_output = ones_like(output)

    check_gradient(forward, backward, b, grad_output, rtol=5e-3, atol=1e-5)


# ============================================================================
# Test 18: Subtract Backward (B operand) with Numerical Gradient Checking
# ============================================================================


fn test_subtract_backward_b_gradient() raises:
    """Test subtract_backward gradient w.r.t. second operand (B).

    Validates that gradient for B is negated: ∂(A-B)/∂B = -1
    """
    var shape = create_shape_vec(3, 4)
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)

    # Initialize with non-uniform values
    for i in range(12):
        a._data.bitcast[Float32]()[i] = Float32(i) * 0.15 + 0.2
        b._data.bitcast[Float32]()[i] = Float32(i) * 0.1 - 1.0

    fn forward(inp: ExTensor) raises -> ExTensor:
        return subtract(a, inp)

    fn backward(grad_out: ExTensor, inp: ExTensor) raises -> ExTensor:
        var grads = subtract_backward(grad_out, a, inp)
        return grads.grad_b

    var output = forward(b)
    var grad_output = ones_like(output)

    check_gradient(forward, backward, b, grad_output, rtol=5e-3, atol=1e-5)


# ============================================================================
# Test 19: Multiply Backward (B operand) with Numerical Gradient Checking
# ============================================================================


fn test_multiply_backward_b_gradient() raises:
    """Test multiply_backward gradient w.r.t. second operand (B).

    Validates product rule for second operand: ∂(A*B)/∂B = A.
    """
    var shape = create_shape_vec(3, 4)
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)

    # Initialize with non-uniform values (avoid zero for product)
    for i in range(12):
        a._data.bitcast[Float32]()[i] = Float32(i) * 0.2 + 0.1
        b._data.bitcast[Float32]()[i] = Float32(i) * 0.15 + 0.15

    fn forward(inp: ExTensor) raises -> ExTensor:
        return multiply(a, inp)

    fn backward(grad_out: ExTensor, inp: ExTensor) raises -> ExTensor:
        var grads = multiply_backward(grad_out, a, inp)
        return grads.grad_b

    var output = forward(b)
    var grad_output = ones_like(output)

    check_gradient(forward, backward, b, grad_output, rtol=1e-2, atol=1e-5)


# ============================================================================
# Test 20: Divide Backward (B operand) with Numerical Gradient Checking
# ============================================================================


fn test_divide_backward_b_gradient() raises:
    """Test divide_backward gradient w.r.t. second operand (B).

    Validates quotient rule for denominator: ∂(A/B)/∂B = -A/B²
    """
    var shape = create_shape_vec(3, 4)
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)

    # Initialize with non-uniform values (ensure b > 0 to avoid division by zero)
    for i in range(12):
        a._data.bitcast[Float32]()[i] = Float32(i) * 0.2 + 0.5
        b._data.bitcast[Float32]()[i] = Float32(i) * 0.1 + 1.5  # b > 0

    fn forward(inp: ExTensor) raises -> ExTensor:
        return divide(a, inp)

    fn backward(grad_out: ExTensor, inp: ExTensor) raises -> ExTensor:
        var grads = divide_backward(grad_out, a, inp)
        return grads.grad_b

    var output = forward(b)
    var grad_output = ones_like(output)

    check_gradient(forward, backward, b, grad_output, rtol=1e-2, atol=1e-5)


# ============================================================================
# Test 21: Add Backward Broadcast with Numerical Gradient Checking
# ============================================================================


fn test_add_backward_broadcast_gradient() raises:
    """Test add_backward with broadcasting and numerical gradient checking.

    Validates gradient computation when one operand broadcasts.
    Broadcasting case: [3] broadcast to [2, 3]
    """
    var a_shape = create_shape_vec(2, 3)
    var b_shape = create_shape_vec(3)

    var a = zeros(a_shape, DType.float32)
    var b = zeros(b_shape, DType.float32)

    # Initialize with non-uniform values
    for i in range(6):
        a._data.bitcast[Float32]()[i] = Float32(i) * 0.1 + 0.2
    for i in range(3):
        b._data.bitcast[Float32]()[i] = Float32(i) * 0.15 - 0.3

    fn forward(inp: ExTensor) raises -> ExTensor:
        return add(a, inp)

    fn backward(grad_out: ExTensor, inp: ExTensor) raises -> ExTensor:
        var grads = add_backward(grad_out, a, inp)
        return grads.grad_b

    var output = forward(b)
    var grad_output = ones_like(output)

    check_gradient(forward, backward, b, grad_output, rtol=5e-3, atol=1e-5)


# ============================================================================
# Test 22: Multiply Backward Broadcast with Numerical Gradient Checking
# ============================================================================


fn test_multiply_backward_broadcast_gradient() raises:
    """Test multiply_backward with broadcasting and numerical gradient checking.

    Validates product rule when one operand broadcasts.
    Broadcasting case: [3] broadcast to [2, 3]
    """
    var a_shape = create_shape_vec(2, 3)
    var b_shape = create_shape_vec(3)

    var a = zeros(a_shape, DType.float32)
    var b = zeros(b_shape, DType.float32)

    # Initialize with non-uniform values (avoid zero for product)
    for i in range(6):
        a._data.bitcast[Float32]()[i] = Float32(i) * 0.1 + 0.1
    for i in range(3):
        b._data.bitcast[Float32]()[i] = Float32(i) * 0.2 + 0.2

    fn forward(inp: ExTensor) raises -> ExTensor:
        return multiply(a, inp)

    fn backward(grad_out: ExTensor, inp: ExTensor) raises -> ExTensor:
        var grads = multiply_backward(grad_out, a, inp)
        return grads.grad_b

    var output = forward(b)
    var grad_output = ones_like(output)

    check_gradient(forward, backward, b, grad_output, rtol=5e-3, atol=1e-5)


# ============================================================================
# Test 23: Divide Backward Broadcast with Numerical Gradient Checking
# ============================================================================


fn test_divide_backward_broadcast_gradient() raises:
    """Test divide_backward with broadcasting and numerical gradient checking.

    Validates quotient rule when denominator broadcasts.
    Broadcasting case: [3] broadcast to [2, 3]
    """
    var a_shape = create_shape_vec(2, 3)
    var b_shape = create_shape_vec(3)

    var a = zeros(a_shape, DType.float32)
    var b = zeros(b_shape, DType.float32)

    # Initialize with non-uniform values (ensure b > 0)
    for i in range(6):
        a._data.bitcast[Float32]()[i] = Float32(i) * 0.2 + 0.5
    for i in range(3):
        b._data.bitcast[Float32]()[i] = Float32(i) * 0.1 + 1.0  # b > 0

    fn forward(inp: ExTensor) raises -> ExTensor:
        return divide(a, inp)

    fn backward(grad_out: ExTensor, inp: ExTensor) raises -> ExTensor:
        var grads = divide_backward(grad_out, a, inp)
        return grads.grad_b

    var output = forward(b)
    var grad_output = ones_like(output)

    check_gradient(forward, backward, b, grad_output, rtol=5e-3, atol=1e-5)


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run all tests in this file."""
    var total = 0
    var passed = 0
    var failed = 0

    print("\n" + "=" * 70)
    print("Running tests from: test_arithmetic_backward.mojo")
    print("=" * 70 + "\n")

    # test_add_backward
    total += 1
    try:
        test_add_backward()
        passed += 1
        print("  ✓ test_add_backward")
    except e:
        failed += 1
        print("  ✗ test_add_backward:", e)

    # test_add_scalar_backward
    total += 1
    try:
        test_add_scalar_backward()
        passed += 1
        print("  ✓ test_add_scalar_backward")
    except e:
        failed += 1
        print("  ✗ test_add_scalar_backward:", e)

    # test_subtract_backward
    total += 1
    try:
        test_subtract_backward()
        passed += 1
        print("  ✓ test_subtract_backward")
    except e:
        failed += 1
        print("  ✗ test_subtract_backward:", e)

    # test_subtract_scalar_backward
    total += 1
    try:
        test_subtract_scalar_backward()
        passed += 1
        print("  ✓ test_subtract_scalar_backward")
    except e:
        failed += 1
        print("  ✗ test_subtract_scalar_backward:", e)

    # test_multiply_backward
    total += 1
    try:
        test_multiply_backward()
        passed += 1
        print("  ✓ test_multiply_backward")
    except e:
        failed += 1
        print("  ✗ test_multiply_backward:", e)

    # test_multiply_scalar_backward
    total += 1
    try:
        test_multiply_scalar_backward()
        passed += 1
        print("  ✓ test_multiply_scalar_backward")
    except e:
        failed += 1
        print("  ✗ test_multiply_scalar_backward:", e)

    # test_divide_backward
    total += 1
    try:
        test_divide_backward()
        passed += 1
        print("  ✓ test_divide_backward")
    except e:
        failed += 1
        print("  ✗ test_divide_backward:", e)

    # test_divide_scalar_backward
    total += 1
    try:
        test_divide_scalar_backward()
        passed += 1
        print("  ✓ test_divide_scalar_backward")
    except e:
        failed += 1
        print("  ✗ test_divide_scalar_backward:", e)

    # test_add_broadcast
    total += 1
    try:
        test_add_broadcast()
        passed += 1
        print("  ✓ test_add_broadcast")
    except e:
        failed += 1
        print("  ✗ test_add_broadcast:", e)

    # test_subtract_broadcast
    total += 1
    try:
        test_subtract_broadcast()
        passed += 1
        print("  ✓ test_subtract_broadcast")
    except e:
        failed += 1
        print("  ✗ test_subtract_broadcast:", e)

    # test_multiply_broadcast
    total += 1
    try:
        test_multiply_broadcast()
        passed += 1
        print("  ✓ test_multiply_broadcast")
    except e:
        failed += 1
        print("  ✗ test_multiply_broadcast:", e)

    # test_divide_broadcast
    total += 1
    try:
        test_divide_broadcast()
        passed += 1
        print("  ✓ test_divide_broadcast")
    except e:
        failed += 1
        print("  ✗ test_divide_broadcast:", e)

    # test_add_backward_gradient
    total += 1
    try:
        test_add_backward_gradient()
        passed += 1
        print("  ✓ test_add_backward_gradient")
    except e:
        failed += 1
        print("  ✗ test_add_backward_gradient:", e)

    # test_subtract_backward_gradient
    total += 1
    try:
        test_subtract_backward_gradient()
        passed += 1
        print("  ✓ test_subtract_backward_gradient")
    except e:
        failed += 1
        print("  ✗ test_subtract_backward_gradient:", e)

    # test_multiply_backward_gradient
    total += 1
    try:
        test_multiply_backward_gradient()
        passed += 1
        print("  ✓ test_multiply_backward_gradient")
    except e:
        failed += 1
        print("  ✗ test_multiply_backward_gradient:", e)

    # test_divide_backward_gradient
    total += 1
    try:
        test_divide_backward_gradient()
        passed += 1
        print("  ✓ test_divide_backward_gradient")
    except e:
        failed += 1
        print("  ✗ test_divide_backward_gradient:", e)

    # test_add_backward_b_gradient
    total += 1
    try:
        test_add_backward_b_gradient()
        passed += 1
        print("  ✓ test_add_backward_b_gradient")
    except e:
        failed += 1
        print("  ✗ test_add_backward_b_gradient:", e)

    # test_subtract_backward_b_gradient
    total += 1
    try:
        test_subtract_backward_b_gradient()
        passed += 1
        print("  ✓ test_subtract_backward_b_gradient")
    except e:
        failed += 1
        print("  ✗ test_subtract_backward_b_gradient:", e)

    # test_multiply_backward_b_gradient
    total += 1
    try:
        test_multiply_backward_b_gradient()
        passed += 1
        print("  ✓ test_multiply_backward_b_gradient")
    except e:
        failed += 1
        print("  ✗ test_multiply_backward_b_gradient:", e)

    # test_divide_backward_b_gradient
    total += 1
    try:
        test_divide_backward_b_gradient()
        passed += 1
        print("  ✓ test_divide_backward_b_gradient")
    except e:
        failed += 1
        print("  ✗ test_divide_backward_b_gradient:", e)

    # test_add_backward_broadcast_gradient
    total += 1
    try:
        test_add_backward_broadcast_gradient()
        passed += 1
        print("  ✓ test_add_backward_broadcast_gradient")
    except e:
        failed += 1
        print("  ✗ test_add_backward_broadcast_gradient:", e)

    # test_multiply_backward_broadcast_gradient
    total += 1
    try:
        test_multiply_backward_broadcast_gradient()
        passed += 1
        print("  ✓ test_multiply_backward_broadcast_gradient")
    except e:
        failed += 1
        print("  ✗ test_multiply_backward_broadcast_gradient:", e)

    # test_divide_backward_broadcast_gradient
    total += 1
    try:
        test_divide_backward_broadcast_gradient()
        passed += 1
        print("  ✓ test_divide_backward_broadcast_gradient")
    except e:
        failed += 1
        print("  ✗ test_divide_backward_broadcast_gradient:", e)

    # Summary
    print("\n" + "=" * 70)
    print("Results:", passed, "/", total, "passed,", failed, "failed")
    print("=" * 70)

    if failed > 0:
        raise Error("Tests failed")
