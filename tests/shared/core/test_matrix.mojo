"""Tests for matrix operations.

Tests cover:
- Matrix multiplication (matmul)
- Transpose
- Dot product
- Outer product
- Backward passes
- Shape validation

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
from shared.core.matrix import (
    matmul,
    transpose,
    dot,
    outer,
    matmul_backward,
    transpose_backward,
)
from tests.helpers.gradient_checking import check_gradient, compute_numerical_gradient, assert_gradients_close


# ============================================================================
# Matrix Multiplication Tests
# ============================================================================


fn test_matmul_shapes() raises:
    """Test that matmul returns correct output shape."""
    var shape_a = List[Int]()
    shape_a[0] = 4
    shape_a[1] = 3

    var shape_b = List[Int]()
    shape_b[0] = 3
    shape_b[1] = 5

    var a = ones(shape_a, DType.float32)
    var b = ones(shape_b, DType.float32)

    var result = matmul(a, b)

    # (4, 3) @ (3, 5) = (4, 5)
    assert_equal(result.shape()[0], 4)
    assert_equal(result.shape()[1], 5)


fn test_matmul_values() raises:
    """Test that matmul computes correct values."""
    var shape_a = List[Int]()
    shape_a[0] = 2
    shape_a[1] = 2

    var shape_b = List[Int]()
    shape_b[0] = 2
    shape_b[1] = 2

    var a = zeros(shape_a, DType.float32)
    var b = zeros(shape_b, DType.float32)

    # A = [[1, 2], [3, 4]]
    a._data.bitcast[Float32]()[0] = 1.0
    a._data.bitcast[Float32]()[1] = 2.0
    a._data.bitcast[Float32]()[2] = 3.0
    a._data.bitcast[Float32]()[3] = 4.0

    # B = [[5, 6], [7, 8]]
    b._data.bitcast[Float32]()[0] = 5.0
    b._data.bitcast[Float32]()[1] = 6.0
    b._data.bitcast[Float32]()[2] = 7.0
    b._data.bitcast[Float32]()[3] = 8.0

    var result = matmul(a, b)

    # Result = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
    #        = [[19, 22], [43, 50]]
    assert_almost_equal(result._data.bitcast[Float32]()[0], Float32(19.0), tolerance=1e-5)
    assert_almost_equal(result._data.bitcast[Float32]()[1], Float32(22.0), tolerance=1e-5)
    assert_almost_equal(result._data.bitcast[Float32]()[2], Float32(43.0), tolerance=1e-5)
    assert_almost_equal(result._data.bitcast[Float32]()[3], Float32(50.0), tolerance=1e-5)


fn test_matmul_identity() raises:
    """Test matmul with identity matrix."""
    var shape = List[Int]()
    shape[0] = 3
    shape[1] = 3

    var a = zeros(shape, DType.float32)
    var identity = zeros(shape, DType.float32)

    # A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    for i in range(9):
        a._data.bitcast[Float32]()[i] = Float32(i + 1)

    # Identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    identity._data.bitcast[Float32]()[0] = 1.0  # (0, 0)
    identity._data.bitcast[Float32]()[4] = 1.0  # (1, 1)
    identity._data.bitcast[Float32]()[8] = 1.0  # (2, 2)

    var result = matmul(a, identity)

    # A @ I = A
    for i in range(9):
        assert_almost_equal(
            result._data.bitcast[Float32]()[i],
            a._data.bitcast[Float32]()[i],
            tolerance=1e-5
        )


fn test_matmul_backward_shapes() raises:
    """Test that matmul_backward returns correct gradient shapes."""
    var shape_a = List[Int]()
    shape_a[0] = 4
    shape_a[1] = 3

    var shape_b = List[Int]()
    shape_b[0] = 3
    shape_b[1] = 5

    var a = ones(shape_a, DType.float32)
    var b = ones(shape_b, DType.float32)

    var result = matmul(a, b)

    var grad_output_shape = List[Int]()
    grad_output_shape[0] = 4
    grad_output_shape[1] = 5
    var grad_output = ones(grad_output_shape, DType.float32)

    var (grad_a, grad_b) = matmul_backward(grad_output, a, b)

    # grad_a should have same shape as a
    assert_equal(grad_a.shape()[0], 4)
    assert_equal(grad_a.shape()[1], 3)

    # grad_b should have same shape as b
    assert_equal(grad_b.shape()[0], 3)
    assert_equal(grad_b.shape()[1], 5)


fn test_matmul_backward_gradient_a() raises:
    """Test matmul_backward gradient w.r.t. input A with numerical checking.

    Validates that gradient w.r.t. A matches finite differences.
    """
    var batch = 2
    var m = 3
    var k = 4
    var n = 2

    # Create input A with shape (batch*m, k)
    var shape_a = List[Int]()
    shape_a[0] = batch * m
    shape_a[1] = k
    var a = zeros(shape_a, DType.float32)

    # Initialize A with non-uniform values
    for i in range(batch * m * k):
        a._data.bitcast[Float32]()[i] = Float32(i) * 0.1 - 1.0

    # Create input B with shape (k, n)
    var shape_b = List[Int]()
    shape_b[0] = k
    shape_b[1] = n
    var b = zeros(shape_b, DType.float32)

    # Initialize B with non-uniform values
    for i in range(k * n):
        b._data.bitcast[Float32]()[i] = Float32(i) * 0.2 - 0.5

    # Forward function wrapper
    fn forward(inp: ExTensor) raises -> ExTensor:
        return matmul(inp, b)

    # Backward function wrapper for grad_a
    fn backward(grad_out: ExTensor, inp: ExTensor) raises -> ExTensor:
        var (grad_a, _) = matmul_backward(grad_out, inp, b)
        return grad_a

    var output = forward(a)
    var grad_output = ones_like(output)

    # Numerical gradient checking
    check_gradient(forward, backward, a, grad_output, rtol=1e-3, atol=1e-6)


fn test_matmul_backward_gradient_b() raises:
    """Test matmul_backward gradient w.r.t. input B with numerical checking.

    Validates that gradient w.r.t. B matches finite differences.
    """
    var m = 3
    var k = 4
    var n = 2

    # Create input A with shape (m, k)
    var shape_a = List[Int]()
    shape_a[0] = m
    shape_a[1] = k
    var a = zeros(shape_a, DType.float32)

    # Initialize A with non-uniform values
    for i in range(m * k):
        a._data.bitcast[Float32]()[i] = Float32(i) * 0.1 - 1.0

    # Create input B with shape (k, n)
    var shape_b = List[Int]()
    shape_b[0] = k
    shape_b[1] = n
    var b = zeros(shape_b, DType.float32)

    # Initialize B with non-uniform values
    for i in range(k * n):
        b._data.bitcast[Float32]()[i] = Float32(i) * 0.2 - 0.5

    # Forward function wrapper
    fn forward(inp: ExTensor) raises -> ExTensor:
        return matmul(a, inp)

    # Backward function wrapper for grad_b
    fn backward(grad_out: ExTensor, inp: ExTensor) raises -> ExTensor:
        var (_, grad_b) = matmul_backward(grad_out, a, inp)
        return grad_b

    var output = forward(b)
    var grad_output = ones_like(output)

    # Numerical gradient checking
    check_gradient(forward, backward, b, grad_output, rtol=1e-3, atol=1e-6)


# ============================================================================
# Transpose Tests
# ============================================================================


fn test_transpose_shapes() raises:
    """Test that transpose returns correct output shape."""
    var shape = List[Int]()
    shape[0] = 4
    shape[1] = 10

    var a = ones(shape, DType.float32)
    var result = transpose(a)

    # (4, 10) -> (10, 4)
    assert_equal(result.shape()[0], 10)
    assert_equal(result.shape()[1], 4)


fn test_transpose_values() raises:
    """Test that transpose computes correct values."""
    var shape = List[Int]()
    shape[0] = 2
    shape[1] = 3

    var a = zeros(shape, DType.float32)

    # A = [[1, 2, 3], [4, 5, 6]]
    a._data.bitcast[Float32]()[0] = 1.0
    a._data.bitcast[Float32]()[1] = 2.0
    a._data.bitcast[Float32]()[2] = 3.0
    a._data.bitcast[Float32]()[3] = 4.0
    a._data.bitcast[Float32]()[4] = 5.0
    a._data.bitcast[Float32]()[5] = 6.0

    var result = transpose(a)

    # A^T = [[1, 4], [2, 5], [3, 6]]
    assert_almost_equal(result._data.bitcast[Float32]()[0], Float32(1.0), tolerance=1e-5)
    assert_almost_equal(result._data.bitcast[Float32]()[1], Float32(4.0), tolerance=1e-5)
    assert_almost_equal(result._data.bitcast[Float32]()[2], Float32(2.0), tolerance=1e-5)
    assert_almost_equal(result._data.bitcast[Float32]()[3], Float32(5.0), tolerance=1e-5)
    assert_almost_equal(result._data.bitcast[Float32]()[4], Float32(3.0), tolerance=1e-5)
    assert_almost_equal(result._data.bitcast[Float32]()[5], Float32(6.0), tolerance=1e-5)


fn test_transpose_double() raises:
    """Test that transpose(transpose(A)) = A."""
    var shape = List[Int]()
    shape[0] = 3
    shape[1] = 4

    var a = zeros(shape, DType.float32)

    # Fill with values
    for i in range(12):
        a._data.bitcast[Float32]()[i] = Float32(i)

    var result = transpose(transpose(a))

    # Should get back original
    for i in range(12):
        assert_almost_equal(
            result._data.bitcast[Float32]()[i],
            a._data.bitcast[Float32]()[i],
            tolerance=1e-5
        )


fn test_transpose_backward_shapes() raises:
    """Test that transpose_backward returns correct gradient shape."""
    var shape = List[Int]()
    shape[0] = 4
    shape[1] = 10

    var a = ones(shape, DType.float32)
    var result = transpose(a)

    var grad_output_shape = List[Int]()
    grad_output_shape[0] = 10
    grad_output_shape[1] = 4
    var grad_output = ones(grad_output_shape, DType.float32)

    var grad_input = transpose_backward(grad_output)

    # Gradient should have same shape as input
    assert_equal(grad_input.shape()[0], 4)
    assert_equal(grad_input.shape()[1], 10)


fn test_transpose_backward_gradient() raises:
    """Test transpose_backward with numerical gradient checking.

    Validates that gradient matches finite differences. Since transpose is
    its own inverse, the gradient should simply be the transposed gradient.
    """
    var m = 3
    var n = 4

    # Create input with shape (m, n)
    var shape = List[Int]()
    shape[0] = m
    shape[1] = n
    var x = zeros(shape, DType.float32)

    # Initialize with non-uniform values
    for i in range(m * n):
        x._data.bitcast[Float32]()[i] = Float32(i) * 0.15 - 2.0

    # Forward function wrapper
    fn forward(inp: ExTensor) raises -> ExTensor:
        return transpose(inp)

    # Backward function wrapper
    fn backward(grad_out: ExTensor, inp: ExTensor) raises -> ExTensor:
        return transpose_backward(grad_out)

    var output = forward(x)
    var grad_output = ones_like(output)

    # Numerical gradient checking
    check_gradient(forward, backward, x, grad_output, rtol=1e-3, atol=1e-6)


# ============================================================================
# Dot Product Tests
# ============================================================================


fn test_dot_shapes() raises:
    """Test that dot returns scalar output."""
    var shape = List[Int]()
    shape[0] = 5

    var a = ones(shape, DType.float32)
    var b = ones(shape, DType.float32)

    var result = dot(a, b)

    # Dot product returns scalar
    assert_equal(result.shape()[0], 1)


fn test_dot_values() raises:
    """Test that dot computes correct values."""
    var shape = List[Int]()
    shape[0] = 3

    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)

    a._data.bitcast[Float32]()[0] = 1.0
    a._data.bitcast[Float32]()[1] = 2.0
    a._data.bitcast[Float32]()[2] = 3.0

    b._data.bitcast[Float32]()[0] = 4.0
    b._data.bitcast[Float32]()[1] = 5.0
    b._data.bitcast[Float32]()[2] = 6.0

    var result = dot(a, b)

    # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    assert_almost_equal(result._data.bitcast[Float32]()[0], Float32(32.0), tolerance=1e-5)


fn test_dot_orthogonal() raises:
    """Test dot product of orthogonal vectors."""
    var shape = List[Int]()
    shape[0] = 2

    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)

    # a = [1, 0], b = [0, 1]
    a._data.bitcast[Float32]()[0] = 1.0
    a._data.bitcast[Float32]()[1] = 0.0

    b._data.bitcast[Float32]()[0] = 0.0
    b._data.bitcast[Float32]()[1] = 1.0

    var result = dot(a, b)

    # Orthogonal vectors have dot product = 0
    assert_almost_equal(result._data.bitcast[Float32]()[0], Float32(0.0), tolerance=1e-5)


# ============================================================================
# Outer Product Tests
# ============================================================================


fn test_outer_shapes() raises:
    """Test that outer returns correct output shape."""
    var shape_a = List[Int]()
    shape_a[0] = 3

    var shape_b = List[Int]()
    shape_b[0] = 4

    var a = ones(shape_a, DType.float32)
    var b = ones(shape_b, DType.float32)

    var result = outer(a, b)

    # (3,) outer (4,) = (3, 4)
    assert_equal(result.shape()[0], 3)
    assert_equal(result.shape()[1], 4)


fn test_outer_values() raises:
    """Test that outer computes correct values."""
    var shape_a = List[Int]()
    shape_a[0] = 2

    var shape_b = List[Int]()
    shape_b[0] = 3

    var a = zeros(shape_a, DType.float32)
    var b = zeros(shape_b, DType.float32)

    # a = [2, 3]
    a._data.bitcast[Float32]()[0] = 2.0
    a._data.bitcast[Float32]()[1] = 3.0

    # b = [4, 5, 6]
    b._data.bitcast[Float32]()[0] = 4.0
    b._data.bitcast[Float32]()[1] = 5.0
    b._data.bitcast[Float32]()[2] = 6.0

    var result = outer(a, b)

    # Outer product = [[2*4, 2*5, 2*6], [3*4, 3*5, 3*6]]
    #                = [[8, 10, 12], [12, 15, 18]]
    assert_almost_equal(result._data.bitcast[Float32]()[0], Float32(8.0), tolerance=1e-5)
    assert_almost_equal(result._data.bitcast[Float32]()[1], Float32(10.0), tolerance=1e-5)
    assert_almost_equal(result._data.bitcast[Float32]()[2], Float32(12.0), tolerance=1e-5)
    assert_almost_equal(result._data.bitcast[Float32]()[3], Float32(12.0), tolerance=1e-5)
    assert_almost_equal(result._data.bitcast[Float32]()[4], Float32(15.0), tolerance=1e-5)
    assert_almost_equal(result._data.bitcast[Float32]()[5], Float32(18.0), tolerance=1e-5)


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run all matrix operation tests."""
    print("Running matrix operation tests...")

    # Matmul tests
    test_matmul_shapes()
    print("✓ test_matmul_shapes")

    test_matmul_values()
    print("✓ test_matmul_values")

    test_matmul_identity()
    print("✓ test_matmul_identity")

    test_matmul_backward_shapes()
    print("✓ test_matmul_backward_shapes")

    test_matmul_backward_gradient_a()
    print("✓ test_matmul_backward_gradient_a")

    test_matmul_backward_gradient_b()
    print("✓ test_matmul_backward_gradient_b")

    # Transpose tests
    test_transpose_shapes()
    print("✓ test_transpose_shapes")

    test_transpose_values()
    print("✓ test_transpose_values")

    test_transpose_double()
    print("✓ test_transpose_double")

    test_transpose_backward_shapes()
    print("✓ test_transpose_backward_shapes")

    test_transpose_backward_gradient()
    print("✓ test_transpose_backward_gradient")

    # Dot product tests
    test_dot_shapes()
    print("✓ test_dot_shapes")

    test_dot_values()
    print("✓ test_dot_values")

    test_dot_orthogonal()
    print("✓ test_dot_orthogonal")

    # Outer product tests
    test_outer_shapes()
    print("✓ test_outer_shapes")

    test_outer_values()
    print("✓ test_outer_values")

    print("\nAll matrix operation tests passed!")
