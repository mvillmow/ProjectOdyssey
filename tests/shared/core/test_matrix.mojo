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
from shared.core.extensor import ExTensor, zeros, ones
from shared.core.matrix import (
    matmul,
    transpose,
    dot,
    outer,
    matmul_backward,
    transpose_backward,
)
from collections.vector import DynamicVector


# ============================================================================
# Matrix Multiplication Tests
# ============================================================================


fn test_matmul_shapes() raises:
    """Test that matmul returns correct output shape."""
    var shape_a = DynamicVector[Int](2)
    shape_a[0] = 4
    shape_a[1] = 3

    var shape_b = DynamicVector[Int](2)
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
    var shape_a = DynamicVector[Int](2)
    shape_a[0] = 2
    shape_a[1] = 2

    var shape_b = DynamicVector[Int](2)
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
    var shape = DynamicVector[Int](2)
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
    var shape_a = DynamicVector[Int](2)
    shape_a[0] = 4
    shape_a[1] = 3

    var shape_b = DynamicVector[Int](2)
    shape_b[0] = 3
    shape_b[1] = 5

    var a = ones(shape_a, DType.float32)
    var b = ones(shape_b, DType.float32)

    var result = matmul(a, b)

    var grad_output_shape = DynamicVector[Int](2)
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


# ============================================================================
# Transpose Tests
# ============================================================================


fn test_transpose_shapes() raises:
    """Test that transpose returns correct output shape."""
    var shape = DynamicVector[Int](2)
    shape[0] = 4
    shape[1] = 10

    var a = ones(shape, DType.float32)
    var result = transpose(a)

    # (4, 10) -> (10, 4)
    assert_equal(result.shape()[0], 10)
    assert_equal(result.shape()[1], 4)


fn test_transpose_values() raises:
    """Test that transpose computes correct values."""
    var shape = DynamicVector[Int](2)
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
    var shape = DynamicVector[Int](2)
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
    var shape = DynamicVector[Int](2)
    shape[0] = 4
    shape[1] = 10

    var a = ones(shape, DType.float32)
    var result = transpose(a)

    var grad_output_shape = DynamicVector[Int](2)
    grad_output_shape[0] = 10
    grad_output_shape[1] = 4
    var grad_output = ones(grad_output_shape, DType.float32)

    var grad_input = transpose_backward(grad_output)

    # Gradient should have same shape as input
    assert_equal(grad_input.shape()[0], 4)
    assert_equal(grad_input.shape()[1], 10)


# ============================================================================
# Dot Product Tests
# ============================================================================


fn test_dot_shapes() raises:
    """Test that dot returns scalar output."""
    var shape = DynamicVector[Int](1)
    shape[0] = 5

    var a = ones(shape, DType.float32)
    var b = ones(shape, DType.float32)

    var result = dot(a, b)

    # Dot product returns scalar
    assert_equal(result.shape()[0], 1)


fn test_dot_values() raises:
    """Test that dot computes correct values."""
    var shape = DynamicVector[Int](1)
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
    var shape = DynamicVector[Int](1)
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
    var shape_a = DynamicVector[Int](1)
    shape_a[0] = 3

    var shape_b = DynamicVector[Int](1)
    shape_b[0] = 4

    var a = ones(shape_a, DType.float32)
    var b = ones(shape_b, DType.float32)

    var result = outer(a, b)

    # (3,) outer (4,) = (3, 4)
    assert_equal(result.shape()[0], 3)
    assert_equal(result.shape()[1], 4)


fn test_outer_values() raises:
    """Test that outer computes correct values."""
    var shape_a = DynamicVector[Int](1)
    shape_a[0] = 2

    var shape_b = DynamicVector[Int](1)
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

    # Transpose tests
    test_transpose_shapes()
    print("✓ test_transpose_shapes")

    test_transpose_values()
    print("✓ test_transpose_values")

    test_transpose_double()
    print("✓ test_transpose_double")

    test_transpose_backward_shapes()
    print("✓ test_transpose_backward_shapes")

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
