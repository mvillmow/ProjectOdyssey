"""Tests for arithmetic operations.

Tests cover:
- Basic arithmetic: add, subtract, multiply, divide
- Floor division and modulo
- Power operation
- Backward passes for all operations
- Shape validation and broadcasting

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
from shared.core.arithmetic import (
    add,
    subtract,
    multiply,
    divide,
    floor_divide,
    modulo,
    power,
    add_backward,
    subtract_backward,
    multiply_backward,
    divide_backward,
)


# ============================================================================
# Addition Tests
# ============================================================================


fn test_add_shapes() raises:
    """Test that add returns correct output shape."""
    var shape = List[Int](2)
    shape[0] = 4
    shape[1] = 10
    var a = ones(shape, DType.float32)
    var b = ones(shape, DType.float32)

    var result = add(a, b)

    assert_equal(result.shape()[0], 4)
    assert_equal(result.shape()[1], 10)


fn test_add_values() raises:
    """Test that add computes correct values."""
    var shape = List[Int](1)
    shape[0] = 3
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)

    a._data.bitcast[Float32]()[0] = 1.0
    a._data.bitcast[Float32]()[1] = 2.0
    a._data.bitcast[Float32]()[2] = 3.0

    b._data.bitcast[Float32]()[0] = 4.0
    b._data.bitcast[Float32]()[1] = 5.0
    b._data.bitcast[Float32]()[2] = 6.0

    var result = add(a, b)

    assert_almost_equal(result._data.bitcast[Float32]()[0], Float32(5.0), tolerance=1e-5)
    assert_almost_equal(result._data.bitcast[Float32]()[1], Float32(7.0), tolerance=1e-5)
    assert_almost_equal(result._data.bitcast[Float32]()[2], Float32(9.0), tolerance=1e-5)


fn test_add_backward() raises:
    """Test add backward pass."""
    var shape = List[Int](2)
    shape[0] = 2
    shape[1] = 3
    var a = ones(shape, DType.float32)
    var b = ones(shape, DType.float32)
    var grad_output = ones(shape, DType.float32)

    var (grad_a, grad_b) = add_backward(grad_output, a, b)

    # Gradient of add is just pass-through
    for i in range(6):
        assert_almost_equal(grad_a._data.bitcast[Float32]()[i], Float32(1.0), tolerance=1e-5)
        assert_almost_equal(grad_b._data.bitcast[Float32]()[i], Float32(1.0), tolerance=1e-5)


# ============================================================================
# Subtraction Tests
# ============================================================================


fn test_subtract_shapes() raises:
    """Test that subtract returns correct output shape."""
    var shape = List[Int](2)
    shape[0] = 4
    shape[1] = 10
    var a = ones(shape, DType.float32)
    var b = ones(shape, DType.float32)

    var result = subtract(a, b)

    assert_equal(result.shape()[0], 4)
    assert_equal(result.shape()[1], 10)


fn test_subtract_values() raises:
    """Test that subtract computes correct values."""
    var shape = List[Int](1)
    shape[0] = 3
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)

    a._data.bitcast[Float32]()[0] = 5.0
    a._data.bitcast[Float32]()[1] = 7.0
    a._data.bitcast[Float32]()[2] = 9.0

    b._data.bitcast[Float32]()[0] = 2.0
    b._data.bitcast[Float32]()[1] = 3.0
    b._data.bitcast[Float32]()[2] = 4.0

    var result = subtract(a, b)

    assert_almost_equal(result._data.bitcast[Float32]()[0], Float32(3.0), tolerance=1e-5)
    assert_almost_equal(result._data.bitcast[Float32]()[1], Float32(4.0), tolerance=1e-5)
    assert_almost_equal(result._data.bitcast[Float32]()[2], Float32(5.0), tolerance=1e-5)


fn test_subtract_backward() raises:
    """Test subtract backward pass."""
    var shape = List[Int](2)
    shape[0] = 2
    shape[1] = 3
    var a = ones(shape, DType.float32)
    var b = ones(shape, DType.float32)
    var grad_output = ones(shape, DType.float32)

    var (grad_a, grad_b) = subtract_backward(grad_output, a, b)

    # Gradient of subtract: d/da = +1, d/db = -1
    for i in range(6):
        assert_almost_equal(grad_a._data.bitcast[Float32]()[i], Float32(1.0), tolerance=1e-5)
        assert_almost_equal(grad_b._data.bitcast[Float32]()[i], Float32(-1.0), tolerance=1e-5)


# ============================================================================
# Multiplication Tests
# ============================================================================


fn test_multiply_shapes() raises:
    """Test that multiply returns correct output shape."""
    var shape = List[Int](2)
    shape[0] = 4
    shape[1] = 10
    var a = ones(shape, DType.float32)
    var b = ones(shape, DType.float32)

    var result = multiply(a, b)

    assert_equal(result.shape()[0], 4)
    assert_equal(result.shape()[1], 10)


fn test_multiply_values() raises:
    """Test that multiply computes correct values."""
    var shape = List[Int](1)
    shape[0] = 3
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)

    a._data.bitcast[Float32]()[0] = 2.0
    a._data.bitcast[Float32]()[1] = 3.0
    a._data.bitcast[Float32]()[2] = 4.0

    b._data.bitcast[Float32]()[0] = 5.0
    b._data.bitcast[Float32]()[1] = 6.0
    b._data.bitcast[Float32]()[2] = 7.0

    var result = multiply(a, b)

    assert_almost_equal(result._data.bitcast[Float32]()[0], Float32(10.0), tolerance=1e-5)
    assert_almost_equal(result._data.bitcast[Float32]()[1], Float32(18.0), tolerance=1e-5)
    assert_almost_equal(result._data.bitcast[Float32]()[2], Float32(28.0), tolerance=1e-5)


fn test_multiply_backward() raises:
    """Test multiply backward pass."""
    var shape = List[Int](1)
    shape[0] = 2
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)
    var grad_output = ones(shape, DType.float32)

    a._data.bitcast[Float32]()[0] = 2.0
    a._data.bitcast[Float32]()[1] = 3.0
    b._data.bitcast[Float32]()[0] = 4.0
    b._data.bitcast[Float32]()[1] = 5.0

    var (grad_a, grad_b) = multiply_backward(grad_output, a, b)

    # Gradient of multiply: d/da = b, d/db = a
    assert_almost_equal(grad_a._data.bitcast[Float32]()[0], Float32(4.0), tolerance=1e-5)
    assert_almost_equal(grad_a._data.bitcast[Float32]()[1], Float32(5.0), tolerance=1e-5)
    assert_almost_equal(grad_b._data.bitcast[Float32]()[0], Float32(2.0), tolerance=1e-5)
    assert_almost_equal(grad_b._data.bitcast[Float32]()[1], Float32(3.0), tolerance=1e-5)


# ============================================================================
# Division Tests
# ============================================================================


fn test_divide_shapes() raises:
    """Test that divide returns correct output shape."""
    var shape = List[Int](2)
    shape[0] = 4
    shape[1] = 10
    var a = ones(shape, DType.float32)
    var b = ones(shape, DType.float32)

    var result = divide(a, b)

    assert_equal(result.shape()[0], 4)
    assert_equal(result.shape()[1], 10)


fn test_divide_values() raises:
    """Test that divide computes correct values."""
    var shape = List[Int](1)
    shape[0] = 3
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)

    a._data.bitcast[Float32]()[0] = 10.0
    a._data.bitcast[Float32]()[1] = 20.0
    a._data.bitcast[Float32]()[2] = 30.0

    b._data.bitcast[Float32]()[0] = 2.0
    b._data.bitcast[Float32]()[1] = 4.0
    b._data.bitcast[Float32]()[2] = 5.0

    var result = divide(a, b)

    assert_almost_equal(result._data.bitcast[Float32]()[0], Float32(5.0), tolerance=1e-5)
    assert_almost_equal(result._data.bitcast[Float32]()[1], Float32(5.0), tolerance=1e-5)
    assert_almost_equal(result._data.bitcast[Float32]()[2], Float32(6.0), tolerance=1e-5)


fn test_divide_backward() raises:
    """Test divide backward pass."""
    var shape = List[Int](1)
    shape[0] = 2
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)
    var grad_output = ones(shape, DType.float32)

    a._data.bitcast[Float32]()[0] = 10.0
    a._data.bitcast[Float32]()[1] = 20.0
    b._data.bitcast[Float32]()[0] = 2.0
    b._data.bitcast[Float32]()[1] = 4.0

    var (grad_a, grad_b) = divide_backward(grad_output, a, b)

    # Gradient of divide: d/da = 1/b, d/db = -a/b^2
    # d/da[0] = 1/2 = 0.5
    # d/da[1] = 1/4 = 0.25
    # d/db[0] = -10/4 = -2.5
    # d/db[1] = -20/16 = -1.25

    assert_almost_equal(grad_a._data.bitcast[Float32]()[0], Float32(0.5), tolerance=1e-5)
    assert_almost_equal(grad_a._data.bitcast[Float32]()[1], Float32(0.25), tolerance=1e-5)
    assert_almost_equal(grad_b._data.bitcast[Float32]()[0], Float32(-2.5), tolerance=1e-4)
    assert_almost_equal(grad_b._data.bitcast[Float32]()[1], Float32(-1.25), tolerance=1e-4)


# ============================================================================
# Floor Division Tests
# ============================================================================


fn test_floor_divide_shapes() raises:
    """Test that floor_divide returns correct output shape."""
    var shape = List[Int](2)
    shape[0] = 4
    shape[1] = 10
    var a = ones(shape, DType.float32)
    var b = ones(shape, DType.float32)

    var result = floor_divide(a, b)

    assert_equal(result.shape()[0], 4)
    assert_equal(result.shape()[1], 10)


fn test_floor_divide_values() raises:
    """Test that floor_divide computes correct values."""
    var shape = List[Int](1)
    shape[0] = 3
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)

    a._data.bitcast[Float32]()[0] = 7.0
    a._data.bitcast[Float32]()[1] = 8.0
    a._data.bitcast[Float32]()[2] = 9.0

    b._data.bitcast[Float32]()[0] = 2.0
    b._data.bitcast[Float32]()[1] = 3.0
    b._data.bitcast[Float32]()[2] = 4.0

    var result = floor_divide(a, b)

    # 7 // 2 = 3, 8 // 3 = 2, 9 // 4 = 2
    assert_almost_equal(result._data.bitcast[Float32]()[0], Float32(3.0), tolerance=1e-5)
    assert_almost_equal(result._data.bitcast[Float32]()[1], Float32(2.0), tolerance=1e-5)
    assert_almost_equal(result._data.bitcast[Float32]()[2], Float32(2.0), tolerance=1e-5)


# ============================================================================
# Modulo Tests
# ============================================================================


fn test_modulo_shapes() raises:
    """Test that modulo returns correct output shape."""
    var shape = List[Int](2)
    shape[0] = 4
    shape[1] = 10
    var a = ones(shape, DType.float32)
    var b = ones(shape, DType.float32)

    var result = modulo(a, b)

    assert_equal(result.shape()[0], 4)
    assert_equal(result.shape()[1], 10)


fn test_modulo_values() raises:
    """Test that modulo computes correct values."""
    var shape = List[Int](1)
    shape[0] = 3
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)

    a._data.bitcast[Float32]()[0] = 7.0
    a._data.bitcast[Float32]()[1] = 8.0
    a._data.bitcast[Float32]()[2] = 9.0

    b._data.bitcast[Float32]()[0] = 3.0
    b._data.bitcast[Float32]()[1] = 5.0
    b._data.bitcast[Float32]()[2] = 4.0

    var result = modulo(a, b)

    # 7 % 3 = 1, 8 % 5 = 3, 9 % 4 = 1
    assert_almost_equal(result._data.bitcast[Float32]()[0], Float32(1.0), tolerance=1e-5)
    assert_almost_equal(result._data.bitcast[Float32]()[1], Float32(3.0), tolerance=1e-5)
    assert_almost_equal(result._data.bitcast[Float32]()[2], Float32(1.0), tolerance=1e-5)


# ============================================================================
# Power Tests
# ============================================================================


fn test_power_shapes() raises:
    """Test that power returns correct output shape."""
    var shape = List[Int](2)
    shape[0] = 4
    shape[1] = 10
    var a = ones(shape, DType.float32)
    var b = ones(shape, DType.float32)

    var result = power(a, b)

    assert_equal(result.shape()[0], 4)
    assert_equal(result.shape()[1], 10)


fn test_power_values() raises:
    """Test that power computes correct values."""
    var shape = List[Int](1)
    shape[0] = 3
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)

    a._data.bitcast[Float32]()[0] = 2.0
    a._data.bitcast[Float32]()[1] = 3.0
    a._data.bitcast[Float32]()[2] = 4.0

    b._data.bitcast[Float32]()[0] = 3.0
    b._data.bitcast[Float32]()[1] = 2.0
    b._data.bitcast[Float32]()[2] = 0.5

    var result = power(a, b)

    # 2^3 = 8, 3^2 = 9, 4^0.5 = 2
    assert_almost_equal(result._data.bitcast[Float32]()[0], Float32(8.0), tolerance=1e-5)
    assert_almost_equal(result._data.bitcast[Float32]()[1], Float32(9.0), tolerance=1e-5)
    assert_almost_equal(result._data.bitcast[Float32]()[2], Float32(2.0), tolerance=1e-5)


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run all arithmetic tests."""
    print("Running arithmetic tests...")

    # Addition tests
    test_add_shapes()
    print("✓ test_add_shapes")

    test_add_values()
    print("✓ test_add_values")

    test_add_backward()
    print("✓ test_add_backward")

    # Subtraction tests
    test_subtract_shapes()
    print("✓ test_subtract_shapes")

    test_subtract_values()
    print("✓ test_subtract_values")

    test_subtract_backward()
    print("✓ test_subtract_backward")

    # Multiplication tests
    test_multiply_shapes()
    print("✓ test_multiply_shapes")

    test_multiply_values()
    print("✓ test_multiply_values")

    test_multiply_backward()
    print("✓ test_multiply_backward")

    # Division tests
    test_divide_shapes()
    print("✓ test_divide_shapes")

    test_divide_values()
    print("✓ test_divide_values")

    test_divide_backward()
    print("✓ test_divide_backward")

    # Floor division tests
    test_floor_divide_shapes()
    print("✓ test_floor_divide_shapes")

    test_floor_divide_values()
    print("✓ test_floor_divide_values")

    # Modulo tests
    test_modulo_shapes()
    print("✓ test_modulo_shapes")

    test_modulo_values()
    print("✓ test_modulo_values")

    # Power tests
    test_power_shapes()
    print("✓ test_power_shapes")

    test_power_values()
    print("✓ test_power_values")

    print("\nAll arithmetic tests passed!")
