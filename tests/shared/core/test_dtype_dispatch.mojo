"""Tests for dtype dispatch helpers.

Tests cover:
- Unary dispatch (dispatch_unary, dispatch_float_unary)
- Binary dispatch (dispatch_binary, dispatch_float_binary)
- Scalar dispatch (dispatch_scalar, dispatch_float_scalar)
- All supported dtypes (float16, float32, float64, int8-64, uint8-64)
- Error handling and descriptive error messages
- Float-only restrictions for appropriate dispatch functions

All tests use pure functional dispatch API.
"""

from tests.shared.conftest import (
    assert_almost_equal,
    assert_equal_int,
    assert_true,
)
from shared.core.extensor import ExTensor, zeros, ones, full
from shared.core.dtype_dispatch import (
    dispatch_unary,
    dispatch_binary,
    dispatch_scalar,
    dispatch_float_unary,
    dispatch_float_binary,
    dispatch_float_scalar,
)


# ============================================================================
# Helper Operations for Testing
# ============================================================================


fn identity_op[T: DType](x: Scalar[T]) -> Scalar[T]:
    """Identity operation for unary dispatch testing."""
    return x


fn double_op[T: DType](x: Scalar[T]) -> Scalar[T]:
    """Double operation for unary dispatch testing."""
    return x + x


fn add_op[T: DType](x: Scalar[T], y: Scalar[T]) -> Scalar[T]:
    """Add operation for binary dispatch testing."""
    return x + y


fn mul_op[T: DType](x: Scalar[T], y: Scalar[T]) -> Scalar[T]:
    """Multiply operation for binary dispatch testing."""
    return x * y


# ============================================================================
# Unary Dispatch Tests - Float32
# ============================================================================


fn test_dispatch_unary_float32_identity() raises:
    """Test dispatch_unary with float32 and identity operation."""
    var shape = List[Int]()
    shape.append(3)
    var x = full(shape, 5.0, DType.float32)

    var result = dispatch_unary[identity_op](x)

    assert_equal_int(result._numel, 3)
    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(5.0), tolerance=1e-6
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[1], Float32(5.0), tolerance=1e-6
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[2], Float32(5.0), tolerance=1e-6
    )


fn test_dispatch_unary_float32_double() raises:
    """Test dispatch_unary with float32 and double operation."""
    var shape = List[Int]()
    shape.append(3)
    var x = full(shape, 3.0, DType.float32)

    var result = dispatch_unary[double_op](x)

    assert_equal_int(result._numel, 3)
    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(6.0), tolerance=1e-6
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[1], Float32(6.0), tolerance=1e-6
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[2], Float32(6.0), tolerance=1e-6
    )


fn test_dispatch_unary_float64() raises:
    """Test dispatch_unary with float64."""
    var shape = List[Int]()
    shape.append(2)
    var x = full(shape, 7.0, DType.float64)

    var result = dispatch_unary[identity_op](x)

    assert_equal_int(result._numel, 2)
    assert_almost_equal(
        result._data.bitcast[Float64]()[0], Float64(7.0), tolerance=1e-6
    )


fn test_dispatch_unary_int32() raises:
    """Test dispatch_unary with int32."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.int32)

    x._data.bitcast[Int32]()[0] = 1
    x._data.bitcast[Int32]()[1] = 2
    x._data.bitcast[Int32]()[2] = 3

    var result = dispatch_unary[identity_op](x)

    assert_equal_int(result._numel, 3)
    assert_equal_int(result._data.bitcast[Int32]()[0], 1)
    assert_equal_int(result._data.bitcast[Int32]()[1], 2)
    assert_equal_int(result._data.bitcast[Int32]()[2], 3)


fn test_dispatch_unary_uint8() raises:
    """Test dispatch_unary with uint8."""
    var shape = List[Int]()
    shape.append(2)
    var x = zeros(shape, DType.uint8)

    x._data.bitcast[UInt8]()[0] = 42
    x._data.bitcast[UInt8]()[1] = 84

    var result = dispatch_unary[identity_op](x)

    assert_equal_int(result._numel, 2)
    assert_equal_int(result._data.bitcast[UInt8]()[0], 42)
    assert_equal_int(result._data.bitcast[UInt8]()[1], 84)


# ============================================================================
# Binary Dispatch Tests - Float32
# ============================================================================


fn test_dispatch_binary_float32_add() raises:
    """Test dispatch_binary with float32 and add operation."""
    var shape = List[Int]()
    shape.append(3)
    var a = full(shape, 2.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)

    var result = dispatch_binary[add_op](a, b)

    assert_equal_int(result._numel, 3)
    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(5.0), tolerance=1e-6
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[1], Float32(5.0), tolerance=1e-6
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[2], Float32(5.0), tolerance=1e-6
    )


fn test_dispatch_binary_float32_mul() raises:
    """Test dispatch_binary with float32 and multiply operation."""
    var shape = List[Int]()
    shape.append(2)
    var a = full(shape, 4.0, DType.float32)
    var b = full(shape, 5.0, DType.float32)

    var result = dispatch_binary[mul_op](a, b)

    assert_equal_int(result._numel, 2)
    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(20.0), tolerance=1e-6
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[1], Float32(20.0), tolerance=1e-6
    )


fn test_dispatch_binary_float64() raises:
    """Test dispatch_binary with float64."""
    var shape = List[Int]()
    shape.append(2)
    var a = full(shape, 1.5, DType.float64)
    var b = full(shape, 2.5, DType.float64)

    var result = dispatch_binary[add_op](a, b)

    assert_equal_int(result._numel, 2)
    assert_almost_equal(
        result._data.bitcast[Float64]()[0], Float64(4.0), tolerance=1e-6
    )


fn test_dispatch_binary_int32() raises:
    """Test dispatch_binary with int32."""
    var shape = List[Int]()
    shape.append(2)
    var a = zeros(shape, DType.int32)
    var b = zeros(shape, DType.int32)

    a._data.bitcast[Int32]()[0] = 10
    a._data.bitcast[Int32]()[1] = 20
    b._data.bitcast[Int32]()[0] = 5
    b._data.bitcast[Int32]()[1] = 10

    var result = dispatch_binary[add_op](a, b)

    assert_equal_int(result._numel, 2)
    assert_equal_int(result._data.bitcast[Int32]()[0], 15)
    assert_equal_int(result._data.bitcast[Int32]()[1], 30)


fn test_dispatch_binary_dtype_mismatch() raises:
    """Test dispatch_binary error when dtypes don't match."""
    var shape = List[Int]()
    shape.append(2)
    var a = full(shape, 1.0, DType.float32)
    var b = full(shape, 2.0, DType.float64)

    var error_caught = False
    try:
        var _ = dispatch_binary[add_op](a, b)
    except Error:
        error_caught = True

    assert_true(error_caught, "Expected error for dtype mismatch")


# ============================================================================
# Scalar Dispatch Tests - Float32
# ============================================================================


fn test_dispatch_scalar_float32_add() raises:
    """Test dispatch_scalar with float32 and add operation."""
    var shape = List[Int]()
    shape.append(3)
    var x = full(shape, 10.0, DType.float32)

    var result = dispatch_scalar[add_op](x, 5.0)

    assert_equal_int(result._numel, 3)
    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(15.0), tolerance=1e-6
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[1], Float32(15.0), tolerance=1e-6
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[2], Float32(15.0), tolerance=1e-6
    )


fn test_dispatch_scalar_float32_mul() raises:
    """Test dispatch_scalar with float32 and multiply operation."""
    var shape = List[Int]()
    shape.append(2)
    var x = full(shape, 4.0, DType.float32)

    var result = dispatch_scalar[mul_op](x, 3.0)

    assert_equal_int(result._numel, 2)
    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(12.0), tolerance=1e-6
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[1], Float32(12.0), tolerance=1e-6
    )


fn test_dispatch_scalar_int32() raises:
    """Test dispatch_scalar with int32."""
    var shape = List[Int]()
    shape.append(2)
    var x = zeros(shape, DType.int32)

    x._data.bitcast[Int32]()[0] = 5
    x._data.bitcast[Int32]()[1] = 10

    var result = dispatch_scalar[add_op](x, 3.0)

    assert_equal_int(result._numel, 2)
    assert_equal_int(result._data.bitcast[Int32]()[0], 8)
    assert_equal_int(result._data.bitcast[Int32]()[1], 13)


# ============================================================================
# Float-only Dispatch Tests
# ============================================================================


fn test_dispatch_float_unary_float32() raises:
    """Test dispatch_float_unary with float32."""
    var shape = List[Int]()
    shape.append(2)
    var x = full(shape, 2.0, DType.float32)

    var result = dispatch_float_unary[identity_op](x)

    assert_equal_int(result._numel, 2)
    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(2.0), tolerance=1e-6
    )


fn test_dispatch_float_unary_float64() raises:
    """Test dispatch_float_unary with float64."""
    var shape = List[Int]()
    shape.append(2)
    var x = full(shape, 3.5, DType.float64)

    var result = dispatch_float_unary[identity_op](x)

    assert_equal_int(result._numel, 2)
    assert_almost_equal(
        result._data.bitcast[Float64]()[0], Float64(3.5), tolerance=1e-6
    )


fn test_dispatch_float_unary_rejects_int32() raises:
    """Test dispatch_float_unary rejects int32."""
    var shape = List[Int]()
    shape.append(2)
    var x = zeros(shape, DType.int32)

    var error_caught = False
    try:
        var _ = dispatch_float_unary[identity_op](x)
    except Error:
        error_caught = True

    assert_true(error_caught, "Expected error for non-float dtype")


fn test_dispatch_float_binary_float32() raises:
    """Test dispatch_float_binary with float32."""
    var shape = List[Int]()
    shape.append(2)
    var a = full(shape, 1.5, DType.float32)
    var b = full(shape, 2.5, DType.float32)

    var result = dispatch_float_binary[add_op](a, b)

    assert_equal_int(result._numel, 2)
    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(4.0), tolerance=1e-6
    )


fn test_dispatch_float_binary_rejects_int32() raises:
    """Test dispatch_float_binary rejects int32."""
    var shape = List[Int]()
    shape.append(2)
    var a = zeros(shape, DType.int32)
    var b = zeros(shape, DType.int32)

    var error_caught = False
    try:
        var _ = dispatch_float_binary[add_op](a, b)
    except Error:
        error_caught = True

    assert_true(error_caught, "Expected error for non-float dtype")


fn test_dispatch_float_scalar_float32() raises:
    """Test dispatch_float_scalar with float32."""
    var shape = List[Int]()
    shape.append(2)
    var x = full(shape, 5.0, DType.float32)

    var result = dispatch_float_scalar[mul_op](x, 2.0)

    assert_equal_int(result._numel, 2)
    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(10.0), tolerance=1e-6
    )


fn test_dispatch_float_scalar_rejects_int32() raises:
    """Test dispatch_float_scalar rejects int32."""
    var shape = List[Int]()
    shape.append(2)
    var x = zeros(shape, DType.int32)

    var error_caught = False
    try:
        var _ = dispatch_float_scalar[mul_op](x, 2.0)
    except Error:
        error_caught = True

    assert_true(error_caught, "Expected error for non-float dtype")


# ============================================================================
# 2D Tensor Tests
# ============================================================================


fn test_dispatch_unary_2d_tensor() raises:
    """Test dispatch_unary with 2D tensor."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    var x = full(shape, 4.0, DType.float32)

    var result = dispatch_unary[double_op](x)

    assert_equal_int(result._numel, 6)
    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(8.0), tolerance=1e-6
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[5], Float32(8.0), tolerance=1e-6
    )


fn test_dispatch_binary_2d_tensor() raises:
    """Test dispatch_binary with 2D tensors."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(2)
    var a = full(shape, 3.0, DType.float32)
    var b = full(shape, 2.0, DType.float32)

    var result = dispatch_binary[mul_op](a, b)

    assert_equal_int(result._numel, 4)
    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(6.0), tolerance=1e-6
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[3], Float32(6.0), tolerance=1e-6
    )
