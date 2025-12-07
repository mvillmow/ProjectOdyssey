"""Tests for basic tensor operations and properties.

Tests cover:
- Tensor creation (zeros, ones, full, empty, arange, eye, linspace)
- Tensor properties (shape, dtype, numel, dim)
- Tensor copying (zeros_like, ones_like, full_like)
- Memory management and views

All tests use pure functional API.
"""

from tests.shared.conftest import (
    assert_true,
    assert_equal_int,
    assert_close_float,
    assert_equal,
    assert_almost_equal,
    assert_dtype_equal,
)
from tests.shared.conftest import TestFixtures
from shared.core.extensor import (
    ExTensor,
    zeros,
    ones,
    full,
    empty,
    arange,
    eye,
    linspace,
    zeros_like,
    ones_like,
    full_like,
)


# ============================================================================
# Tensor Creation Tests
# ============================================================================


fn test_zeros_creation() raises:
    """Test zeros tensor creation."""
    var shape= List[Int]()
    shape.append(3)
    shape.append(4)
    var t = zeros(shape, DType.float32)

    # Check shape
    assert_equal_int(t.shape()[0], 3)
    assert_equal_int(t.shape()[1], 4)

    # Check all values are zero
    for i in range(12):
        assert_close_float(
            Float64(t._data.bitcast[Float32]()[i]), Float64(0.0), atol=1e-5
        )


fn test_ones_creation() raises:
    """Test ones tensor creation."""
    var shape= List[Int]()
    shape.append(2)
    shape.append(3)
    var t = ones(shape, DType.float32)

    # Check shape
    assert_equal(t.shape()[0], 2)
    assert_equal(t.shape()[1], 3)

    # Check all values are one
    for i in range(6):
        assert_almost_equal(
            t._data.bitcast[Float32]()[i], Float32(1.0), tolerance=1e-5
        )


fn test_full_creation() raises:
    """Test full tensor creation with specified value."""
    var shape= List[Int]()
    shape.append(2)
    shape.append(2)
    var t = full(shape, 3.14, DType.float32)

    # Check shape
    assert_equal(t.shape()[0], 2)
    assert_equal(t.shape()[1], 2)

    # Check all values are 3.14
    for i in range(4):
        assert_almost_equal(
            t._data.bitcast[Float32]()[i], Float32(3.14), tolerance=1e-5
        )


fn test_empty_creation() raises:
    """Test empty tensor creation (uninitialized memory)."""
    var shape= List[Int]()
    shape.append(3)
    shape.append(3)
    var t = empty(shape, DType.float32)

    # Check shape is correct (values are uninitialized)
    assert_equal(t.shape()[0], 3)
    assert_equal(t.shape()[1], 3)
    assert_equal(t.numel(), 9)


fn test_arange_creation() raises:
    """Test arange tensor creation (sequential values)."""
    var t = arange(0, 10, 1, DType.float32)

    # Check values: [0, 1, 2, ..., 9]
    assert_equal(t.numel(), 10)
    for i in range(10):
        assert_almost_equal(
            t._data.bitcast[Float32]()[i], Float32(i), tolerance=1e-5
        )


fn test_arange_with_step() raises:
    """Test arange with non-unit step."""
    var t = arange(0, 10, 2, DType.float32)

    # Check values: [0, 2, 4, 6, 8]
    assert_equal(t.numel(), 5)
    assert_almost_equal(
        t._data.bitcast[Float32]()[0], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        t._data.bitcast[Float32]()[1], Float32(2.0), tolerance=1e-5
    )
    assert_almost_equal(
        t._data.bitcast[Float32]()[2], Float32(4.0), tolerance=1e-5
    )
    assert_almost_equal(
        t._data.bitcast[Float32]()[3], Float32(6.0), tolerance=1e-5
    )
    assert_almost_equal(
        t._data.bitcast[Float32]()[4], Float32(8.0), tolerance=1e-5
    )


fn test_eye_creation() raises:
    """Test identity matrix creation."""
    var t = eye(3, 3, 0, DType.float32)

    # Check shape (3x3)
    assert_equal(t.shape()[0], 3)
    assert_equal(t.shape()[1], 3)

    # Check diagonal is 1, off-diagonal is 0
    for i in range(3):
        for j in range(3):
            var idx = i * 3 + j
            var val = t._data.bitcast[Float32]()[idx]
            if i == j:
                assert_almost_equal(val, Float32(1.0), tolerance=1e-5)
            else:
                assert_almost_equal(val, Float32(0.0), tolerance=1e-5)


fn test_linspace_creation() raises:
    """Test linspace tensor creation (evenly spaced values)."""
    var t = linspace(0.0, 1.0, 5, DType.float32)

    # Check values: [0.0, 0.25, 0.5, 0.75, 1.0]
    assert_equal(t.numel(), 5)
    assert_almost_equal(
        t._data.bitcast[Float32]()[0], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        t._data.bitcast[Float32]()[1], Float32(0.25), tolerance=1e-5
    )
    assert_almost_equal(
        t._data.bitcast[Float32]()[2], Float32(0.5), tolerance=1e-5
    )
    assert_almost_equal(
        t._data.bitcast[Float32]()[3], Float32(0.75), tolerance=1e-5
    )
    assert_almost_equal(
        t._data.bitcast[Float32]()[4], Float32(1.0), tolerance=1e-5
    )


# ============================================================================
# Tensor Copying Tests
# ============================================================================


fn test_zeros_like() raises:
    """Test zeros_like creates tensor with same shape and dtype."""
    var shape= List[Int]()
    shape.append(2)
    shape.append(3)
    var x = full(shape, 5.0, DType.float32)
    var y = zeros_like(x)

    # Check shape matches
    assert_equal(y.shape()[0], 2)
    assert_equal(y.shape()[1], 3)

    # Check dtype matches
    assert_dtype_equal(y.dtype(), DType.float32)

    # Check all values are zero
    for i in range(6):
        assert_almost_equal(
            y._data.bitcast[Float32]()[i], Float32(0.0), tolerance=1e-5
        )


fn test_ones_like() raises:
    """Test ones_like creates tensor with same shape and dtype."""
    var shape= List[Int]()
    shape.append(2)
    shape.append(2)
    shape.append(2)
    var x = zeros(shape, DType.float64)
    var y = ones_like(x)

    # Check shape matches
    assert_equal(y.shape()[0], 2)
    assert_equal(y.shape()[1], 2)
    assert_equal(y.shape()[2], 2)

    # Check dtype matches
    assert_dtype_equal(y.dtype(), DType.float64)

    # Check all values are one
    for i in range(8):
        assert_almost_equal(y._data.bitcast[Float64]()[i], 1.0, tolerance=1e-10)


fn test_full_like() raises:
    """Test full_like creates tensor with same shape and dtype."""
    var shape= List[Int]()
    shape.append(3)
    shape.append(2)
    var x = zeros(shape, DType.float32)
    var y = full_like(x, 7.5)

    # Check shape matches
    assert_equal(y.shape()[0], 3)
    assert_equal(y.shape()[1], 2)

    # Check dtype matches
    assert_dtype_equal(y.dtype(), DType.float32)

    # Check all values are 7.5
    for i in range(6):
        assert_almost_equal(
            y._data.bitcast[Float32]()[i], Float32(7.5), tolerance=1e-5
        )


# ============================================================================
# Tensor Properties Tests
# ============================================================================


fn test_tensor_shape() raises:
    """Test tensor shape property."""
    var shape= List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)
    var t = zeros(shape, DType.float32)

    var t_shape = t.shape()
    assert_equal(len(t_shape), 3)
    assert_equal(t_shape[0], 2)
    assert_equal(t_shape[1], 3)
    assert_equal(t_shape[2], 4)


fn test_tensor_dtype() raises:
    """Test tensor dtype property."""
    var shape= List[Int]()
    shape.append(5)

    var t_f32 = zeros(shape, DType.float32)
    assert_dtype_equal(t_f32.dtype(), DType.float32)

    var t_f64 = zeros(shape, DType.float64)
    assert_dtype_equal(t_f64.dtype(), DType.float64)

    var t_i32 = zeros(shape, DType.int32)
    assert_dtype_equal(t_i32.dtype(), DType.int32)


fn test_tensor_numel() raises:
    """Test tensor numel (number of elements)."""
    var shape1= List[Int]()
    shape1.append(10)
    var t1 = zeros(shape1, DType.float32)
    assert_equal(t1.numel(), 10)

    var shape2= List[Int]()
    shape2.append(3)
    shape2.append(4)
    var t2 = zeros(shape2, DType.float32)
    assert_equal(t2.numel(), 12)

    var shape3= List[Int]()
    shape3.append(2)
    shape3.append(3)
    shape3.append(4)
    var t3 = zeros(shape3, DType.float32)
    assert_equal(t3.numel(), 24)


fn test_tensor_dim() raises:
    """Test tensor dim (number of dimensions)."""
    var shape1= List[Int]()
    shape1.append(10)
    var t1 = zeros(shape1, DType.float32)
    assert_equal(t1.dim(), 1)

    var shape2= List[Int]()
    shape2.append(3)
    shape2.append(4)
    var t2 = zeros(shape2, DType.float32)
    assert_equal(t2.dim(), 2)

    var shape3= List[Int]()
    shape3.append(2)
    shape3.append(3)
    shape3.append(4)
    shape3.append(5)
    var t3 = zeros(shape3, DType.float32)
    assert_equal(t3.dim(), 4)


# ============================================================================
# Dtype Support Tests
# ============================================================================


fn test_float_dtypes() raises:
    """Test tensor creation with different float dtypes."""
    var shape= List[Int]()
    shape.append(3)

    # float16
    var t_f16 = zeros(shape, DType.float16)
    assert_dtype_equal(t_f16.dtype(), DType.float16)
    assert_equal(t_f16.numel(), 3)

    # float32
    var t_f32 = zeros(shape, DType.float32)
    assert_dtype_equal(t_f32.dtype(), DType.float32)
    assert_equal(t_f32.numel(), 3)

    # float64
    var t_f64 = zeros(shape, DType.float64)
    assert_dtype_equal(t_f64.dtype(), DType.float64)
    assert_equal(t_f64.numel(), 3)


fn test_int_dtypes() raises:
    """Test tensor creation with different integer dtypes."""
    var shape= List[Int]()
    shape.append(3)

    # int8
    var t_i8 = zeros(shape, DType.int8)
    assert_dtype_equal(t_i8.dtype(), DType.int8)
    assert_equal(t_i8.numel(), 3)

    # int16
    var t_i16 = zeros(shape, DType.int16)
    assert_dtype_equal(t_i16.dtype(), DType.int16)
    assert_equal(t_i16.numel(), 3)

    # int32
    var t_i32 = zeros(shape, DType.int32)
    assert_dtype_equal(t_i32.dtype(), DType.int32)
    assert_equal(t_i32.numel(), 3)

    # int64
    var t_i64 = zeros(shape, DType.int64)
    assert_dtype_equal(t_i64.dtype(), DType.int64)
    assert_equal(t_i64.numel(), 3)


fn test_uint_dtypes() raises:
    """Test tensor creation with different unsigned integer dtypes."""
    var shape= List[Int]()
    shape.append(3)

    # uint8
    var t_u8 = zeros(shape, DType.uint8)
    assert_dtype_equal(t_u8.dtype(), DType.uint8)
    assert_equal(t_u8.numel(), 3)

    # uint16
    var t_u16 = zeros(shape, DType.uint16)
    assert_dtype_equal(t_u16.dtype(), DType.uint16)
    assert_equal(t_u16.numel(), 3)

    # uint32
    var t_u32 = zeros(shape, DType.uint32)
    assert_dtype_equal(t_u32.dtype(), DType.uint32)
    assert_equal(t_u32.numel(), 3)

    # uint64
    var t_u64 = zeros(shape, DType.uint64)
    assert_dtype_equal(t_u64.dtype(), DType.uint64)
    assert_equal(t_u64.numel(), 3)


# ============================================================================
# Edge Case Tests
# ============================================================================


fn test_scalar_tensor() raises:
    """Test scalar tensor (0 dimensions, single element)."""
    var shape= List[Int]()
    shape.append(1)
    var t = full(shape, 42.0, DType.float32)

    assert_equal(t.numel(), 1)
    assert_equal(t.dim(), 1)
    assert_almost_equal(
        t._data.bitcast[Float32]()[0], Float32(42.0), tolerance=1e-5
    )


fn test_large_tensor() raises:
    """Test large tensor creation."""
    var shape= List[Int]()
    shape.append(10)
    shape.append(20)
    shape.append(30)
    var t = zeros(shape, DType.float32)

    assert_equal(t.shape()[0], 10)
    assert_equal(t.shape()[1], 20)
    assert_equal(t.shape()[2], 30)
    assert_equal(t.numel(), 6000)
    assert_equal(t.dim(), 3)


fn test_high_dimensional_tensor() raises:
    """Test high-dimensional tensor (4D, 5D, 6D)."""
    # 4D tensor
    var shape4= List[Int]()
    shape4.append(2)
    shape4.append(3)
    shape4.append(4)
    shape4.append(5)
    var t4 = zeros(shape4, DType.float32)
    assert_equal(t4.dim(), 4)
    assert_equal(t4.numel(), 120)

    # 5D tensor
    var shape5= List[Int]()
    for i in range(5):
        shape5.append(2)
    var t5 = zeros(shape5, DType.float32)
    assert_equal(t5.dim(), 5)
    assert_equal(t5.numel(), 32)


# ============================================================================
# Value Setting and Getting Tests
# ============================================================================


fn test_set_and_get_values() raises:
    """Test setting and getting individual values."""
    var shape= List[Int]()
    shape.append(5)
    var t = zeros(shape, DType.float32)

    # Set values
    t._data.bitcast[Float32]()[0] = 1.0
    t._data.bitcast[Float32]()[1] = 2.0
    t._data.bitcast[Float32]()[2] = 3.0
    t._data.bitcast[Float32]()[3] = 4.0
    t._data.bitcast[Float32]()[4] = 5.0

    # Get and check values
    assert_almost_equal(
        t._data.bitcast[Float32]()[0], Float32(1.0), tolerance=1e-5
    )
    assert_almost_equal(
        t._data.bitcast[Float32]()[1], Float32(2.0), tolerance=1e-5
    )
    assert_almost_equal(
        t._data.bitcast[Float32]()[2], Float32(3.0), tolerance=1e-5
    )
    assert_almost_equal(
        t._data.bitcast[Float32]()[3], Float32(4.0), tolerance=1e-5
    )
    assert_almost_equal(
        t._data.bitcast[Float32]()[4], Float32(5.0), tolerance=1e-5
    )


fn test_2d_indexing() raises:
    """Test 2D tensor indexing."""
    var shape= List[Int]()
    shape.append(3)
    shape.append(4)
    var t = zeros(shape, DType.float32)

    # Set value at (1, 2) -> linear index = 1*4 + 2 = 6
    t._data.bitcast[Float32]()[6] = 42.0

    # Get value at (1, 2)
    var val = t._data.bitcast[Float32]()[6]
    assert_almost_equal(val, Float32(42.0), tolerance=1e-5)


fn test_3d_indexing() raises:
    """Test 3D tensor indexing."""
    var shape= List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)
    var t = zeros(shape, DType.float32)

    # Set value at (1, 2, 3) -> linear index = 1*12 + 2*4 + 3 = 23
    t._data.bitcast[Float32]()[23] = 99.0

    # Get value at (1, 2, 3)
    var val = t._data.bitcast[Float32]()[23]
    assert_almost_equal(val, Float32(99.0), tolerance=1e-5)


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run all tests in this file."""
    var total = 0
    var passed = 0
    var failed = 0

    print("\n" + "=" * 70)
    print("Running tests from: test_tensors.mojo")
    print("=" * 70 + "\n")

    # test_zeros_creation
    total += 1
    try:
        test_zeros_creation()
        passed += 1
        print("  ✓ test_zeros_creation")
    except e:
        failed += 1
        print("  ✗ test_zeros_creation:", e)

    # test_ones_creation
    total += 1
    try:
        test_ones_creation()
        passed += 1
        print("  ✓ test_ones_creation")
    except e:
        failed += 1
        print("  ✗ test_ones_creation:", e)

    # test_full_creation
    total += 1
    try:
        test_full_creation()
        passed += 1
        print("  ✓ test_full_creation")
    except e:
        failed += 1
        print("  ✗ test_full_creation:", e)

    # test_empty_creation
    total += 1
    try:
        test_empty_creation()
        passed += 1
        print("  ✓ test_empty_creation")
    except e:
        failed += 1
        print("  ✗ test_empty_creation:", e)

    # test_arange_creation
    total += 1
    try:
        test_arange_creation()
        passed += 1
        print("  ✓ test_arange_creation")
    except e:
        failed += 1
        print("  ✗ test_arange_creation:", e)

    # test_arange_with_step
    total += 1
    try:
        test_arange_with_step()
        passed += 1
        print("  ✓ test_arange_with_step")
    except e:
        failed += 1
        print("  ✗ test_arange_with_step:", e)

    # test_eye_creation
    total += 1
    try:
        test_eye_creation()
        passed += 1
        print("  ✓ test_eye_creation")
    except e:
        failed += 1
        print("  ✗ test_eye_creation:", e)

    # test_linspace_creation
    total += 1
    try:
        test_linspace_creation()
        passed += 1
        print("  ✓ test_linspace_creation")
    except e:
        failed += 1
        print("  ✗ test_linspace_creation:", e)

    # test_zeros_like
    total += 1
    try:
        test_zeros_like()
        passed += 1
        print("  ✓ test_zeros_like")
    except e:
        failed += 1
        print("  ✗ test_zeros_like:", e)

    # test_ones_like
    total += 1
    try:
        test_ones_like()
        passed += 1
        print("  ✓ test_ones_like")
    except e:
        failed += 1
        print("  ✗ test_ones_like:", e)

    # test_full_like
    total += 1
    try:
        test_full_like()
        passed += 1
        print("  ✓ test_full_like")
    except e:
        failed += 1
        print("  ✗ test_full_like:", e)

    # test_tensor_shape
    total += 1
    try:
        test_tensor_shape()
        passed += 1
        print("  ✓ test_tensor_shape")
    except e:
        failed += 1
        print("  ✗ test_tensor_shape:", e)

    # test_tensor_dtype
    total += 1
    try:
        test_tensor_dtype()
        passed += 1
        print("  ✓ test_tensor_dtype")
    except e:
        failed += 1
        print("  ✗ test_tensor_dtype:", e)

    # test_tensor_numel
    total += 1
    try:
        test_tensor_numel()
        passed += 1
        print("  ✓ test_tensor_numel")
    except e:
        failed += 1
        print("  ✗ test_tensor_numel:", e)

    # test_tensor_dim
    total += 1
    try:
        test_tensor_dim()
        passed += 1
        print("  ✓ test_tensor_dim")
    except e:
        failed += 1
        print("  ✗ test_tensor_dim:", e)

    # test_float_dtypes
    total += 1
    try:
        test_float_dtypes()
        passed += 1
        print("  ✓ test_float_dtypes")
    except e:
        failed += 1
        print("  ✗ test_float_dtypes:", e)

    # test_int_dtypes
    total += 1
    try:
        test_int_dtypes()
        passed += 1
        print("  ✓ test_int_dtypes")
    except e:
        failed += 1
        print("  ✗ test_int_dtypes:", e)

    # test_uint_dtypes
    total += 1
    try:
        test_uint_dtypes()
        passed += 1
        print("  ✓ test_uint_dtypes")
    except e:
        failed += 1
        print("  ✗ test_uint_dtypes:", e)

    # test_scalar_tensor
    total += 1
    try:
        test_scalar_tensor()
        passed += 1
        print("  ✓ test_scalar_tensor")
    except e:
        failed += 1
        print("  ✗ test_scalar_tensor:", e)

    # test_large_tensor
    total += 1
    try:
        test_large_tensor()
        passed += 1
        print("  ✓ test_large_tensor")
    except e:
        failed += 1
        print("  ✗ test_large_tensor:", e)

    # test_high_dimensional_tensor
    total += 1
    try:
        test_high_dimensional_tensor()
        passed += 1
        print("  ✓ test_high_dimensional_tensor")
    except e:
        failed += 1
        print("  ✗ test_high_dimensional_tensor:", e)

    # test_set_and_get_values
    total += 1
    try:
        test_set_and_get_values()
        passed += 1
        print("  ✓ test_set_and_get_values")
    except e:
        failed += 1
        print("  ✗ test_set_and_get_values:", e)

    # test_2d_indexing
    total += 1
    try:
        test_2d_indexing()
        passed += 1
        print("  ✓ test_2d_indexing")
    except e:
        failed += 1
        print("  ✗ test_2d_indexing:", e)

    # test_3d_indexing
    total += 1
    try:
        test_3d_indexing()
        passed += 1
        print("  ✓ test_3d_indexing")
    except e:
        failed += 1
        print("  ✗ test_3d_indexing:", e)

    # Summary
    print("\n" + "=" * 70)
    print("Results:", passed, "/", total, "passed,", failed, "failed")
    print("=" * 70)

    if failed > 0:
        raise Error("Tests failed")
