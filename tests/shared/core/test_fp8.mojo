"""Tests for FP8 data type and tensor conversions.

Tests cover:
- FP8 creation from Float32
- FP8 to Float32 conversion
- Special values (zero, NaN, Inf)
- Range clamping (±240 max value)
- Tensor conversion (to_fp8, from_fp8)
- Round-trip conversion accuracy
"""

from tests.shared.conftest import (
    assert_almost_equal,
    assert_close_float,
    assert_equal,
    assert_equal_int,
    assert_shape,
    assert_shape_equal,
    assert_true,
)
from tests.shared.conftest import TestFixtures
from shared.core.extensor import (
    ExTensor,
    zeros,
    ones,
    full,
)
from shared.core.types.fp8 import FP8
from math import isnan, isinf


# ============================================================================
# FP8 Basic Conversion Tests
# ============================================================================


fn test_fp8_zero() raises:
    """Test FP8 representation of zero."""
    var fp8_zero = FP8.from_float32(0.0)
    var result = fp8_zero.to_float32()

    assert_equal(fp8_zero.value, 0)
    assert_almost_equal(result, Float32(0.0), tolerance=1e-7)


fn test_fp8_positive_values() raises:
    """Test FP8 encoding of positive values."""
    # Test small positive value
    var fp8_small = FP8.from_float32(1.0)
    var result_small = fp8_small.to_float32()
    assert_almost_equal(result_small, Float32(1.0), tolerance=0.2)

    # Test medium positive value
    var fp8_medium = FP8.from_float32(10.0)
    var result_medium = fp8_medium.to_float32()
    assert_almost_equal(result_medium, Float32(10.0), tolerance=2.0)

    # Test large positive value
    var fp8_large = FP8.from_float32(100.0)
    var result_large = fp8_large.to_float32()
    assert_almost_equal(result_large, Float32(100.0), tolerance=15.0)


fn test_fp8_negative_values() raises:
    """Test FP8 encoding of negative values."""
    # Test small negative value
    var fp8_small = FP8.from_float32(-1.0)
    var result_small = fp8_small.to_float32()
    assert_almost_equal(result_small, Float32(-1.0), tolerance=0.2)

    # Test medium negative value
    var fp8_medium = FP8.from_float32(-10.0)
    var result_medium = fp8_medium.to_float32()
    assert_almost_equal(result_medium, Float32(-10.0), tolerance=2.0)

    # Test large negative value
    var fp8_large = FP8.from_float32(-100.0)
    var result_large = fp8_large.to_float32()
    assert_almost_equal(result_large, Float32(-100.0), tolerance=15.0)


fn test_fp8_range_clamping() raises:
    """Test FP8 clamping of values outside representable range."""
    # FP8 E4M3 max value is approximately 240

    # Test positive overflow
    var fp8_overflow = FP8.from_float32(1000.0)
    var result_overflow = fp8_overflow.to_float32()
    assert_true(
        result_overflow <= 240.0, "FP8 should clamp large positive values"
    )
    assert_true(result_overflow > 200.0, "FP8 max should be near 240")

    # Test negative overflow
    var fp8_underflow = FP8.from_float32(-1000.0)
    var result_underflow = fp8_underflow.to_float32()
    assert_true(
        result_underflow >= -240.0, "FP8 should clamp large negative values"
    )
    assert_true(result_underflow < -200.0, "FP8 min should be near -240")


fn test_fp8_subnormal_values() raises:
    """Test FP8 encoding of very small (subnormal) values."""
    # FP8 E4M3 min normal value is 2^-6 = 0.015625

    # Test value in subnormal range
    var fp8_tiny = FP8.from_float32(0.01)
    var result_tiny = fp8_tiny.to_float32()
    assert_true(result_tiny >= 0.0, "FP8 subnormal should be non-negative")
    assert_true(result_tiny < 0.02, "FP8 subnormal should be small")

    # Test value below subnormal range (should be zero)
    var fp8_very_tiny = FP8.from_float32(0.001)
    var result_very_tiny = fp8_very_tiny.to_float32()
    assert_almost_equal(result_very_tiny, Float32(0.0), tolerance=1e-7)


fn test_fp8_special_values_nan() raises:
    """Test FP8 encoding of NaN."""
    var nan_val = Float32(0.0) / Float32(0.0)
    var fp8_nan = FP8.from_float32(nan_val)
    var result = fp8_nan.to_float32()

    # Check that result is NaN
    assert_true(isnan(result), "FP8 should preserve NaN")


fn test_fp8_special_values_inf() raises:
    """Test FP8 encoding of infinity."""
    # Positive infinity
    var pos_inf = Float32(1.0) / Float32(0.0)
    var fp8_pos_inf = FP8.from_float32(pos_inf)
    var result_pos = fp8_pos_inf.to_float32()
    assert_true(
        isinf(result_pos) and result_pos > 0, "FP8 should preserve +Inf"
    )

    # Negative infinity
    var neg_inf = Float32(-1.0) / Float32(0.0)
    var fp8_neg_inf = FP8.from_float32(neg_inf)
    var result_neg = fp8_neg_inf.to_float32()
    assert_true(
        isinf(result_neg) and result_neg < 0, "FP8 should preserve -Inf"
    )


fn test_fp8_equality() raises:
    """Test FP8 equality comparison."""
    var fp8_a = FP8.from_float32(3.14)
    var fp8_b = FP8.from_float32(3.14)
    var fp8_c = FP8.from_float32(2.71)

    assert_true(fp8_a == fp8_b, "Equal FP8 values should compare equal")
    assert_true(fp8_a != fp8_c, "Different FP8 values should compare not equal")


# ============================================================================
# Tensor Conversion Tests
# ============================================================================


fn test_tensor_to_fp8() raises:
    """Test converting Float32 tensor to FP8."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)

    # Create Float32 tensor with specific values
    var t = zeros(shape, DType.float32)
    t._data.bitcast[Float32]()[0] = 1.0
    t._data.bitcast[Float32]()[1] = -2.5
    t._data.bitcast[Float32]()[2] = 10.0
    t._data.bitcast[Float32]()[3] = -5.0
    t._data.bitcast[Float32]()[4] = 0.5
    t._data.bitcast[Float32]()[5] = 100.0

    # Convert to FP8
    var fp8_tensor = t.to_fp8()

    # Check dtype is uint8
    assert_true(
        fp8_tensor.dtype() == DType.uint8, "FP8 tensor should have uint8 dtype"
    )

    # Check shape is preserved
    assert_shape_equal(fp8_tensor.shape(), t.shape())

    # Check that values are encoded (not zero)
    var has_nonzero = False
    for i in range(6):
        if fp8_tensor._data.bitcast[UInt8]()[i] != 0:
            has_nonzero = True
    assert_true(has_nonzero, "FP8 tensor should have encoded values")


fn test_tensor_from_fp8() raises:
    """Test converting FP8 tensor back to Float32."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(2)

    # Create Float32 tensor
    var original = ones(shape, DType.float32)
    original._data.bitcast[Float32]()[0] = 3.0
    original._data.bitcast[Float32]()[1] = -1.5
    original._data.bitcast[Float32]()[2] = 7.0
    original._data.bitcast[Float32]()[3] = -10.0

    # Convert to FP8 and back
    var fp8_tensor = original.to_fp8()
    var restored = fp8_tensor.from_fp8()

    # Check dtype is float32
    assert_true(
        restored.dtype() == DType.float32,
        "Restored tensor should have float32 dtype",
    )

    # Check shape is preserved
    assert_shape_equal(restored.shape(), original.shape())

    # Check values are approximately restored (with FP8 precision loss)
    assert_almost_equal(
        restored._data.bitcast[Float32]()[0], Float32(3.0), tolerance=0.5
    )
    assert_almost_equal(
        restored._data.bitcast[Float32]()[1], Float32(-1.5), tolerance=0.3
    )
    assert_almost_equal(
        restored._data.bitcast[Float32]()[2], Float32(7.0), tolerance=1.0
    )
    assert_almost_equal(
        restored._data.bitcast[Float32]()[3], Float32(-10.0), tolerance=2.0
    )


fn test_tensor_fp8_roundtrip() raises:
    """Test round-trip conversion Float32 -> FP8 -> Float32."""
    var shape = List[Int]()
    shape.append(5)

    # Create tensor with various values
    var original = zeros(shape, DType.float32)
    original._data.bitcast[Float32]()[0] = 0.0
    original._data.bitcast[Float32]()[1] = 1.0
    original._data.bitcast[Float32]()[2] = -5.0
    original._data.bitcast[Float32]()[3] = 20.0
    original._data.bitcast[Float32]()[4] = -50.0

    # Round-trip conversion
    var fp8_tensor = original.to_fp8()
    var restored = fp8_tensor.from_fp8()

    # Verify approximate equality (accounting for FP8 precision loss)
    for i in range(5):
        var orig_val = original._data.bitcast[Float32]()[i]
        var restored_val = restored._data.bitcast[Float32]()[i]

        # Use tolerance proportional to magnitude
        var tolerance = max(abs(orig_val) * 0.15, Float32(0.5))
        assert_almost_equal(restored_val, orig_val, tolerance=tolerance)


fn test_tensor_to_fp8_requires_float() raises:
    """Test that to_fp8() requires floating-point tensor."""
    var shape = List[Int]()
    shape.append(3)

    # Create int32 tensor
    var int_tensor = zeros(shape, DType.int32)

    # Try to convert to FP8 (should raise error)
    var raised_error = False
    try:
        var _ = int_tensor.to_fp8()
    except:
        raised_error = True

    assert_true(
        raised_error, "to_fp8() should raise error for non-float tensor"
    )


fn test_tensor_from_fp8_requires_uint8() raises:
    """Test that from_fp8() requires uint8 tensor."""
    var shape = List[Int]()
    shape.append(3)

    # Create float32 tensor (not uint8)
    var float_tensor = zeros(shape, DType.float32)

    # Try to convert from FP8 (should raise error)
    var raised_error = False
    try:
        var _ = float_tensor.from_fp8()
    except:
        raised_error = True

    assert_true(
        raised_error, "from_fp8() should raise error for non-uint8 tensor"
    )


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run all FP8 tests."""
    print("\n=== FP8 Basic Conversion Tests ===")
    test_fp8_zero()
    print("✓ FP8 zero encoding")

    test_fp8_positive_values()
    print("✓ FP8 positive values")

    test_fp8_negative_values()
    print("✓ FP8 negative values")

    test_fp8_range_clamping()
    print("✓ FP8 range clamping")

    test_fp8_subnormal_values()
    print("✓ FP8 subnormal values")

    test_fp8_special_values_nan()
    print("✓ FP8 NaN handling")

    test_fp8_special_values_inf()
    print("✓ FP8 infinity handling")

    test_fp8_equality()
    print("✓ FP8 equality comparison")

    print("\n=== Tensor Conversion Tests ===")
    test_tensor_to_fp8()
    print("✓ Tensor to FP8 conversion")

    test_tensor_from_fp8()
    print("✓ Tensor from FP8 conversion")

    test_tensor_fp8_roundtrip()
    print("✓ Tensor FP8 round-trip")

    test_tensor_to_fp8_requires_float()
    print("✓ to_fp8() type validation")

    test_tensor_from_fp8_requires_uint8()
    print("✓ from_fp8() type validation")

    print("\n=== All FP8 Tests Passed! ===\n")
