"""Tests for BF8 data type and tensor conversions.

Tests cover:
- BF8 creation from Float32
- BF8 to Float32 conversion
- Special values (zero, NaN, Inf)
- Range clamping (±57344 max value)
- Tensor conversion (to_bf8, from_bf8)
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
from shared.core.types.bf8 import BF8
from math import isnan, isinf


# ============================================================================
# BF8 Basic Conversion Tests
# ============================================================================


fn test_bf8_zero() raises:
    """Test BF8 representation of zero."""
    var bf8_zero = BF8.from_float32(0.0)
    var result = bf8_zero.to_float32()

    assert_equal(bf8_zero.value, 0)
    assert_almost_equal(result, Float32(0.0), tolerance=1e-7)


fn test_bf8_positive_values() raises:
    """Test BF8 encoding of positive values."""
    # Test small positive value
    var bf8_small = BF8.from_float32(1.0)
    var result_small = bf8_small.to_float32()
    assert_almost_equal(result_small, Float32(1.0), tolerance=0.3)

    # Test medium positive value
    var bf8_medium = BF8.from_float32(10.0)
    var result_medium = bf8_medium.to_float32()
    assert_almost_equal(result_medium, Float32(10.0), tolerance=3.0)

    # Test large positive value
    var bf8_large = BF8.from_float32(100.0)
    var result_large = bf8_large.to_float32()
    assert_almost_equal(result_large, Float32(100.0), tolerance=30.0)

    # Test very large positive value (within BF8 range)
    var bf8_vlarge = BF8.from_float32(1000.0)
    var result_vlarge = bf8_vlarge.to_float32()
    assert_almost_equal(result_vlarge, Float32(1000.0), tolerance=300.0)


fn test_bf8_negative_values() raises:
    """Test BF8 encoding of negative values."""
    # Test small negative value
    var bf8_small = BF8.from_float32(-1.0)
    var result_small = bf8_small.to_float32()
    assert_almost_equal(result_small, Float32(-1.0), tolerance=0.3)

    # Test medium negative value
    var bf8_medium = BF8.from_float32(-10.0)
    var result_medium = bf8_medium.to_float32()
    assert_almost_equal(result_medium, Float32(-10.0), tolerance=3.0)

    # Test large negative value
    var bf8_large = BF8.from_float32(-100.0)
    var result_large = bf8_large.to_float32()
    assert_almost_equal(result_large, Float32(-100.0), tolerance=30.0)

    # Test very large negative value (within BF8 range)
    var bf8_vlarge = BF8.from_float32(-1000.0)
    var result_vlarge = bf8_vlarge.to_float32()
    assert_almost_equal(result_vlarge, Float32(-1000.0), tolerance=300.0)


fn test_bf8_range_clamping() raises:
    """Test BF8 clamping of values outside representable range."""
    # BF8 E5M2 max value is approximately 57344

    # Test positive overflow
    var bf8_overflow = BF8.from_float32(100000.0)
    var result_overflow = bf8_overflow.to_float32()
    assert_true(
        result_overflow <= 57344.0, "BF8 should clamp large positive values"
    )
    assert_true(result_overflow > 50000.0, "BF8 max should be near 57344")

    # Test negative overflow
    var bf8_underflow = BF8.from_float32(-100000.0)
    var result_underflow = bf8_underflow.to_float32()
    assert_true(
        result_underflow >= -57344.0, "BF8 should clamp large negative values"
    )
    assert_true(result_underflow < -50000.0, "BF8 min should be near -57344")


fn test_bf8_subnormal_values() raises:
    """Test BF8 encoding of very small (subnormal) values."""
    # BF8 E5M2 min normal value is 2^-14 = 0.00006103515625

    # Test value in subnormal range
    var bf8_tiny = BF8.from_float32(0.00005)
    var result_tiny = bf8_tiny.to_float32()
    assert_true(result_tiny >= 0.0, "BF8 subnormal should be non-negative")
    assert_true(result_tiny < 0.0001, "BF8 subnormal should be small")

    # Test value below subnormal range (should be zero)
    var bf8_very_tiny = BF8.from_float32(0.00001)
    var result_very_tiny = bf8_very_tiny.to_float32()
    assert_almost_equal(result_very_tiny, Float32(0.0), tolerance=1e-7)


fn test_bf8_special_values_nan() raises:
    """Test BF8 encoding of NaN."""
    var nan_val = Float32(0.0) / Float32(0.0)
    var bf8_nan = BF8.from_float32(nan_val)
    var result = bf8_nan.to_float32()

    # Check that result is NaN
    assert_true(isnan(result), "BF8 should preserve NaN")


fn test_bf8_special_values_inf() raises:
    """Test BF8 encoding of infinity."""
    # Positive infinity
    var pos_inf = Float32(1.0) / Float32(0.0)
    var bf8_pos_inf = BF8.from_float32(pos_inf)
    var result_pos = bf8_pos_inf.to_float32()
    assert_true(
        isinf(result_pos) and result_pos > 0, "BF8 should preserve +Inf"
    )

    # Negative infinity
    var neg_inf = Float32(-1.0) / Float32(0.0)
    var bf8_neg_inf = BF8.from_float32(neg_inf)
    var result_neg = bf8_neg_inf.to_float32()
    assert_true(
        isinf(result_neg) and result_neg < 0, "BF8 should preserve -Inf"
    )


fn test_bf8_equality() raises:
    """Test BF8 equality comparison."""
    var bf8_a = BF8.from_float32(3.14)
    var bf8_b = BF8.from_float32(3.14)
    var bf8_c = BF8.from_float32(2.71)

    assert_true(bf8_a == bf8_b, "Equal BF8 values should compare equal")
    assert_true(bf8_a != bf8_c, "Different BF8 values should compare not equal")


# ============================================================================
# Tensor Conversion Tests
# ============================================================================


fn test_tensor_to_bf8() raises:
    """Test converting Float32 tensor to BF8."""
    var shape= List[Int]()
    shape.append(2)
    shape.append(3)

    # Create Float32 tensor with specific values
    var t = zeros(shape, DType.float32)
    t._data.bitcast[Float32]()[0] = 1.0
    t._data.bitcast[Float32]()[1] = -2.5
    t._data.bitcast[Float32]()[2] = 10.0
    t._data.bitcast[Float32]()[3] = -5.0
    t._data.bitcast[Float32]()[4] = 0.5
    t._data.bitcast[Float32]()[5] = 1000.0

    # Convert to BF8
    var bf8_tensor = t.to_bf8()

    # Check dtype is uint8
    assert_true(
        bf8_tensor.dtype() == DType.uint8, "BF8 tensor should have uint8 dtype"
    )

    # Check shape is preserved
    assert_shape_equal(bf8_tensor.shape(), t.shape())

    # Check that values are encoded (not zero)
    var has_nonzero = False
    for i in range(6):
        if bf8_tensor._data.bitcast[UInt8]()[i] != 0:
            has_nonzero = True
    assert_true(has_nonzero, "BF8 tensor should have encoded values")


fn test_tensor_from_bf8() raises:
    """Test converting BF8 tensor back to Float32."""
    var shape= List[Int]()
    shape.append(2)
    shape.append(2)

    # Create Float32 tensor
    var original = ones(shape, DType.float32)
    original._data.bitcast[Float32]()[0] = 3.0
    original._data.bitcast[Float32]()[1] = -1.5
    original._data.bitcast[Float32]()[2] = 7.0
    original._data.bitcast[Float32]()[3] = -10.0

    # Convert to BF8 and back
    var bf8_tensor = original.to_bf8()
    var restored = bf8_tensor.from_bf8()

    # Check dtype is float32
    assert_true(
        restored.dtype() == DType.float32,
        "Restored tensor should have float32 dtype",
    )

    # Check shape is preserved
    assert_shape_equal(restored.shape(), original.shape())

    # Check values are approximately restored (with BF8 precision loss)
    # BF8 has less precision than FP8, so use larger tolerances
    assert_almost_equal(
        restored._data.bitcast[Float32]()[0], Float32(3.0), tolerance=1.0
    )
    assert_almost_equal(
        restored._data.bitcast[Float32]()[1], Float32(-1.5), tolerance=0.5
    )
    assert_almost_equal(
        restored._data.bitcast[Float32]()[2], Float32(7.0), tolerance=2.0
    )
    assert_almost_equal(
        restored._data.bitcast[Float32]()[3], Float32(-10.0), tolerance=3.0
    )


fn test_tensor_bf8_roundtrip() raises:
    """Test round-trip conversion Float32 -> BF8 -> Float32."""
    var shape= List[Int]()
    shape.append(5)

    # Create tensor with various values
    var original = zeros(shape, DType.float32)
    original._data.bitcast[Float32]()[0] = 0.0
    original._data.bitcast[Float32]()[1] = 1.0
    original._data.bitcast[Float32]()[2] = -5.0
    original._data.bitcast[Float32]()[3] = 20.0
    original._data.bitcast[Float32]()[4] = -50.0

    # Round-trip conversion
    var bf8_tensor = original.to_bf8()
    var restored = bf8_tensor.from_bf8()

    # Verify approximate equality (accounting for BF8 precision loss)
    # BF8 has only 2 mantissa bits, so precision loss is significant
    for i in range(5):
        var orig_val = original._data.bitcast[Float32]()[i]
        var restored_val = restored._data.bitcast[Float32]()[i]

        # Use tolerance proportional to magnitude (larger than FP8)
        var tolerance = max(abs(orig_val) * 0.25, Float32(0.5))
        assert_almost_equal(restored_val, orig_val, tolerance=tolerance)


fn test_tensor_to_bf8_requires_float() raises:
    """Test that to_bf8() requires floating-point tensor."""
    var shape= List[Int]()
    shape.append(3)

    # Create int32 tensor
    var int_tensor = zeros(shape, DType.int32)

    # Try to convert to BF8 (should raise error)
    var raised_error = False
    try:
        var _ = int_tensor.to_bf8()
    except:
        raised_error = True

    assert_true(
        raised_error, "to_bf8() should raise error for non-float tensor"
    )


fn test_tensor_from_bf8_requires_uint8() raises:
    """Test that from_bf8() requires uint8 tensor."""
    var shape= List[Int]()
    shape.append(3)

    # Create float32 tensor (not uint8)
    var float_tensor = zeros(shape, DType.float32)

    # Try to convert from BF8 (should raise error)
    var raised_error = False
    try:
        var _ = float_tensor.from_bf8()
    except:
        raised_error = True

    assert_true(
        raised_error, "from_bf8() should raise error for non-uint8 tensor"
    )


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run all BF8 tests."""
    print("\n=== BF8 Basic Conversion Tests ===")
    test_bf8_zero()
    print("✓ BF8 zero encoding")

    test_bf8_positive_values()
    print("✓ BF8 positive values")

    test_bf8_negative_values()
    print("✓ BF8 negative values")

    test_bf8_range_clamping()
    print("✓ BF8 range clamping")

    test_bf8_subnormal_values()
    print("✓ BF8 subnormal values")

    test_bf8_special_values_nan()
    print("✓ BF8 NaN handling")

    test_bf8_special_values_inf()
    print("✓ BF8 infinity handling")

    test_bf8_equality()
    print("✓ BF8 equality comparison")

    print("\n=== Tensor Conversion Tests ===")
    test_tensor_to_bf8()
    print("✓ Tensor to BF8 conversion")

    test_tensor_from_bf8()
    print("✓ Tensor from BF8 conversion")

    test_tensor_bf8_roundtrip()
    print("✓ Tensor BF8 round-trip")

    test_tensor_to_bf8_requires_float()
    print("✓ to_bf8() type validation")

    test_tensor_from_bf8_requires_uint8()
    print("✓ from_bf8() type validation")

    print("\n=== All BF8 Tests Passed! ===\n")
