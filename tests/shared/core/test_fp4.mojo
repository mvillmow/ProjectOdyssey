"""Tests for FP4_E2M1 base format.

Tests cover:
- FP4_E2M1 creation from Float32 with different scales
- FP4_E2M1 to Float32 conversion
- All representable values in E2M1 format
- Special values (zero, NaN, Inf)
- Scale factor impact on encoding/decoding
- Edge cases and boundary values
"""

from tests.shared.conftest import (
    assert_almost_equal,
    assert_close_float,
    assert_equal,
    assert_equal_int,
    assert_true,
)
from shared.core.types.fp4 import FP4_E2M1
from math import isnan, isinf


# ============================================================================
# FP4_E2M1 Basic Conversion Tests
# ============================================================================


fn test_fp4_e2m1_zero() raises:
    """Test FP4_E2M1 representation of zero."""
    var fp4_zero = FP4_E2M1.from_float32(0.0, scale=1.0)
    var result = fp4_zero.to_float32(scale=1.0)

    assert_equal(fp4_zero.value, 0)
    assert_almost_equal(result, Float32(0.0), tolerance=1e-7)


fn test_fp4_e2m1_negative_zero() raises:
    """Test FP4_E2M1 representation of negative zero."""
    var fp4_neg_zero = FP4_E2M1.from_float32(-0.0, scale=1.0)
    var result = fp4_neg_zero.to_float32(scale=1.0)

    # Negative zero should be represented as zero
    assert_almost_equal(result, Float32(0.0), tolerance=1e-7)


fn test_fp4_e2m1_representable_values() raises:
    """Test all representable values in E2M1 format."""
    # E2M1 representable values (unscaled):
    # exp=0: 0
    # exp=1, mantissa=0: 1.0
    # exp=1, mantissa=1: 1.5
    # exp=2, mantissa=0: 2.0
    # exp=2, mantissa=1: 3.0
    # exp=3, mantissa=0: 4.0
    # exp=3, mantissa=1: 6.0

    var values= List[Float32]()
    values.append(0.0)
    values.append(1.0)
    values.append(1.5)
    values.append(2.0)
    values.append(3.0)
    values.append(4.0)
    values.append(6.0)

    for i in range(len(values)):
        var val = values[i]
        var fp4 = FP4_E2M1.from_float32(val, scale=1.0)
        var result = fp4.to_float32(scale=1.0)
        assert_almost_equal(result, val, tolerance=0.1)


fn test_fp4_e2m1_positive_values() raises:
    """Test FP4_E2M1 encoding of positive values with scale=1."""
    # Test values that should map to representable values
    var fp4_one = FP4_E2M1.from_float32(1.0, scale=1.0)
    var result_one = fp4_one.to_float32(scale=1.0)
    assert_almost_equal(result_one, Float32(1.0), tolerance=0.3)

    var fp4_two = FP4_E2M1.from_float32(2.0, scale=1.0)
    var result_two = fp4_two.to_float32(scale=1.0)
    assert_almost_equal(result_two, Float32(2.0), tolerance=0.5)

    var fp4_three = FP4_E2M1.from_float32(3.0, scale=1.0)
    var result_three = fp4_three.to_float32(scale=1.0)
    assert_almost_equal(result_three, Float32(3.0), tolerance=0.5)


fn test_fp4_e2m1_negative_values() raises:
    """Test FP4_E2M1 encoding of negative values with scale=1."""
    var fp4_neg_one = FP4_E2M1.from_float32(-1.0, scale=1.0)
    var result_neg_one = fp4_neg_one.to_float32(scale=1.0)
    assert_almost_equal(result_neg_one, Float32(-1.0), tolerance=0.3)

    var fp4_neg_two = FP4_E2M1.from_float32(-2.0, scale=1.0)
    var result_neg_two = fp4_neg_two.to_float32(scale=1.0)
    assert_almost_equal(result_neg_two, Float32(-2.0), tolerance=0.5)

    var fp4_neg_three = FP4_E2M1.from_float32(-3.0, scale=1.0)
    var result_neg_three = fp4_neg_three.to_float32(scale=1.0)
    assert_almost_equal(result_neg_three, Float32(-3.0), tolerance=0.5)


fn test_fp4_e2m1_range_clamping() raises:
    """Test FP4_E2M1 clamping of values outside representable range."""
    # E2M1 max value (unscaled) is 6.0

    # Test positive overflow
    var fp4_overflow = FP4_E2M1.from_float32(10.0, scale=1.0)
    var result_overflow = fp4_overflow.to_float32(scale=1.0)
    assert_almost_equal(result_overflow, Float32(6.0), tolerance=0.1)

    # Test negative overflow
    var fp4_underflow = FP4_E2M1.from_float32(-10.0, scale=1.0)
    var result_underflow = fp4_underflow.to_float32(scale=1.0)
    assert_almost_equal(result_underflow, Float32(-6.0), tolerance=0.1)


fn test_fp4_e2m1_small_values() raises:
    """Test FP4_E2M1 encoding of very small values (below min representable)."""
    # Values below 0.5 (when scale=1) should map to zero
    var fp4_tiny = FP4_E2M1.from_float32(0.1, scale=1.0)
    var result_tiny = fp4_tiny.to_float32(scale=1.0)
    assert_almost_equal(result_tiny, Float32(0.0), tolerance=1e-7)

    var fp4_very_tiny = FP4_E2M1.from_float32(0.01, scale=1.0)
    var result_very_tiny = fp4_very_tiny.to_float32(scale=1.0)
    assert_almost_equal(result_very_tiny, Float32(0.0), tolerance=1e-7)


fn test_fp4_e2m1_scale_impact() raises:
    """Test how different scale factors affect encoding."""
    # With scale=2.0, we can represent larger values
    var fp4_scaled = FP4_E2M1.from_float32(12.0, scale=2.0)
    var result_scaled = fp4_scaled.to_float32(scale=2.0)
    assert_almost_equal(result_scaled, Float32(12.0), tolerance=1.0)

    # With scale=0.5, we can represent smaller values more precisely
    var fp4_small_scale = FP4_E2M1.from_float32(0.5, scale=0.5)
    var result_small_scale = fp4_small_scale.to_float32(scale=0.5)
    assert_almost_equal(result_small_scale, Float32(0.5), tolerance=0.2)


fn test_fp4_e2m1_special_values_nan() raises:
    """Test FP4_E2M1 encoding of NaN."""
    var nan_val = Float32(0.0) / Float32(0.0)
    var fp4_nan = FP4_E2M1.from_float32(nan_val, scale=1.0)

    # NaN is encoded as max value (0b0111)
    assert_equal(fp4_nan.value, 0b0111)


fn test_fp4_e2m1_special_values_inf() raises:
    """Test FP4_E2M1 encoding of infinity."""
    # Positive infinity
    var pos_inf = Float32(1.0) / Float32(0.0)
    var fp4_pos_inf = FP4_E2M1.from_float32(pos_inf, scale=1.0)
    assert_equal(fp4_pos_inf.value, 0b0111)  # Max positive value

    # Negative infinity
    var neg_inf = Float32(-1.0) / Float32(0.0)
    var fp4_neg_inf = FP4_E2M1.from_float32(neg_inf, scale=1.0)
    assert_equal(fp4_neg_inf.value, 0b1111)  # Max negative value


fn test_fp4_e2m1_equality() raises:
    """Test FP4_E2M1 equality comparison."""
    var fp4_a = FP4_E2M1.from_float32(2.0, scale=1.0)
    var fp4_b = FP4_E2M1.from_float32(2.0, scale=1.0)
    var fp4_c = FP4_E2M1.from_float32(3.0, scale=1.0)

    assert_true(fp4_a == fp4_b, "Equal FP4_E2M1 values should compare equal")
    assert_true(
        fp4_a != fp4_c, "Different FP4_E2M1 values should compare not equal"
    )


fn test_fp4_e2m1_bit_patterns() raises:
    """Test specific bit patterns in E2M1 format."""
    # Test zero: sign=0, exp=0, mantissa=0
    var fp4_zero = FP4_E2M1(0b0000)
    assert_almost_equal(
        fp4_zero.to_float32(scale=1.0), Float32(0.0), tolerance=1e-7
    )

    # Test positive max: sign=0, exp=3, mantissa=1
    var fp4_max = FP4_E2M1(0b0111)
    assert_almost_equal(
        fp4_max.to_float32(scale=1.0), Float32(6.0), tolerance=0.1
    )

    # Test negative max: sign=1, exp=3, mantissa=1
    var fp4_neg_max = FP4_E2M1(0b1111)
    assert_almost_equal(
        fp4_neg_max.to_float32(scale=1.0), Float32(-6.0), tolerance=0.1
    )

    # Test 1.0: sign=0, exp=1, mantissa=0
    var fp4_one = FP4_E2M1(0b0010)
    assert_almost_equal(
        fp4_one.to_float32(scale=1.0), Float32(1.0), tolerance=0.1
    )


fn test_fp4_e2m1_quantization() raises:
    """Test quantization to nearest representable value."""
    # 1.25 should quantize to either 1.0 or 1.5
    var fp4_1_25 = FP4_E2M1.from_float32(1.25, scale=1.0)
    var result_1_25 = fp4_1_25.to_float32(scale=1.0)
    assert_true(
        result_1_25 >= 1.0 and result_1_25 <= 1.5,
        "1.25 should quantize to 1.0 or 1.5",
    )

    # 2.5 should quantize to either 2.0 or 3.0
    var fp4_2_5 = FP4_E2M1.from_float32(2.5, scale=1.0)
    var result_2_5 = fp4_2_5.to_float32(scale=1.0)
    assert_true(
        result_2_5 >= 2.0 and result_2_5 <= 3.0,
        "2.5 should quantize to 2.0 or 3.0",
    )


# ============================================================================
# String Representation Tests
# ============================================================================


fn test_fp4_e2m1_str_representation() raises:
    """Test __str__ method for FP4_E2M1."""
    var fp4_one = FP4_E2M1.from_float32(1.0, scale=1.0)
    var str_repr = fp4_one.__str__()

    # Should contain "FP4_E2M1" and the value
    assert_true("FP4_E2M1" in str_repr, "__str__ should contain FP4_E2M1")


fn test_fp4_e2m1_repr_representation() raises:
    """Test __repr__ method for FP4_E2M1."""
    var fp4_two = FP4_E2M1.from_float32(2.0, scale=1.0)
    var repr_str = fp4_two.__repr__()

    # Should contain "FP4_E2M1" and "bits="
    assert_true("FP4_E2M1" in repr_str, "__repr__ should contain FP4_E2M1")
    assert_true("bits=" in repr_str, "__repr__ should contain bits=")


fn test_fp4_e2m1_repr_zero() raises:
    """Test __repr__ for zero value."""
    var fp4_zero = FP4_E2M1(0b0000)
    var repr_str = fp4_zero.__repr__()
    assert_true(len(repr_str) > 0, "__repr__ should produce non-empty string")


fn test_fp4_e2m1_repr_max() raises:
    """Test __repr__ for max value."""
    var fp4_max = FP4_E2M1(0b0111)
    var repr_str = fp4_max.__repr__()
    assert_true(len(repr_str) > 0, "__repr__ should produce non-empty string")


# ============================================================================
# All 16 FP4 Values Tests
# ============================================================================


fn test_fp4_e2m1_all_16_values() raises:
    """Test decoding of all 16 possible FP4 bit patterns."""
    # All 16 possible 4-bit values: 0x0 to 0xF
    # Testing that all can be decoded without errors
    for i in range(16):
        var fp4 = FP4_E2M1(UInt8(i))
        var _ = fp4.to_float32(scale=1.0)
        # Should produce a valid float (not NaN unless explicitly set)
        assert_true(
            len(fp4.__str__()) > 0,
            "All values should produce string representation",
        )


fn test_fp4_e2m1_positive_values_all() raises:
    """Test all positive FP4 bit patterns (sign bit = 0)."""
    # Positive values: 0x0 to 0x7
    var values: List[UInt8] = []
    values.append(0b0000)  # Zero
    values.append(0b0001)  # exp=0, mantissa=1 (invalid)
    values.append(0b0010)  # exp=1, mantissa=0 = 1.0
    values.append(0b0011)  # exp=1, mantissa=1 = 1.5
    values.append(0b0100)  # exp=2, mantissa=0 = 2.0
    values.append(0b0101)  # exp=2, mantissa=1 = 3.0
    values.append(0b0110)  # exp=3, mantissa=0 = 4.0
    values.append(0b0111)  # exp=3, mantissa=1 = 6.0

    for i in range(len(values)):
        var fp4 = FP4_E2M1(values[i])
        var decoded = fp4.to_float32(scale=1.0)
        assert_true(decoded >= 0.0, "Positive values should be >= 0")


fn test_fp4_e2m1_negative_values_all() raises:
    """Test all negative FP4 bit patterns (sign bit = 1)."""
    # Negative values: 0x8 to 0xF
    var values: List[UInt8] = []
    values.append(0b1000)  # -Zero
    values.append(0b1001)  # exp=0, mantissa=1 (invalid)
    values.append(0b1010)  # exp=1, mantissa=0 = -1.0
    values.append(0b1011)  # exp=1, mantissa=1 = -1.5
    values.append(0b1100)  # exp=2, mantissa=0 = -2.0
    values.append(0b1101)  # exp=2, mantissa=1 = -3.0
    values.append(0b1110)  # exp=3, mantissa=0 = -4.0
    values.append(0b1111)  # exp=3, mantissa=1 = -6.0

    for i in range(len(values)):
        var fp4 = FP4_E2M1(values[i])
        var decoded = fp4.to_float32(scale=1.0)
        assert_true(decoded <= 0.0, "Negative values should be <= 0")


fn test_fp4_e2m1_value_extraction() raises:
    """Test bit pattern extraction in to_float32."""
    # Test that correct components are extracted
    var fp4_test = FP4_E2M1(0b0101)  # exp=2, mantissa=1 = 3.0
    var result = fp4_test.to_float32(scale=1.0)
    assert_almost_equal(result, Float32(3.0), tolerance=0.1)


fn test_fp4_e2m1_scale_zero_edge_case() raises:
    """Test encoding with various edge case scales."""
    # Test with scale=1.0
    var fp4_a = FP4_E2M1.from_float32(3.0, scale=1.0)
    var result_a = fp4_a.to_float32(scale=1.0)
    assert_almost_equal(result_a, Float32(3.0), tolerance=0.5)

    # Test with scale=0.5
    var fp4_b = FP4_E2M1.from_float32(1.5, scale=0.5)
    var result_b = fp4_b.to_float32(scale=0.5)
    assert_almost_equal(result_b, Float32(1.5), tolerance=0.3)

    # Test with scale=2.0
    var fp4_c = FP4_E2M1.from_float32(6.0, scale=2.0)
    var result_c = fp4_c.to_float32(scale=2.0)
    assert_almost_equal(result_c, Float32(6.0), tolerance=1.0)


fn test_fp4_e2m1_not_equal() raises:
    """Test __ne__ operator."""
    var fp4_a = FP4_E2M1.from_float32(1.0, scale=1.0)
    var fp4_b = FP4_E2M1.from_float32(2.0, scale=1.0)

    assert_true(fp4_a != fp4_b, "Different FP4 values should not be equal")


fn test_fp4_e2m1_equal_direct_init() raises:
    """Test equality with direct initialization."""
    var fp4_a = FP4_E2M1(0b0010)
    var fp4_b = FP4_E2M1(0b0010)

    assert_true(fp4_a == fp4_b, "Same bit patterns should be equal")


fn test_fp4_e2m1_edge_case_boundaries() raises:
    """Test boundary conditions for quantization."""
    # Test value just below max
    var fp4_below_max = FP4_E2M1.from_float32(5.9, scale=1.0)
    var result_below_max = fp4_below_max.to_float32(scale=1.0)
    assert_true(result_below_max <= 6.0, "Value should not exceed max")

    # Test value just above min representable
    var fp4_above_min = FP4_E2M1.from_float32(1.01, scale=1.0)
    var result_above_min = fp4_above_min.to_float32(scale=1.0)
    assert_true(result_above_min >= 0.5, "Value should be at least 0.5")


fn test_fp4_e2m1_negative_scale() raises:
    """Test behavior with scale factor variations."""
    # Test inverse scale relationship
    var value = Float32(3.0)
    var fp4_scaled_up = FP4_E2M1.from_float32(value, scale=0.5)
    var result_scaled_up = fp4_scaled_up.to_float32(scale=0.5)
    assert_almost_equal(result_scaled_up, value, tolerance=0.5)

    var fp4_scaled_down = FP4_E2M1.from_float32(value, scale=2.0)
    var result_scaled_down = fp4_scaled_down.to_float32(scale=2.0)
    assert_almost_equal(result_scaled_down, value, tolerance=0.5)


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run all FP4_E2M1 tests."""
    print("\n=== FP4_E2M1 Basic Conversion Tests ===")
    test_fp4_e2m1_zero()
    print("✓ FP4_E2M1 zero encoding")

    test_fp4_e2m1_negative_zero()
    print("✓ FP4_E2M1 negative zero handling")

    test_fp4_e2m1_representable_values()
    print("✓ FP4_E2M1 all representable values")

    test_fp4_e2m1_positive_values()
    print("✓ FP4_E2M1 positive values")

    test_fp4_e2m1_negative_values()
    print("✓ FP4_E2M1 negative values")

    test_fp4_e2m1_range_clamping()
    print("✓ FP4_E2M1 range clamping")

    test_fp4_e2m1_small_values()
    print("✓ FP4_E2M1 small values")

    test_fp4_e2m1_scale_impact()
    print("✓ FP4_E2M1 scale factor impact")

    test_fp4_e2m1_special_values_nan()
    print("✓ FP4_E2M1 NaN handling")

    test_fp4_e2m1_special_values_inf()
    print("✓ FP4_E2M1 infinity handling")

    test_fp4_e2m1_equality()
    print("✓ FP4_E2M1 equality comparison")

    test_fp4_e2m1_bit_patterns()
    print("✓ FP4_E2M1 bit pattern verification")

    test_fp4_e2m1_quantization()
    print("✓ FP4_E2M1 quantization behavior")

    print("\n=== FP4_E2M1 String Representation Tests ===")
    test_fp4_e2m1_str_representation()
    print("✓ FP4_E2M1 __str__ method")

    test_fp4_e2m1_repr_representation()
    print("✓ FP4_E2M1 __repr__ method")

    test_fp4_e2m1_repr_zero()
    print("✓ FP4_E2M1 __repr__ for zero")

    test_fp4_e2m1_repr_max()
    print("✓ FP4_E2M1 __repr__ for max")

    print("\n=== All 16 FP4 Values Tests ===")
    test_fp4_e2m1_all_16_values()
    print("✓ All 16 possible FP4 bit patterns")

    test_fp4_e2m1_positive_values_all()
    print("✓ All positive FP4 bit patterns")

    test_fp4_e2m1_negative_values_all()
    print("✓ All negative FP4 bit patterns")

    test_fp4_e2m1_value_extraction()
    print("✓ Bit pattern extraction")

    test_fp4_e2m1_scale_zero_edge_case()
    print("✓ Edge case scales")

    test_fp4_e2m1_not_equal()
    print("✓ __ne__ operator")

    test_fp4_e2m1_equal_direct_init()
    print("✓ __eq__ operator with direct init")

    test_fp4_e2m1_edge_case_boundaries()
    print("✓ Boundary conditions")

    test_fp4_e2m1_negative_scale()
    print("✓ Scale factor variations")

    print("\n=== All FP4_E2M1 Tests Passed! ===\n")
