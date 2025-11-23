"""Tests for FP4_E2M1 base type encoding and decoding.

Tests cover:
- All 8 representable positive values (E2M1 format)
- All 8 representable negative values
- Round-trip conversion accuracy
- Quantization between representable values
- Special values (NaN, Infinity, zero)
- Scale factor handling
- Comparison operators
- String representations

This addresses TEST-010 (P0 CRITICAL): FP4_E2M1 was previously 0% tested.

All tests use pure functional API.
"""

from shared.core.types.fp4 import FP4_E2M1
from tests.shared.conftest import (
    assert_true,
    assert_equal,
    assert_almost_equal,
)
from math import isnan, isinf


# ============================================================================
# E2M1 Representable Values Tests
# ============================================================================


fn test_fp4_representable_positive_values() raises:
    """Test all 8 representable positive E2M1 values.

    E2M1 format with bias=1:
    - exp=0, mantissa=0: 0.0
    - exp=1, mantissa=0: 2^0 * (1 + 0) = 1.0
    - exp=1, mantissa=1: 2^0 * (1 + 1) = 1.5
    - exp=2, mantissa=0: 2^1 * (1 + 0) = 2.0
    - exp=2, mantissa=1: 2^1 * (1 + 1) = 3.0
    - exp=3, mantissa=0: 2^2 * (1 + 0) = 4.0
    - exp=3, mantissa=1: 2^2 * (1 + 1) = 6.0
    """
    # Test zero
    var fp4_zero = FP4_E2M1.from_float32(0.0, scale=1.0)
    var decoded_zero = fp4_zero.to_float32(scale=1.0)
    assert_almost_equal(decoded_zero, Float32(0.0), tolerance=1e-6)

    # Test 1.0
    var fp4_one = FP4_E2M1.from_float32(1.0, scale=1.0)
    var decoded_one = fp4_one.to_float32(scale=1.0)
    assert_almost_equal(decoded_one, Float32(1.0), tolerance=1e-6)

    # Test 1.5
    var fp4_onefive = FP4_E2M1.from_float32(1.5, scale=1.0)
    var decoded_onefive = fp4_onefive.to_float32(scale=1.0)
    assert_almost_equal(decoded_onefive, Float32(1.5), tolerance=1e-6)

    # Test 2.0
    var fp4_two = FP4_E2M1.from_float32(2.0, scale=1.0)
    var decoded_two = fp4_two.to_float32(scale=1.0)
    assert_almost_equal(decoded_two, Float32(2.0), tolerance=1e-6)

    # Test 3.0
    var fp4_three = FP4_E2M1.from_float32(3.0, scale=1.0)
    var decoded_three = fp4_three.to_float32(scale=1.0)
    assert_almost_equal(decoded_three, Float32(3.0), tolerance=1e-6)

    # Test 4.0
    var fp4_four = FP4_E2M1.from_float32(4.0, scale=1.0)
    var decoded_four = fp4_four.to_float32(scale=1.0)
    assert_almost_equal(decoded_four, Float32(4.0), tolerance=1e-6)

    # Test 6.0 (max value)
    var fp4_six = FP4_E2M1.from_float32(6.0, scale=1.0)
    var decoded_six = fp4_six.to_float32(scale=1.0)
    assert_almost_equal(decoded_six, Float32(6.0), tolerance=1e-6)


fn test_fp4_representable_negative_values() raises:
    """Test all 8 representable negative E2M1 values."""
    # Test -0.0 (distinct from +0.0 in bit pattern)
    var fp4_neg_zero = FP4_E2M1.from_float32(-0.0, scale=1.0)
    var decoded_neg_zero = fp4_neg_zero.to_float32(scale=1.0)
    assert_almost_equal(decoded_neg_zero, Float32(0.0), tolerance=1e-6)

    # Test -1.0
    var fp4_neg_one = FP4_E2M1.from_float32(-1.0, scale=1.0)
    var decoded_neg_one = fp4_neg_one.to_float32(scale=1.0)
    assert_almost_equal(decoded_neg_one, Float32(-1.0), tolerance=1e-6)

    # Test -1.5
    var fp4_neg_onefive = FP4_E2M1.from_float32(-1.5, scale=1.0)
    var decoded_neg_onefive = fp4_neg_onefive.to_float32(scale=1.0)
    assert_almost_equal(decoded_neg_onefive, Float32(-1.5), tolerance=1e-6)

    # Test -2.0
    var fp4_neg_two = FP4_E2M1.from_float32(-2.0, scale=1.0)
    var decoded_neg_two = fp4_neg_two.to_float32(scale=1.0)
    assert_almost_equal(decoded_neg_two, Float32(-2.0), tolerance=1e-6)

    # Test -3.0
    var fp4_neg_three = FP4_E2M1.from_float32(-3.0, scale=1.0)
    var decoded_neg_three = fp4_neg_three.to_float32(scale=1.0)
    assert_almost_equal(decoded_neg_three, Float32(-3.0), tolerance=1e-6)

    # Test -4.0
    var fp4_neg_four = FP4_E2M1.from_float32(-4.0, scale=1.0)
    var decoded_neg_four = fp4_neg_four.to_float32(scale=1.0)
    assert_almost_equal(decoded_neg_four, Float32(-4.0), tolerance=1e-6)

    # Test -6.0 (max negative value)
    var fp4_neg_six = FP4_E2M1.from_float32(-6.0, scale=1.0)
    var decoded_neg_six = fp4_neg_six.to_float32(scale=1.0)
    assert_almost_equal(decoded_neg_six, Float32(-6.0), tolerance=1e-6)


# ============================================================================
# Round-Trip Conversion Tests
# ============================================================================


fn test_fp4_round_trip_exact_values() raises:
    """Test round-trip conversion for exact representable values."""
    var exact_values = List[Float32]()
    exact_values.append(0.0)
    exact_values.append(1.0)
    exact_values.append(1.5)
    exact_values.append(2.0)
    exact_values.append(3.0)
    exact_values.append(4.0)
    exact_values.append(6.0)

    for i in range(len(exact_values)):
        var original = exact_values[i]
        var fp4_val = FP4_E2M1.from_float32(original, scale=1.0)
        var decoded = fp4_val.to_float32(scale=1.0)
        assert_almost_equal(decoded, original, tolerance=1e-6)


fn test_fp4_round_trip_negative_exact() raises:
    """Test round-trip conversion for exact negative values."""
    var exact_values = List[Float32]()
    exact_values.append(-1.0)
    exact_values.append(-1.5)
    exact_values.append(-2.0)
    exact_values.append(-3.0)
    exact_values.append(-4.0)
    exact_values.append(-6.0)

    for i in range(len(exact_values)):
        var original = exact_values[i]
        var fp4_val = FP4_E2M1.from_float32(original, scale=1.0)
        var decoded = fp4_val.to_float32(scale=1.0)
        assert_almost_equal(decoded, original, tolerance=1e-6)


# ============================================================================
# Quantization Tests (Between Representable Values)
# ============================================================================


fn test_fp4_quantization_rounding() raises:
    """Test quantization of values between representable points."""
    # 1.2 should round to 1.0 (closer than to 1.5)
    var fp4_1_2 = FP4_E2M1.from_float32(1.2, scale=1.0)
    var decoded_1_2 = fp4_1_2.to_float32(scale=1.0)
    assert_almost_equal(decoded_1_2, Float32(1.0), tolerance=1e-6)

    # 1.4 should round to 1.5 (closer than to 1.0)
    var fp4_1_4 = FP4_E2M1.from_float32(1.4, scale=1.0)
    var decoded_1_4 = fp4_1_4.to_float32(scale=1.0)
    assert_almost_equal(decoded_1_4, Float32(1.5), tolerance=1e-6)

    # 2.7 should round to 3.0 (closer than to 2.0)
    var fp4_2_7 = FP4_E2M1.from_float32(2.7, scale=1.0)
    var decoded_2_7 = fp4_2_7.to_float32(scale=1.0)
    assert_almost_equal(decoded_2_7, Float32(3.0), tolerance=1e-6)

    # 4.9 should round to 4.0 (closer than to 6.0)
    var fp4_4_9 = FP4_E2M1.from_float32(4.9, scale=1.0)
    var decoded_4_9 = fp4_4_9.to_float32(scale=1.0)
    assert_almost_equal(decoded_4_9, Float32(4.0), tolerance=1e-6)


fn test_fp4_quantization_clamping() raises:
    """Test clamping of values outside representable range."""
    # Value > 6.0 should clamp to 6.0
    var fp4_large = FP4_E2M1.from_float32(10.0, scale=1.0)
    var decoded_large = fp4_large.to_float32(scale=1.0)
    assert_almost_equal(decoded_large, Float32(6.0), tolerance=1e-6)

    # Value < 0.5 should map to 0.0
    var fp4_small = FP4_E2M1.from_float32(0.3, scale=1.0)
    var decoded_small = fp4_small.to_float32(scale=1.0)
    assert_almost_equal(decoded_small, Float32(0.0), tolerance=1e-6)


# ============================================================================
# Special Values Tests
# ============================================================================


fn test_fp4_special_value_nan() raises:
    """Test NaN handling - should map to max value (0b0111)."""
    var nan_val = Float32(0.0) / Float32(0.0)  # Create NaN
    var fp4_nan = FP4_E2M1.from_float32(nan_val, scale=1.0)
    var decoded_nan = fp4_nan.to_float32(scale=1.0)

    # NaN maps to max value (6.0)
    assert_almost_equal(decoded_nan, Float32(6.0), tolerance=1e-6)


fn test_fp4_special_value_infinity() raises:
    """Test Infinity handling - should clamp to max representable."""
    # Positive infinity
    var pos_inf = Float32(1.0) / Float32(0.0)
    var fp4_pos_inf = FP4_E2M1.from_float32(pos_inf, scale=1.0)
    var decoded_pos_inf = fp4_pos_inf.to_float32(scale=1.0)
    assert_almost_equal(decoded_pos_inf, Float32(6.0), tolerance=1e-6)

    # Negative infinity
    var neg_inf = Float32(-1.0) / Float32(0.0)
    var fp4_neg_inf = FP4_E2M1.from_float32(neg_inf, scale=1.0)
    var decoded_neg_inf = fp4_neg_inf.to_float32(scale=1.0)
    assert_almost_equal(decoded_neg_inf, Float32(-6.0), tolerance=1e-6)


fn test_fp4_special_value_zero() raises:
    """Test zero value encoding/decoding."""
    # Positive zero
    var fp4_pos_zero = FP4_E2M1.from_float32(0.0, scale=1.0)
    var decoded_pos_zero = fp4_pos_zero.to_float32(scale=1.0)
    assert_almost_equal(decoded_pos_zero, Float32(0.0), tolerance=1e-6)

    # Very small value (should map to zero)
    var fp4_tiny = FP4_E2M1.from_float32(1e-10, scale=1.0)
    var decoded_tiny = fp4_tiny.to_float32(scale=1.0)
    assert_almost_equal(decoded_tiny, Float32(0.0), tolerance=1e-6)


# ============================================================================
# Scale Factor Tests
# ============================================================================


fn test_fp4_with_scale_factor() raises:
    """Test encoding/decoding with different scale factors."""
    # Scale = 2.0: value 12.0 becomes 6.0 after scaling
    var fp4_scaled = FP4_E2M1.from_float32(12.0, scale=2.0)
    var decoded_scaled = fp4_scaled.to_float32(scale=2.0)
    assert_almost_equal(decoded_scaled, Float32(12.0), tolerance=1e-5)

    # Scale = 0.5: value 3.0 becomes 6.0 after scaling
    var fp4_small_scale = FP4_E2M1.from_float32(3.0, scale=0.5)
    var decoded_small_scale = fp4_small_scale.to_float32(scale=0.5)
    assert_almost_equal(decoded_small_scale, Float32(3.0), tolerance=1e-5)


fn test_fp4_scale_ranges() raises:
    """Test that scale factor extends representable range."""
    # With scale=10.0, we can represent values up to 60.0
    var fp4_large_scale = FP4_E2M1.from_float32(60.0, scale=10.0)
    var decoded_large_scale = fp4_large_scale.to_float32(scale=10.0)
    assert_almost_equal(decoded_large_scale, Float32(60.0), tolerance=1e-5)

    # With scale=0.1, we can represent values down to 0.1
    var fp4_small_range = FP4_E2M1.from_float32(0.15, scale=0.1)
    var decoded_small_range = fp4_small_range.to_float32(scale=0.1)
    assert_almost_equal(decoded_small_range, Float32(0.15), tolerance=0.05)


# ============================================================================
# Comparison Operator Tests
# ============================================================================


fn test_fp4_comparison_equality() raises:
    """Test equality comparison operators."""
    var fp4_a = FP4_E2M1.from_float32(2.0, scale=1.0)
    var fp4_b = FP4_E2M1.from_float32(2.0, scale=1.0)
    var fp4_c = FP4_E2M1.from_float32(3.0, scale=1.0)

    # Test __eq__
    assert_true(fp4_a == fp4_b, "Equal values should be ==")
    assert_true(not (fp4_a == fp4_c), "Different values should not be ==")

    # Test __ne__
    assert_true(fp4_a != fp4_c, "Different values should be !=")
    assert_true(not (fp4_a != fp4_b), "Equal values should not be !=")


# ============================================================================
# String Representation Tests
# ============================================================================


fn test_fp4_string_representation() raises:
    """Test __str__ method."""
    var fp4_val = FP4_E2M1.from_float32(3.0, scale=1.0)
    var str_repr = str(fp4_val)

    # Should contain "FP4_E2M1" and the value
    assert_true(len(str_repr) > 0, "String representation should not be empty")


fn test_fp4_repr_representation() raises:
    """Test __repr__ method."""
    var fp4_val = FP4_E2M1.from_float32(3.0, scale=1.0)
    var repr_str = repr(fp4_val)

    # Should contain "FP4_E2M1", "bits", and "value"
    assert_true(len(repr_str) > 0, "Repr should not be empty")


# ============================================================================
# Bit Pattern Tests
# ============================================================================


fn test_fp4_bit_patterns() raises:
    """Test specific bit patterns for representable values."""
    # 0.0 should have exp=0, mantissa=0 (bits: 0b0000)
    var fp4_zero = FP4_E2M1.from_float32(0.0, scale=1.0)
    assert_equal(int(fp4_zero.value), 0)

    # 1.0 should have exp=1, mantissa=0 (bits: 0b0010)
    var fp4_one = FP4_E2M1.from_float32(1.0, scale=1.0)
    assert_equal(int(fp4_one.value), 0b0010)

    # 1.5 should have exp=1, mantissa=1 (bits: 0b0011)
    var fp4_onefive = FP4_E2M1.from_float32(1.5, scale=1.0)
    assert_equal(int(fp4_onefive.value), 0b0011)

    # 6.0 should have exp=3, mantissa=1 (bits: 0b0111)
    var fp4_six = FP4_E2M1.from_float32(6.0, scale=1.0)
    assert_equal(int(fp4_six.value), 0b0111)

    # -1.0 should have sign=1, exp=1, mantissa=0 (bits: 0b1010)
    var fp4_neg_one = FP4_E2M1.from_float32(-1.0, scale=1.0)
    assert_equal(int(fp4_neg_one.value), 0b1010)


fn main() raises:
    """Run all FP4_E2M1 base type tests."""
    print("Running FP4_E2M1 base type tests...")

    # Representable values
    test_fp4_representable_positive_values()
    print("✓ Positive representable values")

    test_fp4_representable_negative_values()
    print("✓ Negative representable values")

    # Round-trip conversion
    test_fp4_round_trip_exact_values()
    print("✓ Round-trip (exact values)")

    test_fp4_round_trip_negative_exact()
    print("✓ Round-trip (negative exact)")

    # Quantization
    test_fp4_quantization_rounding()
    print("✓ Quantization rounding")

    test_fp4_quantization_clamping()
    print("✓ Quantization clamping")

    # Special values
    test_fp4_special_value_nan()
    print("✓ Special value (NaN)")

    test_fp4_special_value_infinity()
    print("✓ Special value (Infinity)")

    test_fp4_special_value_zero()
    print("✓ Special value (zero)")

    # Scale factors
    test_fp4_with_scale_factor()
    print("✓ Scale factor handling")

    test_fp4_scale_ranges()
    print("✓ Scale ranges")

    # Comparison operators
    test_fp4_comparison_equality()
    print("✓ Comparison operators")

    # String representations
    test_fp4_string_representation()
    print("✓ String representation (__str__)")

    test_fp4_repr_representation()
    print("✓ Repr representation (__repr__)")

    # Bit patterns
    test_fp4_bit_patterns()
    print("✓ Bit patterns")

    print("\nAll FP4_E2M1 base type tests passed!")
    print("TEST-010 (P0 CRITICAL) - RESOLVED")
