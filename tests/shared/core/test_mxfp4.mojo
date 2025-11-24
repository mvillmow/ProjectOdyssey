"""Tests for MXFP4 (Microscaling FP4) data type.

Tests cover:
- E8M0Scale creation and conversion
- MXFP4 creation from Float32
- MXFP4 to Float32 conversion
- Arithmetic operations (+, -, *, /, neg)
- Comparison operations (==, !=, <, <=, >, >=)
- Special values (zero, NaN, Inf)
- Wide dynamic range (2^-127 to 2^128)
- Edge cases and boundary values
"""

from ..helpers.assertions import (
    assert_true,
    assert_equal_int,
    assert_close_float,
)
from shared.core.types.mxfp4 import MXFP4, E8M0Scale
from math import isnan, isinf


# ============================================================================
# E8M0Scale Tests
# ============================================================================


fn test_e8m0_scale_basic() raises:
    """Test E8M0 scale creation and conversion."""
    # Test scale = 1.0 (exponent = 127)
    var scale_one = E8M0Scale.from_float32(1.0)
    assert_equal(scale_one.exponent, 127)
    assert_almost_equal(scale_one.to_float32(), Float32(1.0), tolerance=1e-6)

    # Test scale = 2.0 (exponent = 128)
    var scale_two = E8M0Scale.from_float32(2.0)
    assert_equal(scale_two.exponent, 128)
    assert_almost_equal(scale_two.to_float32(), Float32(2.0), tolerance=1e-6)

    # Test scale = 0.5 (exponent = 126)
    var scale_half = E8M0Scale.from_float32(0.5)
    assert_equal(scale_half.exponent, 126)
    assert_almost_equal(scale_half.to_float32(), Float32(0.5), tolerance=1e-6)


fn test_e8m0_scale_powers_of_two() raises:
    """Test E8M0 scale with various powers of 2."""
    # Test 2^-10 = 0.0009765625
    var scale_small = E8M0Scale.from_float32(0.0009765625)
    var result_small = scale_small.to_float32()
    assert_almost_equal(result_small, Float32(0.0009765625), tolerance=1e-7)

    # Test 2^10 = 1024.0
    var scale_large = E8M0Scale.from_float32(1024.0)
    var result_large = scale_large.to_float32()
    assert_almost_equal(result_large, Float32(1024.0), tolerance=1.0)


fn test_e8m0_scale_edge_cases() raises:
    """Test E8M0 scale edge cases."""
    # Test zero (should return minimum scale)
    var scale_zero = E8M0Scale.from_float32(0.0)
    assert_equal(scale_zero.exponent, 0)

    # Test negative (should return minimum scale)
    var scale_neg = E8M0Scale.from_float32(-1.0)
    assert_equal(scale_neg.exponent, 0)

    # Test infinity (should return maximum scale)
    var pos_inf = Float32(1.0) / Float32(0.0)
    var scale_inf = E8M0Scale.from_float32(pos_inf)
    assert_equal(scale_inf.exponent, 255)


# ============================================================================
# MXFP4 Basic Conversion Tests
# ============================================================================


fn test_mxfp4_zero() raises:
    """Test MXFP4 representation of zero."""
    var mxfp4_zero = MXFP4.from_float32(0.0)
    var result = mxfp4_zero.to_float32()

    assert_equal(mxfp4_zero.value.value, 0)
    assert_almost_equal(result, Float32(0.0), tolerance=1e-7)


fn test_mxfp4_positive_values() raises:
    """Test MXFP4 encoding of positive values."""
    # Test small positive value
    var mxfp4_small = MXFP4.from_float32(1.0)
    var result_small = mxfp4_small.to_float32()
    assert_almost_equal(result_small, Float32(1.0), tolerance=0.3)

    # Test medium positive value
    var mxfp4_medium = MXFP4.from_float32(10.0)
    var result_medium = mxfp4_medium.to_float32()
    assert_almost_equal(result_medium, Float32(10.0), tolerance=2.0)

    # Test large positive value
    var mxfp4_large = MXFP4.from_float32(100.0)
    var result_large = mxfp4_large.to_float32()
    assert_almost_equal(result_large, Float32(100.0), tolerance=20.0)


fn test_mxfp4_negative_values() raises:
    """Test MXFP4 encoding of negative values."""
    # Test small negative value
    var mxfp4_small = MXFP4.from_float32(-1.0)
    var result_small = mxfp4_small.to_float32()
    assert_almost_equal(result_small, Float32(-1.0), tolerance=0.3)

    # Test medium negative value
    var mxfp4_medium = MXFP4.from_float32(-10.0)
    var result_medium = mxfp4_medium.to_float32()
    assert_almost_equal(result_medium, Float32(-10.0), tolerance=2.0)

    # Test large negative value
    var mxfp4_large = MXFP4.from_float32(-100.0)
    var result_large = mxfp4_large.to_float32()
    assert_almost_equal(result_large, Float32(-100.0), tolerance=20.0)


fn test_mxfp4_wide_dynamic_range() raises:
    """Test MXFP4 wide dynamic range (E8M0 scale)."""
    # Test very small value (near 2^-127)
    var mxfp4_tiny = MXFP4.from_float32(1e-30)
    var result_tiny = mxfp4_tiny.to_float32()
    assert_true(result_tiny > 0.0 and result_tiny < 1e-20, "MXFP4 should handle tiny values")

    # Test very large value (within Float32 range)
    var mxfp4_huge = MXFP4.from_float32(1e30)
    var result_huge = mxfp4_huge.to_float32()
    assert_true(result_huge > 1e25, "MXFP4 should handle huge values")


fn test_mxfp4_special_values_nan() raises:
    """Test MXFP4 encoding of NaN."""
    var nan_val = Float32(0.0) / Float32(0.0)
    var mxfp4_nan = MXFP4.from_float32(nan_val)
    var result = mxfp4_nan.to_float32()

    # NaN encoding: E2M1 value should be max (0b0111)
    assert_equal(mxfp4_nan.value.value, 0b0111)


fn test_mxfp4_special_values_inf() raises:
    """Test MXFP4 encoding of infinity."""
    # Positive infinity
    var pos_inf = Float32(1.0) / Float32(0.0)
    var mxfp4_pos_inf = MXFP4.from_float32(pos_inf)
    var result_pos = mxfp4_pos_inf.to_float32()

    # Should encode as max E2M1 value with max scale
    assert_equal(mxfp4_pos_inf.value.value, 0b0111)

    # Negative infinity
    var neg_inf = Float32(-1.0) / Float32(0.0)
    var mxfp4_neg_inf = MXFP4.from_float32(neg_inf)
    var result_neg = mxfp4_neg_inf.to_float32()

    # Should encode as max negative E2M1 value
    assert_equal(mxfp4_neg_inf.value.value, 0b1111)


# ============================================================================
# MXFP4 Arithmetic Operations
# ============================================================================


fn test_mxfp4_addition() raises:
    """Test MXFP4 addition."""
    var a = MXFP4.from_float32(3.0)
    var b = MXFP4.from_float32(2.0)
    var result = a + b

    assert_almost_equal(result.to_float32(), Float32(5.0), tolerance=1.5)


fn test_mxfp4_subtraction() raises:
    """Test MXFP4 subtraction."""
    var a = MXFP4.from_float32(5.0)
    var b = MXFP4.from_float32(2.0)
    var result = a - b

    assert_almost_equal(result.to_float32(), Float32(3.0), tolerance=1.0)


fn test_mxfp4_multiplication() raises:
    """Test MXFP4 multiplication."""
    var a = MXFP4.from_float32(3.0)
    var b = MXFP4.from_float32(2.0)
    var result = a * b

    assert_almost_equal(result.to_float32(), Float32(6.0), tolerance=2.0)


fn test_mxfp4_division() raises:
    """Test MXFP4 division."""
    var a = MXFP4.from_float32(6.0)
    var b = MXFP4.from_float32(2.0)
    var result = a / b

    assert_almost_equal(result.to_float32(), Float32(3.0), tolerance=1.0)


fn test_mxfp4_negation() raises:
    """Test MXFP4 negation."""
    var a = MXFP4.from_float32(3.0)
    var result = -a

    assert_almost_equal(result.to_float32(), Float32(-3.0), tolerance=1.0)

    # Test double negation
    var result2 = -result
    assert_almost_equal(result2.to_float32(), Float32(3.0), tolerance=1.0)


# ============================================================================
# MXFP4 Comparison Operations
# ============================================================================


fn test_mxfp4_equality() raises:
    """Test MXFP4 equality comparison."""
    var a = MXFP4.from_float32(3.14)
    var b = MXFP4.from_float32(3.14)
    var c = MXFP4.from_float32(2.71)

    assert_true(a == b, "Equal MXFP4 values should compare equal")
    assert_true(a != c, "Different MXFP4 values should compare not equal")


fn test_mxfp4_inequality() raises:
    """Test MXFP4 inequality operators."""
    var small = MXFP4.from_float32(1.0)
    var large = MXFP4.from_float32(5.0)

    assert_true(small < large, "Small < Large should be true")
    assert_true(small <= large, "Small <= Large should be true")
    assert_true(large > small, "Large > Small should be true")
    assert_true(large >= small, "Large >= Small should be true")


fn test_mxfp4_comparison_edge_cases() raises:
    """Test MXFP4 comparison with edge cases."""
    var zero = MXFP4.from_float32(0.0)
    var positive = MXFP4.from_float32(1.0)
    var negative = MXFP4.from_float32(-1.0)

    assert_true(negative < zero, "Negative < Zero should be true")
    assert_true(zero < positive, "Zero < Positive should be true")
    assert_true(negative < positive, "Negative < Positive should be true")


# ============================================================================
# MXFP4 Round-trip Tests
# ============================================================================


fn test_mxfp4_roundtrip_small_values() raises:
    """Test MXFP4 round-trip for small values."""
    var original = Float32(0.5)
    var mxfp4 = MXFP4.from_float32(original)
    var restored = mxfp4.to_float32()

    assert_almost_equal(restored, original, tolerance=0.2)


fn test_mxfp4_roundtrip_medium_values() raises:
    """Test MXFP4 round-trip for medium values."""
    var original = Float32(42.0)
    var mxfp4 = MXFP4.from_float32(original)
    var restored = mxfp4.to_float32()

    assert_almost_equal(restored, original, tolerance=10.0)


fn test_mxfp4_roundtrip_large_values() raises:
    """Test MXFP4 round-trip for large values."""
    var original = Float32(1000.0)
    var mxfp4 = MXFP4.from_float32(original)
    var restored = mxfp4.to_float32()

    assert_almost_equal(restored, original, tolerance=250.0)


fn test_mxfp4_precision_vs_fp8() raises:
    """Test that MXFP4 has comparable or better range than FP8."""
    # MXFP4 should handle very large values better than FP8 (max ~240)
    var large_val = Float32(10000.0)
    var mxfp4 = MXFP4.from_float32(large_val)
    var restored = mxfp4.to_float32()

    # Should preserve order of magnitude
    assert_true(restored > 5000.0, "MXFP4 should preserve large values better than FP8")


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run all MXFP4 tests."""
    print("\n=== E8M0Scale Tests ===")
    test_e8m0_scale_basic()
    print("✓ E8M0 scale basic operations")

    test_e8m0_scale_powers_of_two()
    print("✓ E8M0 scale powers of two")

    test_e8m0_scale_edge_cases()
    print("✓ E8M0 scale edge cases")

    print("\n=== MXFP4 Basic Conversion Tests ===")
    test_mxfp4_zero()
    print("✓ MXFP4 zero encoding")

    test_mxfp4_positive_values()
    print("✓ MXFP4 positive values")

    test_mxfp4_negative_values()
    print("✓ MXFP4 negative values")

    test_mxfp4_wide_dynamic_range()
    print("✓ MXFP4 wide dynamic range")

    test_mxfp4_special_values_nan()
    print("✓ MXFP4 NaN handling")

    test_mxfp4_special_values_inf()
    print("✓ MXFP4 infinity handling")

    print("\n=== MXFP4 Arithmetic Operations ===")
    test_mxfp4_addition()
    print("✓ MXFP4 addition")

    test_mxfp4_subtraction()
    print("✓ MXFP4 subtraction")

    test_mxfp4_multiplication()
    print("✓ MXFP4 multiplication")

    test_mxfp4_division()
    print("✓ MXFP4 division")

    test_mxfp4_negation()
    print("✓ MXFP4 negation")

    print("\n=== MXFP4 Comparison Operations ===")
    test_mxfp4_equality()
    print("✓ MXFP4 equality")

    test_mxfp4_inequality()
    print("✓ MXFP4 inequality operators")

    test_mxfp4_comparison_edge_cases()
    print("✓ MXFP4 comparison edge cases")

    print("\n=== MXFP4 Round-trip Tests ===")
    test_mxfp4_roundtrip_small_values()
    print("✓ MXFP4 round-trip (small values)")

    test_mxfp4_roundtrip_medium_values()
    print("✓ MXFP4 round-trip (medium values)")

    test_mxfp4_roundtrip_large_values()
    print("✓ MXFP4 round-trip (large values)")

    test_mxfp4_precision_vs_fp8()
    print("✓ MXFP4 vs FP8 range comparison")

    print("\n=== All MXFP4 Tests Passed! ===\n")
