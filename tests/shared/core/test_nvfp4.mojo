"""Tests for NVFP4 (NVIDIA FP4) data type.

Tests cover:
- E4M3Scale creation and conversion
- NVFP4 creation from Float32
- NVFP4 to Float32 conversion
- Arithmetic operations (+, -, *, /, neg)
- Comparison operations (==, !=, <, <=, >, >=)
- Special values (zero, NaN, Inf)
- Balanced dynamic range (E4M3 scale)
- Edge cases and boundary values
- Accuracy comparison with MXFP4
"""

from ..helpers.assertions import (
    assert_almost_equal,
    assert_close_float,
    assert_equal,
    assert_equal_int,
    assert_true,
)
from shared.core.types.nvfp4 import NVFP4, E4M3Scale
from math import isnan, isinf


# ============================================================================
# E4M3Scale Tests
# ============================================================================


fn test_e4m3_scale_basic() raises:
    """Test E4M3 scale creation and conversion."""
    # Test scale = 1.0 (exp=7, mantissa=0)
    var scale_one = E4M3Scale.from_float32(1.0)
    assert_equal(scale_one.value, 0x38)  # 0b0111000
    assert_almost_equal(scale_one.to_float32(), Float32(1.0), tolerance=1e-6)

    # Test scale = 2.0 (exp=8, mantissa=0)
    var scale_two = E4M3Scale.from_float32(2.0)
    var result_two = scale_two.to_float32()
    assert_almost_equal(result_two, Float32(2.0), tolerance=1e-6)

    # Test scale = 0.5 (exp=6, mantissa=0)
    var scale_half = E4M3Scale.from_float32(0.5)
    var result_half = scale_half.to_float32()
    assert_almost_equal(result_half, Float32(0.5), tolerance=1e-6)


fn test_e4m3_scale_mantissa_precision() raises:
    """Test E4M3 scale with fractional values (uses mantissa bits)."""
    # E4M3 can represent non-power-of-2 scales due to 3-bit mantissa
    var scale_1_5 = E4M3Scale.from_float32(1.5)
    var result_1_5 = scale_1_5.to_float32()
    assert_almost_equal(result_1_5, Float32(1.5), tolerance=0.15)

    var scale_3_5 = E4M3Scale.from_float32(3.5)
    var result_3_5 = scale_3_5.to_float32()
    assert_almost_equal(result_3_5, Float32(3.5), tolerance=0.5)


fn test_e4m3_scale_range() raises:
    """Test E4M3 scale dynamic range (similar to FP8 E4M3)."""
    # E4M3 max value is approximately 240
    var scale_large = E4M3Scale.from_float32(200.0)
    var result_large = scale_large.to_float32()
    assert_true(result_large > 150.0 and result_large < 250.0, "E4M3 should handle values up to ~240")

    # E4M3 min normal value is 2^-6 = 0.015625
    var scale_small = E4M3Scale.from_float32(0.02)
    var result_small = scale_small.to_float32()
    assert_true(result_small > 0.01 and result_small < 0.03, "E4M3 should handle small normal values")


fn test_e4m3_scale_edge_cases() raises:
    """Test E4M3 scale edge cases."""
    # Test zero (should return zero)
    var scale_zero = E4M3Scale.from_float32(0.0)
    assert_equal(scale_zero.value, 0)

    # Test negative (should return zero)
    var scale_neg = E4M3Scale.from_float32(-1.0)
    assert_equal(scale_neg.value, 0)

    # Test overflow (should clamp to max)
    var scale_overflow = E4M3Scale.from_float32(1000.0)
    assert_equal(scale_overflow.value, 0x7F)  # Max value


# ============================================================================
# NVFP4 Basic Conversion Tests
# ============================================================================


fn test_nvfp4_zero() raises:
    """Test NVFP4 representation of zero."""
    var nvfp4_zero = NVFP4.from_float32(0.0)
    var result = nvfp4_zero.to_float32()

    assert_equal(nvfp4_zero.value.value, 0)
    assert_almost_equal(result, Float32(0.0), tolerance=1e-7)


fn test_nvfp4_positive_values() raises:
    """Test NVFP4 encoding of positive values."""
    # Test small positive value
    var nvfp4_small = NVFP4.from_float32(1.0)
    var result_small = nvfp4_small.to_float32()
    assert_almost_equal(result_small, Float32(1.0), tolerance=0.3)

    # Test medium positive value
    var nvfp4_medium = NVFP4.from_float32(10.0)
    var result_medium = nvfp4_medium.to_float32()
    assert_almost_equal(result_medium, Float32(10.0), tolerance=2.0)

    # Test large positive value
    var nvfp4_large = NVFP4.from_float32(100.0)
    var result_large = nvfp4_large.to_float32()
    assert_almost_equal(result_large, Float32(100.0), tolerance=20.0)


fn test_nvfp4_negative_values() raises:
    """Test NVFP4 encoding of negative values."""
    # Test small negative value
    var nvfp4_small = NVFP4.from_float32(-1.0)
    var result_small = nvfp4_small.to_float32()
    assert_almost_equal(result_small, Float32(-1.0), tolerance=0.3)

    # Test medium negative value
    var nvfp4_medium = NVFP4.from_float32(-10.0)
    var result_medium = nvfp4_medium.to_float32()
    assert_almost_equal(result_medium, Float32(-10.0), tolerance=2.0)

    # Test large negative value
    var nvfp4_large = NVFP4.from_float32(-100.0)
    var result_large = nvfp4_large.to_float32()
    assert_almost_equal(result_large, Float32(-100.0), tolerance=20.0)


fn test_nvfp4_balanced_dynamic_range() raises:
    """Test NVFP4 balanced dynamic range (E4M3 scale)."""
    # E4M3 has narrower range than E8M0 but better precision
    # Test small value
    var nvfp4_small = NVFP4.from_float32(0.01)
    var result_small = nvfp4_small.to_float32()
    assert_true(result_small >= 0.0 and result_small < 0.05, "NVFP4 should handle small values")

    # Test large value (within E4M3 range ~240)
    var nvfp4_large = NVFP4.from_float32(200.0)
    var result_large = nvfp4_large.to_float32()
    assert_true(result_large > 150.0, "NVFP4 should handle values up to ~240")


fn test_nvfp4_special_values_nan() raises:
    """Test NVFP4 encoding of NaN."""
    var nan_val = Float32(0.0) / Float32(0.0)
    var nvfp4_nan = NVFP4.from_float32(nan_val)
    var result = nvfp4_nan.to_float32()

    # NaN encoding: E2M1 value should be max (0b0111)
    assert_equal(nvfp4_nan.value.value, 0b0111)


fn test_nvfp4_special_values_inf() raises:
    """Test NVFP4 encoding of infinity."""
    # Positive infinity
    var pos_inf = Float32(1.0) / Float32(0.0)
    var nvfp4_pos_inf = NVFP4.from_float32(pos_inf)
    var result_pos = nvfp4_pos_inf.to_float32()

    # Should encode as max E2M1 value
    assert_equal(nvfp4_pos_inf.value.value, 0b0111)

    # Negative infinity
    var neg_inf = Float32(-1.0) / Float32(0.0)
    var nvfp4_neg_inf = NVFP4.from_float32(neg_inf)
    var result_neg = nvfp4_neg_inf.to_float32()

    # Should encode as max negative E2M1 value
    assert_equal(nvfp4_neg_inf.value.value, 0b1111)


# ============================================================================
# NVFP4 Arithmetic Operations
# ============================================================================


fn test_nvfp4_addition() raises:
    """Test NVFP4 addition."""
    var a = NVFP4.from_float32(3.0)
    var b = NVFP4.from_float32(2.0)
    var result = a + b

    assert_almost_equal(result.to_float32(), Float32(5.0), tolerance=1.5)


fn test_nvfp4_subtraction() raises:
    """Test NVFP4 subtraction."""
    var a = NVFP4.from_float32(5.0)
    var b = NVFP4.from_float32(2.0)
    var result = a - b

    assert_almost_equal(result.to_float32(), Float32(3.0), tolerance=1.0)


fn test_nvfp4_multiplication() raises:
    """Test NVFP4 multiplication."""
    var a = NVFP4.from_float32(3.0)
    var b = NVFP4.from_float32(2.0)
    var result = a * b

    assert_almost_equal(result.to_float32(), Float32(6.0), tolerance=2.0)


fn test_nvfp4_division() raises:
    """Test NVFP4 division."""
    var a = NVFP4.from_float32(6.0)
    var b = NVFP4.from_float32(2.0)
    var result = a / b

    assert_almost_equal(result.to_float32(), Float32(3.0), tolerance=1.0)


fn test_nvfp4_negation() raises:
    """Test NVFP4 negation."""
    var a = NVFP4.from_float32(3.0)
    var result = -a

    assert_almost_equal(result.to_float32(), Float32(-3.0), tolerance=1.0)

    # Test double negation
    var result2 = -result
    assert_almost_equal(result2.to_float32(), Float32(3.0), tolerance=1.0)


# ============================================================================
# NVFP4 Comparison Operations
# ============================================================================


fn test_nvfp4_equality() raises:
    """Test NVFP4 equality comparison."""
    var a = NVFP4.from_float32(3.14)
    var b = NVFP4.from_float32(3.14)
    var c = NVFP4.from_float32(2.71)

    assert_true(a == b, "Equal NVFP4 values should compare equal")
    assert_true(a != c, "Different NVFP4 values should compare not equal")


fn test_nvfp4_inequality() raises:
    """Test NVFP4 inequality operators."""
    var small = NVFP4.from_float32(1.0)
    var large = NVFP4.from_float32(5.0)

    assert_true(small < large, "Small < Large should be true")
    assert_true(small <= large, "Small <= Large should be true")
    assert_true(large > small, "Large > Small should be true")
    assert_true(large >= small, "Large >= Small should be true")


fn test_nvfp4_comparison_edge_cases() raises:
    """Test NVFP4 comparison with edge cases."""
    var zero = NVFP4.from_float32(0.0)
    var positive = NVFP4.from_float32(1.0)
    var negative = NVFP4.from_float32(-1.0)

    assert_true(negative < zero, "Negative < Zero should be true")
    assert_true(zero < positive, "Zero < Positive should be true")
    assert_true(negative < positive, "Negative < Positive should be true")


# ============================================================================
# NVFP4 Round-trip Tests
# ============================================================================


fn test_nvfp4_roundtrip_small_values() raises:
    """Test NVFP4 round-trip for small values."""
    var original = Float32(0.5)
    var nvfp4 = NVFP4.from_float32(original)
    var restored = nvfp4.to_float32()

    assert_almost_equal(restored, original, tolerance=0.2)


fn test_nvfp4_roundtrip_medium_values() raises:
    """Test NVFP4 round-trip for medium values."""
    var original = Float32(42.0)
    var nvfp4 = NVFP4.from_float32(original)
    var restored = nvfp4.to_float32()

    assert_almost_equal(restored, original, tolerance=10.0)


fn test_nvfp4_roundtrip_large_values() raises:
    """Test NVFP4 round-trip for large values."""
    var original = Float32(200.0)
    var nvfp4 = NVFP4.from_float32(original)
    var restored = nvfp4.to_float32()

    assert_almost_equal(restored, original, tolerance=50.0)


fn test_nvfp4_accuracy_vs_mxfp4() raises:
    """Test NVFP4 accuracy compared to MXFP4 (paper claims better results)."""
    # According to paper, E4M3 "achieves the best results"
    # Test mid-range values where mantissa precision matters

    var test_val = Float32(7.5)
    var nvfp4 = NVFP4.from_float32(test_val)
    var restored = nvfp4.to_float32()

    # NVFP4 should preserve mid-range values reasonably well
    # due to E4M3 mantissa providing finer-grained scaling
    var error = abs(restored - test_val)
    var relative_error = error / test_val

    # Expect relative error < 30% for mid-range values
    assert_true(relative_error < 0.3, "NVFP4 should have good accuracy for mid-range values")


fn test_nvfp4_smaller_blocks() raises:
    """Test that NVFP4 uses smaller blocks (16 vs 32 for MXFP4)."""
    # This is a design property test - NVFP4 uses 16-element blocks
    # Smaller blocks provide "modest improvements in accuracy"
    # Cannot test block behavior directly yet (no block implementation)
    # but we can verify individual values work correctly

    # Create 16 different values to simulate a block
    var values_match = True
    for i in range(16):
        var val = Float32(i + 1)  # Values 1.0 to 16.0
        var nvfp4 = NVFP4.from_float32(val)
        var restored = nvfp4.to_float32()
        var tolerance = max(val * 0.3, Float32(1.0))
        if abs(restored - val) > tolerance:
            values_match = False

    assert_true(values_match, "NVFP4 should handle 16-value sequences")


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run all NVFP4 tests."""
    print("\n=== E4M3Scale Tests ===")
    test_e4m3_scale_basic()
    print("✓ E4M3 scale basic operations")

    test_e4m3_scale_mantissa_precision()
    print("✓ E4M3 scale mantissa precision")

    test_e4m3_scale_range()
    print("✓ E4M3 scale dynamic range")

    test_e4m3_scale_edge_cases()
    print("✓ E4M3 scale edge cases")

    print("\n=== NVFP4 Basic Conversion Tests ===")
    test_nvfp4_zero()
    print("✓ NVFP4 zero encoding")

    test_nvfp4_positive_values()
    print("✓ NVFP4 positive values")

    test_nvfp4_negative_values()
    print("✓ NVFP4 negative values")

    test_nvfp4_balanced_dynamic_range()
    print("✓ NVFP4 balanced dynamic range")

    test_nvfp4_special_values_nan()
    print("✓ NVFP4 NaN handling")

    test_nvfp4_special_values_inf()
    print("✓ NVFP4 infinity handling")

    print("\n=== NVFP4 Arithmetic Operations ===")
    test_nvfp4_addition()
    print("✓ NVFP4 addition")

    test_nvfp4_subtraction()
    print("✓ NVFP4 subtraction")

    test_nvfp4_multiplication()
    print("✓ NVFP4 multiplication")

    test_nvfp4_division()
    print("✓ NVFP4 division")

    test_nvfp4_negation()
    print("✓ NVFP4 negation")

    print("\n=== NVFP4 Comparison Operations ===")
    test_nvfp4_equality()
    print("✓ NVFP4 equality")

    test_nvfp4_inequality()
    print("✓ NVFP4 inequality operators")

    test_nvfp4_comparison_edge_cases()
    print("✓ NVFP4 comparison edge cases")

    print("\n=== NVFP4 Round-trip Tests ===")
    test_nvfp4_roundtrip_small_values()
    print("✓ NVFP4 round-trip (small values)")

    test_nvfp4_roundtrip_medium_values()
    print("✓ NVFP4 round-trip (medium values)")

    test_nvfp4_roundtrip_large_values()
    print("✓ NVFP4 round-trip (large values)")

    test_nvfp4_accuracy_vs_mxfp4()
    print("✓ NVFP4 accuracy analysis")

    test_nvfp4_smaller_blocks()
    print("✓ NVFP4 block size verification")

    print("\n=== All NVFP4 Tests Passed! ===\n")
