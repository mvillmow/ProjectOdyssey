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

from tests.shared.conftest import (
    assert_almost_equal,
    assert_close_float,
    assert_equal,
    assert_equal_int,
    assert_true,
)
from shared.core.types.mxfp4 import MXFP4, MXFP4Block, E8M0Scale
from math import isnan, isinf
from collections import List


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
    assert_true(
        result_tiny > 0.0 and result_tiny < 1e-20,
        "MXFP4 should handle tiny values",
    )

    # Test very large value (within Float32 range)
    var mxfp4_huge = MXFP4.from_float32(1e30)
    var result_huge = mxfp4_huge.to_float32()
    assert_true(result_huge > 1e25, "MXFP4 should handle huge values")


fn test_mxfp4_special_values_nan() raises:
    """Test MXFP4 encoding of NaN."""
    var nan_val = Float32(0.0) / Float32(0.0)
    var mxfp4_nan = MXFP4.from_float32(nan_val)
    var _ = mxfp4_nan.to_float32()

    # NaN encoding: E2M1 value should be max (0b0111)
    assert_equal(mxfp4_nan.value.value, 0b0111)


fn test_mxfp4_special_values_inf() raises:
    """Test MXFP4 encoding of infinity."""
    # Positive infinity
    var pos_inf = Float32(1.0) / Float32(0.0)
    var mxfp4_pos_inf = MXFP4.from_float32(pos_inf)
    var _ = mxfp4_pos_inf.to_float32()

    # Should encode as max E2M1 value with max scale
    assert_equal(mxfp4_pos_inf.value.value, 0b0111)

    # Negative infinity
    var neg_inf = Float32(-1.0) / Float32(0.0)
    var mxfp4_neg_inf = MXFP4.from_float32(neg_inf)
    var _ = mxfp4_neg_inf.to_float32()

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
    # Use values that encode to different E2M1 representations
    # 3.0 encodes to exp=2, mantissa=1 (value 3.0)
    # 6.0 encodes to exp=3, mantissa=1 (value 6.0)
    var a = MXFP4.from_float32(3.0)
    var b = MXFP4.from_float32(3.0)
    var c = MXFP4.from_float32(6.0)

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
    """Test MXFP4 round-trip for small values.

    Note: E2M1 has minimum normal value of 1.0, so values like 0.5 will
    quantize to either 0 or 1.0. A tolerance of 0.5 accounts for this.
    """
    var original = Float32(0.5)
    var mxfp4 = MXFP4.from_float32(original)
    var restored = mxfp4.to_float32()

    assert_almost_equal(restored, original, tolerance=0.5)


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
    assert_true(
        restored > 5000.0, "MXFP4 should preserve large values better than FP8"
    )


# ============================================================================
# MXFP4Block Scale Edge Case Tests (GitHub Issue #2379)
# ============================================================================


fn test_e8m0_scale_from_zero() raises:
    """Test E8M0Scale.from_float32(0.0) direct behavior.

    Edge case: Zero scale should return minimum scale (exponent = 0).
    This represents 2^(0 - 127) = 2^-127, the smallest representable scale.
    """
    var scale = E8M0Scale.from_float32(0.0)
    assert_equal(scale.exponent, 0)

    # Convert back to float and verify it's a very small positive number
    var scale_f32 = scale.to_float32()
    assert_true(
        scale_f32 > 0.0, "Zero scale should convert to tiny positive value"
    )
    assert_true(scale_f32 < 1e-30, "Zero scale should be extremely small")


fn test_mxfp4_block_all_zeros() raises:
    """Test MXFP4Block with all zeros.

    Edge case: Block with all zeros should:
    1. Trigger the fallback (max_abs = 0.0 < 1e-10)
    2. Use scale = 1.0 (exponent = 127)
    3. Encode all values as 0 (since 0.0 / scale = 0.0)
    4. Round-trip losslessly to zeros.
    """
    # Create a block with 32 zeros
    var values = List[Float32]()
    for _ in range(32):
        values.append(Float32(0.0))

    # Encode to MXFP4Block
    var block = MXFP4Block.from_float32_array(values)

    # Verify scale is 1.0 (exponent = 127)
    assert_equal(block.scale.exponent, 127)
    var scale_f32 = block.scale.to_float32()
    assert_almost_equal(scale_f32, Float32(1.0), tolerance=1e-6)

    # Verify all packed values are 0
    for i in range(16):
        assert_equal(block.data[i], 0)

    # Decode and verify round-trip
    var decoded = block.to_float32_array()
    assert_equal(len(decoded), 32)
    for i in range(32):
        assert_almost_equal(decoded[i], Float32(0.0), tolerance=1e-7)


fn test_mxfp4_block_near_zero_values() raises:
    """Test MXFP4Block with very small values < 1e-10.

    Edge case: Block with values < 1e-10 should trigger fallback to scale=1.0.
    Normally scale = max(abs) / 6.0 would be tiny. The fallback ensures
    we use scale = 1.0 instead, avoiding numerical instability.
    """
    # Create a block with values < 1e-10
    var values = List[Float32]()
    for i in range(32):
        # Use values like 1e-12, 5e-13, etc.
        values.append(Float32(1e-12) * Float32(i + 1))

    # Encode to MXFP4Block
    var block = MXFP4Block.from_float32_array(values)

    # Verify scale is 1.0 (exponent = 127)
    assert_equal(block.scale.exponent, 127)
    var scale_f32 = block.scale.to_float32()
    assert_almost_equal(scale_f32, Float32(1.0), tolerance=1e-6)

    # Verify no NaN or Inf in decoded values
    var decoded = block.to_float32_array()
    assert_equal(len(decoded), 32)
    for i in range(32):
        # Values should be either 0 or very small (quantized to E2M1 minimum)
        var val = decoded[i]
        assert_true(not isnan(val), "Decoded value should not be NaN")
        assert_true(not isinf(val), "Decoded value should not be Inf")


fn test_mxfp4_block_near_threshold_edge_case() raises:
    """Test MXFP4Block with values near the 1e-10 threshold.

    Edge case: max_abs values near 1e-10 (where max_abs / 6.0 < 1e-10).
    To trigger fallback with threshold 1e-10, max_abs must be < 6e-10.
    Using values around 1e-11 ensures fallback is triggered.
    """
    # Create a block with values that trigger the fallback
    # max_abs = 1e-11 * 32 = 3.2e-10, and 3.2e-10 / 6.0 = 5.3e-11 < 1e-10 (fallback triggers)
    var values = List[Float32]()
    var threshold_val = Float32(1e-11)
    for i in range(32):
        values.append(threshold_val * Float32(i + 1))

    # Encode to MXFP4Block
    var block = MXFP4Block.from_float32_array(values)

    # Verify scale is 1.0 (fallback applies)
    assert_equal(block.scale.exponent, 127)

    # Verify decode works without NaN/Inf
    var decoded = block.to_float32_array()
    assert_equal(len(decoded), 32)
    for i in range(32):
        assert_true(not isnan(decoded[i]), "Should not produce NaN")
        assert_true(not isinf(decoded[i]), "Should not produce Inf")


fn test_mxfp4_block_mixed_zero_and_small() raises:
    """Test MXFP4Block with mixed zeros and tiny values.

    Edge case: Block with some zeros and some tiny values (mixed scenario).
    Should still trigger fallback since max_abs might be < 1e-10.
    """
    var values = List[Float32]()

    # Mix zeros and small values
    for i in range(32):
        if i % 2 == 0:
            values.append(Float32(0.0))
        else:
            values.append(Float32(1e-12) * Float32(i))

    # Encode to MXFP4Block
    var block = MXFP4Block.from_float32_array(values)

    # Should use fallback scale = 1.0
    assert_equal(block.scale.exponent, 127)

    # Round-trip should work
    var decoded = block.to_float32_array()
    assert_equal(len(decoded), 32)

    # Verify decoded values
    for i in range(32):
        if i % 2 == 0:
            # Zeros should round-trip to zero
            assert_almost_equal(decoded[i], Float32(0.0), tolerance=1e-7)
        else:
            # Non-zeros should remain non-negative and not be NaN/Inf
            assert_true(decoded[i] >= 0.0, "Should remain non-negative")
            assert_true(not isnan(decoded[i]), "Should not be NaN")
            assert_true(not isinf(decoded[i]), "Should not be Inf")


fn test_mxfp4_block_zero_roundtrip_lossless() raises:
    """Test that zero blocks round-trip losslessly.

    Requirement: Zero blocks should encode as all-zero with scale=1.0
    and decode back to all zeros without any data corruption.
    """
    var values = List[Float32]()
    for _ in range(32):
        values.append(Float32(0.0))

    # Encode
    var block = MXFP4Block.from_float32_array(values)

    # Verify internal representation
    var scale_f32 = block.scale.to_float32()
    assert_almost_equal(scale_f32, Float32(1.0), tolerance=1e-6)

    # Verify all bits are zero
    for i in range(16):
        assert_equal(block.data[i], 0)

    # Decode
    var decoded = block.to_float32_array()

    # Verify perfect round-trip
    assert_equal(len(decoded), 32)
    for i in range(32):
        assert_almost_equal(decoded[i], Float32(0.0), tolerance=1e-10)


fn test_mxfp4_block_scale_computation_no_division_by_zero() raises:
    """Test that scale computation handles division by zero safely.

    Requirement: When max_abs = 0.0, computing scale = max_abs / 6.0 = 0.0
    should not cause NaN or undefined behavior. The fallback to scale=1.0
    should handle this gracefully.
    """
    # Create multiple blocks with all zeros to ensure consistent behavior
    for _ in range(3):
        var values = List[Float32]()
        for _ in range(32):
            values.append(Float32(0.0))

        var block = MXFP4Block.from_float32_array(values)

        # Should always have scale = 1.0
        var scale_f32 = block.scale.to_float32()
        assert_almost_equal(scale_f32, Float32(1.0), tolerance=1e-6)

        # Decode should be consistent
        var decoded = block.to_float32_array()
        for i in range(32):
            assert_equal(decoded[i], Float32(0.0))


fn test_mxfp4_block_normal_scale_computation_still_works() raises:
    """Test that normal (non-zero) scale computation still works correctly.

    Regression: Ensure that adding fallback logic for zero doesn't break
    normal scale computation for typical values.
    """
    # Create block with normal-sized values (max_abs = 12.0)
    var values = List[Float32]()
    for i in range(32):
        values.append(Float32(i + 1) * 0.4)  # Range 0.4 to 12.8

    # Encode
    var block = MXFP4Block.from_float32_array(values)

    # Scale should be computed as max(12.8) / 6.0 ≈ 2.13
    var scale_f32 = block.scale.to_float32()
    # Scale should be reasonable - definitely above 1.0 for these values
    assert_true(
        scale_f32 >= 1.5,
        "Scale should be computed normally for non-zero values",
    )
    assert_true(scale_f32 <= 4.0, "Scale should be reasonable")

    # Decode and verify reasonable round-trip
    var decoded = block.to_float32_array()
    assert_equal(len(decoded), 32)


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

    print("\n=== MXFP4Block Scale Edge Case Tests ===")
    test_e8m0_scale_from_zero()
    print("✓ E8M0 scale from zero")

    test_mxfp4_block_all_zeros()
    print("✓ MXFP4 block with all zeros")

    test_mxfp4_block_near_zero_values()
    print("✓ MXFP4 block with near-zero values")

    test_mxfp4_block_near_threshold_edge_case()
    print("✓ MXFP4 block near 1e-10 threshold")

    test_mxfp4_block_mixed_zero_and_small()
    print("✓ MXFP4 block mixed zeros and tiny values")

    test_mxfp4_block_zero_roundtrip_lossless()
    print("✓ MXFP4 block zero round-trip (lossless)")

    test_mxfp4_block_scale_computation_no_division_by_zero()
    print("✓ MXFP4 block scale computation (no division by zero)")

    test_mxfp4_block_normal_scale_computation_still_works()
    print("✓ MXFP4 block normal scale computation (regression test)")

    print("\n=== All MXFP4 Tests Passed! ===\n")
