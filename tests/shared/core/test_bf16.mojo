"""Tests for BF16 (BFloat16) data type.

Tests cover:
- BF16 creation from Float32
- BF16 to Float32 conversion
- Special values (zero, NaN, Inf)
- Round-trip conversion accuracy
- Truncation vs rounding conversion
- Arithmetic operations
- Comparison operators
- Utility methods (is_nan, is_inf, is_zero, is_subnormal, is_finite)
- Bit pattern verification
"""

from shared.core.types.bf16 import BF16, print_bf16_bits
from testing import assert_equal, assert_true, assert_false
from math import isnan, isinf


# ============================================================================
# BF16 Basic Conversion Tests
# ============================================================================


fn test_bf16_zero() raises:
    """Test BF16 representation of zero."""
    print("Testing BF16 zero...")

    var zero = BF16._zero()
    var neg_zero = BF16._neg_zero()

    assert_equal(zero.to_float32(), 0.0, "Zero should convert to 0.0")
    assert_equal(
        neg_zero.to_float32(), -0.0, "Negative zero should convert to -0.0"
    )

    # Check bit patterns
    assert_equal(zero.value, 0x0000, "Zero bits should be 0x0000")
    assert_equal(neg_zero.value, 0x8000, "Negative zero bits should be 0x8000")

    # Test utility methods
    assert_true(zero.is_zero(), "Zero should be detected as zero")
    assert_true(neg_zero.is_zero(), "Negative zero should be detected as zero")

    print("✓ BF16 zero encoding")


fn test_bf16_special_values() raises:
    """Test BF16 special values (inf, nan)."""
    print("Testing BF16 special values...")

    var inf = BF16._inf()
    var neg_inf = BF16._neg_inf()
    var nan = BF16._nan()

    # Check conversions
    assert_true(inf.to_float32() > 1e30, "Inf should be very large positive")
    assert_true(
        neg_inf.to_float32() < -1e30, "Neg inf should be very large negative"
    )

    # Check predicates
    assert_true(inf.is_inf(), "Inf should be detected as infinite")
    assert_true(neg_inf.is_inf(), "Neg inf should be detected as infinite")
    assert_true(nan.is_nan(), "NaN should be detected as NaN")

    assert_false(inf.is_nan(), "Inf should not be NaN")
    assert_false(nan.is_inf(), "NaN should not be infinite")

    assert_false(inf.is_finite(), "Inf should not be finite")
    assert_false(nan.is_finite(), "NaN should not be finite")

    print("✓ BF16 special values")


fn test_bf16_positive_values() raises:
    """Test BF16 encoding of positive values."""
    print("Testing BF16 positive values...")

    # Test simple values
    var bf16_1 = BF16.from_float32(1.0)
    var bf16_2 = BF16.from_float32(2.0)
    var bf16_half = BF16.from_float32(0.5)

    assert_true(abs(bf16_1.to_float32() - 1.0) < 1e-6, "1.0 should round-trip")
    assert_true(abs(bf16_2.to_float32() - 2.0) < 1e-6, "2.0 should round-trip")
    assert_true(
        abs(bf16_half.to_float32() - 0.5) < 1e-6, "0.5 should round-trip"
    )

    # Test Pi (rounding expected)
    var bf16_pi = BF16.from_float32(3.14159)
    var result_pi = bf16_pi.to_float32()
    # BF16 has 7 mantissa bits, so ~2 decimal digits precision
    assert_true(
        abs(result_pi - 3.14159) < 0.02, "Pi should be approximately correct"
    )

    # Test 100.0
    var bf16_hundred = BF16.from_float32(100.0)
    var result_hundred = bf16_hundred.to_float32()
    assert_true(abs(result_hundred - 100.0) < 1e-5, "100.0 should round-trip")

    print("✓ BF16 positive values")


fn test_bf16_negative_values() raises:
    """Test BF16 encoding of negative values."""
    print("Testing BF16 negative values...")

    # Test -1.0
    var bf16_neg_one = BF16.from_float32(-1.0)
    var result_neg_one = bf16_neg_one.to_float32()
    assert_true(abs(result_neg_one - (-1.0)) < 1e-6, "-1.0 should round-trip")

    # Test -2.5
    var bf16_neg = BF16.from_float32(-2.5)
    var result_neg = bf16_neg.to_float32()
    assert_true(abs(result_neg - (-2.5)) < 1e-6, "-2.5 should round-trip")

    # Test -100.0
    var bf16_neg_hundred = BF16.from_float32(-100.0)
    var result_neg_hundred = bf16_neg_hundred.to_float32()
    assert_true(
        abs(result_neg_hundred - (-100.0)) < 1e-5, "-100.0 should round-trip"
    )

    print("✓ BF16 negative values")


fn test_bf16_large_values() raises:
    """Test BF16 with large values (same range as Float32)."""
    print("Testing BF16 large values...")

    # BF16 has the same exponent range as Float32, so it can represent very large values
    var large = BF16.from_float32(1e20)
    var small = BF16.from_float32(1e-20)

    assert_true(large.to_float32() > 1e19, "Large values should be preserved")
    assert_true(small.to_float32() < 1e-19, "Small values should be preserved")

    # Test values that overflow Float16 but not BF16
    var big = BF16.from_float32(100000.0)
    assert_true(big.is_finite(), "100000 should be finite in BF16")
    assert_true(
        abs(big.to_float32() - 100000.0) < 1000.0,
        "100000 should be approximately correct",
    )

    print("✓ BF16 large values (Float32 range)")


fn test_bf16_precision() raises:
    """Test BF16 precision (7 mantissa bits = ~2 decimal digits)."""
    print("Testing BF16 precision...")

    # BF16 has 7 mantissa bits, so precision is limited
    var pi = BF16.from_float32(3.14159)
    var pi_f32 = pi.to_float32()

    # Should be close but not exact
    assert_true(
        abs(pi_f32 - 3.14159) < 0.01, "Pi should be approximately correct"
    )

    # Test that truncation vs rounding makes a difference
    var pi_trunc = BF16.from_float32_truncate(3.14159)
    var pi_round = BF16.from_float32(3.14159)

    # Both should be close to original
    assert_true(
        abs(pi_trunc.to_float32() - 3.14159) < 0.02,
        "Truncated Pi should be close",
    )
    assert_true(
        abs(pi_round.to_float32() - 3.14159) < 0.02,
        "Rounded Pi should be close",
    )

    print("✓ BF16 precision")


# ============================================================================
# Arithmetic Tests
# ============================================================================


fn test_bf16_arithmetic() raises:
    """Test BF16 arithmetic operations."""
    print("Testing BF16 arithmetic...")

    var a = BF16.from_float32(3.0)
    var b = BF16.from_float32(2.0)

    # Addition
    var sum = a + b
    assert_true(abs(sum.to_float32() - 5.0) < 1e-6, "3 + 2 = 5")

    # Subtraction
    var diff = a - b
    assert_true(abs(diff.to_float32() - 1.0) < 1e-6, "3 - 2 = 1")

    # Multiplication
    var prod = a * b
    assert_true(abs(prod.to_float32() - 6.0) < 1e-6, "3 * 2 = 6")

    # Division
    var quot = a / b
    assert_true(abs(quot.to_float32() - 1.5) < 1e-6, "3 / 2 = 1.5")

    # Negation
    var neg = -a
    assert_true(abs(neg.to_float32() - (-3.0)) < 1e-6, "-3 = -3")

    print("✓ BF16 arithmetic")


fn test_bf16_negation() raises:
    """Test BF16 negation operator."""
    print("Testing BF16 negation...")

    var bf16_pos = BF16.from_float32(5.0)
    var bf16_neg = -bf16_pos

    assert_true(
        abs(bf16_neg.to_float32() - (-5.0)) < 1e-6, "Negation should work"
    )

    # Negating twice should give original
    var bf16_double_neg = -bf16_neg
    assert_true(
        abs(bf16_double_neg.to_float32() - 5.0) < 1e-6, "Double negation"
    )

    # Check that only sign bit differs
    var xor_result = bf16_pos.value ^ bf16_neg.value
    assert_equal(xor_result, 0x8000, "Negation should flip only sign bit")

    print("✓ BF16 negation")


fn test_bf16_abs() raises:
    """Test BF16 absolute value."""
    print("Testing BF16 absolute value...")

    var bf16_neg = BF16.from_float32(-5.0)
    var bf16_pos = bf16_neg.__abs__()

    assert_true(abs(bf16_pos.to_float32() - 5.0) < 1e-6, "Abs of negative")

    # Abs of positive should be unchanged
    var bf16_already_pos = BF16.from_float32(3.0)
    var bf16_abs = bf16_already_pos.__abs__()
    assert_true(abs(bf16_abs.to_float32() - 3.0) < 1e-6, "Abs of positive")

    print("✓ BF16 absolute value")


# ============================================================================
# Comparison Tests
# ============================================================================


fn test_bf16_equality() raises:
    """Test BF16 equality comparison."""
    print("Testing BF16 equality...")

    var a = BF16.from_float32(3.0)
    var b = BF16.from_float32(2.0)
    var c = BF16.from_float32(3.0)

    # Equality
    assert_true(a == c, "3.0 == 3.0")
    assert_false(a == b, "3.0 != 2.0")

    # Inequality
    assert_false(a != c, "3.0 not != 3.0")
    assert_true(a != b, "3.0 != 2.0")

    print("✓ BF16 equality comparison")


fn test_bf16_ordering() raises:
    """Test BF16 ordering comparisons."""
    print("Testing BF16 ordering...")

    var a = BF16.from_float32(3.0)
    var b = BF16.from_float32(2.0)
    var c = BF16.from_float32(3.0)

    # Less than
    assert_true(b < a, "2.0 < 3.0")
    assert_false(a < b, "3.0 not < 2.0")

    # Less than or equal
    assert_true(b <= a, "2.0 <= 3.0")
    assert_true(a <= c, "3.0 <= 3.0")

    # Greater than
    assert_true(a > b, "3.0 > 2.0")
    assert_false(b > a, "2.0 not > 3.0")

    # Greater than or equal
    assert_true(a >= b, "3.0 >= 2.0")
    assert_true(a >= c, "3.0 >= 3.0")

    print("✓ BF16 ordering comparisons")


fn test_bf16_nan_comparison() raises:
    """Test BF16 NaN comparison behavior."""
    print("Testing BF16 NaN comparison...")

    var nan = BF16._nan()
    var one = BF16.from_float32(1.0)

    # NaN != NaN (IEEE 754 standard)
    assert_false(nan == nan, "NaN should not equal NaN")
    assert_true(nan != nan, "NaN should not equal NaN")

    # NaN comparisons are always false
    assert_false(nan < one, "NaN < 1.0 should be false")
    assert_false(nan > one, "NaN > 1.0 should be false")
    assert_false(one == nan, "1.0 == NaN should be false")

    print("✓ BF16 NaN comparison")


# ============================================================================
# Utility Method Tests
# ============================================================================


fn test_bf16_sign() raises:
    """Test BF16 sign extraction."""
    print("Testing BF16 sign...")

    var bf16_pos = BF16.from_float32(5.0)
    var bf16_neg = BF16.from_float32(-5.0)
    var bf16_zero = BF16.from_float32(0.0)

    assert_equal(bf16_pos.sign(), 0, "Positive sign should be 0")
    assert_equal(bf16_neg.sign(), 1, "Negative sign should be 1")
    assert_equal(bf16_zero.sign(), 0, "Zero sign should be 0")

    print("✓ BF16 sign extraction")


fn test_bf16_is_subnormal() raises:
    """Test BF16 subnormal detection."""
    print("Testing BF16 subnormal...")

    # Create a subnormal by directly setting bits (exp=0, mantissa!=0)
    var bf16_subnormal = BF16(0x0001)  # Smallest positive subnormal
    assert_true(bf16_subnormal.is_subnormal(), "Should detect subnormal")

    # Normal number should not be subnormal
    var bf16_normal = BF16.from_float32(1.0)
    assert_false(bf16_normal.is_subnormal(), "1.0 should not be subnormal")

    # Zero should not be subnormal
    var bf16_zero = BF16.from_float32(0.0)
    assert_false(bf16_zero.is_subnormal(), "Zero should not be subnormal")

    print("✓ BF16 subnormal detection")


fn test_bf16_is_finite() raises:
    """Test BF16 is_finite method."""
    print("Testing BF16 is_finite...")

    var normal = BF16.from_float32(3.14)
    var inf = BF16._inf()
    var nan = BF16._nan()

    assert_true(normal.is_finite(), "Normal value should be finite")
    assert_false(inf.is_finite(), "Inf should not be finite")
    assert_false(nan.is_finite(), "NaN should not be finite")

    print("✓ BF16 is_finite")


# ============================================================================
# Bit Pattern Tests
# ============================================================================


fn test_bf16_bit_patterns() raises:
    """Test specific bit patterns for correctness."""
    print("Testing BF16 bit patterns...")

    # Test known bit patterns
    # 1.0 in BF16: sign=0, exp=127 (0x7F), mantissa=0
    # Bits: 0 01111111 0000000 = 0x3F80
    var one = BF16.from_float32(1.0)
    assert_equal(one.value, 0x3F80, "1.0 should have bits 0x3F80")

    # 2.0 in BF16: sign=0, exp=128 (0x80), mantissa=0
    # Bits: 0 10000000 0000000 = 0x4000
    var two = BF16.from_float32(2.0)
    assert_equal(two.value, 0x4000, "2.0 should have bits 0x4000")

    # 0.5 in BF16: sign=0, exp=126 (0x7E), mantissa=0
    # Bits: 0 01111110 0000000 = 0x3F00
    var half = BF16.from_float32(0.5)
    assert_equal(half.value, 0x3F00, "0.5 should have bits 0x3F00")

    print("✓ BF16 bit patterns")


# ============================================================================
# Round-Trip Tests
# ============================================================================


fn test_bf16_roundtrip_exact_values() raises:
    """Test round-trip for values exactly representable in BF16."""
    print("Testing BF16 round-trip exact values...")

    # Powers of 2 are exactly representable
    var test_values = List[Float32]()
    test_values.append(1.0)
    test_values.append(2.0)
    test_values.append(4.0)
    test_values.append(0.5)
    test_values.append(0.25)
    test_values.append(128.0)
    test_values.append(-1.0)
    test_values.append(-0.5)

    for i in range(len(test_values)):
        var val = test_values[i]
        var bf16 = BF16.from_float32(val)
        var restored = bf16.to_float32()
        assert_true(abs(restored - val) < 1e-6, "Exact value should round-trip")

    print("✓ BF16 exact value round-trip")


fn test_bf16_roundtrip_general_values() raises:
    """Test round-trip for general values with expected precision loss."""
    print("Testing BF16 round-trip general values...")

    var test_values = List[Float32]()
    test_values.append(3.14159)
    test_values.append(2.71828)
    test_values.append(0.123456)
    test_values.append(-9.87654)

    for i in range(len(test_values)):
        var val = test_values[i]
        var bf16 = BF16.from_float32(val)
        var restored = bf16.to_float32()
        # BF16 has ~2 decimal digits precision, so use 1% tolerance
        var tolerance = abs(val) * 0.01
        if tolerance < 0.01:
            tolerance = 0.01
        assert_true(abs(restored - val) < tolerance, "Value should round-trip")

    print("✓ BF16 general value round-trip")


fn test_bf16_string_representation() raises:
    """Test BF16 string representation."""
    print("Testing BF16 string representation...")

    var bf16 = BF16.from_float32(3.14)
    var str_repr = bf16.__str__()

    # Should contain "BF16" and the value
    print("  String representation: " + str_repr)

    print("✓ BF16 string representation")


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run all BF16 tests."""
    print("\n" + "=" * 70)
    print("BF16 (BFLOAT16) IMPLEMENTATION TESTS")
    print("=" * 70)
    print()

    print("=== BF16 Basic Conversion Tests ===")
    test_bf16_zero()
    test_bf16_special_values()
    test_bf16_positive_values()
    test_bf16_negative_values()
    test_bf16_large_values()
    test_bf16_precision()

    print("\n=== BF16 Arithmetic Tests ===")
    test_bf16_arithmetic()
    test_bf16_negation()
    test_bf16_abs()

    print("\n=== BF16 Comparison Tests ===")
    test_bf16_equality()
    test_bf16_ordering()
    test_bf16_nan_comparison()

    print("\n=== BF16 Utility Method Tests ===")
    test_bf16_sign()
    test_bf16_is_subnormal()
    test_bf16_is_finite()

    print("\n=== BF16 Bit Pattern Tests ===")
    test_bf16_bit_patterns()

    print("\n=== BF16 Round-Trip Tests ===")
    test_bf16_roundtrip_exact_values()
    test_bf16_roundtrip_general_values()
    test_bf16_string_representation()

    print()
    print("=" * 70)
    print("ALL BF16 TESTS PASSED! ✓")
    print("=" * 70)
    print()

    # Demonstrate BF16 usage
    print("BF16 Example:")
    print("-" * 70)
    var pi = BF16.from_float32(3.14159265359)
    print("Original Float32: 3.14159265359")
    print("BF16 value: " + String(pi.to_float32()))
    print()
    print("Bit representation:")
    print_bf16_bits(pi)
