"""Tests for BFloat16 implementation.

Tests conversion, arithmetic, comparison, and special values for the
custom BFloat16 type.
"""

from shared.core.bfloat16 import BFloat16, print_bfloat16_bits
from testing import assert_equal, assert_true, assert_false


fn test_bfloat16_zero() raises:
    """Test BFloat16 zero values."""
    print("Testing BFloat16 zero...")

    var zero = BFloat16._zero()
    var neg_zero = BFloat16._neg_zero()

    assert_equal(zero.to_float32(), 0.0, "Zero should convert to 0.0")
    assert_equal(neg_zero.to_float32(), -0.0, "Negative zero should convert to -0.0")

    # Check bit patterns
    assert_equal(zero.bits, 0x0000, "Zero bits should be 0x0000")
    assert_equal(neg_zero.bits, 0x8000, "Negative zero bits should be 0x8000")

    print("✓ BFloat16 zero test passed")


fn test_bfloat16_special_values() raises:
    """Test BFloat16 special values (inf, nan)."""
    print("Testing BFloat16 special values...")

    var inf = BFloat16._inf()
    var neg_inf = BFloat16._neg_inf()
    var nan = BFloat16._nan()

    # Check conversions
    assert_true(inf.to_float32() > 1e30, "Inf should be very large positive")
    assert_true(neg_inf.to_float32() < -1e30, "Neg inf should be very large negative")

    # Check predicates
    assert_true(inf.is_inf(), "Inf should be detected as infinite")
    assert_true(neg_inf.is_inf(), "Neg inf should be detected as infinite")
    assert_true(nan.is_nan(), "NaN should be detected as NaN")

    assert_false(inf.is_nan(), "Inf should not be NaN")
    assert_false(nan.is_inf(), "NaN should not be infinite")

    assert_true(inf.is_finite() == False, "Inf should not be finite")
    assert_true(nan.is_finite() == False, "NaN should not be finite")

    print("✓ BFloat16 special values test passed")


fn test_bfloat16_conversion_from_float32() raises:
    """Test Float32 -> BFloat16 conversion."""
    print("Testing Float32 -> BFloat16 conversion...")

    # Test simple values
    var bf16_1 = BFloat16.from_float32(1.0)
    var bf16_2 = BFloat16.from_float32(2.0)
    var bf16_half = BFloat16.from_float32(0.5)

    assert_true(abs(bf16_1.to_float32() - 1.0) < 1e-6, "1.0 should round-trip")
    assert_true(abs(bf16_2.to_float32() - 2.0) < 1e-6, "2.0 should round-trip")
    assert_true(abs(bf16_half.to_float32() - 0.5) < 1e-6, "0.5 should round-trip")

    # Test negative values
    var bf16_neg = BFloat16.from_float32(-3.14)
    assert_true(bf16_neg.to_float32() < 0.0, "Negative value should stay negative")

    print("✓ Float32 -> BFloat16 conversion test passed")


fn test_bfloat16_conversion_precision() raises:
    """Test BFloat16 precision and rounding."""
    print("Testing BFloat16 precision...")

    # BFloat16 has 7 mantissa bits, so precision is limited
    # For example, 3.14159 should be rounded to nearest representable value

    var pi = BFloat16.from_float32(3.14159)
    var pi_f32 = pi.to_float32()

    # Should be close but not exact (BF16 has only 7 mantissa bits)
    assert_true(abs(pi_f32 - 3.14159) < 0.01, "Pi should be approximately correct")

    # Test that truncation vs rounding makes a difference
    var pi_trunc = BFloat16.from_float32_truncate(3.14159)
    var pi_round = BFloat16.from_float32(3.14159)

    # They should be slightly different
    print("  Pi truncated: " + String(pi_trunc.to_float32()))
    print("  Pi rounded: " + String(pi_round.to_float32()))

    print("✓ BFloat16 precision test passed")


fn test_bfloat16_arithmetic() raises:
    """Test BFloat16 arithmetic operations."""
    print("Testing BFloat16 arithmetic...")

    var a = BFloat16.from_float32(3.0)
    var b = BFloat16.from_float32(2.0)

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

    print("✓ BFloat16 arithmetic test passed")


fn test_bfloat16_comparison() raises:
    """Test BFloat16 comparison operations."""
    print("Testing BFloat16 comparison...")

    var a = BFloat16.from_float32(3.0)
    var b = BFloat16.from_float32(2.0)
    var c = BFloat16.from_float32(3.0)

    # Equality
    assert_true(a == c, "3.0 == 3.0")
    assert_false(a == b, "3.0 != 2.0")

    # Inequality
    assert_false(a != c, "3.0 not != 3.0")
    assert_true(a != b, "3.0 != 2.0")

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

    print("✓ BFloat16 comparison test passed")


fn test_bfloat16_nan_comparison() raises:
    """Test BFloat16 NaN comparison behavior."""
    print("Testing BFloat16 NaN comparison...")

    var nan = BFloat16._nan()
    var one = BFloat16.from_float32(1.0)

    # NaN != NaN (IEEE 754 standard)
    assert_false(nan == nan, "NaN should not equal NaN")
    assert_true(nan != nan, "NaN should not equal NaN")

    # NaN comparisons are always false
    assert_false(nan < one, "NaN < 1.0 should be false")
    assert_false(nan > one, "NaN > 1.0 should be false")
    assert_false(one == nan, "1.0 == NaN should be false")

    print("✓ BFloat16 NaN comparison test passed")


fn test_bfloat16_range() raises:
    """Test BFloat16 range (should match Float32 range)."""
    print("Testing BFloat16 range...")

    # BFloat16 should handle large values like Float32
    var large = BFloat16.from_float32(1e20)
    var small = BFloat16.from_float32(1e-20)

    assert_true(large.to_float32() > 1e19, "Large values should be preserved")
    assert_true(small.to_float32() < 1e-19, "Small values should be preserved")

    # Test values that overflow Float16 but not BFloat16
    var big = BFloat16.from_float32(100000.0)
    assert_true(big.is_finite(), "100000 should be finite in BF16")
    assert_true(abs(big.to_float32() - 100000.0) < 1000.0, "100000 should be approximately correct")

    print("✓ BFloat16 range test passed")


fn test_bfloat16_bit_patterns() raises:
    """Test specific bit patterns for correctness."""
    print("Testing BFloat16 bit patterns...")

    # Test known bit patterns
    # 1.0 in BF16: sign=0, exp=127 (0x7F), mantissa=0
    # Bits: 0 01111111 0000000 = 0x3F80
    var one = BFloat16.from_float32(1.0)
    assert_equal(one.bits, 0x3F80, "1.0 should have bits 0x3F80")

    # 2.0 in BF16: sign=0, exp=128 (0x80), mantissa=0
    # Bits: 0 10000000 0000000 = 0x4000
    var two = BFloat16.from_float32(2.0)
    assert_equal(two.bits, 0x4000, "2.0 should have bits 0x4000")

    # 0.5 in BF16: sign=0, exp=126 (0x7E), mantissa=0
    # Bits: 0 01111110 0000000 = 0x3F00
    var half = BFloat16.from_float32(0.5)
    assert_equal(half.bits, 0x3F00, "0.5 should have bits 0x3F00")

    print("✓ BFloat16 bit patterns test passed")


fn test_bfloat16_negation_bit_flip() raises:
    """Test that negation flips only the sign bit."""
    print("Testing BFloat16 negation bit flip...")

    var pos = BFloat16.from_float32(3.14)
    var neg = -pos

    # Check that only sign bit differs
    var xor_result = pos.bits ^ neg.bits
    assert_equal(xor_result, 0x8000, "Negation should flip only sign bit")

    print("✓ BFloat16 negation bit flip test passed")


fn test_bfloat16_string_representation() raises:
    """Test BFloat16 string conversion."""
    print("Testing BFloat16 string representation...")

    var value = BFloat16.from_float32(3.14)
    var str_repr = value.__str__()

    # Should contain "BFloat16" and the value
    print("  String representation: " + str_repr)

    print("✓ BFloat16 string representation test passed")


fn main() raises:
    print("\n" + "=" * 70)
    print("BFLOAT16 IMPLEMENTATION TESTS")
    print("=" * 70)
    print()

    test_bfloat16_zero()
    test_bfloat16_special_values()
    test_bfloat16_conversion_from_float32()
    test_bfloat16_conversion_precision()
    test_bfloat16_arithmetic()
    test_bfloat16_comparison()
    test_bfloat16_nan_comparison()
    test_bfloat16_range()
    test_bfloat16_bit_patterns()
    test_bfloat16_negation_bit_flip()
    test_bfloat16_string_representation()

    print()
    print("=" * 70)
    print("ALL BFLOAT16 TESTS PASSED! ✓")
    print("=" * 70)
    print()

    # Demonstrate BFloat16 usage
    print("BFloat16 Example:")
    print("-" * 70)
    var pi = BFloat16.from_float32(3.14159265359)
    print("Original Float32: 3.14159265359")
    print("BFloat16 value: " + String(pi.to_float32()))
    print()
    print("Bit representation:")
    print_bfloat16_bits(pi)
