"""Tests for unsigned integer type wrappers (UInt8, UInt16, UInt32, UInt64)."""

from testing import assert_equal, assert_true

from shared.core.types.unsigned import UInt8, UInt16, UInt32, UInt64
from shared.core.extensor import ExTensor, zeros
from shared.core.dtype import DType


fn test_uint8_construction() raises:
    """Test UInt8 construction from various values."""
    # Normal values
    var u1 = UInt8(42)
    assert_equal(String(u1.value), "42")

    var u2 = UInt8(200)
    assert_equal(String(u2.value), "200")

    var u3 = UInt8(0)
    assert_equal(String(u3.value), "0")

    # Boundary values
    var u4 = UInt8(255)  # Max value
    assert_equal(String(u4.value), "255")

    print("✓ test_uint8_construction passed")


fn test_uint8_clamping() raises:
    """Test UInt8 value clamping at boundaries."""
    # Values exceeding max should clamp to 255
    var u1 = UInt8(300)
    assert_equal(String(u1.value), "255")

    var u2 = UInt8(1000)
    assert_equal(String(u2.value), "255")

    # Negative values should clamp to 0
    var u3 = UInt8(-10)
    assert_equal(String(u3.value), "0")

    var u4 = UInt8(-1000)
    assert_equal(String(u4.value), "0")

    print("✓ test_uint8_clamping passed")


fn test_uint8_float_conversion() raises:
    """Test UInt8 conversion to/from float."""
    # Positive values
    var u1 = UInt8.from_float32(42.7)
    assert_equal(String(u1.value), "42")  # Truncates

    var f1 = u1.to_float32()
    assert_equal(String(f1), "42.0")

    # Zero
    var u2 = UInt8.from_float32(0.0)
    assert_equal(String(u2.value), "0")

    # Clamping from float
    var u3 = UInt8.from_float32(500.0)
    assert_equal(String(u3.value), "255")

    var u4 = UInt8.from_float32(-50.0)
    assert_equal(String(u4.value), "0")  # Negative clamped to 0

    print("✓ test_uint8_float_conversion passed")


fn test_uint8_comparisons() raises:
    """Test UInt8 comparison operators."""
    var u1 = UInt8(10)
    var u2 = UInt8(20)
    var u3 = UInt8(10)

    # Equality
    assert_true(u1 == u3)
    assert_true(u1 != u2)

    # Less than / greater than
    assert_true(u1 < u2)
    assert_true(u2 > u1)
    assert_true(u1 <= u3)
    assert_true(u1 >= u3)

    print("✓ test_uint8_comparisons passed")


fn test_uint8_arithmetic() raises:
    """Test UInt8 arithmetic operators."""
    var u1 = UInt8(10)
    var u2 = UInt8(5)

    # Addition
    var sum = u1 + u2
    assert_equal(String(sum.value), "15")

    # Subtraction (with wrapping behavior)
    var diff = u1 - u2
    assert_equal(String(diff.value), "5")

    # Multiplication
    var prod = u1 * u2
    assert_equal(String(prod.value), "50")

    # Division
    var quot = u1 / u2
    assert_equal(String(quot.value), "2")

    # Modulo
    var rem = u1 % UInt8(3)
    assert_equal(String(rem.value), "1")

    print("✓ test_uint8_arithmetic passed")


fn test_uint8_string_representation() raises:
    """Test UInt8 string representation."""
    var u1 = UInt8(42)
    assert_equal(str(u1), "42")
    assert_equal(repr(u1), "UInt8(42)")

    var u2 = UInt8(200)
    assert_equal(str(u2), "200")

    print("✓ test_uint8_string_representation passed")


fn test_uint16_construction() raises:
    """Test UInt16 construction and boundary values."""
    var u1 = UInt16(1000)
    assert_equal(String(u1.value), "1000")

    var u2 = UInt16(0)  # Min value
    assert_equal(String(u2.value), "0")

    var u3 = UInt16(65535)  # Max value
    assert_equal(String(u3.value), "65535")

    # Clamping
    var u4 = UInt16(100000)
    assert_equal(String(u4.value), "65535")

    var u5 = UInt16(-100)
    assert_equal(String(u5.value), "0")

    print("✓ test_uint16_construction passed")


fn test_uint16_conversions() raises:
    """Test UInt16 conversions to/from other types."""
    # From UInt8
    var u8 = UInt8(100)
    var u16 = UInt16.from_uint8(u8.value)
    assert_equal(String(u16.value), "100")

    # To UInt8 with clamping
    var large = UInt16(1000)
    var back_to_u8 = large.to_uint8()
    assert_equal(String(back_to_u8), "255")  # Clamped

    # To UInt32
    var u32 = large.to_uint32()
    assert_equal(String(u32), "1000")

    print("✓ test_uint16_conversions passed")


fn test_uint32_construction() raises:
    """Test UInt32 construction and boundary values."""
    var u1 = UInt32(100000)
    assert_equal(String(u1.value), "100000")

    var u2 = UInt32(0)  # Min value
    assert_equal(String(u2.value), "0")

    var u3 = UInt32(4294967295)  # Max value
    assert_equal(String(u3.value), "4294967295")

    # Negative clamping
    var u4 = UInt32(-100)
    assert_equal(String(u4.value), "0")

    print("✓ test_uint32_construction passed")


fn test_uint32_conversions() raises:
    """Test UInt32 conversions."""
    # From UInt8
    var u8 = UInt8(42)
    var u32 = UInt32.from_uint8(u8.value)
    assert_equal(String(u32.value), "42")

    # From UInt16
    var u16 = UInt16(1000)
    var u32_2 = UInt32.from_uint16(u16.value)
    assert_equal(String(u32_2.value), "1000")

    # To UInt16 with clamping
    var large = UInt32(100000)
    var back_to_u16 = large.to_uint16()
    assert_equal(String(back_to_u16), "65535")

    print("✓ test_uint32_conversions passed")


fn test_uint64_construction() raises:
    """Test UInt64 construction."""
    var u1 = UInt64(9223372036854775807)  # Large value
    assert_equal(String(u1.value), "9223372036854775807")

    var u2 = UInt64(0)
    assert_equal(String(u2.value), "0")

    # Negative clamping
    var u3 = UInt64(-100)
    assert_equal(String(u3.value), "0")

    print("✓ test_uint64_construction passed")


fn test_uint64_conversions() raises:
    """Test UInt64 conversions."""
    # From smaller types
    var u8 = UInt8(42)
    var u64 = UInt64.from_uint8(u8.value)
    assert_equal(String(u64.value), "42")

    var u16 = UInt16(1000)
    var u64_2 = UInt64.from_uint16(u16.value)
    assert_equal(String(u64_2.value), "1000")

    var u32 = UInt32(1000000)
    var u64_3 = UInt64.from_uint32(u32.value)
    assert_equal(String(u64_3.value), "1000000")

    # To smaller types with clamping
    var large = UInt64(100000)
    var back_to_u8 = large.to_uint8()
    assert_equal(String(back_to_u8), "255")

    var back_to_u16 = large.to_uint16()
    assert_equal(String(back_to_u16), "65535")

    var back_to_u32 = large.to_uint32()
    assert_equal(String(back_to_u32), "100000")

    print("✓ test_uint64_conversions passed")


fn test_tensor_uint8_conversion() raises:
    """Test tensor conversion to/from UInt8."""
    # Create a float32 tensor
    var t = zeros(List[Int](2, 3), DType.float32)

    # Set some values
    t._data.bitcast[Float32]()[0] = 10.5
    t._data.bitcast[Float32]()[1] = 200.3
    t._data.bitcast[Float32]()[2] = 255.9
    t._data.bitcast[Float32]()[3] = -10.0  # Should clamp to 0
    t._data.bitcast[Float32]()[4] = 0.0
    t._data.bitcast[Float32]()[5] = 42.7

    # Convert to uint8
    var u8_tensor = t.to_uint8()

    # Verify dtype
    assert_equal(u8_tensor._dtype, DType.uint8)

    # Verify values (truncated and clamped)
    assert_equal(String(u8_tensor._data.bitcast[UInt8]()[0]), "10")
    assert_equal(String(u8_tensor._data.bitcast[UInt8]()[1]), "200")
    assert_equal(String(u8_tensor._data.bitcast[UInt8]()[2]), "255")  # Clamped
    assert_equal(String(u8_tensor._data.bitcast[UInt8]()[3]), "0")  # Clamped from negative
    assert_equal(String(u8_tensor._data.bitcast[UInt8]()[4]), "0")
    assert_equal(String(u8_tensor._data.bitcast[UInt8]()[5]), "42")

    print("✓ test_tensor_uint8_conversion passed")


fn test_tensor_uint16_conversion() raises:
    """Test tensor conversion to/from UInt16."""
    var t = zeros(List[Int](2, 2), DType.float32)

    t._data.bitcast[Float32]()[0] = 1000.5
    t._data.bitcast[Float32]()[1] = 50000.3
    t._data.bitcast[Float32]()[2] = 65535.9
    t._data.bitcast[Float32]()[3] = -100.0

    var u16_tensor = t.to_uint16()

    assert_equal(u16_tensor._dtype, DType.uint16)
    assert_equal(String(u16_tensor._data.bitcast[UInt16]()[0]), "1000")
    assert_equal(String(u16_tensor._data.bitcast[UInt16]()[1]), "50000")
    assert_equal(String(u16_tensor._data.bitcast[UInt16]()[2]), "65535")  # Clamped
    assert_equal(String(u16_tensor._data.bitcast[UInt16]()[3]), "0")  # Clamped from negative

    print("✓ test_tensor_uint16_conversion passed")


fn test_tensor_uint32_conversion() raises:
    """Test tensor conversion to/from UInt32."""
    var t = zeros(List[Int](2, 2), DType.float32)

    t._data.bitcast[Float32]()[0] = 100000.0
    t._data.bitcast[Float32]()[1] = 1000000.0
    t._data.bitcast[Float32]()[2] = 0.0
    t._data.bitcast[Float32]()[3] = -42.5

    var u32_tensor = t.to_uint32()

    assert_equal(u32_tensor._dtype, DType.uint32)
    assert_equal(String(u32_tensor._data.bitcast[UInt32]()[0]), "100000")
    assert_equal(String(u32_tensor._data.bitcast[UInt32]()[1]), "1000000")
    assert_equal(String(u32_tensor._data.bitcast[UInt32]()[2]), "0")
    assert_equal(String(u32_tensor._data.bitcast[UInt32]()[3]), "0")  # Clamped from negative

    print("✓ test_tensor_uint32_conversion passed")


fn test_tensor_uint64_conversion() raises:
    """Test tensor conversion to/from UInt64."""
    var t = zeros(List[Int](2, 2), DType.float32)

    t._data.bitcast[Float32]()[0] = 1000000.0
    t._data.bitcast[Float32]()[1] = 10000000.0
    t._data.bitcast[Float32]()[2] = 0.0
    t._data.bitcast[Float32]()[3] = -12345.5

    var u64_tensor = t.to_uint64()

    assert_equal(u64_tensor._dtype, DType.uint64)
    assert_equal(String(u64_tensor._data.bitcast[UInt64]()[0]), "1000000")
    assert_equal(String(u64_tensor._data.bitcast[UInt64]()[1]), "10000000")
    assert_equal(String(u64_tensor._data.bitcast[UInt64]()[2]), "0")
    assert_equal(String(u64_tensor._data.bitcast[UInt64]()[3]), "0")  # Clamped from negative

    print("✓ test_tensor_uint64_conversion passed")


fn test_tensor_unsigned_round_trip() raises:
    """Test round-trip conversion from float -> uint -> float."""
    var t = zeros(List[Int](3,), DType.float32)

    t._data.bitcast[Float32]()[0] = 10.0
    t._data.bitcast[Float32]()[1] = 200.0
    t._data.bitcast[Float32]()[2] = 0.0

    # Convert to uint8 and back (implicitly through ExTensor operations)
    var u8_tensor = t.to_uint8()

    # Manually convert back to float
    var t_back = zeros(List[Int](3,), DType.float32)
    t_back._data.bitcast[Float32]()[0] = Float32(u8_tensor._data.bitcast[UInt8]()[0])
    t_back._data.bitcast[Float32]()[1] = Float32(u8_tensor._data.bitcast[UInt8]()[1])
    t_back._data.bitcast[Float32]()[2] = Float32(u8_tensor._data.bitcast[UInt8]()[2])

    # Verify values match
    assert_equal(String(t_back._data.bitcast[Float32]()[0]), "10.0")
    assert_equal(String(t_back._data.bitcast[Float32]()[1]), "200.0")
    assert_equal(String(t_back._data.bitcast[Float32]()[2]), "0.0")

    print("✓ test_tensor_unsigned_round_trip passed")


fn test_mixed_signed_unsigned() raises:
    """Test interactions between signed and unsigned types."""
    # Create signed and unsigned values
    var i8 = Int8(-10)
    var u8_from_neg = UInt8.from_float32(i8.to_float32())
    assert_equal(String(u8_from_neg.value), "0")  # Negative clamped to 0

    var u8 = UInt8(200)
    var i8_from_large = Int8.from_float32(u8.to_float32())
    assert_equal(String(i8_from_large.value), "127")  # Large unsigned clamped to signed max

    print("✓ test_mixed_signed_unsigned passed")


fn main() raises:
    """Run all unsigned integer type tests."""
    print("\n=== Running Unsigned Integer Type Tests ===\n")

    # UInt8 tests
    test_uint8_construction()
    test_uint8_clamping()
    test_uint8_float_conversion()
    test_uint8_comparisons()
    test_uint8_arithmetic()
    test_uint8_string_representation()

    # UInt16 tests
    test_uint16_construction()
    test_uint16_conversions()

    # UInt32 tests
    test_uint32_construction()
    test_uint32_conversions()

    # UInt64 tests
    test_uint64_construction()
    test_uint64_conversions()

    # Tensor conversion tests
    test_tensor_uint8_conversion()
    test_tensor_uint16_conversion()
    test_tensor_uint32_conversion()
    test_tensor_uint64_conversion()
    test_tensor_unsigned_round_trip()

    # Mixed signed/unsigned tests
    test_mixed_signed_unsigned()

    print("\n=== All Unsigned Integer Type Tests Passed! ===\n")
