"""Tests for signed integer type wrappers (Int8, Int16, Int32, Int64)."""

from testing import assert_equal, assert_true

from shared.core.types.integer import Int8, Int16, Int32, Int64
from shared.core.extensor import ExTensor, zeros
from shared.core.dtype import DType


fn test_int8_construction() raises:
    """Test Int8 construction from various values."""
    # Normal values
    var i1 = Int8(42)
    assert_equal(String(i1.value), "42")

    var i2 = Int8(-100)
    assert_equal(String(i2.value), "-100")

    var i3 = Int8(0)
    assert_equal(String(i3.value), "0")

    # Boundary values
    var i4 = Int8(-128)  # Min value
    assert_equal(String(i4.value), "-128")

    var i5 = Int8(127)  # Max value
    assert_equal(String(i5.value), "127")

    print("✓ test_int8_construction passed")


fn test_int8_clamping() raises:
    """Test Int8 value clamping at boundaries."""
    # Values exceeding max should clamp to 127
    var i1 = Int8(200)
    assert_equal(String(i1.value), "127")

    var i2 = Int8(1000)
    assert_equal(String(i2.value), "127")

    # Values below min should clamp to -128
    var i3 = Int8(-200)
    assert_equal(String(i3.value), "-128")

    var i4 = Int8(-1000)
    assert_equal(String(i4.value), "-128")

    print("✓ test_int8_clamping passed")


fn test_int8_float_conversion() raises:
    """Test Int8 conversion to/from float."""
    # Positive values
    var i1 = Int8.from_float32(42.7)
    assert_equal(String(i1.value), "42")  # Truncates

    var f1 = i1.to_float32()
    assert_equal(String(f1), "42.0")

    # Negative values
    var i2 = Int8.from_float32(-99.3)
    assert_equal(String(i2.value), "-99")

    # Zero
    var i3 = Int8.from_float32(0.0)
    assert_equal(String(i3.value), "0")

    # Clamping from float
    var i4 = Int8.from_float32(500.0)
    assert_equal(String(i4.value), "127")

    var i5 = Int8.from_float32(-500.0)
    assert_equal(String(i5.value), "-128")

    print("✓ test_int8_float_conversion passed")


fn test_int8_comparisons() raises:
    """Test Int8 comparison operators."""
    var i1 = Int8(10)
    var i2 = Int8(20)
    var i3 = Int8(10)

    # Equality
    assert_true(i1 == i3)
    assert_true(i1 != i2)

    # Less than / greater than
    assert_true(i1 < i2)
    assert_true(i2 > i1)
    assert_true(i1 <= i3)
    assert_true(i1 >= i3)

    print("✓ test_int8_comparisons passed")


fn test_int8_arithmetic() raises:
    """Test Int8 arithmetic operators."""
    var i1 = Int8(10)
    var i2 = Int8(5)

    # Addition
    var sum = i1 + i2
    assert_equal(String(sum.value), "15")

    # Subtraction
    var diff = i1 - i2
    assert_equal(String(diff.value), "5")

    # Multiplication
    var prod = i1 * i2
    assert_equal(String(prod.value), "50")

    # Division
    var quot = i1 / i2
    assert_equal(String(quot.value), "2")

    # Modulo
    var rem = i1 % Int8(3)
    assert_equal(String(rem.value), "1")

    # Negation
    var neg = -i1
    assert_equal(String(neg.value), "-10")

    print("✓ test_int8_arithmetic passed")


fn test_int8_string_representation() raises:
    """Test Int8 string representation."""
    var i1 = Int8(42)
    assert_equal(str(i1), "42")
    assert_equal(repr(i1), "Int8(42)")

    var i2 = Int8(-100)
    assert_equal(str(i2), "-100")

    print("✓ test_int8_string_representation passed")


fn test_int16_construction() raises:
    """Test Int16 construction and boundary values."""
    var i1 = Int16(1000)
    assert_equal(String(i1.value), "1000")

    var i2 = Int16(-32768)  # Min value
    assert_equal(String(i2.value), "-32768")

    var i3 = Int16(32767)  # Max value
    assert_equal(String(i3.value), "32767")

    # Clamping
    var i4 = Int16(50000)
    assert_equal(String(i4.value), "32767")

    var i5 = Int16(-50000)
    assert_equal(String(i5.value), "-32768")

    print("✓ test_int16_construction passed")


fn test_int16_conversions() raises:
    """Test Int16 conversions to/from other types."""
    # From Int8
    var i8 = Int8(100)
    var i16 = Int16.from_int8(i8.value)
    assert_equal(String(i16.value), "100")

    # To Int8 with clamping
    var large = Int16(1000)
    var back_to_i8 = large.to_int8()
    assert_equal(String(back_to_i8), "127")  # Clamped

    # To Int32
    var i32 = large.to_int32()
    assert_equal(String(i32), "1000")

    print("✓ test_int16_conversions passed")


fn test_int32_construction() raises:
    """Test Int32 construction and boundary values."""
    var i1 = Int32(100000)
    assert_equal(String(i1.value), "100000")

    var i2 = Int32(-2147483648)  # Min value
    assert_equal(String(i2.value), "-2147483648")

    var i3 = Int32(2147483647)  # Max value
    assert_equal(String(i3.value), "2147483647")

    print("✓ test_int32_construction passed")


fn test_int32_conversions() raises:
    """Test Int32 conversions."""
    # From Int8
    var i8 = Int8(42)
    var i32 = Int32.from_int8(i8.value)
    assert_equal(String(i32.value), "42")

    # From Int16
    var i16 = Int16(1000)
    var i32_2 = Int32.from_int16(i16.value)
    assert_equal(String(i32_2.value), "1000")

    # To Int16 with clamping
    var large = Int32(100000)
    var back_to_i16 = large.to_int16()
    assert_equal(String(back_to_i16), "32767")

    print("✓ test_int32_conversions passed")


fn test_int64_construction() raises:
    """Test Int64 construction."""
    var i1 = Int64(9223372036854775807)  # Max value
    assert_equal(String(i1.value), "9223372036854775807")

    var i2 = Int64(-9223372036854775808)  # Min value (close to)
    # Note: Can't directly construct the exact min due to literal limits

    var i3 = Int64(0)
    assert_equal(String(i3.value), "0")

    print("✓ test_int64_construction passed")


fn test_int64_conversions() raises:
    """Test Int64 conversions."""
    # From smaller types
    var i8 = Int8(42)
    var i64 = Int64.from_int8(i8.value)
    assert_equal(String(i64.value), "42")

    var i16 = Int16(1000)
    var i64_2 = Int64.from_int16(i16.value)
    assert_equal(String(i64_2.value), "1000")

    var i32 = Int32(1000000)
    var i64_3 = Int64.from_int32(i32.value)
    assert_equal(String(i64_3.value), "1000000")

    # To smaller types with clamping
    var large = Int64(100000)
    var back_to_i8 = large.to_int8()
    assert_equal(String(back_to_i8), "127")

    var back_to_i16 = large.to_int16()
    assert_equal(String(back_to_i16), "32767")

    var back_to_i32 = large.to_int32()
    assert_equal(String(back_to_i32), "100000")

    print("✓ test_int64_conversions passed")


fn test_tensor_int8_conversion() raises:
    """Test tensor conversion to/from Int8."""
    # Create a float32 tensor
    var t = zeros(List[Int](2, 3), DType.float32)

    # Set some values
    t._data.bitcast[Float32]()[0] = 10.5
    t._data.bitcast[Float32]()[1] = -20.3
    t._data.bitcast[Float32]()[2] = 127.9
    t._data.bitcast[Float32]()[3] = -128.1
    t._data.bitcast[Float32]()[4] = 0.0
    t._data.bitcast[Float32]()[5] = 42.7

    # Convert to int8
    var i8_tensor = t.to_int8()

    # Verify dtype
    assert_equal(i8_tensor._dtype, DType.int8)

    # Verify values (truncated and clamped)
    assert_equal(String(i8_tensor._data.bitcast[Int8]()[0]), "10")
    assert_equal(String(i8_tensor._data.bitcast[Int8]()[1]), "-20")
    assert_equal(String(i8_tensor._data.bitcast[Int8]()[2]), "127")  # Clamped
    assert_equal(String(i8_tensor._data.bitcast[Int8]()[3]), "-128")  # Clamped
    assert_equal(String(i8_tensor._data.bitcast[Int8]()[4]), "0")
    assert_equal(String(i8_tensor._data.bitcast[Int8]()[5]), "42")

    print("✓ test_tensor_int8_conversion passed")


fn test_tensor_int16_conversion() raises:
    """Test tensor conversion to/from Int16."""
    var t = zeros(List[Int](2, 2), DType.float32)

    t._data.bitcast[Float32]()[0] = 1000.5
    t._data.bitcast[Float32]()[1] = -1000.3
    t._data.bitcast[Float32]()[2] = 32767.9
    t._data.bitcast[Float32]()[3] = -32768.1

    var i16_tensor = t.to_int16()

    assert_equal(i16_tensor._dtype, DType.int16)
    assert_equal(String(i16_tensor._data.bitcast[Int16]()[0]), "1000")
    assert_equal(String(i16_tensor._data.bitcast[Int16]()[1]), "-1000")
    assert_equal(String(i16_tensor._data.bitcast[Int16]()[2]), "32767")  # Clamped
    assert_equal(String(i16_tensor._data.bitcast[Int16]()[3]), "-32768")  # Clamped

    print("✓ test_tensor_int16_conversion passed")


fn test_tensor_int32_conversion() raises:
    """Test tensor conversion to/from Int32."""
    var t = zeros(List[Int](2, 2), DType.float32)

    t._data.bitcast[Float32]()[0] = 100000.0
    t._data.bitcast[Float32]()[1] = -100000.0
    t._data.bitcast[Float32]()[2] = 0.0
    t._data.bitcast[Float32]()[3] = 42.5

    var i32_tensor = t.to_int32()

    assert_equal(i32_tensor._dtype, DType.int32)
    assert_equal(String(i32_tensor._data.bitcast[Int32]()[0]), "100000")
    assert_equal(String(i32_tensor._data.bitcast[Int32]()[1]), "-100000")
    assert_equal(String(i32_tensor._data.bitcast[Int32]()[2]), "0")
    assert_equal(String(i32_tensor._data.bitcast[Int32]()[3]), "42")

    print("✓ test_tensor_int32_conversion passed")


fn test_tensor_int64_conversion() raises:
    """Test tensor conversion to/from Int64."""
    var t = zeros(List[Int](2, 2), DType.float32)

    t._data.bitcast[Float32]()[0] = 1000000.0
    t._data.bitcast[Float32]()[1] = -1000000.0
    t._data.bitcast[Float32]()[2] = 0.0
    t._data.bitcast[Float32]()[3] = 12345.5

    var i64_tensor = t.to_int64()

    assert_equal(i64_tensor._dtype, DType.int64)
    assert_equal(String(i64_tensor._data.bitcast[Int64]()[0]), "1000000")
    assert_equal(String(i64_tensor._data.bitcast[Int64]()[1]), "-1000000")
    assert_equal(String(i64_tensor._data.bitcast[Int64]()[2]), "0")
    assert_equal(String(i64_tensor._data.bitcast[Int64]()[3]), "12345")

    print("✓ test_tensor_int64_conversion passed")


fn test_tensor_integer_round_trip() raises:
    """Test round-trip conversion from float -> int -> float."""
    var t = zeros(List[Int](3,), DType.float32)

    t._data.bitcast[Float32]()[0] = 10.0
    t._data.bitcast[Float32]()[1] = -20.0
    t._data.bitcast[Float32]()[2] = 0.0

    # Convert to int8 and back (implicitly through ExTensor operations)
    var i8_tensor = t.to_int8()

    # Manually convert back to float
    var t_back = zeros(List[Int](3,), DType.float32)
    t_back._data.bitcast[Float32]()[0] = Float32(i8_tensor._data.bitcast[Int8]()[0])
    t_back._data.bitcast[Float32]()[1] = Float32(i8_tensor._data.bitcast[Int8]()[1])
    t_back._data.bitcast[Float32]()[2] = Float32(i8_tensor._data.bitcast[Int8]()[2])

    # Verify values match
    assert_equal(String(t_back._data.bitcast[Float32]()[0]), "10.0")
    assert_equal(String(t_back._data.bitcast[Float32]()[1]), "-20.0")
    assert_equal(String(t_back._data.bitcast[Float32]()[2]), "0.0")

    print("✓ test_tensor_integer_round_trip passed")


fn main() raises:
    """Run all signed integer type tests."""
    print("\n=== Running Signed Integer Type Tests ===\n")

    # Int8 tests
    test_int8_construction()
    test_int8_clamping()
    test_int8_float_conversion()
    test_int8_comparisons()
    test_int8_arithmetic()
    test_int8_string_representation()

    # Int16 tests
    test_int16_construction()
    test_int16_conversions()

    # Int32 tests
    test_int32_construction()
    test_int32_conversions()

    # Int64 tests
    test_int64_construction()
    test_int64_conversions()

    # Tensor conversion tests
    test_tensor_int8_conversion()
    test_tensor_int16_conversion()
    test_tensor_int32_conversion()
    test_tensor_int64_conversion()
    test_tensor_integer_round_trip()

    print("\n=== All Signed Integer Type Tests Passed! ===\n")
