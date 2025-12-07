"""Example demonstrating the usage of integer type wrappers.

This example shows how to use the Int8, Int16, Int32, Int64, UInt8, UInt16,
UInt32, and UInt64 type wrappers, including:
- Creating integer values
- Converting between types
- Performing arithmetic operations
- Converting tensors to/from integer types
- Handling clamping and overflow
"""

from shared.core.types.integer import Int8, Int16, Int32, Int64
from shared.core.types.unsigned import UInt8, UInt16, UInt32, UInt64
from shared.core.extensor import ExTensor, zeros
from shared.core.dtype import DType


fn example_basic_signed_integers() raises:
    """Demonstrate basic signed integer operations."""
    print("\n=== Basic Signed Integer Operations ===\n")

    # Create Int8 values
    var i8_1 = Int8(42)
    var i8_2 = Int8(-100)
    print("Int8 values:", String(i8_1), String(i8_2))

    # Arithmetic operations
    var sum = i8_1 + Int8(10)
    var diff = i8_1 - Int8(5)
    var prod = i8_1 * Int8(2)
    print(
        "Int8 arithmetic: 42+10=",
        String(sum),
        " 42-5=",
        String(diff),
        " 42*2=",
        String(prod),
    )

    # Comparisons
    if i8_1 > i8_2:
        print("42 > -100: True")

    # Clamping demonstration
    var clamped_high = Int8(200)  # Exceeds max (127)
    var clamped_low = Int8(-200)  # Below min (-128)
    print(
        "Clamping: 200 ->",
        String(clamped_high),
        " -200 ->",
        String(clamped_low),
    )


fn example_basic_unsigned_integers() raises:
    """Demonstrate basic unsigned integer operations."""
    print("\n=== Basic Unsigned Integer Operations ===\n")

    # Create UInt8 values
    var u8_1 = UInt8(255)
    var u8_2 = UInt8(100)
    print("UInt8 values:", String(u8_1), String(u8_2))

    # Arithmetic operations
    var sum = u8_2 + UInt8(50)
    var diff = u8_1 - u8_2
    print("UInt8 arithmetic: 100+50=", String(sum), " 255-100=", String(diff))

    # Clamping demonstration
    var clamped_high = UInt8(300)  # Exceeds max (255)
    var clamped_low = UInt8(-50)  # Below min (0)
    print(
        "Clamping: 300 ->", String(clamped_high), " -50 ->", String(clamped_low)
    )


fn example_type_conversions() raises:
    """Demonstrate conversions between integer types."""
    print("\n=== Type Conversions ===\n")

    # Int8 to larger types (lossless)
    var i8 = Int8(42)
    var i16 = i8.to_int16()
    var i32 = i8.to_int32()
    var i64 = i8.to_int64()
    print(
        "Int8(42) -> Int16:",
        String(i16),
        " Int32:",
        String(i32),
        " Int64:",
        String(i64),
    )

    # Larger types to Int8 (with clamping)
    var large_i32 = Int32(1000)
    var back_to_i8 = large_i32.to_int8()
    print("Int32(1000) -> Int8:", String(back_to_i8), " (clamped to 127)")

    # Float conversions
    var from_float = Int8.from_float32(42.7)
    var to_float = from_float.to_float32()
    print(
        "Float32(42.7) -> Int8:",
        String(from_float),
        " -> Float32:",
        String(to_float),
    )


fn example_tensor_conversions() raises:
    """Demonstrate tensor conversions with integer types."""
    print("\n=== Tensor Conversions ===\n")

    # Create a float32 tensor
    var t = zeros(List[Int](2, 3), DType.float32)

    # Set some values
    t._data.bitcast[Float32]()[0] = 10.5
    t._data.bitcast[Float32]()[1] = -20.3
    t._data.bitcast[Float32]()[2] = 200.0
    t._data.bitcast[Float32]()[3] = -128.5
    t._data.bitcast[Float32]()[4] = 0.0
    t._data.bitcast[Float32]()[5] = 42.9

    print("Original float32 tensor values:")
    print("  [10.5, -20.3, 200.0, -128.5, 0.0, 42.9]")

    # Convert to Int8
    var i8_tensor = t.to_int8()
    print("\nAfter conversion to Int8 (truncated and clamped):")
    print(
        " ",
        String(i8_tensor._data.bitcast[Int8]()[0]),
        String(i8_tensor._data.bitcast[Int8]()[1]),
        String(i8_tensor._data.bitcast[Int8]()[2]),
        String(i8_tensor._data.bitcast[Int8]()[3]),
        String(i8_tensor._data.bitcast[Int8]()[4]),
        String(i8_tensor._data.bitcast[Int8]()[5]),
    )

    # Convert to UInt8
    var u8_tensor = t.to_uint8()
    print("\nAfter conversion to UInt8 (truncated and clamped):")
    print(
        " ",
        String(u8_tensor._data.bitcast[UInt8]()[0]),
        String(u8_tensor._data.bitcast[UInt8]()[1]),
        String(u8_tensor._data.bitcast[UInt8]()[2]),
        String(u8_tensor._data.bitcast[UInt8]()[3]),
        String(u8_tensor._data.bitcast[UInt8]()[4]),
        String(u8_tensor._data.bitcast[UInt8]()[5]),
    )


fn example_16bit_integers() raises:
    """Demonstrate 16-bit integer operations."""
    print("\n=== 16-bit Integer Operations ===\n")

    # Int16
    var i16_1 = Int16(1000)
    var i16_2 = Int16(-500)
    print("Int16 values:", String(i16_1), String(i16_2))

    # UInt16
    var u16_1 = UInt16(60000)
    var u16_2 = UInt16(5000)
    print("UInt16 values:", String(u16_1), String(u16_2))

    # Arithmetic
    var i16_sum = i16_1 + i16_2
    var u16_sum = u16_1 + u16_2
    print("Int16 sum:", String(i16_sum), " UInt16 sum:", String(u16_sum))

    # Clamping boundaries
    var i16_max = Int16(40000)  # Exceeds 32767
    var u16_max = UInt16(70000)  # Exceeds 65535
    print("Int16 clamping: 40000 ->", String(i16_max))
    print("UInt16 clamping: 70000 ->", String(u16_max))


fn example_32bit_integers() raises:
    """Demonstrate 32-bit integer operations."""
    print("\n=== 32-bit Integer Operations ===\n")

    # Int32
    var i32 = Int32(1000000)
    print("Int32 value:", String(i32))

    # UInt32
    var u32 = UInt32(4000000000)
    print("UInt32 value:", String(u32))

    # Conversions
    var i32_from_i8 = Int32.from_int8(Int8(42).value)
    var u32_from_u16 = UInt32.from_uint16(UInt16(1000).value)
    print("Int32 from Int8(42):", String(i32_from_i8))
    print("UInt32 from UInt16(1000):", String(u32_from_u16))


fn example_64bit_integers() raises:
    """Demonstrate 64-bit integer operations."""
    print("\n=== 64-bit Integer Operations ===\n")

    # Int64
    var i64 = Int64(9999999999)
    print("Int64 value:", String(i64))

    # UInt64
    var u64 = UInt64(18446744073709551615)
    print("UInt64 large value:", String(u64))

    # Lossless conversions from smaller types
    var i64_from_i32 = Int64.from_int32(Int32(1000000).value)
    var u64_from_u32 = UInt64.from_uint32(UInt32(4000000000).value)
    print("Int64 from Int32:", String(i64_from_i32))
    print("UInt64 from UInt32:", String(u64_from_u32))


fn example_practical_use_case() raises:
    """Demonstrate a practical use case: quantization."""
    print("\n=== Practical Use Case: Simple Quantization ===\n")

    print(
        "Scenario: Quantizing a float32 tensor to UInt8 for memory efficiency"
    )

    # Create a float tensor with normalized values (0.0 to 1.0)
    var normalized = zeros(
        List[Int](
            5,
        ),
        DType.float32,
    )
    normalized._data.bitcast[Float32]()[0] = 0.0
    normalized._data.bitcast[Float32]()[1] = 0.25
    normalized._data.bitcast[Float32]()[2] = 0.5
    normalized._data.bitcast[Float32]()[3] = 0.75
    normalized._data.bitcast[Float32]()[4] = 1.0

    print("\nOriginal normalized values (0.0 to 1.0):")
    print("  [0.0, 0.25, 0.5, 0.75, 1.0]")

    # Scale to 0-255 range
    var scaled = zeros(
        List[Int](
            5,
        ),
        DType.float32,
    )
    for i in range(5):
        scaled._data.bitcast[Float32]()[i] = (
            normalized._data.bitcast[Float32]()[i] * 255.0
        )

    print("\nScaled to 0-255 range:")
    print("  [0.0, 63.75, 127.5, 191.25, 255.0]")

    # Convert to UInt8 (quantized)
    var quantized = scaled.to_uint8()

    print("\nQuantized to UInt8 (truncated):")
    print(
        " ",
        String(quantized._data.bitcast[UInt8]()[0]),
        String(quantized._data.bitcast[UInt8]()[1]),
        String(quantized._data.bitcast[UInt8]()[2]),
        String(quantized._data.bitcast[UInt8]()[3]),
        String(quantized._data.bitcast[UInt8]()[4]),
    )

    # Dequantize back to float
    var dequantized = zeros(
        List[Int](
            5,
        ),
        DType.float32,
    )
    for i in range(5):
        dequantized._data.bitcast[Float32]()[i] = (
            Float32(quantized._data.bitcast[UInt8]()[i]) / 255.0
        )

    print("\nDequantized back to 0.0-1.0 range:")
    print(
        " ",
        String(dequantized._data.bitcast[Float32]()[0]),
        String(dequantized._data.bitcast[Float32]()[1]),
        String(dequantized._data.bitcast[Float32]()[2]),
        String(dequantized._data.bitcast[Float32]()[3]),
        String(dequantized._data.bitcast[Float32]()[4]),
    )

    print(
        "\nNote: Some precision is lost due to quantization (e.g., 0.25 ->"
        " 0.24901961)"
    )


fn main() raises:
    """Run all integer type examples."""
    print("=" * 60)
    print("Integer Type Wrappers - Usage Examples")
    print("=" * 60)

    example_basic_signed_integers()
    example_basic_unsigned_integers()
    example_type_conversions()
    example_16bit_integers()
    example_32bit_integers()
    example_64bit_integers()
    example_tensor_conversions()
    example_practical_use_case()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60 + "\n")
