"""Example demonstrating BF8 data type usage.

This example shows:
1. Creating BF8 values from Float32
2. Converting BF8 back to Float32
3. Converting tensors to/from BF8 format
4. Memory savings with BF8 (8-bit vs 32-bit)
5. Comparing BF8 (E5M2) vs FP8 (E4M3) characteristics
"""

from shared.core import ExTensor, zeros, BF8, FP8
from collections.vector import DynamicVector


fn main() raises:
    print("\n=== BF8 Data Type Example ===\n")

    # 1. Basic BF8 conversion
    print("1. Basic BF8 Value Conversion")
    print("-" * 40)

    var values = List[Float32](0.0, 1.0, -2.5, 10.0, -100.0, 1000.0, 10000.0)
    print("Original values:")
    for i in range(len(values)):
        print("  ", values[i])

    print("\nBF8 encoded and decoded:")
    for i in range(len(values)):
        var bf8_val = BF8.from_float32(values[i])
        var decoded = bf8_val.to_float32()
        var error = abs(decoded - values[i])
        print("  ", values[i], " -> ", decoded, " (error: ", error, ")")

    # 2. Tensor conversion to BF8
    print("\n2. Tensor Conversion to BF8")
    print("-" * 40)

    var shape = DynamicVector[Int](2)
    shape[0] = 3
    shape[1] = 4

    # Create a Float32 tensor
    var tensor_f32 = zeros(shape, DType.float32)
    for i in range(12):
        tensor_f32._data.bitcast[Float32]()[i] = Float32(i) * 10.0 - 50.0

    print("Original Float32 tensor (3x4):")
    print("  Shape:", tensor_f32.shape()[0], "x", tensor_f32.shape()[1])
    print("  DType:", tensor_f32.dtype())
    print("  Size: 12 elements × 4 bytes = 48 bytes")
    print("  Values:")
    for i in range(12):
        print("    [", i, "]:", tensor_f32._data.bitcast[Float32]()[i])

    # Convert to BF8
    var tensor_bf8 = tensor_f32.to_bf8()

    print("\nBF8-encoded tensor (stored as uint8):")
    print("  Shape:", tensor_bf8.shape()[0], "x", tensor_bf8.shape()[1])
    print("  DType:", tensor_bf8.dtype())
    print("  Size: 12 elements × 1 byte = 12 bytes")
    print("  Memory savings: 75% (48 bytes -> 12 bytes)")

    # Convert back to Float32
    var tensor_restored = tensor_bf8.from_bf8()

    print("\nRestored Float32 tensor:")
    print("  Shape:", tensor_restored.shape()[0], "x", tensor_restored.shape()[1])
    print("  DType:", tensor_restored.dtype())
    print("  Values (with BF8 precision loss):")
    for i in range(12):
        var original = tensor_f32._data.bitcast[Float32]()[i]
        var restored = tensor_restored._data.bitcast[Float32]()[i]
        var error = abs(restored - original)
        print("    [", i, "]:", original, " ->", restored, " (error:", error, ")")

    # 3. Memory efficiency demonstration
    print("\n3. Memory Efficiency")
    print("-" * 40)

    var large_shape = DynamicVector[Int](2)
    large_shape[0] = 1000
    large_shape[1] = 1000

    var large_tensor = zeros(large_shape, DType.float32)
    var large_bf8 = large_tensor.to_bf8()

    var f32_bytes = 1000 * 1000 * 4  # 4 bytes per float32
    var bf8_bytes = 1000 * 1000 * 1  # 1 byte per bf8

    print("Large tensor (1000x1000):")
    print("  Float32 size:", f32_bytes, "bytes (", f32_bytes / 1024 / 1024, "MB)")
    print("  BF8 size:", bf8_bytes, "bytes (", bf8_bytes / 1024 / 1024, "MB)")
    print("  Memory savings:", 100 - (bf8_bytes * 100 // f32_bytes), "%")

    # 4. BF8 vs FP8 comparison
    print("\n4. BF8 (E5M2) vs FP8 (E4M3) Comparison")
    print("-" * 40)

    # Test values that show the difference
    var test_values = List[Float32](1.0, 100.0, 1000.0, 10000.0)

    print("Value    BF8 (E5M2)    FP8 (E4M3)    BF8 Error    FP8 Error")
    print("-" * 70)

    for i in range(len(test_values)):
        var val = test_values[i]

        # BF8 conversion
        var bf8_val = BF8.from_float32(val)
        var bf8_decoded = bf8_val.to_float32()
        var bf8_error = abs(bf8_decoded - val)

        # FP8 conversion
        var fp8_val = FP8.from_float32(val)
        var fp8_decoded = fp8_val.to_float32()
        var fp8_error = abs(fp8_decoded - val)

        print(
            val,
            "    ",
            bf8_decoded,
            "    ",
            fp8_decoded,
            "    ",
            bf8_error,
            "    ",
            fp8_error
        )

    print("\n=== BF8 Example Complete ===\n")
    print("Key Takeaways:")
    print("  • BF8 reduces memory by 75% (vs Float32)")
    print("  • BF8 range: ~±57,344 (much larger than FP8's ~±240)")
    print("  • BF8 precision: 2 mantissa bits (less than FP8's 3 bits)")
    print("  • Use BF8 when range is more important than precision")
    print("  • Use FP8 when precision is more important than range")
    print("  • Trade-off: 8× memory savings for precision loss")
    print("")
