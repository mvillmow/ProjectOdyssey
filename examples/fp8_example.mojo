"""Example demonstrating FP8 data type usage.

This example shows:
1. Creating FP8 values from Float32
2. Converting FP8 back to Float32
3. Converting tensors to/from FP8 format
4. Memory savings with FP8 (8-bit vs 32-bit)
"""

from shared.core import ExTensor, zeros, FP8


fn main() raises:
    print("\n=== FP8 Data Type Example ===\n")

    # 1. Basic FP8 conversion
    print("1. Basic FP8 Value Conversion")
    print("-" * 40)

    var values: List[Float32] = [0.0, 1.0, -2.5, 10.0, -100.0, 200.0]
    print("Original values:")
    for i in range(len(values)):
        print("  ", values[i])

    print("\nFP8 encoded and decoded:")
    for i in range(len(values)):
        var fp8_val = FP8.from_float32(values[i])
        var decoded = fp8_val.to_float32()
        var error = abs(decoded - values[i])
        print("  ", values[i], " -> ", decoded, " (error: ", error, ")")

    # 2. Tensor conversion to FP8
    print("\n2. Tensor Conversion to FP8")
    print("-" * 40)

    var shape = List[Int]()
    shape[0] = 3
    shape[1] = 4

    # Create a Float32 tensor
    var tensor_f32 = zeros(shape, DType.float32)
    for i in range(12):
        tensor_f32._data.bitcast[Float32]()[i] = Float32(i) - 5.0

    print("Original Float32 tensor (3x4):")
    print("  Shape:", tensor_f32.shape()[0], "x", tensor_f32.shape()[1])
    print("  DType:", tensor_f32.dtype())
    print("  Size: 12 elements × 4 bytes = 48 bytes")
    print("  Values:")
    for i in range(12):
        print("    [", i, "]:", tensor_f32._data.bitcast[Float32]()[i])

    # Convert to FP8
    var tensor_fp8 = tensor_f32.to_fp8()

    print("\nFP8-encoded tensor (stored as uint8):")
    print("  Shape:", tensor_fp8.shape()[0], "x", tensor_fp8.shape()[1])
    print("  DType:", tensor_fp8.dtype())
    print("  Size: 12 elements × 1 byte = 12 bytes")
    print("  Memory savings: 75% (48 bytes -> 12 bytes)")

    # Convert back to Float32
    var tensor_restored = tensor_fp8.from_fp8()

    print("\nRestored Float32 tensor:")
    print(
        "  Shape:", tensor_restored.shape()[0], "x", tensor_restored.shape()[1]
    )
    print("  DType:", tensor_restored.dtype())
    print("  Values (with FP8 precision loss):")
    for i in range(12):
        var original = tensor_f32._data.bitcast[Float32]()[i]
        var restored = tensor_restored._data.bitcast[Float32]()[i]
        var error = abs(restored - original)
        print(
            "    [", i, "]:", original, " ->", restored, " (error:", error, ")"
        )

    # 3. Memory efficiency demonstration
    print("\n3. Memory Efficiency")
    print("-" * 40)

    var large_shape = List[Int]()
    large_shape[0] = 1000
    large_shape[1] = 1000

    var large_tensor = zeros(large_shape, DType.float32)
    var large_fp8 = large_tensor.to_fp8()

    var f32_bytes = 1000 * 1000 * 4  # 4 bytes per float32
    var fp8_bytes = 1000 * 1000 * 1  # 1 byte per fp8

    print("Large tensor (1000x1000):")
    print(
        "  Float32 size:", f32_bytes, "bytes (", f32_bytes / 1024 / 1024, "MB)"
    )
    print("  FP8 size:", fp8_bytes, "bytes (", fp8_bytes / 1024 / 1024, "MB)")
    print("  Memory savings:", 100 - (fp8_bytes * 100 // f32_bytes), "%")

    print("\n=== FP8 Example Complete ===\n")
    print("Key Takeaways:")
    print("  • FP8 reduces memory by 75% (vs Float32)")
    print("  • FP8 range: ~±240 with reduced precision")
    print("  • Use for memory-constrained training/inference")
    print("  • Trade-off: 8× memory savings for some precision loss")
    print("")
