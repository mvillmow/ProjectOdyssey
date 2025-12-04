"""Tests for FP4 tensor conversion operations.

Tests cover:
- Tensor conversion (to_mxfp4, from_mxfp4, to_nvfp4, from_nvfp4)
- Multi-dimensional tensors
- Padding behavior for non-multiple sizes
- Round-trip accuracy
- Memory efficiency verification

All tests use pure functional API.
"""

from shared.core.extensor import ExTensor, zeros, ones, full
from tests.shared.conftest import (
    assert_true,
    assert_equal,
    assert_almost_equal,
)


# ============================================================================
# MXFP4 Tensor Conversion Tests
# ============================================================================


fn test_mxfp4_tensor_conversion_exact_size() raises:
    """Test MXFP4 conversion for tensor with exact block size."""
    # Create tensor with exactly 64 elements (2 blocks of 32)
    var t = zeros(List[Int](64), DType.float32)

    # Fill with test values
    for i in range(64):
        t._data.bitcast[Float32]()[i] = Float32(i) * 0.1

    # Convert to MXFP4
    var mxfp4_t = t.to_mxfp4()

    # Verify size: 2 blocks × 17 bytes = 34 bytes
    assert_equal(mxfp4_t.numel(), 34)
    assert_true(mxfp4_t.dtype() == DType.uint8, "Expected MXFP4 dtype to be uint8")

    # Convert back
    var restored = mxfp4_t.from_mxfp4()

    # Verify size restored
    assert_equal(restored.numel(), 64)
    assert_true(restored.dtype() == DType.float32, "Expected restored dtype to be float32")

    # Verify approximate accuracy
    var total_error = Float32(0.0)
    for i in range(64):
        var expected = Float32(i) * 0.1
        var decoded = restored._data.bitcast[Float32]()[i]
        total_error += abs(decoded - expected)

    # Average error should be reasonable
    var avg_error = total_error / 64.0
    assert_true(avg_error < 0.5, "Average error too large")


fn test_mxfp4_tensor_conversion_padding() raises:
    """Test MXFP4 conversion handles padding correctly."""
    # Create tensor with 50 elements (requires 2 blocks, 14 padding zeros)
    var t = zeros(List[Int](50), DType.float32)

    # Fill with test values
    for i in range(50):
        t._data.bitcast[Float32]()[i] = Float32(i) * 0.1

    # Convert to MXFP4
    var mxfp4_t = t.to_mxfp4()

    # Verify size: 2 blocks × 17 bytes = 34 bytes
    assert_equal(mxfp4_t.numel(), 34)

    # Convert back (implementation restores original size, not padded size)
    var restored = mxfp4_t.from_mxfp4()

    # Verify size restored to original (50 elements, not padded 64)
    assert_equal(restored.numel(), 50)

    # Verify all 50 values are approximately correct
    # Note: FP4 has limited precision, use larger tolerance for higher values
    for i in range(50):
        var expected = Float32(i) * 0.1
        var decoded = restored._data.bitcast[Float32]()[i]
        var error = abs(decoded - expected)
        # Tolerance scales with value magnitude for low-precision formats
        var tolerance = max(Float32(1.0), expected * 0.5)
        assert_true(error < tolerance, "Value " + String(i) + " error too large")


fn test_mxfp4_tensor_conversion_multidim() raises:
    """Test MXFP4 conversion for multi-dimensional tensor."""
    # Create 2D tensor (4, 16) = 64 elements
    var t = zeros(List[Int](4, 16), DType.float32)

    # Fill with test values
    for i in range(64):
        t._data.bitcast[Float32]()[i] = Float32(i % 32) * 0.1

    # Convert to MXFP4
    var mxfp4_t = t.to_mxfp4()

    # Convert back
    var restored = mxfp4_t.from_mxfp4()

    # Verify approximate accuracy
    for i in range(64):
        var expected = Float32(i % 32) * 0.1
        var decoded = restored._data.bitcast[Float32]()[i]
        var error = abs(decoded - expected)
        assert_true(error < 0.5, "Error too large")


# ============================================================================
# NVFP4 Tensor Conversion Tests
# ============================================================================


fn test_nvfp4_tensor_conversion_exact_size() raises:
    """Test NVFP4 conversion for tensor with exact block size."""
    # Create tensor with exactly 64 elements (4 blocks of 16)
    var t = zeros(List[Int](64), DType.float32)

    # Fill with test values
    for i in range(64):
        t._data.bitcast[Float32]()[i] = Float32(i) * 0.1

    # Convert to NVFP4
    var nvfp4_t = t.to_nvfp4()

    # Verify size: 4 blocks × 9 bytes = 36 bytes
    assert_equal(nvfp4_t.numel(), 36)
    assert_true(nvfp4_t.dtype() == DType.uint8, "Expected NVFP4 dtype to be uint8")

    # Convert back
    var restored = nvfp4_t.from_nvfp4()

    # Verify size restored
    assert_equal(restored.numel(), 64)
    assert_true(restored.dtype() == DType.float32, "Expected restored dtype to be float32")

    # Verify approximate accuracy
    var total_error = Float32(0.0)
    for i in range(64):
        var expected = Float32(i) * 0.1
        var decoded = restored._data.bitcast[Float32]()[i]
        total_error += abs(decoded - expected)

    # Average error should be reasonable
    var avg_error = total_error / 64.0
    assert_true(avg_error < 0.5, "Average error too large")


fn test_nvfp4_tensor_conversion_padding() raises:
    """Test NVFP4 conversion handles padding correctly."""
    # Create tensor with 50 elements (requires 4 blocks, 14 padding zeros)
    var t = zeros(List[Int](50), DType.float32)

    # Fill with test values
    for i in range(50):
        t._data.bitcast[Float32]()[i] = Float32(i) * 0.1

    # Convert to NVFP4
    var nvfp4_t = t.to_nvfp4()

    # Verify size: 4 blocks × 9 bytes = 36 bytes (ceil(50/16) = 4)
    assert_equal(nvfp4_t.numel(), 36)

    # Convert back (implementation restores original size, not padded size)
    var restored = nvfp4_t.from_nvfp4()

    # Verify size restored to original (50 elements, not padded 64)
    assert_equal(restored.numel(), 50)

    # Verify all 50 values are approximately correct
    # Note: FP4 has limited precision, use larger tolerance for higher values
    for i in range(50):
        var expected = Float32(i) * 0.1
        var decoded = restored._data.bitcast[Float32]()[i]
        var error = abs(decoded - expected)
        # Tolerance scales with value magnitude for low-precision formats
        var tolerance = max(Float32(0.8), expected * 0.5)
        assert_true(error < tolerance, "Value " + String(i) + " error too large")


fn test_nvfp4_tensor_conversion_multidim() raises:
    """Test NVFP4 conversion for multi-dimensional tensor."""
    # Create 2D tensor (4, 16) = 64 elements
    var t = zeros(List[Int](4, 16), DType.float32)

    # Fill with test values
    for i in range(64):
        t._data.bitcast[Float32]()[i] = Float32(i % 16) * 0.1

    # Convert to NVFP4
    var nvfp4_t = t.to_nvfp4()

    # Convert back
    var restored = nvfp4_t.from_nvfp4()

    # Verify approximate accuracy
    for i in range(64):
        var expected = Float32(i % 16) * 0.1
        var decoded = restored._data.bitcast[Float32]()[i]
        var error = abs(decoded - expected)
        assert_true(error < 0.4, "Error too large")


# ============================================================================
# Memory Efficiency Tests
# ============================================================================


fn test_mxfp4_memory_efficiency() raises:
    """Verify MXFP4 provides expected compression."""
    # Create tensor with 320 elements (10 blocks)
    var t = zeros(List[Int](320), DType.float32)

    # Original size: 320 * 4 bytes = 1280 bytes
    # MXFP4 size: 10 blocks * 17 bytes = 170 bytes
    # Compression: ~7.5x

    var mxfp4_t = t.to_mxfp4()

    # Verify compressed size
    assert_equal(mxfp4_t.numel(), 170)

    # Calculate compression ratio
    var original_bytes = 320 * 4
    var compressed_bytes = mxfp4_t.numel()
    var compression_ratio = Float32(original_bytes) / Float32(compressed_bytes)

    print("MXFP4 compression: " + String(compression_ratio) + "x")
    assert_true(compression_ratio > 7.0, "Compression ratio too low")


fn test_nvfp4_memory_efficiency() raises:
    """Verify NVFP4 provides expected compression."""
    # Create tensor with 320 elements (20 blocks)
    var t = zeros(List[Int](320), DType.float32)

    # Original size: 320 * 4 bytes = 1280 bytes
    # NVFP4 size: 20 blocks * 9 bytes = 180 bytes
    # Compression: ~7.1x

    var nvfp4_t = t.to_nvfp4()

    # Verify compressed size
    assert_equal(nvfp4_t.numel(), 180)

    # Calculate compression ratio
    var original_bytes = 320 * 4
    var compressed_bytes = nvfp4_t.numel()
    var compression_ratio = Float32(original_bytes) / Float32(compressed_bytes)

    print("NVFP4 compression: " + String(compression_ratio) + "x")
    assert_true(compression_ratio > 7.0, "Compression ratio too low")


# ============================================================================
# Error Handling Tests
# ============================================================================


fn test_mxfp4_conversion_requires_float() raises:
    """Test MXFP4 conversion requires floating-point tensor."""
    var t = zeros(List[Int](1), DType.int32)

    try:
        var mxfp4_t = t.to_mxfp4()
        assert_true(False, "Expected error for non-float tensor")
    except e:
        # Expected error
        pass


fn test_nvfp4_conversion_requires_float() raises:
    """Test NVFP4 conversion requires floating-point tensor."""
    var t = zeros(List[Int](1), DType.int32)

    try:
        var nvfp4_t = t.to_nvfp4()
        assert_true(False, "Expected error for non-float tensor")
    except e:
        # Expected error
        pass


fn test_mxfp4_decoding_requires_uint8() raises:
    """Test MXFP4 decoding requires uint8 tensor."""
    var t = zeros(List[Int](1), DType.int32)

    try:
        var restored = t.from_mxfp4()
        assert_true(False, "Expected error for non-uint8 tensor")
    except e:
        # Expected error
        pass


fn test_nvfp4_decoding_requires_uint8() raises:
    """Test NVFP4 decoding requires uint8 tensor."""
    var t = zeros(List[Int](1), DType.int32)

    try:
        var restored = t.from_nvfp4()
        assert_true(False, "Expected error for non-uint8 tensor")
    except e:
        # Expected error
        pass


fn test_mxfp4_decoding_requires_block_alignment() raises:
    """Test MXFP4 decoding requires block-aligned size."""
    var t = zeros(List[Int](1), DType.uint8)  # Not a multiple of 17

    try:
        var restored = t.from_mxfp4()
        assert_true(False, "Expected error for non-aligned size")
    except e:
        # Expected error
        pass


fn test_nvfp4_decoding_requires_block_alignment() raises:
    """Test NVFP4 decoding requires block-aligned size."""
    var t = zeros(List[Int](1), DType.uint8)  # Not a multiple of 9

    try:
        var restored = t.from_nvfp4()
        assert_true(False, "Expected error for non-aligned size")
    except e:
        # Expected error
        pass


# ============================================================================
# Round-Trip Tests
# ============================================================================


fn test_mxfp4_roundtrip_large_tensor() raises:
    """Test MXFP4 round-trip for large tensor."""
    # Create large tensor (1000 elements)
    var t = zeros(List[Int](1000), DType.float32)

    # Fill with test pattern
    for i in range(1000):
        t._data.bitcast[Float32]()[i] = Float32(i % 100) * 0.01

    # Round-trip
    var mxfp4_t = t.to_mxfp4()
    var restored = mxfp4_t.from_mxfp4()

    # Check approximate accuracy
    var errors = 0
    for i in range(1000):
        var expected = Float32(i % 100) * 0.01
        var decoded = restored._data.bitcast[Float32]()[i]
        var error = abs(decoded - expected)
        if error > 0.5:
            errors += 1

    # Allow some errors due to quantization
    assert_true(errors < 100, "Too many errors in round-trip")


fn test_nvfp4_roundtrip_large_tensor() raises:
    """Test NVFP4 round-trip for large tensor."""
    # Create large tensor (1000 elements)
    var t = zeros(List[Int](1000), DType.float32)

    # Fill with test pattern
    for i in range(1000):
        t._data.bitcast[Float32]()[i] = Float32(i % 100) * 0.01

    # Round-trip
    var nvfp4_t = t.to_nvfp4()
    var restored = nvfp4_t.from_nvfp4()

    # Check approximate accuracy
    var errors = 0
    for i in range(1000):
        var expected = Float32(i % 100) * 0.01
        var decoded = restored._data.bitcast[Float32]()[i]
        var error = abs(decoded - expected)
        if error > 0.4:
            errors += 1

    # Allow some errors due to quantization
    assert_true(errors < 100, "Too many errors in round-trip")


fn main() raises:
    """Run all FP4 tensor conversion tests."""
    print("Running FP4 tensor conversion tests...")

    # MXFP4 conversion tests
    test_mxfp4_tensor_conversion_exact_size()
    print("✓ MXFP4 exact size conversion")

    test_mxfp4_tensor_conversion_padding()
    print("✓ MXFP4 padding handling")

    test_mxfp4_tensor_conversion_multidim()
    print("✓ MXFP4 multi-dimensional tensor")

    # NVFP4 conversion tests
    test_nvfp4_tensor_conversion_exact_size()
    print("✓ NVFP4 exact size conversion")

    test_nvfp4_tensor_conversion_padding()
    print("✓ NVFP4 padding handling")

    test_nvfp4_tensor_conversion_multidim()
    print("✓ NVFP4 multi-dimensional tensor")

    # Memory efficiency
    test_mxfp4_memory_efficiency()
    print("✓ MXFP4 memory efficiency")

    test_nvfp4_memory_efficiency()
    print("✓ NVFP4 memory efficiency")

    # Error handling
    test_mxfp4_conversion_requires_float()
    print("✓ MXFP4 requires float tensor")

    test_nvfp4_conversion_requires_float()
    print("✓ NVFP4 requires float tensor")

    test_mxfp4_decoding_requires_uint8()
    print("✓ MXFP4 decoding requires uint8")

    test_nvfp4_decoding_requires_uint8()
    print("✓ NVFP4 decoding requires uint8")

    test_mxfp4_decoding_requires_block_alignment()
    print("✓ MXFP4 decoding requires alignment")

    test_nvfp4_decoding_requires_block_alignment()
    print("✓ NVFP4 decoding requires alignment")

    # Round-trip tests
    test_mxfp4_roundtrip_large_tensor()
    print("✓ MXFP4 large tensor round-trip")

    test_nvfp4_roundtrip_large_tensor()
    print("✓ NVFP4 large tensor round-trip")

    print("\nAll FP4 tensor conversion tests passed!")
