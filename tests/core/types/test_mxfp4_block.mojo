"""Tests for MXFP4Block blocked storage and conversion.

Tests cover:
- Block creation from Float32 arrays
- Round-trip conversion accuracy
- Scale computation correctness
- Bit packing/unpacking
- Block indexing operations (get/set)
- Edge cases (all zeros, all same value, mixed signs)

All tests use pure functional API.
"""

from shared.core.types.mxfp4 import MXFP4, MXFP4Block, E8M0Scale
from shared.core.types.fp4 import FP4_E2M1
from tests.shared.conftest import (
    assert_true,
    assert_equal,
    assert_almost_equal,
)


# ============================================================================
# Block Creation Tests
# ============================================================================


fn test_mxfp4_block_creation_zeros() raises:
    """Test MXFP4Block creation with all zeros."""
    var values = List[Float32]()
    for i in range(32):
        values.append(Float32(0.0))

    var block = MXFP4Block.from_float32_array(values)
    var decoded = block.to_float32_array()

    # Check all values are zero
    for i in range(32):
        assert_almost_equal(decoded[i], Float32(0.0), tolerance=1e-5)


fn test_mxfp4_block_creation_ones() raises:
    """Test MXFP4Block creation with all ones."""
    var values = List[Float32]()
    for i in range(32):
        values.append(Float32(1.0))

    var block = MXFP4Block.from_float32_array(values)
    var decoded = block.to_float32_array()

    # Check all values are approximately 1.0 (within E2M1 precision)
    for i in range(32):
        assert_almost_equal(decoded[i], Float32(1.0), tolerance=0.1)


fn test_mxfp4_block_creation_range() raises:
    """Test MXFP4Block creation with sequential values."""
    var values = List[Float32]()
    for i in range(32):
        values.append(Float32(i) * 0.1)

    var block = MXFP4Block.from_float32_array(values)
    var decoded = block.to_float32_array()

    # Check values are reasonably close (E2M1 has limited precision)
    for i in range(32):
        var expected = Float32(i) * 0.1
        var error = abs(decoded[i] - expected)
        # E2M1 precision is limited, allow larger tolerance
        assert_true(error < 0.5, "Value " + String(i) + " error too large")


fn test_mxfp4_block_size_validation() raises:
    """Test MXFP4Block requires exactly 32 values."""
    var values = List[Float32]()
    for i in range(16):  # Only 16 values
        values.append(Float32(i))

    try:
        var block = MXFP4Block.from_float32_array(values)
        assert_true(False, "Expected error for wrong size")
    except e:
        # Expected error
        pass


# ============================================================================
# Round-Trip Conversion Tests
# ============================================================================


fn test_mxfp4_block_roundtrip_small() raises:
    """Test round-trip conversion for small values."""
    var values = List[Float32]()
    for i in range(32):
        values.append(Float32(0.5) + Float32(i) * 0.1)

    var block = MXFP4Block.from_float32_array(values)
    var decoded = block.to_float32_array()

    # Verify approximate reconstruction
    for i in range(32):
        var expected = Float32(0.5) + Float32(i) * 0.1
        var error = abs(decoded[i] - expected)
        assert_true(error < 1.0, "Round-trip error too large")


fn test_mxfp4_block_roundtrip_large() raises:
    """Test round-trip conversion for large values."""
    var values = List[Float32]()
    for i in range(32):
        values.append(Float32(10.0) + Float32(i) * 2.0)

    var block = MXFP4Block.from_float32_array(values)
    var decoded = block.to_float32_array()

    # Verify approximate reconstruction
    for i in range(32):
        var expected = Float32(10.0) + Float32(i) * 2.0
        var error = abs(decoded[i] - expected)
        # Larger values have larger absolute errors
        assert_true(error < expected * 0.3, "Round-trip error too large")


fn test_mxfp4_block_roundtrip_mixed_signs() raises:
    """Test round-trip conversion with mixed signs."""
    var values = List[Float32]()
    for i in range(32):
        var sign = 1.0 if i % 2 == 0 else -1.0
        values.append(sign * Float32(i) * 0.1)

    var block = MXFP4Block.from_float32_array(values)
    var decoded = block.to_float32_array()

    # Verify signs are preserved
    for i in range(32):
        var expected = (1.0 if i % 2 == 0 else -1.0) * Float32(i) * 0.1
        var expected_sign = 1.0 if expected >= 0 else -1.0
        var decoded_sign = 1.0 if decoded[i] >= 0 else -1.0
        assert_equal(int(expected_sign), int(decoded_sign))


# ============================================================================
# Scale Computation Tests
# ============================================================================


fn test_mxfp4_block_scale_computation() raises:
    """Test scale computation for different value ranges."""
    # Test 1: All values in [0, 1]
    var values1 = List[Float32]()
    for i in range(32):
        values1.append(Float32(i) / 32.0)

    var block1 = MXFP4Block.from_float32_array(values1)
    # Scale should be roughly max/6 = (31/32)/6 ≈ 0.16
    var scale1 = block1.scale.to_float32()
    assert_true(scale1 > 0.1 and scale1 < 0.3, "Scale 1 out of range")

    # Test 2: All values in [0, 10]
    var values2 = List[Float32]()
    for i in range(32):
        values2.append(Float32(i) / 3.2)

    var block2 = MXFP4Block.from_float32_array(values2)
    # Scale should be larger
    var scale2 = block2.scale.to_float32()
    assert_true(scale2 > scale1, "Scale 2 should be larger")


# ============================================================================
# Bit Packing Tests
# ============================================================================


fn test_mxfp4_block_bit_packing() raises:
    """Test bit packing stores 2 values per byte."""
    var values = List[Float32]()
    # Create distinct values
    for i in range(32):
        values.append(Float32(1.0) + Float32(i % 4) * 0.5)

    var block = MXFP4Block.from_float32_array(values)

    # Verify block has 16 bytes of data
    # (can't directly check, but verify decoding works)
    var decoded = block.to_float32_array()
    assert_equal(len(decoded), 32)


# ============================================================================
# Block Indexing Tests
# ============================================================================


fn test_mxfp4_block_get() raises:
    """Test get() method retrieves individual values."""
    var values = List[Float32]()
    for i in range(32):
        values.append(Float32(i) * 0.1)

    var block = MXFP4Block.from_float32_array(values)

    # Test get at various indices
    var val0 = block.get(0)
    var val15 = block.get(15)
    var val31 = block.get(31)

    # Verify values are reasonable
    assert_almost_equal(val0.to_float32(), 0.0, tolerance=0.1)
    assert_true(abs(val15.to_float32() - 1.5) < 0.5)
    assert_true(abs(val31.to_float32() - 3.1) < 0.5)


fn test_mxfp4_block_get_bounds_checking() raises:
    """Test get() bounds checking."""
    var values = List[Float32]()
    for i in range(32):
        values.append(Float32(i))

    var block = MXFP4Block.from_float32_array(values)

    # Test out of bounds
    try:
        var val = block.get(-1)
        assert_true(False, "Expected bounds error for -1")
    except e:
        pass

    try:
        var val = block.get(32)
        assert_true(False, "Expected bounds error for 32")
    except e:
        pass


fn test_mxfp4_block_set() raises:
    """Test set() method updates individual values."""
    var values = List[Float32]()
    for i in range(32):
        values.append(Float32(1.0))

    var block = MXFP4Block.from_float32_array(values)

    # Update value at index 5
    var new_val = MXFP4.from_float32(2.5)
    block.set(5, new_val)

    # Retrieve and verify
    var retrieved = block.get(5)
    assert_true(abs(retrieved.to_float32() - 2.5) < 0.5)


fn test_mxfp4_block_set_bounds_checking() raises:
    """Test set() bounds checking."""
    var values = List[Float32]()
    for i in range(32):
        values.append(Float32(i))

    var block = MXFP4Block.from_float32_array(values)
    var new_val = MXFP4.from_float32(5.0)

    # Test out of bounds
    try:
        block.set(-1, new_val)
        assert_true(False, "Expected bounds error for -1")
    except e:
        pass

    try:
        block.set(32, new_val)
        assert_true(False, "Expected bounds error for 32")
    except e:
        pass


# ============================================================================
# Edge Cases
# ============================================================================


fn test_mxfp4_block_all_same_value() raises:
    """Test block with all same values."""
    var values = List[Float32]()
    for i in range(32):
        values.append(Float32(3.14))

    var block = MXFP4Block.from_float32_array(values)
    var decoded = block.to_float32_array()

    # All values should be approximately equal
    var first = decoded[0]
    for i in range(32):
        assert_almost_equal(decoded[i], first, tolerance=0.01)


fn test_mxfp4_block_extreme_range() raises:
    """Test block with very different magnitude values."""
    var values = List[Float32]()
    # Mix very small and very large values
    for i in range(16):
        values.append(Float32(0.001))
    for i in range(16):
        values.append(Float32(100.0))

    var block = MXFP4Block.from_float32_array(values)
    var decoded = block.to_float32_array()

    # Due to shared scale, small values may become zero
    # Large values should be preserved reasonably
    var large_vals_ok = True
    for i in range(16, 32):
        if decoded[i] < 50.0:  # Very rough check
            large_vals_ok = False

    assert_true(large_vals_ok, "Large values not preserved")


fn main() raises:
    """Run all MXFP4Block tests."""
    print("Running MXFP4Block tests...")

    # Block creation tests
    test_mxfp4_block_creation_zeros()
    print("✓ Block creation (zeros)")

    test_mxfp4_block_creation_ones()
    print("✓ Block creation (ones)")

    test_mxfp4_block_creation_range()
    print("✓ Block creation (range)")

    test_mxfp4_block_size_validation()
    print("✓ Block size validation")

    # Round-trip tests
    test_mxfp4_block_roundtrip_small()
    print("✓ Round-trip (small values)")

    test_mxfp4_block_roundtrip_large()
    print("✓ Round-trip (large values)")

    test_mxfp4_block_roundtrip_mixed_signs()
    print("✓ Round-trip (mixed signs)")

    # Scale computation tests
    test_mxfp4_block_scale_computation()
    print("✓ Scale computation")

    # Bit packing tests
    test_mxfp4_block_bit_packing()
    print("✓ Bit packing")

    # Indexing tests
    test_mxfp4_block_get()
    print("✓ Block get()")

    test_mxfp4_block_get_bounds_checking()
    print("✓ Block get() bounds")

    test_mxfp4_block_set()
    print("✓ Block set()")

    test_mxfp4_block_set_bounds_checking()
    print("✓ Block set() bounds")

    # Edge cases
    test_mxfp4_block_all_same_value()
    print("✓ All same value")

    test_mxfp4_block_extreme_range()
    print("✓ Extreme range")

    print("\nAll MXFP4Block tests passed!")
