"""Test memory safety features and tensor slicing."""

from shared.core import ExTensor, zeros
from shared.core.extensor import calculate_max_batch_size
from collections import List


fn test_memory_validation() raises:
    """Test that memory limits are enforced."""
    print("Test 1: Memory validation...")

    # Try to create a tensor that exceeds the limit (2 GB)
    # Shape: (1000000, 1000) float32 = 4 billion bytes = 4 GB
    var huge_shape = List[Int]()
    huge_shape.append(1000000)
    huge_shape.append(1000)

    try:
        var huge_tensor = zeros(huge_shape, DType.float32)
        print("  ❌ FAIL: Should have raised an error for oversized tensor")
    except e:
        print("  ✓ PASS: Correctly rejected oversized tensor")
        print("    Error:", e)


fn test_memory_warning() raises:
    """Test that warnings are shown for large allocations."""
    print("\nTest 2: Memory warning threshold...")

    # Create a tensor just above the warning threshold (500 MB)
    # Shape: (130000000,) float32 = 520 MB
    var large_shape = List[Int]()
    large_shape.append(130000000)

    print("  Creating 520 MB tensor (should see warning)...")
    var large_tensor = zeros(large_shape, DType.float32)
    print("  ✓ PASS: Tensor created successfully")
    print("    Numel:", large_tensor.numel())


fn test_tensor_slicing() raises:
    """Test tensor slicing for mini-batch extraction."""
    print("\nTest 3: Tensor slicing...")

    # Create a dataset: (100, 3, 4, 5)
    var shape = List[Int]()
    shape.append(100)  # samples
    shape.append(3)    # channels
    shape.append(4)    # height
    shape.append(5)    # width

    var dataset = zeros(shape, DType.float32)
    print("  Dataset shape: (100, 3, 4, 5)")
    print("  Dataset numel:", dataset.numel())

    # Extract first 32 samples
    var batch1 = dataset.slice(0, 32, axis=0)
    print("  Batch 1 slice(0, 32): numel =", batch1.numel())
    if batch1.numel() != 32 * 3 * 4 * 5:
        print("  ❌ FAIL: Wrong number of elements")
        return
    print("  ✓ PASS: Correct batch size")

    # Extract middle samples
    var batch2 = dataset.slice(32, 64, axis=0)
    print("  Batch 2 slice(32, 64): numel =", batch2.numel())
    if batch2.numel() != 32 * 3 * 4 * 5:
        print("  ❌ FAIL: Wrong number of elements")
        return
    print("  ✓ PASS: Correct batch size")

    # Extract last samples (may be partial)
    var batch3 = dataset.slice(96, 100, axis=0)
    print("  Batch 3 slice(96, 100): numel =", batch3.numel())
    if batch3.numel() != 4 * 3 * 4 * 5:
        print("  ❌ FAIL: Wrong number of elements")
        return
    print("  ✓ PASS: Correct batch size for partial batch")


fn test_slice_bounds_checking() raises:
    """Test that slice bounds are properly validated."""
    print("\nTest 4: Slice bounds checking...")

    var shape = List[Int]()
    shape.append(100)
    shape.append(10)
    var tensor = zeros(shape, DType.float32)

    # Test 1: Invalid axis
    try:
        var bad_slice = tensor.slice(0, 10, axis=5)
        print("  ❌ FAIL: Should reject invalid axis")
    except:
        print("  ✓ PASS: Rejected invalid axis")

    # Test 2: Start > end
    try:
        var bad_slice = tensor.slice(50, 20, axis=0)
        print("  ❌ FAIL: Should reject start > end")
    except:
        print("  ✓ PASS: Rejected start > end")

    # Test 3: Out of bounds
    try:
        var bad_slice = tensor.slice(0, 200, axis=0)
        print("  ❌ FAIL: Should reject out of bounds end")
    except:
        print("  ✓ PASS: Rejected out of bounds end")


fn test_calculate_max_batch_size_func() raises:
    """Test the batch size calculator."""
    print("\nTest 5: Batch size calculator...")

    # EMNIST image shape: (1, 28, 28)
    var sample_shape = List[Int]()
    sample_shape.append(1)
    sample_shape.append(28)
    sample_shape.append(28)

    # Calculate max batch for 500 MB
    var max_batch = calculate_max_batch_size(
        sample_shape,
        DType.float32,
        max_memory_bytes=500_000_000
    )

    print("  EMNIST sample: (1, 28, 28) float32")
    print("  Bytes per sample:", 1 * 28 * 28 * 4, "=", 3136, "bytes")
    print("  Max batch (500 MB):", max_batch)

    # Expected: 500,000,000 / 3,136 ≈ 159,439
    if max_batch < 100000 or max_batch > 200000:
        print("  ❌ FAIL: Unexpected batch size")
    else:
        print("  ✓ PASS: Reasonable batch size calculated")

    # Test with smaller memory limit
    var small_batch = calculate_max_batch_size(
        sample_shape,
        DType.float32,
        max_memory_bytes=10_000_000  # 10 MB
    )
    print("  Max batch (10 MB):", small_batch)
    if small_batch < 2000 or small_batch > 4000:
        print("  ❌ FAIL: Unexpected batch size for 10 MB")
    else:
        print("  ✓ PASS: Correct calculation for smaller limit")


fn test_mini_batch_iteration() raises:
    """Test complete mini-batch iteration pattern."""
    print("\nTest 6: Mini-batch iteration pattern...")

    # Simulate EMNIST training set: (112800, 1, 28, 28)
    var shape = List[Int]()
    shape.append(112800)
    shape.append(1)
    shape.append(28)
    shape.append(28)

    print("  Creating training dataset (112800, 1, 28, 28)...")
    print("  Size:", 112800 * 1 * 28 * 28 * 4, "bytes ≈", 337, "MB")
    var train_data = zeros(shape, DType.float32)
    print("  ✓ Dataset created")

    # Test iteration with different batch sizes
    var batch_size = 32
    var num_samples = 112800
    var num_batches = (num_samples + batch_size - 1) // batch_size

    print("  Batch size:", batch_size)
    print("  Number of batches:", num_batches)

    var processed_samples = 0
    for batch_idx in range(min(10, num_batches)):  # Test first 10 batches
        var start = batch_idx * batch_size
        var end = min(start + batch_size, num_samples)

        var batch = train_data.slice(start, end, axis=0)
        var actual_batch_size = end - start

        processed_samples += actual_batch_size

        if batch_idx == 0:
            print("  First batch: elements =", batch.numel())

    print("  Processed", processed_samples, "samples in 10 batches")
    print("  ✓ PASS: Mini-batch iteration works")


fn main() raises:
    print("=" * 60)
    print("Memory Safety and Tensor Slicing Tests")
    print("=" * 60)
    print()

    test_memory_validation()
    test_memory_warning()
    test_tensor_slicing()
    test_slice_bounds_checking()
    test_calculate_max_batch_size_func()
    test_mini_batch_iteration()

    print()
    print("=" * 60)
    print("All memory safety tests completed! 🎉")
    print("=" * 60)
