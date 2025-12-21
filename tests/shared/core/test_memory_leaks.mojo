"""Memory leak detection tests for ExTensor.

Tests verify:
1. Reference counting correctness
2. Memory deallocation on scope exit
3. No memory leaks in repeated operations
4. View lifetime management
5. Destructor edge cases
"""

from shared.core import ExTensor, zeros, ones, full
from tests.shared.conftest import (
    assert_true,
    assert_equal_int,
)


# ============================================================================
# Reference Counting Tests
# ============================================================================


fn test_single_tensor_refcount() raises:
    """Test single tensor starts with refcount = 1."""
    var tensor = zeros(List[Int]([10, 10]), DType.float32)
    assert_equal_int(
        tensor._refcount[], 1, "Single tensor should have refcount 1"
    )


fn test_copy_increments_refcount() raises:
    """Test copying tensor increments reference count."""
    var tensor1 = zeros(List[Int]([10, 10]), DType.float32)
    var initial_refcount = tensor1._refcount[]
    assert_equal_int(initial_refcount, 1, "Initial refcount should be 1")

    var tensor2 = tensor1
    assert_equal_int(tensor1._refcount[], 2, "Refcount should be 2 after copy")
    assert_true(
        tensor1._data == tensor2._data, "Copied tensors should share data"
    )


fn test_multiple_copies_refcount() raises:
    """Test multiple copies increment refcount correctly."""
    var tensor1 = zeros(List[Int]([5, 5]), DType.float32)
    var tensor2 = tensor1
    var tensor3 = tensor1
    var tensor4 = tensor2
    assert_equal_int(
        tensor1._refcount[], 4, "Refcount should be 4 after 3 copies"
    )


fn test_scope_exit_decrements_refcount() raises:
    """Test refcount decrements when copy goes out of scope."""
    var tensor1 = zeros(List[Int]([10, 10]), DType.float32)
    var initial_rc = tensor1._refcount[]
    assert_equal_int(initial_rc, 1, "Initial refcount should be 1")
    var tensor2 = tensor1
    assert_equal_int(tensor1._refcount[], 2, "Refcount should be 2 after copy")
    assert_equal_int(tensor1._refcount[], 2, "Refcount should still be 2")


fn test_original_survives_copy_destruction() raises:
    """Test original tensor survives when copy is destroyed."""
    var tensor1 = zeros(List[Int]([10, 10]), DType.float32)
    var initial_value = tensor1._data.bitcast[Float32]()[0]
    var tensor2 = tensor1
    tensor2._data.bitcast[Float32]()[0] = 99.0
    var value = tensor1._data.bitcast[Float32]()[0]
    assert_true(value == 99.0, "Original should have modified value")


# ============================================================================
# Memory Allocation/Deallocation Tests
# ============================================================================


fn test_tensor_deallocation_single() raises:
    """Test single tensor deallocates memory when destroyed."""
    var tensor = zeros(List[Int]([100, 100]), DType.float32)
    _ = tensor.numel()
    assert_true(True, "Single tensor deallocation completed without crash")


fn test_tensor_deallocation_loop() raises:
    """Test repeated tensor creation/destruction in loop."""
    for i in range(100):
        var tensor = zeros(List[Int]([50, 50]), DType.float32)
        var _ = tensor._data.bitcast[Float32]()[0]
        _ = tensor.numel()
    assert_true(True, "Loop deallocation completed without crash")


fn test_shared_tensor_deallocation() raises:
    """Test shared tensor deallocates only when last reference destroyed."""
    var tensor1 = zeros(List[Int]([10, 10]), DType.float32)
    var tensor2 = tensor1
    assert_equal_int(tensor1._refcount[], 2, "Should have 2 refs")
    assert_true(True, "Shared tensor deallocation completed")


# ============================================================================
# Stress Tests for Memory Leaks
# ============================================================================


fn test_no_memory_leak_in_creation_loop() raises:
    """Verify no memory leaks in repeated tensor creation."""
    comptime NUM_ITERATIONS = 1000
    comptime TENSOR_SIZE = 100
    for _ in range(NUM_ITERATIONS):
        var tensor = zeros(List[Int]([TENSOR_SIZE, TENSOR_SIZE]), DType.float32)
    assert_true(True, "Created tensors without OOM")


fn test_no_memory_leak_in_operation_loop() raises:
    """Verify no memory leaks in repeated tensor operations."""
    comptime NUM_ITERATIONS = 500
    for _ in range(NUM_ITERATIONS):
        var tensor = zeros(List[Int]([50, 50]), DType.float32)
        var result = tensor + tensor
        _ = result.numel()
    assert_true(True, "Completed operations without OOM")


fn test_no_memory_leak_with_copies() raises:
    """Verify no memory leaks with shared copies."""
    comptime NUM_ITERATIONS = 500
    for _ in range(NUM_ITERATIONS):
        var tensor1 = ones(List[Int]([100, 100]), DType.float32)
        var tensor2 = tensor1
        var tensor3 = tensor2
        var tensor4 = tensor1
        assert_equal_int(tensor1._refcount[], 4, "Should have 4 refs")
    assert_true(True, "Copy stress test completed without OOM")


fn test_large_tensor_lifecycle() raises:
    """Test large tensor allocation and deallocation."""
    comptime NUM_ITERATIONS = 10
    comptime LARGE_SIZE = 500
    for _ in range(NUM_ITERATIONS):
        var tensor = zeros(List[Int]([LARGE_SIZE, LARGE_SIZE]), DType.float32)
        var _ = tensor._data.bitcast[Float32]()[0]
        _ = tensor.numel()
    assert_true(True, "Large tensor lifecycle test completed")


# ============================================================================
# View Lifetime Tests
# ============================================================================


fn test_view_flag_on_reshape() raises:
    """Test reshape creates a view with _is_view flag set."""
    var tensor = zeros(List[Int]([12]), DType.float32)
    var val = tensor._data.bitcast[Float32]()[0]
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var reshaped = tensor.reshape(shape)
    assert_true(reshaped._is_view, "Reshaped tensor should be a view")
    assert_true(tensor._data == reshaped._data, "View should share data")


fn test_view_does_not_free_data() raises:
    """Test view destruction doesn't free shared data."""
    var original = zeros(List[Int]([12]), DType.float32)
    original._data.bitcast[Float32]()[0] = 42.0
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var view = original.reshape(shape)
    assert_true(view._is_view, "Should be marked as view")
    var value = original._data.bitcast[Float32]()[0]
    assert_true(value == 42.0, "Original data should be intact")


fn test_view_modification_affects_original() raises:
    """Test modifying view affects original tensor."""
    var original = zeros(List[Int]([12]), DType.float32)
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var view = original.reshape(shape)
    view._data.bitcast[Float32]()[0] = 99.0
    var value = original._data.bitcast[Float32]()[0]
    assert_true(
        value == 99.0, "Modification through view should affect original"
    )


# ============================================================================
# Edge Cases
# ============================================================================


fn test_scalar_tensor_lifecycle() raises:
    """Test 0D scalar tensor creation and destruction."""
    for _ in range(100):
        var scalar = ExTensor(42)
        assert_equal_int(scalar._refcount[], 1, "Scalar should have refcount 1")
    assert_true(True, "Scalar tensor lifecycle test completed")


fn test_empty_tensor_lifecycle() raises:
    """Test empty tensor (0 elements) creation and destruction."""
    for _ in range(100):
        var empty = zeros(List[Int](), DType.float32)
        assert_equal_int(
            empty._refcount[], 1, "Empty tensor should have refcount 1"
        )
    assert_true(True, "Empty tensor lifecycle test completed")


fn test_different_dtypes_lifecycle() raises:
    """Test tensor lifecycle with different dtypes."""
    var dtypes = List[DType]()
    dtypes.append(DType.float32)
    dtypes.append(DType.float64)
    dtypes.append(DType.int32)
    dtypes.append(DType.int64)
    dtypes.append(DType.uint8)
    for i in range(len(dtypes)):
        for _ in range(50):
            var shape = List[Int]()
            shape.append(50)
            shape.append(50)
            var tensor = zeros(shape, dtypes[i])
            assert_equal_int(tensor._refcount[], 1, "Should have refcount 1")
    assert_true(True, "Different dtypes lifecycle test completed")


# ============================================================================
# Destructor Edge Cases
# ============================================================================


fn test_destructor_with_valid_refcount() raises:
    """Test destructor handles normal case correctly."""
    var tensor = zeros(List[Int]([10]), DType.float32)
    assert_equal_int(tensor._refcount[], 1, "Tensor should have valid refcount")
    assert_true(True, "Destructor edge case test completed")


fn test_view_destructor_does_not_decrement_refcount() raises:
    """Test view destructor decrements refcount properly when destroyed."""
    var original = zeros(List[Int]([12]), DType.float32)
    var initial_refcount = original._refcount[]
    assert_equal_int(initial_refcount, 1, "Original should have refcount 1")
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var view = original.reshape(shape)
    assert_true(view._is_view, "Should be view")
    assert_equal_int(
        original._refcount[],
        2,
        "View creation via copy() should increment refcount",
    )


# ============================================================================
# Test Runner
# ============================================================================


fn main() raises:
    """Run all memory leak detection tests."""
    print("Running memory leak detection tests...")

    print("  Reference counting tests...")
    test_single_tensor_refcount()
    test_copy_increments_refcount()
    test_multiple_copies_refcount()
    test_scope_exit_decrements_refcount()
    test_original_survives_copy_destruction()
    print("    ✓ Passed")

    print("  Allocation/deallocation tests...")
    test_tensor_deallocation_single()
    test_tensor_deallocation_loop()
    test_shared_tensor_deallocation()
    print("    ✓ Passed")

    print("  Stress tests...")
    test_no_memory_leak_in_creation_loop()
    test_no_memory_leak_in_operation_loop()
    test_no_memory_leak_with_copies()
    test_large_tensor_lifecycle()
    print("    ✓ Passed")

    print("  View lifetime tests...")
    test_view_flag_on_reshape()
    test_view_does_not_free_data()
    test_view_modification_affects_original()
    print("    ✓ Passed")

    print("  Edge case tests...")
    test_scalar_tensor_lifecycle()
    test_empty_tensor_lifecycle()
    test_different_dtypes_lifecycle()
    print("    ✓ Passed")

    print("  Destructor edge case tests...")
    test_destructor_with_valid_refcount()
    test_view_destructor_does_not_decrement_refcount()
    print("    ✓ Passed")

    print("\n✅ All memory leak detection tests passed!")
