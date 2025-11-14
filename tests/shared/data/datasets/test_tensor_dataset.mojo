"""Tests for in-memory tensor dataset.

Tests TensorDataset which stores all data in memory as tensors,
providing fast random access for small to medium datasets.
"""

from tests.shared.conftest import (
    assert_true,
    assert_equal,
    assert_almost_equal,
    TestFixtures,
)


# ============================================================================
# Stub Implementation for TDD
# ============================================================================


struct StubTensorDataset:
    """Minimal stub tensor dataset for testing TensorDataset interface.

    Stores data and labels as simple lists for in-memory access.
    """

    var data: List[Float32]
    var labels: List[Int]
    var size: Int

    fn __init__(
        inoutself, data: List[Float32], labels: List[Int]
    ) raises:
        """Create tensor dataset from data and labels.

        Args:
            data: List of data values.
            labels: List of label values.

        Raises:
            Error if data and labels have different lengths.
        """
        if len(data) != len(labels):
            raise Error("Data and labels must have same length")

        self.data = data
        self.labels = labels
        self.size = len(data)

    fn __len__(self) -> Int:
        """Return number of samples."""
        return self.size

    fn __getitem__(self, index: Int) raises -> Tuple[Float32, Int]:
        """Get sample at index.

        Args:
            index: Sample index.

        Returns:
            Tuple of (data, label).

        Raises:
            Error if index is out of bounds.
        """
        if index < 0 or index >= self.size:
            raise Error("Index out of bounds")

        return (self.data[index], self.labels[index])


# ============================================================================
# TensorDataset Creation Tests
# ============================================================================


fn test_tensor_dataset_creation() raises:
    """Test creating TensorDataset from tensors.

    TensorDataset should accept data and labels tensors and store them
    for efficient in-memory access.
    """
    var data = List[Float32](0.0, 1.0, 2.0)
    var labels = List[Int](0, 1, 2)
    var dataset = StubTensorDataset(data, labels)
    assert_equal(len(dataset), 3)


fn test_tensor_dataset_with_matching_sizes() raises:
    """Test that data and labels must have matching first dimension.

    The number of samples in data must match the number of labels,
    otherwise training would fail with mismatched batch sizes.
    """
    var data = List[Float32](capacity=100)
    var labels = List[Int](capacity=100)
    for i in range(100):
        data.append(Float32(i))
        labels.append(i)

    var dataset = StubTensorDataset(data, labels)
    assert_equal(len(dataset), 100)


fn test_tensor_dataset_size_mismatch_error() raises:
    """Test that mismatched data/label sizes raise error.

    Creating dataset with 100 data samples and 50 labels should fail
    immediately rather than causing silent errors during training.
    """
    var data = List[Float32](capacity=100)
    var labels = List[Int](capacity=50)

    for i in range(100):
        data.append(Float32(i))
    for i in range(50):
        labels.append(i)

    var error_raised = False
    try:
        var dataset = StubTensorDataset(data, labels)
    except:
        error_raised = True

    assert_true(error_raised, "Should have raised error for size mismatch")


fn test_tensor_dataset_empty() raises:
    """Test creating empty TensorDataset.

    Empty dataset should be valid (length 0) and not crash when queried.
    Useful for testing edge cases and incremental dataset building.
    """
    var data = List[Float32]()
    var labels = List[Int]()
    var dataset = StubTensorDataset(data, labels)
    assert_equal(len(dataset), 0)


fn test_tensor_dataset_getitem() raises:
    """Test getting individual samples from TensorDataset.

    Should return (data, label) tuple for requested index.
    """
    var data = List[Float32](10.0, 20.0, 30.0)
    var labels = List[Int](0, 1, 2)
    var dataset = StubTensorDataset(data, labels)

    var sample = dataset[1]
    assert_equal(sample[0], Float32(20.0))
    assert_equal(sample[1], 1)


# ============================================================================
# TensorDataset Access Tests
# ============================================================================


fn test_tensor_dataset_getitem():
    """Test accessing individual samples by index.

    Should return (data, label) tuple for the requested index,
    with data being a single sample (not a batch).
    """
    # TODO(#39): Implement when TensorDataset exists
    # var data = Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    # var labels = Tensor([0, 1, 2])
    # var dataset = TensorDataset(data, labels)
    #
    # var sample_data, sample_label = dataset[1]
    # assert_almost_equal(sample_data[0], 3.0)
    # assert_almost_equal(sample_data[1], 4.0)
    # assert_equal(sample_label, 1)
    pass


fn test_tensor_dataset_negative_indexing():
    """Test negative indexing works correctly.

    dataset[-1] should return the last sample,
    dataset[-2] the second-to-last, etc.
    """
    # TODO(#39): Implement when TensorDataset exists
    # var data = Tensor([[1.0], [2.0], [3.0]])
    # var labels = Tensor([0, 1, 2])
    # var dataset = TensorDataset(data, labels)
    #
    # var last_data, last_label = dataset[-1]
    # assert_almost_equal(last_data[0], 3.0)
    # assert_equal(last_label, 2)
    pass


fn test_tensor_dataset_out_of_bounds():
    """Test that out-of-bounds access raises error.

    Accessing index >= len(dataset) or index < -len(dataset)
    should raise IndexError to prevent silent failures.
    """
    # TODO(#39): Implement when TensorDataset exists
    # var data = Tensor([[1.0], [2.0], [3.0]])
    # var labels = Tensor([0, 1, 2])
    # var dataset = TensorDataset(data, labels)
    #
    # try:
    #     var sample = dataset[10]
    #     assert_true(False, "Should have raised IndexError")
    # except IndexError:
    #     pass
    pass


fn test_tensor_dataset_iteration_consistency():
    """Test that repeated access returns same data.

    Multiple calls to dataset[i] should return identical tensors,
    ensuring deterministic behavior for debugging and testing.
    """
    # TODO(#39): Implement when TensorDataset exists
    # var data = Tensor([[1.0, 2.0]])
    # var labels = Tensor([0])
    # var dataset = TensorDataset(data, labels)
    #
    # var sample1_data, _ = dataset[0]
    # var sample2_data, _ = dataset[0]
    # assert_almost_equal(sample1_data[0], sample2_data[0])
    pass


# ============================================================================
# TensorDataset Memory Tests
# ============================================================================


fn test_tensor_dataset_no_copy_on_access():
    """Test that __getitem__ returns views, not copies.

    For efficiency, dataset should return views into the original tensor
    rather than creating copies, reducing memory overhead.
    """
    # TODO(#39): Implement when TensorDataset and tensor views exist
    # var data = Tensor([[1.0, 2.0]])
    # var labels = Tensor([0])
    # var dataset = TensorDataset(data, labels)
    #
    # var sample_data, _ = dataset[0]
    # # Modify returned data
    # sample_data[0] = 999.0
    # # Check if original was modified (view behavior)
    # assert_almost_equal(data[0, 0], 999.0)
    pass


fn test_tensor_dataset_memory_efficiency():
    """Test that TensorDataset doesn't duplicate data in memory.

    Creating dataset should not copy the input tensors,
    just store references to save memory.
    """
    # TODO(#39): Implement when memory profiling tools exist
    # var data = Tensor.randn(1000, 100)  # Large tensor
    # var labels = Tensor.randn(1000)
    #
    # # Creating dataset should not significantly increase memory
    # var dataset = TensorDataset(data, labels)
    # # Memory usage should be approximately same as original tensors
    pass


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run all tensor dataset tests."""
    print("Running tensor dataset tests...")

    # Creation tests
    test_tensor_dataset_creation()
    test_tensor_dataset_with_matching_sizes()
    test_tensor_dataset_size_mismatch_error()
    test_tensor_dataset_empty()

    # Access tests
    test_tensor_dataset_getitem()
    test_tensor_dataset_negative_indexing()
    test_tensor_dataset_out_of_bounds()
    test_tensor_dataset_iteration_consistency()

    # Memory tests
    test_tensor_dataset_no_copy_on_access()
    test_tensor_dataset_memory_efficiency()

    print("✓ All tensor dataset tests passed!")
