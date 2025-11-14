"""Tests for base data loader interface.

Tests the DataLoader interface that all loaders must implement,
ensuring consistent API for batch iteration during training.
"""

from tests.shared.conftest import assert_true, assert_equal, TestFixtures


# ============================================================================
# Stub Implementations for TDD
# ============================================================================


struct StubDataset:
    """Minimal stub dataset for loader testing."""

    var size: Int

    fn __init__(
        inoutself, size: Int
    ):
        self.size = size

    fn __len__(self) -> Int:
        return self.size

    fn __getitem__(self, index: Int) raises -> Tuple[Float32, Int]:
        if index < 0 or index >= self.size:
            raise Error("Index out of bounds")
        return (Float32(index), index)


struct StubBatch:
    """Minimal stub batch for testing batch operations."""

    var data: List[Float32]
    var labels: List[Int]

    fn __init__(
        inoutself, capacity: Int
    ):
        self.data = List[Float32](capacity=capacity)
        self.labels = List[Int](capacity=capacity)

    fn add_sample(
        inoutself, data: Float32, label: Int
    ):
        self.data.append(data)
        self.labels.append(label)

    fn size(self) -> Int:
        return len(self.data)


struct StubDataLoader:
    """Minimal stub data loader for testing DataLoader interface.

    Provides basic batching functionality without complex features.
    """

    var dataset: StubDataset
    var batch_size: Int
    var drop_last: Bool
    var num_batches: Int

    fn __init__(
        inoutself,
        dataset: StubDataset,
        batch_size: Int,
        drop_last: Bool = False,
    ) raises:
        """Create stub data loader.

        Args:
            dataset: Dataset to load from.
            batch_size: Number of samples per batch.
            drop_last: Whether to drop last incomplete batch.

        Raises:
            Error if batch_size <= 0.
        """
        if batch_size <= 0:
            raise Error("batch_size must be positive")

        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        # Calculate number of batches
        var n_samples = len(dataset)
        if n_samples == 0:
            self.num_batches = 0
        elif drop_last:
            self.num_batches = n_samples // batch_size
        else:
            self.num_batches = (n_samples + batch_size - 1) // batch_size

    fn __len__(self) -> Int:
        """Return number of batches."""
        return self.num_batches

    fn get_batch(self, batch_idx: Int) raises -> StubBatch:
        """Get batch at specified index.

        Args:
            batch_idx: Batch index.

        Returns:
            Batch containing samples.
        """
        var start_idx = batch_idx * self.batch_size
        var end_idx = start_idx + self.batch_size
        var dataset_len = len(self.dataset)

        if end_idx > dataset_len:
            end_idx = dataset_len

        var batch = StubBatch(capacity=self.batch_size)
        for i in range(start_idx, end_idx):
            var sample = self.dataset[i]
            batch.add_sample(sample[0], sample[1])

        return batch


# ============================================================================
# Base DataLoader Interface Tests
# ============================================================================


fn test_loader_has_len_method() raises:
    """Test that DataLoader interface requires __len__ method.

    Should return the number of batches (not samples),
    enabling progress bars and epoch calculations.
    """
    var dataset = StubDataset(size=100)
    var loader = StubDataLoader(dataset, batch_size=32)
    assert_equal(len(loader), 4)  # ceil(100/32) = 4 batches


fn test_loader_batch_size_consistency() raises:
    """Test that batches have consistent size (except possibly last).

    All batches should have batch_size samples except the last batch
    which may be smaller if dataset size is not divisible by batch_size.
    """
    var dataset = StubDataset(size=100)
    var loader = StubDataLoader(dataset, batch_size=32)

    # First 3 batches should be size 32
    var batch0 = loader.get_batch(0)
    assert_equal(batch0.size(), 32)

    var batch1 = loader.get_batch(1)
    assert_equal(batch1.size(), 32)

    var batch2 = loader.get_batch(2)
    assert_equal(batch2.size(), 32)

    # Last batch should be size 4 (100 - 3*32)
    var batch3 = loader.get_batch(3)
    assert_equal(batch3.size(), 4)


fn test_loader_empty_dataset() raises:
    """Test loader behavior with empty dataset.

    Should create valid loader that yields zero batches,
    not crash when iterated.
    """
    var dataset = StubDataset(size=0)
    var loader = StubDataLoader(dataset, batch_size=32)
    assert_equal(len(loader), 0)


fn test_loader_single_sample() raises:
    """Test loader with single sample dataset.

    Should create one batch containing the single sample,
    even though batch_size is larger.
    """
    var dataset = StubDataset(size=1)
    var loader = StubDataLoader(dataset, batch_size=32)

    assert_equal(len(loader), 1)

    var batch = loader.get_batch(0)
    assert_equal(batch.size(), 1)


# ============================================================================
# DataLoader Configuration Tests
# ============================================================================


fn test_loader_batch_size_validation() raises:
    """Test that batch_size must be positive.

    Creating loader with batch_size <= 0 should raise ValueError,
    as it would cause division by zero or infinite loop.
    """
    var dataset = StubDataset(size=100)
    var error_raised = False
    try:
        var loader = StubDataLoader(dataset, batch_size=0)
    except:
        error_raised = True
    assert_true(error_raised, "Should have raised error for batch_size <= 0")


fn test_loader_drop_last_option() raises:
    """Test drop_last parameter.

    When drop_last=True, should discard final partial batch,
    ensuring all batches have exactly batch_size samples.
    """
    var dataset = StubDataset(size=100)
    var loader = StubDataLoader(dataset, batch_size=32, drop_last=True)

    # Should drop last batch of 4 samples
    assert_equal(len(loader), 3)

    # All remaining batches should have exactly batch_size samples
    for i in range(len(loader)):
        var batch = loader.get_batch(i)
        assert_equal(batch.size(), 32)


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run all base loader tests."""
    print("Running base loader tests...")

    # Interface tests
    test_loader_has_iter_method()
    test_loader_has_len_method()
    test_loader_iteration()
    test_loader_batch_size_consistency()
    test_loader_empty_dataset()
    test_loader_single_sample()

    # Configuration tests
    test_loader_batch_size_validation()
    test_loader_drop_last_option()
    test_loader_reset_between_epochs()

    print("✓ All base loader tests passed!")
