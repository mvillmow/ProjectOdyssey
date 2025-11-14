"""Tests for sequential sampler.

Tests SequentialSampler which yields dataset indices in order,
the default sampling strategy for deterministic data loading.
"""

from tests.shared.conftest import assert_true, assert_equal, TestFixtures


# ============================================================================
# Stub Implementation for TDD
# ============================================================================


struct StubSequentialSampler:
    """Minimal stub sequential sampler for testing Sampler interface.

    Yields indices in sequential order [0, 1, 2, ..., n-1].
    """

    var size: Int

    fn __init__(
        inoutself, size: Int
    ):
        """Create sequential sampler.

        Args:
            size: Number of indices to generate.
        """
        self.size = size

    fn __len__(self) -> Int:
        """Return number of indices."""
        return self.size

    fn get_index(self, position: Int) -> Int:
        """Get index at position.

        Args:
            position: Position in sequence (0 to size-1).

        Returns:
            Index value (same as position for sequential sampler).
        """
        return position


# ============================================================================
# SequentialSampler Creation Tests
# ============================================================================


fn test_sequential_sampler_creation() raises:
    """Test creating SequentialSampler with dataset size.

    Should create sampler that will yield indices 0 to n-1 in order,
    deterministic and reproducible.
    """
    var sampler = StubSequentialSampler(size=100)
    assert_equal(len(sampler), 100)


fn test_sequential_sampler_empty() raises:
    """Test creating SequentialSampler with size 0.

    Should create valid sampler that yields no indices,
    useful for edge case testing.
    """
    var sampler = StubSequentialSampler(size=0)
    assert_equal(len(sampler), 0)


# ============================================================================
# SequentialSampler Iteration Tests
# ============================================================================


fn test_sequential_sampler_yields_all_indices() raises:
    """Test that sampler yields all indices exactly once.

    Should produce indices [0, 1, 2, ..., n-1] without
    skipping or duplicating any.
    """
    var sampler = StubSequentialSampler(size=10)

    var indices = List[Int](capacity=10)
    for i in range(len(sampler)):
        indices.append(sampler.get_index(i))

    assert_equal(len(indices), 10)

    # Check all indices present and in order
    for i in range(10):
        assert_equal(indices[i], i)


fn test_sequential_sampler_order() raises:
    """Test that indices are yielded in sequential order.

    Should yield [0, 1, 2, 3, ...], not shuffled or reversed.
    This is the defining property of SequentialSampler.
    """
    var sampler = StubSequentialSampler(size=100)

    var indices = List[Int](capacity=100)
    for i in range(len(sampler)):
        indices.append(sampler.get_index(i))

    # Check indices are in order
    for i in range(100):
        assert_equal(indices[i], i)


fn test_sequential_sampler_deterministic() raises:
    """Test that sampler produces same sequence every time.

    Multiple iterations should yield identical index sequences,
    no randomness involved.
    """
    var sampler = StubSequentialSampler(size=50)

    # First iteration
    var indices1 = List[Int](capacity=50)
    for i in range(len(sampler)):
        indices1.append(sampler.get_index(i))

    # Second iteration
    var indices2 = List[Int](capacity=50)
    for i in range(len(sampler)):
        indices2.append(sampler.get_index(i))

    # Should be identical
    for i in range(50):
        assert_equal(indices1[i], indices2[i])


# ============================================================================
# SequentialSampler Range Tests
# ============================================================================


fn test_sequential_sampler_start_index():
    """Test indices start from 0.

    First yielded index should always be 0,
    not 1 or any other value.
    """
    # TODO(#39): Implement when SequentialSampler exists
    # var sampler = SequentialSampler(size=100)
    # var first_idx = next(iter(sampler))
    # assert_equal(first_idx, 0)
    pass


fn test_sequential_sampler_end_index():
    """Test indices end at size-1.

    Last yielded index should be size-1,
    as indices are 0-based.
    """
    # TODO(#39): Implement when SequentialSampler exists
    # var sampler = SequentialSampler(size=100)
    #
    # var last_idx = 0
    # for idx in sampler:
    #     last_idx = idx
    #
    # assert_equal(last_idx, 99)
    pass


fn test_sequential_sampler_no_negative_indices():
    """Test that sampler never yields negative indices.

    All indices should be >= 0,
    as negative indices would be invalid.
    """
    # TODO(#39): Implement when SequentialSampler exists
    # var sampler = SequentialSampler(size=100)
    #
    # for idx in sampler:
    #     assert_true(idx >= 0)
    pass


# ============================================================================
# SequentialSampler Integration Tests
# ============================================================================


fn test_sequential_sampler_with_dataloader():
    """Test using SequentialSampler with DataLoader.

    DataLoader should use sampler to determine batch order,
    producing deterministic batches.
    """
    # TODO(#39): Implement when DataLoader and SequentialSampler exist
    # var dataset = TestFixtures.sequential_dataset(n_samples=100)
    # var sampler = SequentialSampler(size=100)
    # var loader = DataLoader(dataset, batch_size=32, sampler=sampler)
    #
    # var first_batch = next(iter(loader))
    # # First batch should contain samples [0-31] in order
    # for i in range(32):
    #     assert_equal(first_batch.data[i, 0], Float32(i))
    pass


fn test_sequential_sampler_reusable():
    """Test that sampler can be reused across multiple epochs.

    Same sampler instance should work for multiple epochs,
    yielding same sequence each time.
    """
    # TODO(#39): Implement when SequentialSampler exists
    # var sampler = SequentialSampler(size=50)
    #
    # # Epoch 1
    # var epoch1_indices = List[Int]()
    # for idx in sampler:
    #     epoch1_indices.append(idx)
    #
    # # Epoch 2 (reuse same sampler)
    # var epoch2_indices = List[Int]()
    # for idx in sampler:
    #     epoch2_indices.append(idx)
    #
    # assert_equal(epoch1_indices, epoch2_indices)
    pass


# ============================================================================
# SequentialSampler Performance Tests
# ============================================================================


fn test_sequential_sampler_iteration_speed():
    """Test that iteration is fast.

    Should iterate through indices with minimal overhead,
    as this happens every training step.
    """
    # TODO(#39): Implement when SequentialSampler exists
    # var sampler = SequentialSampler(size=100000)
    #
    # # Should complete quickly
    # var count = 0
    # for idx in sampler:
    #     count += 1
    #
    # assert_equal(count, 100000)
    pass


fn test_sequential_sampler_memory_efficiency():
    """Test that sampler doesn't pre-allocate index array.

    Should generate indices on-the-fly, not store full array,
    to save memory for large datasets.
    """
    # TODO(#39): Implement when SequentialSampler exists
    # # Creating sampler for 1M indices should be instant and lightweight
    # var sampler = SequentialSampler(size=1000000)
    # assert_equal(len(sampler), 1000000)
    pass


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run all sequential sampler tests."""
    print("Running sequential sampler tests...")

    # Creation tests
    test_sequential_sampler_creation()
    test_sequential_sampler_empty()

    # Iteration tests
    test_sequential_sampler_yields_all_indices()
    test_sequential_sampler_order()
    test_sequential_sampler_deterministic()

    # Range tests
    test_sequential_sampler_start_index()
    test_sequential_sampler_end_index()
    test_sequential_sampler_no_negative_indices()

    # Integration tests
    test_sequential_sampler_with_dataloader()
    test_sequential_sampler_reusable()

    # Performance tests
    test_sequential_sampler_iteration_speed()
    test_sequential_sampler_memory_efficiency()

    print("✓ All sequential sampler tests passed!")
