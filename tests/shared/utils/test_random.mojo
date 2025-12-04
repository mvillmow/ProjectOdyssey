"""Tests for random seed management module.

This module tests random seed functionality including:
- Global seed setting for reproducibility
- Random state save/restore
- Cross-library synchronization (Mojo random + future PyTorch interop)
- Seed validation and edge cases
"""

from tests.shared.conftest import (
    assert_true,
    assert_false,
    assert_equal,
    assert_not_equal,
    TestFixtures,
)
from shared.utils.random import (
    set_seed,
    get_global_seed,
    get_random_state,
    set_random_state,
    save_random_state,
    get_saved_state,
    random_uniform,
    random_normal,
    random_int,
    random_choice,
    shuffle,
)


# ============================================================================
# Test Global Seed Setting
# ============================================================================


fn test_set_global_seed() raises:
    """Test setting global random seed."""
    # Set seed to 42
    set_seed(42)
    var val1 = random_uniform()
    var val2 = random_uniform()

    # Reset seed to 42
    set_seed(42)
    var val3 = random_uniform()
    var val4 = random_uniform()

    # Verify: sequences are identical
    assert_equal(val1, val3, "First uniform values should be identical with same seed")
    assert_equal(val2, val4, "Second uniform values should be identical with same seed")


fn test_seed_affects_all_generators() raises:
    """Test global seed affects all random number generators."""
    # Set seed to 42
    set_seed(42)
    var u1 = random_uniform()
    var n1 = random_normal()
    var i1 = random_int(0, 100)

    # Reset seed to 42
    set_seed(42)
    var u2 = random_uniform()
    var n2 = random_normal()
    var i2 = random_int(0, 100)

    # Verify: all sequences are identical
    assert_equal(u1, u2, "Uniform sequences should match with same seed")
    assert_equal(n1, n2, "Normal sequences should match with same seed")
    assert_equal(i1, i2, "Integer sequences should match with same seed")


fn test_different_seeds_produce_different_sequences() raises:
    """Test different seeds produce different random sequences."""
    # Set seed to 42
    set_seed(42)
    var val_a = random_uniform()

    # Set seed to 123
    set_seed(123)
    var val_b = random_uniform()

    # Verify: A != B (sequences differ)
    assert_not_equal(val_a, val_b, "Different seeds should produce different values")


fn test_seed_zero() raises:
    """Test seed value of 0 is valid and reproducible."""
    # Set seed to 0
    set_seed(0)
    var val1 = random_uniform()

    # Reset seed to 0
    set_seed(0)
    var val2 = random_uniform()

    # Verify: sequences match
    assert_equal(val1, val2, "Seed 0 should be reproducible")


fn test_seed_max_value() raises:
    """Test maximum valid seed value."""
    # Set seed to large value
    set_seed(2147483647)
    var val1 = random_uniform()

    # Reset and verify reproducibility
    set_seed(2147483647)
    var val2 = random_uniform()

    # Verify: sequence is reproducible
    assert_equal(val1, val2, "Large seed values should be reproducible")


# ============================================================================
# Test Random State Save/Restore
# ============================================================================


fn test_save_random_state() raises:
    """Test saving current random state."""
    # Set seed to 42
    set_seed(42)
    var _ = random_uniform()
    var _ = random_uniform()

    # Save state
    var state = get_random_state()

    # Verify: state object contains current RNG state
    assert_equal(state.seed_used, 42, "Saved state should have seed 42")


fn test_restore_random_state() raises:
    """Test restoring saved random state.

    Note: Current implementation only saves the seed, not the RNG position.
    So restoring state re-seeds, starting the sequence from the beginning.
    """
    # Set seed to 42
    set_seed(42)
    var a = random_uniform()
    var b = random_uniform()

    # Save state - this saves the seed value
    var saved_state = get_random_state()

    # Verify seed was stored
    assert_equal(saved_state.seed_used, 42, "Saved state should have seed 42")

    # Generate more numbers
    var _ = random_uniform()
    var _ = random_uniform()

    # Restore state - this re-seeds with 42
    set_random_state(saved_state)

    # After restore, we start from the beginning of the seed's sequence
    var a2 = random_uniform()
    var b2 = random_uniform()

    # Verify: a == a2, b == b2 (re-seeding gives same sequence)
    assert_equal(a, a2, "Values after restore should match initial values")
    assert_equal(b, b2, "Values after restore should match initial values")


fn test_state_roundtrip() raises:
    """Test saving and restoring state preserves reproducibility.

    Note: Current implementation saves seed only, so restore re-starts the sequence.
    """
    # Set seed
    set_seed(42)

    # Generate initial values
    var a1 = random_uniform()
    var a2 = random_uniform()

    # Save state (stores seed 42)
    var state = get_random_state()

    # Generate more values
    var _ = random_uniform()
    var _ = random_uniform()

    # Restore state (re-seeds with 42)
    set_random_state(state)

    # Generate values after restore - should match initial sequence
    var a1_again = random_uniform()
    var a2_again = random_uniform()

    # Verify: sequence restarts from beginning of seed
    assert_equal(a1, a1_again, "Restored sequence should restart from beginning")
    assert_equal(a2, a2_again, "Restored sequence should restart from beginning")


fn test_save_multiple_states() raises:
    """Test saving and restoring multiple states."""
    # Set seed to 42
    set_seed(42)

    # Generate a
    var a = random_uniform()

    # Save state1
    save_random_state(get_random_state())

    # Generate b
    var b = random_uniform()

    # Save state2
    save_random_state(get_random_state())

    # Generate c
    var c = random_uniform()

    # Restore state1
    var state1 = get_saved_state(0)
    set_random_state(state1)

    # Generate d
    var d = random_uniform()

    # Verify: d == b (restored to state1)
    assert_equal(d, b, "State 1 should restore to sequence b")

    # Restore state2
    var state2 = get_saved_state(1)
    set_random_state(state2)

    # Generate e
    var e = random_uniform()

    # Verify: e == c (restored to state2)
    assert_equal(e, c, "State 2 should restore to sequence c")


# ============================================================================
# Test Reproducibility
# ============================================================================


fn test_reproducible_training() raises:
    """Test training is reproducible with same seed."""
    # TODO(#44): Implement when full training workflow exists
    # Set seed to 42
    # Train model for 5 epochs
    # Record final loss
    # Reset seed to 42
    # Train same model for 5 epochs
    # Verify: final losses are identical
    pass


fn test_reproducible_data_augmentation() raises:
    """Test data augmentation is reproducible with same seed."""
    # TODO(#44): Implement when data augmentation exists
    # Set seed to 42
    # Apply random augmentation to image
    # Record augmented image
    # Reset seed to 42
    # Apply same augmentation
    # Verify: augmented images are identical
    pass


fn test_reproducible_weight_initialization() raises:
    """Test weight initialization is reproducible with same seed."""
    # TODO(#44): Implement when weight initialization exists
    # Set seed to 42
    # Initialize model weights
    # Record weights
    # Set seed to 42
    # Initialize same model
    # Verify: all weights are identical
    pass


fn test_reproducible_batch_sampling() raises:
    """Test batch sampling is reproducible with same seed."""
    # TODO(#44): Implement when DataLoader exists
    # Set seed to 42
    # Create DataLoader with shuffle=True
    # Get first batch indices
    # Reset seed to 42
    # Create DataLoader again
    # Get first batch indices
    # Verify: batch indices are identical
    pass


# ============================================================================
# Test Cross-Library Synchronization
# ============================================================================


fn test_sync_with_external_library() raises:
    """Test syncing random state with external library (future PyTorch)."""
    # TODO(#44): Implement when external library integration exists
    # Set seed in Mojo: 42
    # Set seed in external lib: 42
    # Generate random numbers in both
    # Verify: sequences match (or have known relationship)
    pass


fn test_restore_external_state() raises:
    """Test restoring state from external library."""
    # TODO(#44): Implement when external state import exists
    # Create state in external library
    # Import state to Mojo
    # Generate numbers in Mojo
    # Verify: continuation matches external library
    pass


# ============================================================================
# Test Thread Safety
# ============================================================================


fn test_seed_thread_local() raises:
    """Test each thread has independent random state."""
    # TODO(#44): Implement when threading support exists
    # Create two threads
    # Set different seeds in each thread
    # Generate numbers in both threads
    # Verify: sequences are different
    # Verify: each thread's sequence is reproducible
    pass


fn test_seed_global_across_threads() raises:
    """Test global seed affects all threads (optional behavior)."""
    # TODO(#44): Implement when threading support exists
    # Set global seed
    # Create multiple threads
    # Generate numbers in each thread
    # Verify: all threads affected by global seed
    # OR verify: each thread maintains independent state
    pass


# ============================================================================
# Test Seed Validation
# ============================================================================


fn test_negative_seed() raises:
    """Test handling of negative seed values."""
    # TODO(#44): Implement when set_seed exists
    # Try to set seed to -1
    # Verify: either converted to positive value OR error raised
    # If converted, verify reproducibility still works
    pass


fn test_seed_type_validation() raises:
    """Test seed must be integer type."""
    # TODO(#44): Implement when set_seed exists
    # Try to set seed to Float32(42.5)
    # Verify: error is raised (type mismatch)
    pass


fn test_seed_overflow() raises:
    """Test handling of seed values beyond valid range."""
    # TODO(#44): Implement when set_seed exists
    # Try to set seed to value > platform max int
    # Verify: either truncated/wrapped OR error raised
    pass


# ============================================================================
# Test Random Number Generation Quality
# ============================================================================


fn test_uniform_distribution() raises:
    """Test random numbers follow uniform distribution."""
    # TODO(#44): Implement when random() exists
    # Generate 10000 random numbers in [0, 1)
    # Verify: mean ≈ 0.5
    # Verify: values are roughly uniformly distributed
    # Use chi-square test or similar
    pass


fn test_normal_distribution() raises:
    """Test random numbers follow normal distribution."""
    # TODO(#44): Implement when randn() exists
    # Generate 10000 random numbers from N(0, 1)
    # Verify: mean ≈ 0.0
    # Verify: std ≈ 1.0
    # Verify: distribution shape is approximately normal
    pass


fn test_integer_range() raises:
    """Test random integers are within specified range."""
    # TODO(#44): Implement when randint() exists
    # Generate 1000 random integers in [0, 10)
    # Verify: all values are in [0, 10)
    # Verify: all values 0-9 appear at least once
    pass


fn test_randomness_independence() raises:
    """Test sequential random numbers are independent."""
    # TODO(#44): Implement when random number generators exist
    # Generate sequence of 1000 random numbers
    # Compute autocorrelation
    # Verify: low autocorrelation (numbers are independent)
    pass


# ============================================================================
# Test Specific Random Functions
# ============================================================================


fn test_randn_shape() raises:
    """Test randn generates tensor with correct shape."""
    # TODO(#44): Implement when randn() exists
    # Generate random tensor: randn(3, 4, 5)
    # Verify: shape is (3, 4, 5)
    # Verify: 60 random values (3*4*5)
    pass


fn test_randint_bounds() raises:
    """Test randint respects low and high bounds."""
    set_seed(42)
    # Generate 100 random integers: randint(low=5, high=15)
    for _ in range(100):
        var val = random_int(5, 15)
        # Verify: all values >= 5
        assert_true(val >= 5, "Random int should be >= 5")
        # Verify: all values < 15
        assert_true(val < 15, "Random int should be < 15")


fn test_random_choice() raises:
    """Test randomly choosing from list of options."""
    set_seed(42)
    var options = List[Int](10, 20, 30, 40)

    # Choose 100 times
    var count_10 = 0
    var count_20 = 0
    var count_30 = 0
    var count_40 = 0

    for _ in range(100):
        var choice = random_choice(options)
        if choice == 10:
            count_10 += 1
        elif choice == 20:
            count_20 += 1
        elif choice == 30:
            count_30 += 1
        elif choice == 40:
            count_40 += 1

    # Verify: all options appear at least once
    assert_true(count_10 > 0, "Option 10 should appear at least once")
    assert_true(count_20 > 0, "Option 20 should appear at least once")
    assert_true(count_30 > 0, "Option 30 should appear at least once")
    assert_true(count_40 > 0, "Option 40 should appear at least once")


fn test_random_permutation() raises:
    """Test random permutation of array."""
    set_seed(42)
    var array = List[Int](0, 1, 2, 3, 4)
    var original_sum = 0 + 1 + 2 + 3 + 4

    # Create a copy and shuffle it
    var shuffled = List[Int](0, 1, 2, 3, 4)
    shuffle(shuffled)

    # Verify: contains same elements (sum should be preserved)
    var shuffled_sum = 0
    for i in range(len(shuffled)):
        shuffled_sum += shuffled[i]

    assert_equal(shuffled_sum, original_sum, "Shuffled array should have same sum (same elements)")


fn test_random_shuffle() raises:
    """Test in-place shuffle of array."""
    set_seed(42)
    var array = List[Int](0, 1, 2, 3, 4)

    # Shuffle in-place
    shuffle(array)

    # Verify: contains same elements (sum preserved)
    var shuffled_sum = 0
    for i in range(len(array)):
        shuffled_sum += array[i]

    assert_equal(shuffled_sum, 10, "Shuffled array should contain same elements (sum=10)")


# ============================================================================
# Test Seed Context Manager
# ============================================================================


fn test_temporary_seed_context() raises:
    """Test using seed as context manager (temporary seed change)."""
    # TODO(#44): Implement when seed context manager exists
    # Set seed to 42
    # Generate a
    # with seed(123):
    #     Generate b (with seed 123)
    # Generate c (back to seed 42 state)
    # Verify: c follows a's sequence (seed 42 restored)
    pass


fn test_nested_seed_contexts() raises:
    """Test nested seed contexts restore properly."""
    # TODO(#44): Implement when seed context manager exists
    # Set seed to 42
    # with seed(100):
    #     Generate a
    #     with seed(200):
    #         Generate b
    #     Generate c (should use seed 100)
    # Generate d (should use seed 42)
    # Verify: contexts restore correctly
    pass


# ============================================================================
# Integration Tests
# ============================================================================


fn test_reproducible_full_workflow() raises:
    """Test entire ML workflow is reproducible with same seed."""
    # TODO(#44): Implement when full workflow exists
    # Set seed to 42
    # Initialize model
    # Load and shuffle data
    # Train for 10 epochs
    # Record final metrics
    # Reset seed to 42
    # Repeat entire workflow
    # Verify: all metrics match exactly
    pass


fn main() raises:
    """Run all tests."""
    test_set_global_seed()
    test_seed_affects_all_generators()
    test_different_seeds_produce_different_sequences()
    test_seed_zero()
    test_seed_max_value()
    test_save_random_state()
    test_restore_random_state()
    test_state_roundtrip()
    test_save_multiple_states()
    test_reproducible_training()
    test_reproducible_data_augmentation()
    test_reproducible_weight_initialization()
    test_reproducible_batch_sampling()
    test_sync_with_external_library()
    test_restore_external_state()
    test_seed_thread_local()
    test_seed_global_across_threads()
    test_negative_seed()
    test_seed_type_validation()
    test_seed_overflow()
    test_uniform_distribution()
    test_normal_distribution()
    test_integer_range()
    test_randomness_independence()
    test_randn_shape()
    test_randint_bounds()
    test_random_choice()
    test_random_permutation()
    test_random_shuffle()
    test_temporary_seed_context()
    test_nested_seed_contexts()
    test_reproducible_full_workflow()
