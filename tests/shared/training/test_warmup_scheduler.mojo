"""Unit tests for Warmup Learning Rate Scheduler.

Tests cover:
- Gradual LR increase from low to target value
- Linear warmup strategy
- Integration with other schedulers
- Warmup period configuration

Following TDD principles - these tests define the expected API
for implementation in Issue #34.
"""

from tests.shared.conftest import (
    assert_true,
    assert_equal,
    assert_almost_equal,
    assert_greater,
    assert_less,
    TestFixtures,
)


# ============================================================================
# Warmup Scheduler Core Tests
# ============================================================================


fn test_warmup_scheduler_initialization() raises:
    """Test LinearWarmup scheduler initialization.

    API Contract:
        LinearWarmup(
            optimizer: Optimizer,
            warmup_epochs: Int,
            start_lr: Float32 = 0.0
        )
        - warmup_epochs: Number of epochs for warmup
        - start_lr: Initial learning rate (default 0.0)
        - Target LR is optimizer's current LR
    """
    from shared.training.stubs import MockWarmupScheduler

    var scheduler = MockWarmupScheduler(base_lr=0.1, warmup_epochs=5)

    Verify parameters
    assert_equal(scheduler.warmup_epochs, 5)
    assert_almost_equal(scheduler.base_lr, 0.1)


fn test_warmup_scheduler_linear_increase() raises:
    """Test LinearWarmup increases LR linearly over warmup period.

    API Contract:
        During warmup (epoch < warmup_epochs):
        lr = start_lr + (target_lr - start_lr) * (epoch / warmup_epochs)

        After warmup (epoch >= warmup_epochs):
        lr = target_lr (constant)

    This is a CRITICAL test for warmup scheduler behavior.
    """
    from shared.training.stubs import MockWarmupScheduler

    var scheduler = MockWarmupScheduler(base_lr=1.0, warmup_epochs=10)

    Initial: epoch 0
    var lr0 = scheduler.get_lr(epoch=0)
    assert_almost_equal(lr0, 0.0)

    Epoch 1: lr = 1.0 * (1/10) = 0.1
    var lr1 = scheduler.get_lr(epoch=1)
    assert_almost_equal(lr1, 0.1)

    Epoch 5: lr = 1.0 * (5/10) = 0.5
    var lr5 = scheduler.get_lr(epoch=5)
    assert_almost_equal(lr5, 0.5)

    Epoch 10: lr = 1.0 (target reached)
    var lr10 = scheduler.get_lr(epoch=10)
    assert_almost_equal(lr10, 1.0)

    Epoch 11+: lr remains at target
    var lr11 = scheduler.get_lr(epoch=11)
    assert_almost_equal(lr11, 1.0)


fn test_warmup_scheduler_reaches_target() raises:
    """Test LinearWarmup reaches target LR after warmup period.

    API Contract:
        After warmup_epochs:
        - LR should equal target_lr
        - LR should remain at target_lr
    """
    from shared.training.stubs import MockWarmupScheduler

    var scheduler = MockWarmupScheduler(base_lr=0.1, warmup_epochs=5)

    After warmup, should be at target
    var lr_after_warmup = scheduler.get_lr(epoch=5)
    assert_almost_equal(lr_after_warmup, 0.1)

    Continue stepping - should remain at target
    var lr_later = scheduler.get_lr(epoch=20)
    assert_almost_equal(lr_later, 0.1)


# ============================================================================
# Start LR Tests
# ============================================================================


fn test_warmup_scheduler_different_start_lrs() raises:
    """Test LinearWarmup with different start_lr values.

    API Contract:
        start_lr can be:
        - 0.0 (common: start from zero)
        - Small value (e.g., 1e-6 for stability)
        - Fraction of target_lr (e.g., 0.1 * target_lr)
    """
    # TODO(#34): Implement when LinearWarmup is available
    var target_lr = Float32(1.0)
    #
    # Test start_lr = 0.0
    var optimizer1 = SGD(learning_rate=target_lr)
    var scheduler1 = LinearWarmup(optimizer1, warmup_epochs=10, start_lr=0.0)
    scheduler1.step(0)
    assert_almost_equal(optimizer1.learning_rate, 0.0)
    #
    # Test start_lr = 0.1
    var optimizer2 = SGD(learning_rate=target_lr)
    var scheduler2 = LinearWarmup(optimizer2, warmup_epochs=10, start_lr=0.1)
    scheduler2.step(0)
    assert_almost_equal(optimizer2.learning_rate, 0.1)


fn test_warmup_scheduler_start_lr_equals_target() raises:
    """Test LinearWarmup when start_lr equals target_lr.

    API Contract:
        When start_lr = target_lr:
        - LR should remain constant (no warmup needed)
    """
    # TODO(#34): Implement when LinearWarmup is available
    var optimizer = SGD(learning_rate=0.1)
    var scheduler = LinearWarmup(optimizer, warmup_epochs=10, start_lr=0.1)
    #
    # LR should remain constant
    for epoch in range(15):
        scheduler.step(epoch)
        assert_almost_equal(optimizer.learning_rate, 0.1)


# ============================================================================
# Warmup Period Tests
# ============================================================================


fn test_warmup_scheduler_different_warmup_periods() raises:
    """Test LinearWarmup with different warmup_epochs values.

    API Contract:
        warmup_epochs determines warmup speed:
        - Small warmup_epochs: Fast warmup
        - Large warmup_epochs: Slow warmup
    """
    # TODO(#34): Implement when LinearWarmup is available
    # Fast warmup (2 epochs)
    var optimizer1 = SGD(learning_rate=1.0)
    var scheduler1 = LinearWarmup(optimizer1, warmup_epochs=2, start_lr=0.0)
    #
    scheduler1.step(1)
    assert_almost_equal(optimizer1.learning_rate, 0.5)  # Halfway after 1 epoch
    #
    scheduler1.step(2)
    assert_almost_equal(optimizer1.learning_rate, 1.0)  # Target after 2 epochs
    #
    # Slow warmup (100 epochs)
    var optimizer2 = SGD(learning_rate=1.0)
    var scheduler2 = LinearWarmup(optimizer2, warmup_epochs=100, start_lr=0.0)
    #
    scheduler2.step(1)
    assert_almost_equal(optimizer2.learning_rate, 0.01)  # Small increase


fn test_warmup_scheduler_single_epoch_warmup() raises:
    """Test LinearWarmup with warmup_epochs=1.

    API Contract:
        warmup_epochs=1 should:
        - Epoch 0: start_lr
        - Epoch 1+: target_lr
    """
    # TODO(#34): Implement when LinearWarmup is available
    var optimizer = SGD(learning_rate=1.0)
    var scheduler = LinearWarmup(optimizer, warmup_epochs=1, start_lr=0.0)
    #
    scheduler.step(0)
    assert_almost_equal(optimizer.learning_rate, 0.0)
    #
    scheduler.step(1)
    assert_almost_equal(optimizer.learning_rate, 1.0)


# ============================================================================
# Integration with Other Schedulers
# ============================================================================


fn test_warmup_scheduler_chained_with_step_lr() raises:
    """Test LinearWarmup chained with StepLR.

    API Contract:
        Common pattern:
        1. Warmup for N epochs
        2. Then step decay

        Can be implemented as:
        - Separate schedulers switched at epoch N
        - Or composite scheduler
    """
    # TODO(#34): Implement when both schedulers available
    var optimizer = SGD(learning_rate=1.0)
    var warmup = LinearWarmup(optimizer, warmup_epochs=5, start_lr=0.1)
    var step_lr = StepLR(optimizer, step_size=10, gamma=0.1)
    #
    # Epochs 0-4: Warmup
    for epoch in range(5):
        warmup.step(epoch)
        # Don't step step_lr during warmup
    #
    # Verify reached target
    assert_almost_equal(optimizer.learning_rate, 1.0)
    #
    # Epochs 5+: Step decay
    for epoch in range(5, 20):
        step_lr.step(epoch - 5)  # Adjust epoch for step_lr
    #
    # At epoch 15 (step_lr epoch 10): LR should be 0.1
    assert_almost_equal(optimizer.learning_rate, 0.1)


fn test_warmup_scheduler_chained_with_cosine() raises:
    """Test LinearWarmup chained with CosineAnnealingLR.

    API Contract:
        Common pattern in modern training:
        1. Warmup for N epochs
        2. Cosine annealing after warmup
    """
    # TODO(#34): Implement when both schedulers available
    var optimizer = SGD(learning_rate=1.0)
    var warmup = LinearWarmup(optimizer, warmup_epochs=5, start_lr=0.0)
    var cosine = CosineAnnealingLR(optimizer, T_max=45, eta_min=0.0)
    #
    # Epochs 0-4: Warmup
    for epoch in range(5):
        warmup.step(epoch)
    #
    # Epochs 5-49: Cosine annealing
    for epoch in range(45):
        cosine.step(epoch)
    #
    # At end of cosine: LR should be near eta_min
    assert_almost_equal(optimizer.learning_rate, 0.0, tolerance=1e-5)


# ============================================================================
# Optimizer Integration Tests
# ============================================================================


fn test_warmup_scheduler_stabilizes_early_training() raises:
    """Test LinearWarmup helps stabilize early training.

    API Contract:
        Starting with low LR and gradually increasing should:
        - Prevent early divergence
        - Allow model to learn stable features first
        - Then ramp up to full learning rate
    """
    # TODO(#34): Implement when warmup and training available
    This is more of an integration test with actual training


# ============================================================================
# Edge Cases
# ============================================================================


fn test_warmup_scheduler_zero_warmup_epochs() raises:
    """Test LinearWarmup with warmup_epochs=0.

    API Contract:
        warmup_epochs=0 should:
        - Immediately set LR to target
        - Or raise error (design choice)
    """
    # TODO(#34): Implement error handling when LinearWarmup is available
    var optimizer = SGD(learning_rate=1.0)
    #
    # Option 1: Immediate target
    var scheduler = LinearWarmup(optimizer, warmup_epochs=0, start_lr=0.0)
    scheduler.step(0)
    assert_almost_equal(optimizer.learning_rate, 1.0)
    #
    # Option 2: Raise error
    # try:
    #     var scheduler = LinearWarmup(optimizer, warmup_epochs=0)
    #     assert_true(False, "Expected error")
    # except Error:
    #     pass


fn test_warmup_scheduler_negative_warmup_epochs() raises:
    """Test LinearWarmup with negative warmup_epochs raises error.

    API Contract:
        warmup_epochs must be non-negative.
    """
    # TODO(#34): Implement error handling when LinearWarmup is available
    var optimizer = SGD(learning_rate=1.0)
    #
    try:
        var scheduler = LinearWarmup(optimizer, warmup_epochs=-5)
        assert_true(False, "Expected error for negative warmup_epochs")
    except Error:
        pass  # Expected


# ============================================================================
# Property-Based Tests
# ============================================================================


fn test_warmup_scheduler_property_monotonic_increase() raises:
    """Property: LR should monotonically increase during warmup.

    During warmup period, LR should never decrease.
    """
    # TODO(#34): Implement when LinearWarmup is available
    var optimizer = SGD(learning_rate=1.0)
    var scheduler = LinearWarmup(optimizer, warmup_epochs=20, start_lr=0.0)
    #
    var previous_lr = Float32(0.0)
    for epoch in range(21):
        scheduler.step(epoch)
        var current_lr = optimizer.learning_rate
    #
        # LR should not decrease
        assert_greater_or_equal(current_lr, previous_lr)
        previous_lr = current_lr


fn test_warmup_scheduler_property_linear() raises:
    """Property: LR increase should be perfectly linear.

    Equal epoch increments should produce equal LR increments.
    """
    # TODO(#34): Implement when LinearWarmup is available
    var optimizer = SGD(learning_rate=1.0)
    var scheduler = LinearWarmup(optimizer, warmup_epochs=10, start_lr=0.0)
    #
    var lr_values = List[Float32]()
    for epoch in range(11):
        scheduler.step(epoch)
        lr_values.append(optimizer.learning_rate)
    #
    # Check linear spacing
    var increment = lr_values[1] - lr_values[0]
    for i in range(1, 10):
        var current_increment = lr_values[i+1] - lr_values[i]
        assert_almost_equal(current_increment, increment, tolerance=1e-6)


# ============================================================================
# Test Main
# ============================================================================


fn main() raises:
    """Run all LinearWarmup scheduler tests."""
    print("Running LinearWarmup core tests...")
    test_warmup_scheduler_initialization()
    test_warmup_scheduler_linear_increase()
    test_warmup_scheduler_reaches_target()

    print("Running start_lr tests...")
    test_warmup_scheduler_different_start_lrs()
    test_warmup_scheduler_start_lr_equals_target()

    print("Running warmup period tests...")
    test_warmup_scheduler_different_warmup_periods()
    test_warmup_scheduler_single_epoch_warmup()

    print("Running scheduler chaining tests...")
    test_warmup_scheduler_chained_with_step_lr()
    test_warmup_scheduler_chained_with_cosine()

    print("Running optimizer integration tests...")
    test_warmup_scheduler_stabilizes_early_training()

    print("Running edge cases...")
    test_warmup_scheduler_zero_warmup_epochs()
    test_warmup_scheduler_negative_warmup_epochs()

    print("Running property-based tests...")
    test_warmup_scheduler_property_monotonic_increase()
    test_warmup_scheduler_property_linear()

    print("\nAll LinearWarmup scheduler tests passed! ✓")
