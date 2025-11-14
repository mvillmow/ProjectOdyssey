"""Unit tests for Early Stopping Callback.

Tests cover:
- Monitoring validation metrics
- Stopping when no improvement
- Patience parameter
- Restoring best weights

Following TDD principles - these tests define the expected API
for implementation in Issue #34.
"""

from tests.shared.conftest import (
    assert_true,
    assert_false,
    assert_equal,
    assert_almost_equal,
    TestFixtures,
)


# ============================================================================
# Early Stopping Core Tests
# ============================================================================


fn test_early_stopping_initialization() raises:
    """Test EarlyStopping callback initialization.

    API Contract:
        EarlyStopping(
            monitor: String = "val_loss",
            patience: Int = 5,
            min_delta: Float32 = 0.0,
            restore_best_weights: Bool = True
        )
        - monitor: Metric to monitor
        - patience: Number of epochs with no improvement before stopping
        - min_delta: Minimum change to qualify as improvement
        - restore_best_weights: Restore model to best weights when stopping
    """
    from shared.training.stubs import MockEarlyStopping

    var early_stop = MockEarlyStopping(
        monitor="val_loss", patience=5, min_delta=0.001
    )

    Verify parameters
    assert_equal(early_stop.monitor, "val_loss")
    assert_equal(early_stop.patience, 5)
    assert_almost_equal(early_stop.min_delta, 0.001)


fn test_early_stopping_triggers_after_patience() raises:
    """Test EarlyStopping stops training after patience epochs.

    API Contract:
        After patience epochs without improvement:
        - Set stop_training flag to True
        - Trainer should check this flag and stop

    This is a CRITICAL test for early stopping behavior.
    """
    from shared.training.stubs import MockEarlyStopping
    from shared.training.base import TrainingState

    var early_stop = MockEarlyStopping(monitor="val_loss", patience=3)

    Initialize state
    var state = TrainingState(epoch=1, learning_rate=0.1)

    Initial best: 0.5
    state.metrics["val_loss"] = 0.5
    _ = early_stop.on_epoch_end(state)
    assert_false(early_stop.should_stop())

    No improvement for epoch 2
    state.epoch = 2
    state.metrics["val_loss"] = 0.6
    _ = early_stop.on_epoch_end(state)
    assert_false(early_stop.should_stop())

    No improvement for epoch 3
    state.epoch = 3
    state.metrics["val_loss"] = 0.6
    _ = early_stop.on_epoch_end(state)
    assert_false(early_stop.should_stop())

    No improvement for epoch 4 - patience exhausted
    state.epoch = 4
    state.metrics["val_loss"] = 0.6
    _ = early_stop.on_epoch_end(state)
    assert_true(early_stop.should_stop())  # Patience exhausted


fn test_early_stopping_resets_patience_on_improvement() raises:
    """Test EarlyStopping resets patience counter when metric improves.

    API Contract:
        When monitored metric improves:
        - Reset patience counter to 0
        - Update best value
        - Continue training
    """
    from shared.training.stubs import MockEarlyStopping
    from shared.training.base import TrainingState

    var early_stop = MockEarlyStopping(monitor="val_loss", patience=3)
    var state = TrainingState(epoch=1, learning_rate=0.1)

    Initial: 0.5
    state.metrics["val_loss"] = 0.5
    _ = early_stop.on_epoch_end(state)

    No improvement for 2 epochs
    state.epoch = 2
    state.metrics["val_loss"] = 0.6
    _ = early_stop.on_epoch_end(state)
    state.epoch = 3
    _ = early_stop.on_epoch_end(state)

    Improvement! Reset patience
    state.epoch = 4
    state.metrics["val_loss"] = 0.4
    _ = early_stop.on_epoch_end(state)
    assert_false(early_stop.should_stop())

    Verify wait count was reset (counter should be 0 after improvement)
    assert_equal(early_stop.wait_count, 0)


# ============================================================================
# Min Delta Tests
# ============================================================================


fn test_early_stopping_min_delta() raises:
    """Test EarlyStopping min_delta for improvement threshold.

    API Contract:
        Improvement is counted only if:
        |new_value - best_value| > min_delta

        Small improvements below threshold don't reset patience.
    """
    # TODO(#34): Implement when EarlyStopping is available
    var early_stop = EarlyStopping(
        monitor="val_loss",
        patience=2,
        min_delta=0.01
    )
    #
    # Initial: 0.5
    early_stop.on_epoch_end(1, {"val_loss": 0.5})
    #
    # Small improvement (0.495, delta=0.005 < 0.01): NOT counted
    early_stop.on_epoch_end(2, {"val_loss": 0.495})
    assert_false(early_stop.should_stop())
    #
    # Another small non-improvement
    early_stop.on_epoch_end(3, {"val_loss": 0.496})
    assert_false(early_stop.should_stop())
    #
    # Patience exhausted (2 epochs without significant improvement)
    early_stop.on_epoch_end(4, {"val_loss": 0.496})
    assert_true(early_stop.should_stop())


fn test_early_stopping_min_delta_large_improvement() raises:
    """Test EarlyStopping counts large improvements above min_delta.

    API Contract:
        Improvement > min_delta resets patience.
    """
    # TODO(#34): Implement when EarlyStopping is available
    var early_stop = EarlyStopping(
        monitor="val_loss",
        patience=2,
        min_delta=0.01
    )
    #
    # Initial: 0.5
    early_stop.on_epoch_end(1, {"val_loss": 0.5})
    #
    # Large improvement (0.48, delta=0.02 > 0.01): Counted
    early_stop.on_epoch_end(2, {"val_loss": 0.48})
    assert_false(early_stop.should_stop())
    #
    # Can continue for another patience epochs
    early_stop.on_epoch_end(3, {"val_loss": 0.49})
    early_stop.on_epoch_end(4, {"val_loss": 0.49})
    assert_false(early_stop.should_stop())


# ============================================================================
# Restore Best Weights Tests
# ============================================================================


fn test_early_stopping_restore_best_weights() raises:
    """Test EarlyStopping restores model to best weights when stopping.

    API Contract:
        With restore_best_weights=True:
        - Track best model weights during training
        - When stopping, restore model to best weights
        - Model should have best performance, not final performance

    This is CRITICAL for getting best model at early stopping.
    """
    # TODO(#34): Implement when EarlyStopping is available
    var model = create_simple_model()
    var early_stop = EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    )
    #
    # Epoch 1: val_loss = 0.5 (best so far)
    early_stop.on_epoch_end(1, {"val_loss": 0.5}, model=model)
    var best_weights = model.get_weights().copy()
    #
    # Epoch 2: val_loss = 0.4 (new best)
    train_one_epoch(model)  # Weights change
    early_stop.on_epoch_end(2, {"val_loss": 0.4}, model=model)
    var new_best_weights = model.get_weights().copy()
    #
    # Epochs 3-4: No improvement, weights continue changing
    train_one_epoch(model)
    early_stop.on_epoch_end(3, {"val_loss": 0.5}, model=model)
    train_one_epoch(model)
    early_stop.on_epoch_end(4, {"val_loss": 0.5}, model=model)
    #
    # Epoch 5: Patience exhausted, should restore best weights
    var final_weights_before_restore = model.get_weights().copy()
    early_stop.on_epoch_end(5, {"val_loss": 0.5}, model=model)
    #
    # Weights should be restored to epoch 2 (best)
    var restored_weights = model.get_weights()
    assert_tensor_equal(restored_weights, new_best_weights)
    assert_not_equal_tensor(restored_weights, final_weights_before_restore)


fn test_early_stopping_no_restore() raises:
    """Test EarlyStopping without restoring best weights.

    API Contract:
        With restore_best_weights=False:
        - Don't track best weights
        - When stopping, keep current weights
    """
    # TODO(#34): Implement when EarlyStopping is available
    var model = create_simple_model()
    var early_stop = EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=False
    )
    #
    # Train for several epochs
    for epoch in range(1, 6):
        train_one_epoch(model)
        early_stop.on_epoch_end(epoch, {"val_loss": 0.5}, model=model)
    #
    # Get final weights
    var final_weights = model.get_weights()
    #
    # Trigger early stopping
    early_stop.on_epoch_end(6, {"val_loss": 0.5}, model=model)
    #
    # Weights should be unchanged (not restored)
    var weights_after_stop = model.get_weights()
    assert_tensor_equal(weights_after_stop, final_weights)


# ============================================================================
# Monitor Metric Tests
# ============================================================================


fn test_early_stopping_monitor_accuracy() raises:
    """Test EarlyStopping monitoring accuracy (higher is better).

    API Contract:
        For metrics where higher is better (accuracy, F1):
        - mode="max" or auto-detect from metric name
        - Improvement = new_value > best_value
    """
    # TODO(#34): Implement when EarlyStopping is available
    var early_stop = EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        mode="max"  # Maximize accuracy
    )
    #
    # Initial: 0.5
    early_stop.on_epoch_end(1, {"val_accuracy": 0.5})
    #
    # Improvement: 0.6 > 0.5
    early_stop.on_epoch_end(2, {"val_accuracy": 0.6})
    assert_false(early_stop.should_stop())
    #
    # No improvement: 0.5 < 0.6
    early_stop.on_epoch_end(3, {"val_accuracy": 0.5})
    early_stop.on_epoch_end(4, {"val_accuracy": 0.5})
    early_stop.on_epoch_end(5, {"val_accuracy": 0.5})
    assert_false(early_stop.should_stop())
    #
    # Patience exhausted
    early_stop.on_epoch_end(6, {"val_accuracy": 0.5})
    assert_true(early_stop.should_stop())


# ============================================================================
# Integration with Trainer Tests
# ============================================================================


fn test_early_stopping_integration_with_trainer() raises:
    """Test EarlyStopping integrates with training loop.

    API Contract:
        Trainer should:
        - Call early_stop.on_epoch_end() after each epoch
        - Check early_stop.should_stop()
        - Break training loop if True
    """
    # TODO(#34): Implement when Trainer and EarlyStopping are available
    var model = create_simple_model()
    var optimizer = SGD(learning_rate=0.1)
    var early_stop = EarlyStopping(monitor="val_loss", patience=3)
    #
    var trainer = Trainer(model, optimizer, loss_fn, callbacks=[early_stop])
    #
    # Train for max 100 epochs, but should stop early
    var results = trainer.train(
        epochs=100,
        train_loader=create_plateaued_dataset(),  # Loss stops improving
        val_loader=create_plateaued_dataset()
    )
    #
    # Should stop before 100 epochs
    var actual_epochs = len(results["train_loss"])
    assert_less(actual_epochs, 100)
    #
    # Should stop around when patience exhausted
    # (exact number depends on when loss plateaus)


# ============================================================================
# Edge Cases
# ============================================================================


fn test_early_stopping_zero_patience() raises:
    """Test EarlyStopping with patience=0.

    API Contract:
        patience=0 should:
        - Stop after first epoch with no improvement
        - Effectively requires improvement every epoch
    """
    # TODO(#34): Implement when EarlyStopping is available
    var early_stop = EarlyStopping(monitor="val_loss", patience=0)
    #
    # Initial: 0.5
    early_stop.on_epoch_end(1, {"val_loss": 0.5})
    #
    # No improvement immediately triggers stop
    early_stop.on_epoch_end(2, {"val_loss": 0.5})
    assert_true(early_stop.should_stop())


fn test_early_stopping_missing_monitored_metric() raises:
    """Test EarlyStopping handles missing monitored metric.

    API Contract:
        If monitored metric not in logs:
        - Raise error OR
        - Skip epoch with warning
    """
    # TODO(#34): Implement error handling when EarlyStopping is available
    var early_stop = EarlyStopping(monitor="val_loss", patience=3)
    #
    # Logs missing val_loss
    try:
        early_stop.on_epoch_end(1, {"train_loss": 0.5})
        # Should raise error or handle gracefully
    except Error:
        pass  # Expected


# ============================================================================
# Test Main
# ============================================================================


fn main() raises:
    """Run all early stopping callback tests."""
    print("Running early stopping core tests...")
    test_early_stopping_initialization()
    test_early_stopping_triggers_after_patience()
    test_early_stopping_resets_patience_on_improvement()

    print("Running min_delta tests...")
    test_early_stopping_min_delta()
    test_early_stopping_min_delta_large_improvement()

    print("Running restore best weights tests...")
    test_early_stopping_restore_best_weights()
    test_early_stopping_no_restore()

    print("Running monitor metric tests...")
    test_early_stopping_monitor_accuracy()

    print("Running trainer integration tests...")
    test_early_stopping_integration_with_trainer()

    print("Running edge cases...")
    test_early_stopping_zero_patience()
    test_early_stopping_missing_monitored_metric()

    print("\nAll early stopping callback tests passed! ✓")
