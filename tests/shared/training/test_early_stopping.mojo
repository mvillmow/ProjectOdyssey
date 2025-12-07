"""Unit tests for Early Stopping Callback.

Tests cover:
- Monitoring validation metrics
- Stopping when no improvement
- Patience parameter
- Min delta threshold
- Both minimize (loss) and maximize (accuracy) modes

All tests use the real EarlyStopping implementation.
"""

from tests.shared.conftest import (
    assert_true,
    assert_false,
    assert_equal,
    assert_almost_equal,
    TestFixtures,
)
from shared.training.callbacks import EarlyStopping
from shared.training.base import TrainingState


# ============================================================================
# Early Stopping Core Tests
# ============================================================================


fn test_early_stopping_initialization() raises:
    """Test EarlyStopping callback initialization with parameters."""
    var early_stop = EarlyStopping(
        monitor="val_loss", patience=5, min_delta=0.001, mode="min"
    )

    # Verify parameters
    assert_equal(early_stop.monitor, "val_loss")
    assert_equal(early_stop.patience, 5)
    assert_almost_equal(early_stop.min_delta, 0.001)
    assert_equal(early_stop.mode, "min")


fn test_early_stopping_triggers_after_patience() raises:
    """Test EarlyStopping stops training after patience epochs without improvement.
    """
    var early_stop = EarlyStopping(monitor="val_loss", patience=3, mode="min")
    var state = TrainingState(epoch=1, learning_rate=0.1)

    # Initial best: 0.5
    state.metrics["val_loss"] = 0.5
    _ = early_stop.on_epoch_end(state)
    assert_false(early_stop.should_stop())

    # No improvement for epoch 2
    state.epoch = 2
    state.metrics["val_loss"] = 0.6
    _ = early_stop.on_epoch_end(state)
    assert_false(early_stop.should_stop())

    # No improvement for epoch 3
    state.epoch = 3
    state.metrics["val_loss"] = 0.6
    _ = early_stop.on_epoch_end(state)
    assert_false(early_stop.should_stop())

    # No improvement for epoch 4 - patience exhausted
    state.epoch = 4
    state.metrics["val_loss"] = 0.6
    _ = early_stop.on_epoch_end(state)
    assert_true(early_stop.should_stop())  # Patience exhausted


fn test_early_stopping_resets_patience_on_improvement() raises:
    """Test EarlyStopping resets patience counter when metric improves."""
    var early_stop = EarlyStopping(monitor="val_loss", patience=3, mode="min")
    var state = TrainingState(epoch=1, learning_rate=0.1)

    # Initial: 0.5
    state.metrics["val_loss"] = 0.5
    _ = early_stop.on_epoch_end(state)

    # No improvement for 2 epochs
    state.epoch = 2
    state.metrics["val_loss"] = 0.6
    _ = early_stop.on_epoch_end(state)
    state.epoch = 3
    _ = early_stop.on_epoch_end(state)

    # Improvement! Reset patience
    state.epoch = 4
    state.metrics["val_loss"] = 0.4
    _ = early_stop.on_epoch_end(state)
    assert_false(early_stop.should_stop())

    # Verify wait count was reset
    assert_equal(early_stop.wait_count, 0)


# ============================================================================
# Min Delta Tests
# ============================================================================


fn test_early_stopping_min_delta() raises:
    """Test EarlyStopping min_delta for improvement threshold.

    Small improvements below threshold don't reset patience.
    """
    var early_stop = EarlyStopping(
        monitor="val_loss", patience=2, min_delta=0.01, mode="min"
    )
    var state = TrainingState(epoch=1, learning_rate=0.1)

    # Initial: 0.5
    state.metrics["val_loss"] = 0.5
    _ = early_stop.on_epoch_end(state)

    # Small improvement (0.495, delta=0.005 < 0.01): NOT counted
    state.epoch = 2
    state.metrics["val_loss"] = 0.495
    _ = early_stop.on_epoch_end(state)
    assert_false(early_stop.should_stop())

    # Another small non-improvement - Patience exhausted (2 epochs without significant improvement)
    state.epoch = 3
    state.metrics["val_loss"] = 0.496
    _ = early_stop.on_epoch_end(state)
    assert_true(early_stop.should_stop())  # wait_count=2 >= patience=2


fn test_early_stopping_min_delta_large_improvement() raises:
    """Test EarlyStopping counts large improvements above min_delta."""
    var early_stop = EarlyStopping(
        monitor="val_loss", patience=2, min_delta=0.01, mode="min"
    )
    var state = TrainingState(epoch=1, learning_rate=0.1)

    # Initial: 0.5
    state.metrics["val_loss"] = 0.5
    _ = early_stop.on_epoch_end(state)

    # Large improvement (0.48, delta=0.02 > 0.01): Counted
    state.epoch = 2
    state.metrics["val_loss"] = 0.48
    _ = early_stop.on_epoch_end(state)
    assert_false(early_stop.should_stop())
    assert_equal(early_stop.wait_count, 0)  # Reset

    # No improvement for 1 epoch - within patience
    state.epoch = 3
    state.metrics["val_loss"] = 0.49
    _ = early_stop.on_epoch_end(state)
    assert_false(early_stop.should_stop())  # wait_count=1 < patience=2

    # No improvement for 2 epochs - patience exhausted
    state.epoch = 4
    state.metrics["val_loss"] = 0.49
    _ = early_stop.on_epoch_end(state)
    assert_true(early_stop.should_stop())  # wait_count=2 >= patience=2


# ============================================================================
# Monitor Metric Tests
# ============================================================================


fn test_early_stopping_monitor_accuracy() raises:
    """Test EarlyStopping monitoring accuracy (higher is better) with mode='max'.
    """
    var early_stop = EarlyStopping(
        monitor="val_accuracy", patience=3, mode="max"
    )
    var state = TrainingState(epoch=1, learning_rate=0.1)

    # Initial: 0.5
    state.metrics["val_accuracy"] = 0.5
    _ = early_stop.on_epoch_end(state)

    # Improvement: 0.6 > 0.5
    state.epoch = 2
    state.metrics["val_accuracy"] = 0.6
    _ = early_stop.on_epoch_end(state)
    assert_false(early_stop.should_stop())
    assert_equal(early_stop.wait_count, 0)

    # No improvement: 0.5 < 0.6 (epochs 3, 4)
    state.epoch = 3
    state.metrics["val_accuracy"] = 0.5
    _ = early_stop.on_epoch_end(state)
    state.epoch = 4
    state.metrics["val_accuracy"] = 0.5
    _ = early_stop.on_epoch_end(state)
    assert_false(early_stop.should_stop())  # wait_count=2 < patience=3

    # Patience exhausted after 3 epochs without improvement
    state.epoch = 5
    state.metrics["val_accuracy"] = 0.5
    _ = early_stop.on_epoch_end(state)
    assert_true(early_stop.should_stop())  # wait_count=3 >= patience=3


fn test_early_stopping_mode_min() raises:
    """Test EarlyStopping with mode='min' for loss minimization."""
    var early_stop = EarlyStopping(
        monitor="val_loss", patience=2, min_delta=0.0, mode="min"
    )
    var state = TrainingState(epoch=1, learning_rate=0.1)

    # Initial: 1.0
    state.metrics["val_loss"] = 1.0
    _ = early_stop.on_epoch_end(state)

    # Improvement: 0.8 < 1.0
    state.epoch = 2
    state.metrics["val_loss"] = 0.8
    _ = early_stop.on_epoch_end(state)
    assert_equal(early_stop.wait_count, 0)

    # Improvement: 0.6 < 0.8
    state.epoch = 3
    state.metrics["val_loss"] = 0.6
    _ = early_stop.on_epoch_end(state)
    assert_equal(early_stop.wait_count, 0)
    assert_false(early_stop.should_stop())


fn test_early_stopping_mode_max() raises:
    """Test EarlyStopping with mode='max' for accuracy maximization."""
    var early_stop = EarlyStopping(
        monitor="val_accuracy", patience=2, min_delta=0.0, mode="max"
    )
    var state = TrainingState(epoch=1, learning_rate=0.1)

    # Initial: 0.6
    state.metrics["val_accuracy"] = 0.6
    _ = early_stop.on_epoch_end(state)

    # Improvement: 0.7 > 0.6
    state.epoch = 2
    state.metrics["val_accuracy"] = 0.7
    _ = early_stop.on_epoch_end(state)
    assert_equal(early_stop.wait_count, 0)

    # Improvement: 0.8 > 0.7
    state.epoch = 3
    state.metrics["val_accuracy"] = 0.8
    _ = early_stop.on_epoch_end(state)
    assert_equal(early_stop.wait_count, 0)
    assert_false(early_stop.should_stop())


# ============================================================================
# Edge Cases
# ============================================================================


fn test_early_stopping_zero_patience() raises:
    """Test EarlyStopping with patience=0 stops after first non-improvement."""
    var early_stop = EarlyStopping(monitor="val_loss", patience=0, mode="min")
    var state = TrainingState(epoch=1, learning_rate=0.1)

    # Initial: 0.5
    state.metrics["val_loss"] = 0.5
    _ = early_stop.on_epoch_end(state)

    # No improvement immediately triggers stop
    state.epoch = 2
    state.metrics["val_loss"] = 0.5
    _ = early_stop.on_epoch_end(state)
    assert_true(early_stop.should_stop())


fn test_early_stopping_missing_monitored_metric() raises:
    """Test EarlyStopping handles missing monitored metric gracefully."""
    var early_stop = EarlyStopping(monitor="val_loss", patience=3, mode="min")
    var state = TrainingState(epoch=1, learning_rate=0.1)

    # Metrics missing val_loss - should continue without error
    state.metrics["train_loss"] = 0.5
    _ = early_stop.on_epoch_end(state)

    # Should not stop (metric doesn't exist)
    assert_false(early_stop.should_stop())


fn test_early_stopping_on_train_begin_resets() raises:
    """Test on_train_begin resets all state."""
    var early_stop = EarlyStopping(monitor="val_loss", patience=2, mode="min")
    var state = TrainingState(epoch=1, learning_rate=0.1)

    # Simulate some training
    state.metrics["val_loss"] = 0.5
    _ = early_stop.on_epoch_end(state)
    state.epoch = 2
    state.metrics["val_loss"] = 0.6
    _ = early_stop.on_epoch_end(state)

    # Reset with on_train_begin
    _ = early_stop.on_train_begin(state)

    # State should be reset
    assert_equal(early_stop.wait_count, 0)
    assert_false(early_stop.stopped)
    assert_almost_equal(
        early_stop.best_value, 1e9
    )  # Reset to initial for min mode


# ============================================================================
# Best Value Tracking Tests
# ============================================================================


fn test_early_stopping_tracks_best_value() raises:
    """Test EarlyStopping correctly tracks best value seen."""
    var early_stop = EarlyStopping(monitor="val_loss", patience=3, mode="min")
    var state = TrainingState(epoch=1, learning_rate=0.1)

    # Initial: 0.5
    state.metrics["val_loss"] = 0.5
    _ = early_stop.on_epoch_end(state)
    assert_almost_equal(early_stop.best_value, 0.5)

    # Worse: 0.6 (best stays 0.5)
    state.epoch = 2
    state.metrics["val_loss"] = 0.6
    _ = early_stop.on_epoch_end(state)
    assert_almost_equal(early_stop.best_value, 0.5)

    # Better: 0.3 (best updates)
    state.epoch = 3
    state.metrics["val_loss"] = 0.3
    _ = early_stop.on_epoch_end(state)
    assert_almost_equal(early_stop.best_value, 0.3)


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

    print("Running monitor metric tests...")
    test_early_stopping_monitor_accuracy()
    test_early_stopping_mode_min()
    test_early_stopping_mode_max()

    print("Running edge cases...")
    test_early_stopping_zero_patience()
    test_early_stopping_missing_monitored_metric()
    test_early_stopping_on_train_begin_resets()

    print("Running best value tracking tests...")
    test_early_stopping_tracks_best_value()

    print("\nAll early stopping callback tests passed! ✓")
