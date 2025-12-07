"""Tests for training callbacks.

Tests cover:
- EarlyStopping callback with min/max modes
- ModelCheckpoint callback with frequency and best-only modes
- LoggingCallback with metric logging
- Callback signal handling
- Metric tracking and state transitions
"""

from testing import assert_true, assert_equal
from collections import Dict
from shared.training.base import (
    Callback,
    CallbackSignal,
    CONTINUE,
    STOP,
    TrainingState,
)
from shared.training.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LoggingCallback,
)


# ============================================================================
# EarlyStopping Tests
# ============================================================================


fn test_early_stopping_min_mode_improves() raises:
    """Test early stopping in min mode when loss improves."""
    var early_stop = EarlyStopping(
        monitor="val_loss", patience=3, min_delta=0.001, mode="min"
    )

    var state = TrainingState(epoch=0, batch=0, learning_rate=0.01)

    # Start training
    var signal = early_stop.on_train_begin(state)
    assert_equal(signal.value, 0, "Should return CONTINUE at train begin")

    # Epoch 0: val_loss = 1.0 (first value, should be improvement)
    state.metrics["val_loss"] = 1.0
    signal = early_stop.on_epoch_end(state)
    assert_equal(signal.value, 0, "Should return CONTINUE on first epoch")
    assert_equal(
        early_stop.wait_count, 0, "Wait count should be 0 after improvement"
    )

    # Epoch 1: val_loss = 0.998 (improved by > 0.001)
    state.epoch = 1
    state.metrics["val_loss"] = 0.998
    signal = early_stop.on_epoch_end(state)
    assert_equal(signal.value, 0, "Should return CONTINUE on improvement")
    assert_equal(early_stop.wait_count, 0, "Wait count should reset to 0")


fn test_early_stopping_min_mode_no_improvement() raises:
    """Test early stopping in min mode when loss doesn't improve enough."""
    var early_stop = EarlyStopping(
        monitor="val_loss", patience=2, min_delta=0.001, mode="min"
    )

    var state = TrainingState(epoch=0, batch=0, learning_rate=0.01)
    early_stop.on_train_begin(state)

    # Epoch 0: val_loss = 1.0
    state.metrics["val_loss"] = 1.0
    early_stop.on_epoch_end(state)

    # Epoch 1: val_loss = 0.9999 (didn't improve by min_delta)
    state.epoch = 1
    state.metrics["val_loss"] = 0.9999
    var signal = early_stop.on_epoch_end(state)
    assert_equal(signal.value, 0, "Should return CONTINUE")
    assert_equal(early_stop.wait_count, 1, "Wait count should be 1")

    # Epoch 2: val_loss = 0.9998 (still no improvement)
    state.epoch = 2
    state.metrics["val_loss"] = 0.9998
    signal = early_stop.on_epoch_end(state)
    assert_equal(signal.value, 1, "Should return STOP after patience exhausted")
    assert_true(early_stop.stopped, "Stopped flag should be True")


fn test_early_stopping_max_mode_improves() raises:
    """Test early stopping in max mode (accuracy) when metric improves."""
    var early_stop = EarlyStopping(
        monitor="val_accuracy", patience=2, min_delta=0.001, mode="max"
    )

    var state = TrainingState(epoch=0, batch=0, learning_rate=0.01)
    early_stop.on_train_begin(state)

    # Epoch 0: accuracy = 0.8
    state.metrics["val_accuracy"] = 0.8
    var signal = early_stop.on_epoch_end(state)
    assert_equal(signal.value, 0, "Should return CONTINUE")
    assert_equal(early_stop.wait_count, 0, "Wait count should be 0")

    # Epoch 1: accuracy = 0.805 (improved by > 0.001)
    state.epoch = 1
    state.metrics["val_accuracy"] = 0.805
    signal = early_stop.on_epoch_end(state)
    assert_equal(signal.value, 0, "Should return CONTINUE")
    assert_equal(early_stop.wait_count, 0, "Wait count should reset")


fn test_early_stopping_max_mode_no_improvement() raises:
    """Test early stopping in max mode when accuracy doesn't improve."""
    var early_stop = EarlyStopping(
        monitor="val_accuracy", patience=1, min_delta=0.001, mode="max"
    )

    var state = TrainingState(epoch=0, batch=0, learning_rate=0.01)
    early_stop.on_train_begin(state)

    # Epoch 0: accuracy = 0.8
    state.metrics["val_accuracy"] = 0.8
    early_stop.on_epoch_end(state)

    # Epoch 1: accuracy = 0.7999 (degraded)
    state.epoch = 1
    state.metrics["val_accuracy"] = 0.7999
    var signal = early_stop.on_epoch_end(state)
    assert_equal(signal.value, 1, "Should return STOP after patience exhausted")
    assert_true(early_stop.stopped, "Stopped flag should be True")


fn test_early_stopping_missing_metric() raises:
    """Test early stopping when monitored metric is missing."""
    var early_stop = EarlyStopping(
        monitor="val_loss", patience=2, min_delta=0.001, mode="min"
    )

    var state = TrainingState(epoch=0, batch=0, learning_rate=0.01)
    early_stop.on_train_begin(state)

    # Epoch has no val_loss metric
    state.metrics["train_loss"] = 0.5
    var signal = early_stop.on_epoch_end(state)
    assert_equal(signal.value, 0, "Should return CONTINUE when metric missing")
    assert_equal(early_stop.wait_count, 0, "Wait count should not change")


fn test_early_stopping_should_stop_method() raises:
    """Test should_stop method."""
    var early_stop = EarlyStopping(monitor="val_loss", patience=1, mode="min")

    assert_true(
        not early_stop.should_stop(), "should_stop should be False initially"
    )

    var state = TrainingState(epoch=0, batch=0, learning_rate=0.01)
    early_stop.on_train_begin(state)

    state.metrics["val_loss"] = 1.0
    early_stop.on_epoch_end(state)

    state.epoch = 1
    state.metrics[
        "val_loss"
    ] = 1.1  # Worse than 1.0 in min mode - no improvement
    early_stop.on_epoch_end(state)

    assert_true(
        early_stop.should_stop(),
        "should_stop should be True after patience exhausted",
    )


# ============================================================================
# ModelCheckpoint Tests
# ============================================================================


fn test_model_checkpoint_save_frequency() raises:
    """Test checkpoint saving at specified frequency."""
    var checkpoint = ModelCheckpoint(
        filepath="checkpoint_{epoch}.pt", save_frequency=2
    )

    var state = TrainingState(epoch=0, batch=0, learning_rate=0.01)

    # Epoch 0: No save (0 % 2 == 0 but save_best_only is off)
    state.epoch = 0
    var signal = checkpoint.on_epoch_end(state)
    assert_equal(signal.value, 0, "Should return CONTINUE")
    assert_equal(checkpoint.get_save_count(), 1, "Should have saved at epoch 0")

    # Epoch 1: No save
    state.epoch = 1
    signal = checkpoint.on_epoch_end(state)
    assert_equal(checkpoint.get_save_count(), 1, "Should not save at epoch 1")

    # Epoch 2: Save
    state.epoch = 2
    signal = checkpoint.on_epoch_end(state)
    assert_equal(checkpoint.get_save_count(), 2, "Should have saved at epoch 2")


fn test_model_checkpoint_save_best_only_min_mode() raises:
    """Test checkpoint saving only when best metric improves (min mode)."""
    var checkpoint = ModelCheckpoint(
        filepath="best_model.pt",
        monitor="val_loss",
        save_best_only=True,
        mode="min",
    )

    var state = TrainingState(epoch=0, batch=0, learning_rate=0.01)

    # Epoch 0: val_loss = 1.0 (first value, should save)
    state.epoch = 0
    state.metrics["val_loss"] = 1.0
    checkpoint.on_epoch_end(state)
    assert_equal(checkpoint.get_save_count(), 1, "Should save on first epoch")

    # Epoch 1: val_loss = 0.9 (improved, should save)
    state.epoch = 1
    state.metrics["val_loss"] = 0.9
    checkpoint.on_epoch_end(state)
    assert_equal(checkpoint.get_save_count(), 2, "Should save on improvement")

    # Epoch 2: val_loss = 0.95 (degraded, should not save)
    state.epoch = 2
    state.metrics["val_loss"] = 0.95
    checkpoint.on_epoch_end(state)
    assert_equal(
        checkpoint.get_save_count(), 2, "Should not save on degradation"
    )


fn test_model_checkpoint_save_best_only_max_mode() raises:
    """Test checkpoint saving only when best metric improves (max mode)."""
    var checkpoint = ModelCheckpoint(
        filepath="best_model.pt",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
    )

    var state = TrainingState(epoch=0, batch=0, learning_rate=0.01)

    # Epoch 0: accuracy = 0.8 (first value, should save)
    state.epoch = 0
    state.metrics["val_accuracy"] = 0.8
    checkpoint.on_epoch_end(state)
    assert_equal(checkpoint.get_save_count(), 1, "Should save on first epoch")

    # Epoch 1: accuracy = 0.85 (improved, should save)
    state.epoch = 1
    state.metrics["val_accuracy"] = 0.85
    checkpoint.on_epoch_end(state)
    assert_equal(checkpoint.get_save_count(), 2, "Should save on improvement")

    # Epoch 2: accuracy = 0.83 (degraded, should not save)
    state.epoch = 2
    state.metrics["val_accuracy"] = 0.83
    checkpoint.on_epoch_end(state)
    assert_equal(
        checkpoint.get_save_count(), 2, "Should not save on degradation"
    )


fn test_model_checkpoint_epoch_placeholder() raises:
    """Test checkpoint path with {epoch} placeholder."""
    var checkpoint = ModelCheckpoint(
        filepath="checkpoints/model_epoch_{epoch}.pt", save_frequency=1
    )

    var state = TrainingState(epoch=5, batch=0, learning_rate=0.01)

    # Should replace {epoch} with 5
    checkpoint.on_epoch_end(state)
    assert_equal(checkpoint.get_save_count(), 1, "Should have saved checkpoint")


fn test_model_checkpoint_error_count() raises:
    """Test error tracking in checkpoint saving."""
    var checkpoint = ModelCheckpoint(filepath="checkpoint.pt", save_frequency=1)

    var state = TrainingState(epoch=0, batch=0, learning_rate=0.01)

    # Initially no errors
    assert_equal(
        checkpoint.get_error_count(), 0, "Error count should start at 0"
    )

    # Save a checkpoint (no error)
    checkpoint.on_epoch_end(state)
    assert_equal(
        checkpoint.get_error_count(), 0, "No errors from successful save"
    )


fn test_model_checkpoint_always_returns_continue() raises:
    """Test that checkpoint always returns CONTINUE signal."""
    var checkpoint = ModelCheckpoint(filepath="checkpoint.pt", save_frequency=1)

    var state = TrainingState(epoch=0, batch=0, learning_rate=0.01)
    state.metrics["val_loss"] = 1.0

    var signal = checkpoint.on_epoch_end(state)
    assert_equal(
        signal.value, 0, "Should always return CONTINUE to not stop training"
    )


# ============================================================================
# LoggingCallback Tests
# ============================================================================


fn test_logging_callback_logs_at_interval() raises:
    """Test logging callback logs at specified interval."""
    var logger = LoggingCallback(log_interval=1)

    var state = TrainingState(epoch=0, batch=0, learning_rate=0.01)
    state.metrics["train_loss"] = 0.5
    state.metrics["val_loss"] = 0.6

    # Epoch 0: Should log (0 % 1 == 0)
    logger.on_epoch_end(state)
    assert_equal(logger.get_log_count(), 1, "Should have logged at epoch 0")

    # Epoch 1: Should log
    state.epoch = 1
    logger.on_epoch_end(state)
    assert_equal(logger.get_log_count(), 2, "Should have logged at epoch 1")


fn test_logging_callback_skips_non_intervals() raises:
    """Test logging callback skips non-interval epochs."""
    var logger = LoggingCallback(log_interval=2)

    var state = TrainingState(epoch=0, batch=0, learning_rate=0.01)
    state.metrics["train_loss"] = 0.5

    # Epoch 0: Should log (0 % 2 == 0)
    logger.on_epoch_end(state)
    assert_equal(logger.get_log_count(), 1, "Should have logged at epoch 0")

    # Epoch 1: Should not log
    state.epoch = 1
    logger.on_epoch_end(state)
    assert_equal(logger.get_log_count(), 1, "Should not log at epoch 1")

    # Epoch 2: Should log (2 % 2 == 0)
    state.epoch = 2
    logger.on_epoch_end(state)
    assert_equal(logger.get_log_count(), 2, "Should have logged at epoch 2")


fn test_logging_callback_with_metrics() raises:
    """Test logging callback handles multiple metrics."""
    var logger = LoggingCallback(log_interval=1)

    var state = TrainingState(epoch=0, batch=0, learning_rate=0.01)
    state.metrics["train_loss"] = 0.5
    state.metrics["val_loss"] = 0.6
    state.metrics["accuracy"] = 0.95

    logger.on_epoch_end(state)
    assert_equal(logger.get_log_count(), 1, "Should have logged metrics")


fn test_logging_callback_with_learning_rate() raises:
    """Test logging callback includes learning rate."""
    var logger = LoggingCallback(log_interval=1)

    var state = TrainingState(epoch=0, batch=0, learning_rate=0.001)
    state.metrics["train_loss"] = 0.5

    logger.on_epoch_end(state)
    assert_equal(
        logger.get_log_count(), 1, "Should have logged with learning rate"
    )


fn test_logging_callback_empty_metrics() raises:
    """Test logging callback with no metrics."""
    var logger = LoggingCallback(log_interval=1)

    var state = TrainingState(epoch=0, batch=0, learning_rate=0.01)
    # No metrics added

    logger.on_epoch_end(state)
    assert_equal(logger.get_log_count(), 1, "Should log even with no metrics")


fn test_logging_callback_returns_continue() raises:
    """Test logging callback always returns CONTINUE."""
    var logger = LoggingCallback(log_interval=1)

    var state = TrainingState(epoch=0, batch=0, learning_rate=0.01)
    var signal = logger.on_epoch_end(state)
    assert_equal(signal.value, 0, "Should always return CONTINUE")


# ============================================================================
# Integration Tests
# ============================================================================


fn test_multiple_callbacks_together() raises:
    """Test multiple callbacks working together."""
    var early_stop = EarlyStopping(monitor="val_loss", patience=2, mode="min")
    var checkpoint = ModelCheckpoint(
        filepath="checkpoint.pt",
        monitor="val_loss",
        save_best_only=True,
        mode="min",
    )
    var logger = LoggingCallback(log_interval=1)

    var state = TrainingState(epoch=0, batch=0, learning_rate=0.01)
    state.metrics["val_loss"] = 1.0

    # All callbacks should work together
    early_stop.on_train_begin(state)
    early_stop.on_epoch_end(state)
    checkpoint.on_epoch_end(state)
    logger.on_epoch_end(state)

    assert_equal(early_stop.wait_count, 0, "Early stop initialized")
    assert_equal(checkpoint.get_save_count(), 1, "Checkpoint saved")
    assert_equal(logger.get_log_count(), 1, "Logger logged")


fn test_early_stopping_sets_should_stop_flag() raises:
    """Test that early stopping sets the should_stop flag."""
    var early_stop = EarlyStopping(monitor="val_loss", patience=1, mode="min")

    var state = TrainingState(epoch=0, batch=0, learning_rate=0.01)
    early_stop.on_train_begin(state)

    # First epoch
    state.metrics["val_loss"] = 1.0
    early_stop.on_epoch_end(state)
    assert_true(not state.should_stop, "should_stop should be False initially")

    # Second epoch - no improvement, patience exhausted
    state.epoch = 1
    state.metrics["val_loss"] = 1.1  # Worse than 1.0 in min mode
    early_stop.on_epoch_end(state)
    assert_true(
        state.should_stop, "should_stop should be set by early stopping"
    )


fn main() raises:
    """Run all tests."""
    # EarlyStopping tests
    test_early_stopping_min_mode_improves()
    test_early_stopping_min_mode_no_improvement()
    test_early_stopping_max_mode_improves()
    test_early_stopping_max_mode_no_improvement()
    test_early_stopping_missing_metric()
    test_early_stopping_should_stop_method()

    # ModelCheckpoint tests
    test_model_checkpoint_save_frequency()
    test_model_checkpoint_save_best_only_min_mode()
    test_model_checkpoint_save_best_only_max_mode()
    test_model_checkpoint_epoch_placeholder()
    test_model_checkpoint_error_count()
    test_model_checkpoint_always_returns_continue()

    # LoggingCallback tests
    test_logging_callback_logs_at_interval()
    test_logging_callback_skips_non_intervals()
    test_logging_callback_with_metrics()
    test_logging_callback_with_learning_rate()
    test_logging_callback_empty_metrics()
    test_logging_callback_returns_continue()

    # Integration tests
    test_multiple_callbacks_together()
    test_early_stopping_sets_should_stop_flag()

    print("All tests passed!")
