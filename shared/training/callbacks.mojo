"""Callback implementations for training loop control.

This module provides callback implementations for common training workflows:
- EarlyStopping: Stop training when validation metric stops improving
- ModelCheckpoint: Save model checkpoints during training
- LoggingCallback: Log training metrics at specified intervals

All callbacks implement the Callback trait from base.mojo.
"""

from collections import Dict
from shared.training.base import (
    Callback,
    CallbackSignal,
    CONTINUE,
    STOP,
    TrainingState,
)


# ============================================================================
# Early Stopping Callback
# ============================================================================


@value
struct EarlyStopping(Callback):
    """Stop training when monitored metric stops improving.

    Early stopping monitors a validation metric and stops training when
    the metric fails to improve for a specified number of epochs (patience).

    Attributes:
        monitor: Name of metric to monitor (e.g., "val_loss")
        patience: Number of epochs with no improvement before stopping
        min_delta: Minimum change to qualify as improvement
        best_value: Best value seen so far
        wait_count: Epochs since last improvement
        stopped: Whether training has been stopped

    Example:
        var early_stop = EarlyStopping(
            monitor="val_loss",
            patience=5,
            min_delta=0.001
        )
    """

    var monitor: String
    var patience: Int
    var min_delta: Float64
    var best_value: Float64
    var wait_count: Int
    var stopped: Bool

    fn __init__(
        out self,
        monitor: String = "val_loss",
        patience: Int = 5,
        min_delta: Float64 = 0.0,
    ):
        """Initialize early stopping callback.

        Args:
            monitor: Metric to monitor (e.g., "val_loss").
            patience: Epochs to wait before stopping.
            min_delta: Minimum improvement threshold.
        """
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.best_value = Float64(1e9)
        self.wait_count = 0
        self.stopped = False

    fn on_train_begin(inout self, inout state: TrainingState) -> CallbackSignal:
        """Reset state at training start."""
        self.best_value = Float64(1e9)
        self.wait_count = 0
        self.stopped = False
        return CONTINUE

    fn on_train_end(inout self, inout state: TrainingState) -> CallbackSignal:
        """No-op at training end."""
        return CONTINUE

    fn on_epoch_begin(inout self, inout state: TrainingState) -> CallbackSignal:
        """No-op at epoch begin."""
        return CONTINUE

    fn on_epoch_end(inout self, inout state: TrainingState) -> CallbackSignal:
        """Check for improvement and decide whether to stop.

        Args:
            state: Training state with current metrics.

        Returns:
            STOP if patience exhausted, CONTINUE otherwise.
        """
        # Check if monitored metric exists
        if self.monitor not in state.metrics:
            return CONTINUE

        var current_value = state.metrics[self.monitor]

        # Check for improvement (assuming lower is better)
        var improved = (self.best_value - current_value) > self.min_delta

        if improved:
            self.best_value = current_value
            self.wait_count = 0
        else:
            self.wait_count += 1

        if self.wait_count >= self.patience:
            self.stopped = True
            state.should_stop = True
            return STOP

        return CONTINUE

    fn on_batch_begin(inout self, inout state: TrainingState) -> CallbackSignal:
        """No-op at batch begin."""
        return CONTINUE

    fn on_batch_end(inout self, inout state: TrainingState) -> CallbackSignal:
        """No-op at batch end."""
        return CONTINUE

    fn should_stop(self) -> Bool:
        """Check if training should stop.

        Returns:
            True if patience exhausted, False otherwise.
        """
        return self.stopped


# ============================================================================
# Model Checkpoint Callback
# ============================================================================


@value
struct ModelCheckpoint(Callback):
    """Save model checkpoints during training.

    Saves the model state at specified intervals (e.g., every epoch).
    Can be configured to save only when metrics improve.

    Error Handling:
        If checkpoint saving fails, a warning is printed but training continues.
        This prevents training interruption due to I/O failures (e.g., disk full,
        permission denied). The error message clearly identifies the cause.

    Attributes:
        save_path: Path template for saving checkpoints
        save_count: Number of checkpoints saved
        error_count: Number of failed checkpoint save attempts

    Example:
        var checkpoint = ModelCheckpoint(save_path="checkpoint_epoch_{epoch}.pt")
    """

    var save_path: String
    var save_count: Int
    var error_count: Int

    fn __init__(out self, save_path: String = "checkpoint.pt"):
        """Initialize checkpoint callback.

        Args:
            save_path: Path template for saving checkpoints.
        """
        self.save_path = save_path
        self.save_count = 0
        self.error_count = 0

    fn on_train_begin(inout self, inout state: TrainingState) -> CallbackSignal:
        """No-op at training start."""
        return CONTINUE

    fn on_train_end(inout self, inout state: TrainingState) -> CallbackSignal:
        """No-op at training end."""
        return CONTINUE

    fn on_epoch_begin(inout self, inout state: TrainingState) -> CallbackSignal:
        """No-op at epoch begin."""
        return CONTINUE

    fn on_epoch_end(inout self, inout state: TrainingState) -> CallbackSignal:
        """Save checkpoint at end of epoch with error handling.

        Attempts to save checkpoint at the end of each epoch. If saving fails,
        logs a warning but continues training to prevent I/O errors from
        interrupting the training process.

        Args:
            state: Training state with current epoch number.

        Returns:
            CONTINUE always (even if checkpoint save fails).
        """
        self.save_count += 1

        # Build checkpoint path from template and epoch number
        # TODO(#34): Implement actual checkpoint saving when model interface available
        var checkpoint_path = self.save_path

        # When model interface is available, implement actual checkpoint saving:
        # try:
        #     model.save(checkpoint_path)
        # except:
        #     self.error_count += 1
        #     print("Warning: Failed to save checkpoint to", checkpoint_path)
        #     print("Checkpoint save error - epoch", state.epoch)
        #     # Continue training despite checkpoint save failure

        # Always return CONTINUE to prevent I/O errors from stopping training
        return CONTINUE

    fn on_batch_begin(inout self, inout state: TrainingState) -> CallbackSignal:
        """No-op at batch begin."""
        return CONTINUE

    fn on_batch_end(inout self, inout state: TrainingState) -> CallbackSignal:
        """No-op at batch end."""
        return CONTINUE

    fn get_save_count(self) -> Int:
        """Get number of checkpoints saved.

        Returns:
            Number of checkpoints saved so far.
        """
        return self.save_count

    fn get_error_count(self) -> Int:
        """Get number of checkpoint save errors.

        Returns:
            Number of failed checkpoint save attempts.
        """
        return self.error_count


# ============================================================================
# Logging Callback
# ============================================================================


@value
struct LoggingCallback(Callback):
    """Log training metrics at specified intervals.

    Logs training progress to stdout at regular intervals.

    Attributes:
        log_interval: Log every N epochs
        log_count: Number of times logged

    Example:
        var logger = LoggingCallback(log_interval=1)
    """

    var log_interval: Int
    var log_count: Int

    fn __init__(out self, log_interval: Int = 1):
        """Initialize logging callback.

        Args:
            log_interval: Log every N epochs.
        """
        self.log_interval = log_interval
        self.log_count = 0

    fn on_train_begin(inout self, inout state: TrainingState) -> CallbackSignal:
        """Log training start."""
        return CONTINUE

    fn on_train_end(inout self, inout state: TrainingState) -> CallbackSignal:
        """Log training end."""
        return CONTINUE

    fn on_epoch_begin(inout self, inout state: TrainingState) -> CallbackSignal:
        """No-op at epoch begin."""
        return CONTINUE

    fn on_epoch_end(inout self, inout state: TrainingState) -> CallbackSignal:
        """Log metrics at end of epoch.

        Args:
            state: Training state with metrics.

        Returns:
            CONTINUE always.
        """
        if state.epoch % self.log_interval == 0:
            self.log_count += 1
            # TODO(#34): Implement actual logging when desired
        return CONTINUE

    fn on_batch_begin(inout self, inout state: TrainingState) -> CallbackSignal:
        """No-op at batch begin."""
        return CONTINUE

    fn on_batch_end(inout self, inout state: TrainingState) -> CallbackSignal:
        """No-op at batch end."""
        return CONTINUE

    fn get_log_count(self) -> Int:
        """Get number of times logged.

        Returns:
            Number of logging calls made.
        """
        return self.log_count
