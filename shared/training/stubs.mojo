"""Stub implementations for TDD test execution.

These stubs provide minimal implementations to make tests executable
during the Test phase (Issue #33). They will be replaced with full
implementations in the Implementation phase (Issue #34).

IMPORTANT: These are NOT production code - they are scaffolding for TDD.
"""

from collections import Dict
from shared.training.base import (
    Callback,
    CallbackSignal,
    CONTINUE,
    STOP,
    TrainingState,
    LRScheduler,
)


# ============================================================================
# Stub Trainer Interface
# ============================================================================


@fieldwise_init
struct MockTrainer(Copyable, Movable):
    """Stub trainer for testing trainer interface contract.

    This minimal implementation provides just enough functionality.
    to make interface tests executable.
    """

    var epoch_count: Int
    var should_fail: Bool

    fn __init__(out self):
        """Initialize mock trainer."""
        self.epoch_count = 0
        self.should_fail = False

    fn train(
        mut self, epochs: Int, batch_size: Int = 32
    ) raises -> Dict[String, Float64]:
        """Stub train method.

        Args:
            epochs: Number of epochs to simulate.
            batch_size: Batch size (unused in stub).

        Returns:
            Dictionary with train_loss and val_loss lists.
        """
        var results = Dict[String, Float64]()

        # Simulate training for specified epochs
        for epoch in range(epochs):
            self.epoch_count += 1
            # Stub: Loss decreases linearly
            var train_loss = 1.0 - (Float64(epoch) * 0.1)
            var val_loss = 1.2 - (Float64(epoch) * 0.1)

            results["train_loss_" + String(epoch)] = train_loss
            results["val_loss_" + String(epoch)] = val_loss

        return results

    fn validate(self) raises -> Dict[String, Float64]:
        """Stub validate method.

        Returns:
            Dictionary with loss and optional accuracy.
        """
        var results = Dict[String, Float64]()
        results["loss"] = 0.5
        results["accuracy"] = 0.85
        return results

    fn save_checkpoint(self, path: String) raises:
        """Stub save checkpoint (no-op)."""
        pass

    fn load_checkpoint(mut self, path: String) raises:
        """Stub load checkpoint (no-op)."""
        pass


# ============================================================================
# Stub Learning Rate Schedulers
# ============================================================================


@value
struct MockStepLR(LRScheduler):
    """Stub StepLR scheduler for testing.

    Implements step decay: lr = base_lr * gamma^(epoch // step_size)
    """

    var base_lr: Float64
    var step_size: Int
    var gamma: Float64

    fn __init__(out self, base_lr: Float64, step_size: Int, gamma: Float64):
        """Initialize StepLR scheduler.

        Args:
            base_lr: Initial learning rate.
            step_size: Number of epochs between LR reductions.
            gamma: Multiplicative factor for LR reduction.
        """
        self.base_lr = base_lr
        self.step_size = step_size
        self.gamma = gamma

    fn get_lr(self, epoch: Int, batch: Int = 0) -> Float64:
        """Compute learning rate using step decay formula.

        Args:
            epoch: Current epoch (0-indexed).
            batch: Current batch (unused for epoch-based scheduler).

        Returns:
            Learning rate for this epoch.
        """
        if self.step_size <= 0:
            return self.base_lr

        var num_steps = epoch // self.step_size
        var decay_factor = self.gamma**num_steps
        return self.base_lr * decay_factor


@value
struct MockCosineAnnealingLR(LRScheduler):
    """Stub Cosine Annealing scheduler for testing.

    Implements: lr = eta_min + (base_lr - eta_min) * (1 + cos(pi * epoch / T_max)) / 2
    """

    var base_lr: Float64
    var T_max: Int
    var eta_min: Float64

    fn __init__(
        out self, base_lr: Float64, T_max: Int, eta_min: Float64 = 0.0
    ):
        """Initialize Cosine Annealing scheduler.

        Args:
            base_lr: Initial learning rate.
            T_max: Maximum number of epochs (period).
            eta_min: Minimum learning rate.
        """
        self.base_lr = base_lr
        self.T_max = T_max
        self.eta_min = eta_min

    fn get_lr(self, epoch: Int, batch: Int = 0) -> Float64:
        """Compute learning rate using cosine annealing formula.

        Args:
            epoch: Current epoch (0-indexed).
            batch: Current batch (unused).

        Returns:
            Learning rate for this epoch.
        """
        if self.T_max <= 0:
            return self.base_lr

        # Cosine formula (simplified for stub)
        # TODO(#34): Import proper math.cos function
        var progress = Float64(epoch) / Float64(self.T_max)
        # Approximate cosine with linear decay for now
        var factor = 1.0 - progress
        return self.eta_min + (self.base_lr - self.eta_min) * factor


@value
struct MockWarmupScheduler(LRScheduler):
    """Stub Warmup scheduler for testing.

    Linear warmup: lr increases from 0 to base_lr over warmup_epochs.
    """

    var base_lr: Float64
    var warmup_epochs: Int

    fn __init__(out self, base_lr: Float64, warmup_epochs: Int):
        """Initialize Warmup scheduler.

        Args:
            base_lr: Target learning rate after warmup.
            warmup_epochs: Number of epochs for warmup.
        """
        self.base_lr = base_lr
        self.warmup_epochs = warmup_epochs

    fn get_lr(self, epoch: Int, batch: Int = 0) -> Float64:
        """Compute learning rate with linear warmup.

        Args:
            epoch: Current epoch (0-indexed).
            batch: Current batch (unused).

        Returns:
            Learning rate for this epoch.
        """
        if self.warmup_epochs <= 0:
            return self.base_lr

        if epoch >= self.warmup_epochs:
            return self.base_lr

        # Linear warmup
        var progress = Float64(epoch) / Float64(self.warmup_epochs)
        return self.base_lr * progress


# ============================================================================
# Stub Callbacks
# ============================================================================


@value
struct MockEarlyStopping(Callback):
    """Stub Early Stopping callback for testing."""

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
        """Initialize early stopping.

        Args:.            `monitor`: Metric to monitor.
            `patience`: Epochs to wait before stopping.
            `min_delta`: Minimum improvement threshold.
        """
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.best_value = Float64(1e9)  # Start with large value (for loss)
        self.wait_count = 0
        self.stopped = False

    fn on_train_begin(mut self, mut state: TrainingState) -> CallbackSignal:
        """Reset state at training start."""
        self.best_value = Float64(1e9)
        self.wait_count = 0
        self.stopped = False
        return CONTINUE

    fn on_train_end(mut self, mut state: TrainingState) -> CallbackSignal:
        """No-op at training end."""
        return CONTINUE

    fn on_epoch_begin(mut self, mut state: TrainingState) -> CallbackSignal:
        """No-op at epoch begin."""
        return CONTINUE

    fn on_epoch_end(mut self, mut state: TrainingState) -> CallbackSignal:
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

    fn on_batch_begin(mut self, mut state: TrainingState) -> CallbackSignal:
        """No-op at batch begin."""
        return CONTINUE

    fn on_batch_end(mut self, mut state: TrainingState) -> CallbackSignal:
        """No-op at batch end."""
        return CONTINUE

    fn should_stop(self) -> Bool:
        """Check if training should stop.

        Returns:
            True if patience exhausted, False otherwise.
        """
        return self.stopped


@value
struct MockCheckpoint(Callback):
    """Stub Checkpoint callback for testing."""

    var save_path: String
    var save_count: Int

    fn __init__(out self, save_path: String = "checkpoint.pt"):
        """Initialize checkpoint callback.

        Args:.            `save_path`: Path template for saving checkpoints.
        """
        self.save_path = save_path
        self.save_count = 0

    fn on_train_begin(mut self, mut state: TrainingState) -> CallbackSignal:
        """No-op at training start."""
        return CONTINUE

    fn on_train_end(mut self, mut state: TrainingState) -> CallbackSignal:
        """No-op at training end."""
        return CONTINUE

    fn on_epoch_begin(mut self, mut state: TrainingState) -> CallbackSignal:
        """No-op at epoch begin."""
        return CONTINUE

    fn on_epoch_end(mut self, mut state: TrainingState) -> CallbackSignal:
        """Save checkpoint at end of epoch (stub).

        Args:
            state: Training state.

        Returns:
            CONTINUE always.
        """
        # Stub: Just increment counter
        self.save_count += 1
        return CONTINUE

    fn on_batch_begin(mut self, mut state: TrainingState) -> CallbackSignal:
        """No-op at batch begin."""
        return CONTINUE

    fn on_batch_end(mut self, mut state: TrainingState) -> CallbackSignal:
        """No-op at batch end."""
        return CONTINUE


@value
struct MockLoggingCallback(Callback):
    """Stub Logging callback for testing."""

    var log_interval: Int
    var log_count: Int

    fn __init__(out self, log_interval: Int = 1):
        """Initialize logging callback.

        Args:.            `log_interval`: Log every N epochs.
        """
        self.log_interval = log_interval
        self.log_count = 0

    fn on_train_begin(mut self, mut state: TrainingState) -> CallbackSignal:
        """No-op at training start."""
        return CONTINUE

    fn on_train_end(mut self, mut state: TrainingState) -> CallbackSignal:
        """No-op at training end."""
        return CONTINUE

    fn on_epoch_begin(mut self, mut state: TrainingState) -> CallbackSignal:
        """No-op at epoch begin."""
        return CONTINUE

    fn on_epoch_end(mut self, mut state: TrainingState) -> CallbackSignal:
        """Log metrics at end of epoch (stub).

        Args:
            state: Training state with metrics.

        Returns:
            CONTINUE always.
        """
        if state.epoch % self.log_interval == 0:
            self.log_count += 1
        return CONTINUE

    fn on_batch_begin(mut self, mut state: TrainingState) -> CallbackSignal:
        """No-op at batch begin."""
        return CONTINUE

    fn on_batch_end(mut self, mut state: TrainingState) -> CallbackSignal:
        """No-op at batch end."""
        return CONTINUE
