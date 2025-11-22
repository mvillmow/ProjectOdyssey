"""Base training interfaces and contracts.

This module defines the core traits and types for the training subsystem:
- Callback: Lifecycle hooks for training events
- TrainingState: Shared state passed to callbacks
- LRScheduler: Learning rate scheduling interface

These contracts establish clear APIs for Issue #34 implementation.
"""

from collections import Dict


# ============================================================================
# Callback System
# ============================================================================


@value
struct CallbackSignal:
    """Signal returned by callbacks to control training flow.

    Values:
        CONTINUE (0): Continue training normally
        STOP (1): Stop training immediately
    """

    var value: Int

    fn __init__(out self, value: Int):
        """Initialize callback signal.

        Args:
            value: Signal value (0=CONTINUE, 1=STOP).
        """
        self.value = value


# Callback signal constants
alias CONTINUE = CallbackSignal(0)
alias STOP = CallbackSignal(1)


@value
struct TrainingState:
    """Training state passed to callbacks.

    This struct provides callbacks with access to training metrics and control.
    It uses borrowed references to avoid ownership issues.

    Lifecycle:
        Created at training start, updated each epoch/batch, destroyed at end.

    Fields:
        epoch: Current epoch number (0-indexed)
        batch: Current batch number within epoch (0-indexed)
        metrics: Dictionary of metric name -> value
        learning_rate: Current learning rate
        should_stop: Flag set by callbacks to request training stop

    Example:
        var state = TrainingState(epoch=0, batch=0, metrics={}, lr=0.1)
        state.metrics["train_loss"] = 0.5
        state.metrics["val_loss"] = 0.6
    """

    var epoch: Int
    var batch: Int
    var metrics: Dict[String, Float64]
    var learning_rate: Float64
    var should_stop: Bool

    fn __init__(
        out self,
        epoch: Int = 0,
        batch: Int = 0,
        learning_rate: Float64 = 0.0,
    ):
        """Initialize training state.

        Args:
            epoch: Current epoch number.
            batch: Current batch number.
            learning_rate: Current learning rate.
        """
        self.epoch = epoch
        self.batch = batch
        self.metrics = Dict[String, Float64]()
        self.learning_rate = learning_rate
        self.should_stop = False


trait Callback:
    """Callback interface for training lifecycle hooks.

    Callbacks receive training events at specific points in the training loop.
    They can monitor metrics, save checkpoints, or request early stopping.

    Lifecycle Event Order:
        1. on_train_begin()        # Once at training start
        2. For each epoch:
            a. on_epoch_begin()    # Start of epoch
            b. For each batch:
                i.  on_batch_begin()   # Before forward pass
                ii. on_batch_end()     # After backward pass and optimizer step
            c. on_epoch_end()      # After validation (if any)
        3. on_train_end()          # Once at training end

    State Modification:
        - Callbacks can READ all fields of TrainingState
        - Callbacks can WRITE metrics dictionary (add new metrics)
        - Callbacks can SET should_stop flag (request early stopping)
        - Callbacks CANNOT modify model/optimizer directly (no references provided)

    Return Values:
        - Most hooks return CallbackSignal (CONTINUE or STOP)
        - STOP signal triggers graceful training shutdown
        - Multiple callbacks: first STOP signal takes precedence

    Example:
        struct MyCallback(Callback):
            fn on_epoch_end(inout self, inout state: TrainingState) -> CallbackSignal:
                print("Epoch", state.epoch, "loss:", state.metrics["train_loss"])
                return CONTINUE
    """

    fn on_train_begin(mut self, mut state: TrainingState) -> CallbackSignal:
        """Called once at the start of training.

        Args:
            state: Training state (epoch=0, batch=0, empty metrics).

        Returns:
            CallbackSignal (CONTINUE or STOP).
        """
        ...

    fn on_train_end(mut self, mut state: TrainingState) -> CallbackSignal:
        """Called once at the end of training.

        Args:
            state: Final training state with complete metrics history.

        Returns:
            CallbackSignal (ignored, training already ending).
        """
        ...

    fn on_epoch_begin(mut self, mut state: TrainingState) -> CallbackSignal:
        """Called at the start of each epoch.

        Args:
            state: Training state (epoch set, batch=0, previous epoch metrics).

        Returns:
            CallbackSignal (CONTINUE or STOP).
        """
        ...

    fn on_epoch_end(mut self, mut state: TrainingState) -> CallbackSignal:
        """Called at the end of each epoch (after validation).

        Args:
            state: Training state with current epoch metrics (train_loss, val_loss, etc.).

        Returns:
            CallbackSignal (CONTINUE or STOP).
        """
        ...

    fn on_batch_begin(mut self, mut state: TrainingState) -> CallbackSignal:
        """Called at the start of each batch.

        Args:
            state: Training state (epoch, batch set).

        Returns:
            CallbackSignal (CONTINUE or STOP).
        """
        ...

    fn on_batch_end(mut self, mut state: TrainingState) -> CallbackSignal:
        """Called at the end of each batch (after optimizer step).

        Args:
            state: Training state (may include batch metrics like batch_loss).

        Returns:
            CallbackSignal (CONTINUE or STOP).
        """
        ...


# ============================================================================
# Learning Rate Scheduler Interface
# ============================================================================


trait LRScheduler:
    """Learning rate scheduler interface.

    Schedulers compute learning rates based on training progress.
    They do NOT modify the optimizer directly - the training loop
    is responsible for calling get_lr() and updating the optimizer.

    Integration Pattern:
        1. Training loop calls scheduler.get_lr(epoch, batch)
        2. Training loop sets optimizer.learning_rate = new_lr
        3. Scheduler is stateless (pure function of epoch/batch)

    Scheduler Types:
        - Step decay: Reduce LR at fixed intervals
        - Cosine annealing: Smooth cosine decay
        - Warmup: Linear increase then constant

    Example:
        struct StepLR(LRScheduler):
            var base_lr: Float64
            var step_size: Int
            var gamma: Float64

            fn get_lr(self, epoch: Int, batch: Int) -> Float64:
                let steps = epoch // self.step_size
                return self.base_lr * (self.gamma ** steps)
    """

    fn get_lr(self, epoch: Int, batch: Int = 0) -> Float64:
        """Compute learning rate for given epoch and batch.

        Args:
            epoch: Current epoch number (0-indexed).
            batch: Current batch number within epoch (0-indexed).

        Returns:
            Learning rate to use for this step.

        Notes:
            - Schedulers should be deterministic (same inputs -> same output)
            - epoch and batch are 0-indexed
            - batch parameter is optional (defaults to 0 for epoch-based schedulers)
        """
        ...


# ============================================================================
# Numerical Safety
# ============================================================================


fn is_valid_loss(loss: Float64) -> Bool:
    """Check if loss value is valid (not NaN or inf).

    Args:
        loss: Loss value to check.

    Returns:
        True if loss is finite (not NaN, not inf), False otherwise.

    Example:
        if not is_valid_loss(loss):
            print("Training diverged! Loss is", loss)
            break

    Warning:
        This is a placeholder implementation that always returns True.
        It will be replaced with proper NaN/inf detection in Issue #34.
    """
    print("[WARNING] is_valid_loss is a placeholder - always returns True")
    # TODO(#34): Implement NaN/inf detection with proper Float64 checks
    return True


fn clip_gradients(gradients: List[Float64], max_norm: Float64) -> List[Float64]:
    """Clip gradients by global norm to prevent exploding gradients.

    Args:
        gradients: List of gradient values.
        max_norm: Maximum allowed gradient norm.

    Returns:
        Clipped gradients with norm <= max_norm.

    Example:
        clipped_grads = clip_gradients(grads, max_norm=1.0)

    Warning:
        This is a placeholder implementation that returns gradients unchanged.
        It will be replaced with proper gradient clipping in Issue #34.
    """
    print("[WARNING] clip_gradients is a placeholder - returns gradients unchanged")
    # TODO(#34): Implement gradient norm computation and clipping
    return gradients
