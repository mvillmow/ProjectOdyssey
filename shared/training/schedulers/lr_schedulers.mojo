"""Learning rate scheduler implementations.

This module provides implementations of common learning rate scheduling strategies:
- StepLR: Decay learning rate at fixed intervals
- CosineAnnealingLR: Smooth cosine decay
- WarmupLR: Linear warmup phase

All schedulers implement the LRScheduler trait from base.mojo.
"""

from math import pi, cos
from shared.training.base import LRScheduler


# ============================================================================
# Step Learning Rate Scheduler
# ============================================================================


struct StepLR(LRScheduler, Copyable, Movable):
    """Step decay: reduce learning rate at fixed intervals.

    Reduces the learning rate by a factor of gamma every step_size epochs.

    Formula:
        lr = base_lr * gamma^(epoch // step_size)

    Attributes:
        `base_lr`: Initial learning rate.
        `step_size`: Number of epochs between LR reductions.
        `gamma`: Multiplicative factor for LR reduction.

    Example:.        var scheduler = StepLR(
            base_lr=0.1,
            step_size=10,
            gamma=0.1
        )
        # After 10 epochs: LR = 0.1 * 0.1 = 0.01
        # After 20 epochs: LR = 0.1 * 0.01 = 0.001
    """

    var base_lr: Float64
    var step_size: Int
    var gamma: Float64

    fn __init__(
        out self, base_lr: Float64, step_size: Int, gamma: Float64
    ):
        """Initialize StepLR scheduler.

        Args:.            `base_lr`: Initial learning rate.
            `step_size`: Number of epochs between LR reductions.
            `gamma`: Multiplicative factor for LR reduction.
        """
        self.base_lr = base_lr
        self.step_size = step_size
        self.gamma = gamma

    fn get_lr(self, epoch: Int, batch: Int = 0) -> Float64:
        """Compute learning rate using step decay formula.

        Args:.            `epoch`: Current epoch (0-indexed).
            `batch`: Current batch (unused for epoch-based scheduler).

        Returns:.            Learning rate for this epoch.
        """
        if self.step_size <= 0:
            return self.base_lr

        var num_steps = epoch // self.step_size
        var decay_factor = self.gamma ** num_steps
        return self.base_lr * decay_factor


# ============================================================================
# Cosine Annealing Learning Rate Scheduler
# ============================================================================


struct CosineAnnealingLR(LRScheduler, Copyable, Movable):
    """Cosine annealing: smooth cosine decay from base_lr to eta_min.

    The learning rate follows a cosine curve, starting at base_lr and.
    smoothly decaying to eta_min over T_max epochs.

    Formula:
        lr = eta_min + (base_lr - eta_min) * (1 + cos(pi * epoch / T_max)) / 2

    Attributes:
        `base_lr`: Initial learning rate.
        T_max: Maximum number of epochs (period)
        `eta_min`: Minimum learning rate.

    Example:.        var scheduler = CosineAnnealingLR(
            base_lr=0.1,
            T_max=100,
            eta_min=0.0
        )
        # At epoch 0: LR = 0.1 (maximum)
        # At epoch 50: LR = 0.05 (halfway)
        # At epoch 100: LR = 0.0 (minimum)
    """

    var base_lr: Float64
    var T_max: Int
    var eta_min: Float64

    fn __init__(
        out self, base_lr: Float64, T_max: Int, eta_min: Float64 = 0.0
    ):
        """Initialize Cosine Annealing scheduler.

        Args:.            `base_lr`: Initial learning rate.
            T_max: Maximum number of epochs (period).
            `eta_min`: Minimum learning rate.
        """
        self.base_lr = base_lr
        self.T_max = T_max
        self.eta_min = eta_min

    fn get_lr(self, epoch: Int, batch: Int = 0) -> Float64:
        """Compute learning rate using cosine annealing formula.

        Args:.            `epoch`: Current epoch (0-indexed).
            `batch`: Current batch (unused).

        Returns:.            Learning rate for this epoch.
        """
        if self.T_max <= 0:
            return self.base_lr

        # Clamp epoch to T_max range
        var clamped_epoch = epoch
        if clamped_epoch > self.T_max:
            clamped_epoch = self.T_max

        # Cosine annealing formula
        var progress = Float64(clamped_epoch) / Float64(self.T_max)
        var cosine_factor = (1.0 + cos(pi * progress)) / 2.0
        return self.eta_min + (self.base_lr - self.eta_min) * cosine_factor


# ============================================================================
# Warmup Learning Rate Scheduler
# ============================================================================


struct WarmupLR(LRScheduler, Copyable, Movable):
    """Linear warmup: gradually increase learning rate during initial epochs.

    The learning rate increases linearly from 0 to base_lr over warmup_epochs,
    then remains constant at base_lr.

    Formula:
        lr = base_lr * (epoch / warmup_epochs)  for epoch < warmup_epochs
        lr = base_lr                            for epoch >= warmup_epochs

    Attributes:
        `base_lr`: Target learning rate after warmup.
        `warmup_epochs`: Number of epochs for warmup phase.

    Example:.        var scheduler = WarmupLR(
            base_lr=0.1,
            warmup_epochs=10
        )
        # Epochs 0-9: LR increases from 0 to 0.1
        # Epochs 10+: LR = 0.1
    """

    var base_lr: Float64
    var warmup_epochs: Int

    fn __init__(out self, base_lr: Float64, warmup_epochs: Int):
        """Initialize Warmup scheduler.

        Args:.            `base_lr`: Target learning rate after warmup.
            `warmup_epochs`: Number of epochs for warmup.
        """
        self.base_lr = base_lr
        self.warmup_epochs = warmup_epochs

    fn get_lr(self, epoch: Int, batch: Int = 0) -> Float64:
        """Compute learning rate with linear warmup.

        Args:.            `epoch`: Current epoch (0-indexed).
            `batch`: Current batch (unused).

        Returns:.            Learning rate for this epoch.
        """
        if self.warmup_epochs <= 0:
            return self.base_lr

        if epoch >= self.warmup_epochs:
            return self.base_lr

        # Linear warmup
        var progress = Float64(epoch) / Float64(self.warmup_epochs)
        return self.base_lr * progress
