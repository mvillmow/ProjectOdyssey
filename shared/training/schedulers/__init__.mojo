"""Learning Rate Schedulers

Scheduler implementations for adjusting learning rates during training

Includes:
- StepLR: Step decay scheduler (decay every N epochs)
- CosineAnnealingLR: Cosine annealing scheduler
- WarmupLR: Linear warmup scheduler
- ExponentialLR: Exponential decay scheduler
- MultiStepLR: Multi-step decay scheduler
- ReduceLROnPlateau: Metric-based decay scheduler
- WarmupCosineAnnealingLR: Combined warmup and cosine annealing
- WarmupStepLR: Combined warmup and step decay

All schedulers are struct-based implementations of the LRScheduler trait
"""

# Export scheduler implementations
from .lr_schedulers import (
    StepLR,
    CosineAnnealingLR,
    WarmupLR,
    ExponentialLR,
    MultiStepLR,
    ReduceLROnPlateau,
    WarmupCosineAnnealingLR,
    WarmupStepLR,
)

# Also export pure function implementations for backward compatibility
from .step_decay import step_lr, multistep_lr, exponential_lr, constant_lr
