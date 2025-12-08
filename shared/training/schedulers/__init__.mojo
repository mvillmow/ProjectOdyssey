"""Learning Rate Schedulers

Scheduler implementations for adjusting learning rates during training

Includes:
- StepLR: Step decay scheduler (decay every N epochs)
- CosineAnnealingLR: Cosine annealing scheduler
- WarmupLR: Linear warmup scheduler

All schedulers are struct-based implementations of the LRScheduler trait
"""

# Export scheduler implementations
from .lr_schedulers import (
    StepLR,
    CosineAnnealingLR,
    WarmupLR,
    ReduceLROnPlateau,
)

# Also export pure function implementations for backward compatibility
from .step_decay import step_lr, multistep_lr, exponential_lr, constant_lr
