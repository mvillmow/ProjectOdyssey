"""
Learning Rate Schedulers

Scheduler implementations for adjusting learning rates during training.

Includes:
- step_lr: Step decay scheduler (decay every N epochs)
- multistep_lr: Multi-step decay at specific milestones
- exponential_lr: Exponential decay per epoch
- constant_lr: No decay (baseline)

All schedulers are pure functions - caller passes epoch and gets learning rate.
Future: Add cosine annealing and warmup schedulers.
"""

# Export scheduler implementations
from .step_decay import step_lr, multistep_lr, exponential_lr, constant_lr

# Export scheduler classes
from ..schedulers import StepLR, CosineAnnealingLR, WarmupLR
