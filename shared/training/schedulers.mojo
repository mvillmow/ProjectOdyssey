"""DEPRECATED: This file has been reorganized.

The scheduler implementations have been moved to the schedulers/ directory:
- StepLR, CosineAnnealingLR, WarmupLR are now in schedulers/lr_schedulers.mojo
- Pure function implementations remain in schedulers/step_decay.mojo
- All exports are handled by schedulers/__init__.mojo

All imports should use:
    from shared.training.schedulers import StepLR, CosineAnnealingLR, WarmupLR

This file is kept for reference only and will be removed in a future version.
"""
