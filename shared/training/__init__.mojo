"""
Training Library

The training library provides reusable training infrastructure including optimizers,
schedulers, metrics, callbacks, and training loops for ML Odyssey paper implementations.

All components are implemented in Mojo for maximum performance.
"""

# Package version
alias VERSION = "0.1.0"

# ============================================================================
# Exports - Training Components
# ============================================================================

# Export base interfaces and utilities
from .base import (
    Callback,
    CallbackSignal,
    CONTINUE,
    STOP,
    TrainingState,
    LRScheduler,
    is_valid_loss,
    clip_gradients,
)

# Export scheduler implementations
from .schedulers import StepLR, CosineAnnealingLR, WarmupLR

# Export callback implementations
from .callbacks import EarlyStopping, ModelCheckpoint, LoggingCallback

# ============================================================================
# Public API
# ============================================================================
