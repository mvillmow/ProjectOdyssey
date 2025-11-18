"""
Optimizers

Optimizer implementations for training neural networks.

Includes:
- SGD (Stochastic Gradient Descent) with momentum
- Adam (Adaptive Moment Estimation)
- AdamW (Adam with Weight Decay)
- RMSprop (Root Mean Square Propagation)

All optimizers implement the Optimizer trait for consistent interface.
"""

# Export optimizer implementations

# SGD optimizer (basic implementation)
from .sgd import sgd_step, sgd_step_simple

# TODO: Implement remaining optimizers
# from .base import Optimizer
# from .adam import Adam
# from .adamw import AdamW
# from .rmsprop import RMSprop
