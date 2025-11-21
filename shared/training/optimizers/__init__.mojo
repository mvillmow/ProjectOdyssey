"""
Optimizers

Optimizer implementations for training neural networks.

Includes:
- SGD (Stochastic Gradient Descent) with momentum
- Adam (Adaptive Moment Estimation)
- AdamW (Adam with Weight Decay)
- RMSprop (Root Mean Square Propagation)

All optimizers follow pure functional design - caller manages state.
"""

# Export optimizer implementations

# SGD optimizer (functional implementation and in-place mutation)
from .sgd import sgd_step, sgd_step_simple, sgd_momentum_update_inplace

# Adam optimizer (functional implementation)
from .adam import adam_step, adam_step_simple

# RMSprop optimizer (functional implementation)
from .rmsprop import rmsprop_step, rmsprop_step_simple

# TODO: Implement remaining optimizers
# from .adamw import adamw_step
