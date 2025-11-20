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

# SGD optimizer (functional implementation)
from .sgd import sgd_step, sgd_step_simple

# Adam optimizer (functional implementation)
from .adam import adam_step, adam_step_simple

# TODO: Implement remaining optimizers
# from .adamw import adamw_step
# from .rmsprop import rmsprop_step
