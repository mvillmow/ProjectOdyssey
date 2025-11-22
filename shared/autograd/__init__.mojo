"""Autograd - Automatic Differentiation for ML Odyssey.

This module provides gradient computation capabilities for training neural networks.

Core Components:
- Variable: Tensor wrapper with gradient tracking (foundation)
- GradientTape: Operation recording structure (foundation)
- SGD: Stochastic gradient descent optimizer
- Functional helpers: Practical gradient computation for common patterns

Recommended API (Functional Helpers):
    from shared.autograd import mse_loss_and_grad, SGD

    # Compute loss and gradient in one call
    var result = mse_loss_and_grad(predictions, targets)
    var loss = result.loss
    var grad = result.grad

    # Update parameters using optimizer
    var optimizer = SGD(learning_rate=0.01)
    optimizer.step(parameters)
    optimizer.zero_grad(parameters)

Available Loss+Grad Helpers:
- mse_loss_and_grad: Mean squared error (regression)
- bce_loss_and_grad: Binary cross-entropy (binary classification)
- ce_loss_and_grad: Cross-entropy with softmax (multi-class classification)

Design Philosophy:
    YAGNI + KISS: Provide practical helpers for common patterns rather than
    complex computation graph autograd. This works today with current Mojo
    constraints and covers 90% of real use cases.

Status:
    ✅ Functional gradient helpers (mse, bce, ce)
    ✅ SGD optimizer
    ✅ Variable/Tape foundation (for future full autograd)
    🔮 Full automatic differentiation (future, see DESIGN.md)

References:
    - Gradient helpers: functional.mojo
    - Design rationale: DESIGN.md
    - Existing backward passes: /home/user/ml-odyssey/shared/core/
"""

from .variable import Variable
from .tape import GradientTape, TapeNode
from .optimizers import SGD
from .functional import (
    LossAndGrad,
    mse_loss_and_grad,
    bce_loss_and_grad,
    ce_loss_and_grad,
    compute_gradient,
    multiply_scalar,
    add_scalar,
    subtract_scalar,
    divide_scalar,
    apply_gradient,
    apply_gradients,
)

__all__ = [
    "Variable",
    "GradientTape",
    "TapeNode",
    "SGD",
    "LossAndGrad",
    "mse_loss_and_grad",
    "bce_loss_and_grad",
    "ce_loss_and_grad",
    "compute_gradient",
    "multiply_scalar",
    "add_scalar",
    "subtract_scalar",
    "divide_scalar",
    "apply_gradient",
    "apply_gradients",
]
