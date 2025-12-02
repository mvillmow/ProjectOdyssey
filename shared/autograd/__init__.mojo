"""Autograd - Automatic Differentiation for ML Odyssey.

This module provides gradient computation capabilities for training neural networks.

Core Components:
- Variable: Tensor wrapper with gradient tracking (foundation)
- GradientTape: Operation recording structure (foundation)
- SGD: Stochastic gradient descent optimizer
- Functional helpers: Practical gradient computation for common patterns
- Backward functions: Unified API for all gradient computations

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

Backward Functions (Unified API from shared.core):

NOTE: Backward function implementations are in progress. See TODO.md for current status.
Once implemented, the following backward passes will be available:

Activation: relu_backward, leaky_relu_backward, prelu_backward, sigmoid_backward, tanh_backward,
gelu_backward, softmax_backward, swish_backward, mish_backward, elu_backward

Loss: binary_cross_entropy_backward, mean_squared_error_backward, cross_entropy_backward

Matrix Ops: matmul_backward, transpose_backward

Arithmetic: add_backward, subtract_backward, multiply_backward, divide_backward

Element-wise: exp_backward, log_backward, log10_backward, log2_backward, sqrt_backward,
abs_backward, clip_backward

Reduction: sum_backward, mean_backward, max_reduce_backward, min_reduce_backward

Network Layers: linear_backward, linear_no_bias_backward, conv2d_backward, conv2d_no_bias_backward

Pooling: maxpool2d_backward, avgpool2d_backward, global_avgpool2d_backward

Normalization: batch_norm2d_backward

Dropout: dropout_backward, dropout2d_backward

Design Philosophy:
    YAGNI + KISS: Provide practical helpers for common patterns rather than
    complex computation graph autograd. This works today with current Mojo
    constraints and covers 90% of real use cases.

Status:
    ✅ Functional gradient helpers (mse, bce, ce)
    ✅ SGD optimizer
    ✅ Variable/Tape foundation (for future full autograd)
    ✅ Unified backward function API (Issue #2196)
    🔮 Full automatic differentiation (future, see DESIGN.md)

References:
    - Gradient helpers: functional.mojo
    - Design rationale: DESIGN.md
    - Backward functions: ../core/ modules
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

# ============================================================================
# Re-export Backward Functions from shared.core (Issue #2196)
# ============================================================================
# NOTE: Backward function implementations are in progress.
# These imports are commented out until the functions are available.
# See TODO.md for current status.

# # Activation backward passes
# from ..core.activation import (
#     relu_backward,
#     leaky_relu_backward,
#     prelu_backward,
#     sigmoid_backward,
#     tanh_backward,
#     gelu_backward,
#     softmax_backward,
#     swish_backward,
#     mish_backward,
#     elu_backward,
# )

# # Loss backward passes
# from ..core.loss import (
#     binary_cross_entropy_backward,
#     mean_squared_error_backward,
#     cross_entropy_backward,
# )

# # Matrix operation backward passes
# from ..core.matrix import (
#     matmul_backward,
#     transpose_backward,
# )

# # Arithmetic backward passes
# from ..core.arithmetic import (
#     add_backward,
#     subtract_backward,
#     multiply_backward,
#     divide_backward,
# )

# # Element-wise backward passes
# from ..core.elementwise import (
#     exp_backward,
#     log_backward,
#     log10_backward,
#     log2_backward,
#     sqrt_backward,
#     abs_backward,
#     clip_backward,
# )

# # Reduction backward passes
# from ..core.reduction import (
#     sum_backward,
#     mean_backward,
#     max_reduce_backward,
#     min_reduce_backward,
# )

# # Network layer backward passes
# from ..core.linear import (
#     linear_backward,
#     linear_no_bias_backward,
# )

# from ..core.conv import (
#     conv2d_backward,
#     conv2d_no_bias_backward,
# )

# # Pooling backward passes
# from ..core.pooling import (
#     maxpool2d_backward,
#     avgpool2d_backward,
#     global_avgpool2d_backward,
# )

# # Normalization backward passes
# from ..core.normalization import batch_norm2d_backward

# # Dropout backward passes
# from ..core.dropout import (
#     dropout_backward,
#     dropout2d_backward,
# )

__all__ = [
    # Foundation components
    "Variable",
    "GradientTape",
    "TapeNode",
    "SGD",
    # Functional helpers
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
    # Backward passes from shared.core (to be added when implementations are complete)
    # See TODO.md for progress on backward function implementations
]
