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

Activation Backward Passes:
- relu_backward: ReLU gradient
- leaky_relu_backward: Leaky ReLU gradient
- prelu_backward: PReLU gradient (returns GradientPair for input and alpha)
- sigmoid_backward: Sigmoid gradient
- tanh_backward: Tanh gradient
- gelu_backward: GELU gradient (supports exact and approximate)
- softmax_backward: Softmax gradient
- swish_backward: Swish activation gradient
- mish_backward: Mish activation gradient
- elu_backward: ELU gradient

Loss Backward Passes:
- binary_cross_entropy_backward: Binary cross-entropy gradient
- mean_squared_error_backward: MSE gradient
- cross_entropy_backward: Cross-entropy gradient

Matrix Operation Backward Passes:
- matmul_backward: Matrix multiplication gradient (returns GradientPair)
- transpose_backward: Transpose gradient

Arithmetic Backward Passes:
- add_backward: Addition gradient (returns GradientPair)
- subtract_backward: Subtraction gradient (returns GradientPair)
- multiply_backward: Element-wise multiplication gradient (returns GradientPair)
- divide_backward: Division gradient (returns GradientPair)

Element-wise Backward Passes:
- exp_backward: Exponential gradient
- log_backward: Natural log gradient
- log10_backward: Base-10 log gradient
- log2_backward: Base-2 log gradient
- sqrt_backward: Square root gradient
- abs_backward: Absolute value gradient
- clip_backward: Clip gradient

Reduction Backward Passes:
- sum_backward: Sum reduction gradient
- mean_backward: Mean reduction gradient
- max_reduce_backward: Max reduction gradient
- min_reduce_backward: Min reduction gradient

Network Layer Backward Passes:
- linear_backward: Linear/dense layer gradient (returns GradientPair for input, weight, bias)
- linear_no_bias_backward: Linear layer without bias gradient
- conv2d_backward: 2D convolution gradient
- conv2d_no_bias_backward: 2D convolution without bias gradient

Pooling Backward Passes:
- maxpool2d_backward: Max pooling 2D gradient
- avgpool2d_backward: Average pooling 2D gradient
- global_avgpool2d_backward: Global average pooling 2D gradient

Normalization Backward Passes:
- batch_norm2d_backward: Batch normalization 2D gradient

Dropout Backward Passes:
- dropout_backward: Dropout gradient (requires mask from forward pass)
- dropout2d_backward: Dropout 2D gradient (requires mask from forward pass)

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

# Activation backward passes
from ..core.activation import (
    relu_backward,
    leaky_relu_backward,
    prelu_backward,
    sigmoid_backward,
    tanh_backward,
    gelu_backward,
    softmax_backward,
    swish_backward,
    mish_backward,
    elu_backward,
)

# Loss backward passes
from ..core.loss import (
    binary_cross_entropy_backward,
    mean_squared_error_backward,
    cross_entropy_backward,
)

# Matrix operation backward passes
from ..core.matrix import (
    matmul_backward,
    transpose_backward,
)

# Arithmetic backward passes
from ..core.arithmetic import (
    add_backward,
    subtract_backward,
    multiply_backward,
    divide_backward,
)

# Element-wise backward passes
from ..core.elementwise import (
    exp_backward,
    log_backward,
    log10_backward,
    log2_backward,
    sqrt_backward,
    abs_backward,
    clip_backward,
)

# Reduction backward passes
from ..core.reduction import (
    sum_backward,
    mean_backward,
    max_reduce_backward,
    min_reduce_backward,
)

# Network layer backward passes
from ..core.linear import (
    linear_backward,
    linear_no_bias_backward,
)

from ..core.conv import (
    conv2d_backward,
    conv2d_no_bias_backward,
)

# Pooling backward passes
from ..core.pooling import (
    maxpool2d_backward,
    avgpool2d_backward,
    global_avgpool2d_backward,
)

# Normalization backward passes
from ..core.normalization import batch_norm2d_backward

# Dropout backward passes
from ..core.dropout import (
    dropout_backward,
    dropout2d_backward,
)

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
    # Activation backward passes
    "relu_backward",
    "leaky_relu_backward",
    "prelu_backward",
    "sigmoid_backward",
    "tanh_backward",
    "gelu_backward",
    "softmax_backward",
    "swish_backward",
    "mish_backward",
    "elu_backward",
    # Loss backward passes
    "binary_cross_entropy_backward",
    "mean_squared_error_backward",
    "cross_entropy_backward",
    # Matrix operation backward passes
    "matmul_backward",
    "transpose_backward",
    # Arithmetic backward passes
    "add_backward",
    "subtract_backward",
    "multiply_backward",
    "divide_backward",
    # Element-wise backward passes
    "exp_backward",
    "log_backward",
    "log10_backward",
    "log2_backward",
    "sqrt_backward",
    "abs_backward",
    "clip_backward",
    # Reduction backward passes
    "sum_backward",
    "mean_backward",
    "max_reduce_backward",
    "min_reduce_backward",
    # Network layer backward passes
    "linear_backward",
    "linear_no_bias_backward",
    "conv2d_backward",
    "conv2d_no_bias_backward",
    # Pooling backward passes
    "maxpool2d_backward",
    "avgpool2d_backward",
    "global_avgpool2d_backward",
    # Normalization backward passes
    "batch_norm2d_backward",
    # Dropout backward passes
    "dropout_backward",
    "dropout2d_backward",
]
