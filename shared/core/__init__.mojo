"""
Core Library - Pure functional operations for ML Odyssey.

This package provides pure functional operations for neural networks and mathematical
operations. All functions are stateless - they process inputs to produce outputs
without maintaining internal state.

Architecture:
    - Pure functional design - no classes, no internal state
    - All functions work with ExTensor (no Tensor alias)
    - Caller manages all state (weights, biases, momentum, etc.)
    - Functions return new values, never mutate inputs

Modules:
    extensor: Core tensor type and creation functions
    arithmetic: Element-wise arithmetic operations (add, subtract, multiply, divide)
    matrix: Matrix operations (matmul, transpose, dot, outer)
    activation: Activation functions (relu, sigmoid, tanh, softmax, gelu)
    linear: Linear transformations
    conv: Convolutional operations
    pooling: Pooling operations
    elementwise: Element-wise math functions (exp, log, sqrt, abs, clip)
    comparison: Comparison operations (equal, less, greater)
    broadcasting: Broadcasting utilities
    initializers: Weight initialization functions
    loss: Loss functions
    numerical_safety: NaN/Inf detection, gradient monitoring, numerical stability checks
    dtype_dispatch: Generic dtype dispatch helpers for eliminating dtype branching

Example:
    from shared.core.extensor import ExTensor, zeros
    from shared.core.linear import linear
    from shared.core.activation import relu
    from shared.core.matrix import matmul, transpose

    # Create tensors
    var x = zeros([32, 784])
    var weights = zeros([128, 784])
    var bias = zeros([128])

    # Forward pass (pure functional)
    var h1 = linear(x, weights, bias)
    var a1 = relu(h1)
"""

# Package version
alias VERSION = "0.1.0"

# ============================================================================
# Core Tensor Type and Creation Functions
# ============================================================================

from .extensor import (
    ExTensor,
    zeros,
    ones,
    full,
    empty,
    arange,
    eye,
    linspace,
    ones_like,
    zeros_like,
    full_like,
)

# ============================================================================
# Arithmetic Operations
# ============================================================================

from .arithmetic import (
    add,
    subtract,
    multiply,
    divide,
    floor_divide,
    modulo,
    power,
    add_backward,
    subtract_backward,
    multiply_backward,
    divide_backward,
)

# ============================================================================
# Matrix Operations
# ============================================================================

from .matrix import (
    matmul,
    transpose,
    dot,
    outer,
    matmul_backward,
    transpose_backward,
)

# ============================================================================
# Activation Functions
# ============================================================================

from .activation import (
    relu,
    leaky_relu,
    prelu,
    sigmoid,
    tanh,
    softmax,
    gelu,
    swish,
    mish,
    elu,
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

# ============================================================================
# Neural Network Operations
# ============================================================================

from .linear import (
    linear,
    linear_no_bias,
    linear_backward,
    linear_no_bias_backward,
)

from .conv import (
    conv2d,
    conv2d_no_bias,
    conv2d_backward,
    conv2d_no_bias_backward,
)

from .pooling import (
    maxpool2d,
    avgpool2d,
    global_avgpool2d,
    maxpool2d_backward,
    avgpool2d_backward,
    global_avgpool2d_backward,
)

from .dropout import (
    dropout,
    dropout2d,
    dropout_backward,
    dropout2d_backward,
)

from .normalization import (
    batch_norm2d,
    layer_norm,
)

# ============================================================================
# Element-wise Operations
# ============================================================================

from .elementwise import (
    abs,
    sign,
    exp,
    log,
    sqrt,
    sin,
    cos,
    clip,
    ceil,
    floor,
    round,
    trunc,
    logical_and,
    logical_or,
    logical_not,
    logical_xor,
    log10,
    log2,
    exp_backward,
    log_backward,
    sqrt_backward,
    abs_backward,
    clip_backward,
    log10_backward,
    log2_backward,
)

# ============================================================================
# Comparison Operations
# ============================================================================

from .comparison import (
    equal,
    not_equal,
    less,
    less_equal,
    greater,
    greater_equal,
)

# ============================================================================
# Broadcasting Utilities
# ============================================================================

from .broadcasting import (
    broadcast_shapes,
    are_shapes_broadcastable,
    compute_broadcast_strides,
)

# ============================================================================
# Initialization Functions
# ============================================================================

from .initializers import (
    xavier_uniform,
    xavier_normal,
    kaiming_uniform,
    kaiming_normal,
    uniform,
    normal,
    constant,
)

# ============================================================================
# Loss Functions
# ============================================================================

from .loss import (
    binary_cross_entropy,
    mean_squared_error,
    cross_entropy,
    binary_cross_entropy_backward,
    mean_squared_error_backward,
    cross_entropy_backward,
)

from .numerical_safety import (
    has_nan,
    has_inf,
    count_nan,
    count_inf,
    check_tensor_safety,
    tensor_min,
    tensor_max,
    check_tensor_range,
    compute_tensor_l2_norm,
    check_gradient_norm,
    check_gradient_vanishing,
    check_gradient_safety,
)

# ============================================================================
# Dtype Dispatch Helpers
# ============================================================================

from .dtype_dispatch import (
    dispatch_unary,
    dispatch_binary,
    dispatch_scalar,
    dispatch_float_unary,
    dispatch_float_binary,
    dispatch_float_scalar,
)

# ============================================================================
# Public API
# ============================================================================

__all__ = [
    # Core tensor type
    "ExTensor",
    # Tensor creation
    "zeros",
    "ones",
    "full",
    "empty",
    "arange",
    "eye",
    "linspace",
    "ones_like",
    "zeros_like",
    "full_like",
    # Arithmetic
    "add",
    "subtract",
    "multiply",
    "divide",
    "floor_divide",
    "modulo",
    "power",
    "add_backward",
    "subtract_backward",
    "multiply_backward",
    "divide_backward",
    # Matrix operations
    "matmul",
    "transpose",
    "dot",
    "outer",
    "matmul_backward",
    "transpose_backward",
    # Activations
    "relu",
    "leaky_relu",
    "prelu",
    "sigmoid",
    "tanh",
    "softmax",
    "gelu",
    "swish",
    "mish",
    "elu",
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
    # Neural network operations
    "linear",
    "linear_no_bias",
    "linear_backward",
    "linear_no_bias_backward",
    "conv2d",
    "conv2d_no_bias",
    "conv2d_backward",
    "conv2d_no_bias_backward",
    "maxpool2d",
    "avgpool2d",
    "global_avgpool2d",
    "maxpool2d_backward",
    "avgpool2d_backward",
    "global_avgpool2d_backward",
    "dropout",
    "dropout2d",
    "dropout_backward",
    "dropout2d_backward",
    "batch_norm2d",
    "layer_norm",
    # Element-wise
    "abs",
    "sign",
    "exp",
    "log",
    "sqrt",
    "sin",
    "cos",
    "clip",
    "ceil",
    "floor",
    "round",
    "trunc",
    "logical_and",
    "logical_or",
    "logical_not",
    "logical_xor",
    "log10",
    "log2",
    "exp_backward",
    "log_backward",
    "sqrt_backward",
    "abs_backward",
    "clip_backward",
    "log10_backward",
    "log2_backward",
    # Comparison
    "equal",
    "not_equal",
    "less",
    "less_equal",
    "greater",
    "greater_equal",
    # Broadcasting
    "broadcast_shapes",
    "are_shapes_broadcastable",
    "compute_broadcast_strides",
    # Initializers
    "xavier_uniform",
    "xavier_normal",
    "kaiming_uniform",
    "kaiming_normal",
    "uniform",
    "normal",
    "constant",
    # Loss functions
    "binary_cross_entropy",
    "mean_squared_error",
    "cross_entropy",
    "binary_cross_entropy_backward",
    "mean_squared_error_backward",
    "cross_entropy_backward",
    # Numerical safety
    "has_nan",
    "has_inf",
    "count_nan",
    "count_inf",
    "check_tensor_safety",
    "tensor_min",
    "tensor_max",
    "check_tensor_range",
    "compute_tensor_l2_norm",
    "check_gradient_norm",
    "check_gradient_vanishing",
    "check_gradient_safety",
    # Dtype dispatch helpers
    "dispatch_unary",
    "dispatch_binary",
    "dispatch_scalar",
    "dispatch_float_unary",
    "dispatch_float_binary",
    "dispatch_float_scalar",
]
