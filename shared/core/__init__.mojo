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
    types: Custom data types (FP8 for E4M3, BF8 for E5M2 8-bit floating point)
    arithmetic: Element-wise arithmetic operations (add, subtract, multiply, divide)
    matrix: Matrix operations (matmul, transpose, dot, outer)
    activation: Activation functions (relu, sigmoid, tanh, softmax, gelu)
    activation_ops: Activation operation utilities (scalar exp functions)
    linear: Linear transformations
    conv: Convolutional operations
    pooling: Pooling operations
    elementwise: Element-wise math functions (exp, log, sqrt, abs, clip)
    comparison: Comparison operations (equal, less, greater)
    broadcasting: Broadcasting utilities
    initializers: Weight initialization functions
    loss: Loss functions
    loss_utils: Utility functions for loss computation (clipping, epsilon handling, blending)
    numerical_safety: NaN/Inf detection, gradient monitoring, numerical stability checks
    dtype_dispatch: Generic dtype dispatch helpers for eliminating dtype branching
    reduction: Reduction operations (sum, mean, max, min) with forward and backward passes
    reduction_utils: Utility functions for reduction operations (coordinate/stride computation)
    utils: Utility functions (argmax, top_k_indices, top_k, argsort)
    scalar_ops: Scalar mathematical operations (sqrt, pow for float32 and float64)
    normalize_ops: Normalization operations for data preprocessing (RGB normalization)

Example:
   ```mojo
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
    ```

FIXME(#2715): Placeholder import tests in tests/shared/test_imports.mojo require:
- test_core_imports (line 17)
- test_core_layers_imports (line 31)
- test_core_activations_imports (line 46)
- test_core_types_imports (line 60)
All tests marked as "(placeholder - awaiting implementation)" and require module
imports to be uncommented as Issue #49 progresses. See Issue #49 for details
"""

# Package version
from ..version import VERSION

# ============================================================================
# Default Hyperparameters
# ============================================================================

from .defaults import (
    DEFAULT_LEAKY_RELU_ALPHA,
    DEFAULT_ELU_ALPHA,
    DEFAULT_HARD_TANH_MIN,
    DEFAULT_HARD_TANH_MAX,
    DEFAULT_DROPOUT_RATE,
    DEFAULT_BATCHNORM_MOMENTUM,
    DEFAULT_UNIFORM_LOW,
    DEFAULT_UNIFORM_HIGH,
    DEFAULT_AUGMENTATION_PROB,
    DEFAULT_TEXT_AUGMENTATION_PROB,
    DEFAULT_RANDOM_SEED,
)

# ============================================================================
# Mathematical Constants
# ============================================================================

from .math_constants import (
    PI,
    SQRT_2,
    SQRT_2_OVER_PI,
    INV_SQRT_2PI,
    GELU_COEFF,
    LN2,
    LN10,
)

# ============================================================================
# Numerical Stability Constants
# ============================================================================

from .numerical_constants import (
    EPSILON_DIV,
    EPSILON_LOSS,
    EPSILON_NORM,
    GRADIENT_MAX_NORM,
    GRADIENT_MIN_NORM,
)

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
# Shape Manipulation Operations
# ============================================================================

from .shape import (
    reshape,
    squeeze,
    unsqueeze,
    expand_dims,
    flatten,
    ravel,
    concatenate,
    stack,
    is_contiguous,
    as_contiguous,
    view,
    conv2d_output_shape,
    pool_output_shape,
    flatten_size,
    flatten_to_2d,
    transposed_conv2d_output_shape,
    global_avgpool_output_shape,
    linear_output_shape,
)

# ============================================================================
# Custom Data Types
# ============================================================================

from .types.fp8 import FP8
from .types.bf8 import BF8

# ============================================================================
# Gradient Container Types
# ============================================================================

from .gradient_types import (
    GradientPair,
    GradientTriple,
    GradientQuad,
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
    hard_sigmoid,
    hard_swish,
    hard_tanh,
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
    hard_sigmoid_backward,
    hard_swish_backward,
    hard_tanh_backward,
)

from .activation_ops import (
    exp_scalar_f32,
    exp_scalar_f64,
)

# ============================================================================
# Neural Network Operations
# ============================================================================

from .linear import (
    linear,
    linear_no_bias,
    linear_backward,
    linear_no_bias_backward,
    LinearBackwardResult,
    LinearNoBiasBackwardResult,
)

from .conv import (
    conv2d,
    conv2d_no_bias,
    conv2d_backward,
    conv2d_no_bias_backward,
    Conv2dBackwardResult,
    Conv2dNoBiasBackwardResult,
    depthwise_conv2d,
    depthwise_conv2d_no_bias,
    depthwise_conv2d_backward,
    depthwise_conv2d_no_bias_backward,
    DepthwiseConv2dBackwardResult,
    DepthwiseConv2dNoBiasBackwardResult,
    depthwise_separable_conv2d,
    depthwise_separable_conv2d_no_bias,
    depthwise_separable_conv2d_backward,
    depthwise_separable_conv2d_no_bias_backward,
    DepthwiseSeparableConv2dBackwardResult,
    DepthwiseSeparableConv2dNoBiasBackwardResult,
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
    batch_norm2d_backward,
    layer_norm,
    layer_norm_backward,
    group_norm,
    group_norm_backward,
    instance_norm,
    instance_norm_backward,
)

from .normalize_ops import normalize_rgb

from .scalar_ops import (
    sqrt_scalar_f32,
    sqrt_scalar_f64,
    pow_scalar_f32,
    pow_scalar_f64,
)

# ============================================================================
# Attention Mechanisms
# ============================================================================

from .attention import (
    scaled_dot_product_attention,
    scaled_dot_product_attention_masked,
    scaled_dot_product_attention_backward,
    scaled_dot_product_attention_backward_masked,
    ScaledDotProductAttentionBackwardResult,
    create_causal_mask,
    multi_head_attention,
    multi_head_attention_masked,
    multi_head_attention_backward,
    MultiHeadAttentionWeights,
    MultiHeadAttentionResult,
    MultiHeadAttentionBackwardResult,
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
    BroadcastIterator,
)

# ============================================================================
# Initialization Functions
# ============================================================================

from .initializers import (
    xavier_uniform,
    xavier_normal,
    kaiming_uniform,
    kaiming_normal,
    he_uniform,
    he_normal,
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
    smooth_l1_loss,
    hinge_loss,
    focal_loss,
    kl_divergence,
    binary_cross_entropy_backward,
    mean_squared_error_backward,
    cross_entropy_backward,
    smooth_l1_loss_backward,
    hinge_loss_backward,
    focal_loss_backward,
    kl_divergence_backward,
)

from .loss_utils import (
    clip_predictions,
    create_epsilon_tensor,
    validate_tensor_shapes,
    validate_tensor_dtypes,
    compute_one_minus_tensor,
    compute_sign_tensor,
    blend_tensors,
    compute_max_stable,
    compute_difference,
    compute_product,
    compute_ratio,
    negate_tensor,
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
    dispatch_softmax,
    dispatch_softmax_backward,
    dispatch_gelu,
    dispatch_gelu_backward,
    dispatch_hard_sigmoid,
    dispatch_hard_sigmoid_backward,
    dispatch_hard_swish,
    dispatch_hard_swish_backward,
    dispatch_hard_tanh,
    dispatch_hard_tanh_backward,
)

# ============================================================================
# Reduction Operations
# ============================================================================

from .reduction import (
    sum,
    mean,
    max_reduce,
    min_reduce,
    variance,
    std,
    median,
    percentile,
    sum_backward,
    mean_backward,
    max_reduce_backward,
    min_reduce_backward,
    variance_backward,
    std_backward,
    median_backward,
    percentile_backward,
)

from .reduction_ops import (
    ReduceOp,
    ReduceBackwardOp,
    SumOp,
    MeanOp,
    MaxOp,
    MinOp,
    SumBackwardOp,
    MeanBackwardOp,
    MaxBackwardOp,
    MinBackwardOp,
)

from .reduction_utils import (
    compute_strides,
    linear_to_coords,
    coords_to_linear,
    map_result_to_input_coords,
    create_result_coords,
)

# ============================================================================
# Utility Functions
# ============================================================================

from .utils import (
    argmax,
    top_k_indices,
    top_k,
    argsort,
)

# ============================================================================
# Tensor Validation Functions
# ============================================================================

from .validation import (
    validate_tensor_shape,
    validate_tensor_dtype,
    validate_matching_tensors,
    validate_2d_input,
    validate_4d_input,
)

# ============================================================================
# Module Interface for Layer Composition
# ============================================================================

from .module import Module

# Note: Mojo does not support Python's __all__ mechanism.
# All imported symbols are automatically available to package consumers.
