"""Activation functions for neural networks.

Provides common activation functions including ReLU family, sigmoid, tanh,
softmax, and GELU with numerically stable implementations and gradient support.

Reference implementations follow PyTorch and TensorFlow conventions:
- ReLU family: max(0, x), max(alpha*x, x), parametric variants
- Sigmoid/Tanh: Numerically stable with input clipping
- Softmax: Log-sum-exp trick for numerical stability
- GELU: Both exact (erf) and approximate (tanh) implementations

Issues covered:
- #238-242: ReLU Family (ReLU, Leaky ReLU, PReLU)
- #243-247: Sigmoid and Tanh
- #248-252: Softmax and GELU
- #253-257: Activations module integration
"""

from math import exp, erf, sqrt, tanh as math_tanh
from collections.vector import DynamicVector
from .extensor import ExTensor
from .reduction import sum as tensor_sum, max as tensor_max


# ============================================================================
# ReLU Family (#238-242)
# ============================================================================


fn relu(tensor: ExTensor) raises -> ExTensor:
    """Apply ReLU (Rectified Linear Unit) activation: max(0, x).

    ReLU zeros out negative values while preserving positive values unchanged.
    This is the most common activation function in deep learning, promoting
    sparse activation patterns.

    Args:
        tensor: Input tensor of any shape

    Returns:
        New tensor with ReLU applied element-wise

    Examples:
        var x = ExTensor(...)  # [-2, -1, 0, 1, 2]
        var y = relu(x)        # [0, 0, 0, 1, 2]
    """
    var result = ExTensor(tensor._shape, tensor._dtype)

    if tensor._dtype == DType.float32:
        for i in range(tensor._numel):
            var val = tensor._data.bitcast[Float32]()[i]
            result._data.bitcast[Float32]()[i] = max(0.0, val)
    elif tensor._dtype == DType.float64:
        for i in range(tensor._numel):
            var val = tensor._data.bitcast[Float64]()[i]
            result._data.bitcast[Float64]()[i] = max(0.0, val)
    else:
        raise Error("relu: only float32 and float64 dtypes supported")

    return result


fn leaky_relu(tensor: ExTensor, alpha: Float64 = 0.01) raises -> ExTensor:
    """Apply Leaky ReLU activation: max(alpha*x, x).

    Leaky ReLU introduces a small slope for negative values to prevent
    "dying ReLU" problem where neurons can become permanently inactive.

    Args:
        tensor: Input tensor of any shape
        alpha: Slope for negative values (default: 0.01)

    Returns:
        New tensor with Leaky ReLU applied element-wise

    Examples:
        var x = ExTensor(...)           # [-2, -1, 0, 1, 2]
        var y = leaky_relu(x, 0.01)     # [-0.02, -0.01, 0, 1, 2]
    """
    var result = ExTensor(tensor._shape, tensor._dtype)

    if tensor._dtype == DType.float32:
        var alpha32 = Float32(alpha)
        for i in range(tensor._numel):
            var val = tensor._data.bitcast[Float32]()[i]
            result._data.bitcast[Float32]()[i] = max(alpha32 * val, val)
    elif tensor._dtype == DType.float64:
        for i in range(tensor._numel):
            var val = tensor._data.bitcast[Float64]()[i]
            result._data.bitcast[Float64]()[i] = max(alpha * val, val)
    else:
        raise Error("leaky_relu: only float32 and float64 dtypes supported")

    return result


fn prelu(tensor: ExTensor, alpha: ExTensor) raises -> ExTensor:
    """Apply PReLU (Parametric ReLU) activation: max(alpha*x, x).

    PReLU is similar to Leaky ReLU but uses learnable parameters for the
    negative slope. Alpha can be a scalar or have the same shape as tensor
    for per-element or per-channel learned slopes.

    Args:
        tensor: Input tensor of any shape
        alpha: Learnable slope parameter (scalar or matching shape)

    Returns:
        New tensor with PReLU applied element-wise

    Raises:
        Error: If alpha shape is incompatible with tensor shape

    Examples:
        var x = ExTensor(...)      # [-2, -1, 0, 1, 2]
        var a = full(x.shape(), 0.25, DType.float32)
        var y = prelu(x, a)        # [-0.5, -0.25, 0, 1, 2]
    """
    # Validate alpha is scalar or compatible shape
    if alpha._numel != 1 and alpha._numel != tensor._numel:
        raise Error("prelu: alpha must be scalar or match tensor shape")

    if tensor._dtype != alpha._dtype:
        raise Error("prelu: tensor and alpha must have same dtype")

    var result = ExTensor(tensor._shape, tensor._dtype)
    var is_scalar = alpha._numel == 1

    if tensor._dtype == DType.float32:
        for i in range(tensor._numel):
            var val = tensor._data.bitcast[Float32]()[i]
            var a = alpha._data.bitcast[Float32]()[0] if is_scalar else alpha._data.bitcast[Float32]()[i]
            result._data.bitcast[Float32]()[i] = max(a * val, val)
    elif tensor._dtype == DType.float64:
        for i in range(tensor._numel):
            var val = tensor._data.bitcast[Float64]()[i]
            var a = alpha._data.bitcast[Float64]()[0] if is_scalar else alpha._data.bitcast[Float64]()[i]
            result._data.bitcast[Float64]()[i] = max(a * val, val)
    else:
        raise Error("prelu: only float32 and float64 dtypes supported")

    return result


# ============================================================================
# Sigmoid and Tanh (#243-247)
# ============================================================================


fn sigmoid(tensor: ExTensor) raises -> ExTensor:
    """Apply sigmoid activation: 1 / (1 + exp(-x)).

    Sigmoid maps inputs to (0, 1) range. Uses numerically stable implementation
    with input clipping to prevent overflow in exp() computation.

    For large |x| (>20), uses approximations:
    - x > 20: sigmoid(x) ≈ 1.0
    - x < -20: sigmoid(x) ≈ 0.0

    Args:
        tensor: Input tensor of any shape

    Returns:
        New tensor with sigmoid applied element-wise, values in (0, 1)

    Examples:
        var x = ExTensor(...)  # [-2, 0, 2]
        var y = sigmoid(x)     # [0.119, 0.5, 0.881]
    """
    var result = ExTensor(tensor._shape, tensor._dtype)

    if tensor._dtype == DType.float32:
        for i in range(tensor._numel):
            var x = tensor._data.bitcast[Float32]()[i]
            var sig: Float32

            # Numerically stable sigmoid with clipping
            if x > 20.0:
                sig = 1.0
            elif x < -20.0:
                sig = 0.0
            else:
                sig = 1.0 / (1.0 + exp(-x))

            result._data.bitcast[Float32]()[i] = sig
    elif tensor._dtype == DType.float64:
        for i in range(tensor._numel):
            var x = tensor._data.bitcast[Float64]()[i]
            var sig: Float64

            # Numerically stable sigmoid with clipping
            if x > 20.0:
                sig = 1.0
            elif x < -20.0:
                sig = 0.0
            else:
                sig = 1.0 / (1.0 + exp(-x))

            result._data.bitcast[Float64]()[i] = sig
    else:
        raise Error("sigmoid: only float32 and float64 dtypes supported")

    return result


fn tanh(tensor: ExTensor) raises -> ExTensor:
    """Apply tanh (hyperbolic tangent) activation.

    Tanh maps inputs to (-1, 1) range. This is a numerically stable
    implementation that leverages the relationship: tanh(x) = 2*sigmoid(2x) - 1.

    Args:
        tensor: Input tensor of any shape

    Returns:
        New tensor with tanh applied element-wise, values in (-1, 1)

    Examples:
        var x = ExTensor(...)  # [-2, 0, 2]
        var y = tanh(x)        # [-0.964, 0, 0.964]
    """
    var result = ExTensor(tensor._shape, tensor._dtype)

    if tensor._dtype == DType.float32:
        for i in range(tensor._numel):
            var x = tensor._data.bitcast[Float32]()[i]
            result._data.bitcast[Float32]()[i] = math_tanh(x)
    elif tensor._dtype == DType.float64:
        for i in range(tensor._numel):
            var x = tensor._data.bitcast[Float64]()[i]
            result._data.bitcast[Float64]()[i] = math_tanh(x)
    else:
        raise Error("tanh: only float32 and float64 dtypes supported")

    return result


# ============================================================================
# Softmax and GELU (#248-252)
# ============================================================================


fn softmax(tensor: ExTensor, axis: Int = -1) raises -> ExTensor:
    """Apply softmax activation: exp(x) / sum(exp(x)) along specified axis.

    Softmax converts logits to probability distribution. Uses log-sum-exp trick
    for numerical stability by subtracting max value before exponentiation.

    Outputs sum to 1.0 along the specified axis.

    Args:
        tensor: Input tensor (logits)
        axis: Axis along which to compute softmax (default: -1, last axis)

    Returns:
        New tensor with softmax applied, values sum to 1.0 along axis

    Raises:
        Error: If axis is out of bounds

    Examples:
        var logits = ExTensor(...)  # [[1, 2, 3], [4, 5, 6]]
        var probs = softmax(logits, axis=-1)
        # [[0.09, 0.24, 0.67], [0.09, 0.24, 0.67]]
    """
    # Normalize axis
    var ndim = len(tensor._shape)
    var norm_axis = axis if axis >= 0 else ndim + axis

    if norm_axis < 0 or norm_axis >= ndim:
        raise Error("softmax: axis out of bounds")

    # For simplicity, implement for last axis first
    # TODO: Support arbitrary axis with proper reduction
    if norm_axis != ndim - 1:
        raise Error("softmax: only last axis currently supported")

    var result = ExTensor(tensor._shape, tensor._dtype)

    # Calculate outer size (all dimensions before axis) and inner size (axis dimension)
    var outer_size = 1
    for i in range(norm_axis):
        outer_size *= tensor._shape[i]
    var inner_size = tensor._shape[norm_axis]

    if tensor._dtype == DType.float32:
        for outer_idx in range(outer_size):
            var base_idx = outer_idx * inner_size

            # Find max value for numerical stability
            var max_val: Float32 = tensor._data.bitcast[Float32]()[base_idx]
            for i in range(1, inner_size):
                var val = tensor._data.bitcast[Float32]()[base_idx + i]
                if val > max_val:
                    max_val = val

            # Compute exp(x - max) and sum
            var sum_exp: Float32 = 0.0
            for i in range(inner_size):
                var val = tensor._data.bitcast[Float32]()[base_idx + i]
                var exp_val = exp(val - max_val)
                result._data.bitcast[Float32]()[base_idx + i] = exp_val
                sum_exp += exp_val

            # Normalize by sum
            for i in range(inner_size):
                result._data.bitcast[Float32]()[base_idx + i] /= sum_exp

    elif tensor._dtype == DType.float64:
        for outer_idx in range(outer_size):
            var base_idx = outer_idx * inner_size

            # Find max value for numerical stability
            var max_val: Float64 = tensor._data.bitcast[Float64]()[base_idx]
            for i in range(1, inner_size):
                var val = tensor._data.bitcast[Float64]()[base_idx + i]
                if val > max_val:
                    max_val = val

            # Compute exp(x - max) and sum
            var sum_exp: Float64 = 0.0
            for i in range(inner_size):
                var val = tensor._data.bitcast[Float64]()[base_idx + i]
                var exp_val = exp(val - max_val)
                result._data.bitcast[Float64]()[base_idx + i] = exp_val
                sum_exp += exp_val

            # Normalize by sum
            for i in range(inner_size):
                result._data.bitcast[Float64]()[base_idx + i] /= sum_exp
    else:
        raise Error("softmax: only float32 and float64 dtypes supported")

    return result


fn gelu(tensor: ExTensor, approximate: Bool = False) raises -> ExTensor:
    """Apply GELU (Gaussian Error Linear Unit) activation.

    GELU provides smooth, non-linear activation used in transformers (BERT, GPT).

    Exact formula: GELU(x) = x * Φ(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
    Approximate formula: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))

    The approximate version is faster and was used in the original BERT implementation.

    Args:
        tensor: Input tensor of any shape
        approximate: Use tanh approximation (True) or exact erf (False)

    Returns:
        New tensor with GELU applied element-wise

    Examples:
        var x = ExTensor(...)     # [-2, 0, 2]
        var y_exact = gelu(x, approximate=False)
        var y_approx = gelu(x, approximate=True)
    """
    var result = ExTensor(tensor._shape, tensor._dtype)

    # Constants for approximate GELU
    alias SQRT_2_OVER_PI = 0.7978845608028654  # sqrt(2/π)
    alias GELU_COEFF = 0.044715

    if tensor._dtype == DType.float32:
        if approximate:
            # Approximate: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
            for i in range(tensor._numel):
                var x = tensor._data.bitcast[Float32]()[i]
                var x_cubed = x * x * x
                var inner = Float32(SQRT_2_OVER_PI) * (x + Float32(GELU_COEFF) * x_cubed)
                var tanh_val = math_tanh(inner)
                result._data.bitcast[Float32]()[i] = 0.5 * x * (1.0 + tanh_val)
        else:
            # Exact: x * 0.5 * (1 + erf(x / sqrt(2)))
            alias SQRT_2 = 1.4142135623730951
            for i in range(tensor._numel):
                var x = tensor._data.bitcast[Float32]()[i]
                var erf_val = erf(x / Float32(SQRT_2))
                result._data.bitcast[Float32]()[i] = x * 0.5 * (1.0 + erf_val)

    elif tensor._dtype == DType.float64:
        if approximate:
            # Approximate: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
            for i in range(tensor._numel):
                var x = tensor._data.bitcast[Float64]()[i]
                var x_cubed = x * x * x
                var inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x_cubed)
                var tanh_val = math_tanh(inner)
                result._data.bitcast[Float64]()[i] = 0.5 * x * (1.0 + tanh_val)
        else:
            # Exact: x * 0.5 * (1 + erf(x / sqrt(2)))
            alias SQRT_2 = 1.4142135623730951
            for i in range(tensor._numel):
                var x = tensor._data.bitcast[Float64]()[i]
                var erf_val = erf(x / SQRT_2)
                result._data.bitcast[Float64]()[i] = x * 0.5 * (1.0 + erf_val)
    else:
        raise Error("gelu: only float32 and float64 dtypes supported")

    return result
