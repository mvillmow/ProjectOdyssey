"""Activation functions for neural networks.

Provides common activation functions including ReLU family, sigmoid, tanh,
softmax, and GELU with numerically stable implementations and gradient support.

Reference implementations follow PyTorch and TensorFlow conventions:
- ReLU family: max(0, x), max(alpha*x, x), parametric variants
- Sigmoid/Tanh: Numerically stable with input clipping
- Softmax: Log-sum-exp trick for numerical stability
- GELU: Both exact (erf) and approximate (tanh) implementations

Type support:
- ReLU family: float16, float32, float64, int8, int16, int32, int64, uint8, uint16, uint32, uint64
- Sigmoid/Tanh/Softmax/GELU: float16, float32, float64

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

    Supported dtypes: float16, float32, float64, int8, int16, int32, int64,
                      uint8, uint16, uint32, uint64

    Args:
        tensor: Input tensor of any shape

    Returns:
        New tensor with ReLU applied element-wise

    Examples:
        var x = ExTensor(...)  # [-2, -1, 0, 1, 2]
        var y = relu(x)        # [0, 0, 0, 1, 2]
    """
    var result = ExTensor(tensor._shape, tensor._dtype)

    if tensor._dtype == DType.float16:
        for i in range(tensor._numel):
            var val = tensor._data.bitcast[Float16]()[i]
            result._data.bitcast[Float16]()[i] = max(Float16(0.0), val)
    elif tensor._dtype == DType.float32:
        for i in range(tensor._numel):
            var val = tensor._data.bitcast[Float32]()[i]
            result._data.bitcast[Float32]()[i] = max(0.0, val)
    elif tensor._dtype == DType.float64:
        for i in range(tensor._numel):
            var val = tensor._data.bitcast[Float64]()[i]
            result._data.bitcast[Float64]()[i] = max(0.0, val)
    elif tensor._dtype == DType.int8:
        for i in range(tensor._numel):
            var val = tensor._data.bitcast[Int8]()[i]
            result._data.bitcast[Int8]()[i] = max(Int8(0), val)
    elif tensor._dtype == DType.int16:
        for i in range(tensor._numel):
            var val = tensor._data.bitcast[Int16]()[i]
            result._data.bitcast[Int16]()[i] = max(Int16(0), val)
    elif tensor._dtype == DType.int32:
        for i in range(tensor._numel):
            var val = tensor._data.bitcast[Int32]()[i]
            result._data.bitcast[Int32]()[i] = max(0, val)
    elif tensor._dtype == DType.int64:
        for i in range(tensor._numel):
            var val = tensor._data.bitcast[Int64]()[i]
            result._data.bitcast[Int64]()[i] = max(0, val)
    elif tensor._dtype == DType.uint8:
        # Unsigned types are already >= 0, just copy
        for i in range(tensor._numel):
            result._data.bitcast[UInt8]()[i] = tensor._data.bitcast[UInt8]()[i]
    elif tensor._dtype == DType.uint16:
        for i in range(tensor._numel):
            result._data.bitcast[UInt16]()[i] = tensor._data.bitcast[UInt16]()[i]
    elif tensor._dtype == DType.uint32:
        for i in range(tensor._numel):
            result._data.bitcast[UInt32]()[i] = tensor._data.bitcast[UInt32]()[i]
    elif tensor._dtype == DType.uint64:
        for i in range(tensor._numel):
            result._data.bitcast[UInt64]()[i] = tensor._data.bitcast[UInt64]()[i]
    else:
        raise Error("relu: unsupported dtype")

    return result


fn leaky_relu(tensor: ExTensor, alpha: Float64 = 0.01) raises -> ExTensor:
    """Apply Leaky ReLU activation: max(alpha*x, x).

    Leaky ReLU introduces a small slope for negative values to prevent
    "dying ReLU" problem where neurons can become permanently inactive.

    Supported dtypes: float16, float32, float64, int8, int16, int32, int64

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

    if tensor._dtype == DType.float16:
        var alpha16 = Float16(alpha)
        for i in range(tensor._numel):
            var val = tensor._data.bitcast[Float16]()[i]
            result._data.bitcast[Float16]()[i] = max(alpha16 * val, val)
    elif tensor._dtype == DType.float32:
        var alpha32 = Float32(alpha)
        for i in range(tensor._numel):
            var val = tensor._data.bitcast[Float32]()[i]
            result._data.bitcast[Float32]()[i] = max(alpha32 * val, val)
    elif tensor._dtype == DType.float64:
        for i in range(tensor._numel):
            var val = tensor._data.bitcast[Float64]()[i]
            result._data.bitcast[Float64]()[i] = max(alpha * val, val)
    elif tensor._dtype == DType.int8:
        var alpha_scaled = Int8(alpha * 128)  # Scale for fixed-point
        for i in range(tensor._numel):
            var val = tensor._data.bitcast[Int8]()[i]
            var scaled = (alpha_scaled * val) >> 7  # Divide by 128
            result._data.bitcast[Int8]()[i] = max(scaled, val)
    elif tensor._dtype == DType.int16:
        var alpha_scaled = Int16(alpha * 32768)
        for i in range(tensor._numel):
            var val = tensor._data.bitcast[Int16]()[i]
            var scaled = (alpha_scaled * val) >> 15
            result._data.bitcast[Int16]()[i] = max(scaled, val)
    elif tensor._dtype == DType.int32:
        var alpha32 = Float32(alpha)
        for i in range(tensor._numel):
            var val = tensor._data.bitcast[Int32]()[i]
            var scaled = Int32(alpha32 * Float32(val))
            result._data.bitcast[Int32]()[i] = max(scaled, val)
    elif tensor._dtype == DType.int64:
        for i in range(tensor._numel):
            var val = tensor._data.bitcast[Int64]()[i]
            var scaled = Int64(alpha * Float64(val))
            result._data.bitcast[Int64]()[i] = max(scaled, val)
    else:
        raise Error("leaky_relu: unsupported dtype (use float16/32/64 or int8/16/32/64)")

    return result


fn prelu(tensor: ExTensor, alpha: ExTensor) raises -> ExTensor:
    """Apply PReLU (Parametric ReLU) activation: max(alpha*x, x).

    PReLU is similar to Leaky ReLU but uses learnable parameters for the
    negative slope. Alpha can be a scalar or have the same shape as tensor
    for per-element or per-channel learned slopes.

    Supported dtypes: float16, float32, float64

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

    if tensor._dtype == DType.float16:
        for i in range(tensor._numel):
            var val = tensor._data.bitcast[Float16]()[i]
            var a = alpha._data.bitcast[Float16]()[0] if is_scalar else alpha._data.bitcast[Float16]()[i]
            result._data.bitcast[Float16]()[i] = max(a * val, val)
    elif tensor._dtype == DType.float32:
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
        raise Error("prelu: only float16, float32, and float64 dtypes supported")

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

    Supported dtypes: float16, float32, float64

    Args:
        tensor: Input tensor of any shape

    Returns:
        New tensor with sigmoid applied element-wise, values in (0, 1)

    Examples:
        var x = ExTensor(...)  # [-2, 0, 2]
        var y = sigmoid(x)     # [0.119, 0.5, 0.881]
    """
    var result = ExTensor(tensor._shape, tensor._dtype)

    if tensor._dtype == DType.float16:
        for i in range(tensor._numel):
            var x = tensor._data.bitcast[Float16]()[i]
            var sig: Float16

            # Numerically stable sigmoid with clipping
            if x > Float16(20.0):
                sig = Float16(1.0)
            elif x < Float16(-20.0):
                sig = Float16(0.0)
            else:
                sig = Float16(1.0) / (Float16(1.0) + exp(-Float32(x)))

            result._data.bitcast[Float16]()[i] = sig
    elif tensor._dtype == DType.float32:
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
        raise Error("sigmoid: only float16, float32, and float64 dtypes supported")

    return result


fn tanh(tensor: ExTensor) raises -> ExTensor:
    """Apply tanh (hyperbolic tangent) activation.

    Tanh maps inputs to (-1, 1) range. This is a numerically stable
    implementation that leverages the math library tanh function.

    Supported dtypes: float16, float32, float64

    Args:
        tensor: Input tensor of any shape

    Returns:
        New tensor with tanh applied element-wise, values in (-1, 1)

    Examples:
        var x = ExTensor(...)  # [-2, 0, 2]
        var y = tanh(x)        # [-0.964, 0, 0.964]
    """
    var result = ExTensor(tensor._shape, tensor._dtype)

    if tensor._dtype == DType.float16:
        for i in range(tensor._numel):
            var x = tensor._data.bitcast[Float16]()[i]
            result._data.bitcast[Float16]()[i] = Float16(math_tanh(Float32(x)))
    elif tensor._dtype == DType.float32:
        for i in range(tensor._numel):
            var x = tensor._data.bitcast[Float32]()[i]
            result._data.bitcast[Float32]()[i] = math_tanh(x)
    elif tensor._dtype == DType.float64:
        for i in range(tensor._numel):
            var x = tensor._data.bitcast[Float64]()[i]
            result._data.bitcast[Float64]()[i] = math_tanh(x)
    else:
        raise Error("tanh: only float16, float32, and float64 dtypes supported")

    return result


# ============================================================================
# Softmax and GELU (#248-252)
# ============================================================================


fn softmax(tensor: ExTensor, axis: Int = -1) raises -> ExTensor:
    """Apply softmax activation: exp(x) / sum(exp(x)) along specified axis.

    Softmax converts logits to probability distribution. Uses log-sum-exp trick
    for numerical stability by subtracting max value before exponentiation.

    Outputs sum to 1.0 along the specified axis.

    Supported dtypes: float16, float32, float64

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

    if tensor._dtype == DType.float16:
        for outer_idx in range(outer_size):
            var base_idx = outer_idx * inner_size

            # Find max value for numerical stability
            var max_val: Float16 = tensor._data.bitcast[Float16]()[base_idx]
            for i in range(1, inner_size):
                var val = tensor._data.bitcast[Float16]()[base_idx + i]
                if val > max_val:
                    max_val = val

            # Compute exp(x - max) and sum
            var sum_exp: Float32 = 0.0
            for i in range(inner_size):
                var val = tensor._data.bitcast[Float16]()[base_idx + i]
                var exp_val = exp(Float32(val - max_val))
                result._data.bitcast[Float16]()[base_idx + i] = Float16(exp_val)
                sum_exp += exp_val

            # Normalize by sum
            for i in range(inner_size):
                var current = Float32(result._data.bitcast[Float16]()[base_idx + i])
                result._data.bitcast[Float16]()[base_idx + i] = Float16(current / sum_exp)

    elif tensor._dtype == DType.float32:
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
        raise Error("softmax: only float16, float32, and float64 dtypes supported")

    return result


fn gelu(tensor: ExTensor, approximate: Bool = False) raises -> ExTensor:
    """Apply GELU (Gaussian Error Linear Unit) activation.

    GELU provides smooth, non-linear activation used in transformers (BERT, GPT).

    Exact formula: GELU(x) = x * Φ(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
    Approximate formula: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))

    The approximate version is faster and was used in the original BERT implementation.

    Supported dtypes: float16, float32, float64

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
    alias SQRT_2 = 1.4142135623730951

    if tensor._dtype == DType.float16:
        if approximate:
            # Approximate: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
            for i in range(tensor._numel):
                var x = Float32(tensor._data.bitcast[Float16]()[i])
                var x_cubed = x * x * x
                var inner = Float32(SQRT_2_OVER_PI) * (x + Float32(GELU_COEFF) * x_cubed)
                var tanh_val = math_tanh(inner)
                result._data.bitcast[Float16]()[i] = Float16(0.5 * x * (1.0 + tanh_val))
        else:
            # Exact: x * 0.5 * (1 + erf(x / sqrt(2)))
            for i in range(tensor._numel):
                var x = Float32(tensor._data.bitcast[Float16]()[i])
                var erf_val = erf(x / Float32(SQRT_2))
                result._data.bitcast[Float16]()[i] = Float16(x * 0.5 * (1.0 + erf_val))

    elif tensor._dtype == DType.float32:
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
            for i in range(tensor._numel):
                var x = tensor._data.bitcast[Float64]()[i]
                var erf_val = erf(x / SQRT_2)
                result._data.bitcast[Float64]()[i] = x * 0.5 * (1.0 + erf_val)
    else:
        raise Error("gelu: only float16, float32, and float64 dtypes supported")

    return result


# ============================================================================
# Backward Pass (Gradient Computation)
# ============================================================================


fn relu_backward(grad_output: ExTensor, x: ExTensor) raises -> ExTensor:
    """Compute gradient of ReLU activation.

    ReLU gradient: ∂L/∂x = ∂L/∂y * (x > 0)

    Args:
        grad_output: Gradient from upstream (∂L/∂y)
        x: Input tensor from forward pass

    Returns:
        Gradient with respect to input (∂L/∂x)

    Examples:
        var x = ExTensor(...)  # Input
        var y = relu(x)        # Forward pass
        var grad_y = ExTensor(...)  # Gradient from loss
        var grad_x = relu_backward(grad_y, x)  # Backward pass
    """
    if grad_output._dtype != x._dtype:
        raise Error("relu_backward: grad_output and x must have same dtype")
    if grad_output._numel != x._numel:
        raise Error("relu_backward: grad_output and x must have same shape")

    var result = ExTensor(x._shape, x._dtype)

    if x._dtype == DType.float16:
        for i in range(x._numel):
            var x_val = x._data.bitcast[Float16]()[i]
            var grad = grad_output._data.bitcast[Float16]()[i]
            result._data.bitcast[Float16]()[i] = grad if x_val > Float16(0) else Float16(0)
    elif x._dtype == DType.float32:
        for i in range(x._numel):
            var x_val = x._data.bitcast[Float32]()[i]
            var grad = grad_output._data.bitcast[Float32]()[i]
            result._data.bitcast[Float32]()[i] = grad if x_val > 0.0 else 0.0
    elif x._dtype == DType.float64:
        for i in range(x._numel):
            var x_val = x._data.bitcast[Float64]()[i]
            var grad = grad_output._data.bitcast[Float64]()[i]
            result._data.bitcast[Float64]()[i] = grad if x_val > 0.0 else 0.0
    elif x._dtype == DType.int32:
        for i in range(x._numel):
            var x_val = x._data.bitcast[Int32]()[i]
            var grad = grad_output._data.bitcast[Int32]()[i]
            result._data.bitcast[Int32]()[i] = grad if x_val > 0 else 0
    elif x._dtype == DType.int64:
        for i in range(x._numel):
            var x_val = x._data.bitcast[Int64]()[i]
            var grad = grad_output._data.bitcast[Int64]()[i]
            result._data.bitcast[Int64]()[i] = grad if x_val > 0 else 0
    else:
        raise Error("relu_backward: only float16/32/64 and int32/64 supported for gradients")

    return result


fn leaky_relu_backward(grad_output: ExTensor, x: ExTensor, alpha: Float64 = 0.01) raises -> ExTensor:
    """Compute gradient of Leaky ReLU activation.

    Leaky ReLU gradient: ∂L/∂x = ∂L/∂y * (1 if x > 0 else alpha)

    Args:
        grad_output: Gradient from upstream (∂L/∂y)
        x: Input tensor from forward pass
        alpha: Slope for negative values (default: 0.01)

    Returns:
        Gradient with respect to input (∂L/∂x)
    """
    if grad_output._dtype != x._dtype:
        raise Error("leaky_relu_backward: grad_output and x must have same dtype")
    if grad_output._numel != x._numel:
        raise Error("leaky_relu_backward: grad_output and x must have same shape")

    var result = ExTensor(x._shape, x._dtype)

    if x._dtype == DType.float16:
        var alpha16 = Float16(alpha)
        for i in range(x._numel):
            var x_val = x._data.bitcast[Float16]()[i]
            var grad = grad_output._data.bitcast[Float16]()[i]
            result._data.bitcast[Float16]()[i] = grad if x_val > Float16(0) else grad * alpha16
    elif x._dtype == DType.float32:
        var alpha32 = Float32(alpha)
        for i in range(x._numel):
            var x_val = x._data.bitcast[Float32]()[i]
            var grad = grad_output._data.bitcast[Float32]()[i]
            result._data.bitcast[Float32]()[i] = grad if x_val > 0.0 else grad * alpha32
    elif x._dtype == DType.float64:
        for i in range(x._numel):
            var x_val = x._data.bitcast[Float64]()[i]
            var grad = grad_output._data.bitcast[Float64]()[i]
            result._data.bitcast[Float64]()[i] = grad if x_val > 0.0 else grad * alpha
    else:
        raise Error("leaky_relu_backward: only float16/32/64 dtypes supported")

    return result


fn prelu_backward(grad_output: ExTensor, x: ExTensor, alpha: ExTensor) raises -> (ExTensor, ExTensor):
    """Compute gradients of PReLU activation.

    PReLU gradients:
    - ∂L/∂x = ∂L/∂y * (1 if x > 0 else alpha)
    - ∂L/∂alpha = sum(∂L/∂y * x) where x <= 0

    Args:
        grad_output: Gradient from upstream (∂L/∂y)
        x: Input tensor from forward pass
        alpha: Learnable slope parameter

    Returns:
        Tuple of (grad_input, grad_alpha)
    """
    if grad_output._dtype != x._dtype or grad_output._dtype != alpha._dtype:
        raise Error("prelu_backward: all tensors must have same dtype")
    if grad_output._numel != x._numel:
        raise Error("prelu_backward: grad_output and x must have same shape")

    var grad_input = ExTensor(x._shape, x._dtype)
    var grad_alpha = ExTensor(alpha._shape, alpha._dtype)
    var is_scalar = alpha._numel == 1

    # Initialize grad_alpha to zero
    for i in range(grad_alpha._numel):
        if grad_alpha._dtype == DType.float16:
            grad_alpha._data.bitcast[Float16]()[i] = Float16(0)
        elif grad_alpha._dtype == DType.float32:
            grad_alpha._data.bitcast[Float32]()[i] = 0.0
        elif grad_alpha._dtype == DType.float64:
            grad_alpha._data.bitcast[Float64]()[i] = 0.0

    if x._dtype == DType.float16:
        for i in range(x._numel):
            var x_val = x._data.bitcast[Float16]()[i]
            var grad = grad_output._data.bitcast[Float16]()[i]
            var a = alpha._data.bitcast[Float16]()[0] if is_scalar else alpha._data.bitcast[Float16]()[i]

            if x_val > Float16(0):
                grad_input._data.bitcast[Float16]()[i] = grad
            else:
                grad_input._data.bitcast[Float16]()[i] = grad * a
                # Accumulate gradient for alpha
                var alpha_idx = 0 if is_scalar else i
                grad_alpha._data.bitcast[Float16]()[alpha_idx] += grad * x_val

    elif x._dtype == DType.float32:
        for i in range(x._numel):
            var x_val = x._data.bitcast[Float32]()[i]
            var grad = grad_output._data.bitcast[Float32]()[i]
            var a = alpha._data.bitcast[Float32]()[0] if is_scalar else alpha._data.bitcast[Float32]()[i]

            if x_val > 0.0:
                grad_input._data.bitcast[Float32]()[i] = grad
            else:
                grad_input._data.bitcast[Float32]()[i] = grad * a
                var alpha_idx = 0 if is_scalar else i
                grad_alpha._data.bitcast[Float32]()[alpha_idx] += grad * x_val

    elif x._dtype == DType.float64:
        for i in range(x._numel):
            var x_val = x._data.bitcast[Float64]()[i]
            var grad = grad_output._data.bitcast[Float64]()[i]
            var a = alpha._data.bitcast[Float64]()[0] if is_scalar else alpha._data.bitcast[Float64]()[i]

            if x_val > 0.0:
                grad_input._data.bitcast[Float64]()[i] = grad
            else:
                grad_input._data.bitcast[Float64]()[i] = grad * a
                var alpha_idx = 0 if is_scalar else i
                grad_alpha._data.bitcast[Float64]()[alpha_idx] += grad * x_val
    else:
        raise Error("prelu_backward: only float16/32/64 dtypes supported")

    return (grad_input, grad_alpha)


fn sigmoid_backward(grad_output: ExTensor, output: ExTensor) raises -> ExTensor:
    """Compute gradient of sigmoid activation.

    Sigmoid gradient: ∂L/∂x = ∂L/∂y * y * (1 - y)
    where y = sigmoid(x)

    Args:
        grad_output: Gradient from upstream (∂L/∂y)
        output: Output from forward pass (sigmoid(x))

    Returns:
        Gradient with respect to input (∂L/∂x)

    Note:
        Takes output instead of input to avoid recomputing sigmoid.
    """
    if grad_output._dtype != output._dtype:
        raise Error("sigmoid_backward: grad_output and output must have same dtype")
    if grad_output._numel != output._numel:
        raise Error("sigmoid_backward: grad_output and output must have same shape")

    var result = ExTensor(output._shape, output._dtype)

    if output._dtype == DType.float16:
        for i in range(output._numel):
            var y = output._data.bitcast[Float16]()[i]
            var grad = grad_output._data.bitcast[Float16]()[i]
            result._data.bitcast[Float16]()[i] = grad * y * (Float16(1.0) - y)
    elif output._dtype == DType.float32:
        for i in range(output._numel):
            var y = output._data.bitcast[Float32]()[i]
            var grad = grad_output._data.bitcast[Float32]()[i]
            result._data.bitcast[Float32]()[i] = grad * y * (1.0 - y)
    elif output._dtype == DType.float64:
        for i in range(output._numel):
            var y = output._data.bitcast[Float64]()[i]
            var grad = grad_output._data.bitcast[Float64]()[i]
            result._data.bitcast[Float64]()[i] = grad * y * (1.0 - y)
    else:
        raise Error("sigmoid_backward: only float16/32/64 dtypes supported")

    return result


fn tanh_backward(grad_output: ExTensor, output: ExTensor) raises -> ExTensor:
    """Compute gradient of tanh activation.

    Tanh gradient: ∂L/∂x = ∂L/∂y * (1 - y²)
    where y = tanh(x)

    Args:
        grad_output: Gradient from upstream (∂L/∂y)
        output: Output from forward pass (tanh(x))

    Returns:
        Gradient with respect to input (∂L/∂x)

    Note:
        Takes output instead of input to avoid recomputing tanh.
    """
    if grad_output._dtype != output._dtype:
        raise Error("tanh_backward: grad_output and output must have same dtype")
    if grad_output._numel != output._numel:
        raise Error("tanh_backward: grad_output and output must have same shape")

    var result = ExTensor(output._shape, output._dtype)

    if output._dtype == DType.float16:
        for i in range(output._numel):
            var y = output._data.bitcast[Float16]()[i]
            var grad = grad_output._data.bitcast[Float16]()[i]
            result._data.bitcast[Float16]()[i] = grad * (Float16(1.0) - y * y)
    elif output._dtype == DType.float32:
        for i in range(output._numel):
            var y = output._data.bitcast[Float32]()[i]
            var grad = grad_output._data.bitcast[Float32]()[i]
            result._data.bitcast[Float32]()[i] = grad * (1.0 - y * y)
    elif output._dtype == DType.float64:
        for i in range(output._numel):
            var y = output._data.bitcast[Float64]()[i]
            var grad = grad_output._data.bitcast[Float64]()[i]
            result._data.bitcast[Float64]()[i] = grad * (1.0 - y * y)
    else:
        raise Error("tanh_backward: only float16/32/64 dtypes supported")

    return result


fn gelu_backward(grad_output: ExTensor, x: ExTensor, approximate: Bool = False) raises -> ExTensor:
    """Compute gradient of GELU activation.

    GELU gradient (exact): ∂L/∂x = ∂L/∂y * [Φ(x) + x*φ(x)]
    where Φ is CDF and φ is PDF of standard normal

    GELU gradient (approximate): Uses derivative of tanh approximation

    Args:
        grad_output: Gradient from upstream (∂L/∂y)
        x: Input tensor from forward pass
        approximate: Use tanh approximation (True) or exact erf (False)

    Returns:
        Gradient with respect to input (∂L/∂x)
    """
    if grad_output._dtype != x._dtype:
        raise Error("gelu_backward: grad_output and x must have same dtype")
    if grad_output._numel != x._numel:
        raise Error("gelu_backward: grad_output and x must have same shape")

    var result = ExTensor(x._shape, x._dtype)

    alias SQRT_2 = 1.4142135623730951
    alias SQRT_2_OVER_PI = 0.7978845608028654
    alias GELU_COEFF = 0.044715
    alias INV_SQRT_2PI = 0.3989422804014327  # 1/sqrt(2π)

    if x._dtype == DType.float32:
        if approximate:
            # d/dx[0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))]
            for i in range(x._numel):
                var x_val = x._data.bitcast[Float32]()[i]
                var grad = grad_output._data.bitcast[Float32]()[i]

                var x_cubed = x_val * x_val * x_val
                var inner = Float32(SQRT_2_OVER_PI) * (x_val + Float32(GELU_COEFF) * x_cubed)
                var tanh_val = math_tanh(inner)
                var sech2 = 1.0 - tanh_val * tanh_val

                # Derivative computation
                var dtanh = Float32(SQRT_2_OVER_PI) * (1.0 + 3.0 * Float32(GELU_COEFF) * x_val * x_val)
                var dgelu = 0.5 * (1.0 + tanh_val) + 0.5 * x_val * sech2 * dtanh

                result._data.bitcast[Float32]()[i] = grad * dgelu
        else:
            # d/dx[x * 0.5 * (1 + erf(x / sqrt(2)))] = 0.5 * [1 + erf(x/√2)] + x * φ(x)
            for i in range(x._numel):
                var x_val = x._data.bitcast[Float32]()[i]
                var grad = grad_output._data.bitcast[Float32]()[i]

                var erf_val = erf(x_val / Float32(SQRT_2))
                var pdf = Float32(INV_SQRT_2PI) * exp(-0.5 * x_val * x_val)  # φ(x)
                var dgelu = 0.5 * (1.0 + erf_val) + x_val * pdf

                result._data.bitcast[Float32]()[i] = grad * dgelu

    elif x._dtype == DType.float64:
        if approximate:
            for i in range(x._numel):
                var x_val = x._data.bitcast[Float64]()[i]
                var grad = grad_output._data.bitcast[Float64]()[i]

                var x_cubed = x_val * x_val * x_val
                var inner = SQRT_2_OVER_PI * (x_val + GELU_COEFF * x_cubed)
                var tanh_val = math_tanh(inner)
                var sech2 = 1.0 - tanh_val * tanh_val

                var dtanh = SQRT_2_OVER_PI * (1.0 + 3.0 * GELU_COEFF * x_val * x_val)
                var dgelu = 0.5 * (1.0 + tanh_val) + 0.5 * x_val * sech2 * dtanh

                result._data.bitcast[Float64]()[i] = grad * dgelu
        else:
            for i in range(x._numel):
                var x_val = x._data.bitcast[Float64]()[i]
                var grad = grad_output._data.bitcast[Float64]()[i]

                var erf_val = erf(x_val / SQRT_2)
                var pdf = INV_SQRT_2PI * exp(-0.5 * x_val * x_val)
                var dgelu = 0.5 * (1.0 + erf_val) + x_val * pdf

                result._data.bitcast[Float64]()[i] = grad * dgelu

    elif x._dtype == DType.float16:
        # Use float32 intermediate precision
        if approximate:
            for i in range(x._numel):
                var x_val = Float32(x._data.bitcast[Float16]()[i])
                var grad = Float32(grad_output._data.bitcast[Float16]()[i])

                var x_cubed = x_val * x_val * x_val
                var inner = Float32(SQRT_2_OVER_PI) * (x_val + Float32(GELU_COEFF) * x_cubed)
                var tanh_val = math_tanh(inner)
                var sech2 = 1.0 - tanh_val * tanh_val

                var dtanh = Float32(SQRT_2_OVER_PI) * (1.0 + 3.0 * Float32(GELU_COEFF) * x_val * x_val)
                var dgelu = 0.5 * (1.0 + tanh_val) + 0.5 * x_val * sech2 * dtanh

                result._data.bitcast[Float16]()[i] = Float16(grad * dgelu)
        else:
            for i in range(x._numel):
                var x_val = Float32(x._data.bitcast[Float16]()[i])
                var grad = Float32(grad_output._data.bitcast[Float16]()[i])

                var erf_val = erf(x_val / Float32(SQRT_2))
                var pdf = Float32(INV_SQRT_2PI) * exp(-0.5 * x_val * x_val)
                var dgelu = 0.5 * (1.0 + erf_val) + x_val * pdf

                result._data.bitcast[Float16]()[i] = Float16(grad * dgelu)
    else:
        raise Error("gelu_backward: only float16/32/64 dtypes supported")

    return result


fn softmax_backward(grad_output: ExTensor, output: ExTensor, axis: Int = -1) raises -> ExTensor:
    """Compute gradient of softmax activation.

    Softmax gradient (along axis):
        ∂L/∂x_i = y_i * (∂L/∂y_i - sum_j(∂L/∂y_j * y_j))

    where y = softmax(x) is the output from the forward pass.

    The gradient takes into account that each output depends on all inputs
    through the normalization term, creating a Jacobian matrix.

    Supported dtypes: float16, float32, float64

    Args:
        grad_output: Gradient from upstream (∂L/∂y)
        output: Softmax output from forward pass (y = softmax(x))
        axis: Axis along which softmax was computed (default: -1)

    Returns:
        Gradient with respect to input (∂L/∂x)

    Raises:
        Error: If dtypes don't match or shapes incompatible

    Examples:
        var x = ExTensor(...)  # Input
        var y = softmax(x)     # Forward pass
        var grad_y = ExTensor(...)  # Gradient from loss
        var grad_x = softmax_backward(grad_y, y)  # Backward pass

    References:
        The gradient formula accounts for the fact that softmax output y_i
        depends on all inputs x_j, not just x_i, due to the normalization.
    """
    if grad_output._dtype != output._dtype:
        raise Error("softmax_backward: grad_output and output must have same dtype")
    if grad_output._numel != output._numel:
        raise Error("softmax_backward: grad_output and output must have same shape")

    var result = ExTensor(output._shape, output._dtype)

    # Normalize axis to positive index
    var ndim = len(output._shape)
    var normalized_axis = axis if axis >= 0 else ndim + axis

    if normalized_axis < 0 or normalized_axis >= ndim:
        raise Error("softmax_backward: axis out of bounds")

    # Calculate stride for the softmax axis
    var axis_size = output._shape[normalized_axis]
    var axis_stride = 1
    for i in range(normalized_axis + 1, ndim):
        axis_stride *= output._shape[i]

    # Calculate outer size (product of dimensions before axis)
    var outer_size = 1
    for i in range(normalized_axis):
        outer_size *= output._shape[i]

    if output._dtype == DType.float32:
        # For each outer position
        for outer in range(outer_size):
            # For each inner position
            for inner in range(axis_stride):
                # Calculate sum(grad_output * output) along axis
                var dot_sum: Float32 = 0.0
                for k in range(axis_size):
                    var idx = (outer * axis_size + k) * axis_stride + inner
                    var grad_val = grad_output._data.bitcast[Float32]()[idx]
                    var out_val = output._data.bitcast[Float32]()[idx]
                    dot_sum += grad_val * out_val

                # Compute gradient for each position along axis
                for k in range(axis_size):
                    var idx = (outer * axis_size + k) * axis_stride + inner
                    var grad_val = grad_output._data.bitcast[Float32]()[idx]
                    var out_val = output._data.bitcast[Float32]()[idx]
                    result._data.bitcast[Float32]()[idx] = out_val * (grad_val - dot_sum)

    elif output._dtype == DType.float64:
        # For each outer position
        for outer in range(outer_size):
            # For each inner position
            for inner in range(axis_stride):
                # Calculate sum(grad_output * output) along axis
                var dot_sum: Float64 = 0.0
                for k in range(axis_size):
                    var idx = (outer * axis_size + k) * axis_stride + inner
                    var grad_val = grad_output._data.bitcast[Float64]()[idx]
                    var out_val = output._data.bitcast[Float64]()[idx]
                    dot_sum += grad_val * out_val

                # Compute gradient for each position along axis
                for k in range(axis_size):
                    var idx = (outer * axis_size + k) * axis_stride + inner
                    var grad_val = grad_output._data.bitcast[Float64]()[idx]
                    var out_val = output._data.bitcast[Float64]()[idx]
                    result._data.bitcast[Float64]()[idx] = out_val * (grad_val - dot_sum)

    elif output._dtype == DType.float16:
        # Use float32 intermediate precision
        for outer in range(outer_size):
            for inner in range(axis_stride):
                # Calculate sum(grad_output * output) along axis
                var dot_sum: Float32 = 0.0
                for k in range(axis_size):
                    var idx = (outer * axis_size + k) * axis_stride + inner
                    var grad_val = Float32(grad_output._data.bitcast[Float16]()[idx])
                    var out_val = Float32(output._data.bitcast[Float16]()[idx])
                    dot_sum += grad_val * out_val

                # Compute gradient for each position along axis
                for k in range(axis_size):
                    var idx = (outer * axis_size + k) * axis_stride + inner
                    var grad_val = Float32(grad_output._data.bitcast[Float16]()[idx])
                    var out_val = Float32(output._data.bitcast[Float16]()[idx])
                    result._data.bitcast[Float16]()[idx] = Float16(out_val * (grad_val - dot_sum))
    else:
        raise Error("softmax_backward: only float16/32/64 dtypes supported")

    return result


# ============================================================================
# Advanced Activation Functions
# ============================================================================


fn swish(tensor: ExTensor) raises -> ExTensor:
    """Swish activation function (also known as SiLU - Sigmoid Linear Unit).

    Swish is a smooth, non-monotonic activation function that performs better
    than ReLU in deep networks.

    Formula:
        swish(x) = x * sigmoid(x)

    Properties:
        - Smooth and continuously differentiable
        - Non-monotonic (has a slight dip for negative values)
        - Self-gated (uses its own value as gate)
        - Bounded below, unbounded above

    Args:
        tensor: Input tensor of any shape

    Returns:
        Output tensor with swish applied element-wise

    Example:
        ```mojo
        from shared.core import ExTensor, swish

        var x = ExTensor.from_list([...])
        var activated = swish(x)
        ```

    Reference:
        Ramachandran et al., "Searching for Activation Functions" (2017)
    """
    # swish(x) = x * sigmoid(x)
    var sig = sigmoid(tensor)
    return multiply(tensor, sig)


fn mish(tensor: ExTensor) raises -> ExTensor:
    """Mish activation function.

    Mish is a smooth, self-regularized non-monotonic activation function
    that has shown improvements over ReLU and Swish in some tasks.

    Formula:
        mish(x) = x * tanh(softplus(x))
        where softplus(x) = log(1 + exp(x))

    Properties:
        - Smooth and continuously differentiable
        - Non-monotonic (has a slight dip for negative values)
        - Self-regularized (bounded below)
        - Unbounded above

    Args:
        tensor: Input tensor of any shape

    Returns:
        Output tensor with mish applied element-wise

    Example:
        ```mojo
        from shared.core import ExTensor, mish

        var x = ExTensor.from_list([...])
        var activated = mish(x)
        ```

    Reference:
        Misra, "Mish: A Self Regularized Non-Monotonic Activation Function" (2019)
    """
    # mish(x) = x * tanh(softplus(x))
    # softplus(x) = log(1 + exp(x))

    # Compute softplus with numerical stability
    var exp_x = exp(tensor)
    var one_plus_exp = add(exp_x, ExTensor.from_scalar(1.0, tensor.dtype()))
    var softplus = log(one_plus_exp)

    var tanh_softplus = tanh(softplus)
    return multiply(tensor, tanh_softplus)


fn elu(tensor: ExTensor, alpha: Float64 = 1.0) raises -> ExTensor:
    """Exponential Linear Unit (ELU) activation function.

    ELU has negative values which pushes mean unit activations closer to zero,
    reducing bias shift and improving learning.

    Formula:
        elu(x) = x if x > 0
        elu(x) = alpha * (exp(x) - 1) if x <= 0

    Properties:
        - Smooth and continuously differentiable
        - Has negative values (unlike ReLU)
        - Saturates for large negative values
        - Reduces bias shift

    Args:
        tensor: Input tensor of any shape
        alpha: Scale for negative values (default: 1.0)

    Returns:
        Output tensor with ELU applied element-wise

    Example:
        ```mojo
        from shared.core import ExTensor, elu

        var x = ExTensor.from_list([...])
        var activated = elu(x, alpha=1.0)
        ```

    Reference:
        Clevert et al., "Fast and Accurate Deep Network Learning by
        Exponential Linear Units (ELUs)" (2015)
    """
    var result = zeros_like(tensor)
    var data_ptr = tensor._data
    var result_ptr = result._data
    var size = tensor.numel()

    if tensor.dtype() == DType.float32:
        for i in range(size):
            var val = data_ptr.bitcast[Float32]()[i]
            if val > 0:
                result_ptr.bitcast[Float32]()[i] = val
            else:
                # alpha * (exp(val) - 1)
                var exp_val = exp_scalar_f32(val)
                result_ptr.bitcast[Float32]()[i] = Float32(alpha) * (exp_val - 1.0)
    elif tensor.dtype() == DType.float64:
        for i in range(size):
            var val = data_ptr.bitcast[Float64]()[i]
            if val > 0:
                result_ptr.bitcast[Float64]()[i] = val
            else:
                var exp_val = exp_scalar_f64(val)
                result_ptr.bitcast[Float64]()[i] = alpha * (exp_val - 1.0)
    elif tensor.dtype() == DType.float16:
        for i in range(size):
            var val = Float32(data_ptr.bitcast[Float16]()[i])
            if val > 0:
                result_ptr.bitcast[Float16]()[i] = Float16(val)
            else:
                var exp_val = exp_scalar_f32(val)
                result_ptr.bitcast[Float16]()[i] = Float16(Float32(alpha) * (exp_val - 1.0))
    else:
        raise Error("elu: only float16/32/64 dtypes supported")

    return result


# Helper functions for scalar exp (used in ELU)
fn exp_scalar_f32(x: Float32) -> Float32:
    """Compute exp of a scalar float32."""
    # Using power operator for exponential
    return Float32(2.718281828459045) ** x


fn exp_scalar_f64(x: Float64) -> Float64:
    """Compute exp of a scalar float64."""
    return 2.718281828459045 ** x


# ============================================================================
# Backward Passes for Advanced Activations
# ============================================================================


fn swish_backward(grad_output: ExTensor, x: ExTensor) raises -> ExTensor:
    """Backward pass for Swish activation.

    The derivative of swish is:
        d/dx[swish(x)] = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
                       = sigmoid(x) * (1 + x * (1 - sigmoid(x)))

    Args:
        grad_output: Gradient from upstream
        x: Input from forward pass

    Returns:
        Gradient with respect to input

    Example:
        ```mojo
        from shared.core import swish, swish_backward

        # Forward
        var output = swish(x)
        # Backward
        var grad_x = swish_backward(grad_output, x)
        ```
    """
    # Compute sigmoid(x)
    var sig = sigmoid(x)

    # Derivative: sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    var one_minus_sig = subtract(ExTensor.from_scalar(1.0, sig.dtype()), sig)
    var x_term = multiply(x, one_minus_sig)
    var one_plus_x_term = add(ExTensor.from_scalar(1.0, x_term.dtype()), x_term)
    var derivative = multiply(sig, one_plus_x_term)

    # Apply chain rule
    return multiply(grad_output, derivative)


fn mish_backward(grad_output: ExTensor, x: ExTensor) raises -> ExTensor:
    """Backward pass for Mish activation.

    The derivative involves the derivative of tanh(softplus(x)).

    Args:
        grad_output: Gradient from upstream
        x: Input from forward pass

    Returns:
        Gradient with respect to input

    Example:
        ```mojo
        from shared.core import mish, mish_backward

        # Forward
        var output = mish(x)
        # Backward
        var grad_x = mish_backward(grad_output, x)
        ```
    """
    # Compute softplus(x) = log(1 + exp(x))
    var exp_x = exp(x)
    var one_plus_exp = add(exp_x, ExTensor.from_scalar(1.0, x.dtype()))
    var softplus = log(one_plus_exp)

    # Compute tanh(softplus(x))
    var tanh_sp = tanh(softplus)

    # Compute sigmoid(x) for derivative
    var sig = sigmoid(x)

    # Derivative of mish:
    # = tanh(softplus(x)) + x * sech²(softplus(x)) * sigmoid(x)
    # where sech²(y) = 1 - tanh²(y)

    var tanh_sq = multiply(tanh_sp, tanh_sp)
    var sech_sq = subtract(ExTensor.from_scalar(1.0, tanh_sq.dtype()), tanh_sq)
    var x_sech_sig = multiply(multiply(x, sech_sq), sig)
    var derivative = add(tanh_sp, x_sech_sig)

    # Apply chain rule
    return multiply(grad_output, derivative)


fn elu_backward(grad_output: ExTensor, x: ExTensor, alpha: Float64 = 1.0) raises -> ExTensor:
    """Backward pass for ELU activation.

    The derivative is:
        d/dx[elu(x)] = 1 if x > 0
        d/dx[elu(x)] = alpha * exp(x) if x <= 0

    Args:
        grad_output: Gradient from upstream
        x: Input from forward pass
        alpha: Scale for negative values (must match forward pass)

    Returns:
        Gradient with respect to input

    Example:
        ```mojo
        from shared.core import elu, elu_backward

        # Forward
        var output = elu(x, alpha=1.0)
        # Backward
        var grad_x = elu_backward(grad_output, x, alpha=1.0)
        ```
    """
    var result = zeros_like(x)
    var x_ptr = x._data
    var grad_ptr = grad_output._data
    var result_ptr = result._data
    var size = x.numel()

    if x.dtype() == DType.float32:
        for i in range(size):
            var val = x_ptr.bitcast[Float32]()[i]
            var grad_val = grad_ptr.bitcast[Float32]()[i]

            if val > 0:
                result_ptr.bitcast[Float32]()[i] = grad_val
            else:
                # alpha * exp(val)
                var exp_val = exp_scalar_f32(val)
                result_ptr.bitcast[Float32]()[i] = grad_val * Float32(alpha) * exp_val
    elif x.dtype() == DType.float64:
        for i in range(size):
            var val = x_ptr.bitcast[Float64]()[i]
            var grad_val = grad_ptr.bitcast[Float64]()[i]

            if val > 0:
                result_ptr.bitcast[Float64]()[i] = grad_val
            else:
                var exp_val = exp_scalar_f64(val)
                result_ptr.bitcast[Float64]()[i] = grad_val * alpha * exp_val
    elif x.dtype() == DType.float16:
        for i in range(size):
            var val = Float32(x_ptr.bitcast[Float16]()[i])
            var grad_val = Float32(grad_ptr.bitcast[Float16]()[i])

            if val > 0:
                result_ptr.bitcast[Float16]()[i] = Float16(grad_val)
            else:
                var exp_val = exp_scalar_f32(val)
                result_ptr.bitcast[Float16]()[i] = Float16(grad_val * Float32(alpha) * exp_val)
    else:
        raise Error("elu_backward: only float16/32/64 dtypes supported")

    return result
