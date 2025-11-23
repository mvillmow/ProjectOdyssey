"""Element-wise mathematical operations for ExTensor.

Implements mathematical functions like exp, log, sqrt, trigonometric functions, etc.
"""

from .extensor import ExTensor
from .dtype_dispatch import dispatch_unary, dispatch_binary, dispatch_float_unary, dispatch_float_binary
from math import sqrt as math_sqrt
from math import exp as math_exp
from math import log as math_log
from math import sin as math_sin
from math import cos as math_cos
from math import tanh as math_tanh
from math import ceil as math_ceil
from math import floor as math_floor
from math import trunc as math_trunc
from memory import UnsafePointer


# ============================================================================
# Unary Operations (Element-wise)
# ============================================================================


@always_inline
fn _abs_op[T: DType](x: Scalar[T]) -> Scalar[T]:
    """Absolute value operation."""
    @parameter
    if T == DType.float16 or T == DType.float32:
        return Scalar[T](abs(Float32(x)))
    else:
        return Scalar[T](abs(Float64(x)))


fn abs(tensor: ExTensor) raises -> ExTensor:
    """Absolute value element-wise.

    Args:
        tensor: Input tensor

    Returns:
        A new tensor with absolute values

    Examples:
        var a = full(shape, -3.0, DType.float32)
        var b = abs(a)  # All values become 3.0
    """
    return dispatch_unary[_abs_op](tensor)


@always_inline
fn _sign_op[T: DType](x: Scalar[T]) -> Scalar[T]:
    """Sign operation: -1, 0, or 1."""
    if x > Scalar[T](0):
        return Scalar[T](1)
    elif x < Scalar[T](0):
        return Scalar[T](-1)
    else:
        return Scalar[T](0)


fn sign(tensor: ExTensor) raises -> ExTensor:
    """Sign function element-wise (-1, 0, or 1).

    Args:.        `tensor`: Input tensor.

    Returns:.        A new tensor with sign values (-1 for negative, 0 for zero, 1 for positive)

    Examples:
        var a = tensor([-2.0, 0.0, 3.0])
        var b = sign(a)  # [-1.0, 0.0, 1.0]
    """
    return dispatch_unary[_sign_op](tensor)


@always_inline
fn _exp_op[T: DType](x: Scalar[T]) -> Scalar[T]:
    """Exponential operation."""
    @parameter
    if T == DType.float16 or T == DType.float32:
        return Scalar[T](math_exp(Float32(x)))
    else:
        return Scalar[T](math_exp(Float64(x)))


fn exp(tensor: ExTensor) raises -> ExTensor:
    """Exponential function element-wise (e^x).

    Args:
        tensor: Input tensor

    Returns:
        A new tensor with exponential values

    Examples:
        var a = zeros(shape, DType.float32)
        var b = exp(a)  # All values become 1.0 (e^0)
    """
    return dispatch_float_unary[_exp_op](tensor)


@always_inline
fn _log_op[T: DType](x: Scalar[T]) -> Scalar[T]:
    """Natural logarithm operation."""
    @parameter
    if T == DType.float16 or T == DType.float32:
        return Scalar[T](math_log(Float32(x)))
    else:
        return Scalar[T](math_log(Float64(x)))


fn log(tensor: ExTensor) raises -> ExTensor:
    """Natural logarithm element-wise (ln(x)).

    Args:.        `tensor`: Input tensor (must have positive values)

    Returns:.        A new tensor with logarithm values.

    Raises:.        Error if any value is <= 0.

    Examples:
        var a = ones(shape, DType.float32)
        var b = log(a)  # All values become 0.0 (ln(1))
    """
    return dispatch_float_unary[_log_op](tensor)


@always_inline
fn _sqrt_op[T: DType](x: Scalar[T]) -> Scalar[T]:
    """Square root operation."""
    @parameter
    if T == DType.float16 or T == DType.float32:
        return Scalar[T](math_sqrt(Float32(x)))
    else:
        return Scalar[T](math_sqrt(Float64(x)))


fn sqrt(tensor: ExTensor) raises -> ExTensor:
    """Square root element-wise.

    Args:
        tensor: Input tensor (must have non-negative values)

    Returns:
        A new tensor with square root values

    Raises:
        Error if any value is < 0

    Examples:
        var a = full(shape, 4.0, DType.float32)
        var b = sqrt(a)  # All values become 2.0
    """
    return dispatch_float_unary[_sqrt_op](tensor)


@always_inline
fn _sin_op[T: DType](x: Scalar[T]) -> Scalar[T]:
    """Sine operation."""
    @parameter
    if T == DType.float16 or T == DType.float32:
        return Scalar[T](math_sin(Float32(x)))
    else:
        return Scalar[T](math_sin(Float64(x)))


fn sin(tensor: ExTensor) raises -> ExTensor:
    """Sine function element-wise.

    Args:.        `tensor`: Input tensor (values in radians)

    Returns:.        A new tensor with sine values.

    Examples:
        var a = zeros(shape, DType.float32)
        var b = sin(a)  # All values become 0.0 (sin(0))
    """
    return dispatch_float_unary[_sin_op](tensor)


@always_inline
fn _cos_op[T: DType](x: Scalar[T]) -> Scalar[T]:
    """Cosine operation."""
    @parameter
    if T == DType.float16 or T == DType.float32:
        return Scalar[T](math_cos(Float32(x)))
    else:
        return Scalar[T](math_cos(Float64(x)))


fn cos(tensor: ExTensor) raises -> ExTensor:
    """Cosine function element-wise.

    Args:
        tensor: Input tensor (values in radians)

    Returns:
        A new tensor with cosine values

    Examples:
        var a = zeros(shape, DType.float32)
        var b = cos(a)  # All values become 1.0 (cos(0))
    """
    return dispatch_float_unary[_cos_op](tensor)


@always_inline
fn _tanh_op[T: DType](x: Scalar[T]) -> Scalar[T]:
    """Hyperbolic tangent operation."""
    @parameter
    if T == DType.float16 or T == DType.float32:
        return Scalar[T](math_tanh(Float32(x)))
    else:
        return Scalar[T](math_tanh(Float64(x)))


fn tanh(tensor: ExTensor) raises -> ExTensor:
    """Hyperbolic tangent function element-wise.

    Args:.        `tensor`: Input tensor.

    Returns:.        A new tensor with tanh values (range: -1 to 1)

    Examples:
        var a = zeros(shape, DType.float32)
        var b = tanh(a)  # All values become 0.0 (tanh(0))
    """
    return dispatch_float_unary[_tanh_op](tensor)


fn clip(tensor: ExTensor, min_val: Float64, max_val: Float64) raises -> ExTensor:
    """Clip (clamp) values to a range element-wise.

    Args:.        `tensor`: Input tensor.
        `min_val`: Minimum value.
        `max_val`: Maximum value.

    Returns:.        A new tensor with clipped values.

    Raises:.        Error if min_val > max_val.

    Examples:
        var a = tensor([-5.0, 0.0, 10.0])
        var b = clip(a, 0.0, 5.0)  # [0.0, 0.0, 5.0]
    """
    if min_val > max_val:
        raise Error("clip requires min_val <= max_val")

    var result = ExTensor(tensor.shape(), tensor.dtype())

    var numel = tensor.numel()
    for i in range(numel):
        var val = tensor._get_float64(i)
        if val < min_val:
            val = min_val
        elif val > max_val:
            val = max_val
        result._set_float64(i, val)

    return result^


# ============================================================================
# Rounding operations
# ============================================================================


@always_inline
fn _ceil_op[T: DType](x: Scalar[T]) -> Scalar[T]:
    """Ceiling operation."""
    @parameter
    if T == DType.float16 or T == DType.float32:
        return Scalar[T](math_ceil(Float32(x)))
    else:
        return Scalar[T](math_ceil(Float64(x)))


fn ceil(tensor: ExTensor) raises -> ExTensor:
    """Ceiling function element-wise (round up to nearest integer).

    Args:
        tensor: Input tensor

    Returns:
        A new tensor with ceiling values

    Examples:
        var a = tensor([1.2, 2.5, 3.9])
        var b = ceil(a)  # [2.0, 3.0, 4.0]
    """
    return dispatch_float_unary[_ceil_op](tensor)


@always_inline
fn _floor_op[T: DType](x: Scalar[T]) -> Scalar[T]:
    """Floor operation."""
    @parameter
    if T == DType.float16 or T == DType.float32:
        return Scalar[T](math_floor(Float32(x)))
    else:
        return Scalar[T](math_floor(Float64(x)))


fn floor(tensor: ExTensor) raises -> ExTensor:
    """Floor function element-wise (round down to nearest integer).

    Args:.        `tensor`: Input tensor.

    Returns:.        A new tensor with floor values.

    Examples:
        var a = tensor([1.2, 2.5, 3.9])
        var b = floor(a)  # [1.0, 2.0, 3.0]
    """
    return dispatch_float_unary[_floor_op](tensor)


@always_inline
fn _round_op[T: DType](x: Scalar[T]) -> Scalar[T]:
    """Round operation."""
    @parameter
    if T == DType.float16 or T == DType.float32:
        return Scalar[T](round(Float32(x)))
    else:
        return Scalar[T](round(Float64(x)))


fn round(tensor: ExTensor) raises -> ExTensor:
    """Round to nearest integer element-wise.

    Args:
        tensor: Input tensor

    Returns:
        A new tensor with rounded values

    Examples:
        var a = tensor([1.2, 2.5, 3.9])
        var b = round(a)  # [1.0, 2.0, 4.0] (or [1.0, 3.0, 4.0] depending on rounding mode)
    """
    return dispatch_float_unary[_round_op](tensor)


@always_inline
fn _trunc_op[T: DType](x: Scalar[T]) -> Scalar[T]:
    """Truncate operation."""
    @parameter
    if T == DType.float16 or T == DType.float32:
        return Scalar[T](math_trunc(Float32(x)))
    else:
        return Scalar[T](math_trunc(Float64(x)))


fn trunc(tensor: ExTensor) raises -> ExTensor:
    """Truncate to integer element-wise (round toward zero).

    Args:.        `tensor`: Input tensor.

    Returns:.        A new tensor with truncated values.

    Examples:
        var a = tensor([1.9, -2.9, 3.1])
        var b = trunc(a)  # [1.0, -2.0, 3.0]
    """
    var result = ExTensor(tensor.shape(), tensor.dtype())

    var numel = tensor.numel()
    for i in range(numel):
        var val = tensor._get_float64(i)
        result._set_float64(i, math_trunc(val))

    return result^


# ============================================================================
# Logical operations
# ============================================================================

fn logical_and(a: ExTensor, b: ExTensor) raises -> ExTensor:
    """Logical AND element-wise.

    Args:.        `a`: First input tensor.
        `b`: Second input tensor.

    Returns:.        Boolean tensor (True where both are non-zero)

    Examples:
        var a = tensor([0.0, 1.0, 2.0])
        var b = tensor([0.0, 0.0, 1.0])
        var c = logical_and(a, b)  # [False, False, True]
    """
    if a.dtype() != b.dtype():
        raise Error("logical_and: tensors must have same dtype")

    var shape_a = a.shape()
    var shape_b = b.shape()

    if len(shape_a) != len(shape_b):
        raise Error("logical_and: tensors must have same shape (broadcasting TODO)")

    for i in range(len(shape_a)):
        if shape_a[i] != shape_b[i]:
            raise Error("logical_and: tensors must have same shape (broadcasting TODO)")

    var result = ExTensor(a.shape(), DType.bool)

    var numel = a.numel()
    for i in range(numel):
        var val_a = a._get_float64(i)
        var val_b = b._get_float64(i)
        # True if both non-zero
        var bool_result = (val_a != 0.0) and (val_b != 0.0)
        result._set_float64(i, 1.0 if bool_result else 0.0)

    return result^


fn logical_or(a: ExTensor, b: ExTensor) raises -> ExTensor:
    """Logical OR element-wise.

    Args:.        `a`: First input tensor.
        `b`: Second input tensor.

    Returns:.        Boolean tensor (True where either is non-zero)

    Examples:
        var a = tensor([0.0, 1.0, 2.0])
        var b = tensor([0.0, 0.0, 1.0])
        var c = logical_or(a, b)  # [False, True, True]
    """
    if a.dtype() != b.dtype():
        raise Error("logical_or: tensors must have same dtype")

    var shape_a = a.shape()
    var shape_b = b.shape()

    if len(shape_a) != len(shape_b):
        raise Error("logical_or: tensors must have same shape (broadcasting TODO)")

    for i in range(len(shape_a)):
        if shape_a[i] != shape_b[i]:
            raise Error("logical_or: tensors must have same shape (broadcasting TODO)")

    var result = ExTensor(a.shape(), DType.bool)

    var numel = a.numel()
    for i in range(numel):
        var val_a = a._get_float64(i)
        var val_b = b._get_float64(i)
        # True if either non-zero
        var bool_result = (val_a != 0.0) or (val_b != 0.0)
        result._set_float64(i, 1.0 if bool_result else 0.0)

    return result^


fn logical_not(tensor: ExTensor) raises -> ExTensor:
    """Logical NOT element-wise.

    Args:.        `tensor`: Input tensor.

    Returns:.        Boolean tensor (True where input is zero)

    Examples:
        var a = tensor([0.0, 1.0, 2.0])
        var b = logical_not(a)  # [True, False, False]
    """
    var result = ExTensor(tensor.shape(), DType.bool)

    var numel = tensor.numel()
    for i in range(numel):
        var val = tensor._get_float64(i)
        # True if zero
        var bool_result = (val == 0.0)
        result._set_float64(i, 1.0 if bool_result else 0.0)

    return result^


fn logical_xor(a: ExTensor, b: ExTensor) raises -> ExTensor:
    """Logical XOR element-wise.

    Args:.        `a`: First input tensor.
        `b`: Second input tensor.

    Returns:.        Boolean tensor (True where exactly one is non-zero)

    Examples:
        var a = tensor([0.0, 1.0, 0.0, 1.0])
        var b = tensor([0.0, 0.0, 1.0, 1.0])
        var c = logical_xor(a, b)  # [False, True, True, False]
    """
    if a.dtype() != b.dtype():
        raise Error("logical_xor: tensors must have same dtype")

    var shape_a = a.shape()
    var shape_b = b.shape()

    if len(shape_a) != len(shape_b):
        raise Error("logical_xor: tensors must have same shape (broadcasting TODO)")

    for i in range(len(shape_a)):
        if shape_a[i] != shape_b[i]:
            raise Error("logical_xor: tensors must have same shape (broadcasting TODO)")

    var result = ExTensor(a.shape(), DType.bool)

    var numel = a.numel()
    for i in range(numel):
        var val_a = a._get_float64(i)
        var val_b = b._get_float64(i)
        # True if exactly one is non-zero
        var bool_a = (val_a != 0.0)
        var bool_b = (val_b != 0.0)
        var bool_result = bool_a != bool_b  # XOR
        result._set_float64(i, 1.0 if bool_result else 0.0)

    return result^


# ============================================================================
# Additional transcendental functions
# ============================================================================

fn log10(tensor: ExTensor) raises -> ExTensor:
    """Base-10 logarithm element-wise.

    Args:.        `tensor`: Input tensor (must have positive values)

    Returns:.        A new tensor with log10 values.

    Raises:.        Error if any value is <= 0.

    Examples:
        var a = tensor([1.0, 10.0, 100.0])
        var b = log10(a)  # [0.0, 1.0, 2.0]
    """
    var result = ExTensor(tensor.shape(), tensor.dtype())

    var numel = tensor.numel()
    for i in range(numel):
        var val = tensor._get_float64(i)
        if val <= 0.0:
            raise Error("log10 requires positive values, got " + str(val))
        # log10(x) = log(x) / log(10)
        result._set_float64(i, math_log(val) / math_log(10.0))

    return result^


fn log2(tensor: ExTensor) raises -> ExTensor:
    """Base-2 logarithm element-wise.

    Args:.        `tensor`: Input tensor (must have positive values)

    Returns:.        A new tensor with log2 values.

    Raises:.        Error if any value is <= 0.

    Examples:
        var a = tensor([1.0, 2.0, 8.0])
        var b = log2(a)  # [0.0, 1.0, 3.0]
    """
    var result = ExTensor(tensor.shape(), tensor.dtype())

    var numel = tensor.numel()
    for i in range(numel):
        var val = tensor._get_float64(i)
        if val <= 0.0:
            raise Error("log2 requires positive values, got " + str(val))
        # log2(x) = log(x) / log(2)
        result._set_float64(i, math_log(val) / math_log(2.0))

    return result^


# ============================================================================
# Backward Pass (Gradient Computation)
# ============================================================================


fn exp_backward(grad_output: ExTensor, output: ExTensor) raises -> ExTensor:
    """Compute gradient for exponential function.

    For Y = exp(X), given ∂L/∂Y, computes:
        ∂L/∂X = ∂L/∂Y * exp(X) = ∂L/∂Y * Y

    Uses output from forward pass to avoid recomputing exp(X).

    Args:.        `grad_output`: Gradient from upstream (∂L/∂Y)
        `output`: Output from forward pass (Y = exp(X))

    Returns:.        Gradient w.r.t. input (∂L/∂X)

    Examples:
        var x = ones([3, 4])
        var y = exp(x)  # Forward pass
        var grad_y = ones([3, 4])
        var grad_x = exp_backward(grad_y, y)  # grad_x = grad_y * y
    """
    var result = ExTensor(grad_output.shape(), grad_output.dtype())

    for i in range(grad_output.numel()):
        var grad = grad_output._get_float64(i)
        var out_val = output._get_float64(i)
        result._set_float64(i, grad * out_val)

    return result


fn log_backward(grad_output: ExTensor, x: ExTensor) raises -> ExTensor:
    """Compute gradient for natural logarithm.

    For Y = log(X), given ∂L/∂Y, computes:
        ∂L/∂X = ∂L/∂Y / X

    Includes numerical stability: adds epsilon to prevent division by zero.

    Args:.        `grad_output`: Gradient from upstream (∂L/∂Y)
        `x`: Input from forward pass (must be positive)

    Returns:.        Gradient w.r.t. input (∂L/∂X)

    Examples:
        var x = full([3, 4], 2.0)
        var y = log(x)
        var grad_y = ones([3, 4])
        var grad_x = log_backward(grad_y, x)  # grad_x = grad_y / x

    Numerical Stability:
        Uses epsilon = 1e-10 to prevent division by zero.
    """
    alias EPSILON = 1e-10

    var result = ExTensor(grad_output.shape(), grad_output.dtype())

    for i in range(grad_output.numel()):
        var grad = grad_output._get_float64(i)
        var x_val = x._get_float64(i)
        # Add epsilon for numerical stability
        result._set_float64(i, grad / (x_val + EPSILON))

    return result


fn sqrt_backward(grad_output: ExTensor, output: ExTensor) raises -> ExTensor:
    """Compute gradient for square root.

    For Y = sqrt(X), given ∂L/∂Y, computes:
        ∂L/∂X = ∂L/∂Y / (2 * sqrt(X)) = ∂L/∂Y / (2 * Y)

    Uses output from forward pass to avoid recomputing sqrt(X).
    Includes numerical stability.

    Args:.        `grad_output`: Gradient from upstream (∂L/∂Y)
        `output`: Output from forward pass (Y = sqrt(X))

    Returns:.        Gradient w.r.t. input (∂L/∂X)

    Examples:
        var x = full([3, 4], 4.0)
        var y = sqrt(x)  # y = 2.0
        var grad_y = ones([3, 4])
        var grad_x = sqrt_backward(grad_y, y)  # grad_x = grad_y / (2 * 2.0) = 0.25

    Numerical Stability:
        Uses epsilon = 1e-10 to prevent division by zero when Y ≈ 0.
    """
    alias EPSILON = 1e-10

    var result = ExTensor(grad_output.shape(), grad_output.dtype())

    for i in range(grad_output.numel()):
        var grad = grad_output._get_float64(i)
        var out_val = output._get_float64(i)
        # grad / (2 * sqrt(x)) = grad / (2 * output)
        # Add epsilon for numerical stability
        result._set_float64(i, grad / (2.0 * out_val + EPSILON))

    return result


fn abs_backward(grad_output: ExTensor, x: ExTensor) raises -> ExTensor:
    """Compute gradient for absolute value.

    For Y = |X|, given ∂L/∂Y, computes:
        ∂L/∂X = ∂L/∂Y * sign(X)

    where sign(X) = 1 if X > 0, -1 if X < 0, 0 if X = 0.

    Note: Gradient at X=0 is technically undefined, we use 0 by convention.

    Args:.        `grad_output`: Gradient from upstream (∂L/∂Y)
        `x`: Input from forward pass.

    Returns:.        Gradient w.r.t. input (∂L/∂X)

    Examples:
        var x = tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        var y = abs(x)
        var grad_y = ones([5])
        var grad_x = abs_backward(grad_y, x)  # [-1, -1, 0, 1, 1]
    """
    var result = ExTensor(grad_output.shape(), grad_output.dtype())

    for i in range(grad_output.numel()):
        var grad = grad_output._get_float64(i)
        var x_val = x._get_float64(i)

        # Compute sign(x): 1 if x > 0, -1 if x < 0, 0 if x = 0
        var sign_x: Float64 = 0.0
        if x_val > 0.0:
            sign_x = 1.0
        elif x_val < 0.0:
            sign_x = -1.0
        # else: sign_x = 0.0 (at x=0, gradient is undefined, use 0)

        result._set_float64(i, grad * sign_x)

    return result


fn clip_backward(grad_output: ExTensor, x: ExTensor, min_val: Float64, max_val: Float64) raises -> ExTensor:
    """Compute gradient for clip (clamp) operation.

    For Y = clip(X, min, max), given ∂L/∂Y, computes:
        ∂L/∂X = ∂L/∂Y  if min <= X <= max
        ∂L/∂X = 0       if X < min or X > max

    Gradient flows through only where input is within bounds.

    Args:.        `grad_output`: Gradient from upstream (∂L/∂Y)
        `x`: Input from forward pass.
        `min_val`: Minimum clip value.
        `max_val`: Maximum clip value.

    Returns:.        Gradient w.r.t. input (∂L/∂X)

    Examples:
        var x = tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        var y = clip(x, -1.0, 1.0)  # [-1, -1, 0, 1, 1]
        var grad_y = ones([5])
        var grad_x = clip_backward(grad_y, x, -1.0, 1.0)  # [0, 1, 1, 1, 0]
    """
    var result = ExTensor(grad_output.shape(), grad_output.dtype())

    for i in range(grad_output.numel()):
        var grad = grad_output._get_float64(i)
        var x_val = x._get_float64(i)

        # Gradient flows through only if min <= x <= max
        if x_val >= min_val and x_val <= max_val:
            result._set_float64(i, grad)
        else:
            result._set_float64(i, 0.0)

    return result


fn log10_backward(grad_output: ExTensor, x: ExTensor) raises -> ExTensor:
    """Compute gradient for base-10 logarithm.

    For Y = log10(X), given ∂L/∂Y, computes:
        ∂L/∂X = ∂L/∂Y / (X * ln(10))

    Args:.        `grad_output`: Gradient from upstream (∂L/∂Y)
        `x`: Input from forward pass (must be positive)

    Returns:.        Gradient w.r.t. input (∂L/∂X)

    Numerical Stability:
        Uses epsilon = 1e-10 to prevent division by zero.
    """
    alias EPSILON = 1e-10
    alias LN10 = 2.302585092994046  # ln(10)

    var result = ExTensor(grad_output.shape(), grad_output.dtype())

    for i in range(grad_output.numel()):
        var grad = grad_output._get_float64(i)
        var x_val = x._get_float64(i)
        result._set_float64(i, grad / (x_val * LN10 + EPSILON))

    return result


fn log2_backward(grad_output: ExTensor, x: ExTensor) raises -> ExTensor:
    """Compute gradient for base-2 logarithm.

    For Y = log2(X), given ∂L/∂Y, computes:
        ∂L/∂X = ∂L/∂Y / (X * ln(2))

    Args:.        `grad_output`: Gradient from upstream (∂L/∂Y)
        `x`: Input from forward pass (must be positive)

    Returns:.        Gradient w.r.t. input (∂L/∂X)

    Numerical Stability:
        Uses epsilon = 1e-10 to prevent division by zero.
    """
    alias EPSILON = 1e-10
    alias LN2 = 0.6931471805599453  # ln(2)

    var result = ExTensor(grad_output.shape(), grad_output.dtype())

    for i in range(grad_output.numel()):
        var grad = grad_output._get_float64(i)
        var x_val = x._get_float64(i)
        result._set_float64(i, grad / (x_val * LN2 + EPSILON))

    return result
