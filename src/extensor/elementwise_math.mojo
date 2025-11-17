"""Element-wise mathematical operations for ExTensor.

Implements mathematical functions like exp, log, sqrt, trigonometric functions, etc.
"""

from .extensor import ExTensor
from math import sqrt as math_sqrt
from math import exp as math_exp
from math import log as math_log
from math import sin as math_sin
from math import cos as math_cos
from math import tanh as math_tanh
from math import abs as math_abs
from math import ceil as math_ceil
from math import floor as math_floor
from math import round as math_round
from math import trunc as math_trunc
from sys import DType


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
    var result = ExTensor(tensor.shape(), tensor.dtype())

    let numel = tensor.numel()
    for i in range(numel):
        let val = tensor._get_float64(i)
        result._set_float64(i, math_abs(val))

    return result^


fn sign(tensor: ExTensor) raises -> ExTensor:
    """Sign function element-wise (-1, 0, or 1).

    Args:
        tensor: Input tensor

    Returns:
        A new tensor with sign values (-1 for negative, 0 for zero, 1 for positive)

    Examples:
        var a = tensor([-2.0, 0.0, 3.0])
        var b = sign(a)  # [-1.0, 0.0, 1.0]
    """
    var result = ExTensor(tensor.shape(), tensor.dtype())

    let numel = tensor.numel()
    for i in range(numel):
        let val = tensor._get_float64(i)
        var sign_val: Float64
        if val > 0.0:
            sign_val = 1.0
        elif val < 0.0:
            sign_val = -1.0
        else:
            sign_val = 0.0
        result._set_float64(i, sign_val)

    return result^


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
    var result = ExTensor(tensor.shape(), tensor.dtype())

    let numel = tensor.numel()
    for i in range(numel):
        let val = tensor._get_float64(i)
        result._set_float64(i, math_exp(val))

    return result^


fn log(tensor: ExTensor) raises -> ExTensor:
    """Natural logarithm element-wise (ln(x)).

    Args:
        tensor: Input tensor (must have positive values)

    Returns:
        A new tensor with logarithm values

    Raises:
        Error if any value is <= 0

    Examples:
        var a = ones(shape, DType.float32)
        var b = log(a)  # All values become 0.0 (ln(1))
    """
    var result = ExTensor(tensor.shape(), tensor.dtype())

    let numel = tensor.numel()
    for i in range(numel):
        let val = tensor._get_float64(i)
        if val <= 0.0:
            raise Error("log requires positive values, got " + str(val))
        result._set_float64(i, math_log(val))

    return result^


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
    var result = ExTensor(tensor.shape(), tensor.dtype())

    let numel = tensor.numel()
    for i in range(numel):
        let val = tensor._get_float64(i)
        if val < 0.0:
            raise Error("sqrt requires non-negative values, got " + str(val))
        result._set_float64(i, math_sqrt(val))

    return result^


fn sin(tensor: ExTensor) raises -> ExTensor:
    """Sine function element-wise.

    Args:
        tensor: Input tensor (values in radians)

    Returns:
        A new tensor with sine values

    Examples:
        var a = zeros(shape, DType.float32)
        var b = sin(a)  # All values become 0.0 (sin(0))
    """
    var result = ExTensor(tensor.shape(), tensor.dtype())

    let numel = tensor.numel()
    for i in range(numel):
        let val = tensor._get_float64(i)
        result._set_float64(i, math_sin(val))

    return result^


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
    var result = ExTensor(tensor.shape(), tensor.dtype())

    let numel = tensor.numel()
    for i in range(numel):
        let val = tensor._get_float64(i)
        result._set_float64(i, math_cos(val))

    return result^


fn tanh(tensor: ExTensor) raises -> ExTensor:
    """Hyperbolic tangent function element-wise.

    Args:
        tensor: Input tensor

    Returns:
        A new tensor with tanh values (range: -1 to 1)

    Examples:
        var a = zeros(shape, DType.float32)
        var b = tanh(a)  # All values become 0.0 (tanh(0))
    """
    var result = ExTensor(tensor.shape(), tensor.dtype())

    let numel = tensor.numel()
    for i in range(numel):
        let val = tensor._get_float64(i)
        result._set_float64(i, math_tanh(val))

    return result^


fn clip(tensor: ExTensor, min_val: Float64, max_val: Float64) raises -> ExTensor:
    """Clip (clamp) values to a range element-wise.

    Args:
        tensor: Input tensor
        min_val: Minimum value
        max_val: Maximum value

    Returns:
        A new tensor with clipped values

    Raises:
        Error if min_val > max_val

    Examples:
        var a = tensor([-5.0, 0.0, 10.0])
        var b = clip(a, 0.0, 5.0)  # [0.0, 0.0, 5.0]
    """
    if min_val > max_val:
        raise Error("clip requires min_val <= max_val")

    var result = ExTensor(tensor.shape(), tensor.dtype())

    let numel = tensor.numel()
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
    var result = ExTensor(tensor.shape(), tensor.dtype())

    let numel = tensor.numel()
    for i in range(numel):
        let val = tensor._get_float64(i)
        result._set_float64(i, math_ceil(val))

    return result^


fn floor(tensor: ExTensor) raises -> ExTensor:
    """Floor function element-wise (round down to nearest integer).

    Args:
        tensor: Input tensor

    Returns:
        A new tensor with floor values

    Examples:
        var a = tensor([1.2, 2.5, 3.9])
        var b = floor(a)  # [1.0, 2.0, 3.0]
    """
    var result = ExTensor(tensor.shape(), tensor.dtype())

    let numel = tensor.numel()
    for i in range(numel):
        let val = tensor._get_float64(i)
        result._set_float64(i, math_floor(val))

    return result^


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
    var result = ExTensor(tensor.shape(), tensor.dtype())

    let numel = tensor.numel()
    for i in range(numel):
        let val = tensor._get_float64(i)
        result._set_float64(i, math_round(val))

    return result^


fn trunc(tensor: ExTensor) raises -> ExTensor:
    """Truncate to integer element-wise (round toward zero).

    Args:
        tensor: Input tensor

    Returns:
        A new tensor with truncated values

    Examples:
        var a = tensor([1.9, -2.9, 3.1])
        var b = trunc(a)  # [1.0, -2.0, 3.0]
    """
    var result = ExTensor(tensor.shape(), tensor.dtype())

    let numel = tensor.numel()
    for i in range(numel):
        let val = tensor._get_float64(i)
        result._set_float64(i, math_trunc(val))

    return result^


# ============================================================================
# Logical operations
# ============================================================================

fn logical_and(a: ExTensor, b: ExTensor) raises -> ExTensor:
    """Logical AND element-wise.

    Args:
        a: First input tensor
        b: Second input tensor

    Returns:
        Boolean tensor (True where both are non-zero)

    Examples:
        var a = tensor([0.0, 1.0, 2.0])
        var b = tensor([0.0, 0.0, 1.0])
        var c = logical_and(a, b)  # [False, False, True]
    """
    if a.dtype() != b.dtype():
        raise Error("logical_and: tensors must have same dtype")

    let shape_a = a.shape()
    let shape_b = b.shape()

    if len(shape_a) != len(shape_b):
        raise Error("logical_and: tensors must have same shape (broadcasting TODO)")

    for i in range(len(shape_a)):
        if shape_a[i] != shape_b[i]:
            raise Error("logical_and: tensors must have same shape (broadcasting TODO)")

    var result = ExTensor(a.shape(), DType.bool)

    let numel = a.numel()
    for i in range(numel):
        let val_a = a._get_float64(i)
        let val_b = b._get_float64(i)
        # True if both non-zero
        let bool_result = (val_a != 0.0) and (val_b != 0.0)
        result._set_float64(i, 1.0 if bool_result else 0.0)

    return result^


fn logical_or(a: ExTensor, b: ExTensor) raises -> ExTensor:
    """Logical OR element-wise.

    Args:
        a: First input tensor
        b: Second input tensor

    Returns:
        Boolean tensor (True where either is non-zero)

    Examples:
        var a = tensor([0.0, 1.0, 2.0])
        var b = tensor([0.0, 0.0, 1.0])
        var c = logical_or(a, b)  # [False, True, True]
    """
    if a.dtype() != b.dtype():
        raise Error("logical_or: tensors must have same dtype")

    let shape_a = a.shape()
    let shape_b = b.shape()

    if len(shape_a) != len(shape_b):
        raise Error("logical_or: tensors must have same shape (broadcasting TODO)")

    for i in range(len(shape_a)):
        if shape_a[i] != shape_b[i]:
            raise Error("logical_or: tensors must have same shape (broadcasting TODO)")

    var result = ExTensor(a.shape(), DType.bool)

    let numel = a.numel()
    for i in range(numel):
        let val_a = a._get_float64(i)
        let val_b = b._get_float64(i)
        # True if either non-zero
        let bool_result = (val_a != 0.0) or (val_b != 0.0)
        result._set_float64(i, 1.0 if bool_result else 0.0)

    return result^


fn logical_not(tensor: ExTensor) raises -> ExTensor:
    """Logical NOT element-wise.

    Args:
        tensor: Input tensor

    Returns:
        Boolean tensor (True where input is zero)

    Examples:
        var a = tensor([0.0, 1.0, 2.0])
        var b = logical_not(a)  # [True, False, False]
    """
    var result = ExTensor(tensor.shape(), DType.bool)

    let numel = tensor.numel()
    for i in range(numel):
        let val = tensor._get_float64(i)
        # True if zero
        let bool_result = (val == 0.0)
        result._set_float64(i, 1.0 if bool_result else 0.0)

    return result^


fn logical_xor(a: ExTensor, b: ExTensor) raises -> ExTensor:
    """Logical XOR element-wise.

    Args:
        a: First input tensor
        b: Second input tensor

    Returns:
        Boolean tensor (True where exactly one is non-zero)

    Examples:
        var a = tensor([0.0, 1.0, 0.0, 1.0])
        var b = tensor([0.0, 0.0, 1.0, 1.0])
        var c = logical_xor(a, b)  # [False, True, True, False]
    """
    if a.dtype() != b.dtype():
        raise Error("logical_xor: tensors must have same dtype")

    let shape_a = a.shape()
    let shape_b = b.shape()

    if len(shape_a) != len(shape_b):
        raise Error("logical_xor: tensors must have same shape (broadcasting TODO)")

    for i in range(len(shape_a)):
        if shape_a[i] != shape_b[i]:
            raise Error("logical_xor: tensors must have same shape (broadcasting TODO)")

    var result = ExTensor(a.shape(), DType.bool)

    let numel = a.numel()
    for i in range(numel):
        let val_a = a._get_float64(i)
        let val_b = b._get_float64(i)
        # True if exactly one is non-zero
        let bool_a = (val_a != 0.0)
        let bool_b = (val_b != 0.0)
        let bool_result = bool_a != bool_b  # XOR
        result._set_float64(i, 1.0 if bool_result else 0.0)

    return result^


# ============================================================================
# Additional transcendental functions
# ============================================================================

fn log10(tensor: ExTensor) raises -> ExTensor:
    """Base-10 logarithm element-wise.

    Args:
        tensor: Input tensor (must have positive values)

    Returns:
        A new tensor with log10 values

    Raises:
        Error if any value is <= 0

    Examples:
        var a = tensor([1.0, 10.0, 100.0])
        var b = log10(a)  # [0.0, 1.0, 2.0]
    """
    var result = ExTensor(tensor.shape(), tensor.dtype())

    let numel = tensor.numel()
    for i in range(numel):
        let val = tensor._get_float64(i)
        if val <= 0.0:
            raise Error("log10 requires positive values, got " + str(val))
        # log10(x) = log(x) / log(10)
        result._set_float64(i, math_log(val) / math_log(10.0))

    return result^


fn log2(tensor: ExTensor) raises -> ExTensor:
    """Base-2 logarithm element-wise.

    Args:
        tensor: Input tensor (must have positive values)

    Returns:
        A new tensor with log2 values

    Raises:
        Error if any value is <= 0

    Examples:
        var a = tensor([1.0, 2.0, 8.0])
        var b = log2(a)  # [0.0, 1.0, 3.0]
    """
    var result = ExTensor(tensor.shape(), tensor.dtype())

    let numel = tensor.numel()
    for i in range(numel):
        let val = tensor._get_float64(i)
        if val <= 0.0:
            raise Error("log2 requires positive values, got " + str(val))
        # log2(x) = log(x) / log(2)
        result._set_float64(i, math_log(val) / math_log(2.0))

    return result^
