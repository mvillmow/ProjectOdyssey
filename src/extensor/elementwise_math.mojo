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
