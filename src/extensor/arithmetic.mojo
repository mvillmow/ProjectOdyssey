"""Arithmetic operations for ExTensor with broadcasting support.

Implements element-wise arithmetic operations following NumPy-style broadcasting.
"""

from .extensor import ExTensor
from .broadcasting import broadcast_shapes, compute_broadcast_strides


fn add(a: ExTensor, b: ExTensor) raises -> ExTensor:
    """Element-wise addition with broadcasting.

    Args:
        a: First tensor
        b: Second tensor

    Returns:
        A new tensor containing a + b

    Raises:
        Error if shapes are not broadcast-compatible or dtypes don't match

    Examples:
        var a = zeros(DynamicVector[Int](3, 4), DType.float32)
        var b = ones(DynamicVector[Int](3, 4), DType.float32)
        var c = add(a, b)  # Shape (3, 4), all ones
    """
    # Check dtype compatibility
    if a.dtype() != b.dtype():
        raise Error("Cannot add tensors with different dtypes")

    # Compute broadcast shape
    let result_shape = broadcast_shapes(a.shape(), b.shape())

    # Create result tensor
    var result = ExTensor(result_shape, a.dtype())

    # Simple case: same shape (no broadcasting needed)
    if len(a.shape()) == len(b.shape()):
        var same_shape = True
        for i in range(len(a.shape())):
            if a.shape()[i] != b.shape()[i]:
                same_shape = False
                break

        if same_shape:
            # Direct element-wise addition (no broadcasting)
            for i in range(a.numel()):
                let a_val = a._get_float64(i)
                let b_val = b._get_float64(i)
                result._set_float64(i, a_val + b_val)
            return result^

    # TODO: Implement full broadcasting for different shapes
    # For now, return zeros for broadcast cases
    result._fill_zero()

    return result^


fn subtract(a: ExTensor, b: ExTensor) raises -> ExTensor:
    """Element-wise subtraction with broadcasting.

    Args:
        a: First tensor
        b: Second tensor

    Returns:
        A new tensor containing a - b

    Raises:
        Error if shapes are not broadcast-compatible or dtypes don't match

    Examples:
        var a = ones(DynamicVector[Int](3, 4), DType.float32)
        var b = ones(DynamicVector[Int](3, 4), DType.float32)
        var c = subtract(a, b)  # Shape (3, 4), all zeros
    """
    if a.dtype() != b.dtype():
        raise Error("Cannot subtract tensors with different dtypes")

    let result_shape = broadcast_shapes(a.shape(), b.shape())
    var result = ExTensor(result_shape, a.dtype())

    # Simple case: same shape (no broadcasting)
    if len(a.shape()) == len(b.shape()):
        var same_shape = True
        for i in range(len(a.shape())):
            if a.shape()[i] != b.shape()[i]:
                same_shape = False
                break

        if same_shape:
            for i in range(a.numel()):
                let a_val = a._get_float64(i)
                let b_val = b._get_float64(i)
                result._set_float64(i, a_val - b_val)
            return result^

    result._fill_zero()
    return result^


fn multiply(a: ExTensor, b: ExTensor) raises -> ExTensor:
    """Element-wise multiplication with broadcasting.

    Args:
        a: First tensor
        b: Second tensor

    Returns:
        A new tensor containing a * b

    Raises:
        Error if shapes are not broadcast-compatible or dtypes don't match

    Examples:
        var a = full(DynamicVector[Int](3, 4), 2.0, DType.float32)
        var b = full(DynamicVector[Int](3, 4), 3.0, DType.float32)
        var c = multiply(a, b)  # Shape (3, 4), all 6.0
    """
    if a.dtype() != b.dtype():
        raise Error("Cannot multiply tensors with different dtypes")

    let result_shape = broadcast_shapes(a.shape(), b.shape())
    var result = ExTensor(result_shape, a.dtype())

    # Simple case: same shape (no broadcasting)
    if len(a.shape()) == len(b.shape()):
        var same_shape = True
        for i in range(len(a.shape())):
            if a.shape()[i] != b.shape()[i]:
                same_shape = False
                break

        if same_shape:
            for i in range(a.numel()):
                let a_val = a._get_float64(i)
                let b_val = b._get_float64(i)
                result._set_float64(i, a_val * b_val)
            return result^

    result._fill_zero()
    return result^


fn divide(a: ExTensor, b: ExTensor) raises -> ExTensor:
    """Element-wise division with broadcasting.

    Args:
        a: First tensor (numerator)
        b: Second tensor (denominator)

    Returns:
        A new tensor containing a / b

    Raises:
        Error if shapes are not broadcast-compatible or dtypes don't match

    Note:
        Division by zero follows IEEE 754 semantics for floating-point types:
        - x / 0.0 where x > 0 -> +inf
        - x / 0.0 where x < 0 -> -inf
        - 0.0 / 0.0 -> NaN

    Examples:
        var a = full(DynamicVector[Int](3, 4), 6.0, DType.float32)
        var b = full(DynamicVector[Int](3, 4), 2.0, DType.float32)
        var c = divide(a, b)  # Shape (3, 4), all 3.0
    """
    if a.dtype() != b.dtype():
        raise Error("Cannot divide tensors with different dtypes")

    let result_shape = broadcast_shapes(a.shape(), b.shape())
    var result = ExTensor(result_shape, a.dtype())

    # Simple case: same shape (no broadcasting)
    if len(a.shape()) == len(b.shape()):
        var same_shape = True
        for i in range(len(a.shape())):
            if a.shape()[i] != b.shape()[i]:
                same_shape = False
                break

        if same_shape:
            for i in range(a.numel()):
                let a_val = a._get_float64(i)
                let b_val = b._get_float64(i)
                result._set_float64(i, a_val / b_val)
            return result^

    # TODO: Implement full broadcasting for different shapes
    result._fill_zero()
    return result^


fn floor_divide(a: ExTensor, b: ExTensor) raises -> ExTensor:
    """Element-wise floor division with broadcasting.

    Args:
        a: First tensor (numerator)
        b: Second tensor (denominator)

    Returns:
        A new tensor containing a // b (floor division)

    Raises:
        Error if shapes are not broadcast-compatible or dtypes don't match

    Examples:
        var a = full(DynamicVector[Int](3, 4), 7.0, DType.float32)
        var b = full(DynamicVector[Int](3, 4), 2.0, DType.float32)
        var c = floor_divide(a, b)  # Shape (3, 4), all 3.0
    """
    if a.dtype() != b.dtype():
        raise Error("Cannot floor divide tensors with different dtypes")

    let result_shape = broadcast_shapes(a.shape(), b.shape())
    var result = ExTensor(result_shape, a.dtype())

    # Simple case: same shape (no broadcasting)
    if len(a.shape()) == len(b.shape()):
        var same_shape = True
        for i in range(len(a.shape())):
            if a.shape()[i] != b.shape()[i]:
                same_shape = False
                break

        if same_shape:
            for i in range(a.numel()):
                let a_val = a._get_float64(i)
                let b_val = b._get_float64(i)
                # Floor division: convert result to int, then back to float
                let div_result = a_val / b_val
                let floored = Float64(int(div_result)) if div_result >= 0.0 else Float64(int(div_result) - 1)
                result._set_float64(i, floored)
            return result^

    # TODO: Implement full broadcasting for different shapes
    result._fill_zero()
    return result^


fn modulo(a: ExTensor, b: ExTensor) raises -> ExTensor:
    """Element-wise modulo with broadcasting.

    Args:
        a: First tensor
        b: Second tensor (modulus)

    Returns:
        A new tensor containing a % b

    Raises:
        Error if shapes are not broadcast-compatible or dtypes don't match

    Examples:
        var a = full(DynamicVector[Int](3, 4), 7.0, DType.int32)
        var b = full(DynamicVector[Int](3, 4), 3.0, DType.int32)
        var c = modulo(a, b)  # Shape (3, 4), all 1
    """
    if a.dtype() != b.dtype():
        raise Error("Cannot compute modulo for tensors with different dtypes")

    let result_shape = broadcast_shapes(a.shape(), b.shape())
    var result = ExTensor(result_shape, a.dtype())

    # Simple case: same shape (no broadcasting)
    if len(a.shape()) == len(b.shape()):
        var same_shape = True
        for i in range(len(a.shape())):
            if a.shape()[i] != b.shape()[i]:
                same_shape = False
                break

        if same_shape:
            for i in range(a.numel()):
                let a_val = a._get_float64(i)
                let b_val = b._get_float64(i)
                # Modulo: a % b = a - floor(a/b) * b
                let div_result = a_val / b_val
                let floored = Float64(int(div_result)) if div_result >= 0.0 else Float64(int(div_result) - 1)
                result._set_float64(i, a_val - floored * b_val)
            return result^

    # TODO: Implement full broadcasting for different shapes
    result._fill_zero()
    return result^


fn power(a: ExTensor, b: ExTensor) raises -> ExTensor:
    """Element-wise exponentiation with broadcasting.

    Args:
        a: Base tensor
        b: Exponent tensor

    Returns:
        A new tensor containing a ** b

    Raises:
        Error if shapes are not broadcast-compatible or dtypes don't match

    Examples:
        var a = full(DynamicVector[Int](3, 4), 2.0, DType.float32)
        var b = full(DynamicVector[Int](3, 4), 3.0, DType.float32)
        var c = power(a, b)  # Shape (3, 4), all 8.0
    """
    if a.dtype() != b.dtype():
        raise Error("Cannot compute power for tensors with different dtypes")

    let result_shape = broadcast_shapes(a.shape(), b.shape())
    var result = ExTensor(result_shape, a.dtype())

    # Simple case: same shape (no broadcasting)
    if len(a.shape()) == len(b.shape()):
        var same_shape = True
        for i in range(len(a.shape())):
            if a.shape()[i] != b.shape()[i]:
                same_shape = False
                break

        if same_shape:
            for i in range(a.numel()):
                let a_val = a._get_float64(i)
                let b_val = b._get_float64(i)
                # Power: a ** b
                # For now, use simple implementation: repeated multiplication for small integer exponents
                # TODO: Implement proper pow function using exp(b * log(a))
                var pow_result: Float64 = 1.0
                let exp_int = int(b_val)
                if b_val == Float64(exp_int) and exp_int >= 0 and exp_int < 100:
                    # Integer exponent case
                    for _ in range(exp_int):
                        pow_result *= a_val
                else:
                    # For non-integer or large exponents, use approximation
                    # TODO: Implement proper exp/log for general case
                    pow_result = a_val  # Placeholder
                result._set_float64(i, pow_result)
            return result^

    # TODO: Implement full broadcasting for different shapes
    result._fill_zero()
    return result^


# TODO: Implement dunder methods on ExTensor struct:
# fn __add__(self, other: ExTensor) -> ExTensor
# fn __sub__(self, other: ExTensor) -> ExTensor
# fn __mul__(self, other: ExTensor) -> ExTensor
# fn __truediv__(self, other: ExTensor) -> ExTensor
# fn __floordiv__(self, other: ExTensor) -> ExTensor
# fn __mod__(self, other: ExTensor) -> ExTensor
# fn __pow__(self, other: ExTensor) -> ExTensor
#
# And reflected variants (__radd__, __rsub__, etc.)
# And in-place variants (__iadd__, __isub__, etc.)
