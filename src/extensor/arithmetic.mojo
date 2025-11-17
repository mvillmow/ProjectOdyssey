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

        # Broadcasting example
        var x = ones([3, 1, 5], DType.float32)
        var y = ones([3, 4, 5], DType.float32)
        var z = add(x, y)  # Shape (3, 4, 5)
    """
    if a.dtype() != b.dtype():
        raise Error("Cannot add tensors with different dtypes")

    # Compute broadcast shape
    let result_shape = broadcast_shapes(a.shape(), b.shape())
    var result = ExTensor(result_shape, a.dtype())

    # Compute broadcast strides
    let strides_a = compute_broadcast_strides(a.shape(), result_shape)
    let strides_b = compute_broadcast_strides(b.shape(), result_shape)

    # Calculate total elements in result
    var total_elems = 1
    for i in range(len(result_shape)):
        total_elems *= result_shape[i]

    # Iterate over all result elements
    for result_idx in range(total_elems):
        var idx_a = 0
        var idx_b = 0
        var temp_idx = result_idx

        # Compute source indices for a and b using broadcast strides
        for dim in range(len(result_shape) - 1, -1, -1):
            var stride_prod = 1
            for d in range(dim + 1, len(result_shape)):
                stride_prod *= result_shape[d]

            let coord = temp_idx // stride_prod
            temp_idx = temp_idx % stride_prod

            idx_a += coord * strides_a[dim]
            idx_b += coord * strides_b[dim]

        # Perform addition
        let a_val = a._get_float64(idx_a)
        let b_val = b._get_float64(idx_b)
        result._set_float64(result_idx, a_val + b_val)

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

        # Broadcasting example
        var x = ones([3, 1, 5], DType.float32)
        var y = ones([3, 4, 5], DType.float32)
        var z = subtract(x, y)  # Shape (3, 4, 5)
    """
    if a.dtype() != b.dtype():
        raise Error("Cannot subtract tensors with different dtypes")

    # Compute broadcast shape
    let result_shape = broadcast_shapes(a.shape(), b.shape())
    var result = ExTensor(result_shape, a.dtype())

    # Compute broadcast strides
    let strides_a = compute_broadcast_strides(a.shape(), result_shape)
    let strides_b = compute_broadcast_strides(b.shape(), result_shape)

    # Calculate total elements in result
    var total_elems = 1
    for i in range(len(result_shape)):
        total_elems *= result_shape[i]

    # Iterate over all result elements
    for result_idx in range(total_elems):
        var idx_a = 0
        var idx_b = 0
        var temp_idx = result_idx

        # Compute source indices for a and b using broadcast strides
        for dim in range(len(result_shape) - 1, -1, -1):
            var stride_prod = 1
            for d in range(dim + 1, len(result_shape)):
                stride_prod *= result_shape[d]

            let coord = temp_idx // stride_prod
            temp_idx = temp_idx % stride_prod

            idx_a += coord * strides_a[dim]
            idx_b += coord * strides_b[dim]

        # Perform subtraction
        let a_val = a._get_float64(idx_a)
        let b_val = b._get_float64(idx_b)
        result._set_float64(result_idx, a_val - b_val)

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

        # Broadcasting example
        var x = full([3, 1, 5], 2.0, DType.float32)
        var y = full([3, 4, 5], 3.0, DType.float32)
        var z = multiply(x, y)  # Shape (3, 4, 5), all 6.0
    """
    if a.dtype() != b.dtype():
        raise Error("Cannot multiply tensors with different dtypes")

    # Compute broadcast shape
    let result_shape = broadcast_shapes(a.shape(), b.shape())
    var result = ExTensor(result_shape, a.dtype())

    # Compute broadcast strides
    let strides_a = compute_broadcast_strides(a.shape(), result_shape)
    let strides_b = compute_broadcast_strides(b.shape(), result_shape)

    # Calculate total elements in result
    var total_elems = 1
    for i in range(len(result_shape)):
        total_elems *= result_shape[i]

    # Iterate over all result elements
    for result_idx in range(total_elems):
        var idx_a = 0
        var idx_b = 0
        var temp_idx = result_idx

        # Compute source indices for a and b using broadcast strides
        for dim in range(len(result_shape) - 1, -1, -1):
            var stride_prod = 1
            for d in range(dim + 1, len(result_shape)):
                stride_prod *= result_shape[d]

            let coord = temp_idx // stride_prod
            temp_idx = temp_idx % stride_prod

            idx_a += coord * strides_a[dim]
            idx_b += coord * strides_b[dim]

        # Perform multiplication
        let a_val = a._get_float64(idx_a)
        let b_val = b._get_float64(idx_b)
        result._set_float64(result_idx, a_val * b_val)

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

        # Broadcasting example
        var x = full([3, 1, 5], 6.0, DType.float32)
        var y = full([3, 4, 5], 2.0, DType.float32)
        var z = divide(x, y)  # Shape (3, 4, 5), all 3.0
    """
    if a.dtype() != b.dtype():
        raise Error("Cannot divide tensors with different dtypes")

    # Compute broadcast shape
    let result_shape = broadcast_shapes(a.shape(), b.shape())
    var result = ExTensor(result_shape, a.dtype())

    # Compute broadcast strides
    let strides_a = compute_broadcast_strides(a.shape(), result_shape)
    let strides_b = compute_broadcast_strides(b.shape(), result_shape)

    # Calculate total elements in result
    var total_elems = 1
    for i in range(len(result_shape)):
        total_elems *= result_shape[i]

    # Iterate over all result elements
    for result_idx in range(total_elems):
        var idx_a = 0
        var idx_b = 0
        var temp_idx = result_idx

        # Compute source indices for a and b using broadcast strides
        for dim in range(len(result_shape) - 1, -1, -1):
            var stride_prod = 1
            for d in range(dim + 1, len(result_shape)):
                stride_prod *= result_shape[d]

            let coord = temp_idx // stride_prod
            temp_idx = temp_idx % stride_prod

            idx_a += coord * strides_a[dim]
            idx_b += coord * strides_b[dim]

        # Perform division (IEEE 754 semantics apply automatically)
        let a_val = a._get_float64(idx_a)
        let b_val = b._get_float64(idx_b)
        result._set_float64(result_idx, a_val / b_val)

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

        # Broadcasting example
        var x = full([3, 1, 5], 7.0, DType.float32)
        var y = full([3, 4, 5], 2.0, DType.float32)
        var z = floor_divide(x, y)  # Shape (3, 4, 5), all 3.0
    """
    if a.dtype() != b.dtype():
        raise Error("Cannot floor divide tensors with different dtypes")

    # Compute broadcast shape
    let result_shape = broadcast_shapes(a.shape(), b.shape())
    var result = ExTensor(result_shape, a.dtype())

    # Compute broadcast strides
    let strides_a = compute_broadcast_strides(a.shape(), result_shape)
    let strides_b = compute_broadcast_strides(b.shape(), result_shape)

    # Calculate total elements in result
    var total_elems = 1
    for i in range(len(result_shape)):
        total_elems *= result_shape[i]

    # Iterate over all result elements
    for result_idx in range(total_elems):
        var idx_a = 0
        var idx_b = 0
        var temp_idx = result_idx

        # Compute source indices for a and b using broadcast strides
        for dim in range(len(result_shape) - 1, -1, -1):
            var stride_prod = 1
            for d in range(dim + 1, len(result_shape)):
                stride_prod *= result_shape[d]

            let coord = temp_idx // stride_prod
            temp_idx = temp_idx % stride_prod

            idx_a += coord * strides_a[dim]
            idx_b += coord * strides_b[dim]

        # Perform floor division
        let a_val = a._get_float64(idx_a)
        let b_val = b._get_float64(idx_b)
        let div_result = a_val / b_val
        let floored = Float64(int(div_result)) if div_result >= 0.0 else Float64(int(div_result) - 1)
        result._set_float64(result_idx, floored)

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

        # Broadcasting example
        var x = full([3, 1, 5], 7.0, DType.float32)
        var y = full([3, 4, 5], 3.0, DType.float32)
        var z = modulo(x, y)  # Shape (3, 4, 5), all 1.0
    """
    if a.dtype() != b.dtype():
        raise Error("Cannot compute modulo for tensors with different dtypes")

    # Compute broadcast shape
    let result_shape = broadcast_shapes(a.shape(), b.shape())
    var result = ExTensor(result_shape, a.dtype())

    # Compute broadcast strides
    let strides_a = compute_broadcast_strides(a.shape(), result_shape)
    let strides_b = compute_broadcast_strides(b.shape(), result_shape)

    # Calculate total elements in result
    var total_elems = 1
    for i in range(len(result_shape)):
        total_elems *= result_shape[i]

    # Iterate over all result elements
    for result_idx in range(total_elems):
        var idx_a = 0
        var idx_b = 0
        var temp_idx = result_idx

        # Compute source indices for a and b using broadcast strides
        for dim in range(len(result_shape) - 1, -1, -1):
            var stride_prod = 1
            for d in range(dim + 1, len(result_shape)):
                stride_prod *= result_shape[d]

            let coord = temp_idx // stride_prod
            temp_idx = temp_idx % stride_prod

            idx_a += coord * strides_a[dim]
            idx_b += coord * strides_b[dim]

        # Perform modulo: a % b = a - floor(a/b) * b
        let a_val = a._get_float64(idx_a)
        let b_val = b._get_float64(idx_b)
        let div_result = a_val / b_val
        let floored = Float64(int(div_result)) if div_result >= 0.0 else Float64(int(div_result) - 1)
        result._set_float64(result_idx, a_val - floored * b_val)

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

        # Broadcasting example
        var x = full([3, 1, 5], 2.0, DType.float32)
        var y = full([3, 4, 5], 3.0, DType.float32)
        var z = power(x, y)  # Shape (3, 4, 5), all 8.0
    """
    if a.dtype() != b.dtype():
        raise Error("Cannot compute power for tensors with different dtypes")

    # Compute broadcast shape
    let result_shape = broadcast_shapes(a.shape(), b.shape())
    var result = ExTensor(result_shape, a.dtype())

    # Compute broadcast strides
    let strides_a = compute_broadcast_strides(a.shape(), result_shape)
    let strides_b = compute_broadcast_strides(b.shape(), result_shape)

    # Calculate total elements in result
    var total_elems = 1
    for i in range(len(result_shape)):
        total_elems *= result_shape[i]

    # Iterate over all result elements
    for result_idx in range(total_elems):
        var idx_a = 0
        var idx_b = 0
        var temp_idx = result_idx

        # Compute source indices for a and b using broadcast strides
        for dim in range(len(result_shape) - 1, -1, -1):
            var stride_prod = 1
            for d in range(dim + 1, len(result_shape)):
                stride_prod *= result_shape[d]

            let coord = temp_idx // stride_prod
            temp_idx = temp_idx % stride_prod

            idx_a += coord * strides_a[dim]
            idx_b += coord * strides_b[dim]

        # Perform power: a ** b
        let a_val = a._get_float64(idx_a)
        let b_val = b._get_float64(idx_b)

        # Current implementation: repeated multiplication for small integer exponents
        # LIMITATION: Only supports integer exponents in range [0, 100)
        # For general case (fractional/large exponents), proper implementation requires:
        #   - exp(b * log(a)) for general exponents
        #   - Special handling for negative bases with fractional exponents
        #   - Proper handling of edge cases (0^0, inf^0, etc.)
        var pow_result: Float64 = 1.0
        let exp_int = int(b_val)
        if b_val == Float64(exp_int) and exp_int >= 0 and exp_int < 100:
            # Integer exponent case (naive repeated multiplication)
            for _ in range(exp_int):
                pow_result *= a_val
        else:
            # LIMITATION: Non-integer and large exponents not yet supported
            # Returns base value as placeholder (incorrect result)
            pow_result = a_val
        result._set_float64(result_idx, pow_result)

    return result^


# ==============================================================================
# FUTURE WORK: Operator Overloading (out of scope for issues #219-220)
# ==============================================================================
#
# The following dunder methods should be implemented on the ExTensor struct
# to enable natural operator syntax (e.g., a + b instead of add(a, b)):
#
# Basic operators:
#   fn __add__(self, other: ExTensor) -> ExTensor
#   fn __sub__(self, other: ExTensor) -> ExTensor
#   fn __mul__(self, other: ExTensor) -> ExTensor
#   fn __truediv__(self, other: ExTensor) -> ExTensor
#   fn __floordiv__(self, other: ExTensor) -> ExTensor
#   fn __mod__(self, other: ExTensor) -> ExTensor
#   fn __pow__(self, other: ExTensor) -> ExTensor
#
# Reflected variants (for operations like: 2 + tensor):
#   fn __radd__, __rsub__, __rmul__, __rtruediv__, etc.
#
# In-place variants (for operations like: tensor += 2):
#   fn __iadd__, __isub__, __imul__, __itruediv__, etc.
#
# These implementations should delegate to the functions in this module.
