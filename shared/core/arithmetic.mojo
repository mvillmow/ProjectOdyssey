"""Arithmetic operations for ExTensor with broadcasting support.

Implements element-wise arithmetic operations following NumPy-style broadcasting.
"""

from collections.vector import DynamicVector
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
# Backward Pass (Gradient Computation)
# ==============================================================================


fn _reduce_broadcast_dims(grad: ExTensor, original_shape: DynamicVector[Int]) raises -> ExTensor:
    """Reduce gradient from broadcast shape back to original shape.

    When forward pass broadcasts input from original_shape to grad.shape(),
    backward pass must sum gradient back to original_shape.

    Args:
        grad: Gradient tensor (broadcast shape)
        original_shape: Original input shape before broadcasting

    Returns:
        Reduced gradient matching original_shape

    Examples:
        # Broadcasting (3, 1, 5) → (3, 4, 5)
        var grad = ones([3, 4, 5])  # Gradient from loss
        var original = [3, 1, 5]    # Original input shape
        var reduced = _reduce_broadcast_dims(grad, original)  # Shape (3, 1, 5)

        # Prepended dimensions: (5,) → (3, 4, 5)
        var grad2 = ones([3, 4, 5])
        var original2 = [5]
        var reduced2 = _reduce_broadcast_dims(grad2, original2)  # Shape (5,)
    """
    from .reduction import sum

    var result = grad
    let grad_shape = grad.shape()
    let grad_ndim = len(grad_shape)
    let orig_ndim = len(original_shape)

    # Handle prepended dimensions (when original had fewer dims)
    # Example: original (5,) broadcast to (3, 4, 5)
    # Need to sum over first (grad_ndim - orig_ndim) dimensions
    if orig_ndim < grad_ndim:
        let dims_to_sum = grad_ndim - orig_ndim
        for i in range(dims_to_sum):
            # Always sum over axis 0 since shape shrinks each time
            result = sum(result, axis=0, keepdims=False)

    # Now handle dimensions that were size 1 and got broadcast
    # Example: (3, 1, 5) → (3, 4, 5), sum over axis 1 keeping dims
    for i in range(min(orig_ndim, grad_ndim)):
        let dim_idx = i if orig_ndim < grad_ndim else i + (grad_ndim - orig_ndim)
        if i < orig_ndim and original_shape[i] == 1 and i < len(result.shape()) and result.shape()[i] > 1:
            result = sum(result, axis=i, keepdims=True)

    return result


fn add_backward(grad_output: ExTensor, a_shape: DynamicVector[Int], b_shape: DynamicVector[Int]) raises -> (ExTensor, ExTensor):
    """Compute gradients for element-wise addition.

    For C = A + B, given ∂L/∂C, computes:
        ∂L/∂A = ∂L/∂C (summed over broadcasted dimensions)
        ∂L/∂B = ∂L/∂C (summed over broadcasted dimensions)

    Handles broadcasting: If input was broadcast to output shape, gradient
    is summed back to the original input shape.

    Args:
        grad_output: Gradient from upstream (∂L/∂C)
        a_shape: Original shape of first input (A)
        b_shape: Original shape of second input (B)

    Returns:
        Tuple of (grad_a, grad_b) - gradients w.r.t. inputs

    Examples:
        # No broadcasting
        var a = ones(DynamicVector[Int](3, 4), DType.float32)
        var b = ones(DynamicVector[Int](3, 4), DType.float32)
        var c = add(a, b)
        var grad_c = ones(DynamicVector[Int](3, 4), DType.float32)
        var grads = add_backward(grad_c, a.shape(), b.shape())

        # With broadcasting
        var x = ones(DynamicVector[Int](3, 1), DType.float32)
        var y = ones(DynamicVector[Int](3, 4), DType.float32)
        var z = add(x, y)  # Shape (3, 4)
        var grad_z = ones(DynamicVector[Int](3, 4), DType.float32)
        var grads = add_backward(grad_z, x.shape(), y.shape())
        # grad_x will be shape (3, 1) - summed over broadcast dimension
    """
    # For addition, gradient passes through but must be reduced for broadcasting
    var grad_a = _reduce_broadcast_dims(grad_output, a_shape)
    var grad_b = _reduce_broadcast_dims(grad_output, b_shape)

    return (grad_a, grad_b)


fn subtract_backward(grad_output: ExTensor, a_shape: DynamicVector[Int], b_shape: DynamicVector[Int]) raises -> (ExTensor, ExTensor):
    """Compute gradients for element-wise subtraction.

    For C = A - B, given ∂L/∂C, computes:
        ∂L/∂A = ∂L/∂C (reduced for broadcasting)
        ∂L/∂B = -∂L/∂C (negated and reduced for broadcasting)

    The gradient for B is negated since ∂(A-B)/∂B = -1.

    Args:
        grad_output: Gradient from upstream (∂L/∂C)
        a_shape: Original shape of first input (A)
        b_shape: Original shape of second input (B)

    Returns:
        Tuple of (grad_a, grad_b) - gradients w.r.t. inputs
    """
    # Gradient for A passes through unchanged (but reduced for broadcasting)
    var grad_a = _reduce_broadcast_dims(grad_output, a_shape)

    # Gradient for B is negated
    # Create a tensor of -1s with same shape as grad_output
    var neg_grad = ExTensor(grad_output.shape(), grad_output.dtype())
    for i in range(grad_output.numel()):
        neg_grad._set_float64(i, -grad_output._get_float64(i))

    # Reduce for broadcasting
    var grad_b = _reduce_broadcast_dims(neg_grad, b_shape)

    return (grad_a, grad_b)


fn multiply_backward(grad_output: ExTensor, a: ExTensor, b: ExTensor) raises -> (ExTensor, ExTensor):
    """Compute gradients for element-wise multiplication.

    For C = A * B, given ∂L/∂C, computes:
        ∂L/∂A = ∂L/∂C * B  (product rule, reduced for broadcasting)
        ∂L/∂B = ∂L/∂C * A  (reduced for broadcasting)

    Args:
        grad_output: Gradient from upstream (∂L/∂C)
        a: First input from forward pass (A)
        b: Second input from forward pass (B)

    Returns:
        Tuple of (grad_a, grad_b) - gradients w.r.t. inputs

    Examples:
        var a = ones(DynamicVector[Int](3, 4), DType.float32)
        var b = ones(DynamicVector[Int](3, 4), DType.float32)
        var c = multiply(a, b)
        var grad_c = ones(DynamicVector[Int](3, 4), DType.float32)
        var grads = multiply_backward(grad_c, a, b)
    """
    # grad_a = grad_output * b (then reduce for broadcasting)
    var grad_a_unreduced = multiply(grad_output, b)
    var grad_a = _reduce_broadcast_dims(grad_a_unreduced, a.shape())

    # grad_b = grad_output * a (then reduce for broadcasting)
    var grad_b_unreduced = multiply(grad_output, a)
    var grad_b = _reduce_broadcast_dims(grad_b_unreduced, b.shape())

    return (grad_a, grad_b)


fn divide_backward(grad_output: ExTensor, a: ExTensor, b: ExTensor) raises -> (ExTensor, ExTensor):
    """Compute gradients for element-wise division.

    For C = A / B, given ∂L/∂C, computes:
        ∂L/∂A = ∂L/∂C / B  (quotient rule numerator)
        ∂L/∂B = -∂L/∂C * A / B²  (quotient rule denominator)

    Includes numerical stability: adds small epsilon to prevent division by zero.

    Args:
        grad_output: Gradient from upstream (∂L/∂C)
        a: First input from forward pass (A)
        b: Second input from forward pass (B)

    Returns:
        Tuple of (grad_a, grad_b) - gradients w.r.t. inputs

    Examples:
        var a = ones(DynamicVector[Int](3, 4), DType.float32)
        var b = full(DynamicVector[Int](3, 4), 2.0, DType.float32)
        var c = divide(a, b)
        var grad_c = ones(DynamicVector[Int](3, 4), DType.float32)
        var grads = divide_backward(grad_c, a, b)

    Numerical Stability:
        Uses epsilon = 1e-10 to prevent division by zero in b².
    """
    alias EPSILON = 1e-10

    # grad_a = grad_output / b (then reduce for broadcasting)
    var grad_a_unreduced = divide(grad_output, b)
    var grad_a = _reduce_broadcast_dims(grad_a_unreduced, a.shape())

    # grad_b = -grad_output * a / b²
    # Add epsilon to b² for numerical stability
    var b_squared = multiply(b, b)

    # Add epsilon to prevent division by zero
    var b_squared_safe = ExTensor(b_squared.shape(), b_squared.dtype())
    for i in range(b_squared.numel()):
        let val = b_squared._get_float64(i)
        b_squared_safe._set_float64(i, val + EPSILON)

    # Compute -grad_output * a / b²
    var temp = multiply(grad_output, a)
    var grad_b_positive = divide(temp, b_squared_safe)

    # Negate it
    var grad_b_unreduced = ExTensor(grad_b_positive.shape(), grad_b_positive.dtype())
    for i in range(grad_b_positive.numel()):
        grad_b_unreduced._set_float64(i, -grad_b_positive._get_float64(i))

    # Reduce for broadcasting
    var grad_b = _reduce_broadcast_dims(grad_b_unreduced, b.shape())

    return (grad_a, grad_b)


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
