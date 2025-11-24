"""Assertion helpers for ExTensor testing.

Provides comprehensive assertion functions for validating tensor properties,
values, shapes, dtypes, and numerical accuracy.

Note: These functions work with ExTensor through duck typing.
Import ExTensor in your test files before using these assertions.
"""

from math import isnan, isinf


fn assert_true(condition: Bool, message: String = "") raises:
    """Assert a condition is true.

    Args:
        condition: The condition to check
        message: Optional error message

    Raises:
        Error if condition is false
    """
    if not condition:
        if message:
            raise Error("Assertion failed: " + message)
        else:
            raise Error("Assertion failed")


fn assert_false(condition: Bool, message: String = "") raises:
    """Assert a condition is false.

    Args:
        condition: The condition to check
        message: Optional error message

    Raises:
        Error if condition is true
    """
    if condition:
        if message:
            raise Error("Assertion failed: " + message)
        else:
            raise Error("Assertion failed: expected false")


fn assert_equal_int(a: Int, b: Int, message: String = "") raises:
    """Assert two integers are equal.

    Args:
        a: First integer
        b: Second integer
        message: Optional error message

    Raises:
        Error if integers are not equal
    """
    if a != b:
        var msg = "Expected " + String(a) + " == " + String(b)
        if message:
            msg = message + ": " + msg
        raise Error(msg)


fn assert_equal_float(a: Float64, b: Float64, tolerance: Float64 = 1e-8, message: String = "") raises:
    """Assert two floats are equal within tolerance.

    Args:
        a: First float
        b: Second float
        tolerance: Numerical tolerance
        message: Optional error message

    Raises:
        Error if floats differ beyond tolerance
    """
    var diff = abs(a - b)
    if diff > tolerance:
        var msg = "Expected " + String(a) + " ≈ " + String(b) + " (diff=" + String(diff) + ")"
        if message:
            msg = message + ": " + msg
        raise Error(msg)


fn assert_close_float(
    a: Float64,
    b: Float64,
    rtol: Float64 = 1e-5,
    atol: Float64 = 1e-8,
    message: String = ""
) raises:
    """Assert two floats are numerically close.

    Uses the formula: |a - b| <= atol + rtol * |b|

    Args:
        a: First float
        b: Second float
        rtol: Relative tolerance
        atol: Absolute tolerance
        message: Optional error message

    Raises:
        Error if floats differ beyond tolerance
    """
    # Handle NaN and inf
    var a_is_nan = isnan(a)
    var b_is_nan = isnan(b)
    var a_is_inf = isinf(a)
    var b_is_inf = isinf(b)

    if a_is_nan and b_is_nan:
        return  # Both NaN, considered equal

    if a_is_nan or b_is_nan:
        var msg = "NaN mismatch: " + String(a) + " vs " + String(b)
        if message:
            msg = message + ": " + msg
        raise Error(msg)

    if a_is_inf or b_is_inf:
        if a != b:
            var msg = "Infinity mismatch: " + String(a) + " vs " + String(b)
            if message:
                msg = message + ": " + msg
            raise Error(msg)
        return

    # Check numeric closeness
    var diff = abs(a - b)
    var threshold = atol + rtol * abs(b)

    if diff > threshold:
        var msg = (
            "Values differ: " + String(a) + " vs " + String(b) + " (diff=" + String(diff) +
            ", threshold=" + String(threshold) + ")"
        )
        if message:
            msg = message + ": " + msg
        raise Error(msg)


# ============================================================================
# ExTensor-Specific Assertions
# ============================================================================
# Note: Import ExTensor in your test files before using these assertions

fn assert_shape[T: AnyType](tensor: T, expected: List[Int], message: String = "") raises:
    """Assert tensor has expected shape.

    Args:
        tensor: ExTensor to check
        expected: Expected shape as List
        message: Optional error message

    Raises:
        Error if shapes don't match
    """
    # Get actual shape
    var actual_shape = tensor.shape()

    # Check dimensions match
    if len(actual_shape) != len(expected):
        var msg = "Shape dimension mismatch: expected " + String(len(expected)) + " dims, got " + String(len(actual_shape))
        if message:
            msg = message + ": " + msg
        raise Error(msg)

    # Check each dimension
    for i in range(len(expected)):
        if actual_shape[i] != expected[i]:
            var msg = "Shape mismatch at dim " + String(i) + ": expected " + String(expected[i]) + ", got " + String(actual_shape[i])
            if message:
                msg = message + ": " + msg
            raise Error(msg)


fn assert_dtype[T: AnyType](tensor: T, expected_dtype: DType, message: String = "") raises:
    """Assert tensor has expected dtype.

    Args:
        tensor: ExTensor to check
        expected_dtype: Expected DType
        message: Optional error message

    Raises:
        Error if dtypes don't match
    """
    var actual_dtype = tensor.dtype()
    if actual_dtype != expected_dtype:
        var msg = "DType mismatch: expected " + String(expected_dtype) + ", got " + String(actual_dtype)
        if message:
            msg = message + ": " + msg
        raise Error(msg)


fn assert_numel[T: AnyType](tensor: T, expected_numel: Int, message: String = "") raises:
    """Assert tensor has expected number of elements.

    Args:
        tensor: ExTensor to check
        expected_numel: Expected number of elements
        message: Optional error message

    Raises:
        Error if numel doesn't match
    """
    var actual_numel = tensor.numel()
    if actual_numel != expected_numel:
        var msg = "Element count mismatch: expected " + String(expected_numel) + ", got " + String(actual_numel)
        if message:
            msg = message + ": " + msg
        raise Error(msg)


fn assert_dim[T: AnyType](tensor: T, expected_dim: Int, message: String = "") raises:
    """Assert tensor has expected number of dimensions.

    Args:
        tensor: ExTensor to check
        expected_dim: Expected number of dimensions
        message: Optional error message

    Raises:
        Error if dimensions don't match
    """
    var actual_dim = len(tensor.shape())
    if actual_dim != expected_dim:
        var msg = "Dimension count mismatch: expected " + String(expected_dim) + ", got " + String(actual_dim)
        if message:
            msg = message + ": " + msg
        raise Error(msg)


fn assert_value_at[T: AnyType](tensor: T, index: Int, expected_value: Float64, tolerance: Float64 = 1e-8, message: String = "") raises:
    """Assert tensor has expected value at given index.

    Args:
        tensor: ExTensor to check
        index: Flat index to check
        expected_value: Expected value
        tolerance: Numerical tolerance
        message: Optional error message

    Raises:
        Error if value doesn't match
    """
    var actual_value = tensor._get_float64(index)
    var diff = math_abs(actual_value - expected_value)

    if diff > tolerance:
        var msg = "Value mismatch at index " + String(index) + ": expected " + String(expected_value) + ", got " + String(actual_value) + " (diff=" + String(diff) + ")"
        if message:
            msg = message + ": " + msg
        raise Error(msg)


fn assert_all_values[T: AnyType](tensor: T, expected_value: Float64, tolerance: Float64 = 1e-8, message: String = "") raises:
    """Assert all tensor elements equal expected value.

    Args:
        tensor: ExTensor to check
        expected_value: Expected value for all elements
        tolerance: Numerical tolerance
        message: Optional error message

    Raises:
        Error if any value doesn't match
    """
    var numel = tensor.numel()
    for i in range(numel):
        var actual_value = tensor._get_float64(i)
        var diff = abs(actual_value - expected_value)

        if diff > tolerance:
            var msg = "Value mismatch at index " + String(i) + ": expected " + String(expected_value) + ", got " + String(actual_value)
            if message:
                msg = message + ": " + msg
            raise Error(msg)


fn assert_all_close[T: AnyType](
    a: T,
    b: T,
    rtol: Float64 = 1e-5,
    atol: Float64 = 1e-8,
    message: String = ""
) raises:
    """Assert two tensors have numerically close values.

    Uses the formula: |a - b| <= atol + rtol * |b|

    Args:
        a: First ExTensor
        b: Second ExTensor
        rtol: Relative tolerance
        atol: Absolute tolerance
        message: Optional error message

    Raises:
        Error if shapes don't match or values differ beyond tolerance
    """
    # Check shapes match
    var shape_a = a.shape()
    var shape_b = b.shape()

    if len(shape_a) != len(shape_b):
        raise Error("Shape dimension mismatch: " + String(len(shape_a)) + " vs " + String(len(shape_b)))

    for i in range(len(shape_a)):
        if shape_a[i] != shape_b[i]:
            raise Error("Shape mismatch at dim " + String(i) + ": " + String(shape_a[i]) + " vs " + String(shape_b[i]))

    # Check all values
    var numel = a.numel()
    for i in range(numel):
        var val_a = a._get_float64(i)
        var val_b = b._get_float64(i)

        # Handle NaN
        if isnan(val_a) and isnan(val_b):
            continue

        if isnan(val_a) or isnan(val_b):
            var msg = "NaN mismatch at index " + String(i) + ": " + String(val_a) + " vs " + String(val_b)
            if message:
                msg = message + ": " + msg
            raise Error(msg)

        # Handle infinity
        if isinf(val_a) or isinf(val_b):
            if val_a != val_b:
                var msg = "Infinity mismatch at index " + String(i) + ": " + String(val_a) + " vs " + String(val_b)
                if message:
                    msg = message + ": " + msg
                raise Error(msg)
            continue

        # Check numeric closeness
        var diff = math_abs(val_a - val_b)
        var threshold = atol + rtol * abs(val_b)

        if diff > threshold:
            var msg = "Value mismatch at index " + String(i) + ": " + String(val_a) + " vs " + String(val_b) + " (diff=" + String(diff) + ", threshold=" + String(threshold) + ")"
            if message:
                msg = message + ": " + msg
            raise Error(msg)


fn assert_contiguous[T: AnyType](tensor: T, message: String = "") raises:
    """Assert tensor is contiguous in memory.

    Args:
        tensor: ExTensor to check
        message: Optional error message

    Raises:
        Error if tensor is not contiguous
    """
    if not tensor.is_contiguous():
        var msg = "Tensor is not contiguous"
        if message:
            msg = message + ": " + msg
        raise Error(msg)
