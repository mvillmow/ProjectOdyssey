"""Tensor validation utilities for shape and dtype checking.

Provides common validation functions for tensor shape and data type compatibility.
These utilities are used throughout the library to validate tensor arguments and
detect shape/dtype mismatches early.

Functions:
    validate_tensor_shape: Validate tensor has expected shape
    validate_tensor_dtype: Validate tensor has expected dtype
    validate_matching_tensors: Validate two tensors have matching shape and dtype
    validate_2d_input: Validate tensor is 2D
    validate_4d_input: Validate tensor is 4D
"""

from .extensor import ExTensor


fn validate_tensor_shape(
    tensor: ExTensor, expected_shape: List[Int], name: String
) raises:
    """Validate that a tensor has the expected shape.

    Checks if the tensor's shape matches the expected shape. If not, raises
    an error with a descriptive message including the tensor name.

Args:
        tensor: The tensor to validate.
        expected_shape: The expected shape as a List[Int].
        name: The name of the tensor (for error messages).

Raises:
        Error: If the tensor shape does not match the expected shape.

    Example:
        ```mojo
        var x = zeros([2, 3], DType.float32)
        var expected : List[Int] = [2, 3]
        validate_tensor_shape(x, expected, "input")  # Passes
        ```
    """
    var actual_shape = tensor.shape()

    # Check number of dimensions
    if len(actual_shape) != len(expected_shape):
        raise Error(
            name
            + ": expected "
            + String(len(expected_shape))
            + "D tensor, got "
            + String(len(actual_shape))
            + "D"
        )

    # Check each dimension
    for i in range(len(expected_shape)):
        if actual_shape[i] != expected_shape[i]:
            raise Error(
                name
                + ": expected shape ["
                + _shape_to_string(expected_shape)
                + "], got ["
                + _shape_to_string(actual_shape)
                + "]"
            )


fn validate_tensor_dtype(
    tensor: ExTensor, expected_dtype: DType, name: String
) raises:
    """Validate that a tensor has the expected data type.

    Checks if the tensor's dtype matches the expected dtype. If not, raises
    an error with a descriptive message including the tensor name.

Args:
        tensor: The tensor to validate.
        expected_dtype: The expected data type.
        name: The name of the tensor (for error messages).

Raises:
        Error: If the tensor dtype does not match the expected dtype.

    Example:
        ```mojo
        var x = zeros([2, 3], DType.float32)
        validate_tensor_dtype(x, DType.float32, "input")  # Passes
        validate_tensor_dtype(x, DType.float64, "input")  # Raises error
        ```
    """
    var actual_dtype = tensor.dtype()
    if actual_dtype != expected_dtype:
        raise Error(
            name
            + ": expected dtype "
            + _dtype_to_string(expected_dtype)
            + ", got "
            + _dtype_to_string(actual_dtype)
        )


fn validate_matching_tensors(
    a: ExTensor, b: ExTensor, a_name: String, b_name: String
) raises:
    """Validate that two tensors have matching shape and dtype.

    Checks if two tensors have the same shape and data type. Useful for
    verifying that tensors can be used together in element-wise operations.

Args:
        a: The first tensor.
        b: The second tensor.
        a_name: The name of the first tensor (for error messages).
        b_name: The name of the second tensor (for error messages).

Raises:
        Error: If tensors have different shapes or dtypes.

    Example:
        ```mojo
        var x = zeros([2, 3], DType.float32)
        var y = ones([2, 3], DType.float32)
        validate_matching_tensors(x, y, "x", "y")  # Passes

        var z = zeros([3, 2], DType.float32)
        validate_matching_tensors(x, z, "x", "z")  # Raises error
        ```
    """
    var shape_a = a.shape()
    var shape_b = b.shape()
    var dtype_a = a.dtype()
    var dtype_b = b.dtype()

    # Check dtype match
    if dtype_a != dtype_b:
        raise Error(
            a_name
            + " and "
            + b_name
            + " have mismatched dtypes: "
            + _dtype_to_string(dtype_a)
            + " vs "
            + _dtype_to_string(dtype_b)
        )

    # Check shape match
    if len(shape_a) != len(shape_b):
        raise Error(
            a_name
            + " and "
            + b_name
            + " have mismatched number of dimensions: "
            + String(len(shape_a))
            + " vs "
            + String(len(shape_b))
        )

    for i in range(len(shape_a)):
        if shape_a[i] != shape_b[i]:
            raise Error(
                a_name
                + " and "
                + b_name
                + " have mismatched shapes: ["
                + _shape_to_string(shape_a)
                + "] vs ["
                + _shape_to_string(shape_b)
                + "]"
            )


fn validate_2d_input(tensor: ExTensor, name: String) raises:
    """Validate that a tensor is exactly 2-dimensional.

Args:
        tensor: The tensor to validate.
        name: The name of the tensor (for error messages).

Raises:
        Error: If the tensor is not 2D.

    Example:
        ```mojo
        var x = zeros([2, 3], DType.float32)
        validate_2d_input(x, "input")  # Passes

        var y = zeros([2, 3, 4], DType.float32)
        validate_2d_input(y, "input")  # Raises error
        ```
    """
    var shape = tensor.shape()
    var ndim = len(shape)

    if ndim != 2:
        raise Error(
            name
            + ": expected 2D tensor, got "
            + String(ndim)
            + "D with shape ["
            + _shape_to_string(shape)
            + "]"
        )


fn validate_4d_input(tensor: ExTensor, name: String) raises:
    """Validate that a tensor is exactly 4-dimensional.

Args:
        tensor: The tensor to validate.
        name: The name of the tensor (for error messages).

Raises:
        Error: If the tensor is not 4D.

    Example:
        ```mojo
        var x = zeros([2, 3, 4, 5], DType.float32)
        validate_4d_input(x, "input")  # Passes

        var y = zeros([2, 3, 4], DType.float32)
        validate_4d_input(y, "input")  # Raises error
        ```
    """
    var shape = tensor.shape()
    var ndim = len(shape)

    if ndim != 4:
        raise Error(
            name
            + ": expected 4D tensor, got "
            + String(ndim)
            + "D with shape ["
            + _shape_to_string(shape)
            + "]"
        )


# ============================================================================
# Helper Functions for Error Message Formatting
# ============================================================================


fn _shape_to_string(shape: List[Int]) -> String:
    """Convert a shape list to a string representation.

Args:
        shape: The shape as a List[Int].

Returns:
        A string like "2, 3, 4" for shape [2, 3, 4].
    """
    if len(shape) == 0:
        return ""

    var result = String(shape[0])
    for i in range(1, len(shape)):
        result += ", " + String(shape[i])

    return result


fn _dtype_to_string(dtype: DType) -> String:
    """Convert a DType to a readable string representation.

Args:
        dtype: The data type.

Returns:
        A string like "float32", "int64", etc.
    """
    if dtype == DType.float32:
        return "float32"
    elif dtype == DType.float64:
        return "float64"
    elif dtype == DType.float16:
        return "float16"
    elif dtype == DType.int32:
        return "int32"
    elif dtype == DType.int64:
        return "int64"
    elif dtype == DType.int16:
        return "int16"
    elif dtype == DType.int8:
        return "int8"
    elif dtype == DType.uint32:
        return "uint32"
    elif dtype == DType.uint64:
        return "uint64"
    elif dtype == DType.uint16:
        return "uint16"
    elif dtype == DType.uint8:
        return "uint8"
    elif dtype == DType.bool:
        return "bool"
    else:
        return "unknown"
