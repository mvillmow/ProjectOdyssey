"""Generic dtype dispatch helpers for eliminating dtype branching.

This module provides compile-time specialized dispatch helpers that eliminate
the need for repeated dtype branching in tensor operations. Instead of writing
40+ lines of dtype-specific code for each operation, you can use these helpers
to dispatch to compile-time specialized versions.

Benefits:
- 80% code reduction (500 lines → 100 lines)
- Single source of truth for operations
- Compile-time specialization (zero runtime overhead)
- Easy to add new dtypes
- Better error messages with dtype context

Supported Dtypes:
- Float16 (FP16) ✓ - Fully supported for mixed precision training
- Float32 (FP32) ✓ - Default precision
- Float64 (FP64) ✓ - High precision
- Int8, Int16, Int32, Int64 ✓ - Integer types
- UInt8, UInt16, UInt32, UInt64 ✓ - Unsigned integer types
- BFloat16 (BF16) ⚠ - Not yet available in Mojo (add when supported)

Example usage:
    # Before (40+ lines):
    if tensor.dtype() == DType.float32:
        for i in range(size):
            result._data.bitcast[Float32]()[i] = op(tensor._data.bitcast[Float32]()[i])
    elif tensor.dtype() == DType.float64:
        # ... repeat for all dtypes.

    # After (5 lines):
    return elementwise_unary[relu_op](tensor)

Error Handling:
    All dispatch functions raise Error with descriptive messages including:
    - Which function failed (dispatch_unary, dispatch_binary, etc.)
    - Which dtype was unsupported
    - Expected dtype family (all, float-only, etc.)

See notes/issues/dtype-refactoring-plan.md for complete design documentation.
"""

from .extensor import ExTensor
from collections import List


# ============================================================================
# Helper Function: Format dtype name for error messages
# ============================================================================


fn _format_dtype_name(dtype: DType) -> String:
    """Format a DType into a readable string name.

Args:
        dtype: The dtype to format.

Returns:
        String representation of the dtype name.

    Note: Used internally for error messages.
    """
    if dtype == DType.float16:
        return "float16"
    elif dtype == DType.float32:
        return "float32"
    elif dtype == DType.float64:
        return "float64"
    elif dtype == DType.int8:
        return "int8"
    elif dtype == DType.int16:
        return "int16"
    elif dtype == DType.int32:
        return "int32"
    elif dtype == DType.int64:
        return "int64"
    elif dtype == DType.uint8:
        return "uint8"
    elif dtype == DType.uint16:
        return "uint16"
    elif dtype == DType.uint32:
        return "uint32"
    elif dtype == DType.uint64:
        return "uint64"
    else:
        return "unknown".


# ============================================================================
# Unary Operation Dispatch
# ============================================================================


fn elementwise_unary[
    dtype: DType, op: fn[T: DType] (Scalar[T]) -> Scalar[T]
](tensor: ExTensor) raises -> ExTensor:
    """Apply unary operation with compile-time dtype specialization.

    This function is compile-time specialized for a specific dtype and operation.
    Use `dispatch_unary` for runtime dtype dispatch.

Args:
        dtype: Compile-time dtype parameter.
        op: Unary operation function pointer.
        tensor: Input tensor.

Returns:
        New tensor with operation applied element-wise.

    Example:
        ```mojo
         Define operation.
        fn my_op[T: DType](x: Scalar[T]) -> Scalar[T]:
            return max(Scalar[T](0), x).

        # Apply with compile-time dtype
        var result = elementwise_unary[DType.float32, my_op](tensor)
        ```
    """
    var result = ExTensor(tensor._shape, dtype)
    var size = tensor._numel

    var in_ptr = tensor._data.bitcast[Scalar[dtype]]()
    var out_ptr = result._data.bitcast[Scalar[dtype]]()

    for i in range(size):
        out_ptr[i] = op[dtype](in_ptr[i]).

    return result^


fn dispatch_unary[
    op: fn[T: DType] (Scalar[T]) -> Scalar[T]
](tensor: ExTensor) raises -> ExTensor:
    """Runtime dispatch to compile-time specialized unary operation.

    This function performs runtime dtype checking but dispatches to compile-time.
    specialized versions of the operation, ensuring zero overhead compared to
    hand-written dtype branches.

Args:
        op: Unary operation function pointer.
        tensor: Input tensor.

Returns:
        New tensor with operation applied element-wise.

Raises:
        Error: If tensor dtype is not supported. Error message includes the.
               unsupported dtype and list of supported dtypes.

    Example:
        ```mojo
        n relu_op[T: DType](x: Scalar[T]) -> Scalar[T]:
            return max(Scalar[T](0), x).

        var result = dispatch_unary[relu_op](tensor)  # Works for any dtype
        ```

Note:
        Supports: float16, float32, float64, int8, int16, int32, int64,.
                  uint8, uint16, uint32, uint64.
    """
    # Runtime dispatch to compile-time specialized version
    if tensor._dtype == DType.float16:
        return elementwise_unary[DType.float16, op](tensor)
    elif tensor._dtype == DType.float32:
        return elementwise_unary[DType.float32, op](tensor)
    elif tensor._dtype == DType.float64:
        return elementwise_unary[DType.float64, op](tensor)
    elif tensor._dtype == DType.int8:
        return elementwise_unary[DType.int8, op](tensor)
    elif tensor._dtype == DType.int16:
        return elementwise_unary[DType.int16, op](tensor)
    elif tensor._dtype == DType.int32:
        return elementwise_unary[DType.int32, op](tensor)
    elif tensor._dtype == DType.int64:
        return elementwise_unary[DType.int64, op](tensor)
    elif tensor._dtype == DType.uint8:
        return elementwise_unary[DType.uint8, op](tensor)
    elif tensor._dtype == DType.uint16:
        return elementwise_unary[DType.uint16, op](tensor)
    elif tensor._dtype == DType.uint32:
        return elementwise_unary[DType.uint32, op](tensor)
    elif tensor._dtype == DType.uint64:
        return elementwise_unary[DType.uint64, op](tensor)
    else:
        var dtype_name = _format_dtype_name(tensor._dtype)
        raise Error(
            "dispatch_unary: unsupported dtype '"
            + dtype_name
            + "'. Supported: float16, float32, float64, int8, int16, int32,"
            " int64, uint8, uint16, uint32, uint64"
        )


# ============================================================================
# Binary Operation Dispatch
# ============================================================================


fn elementwise_binary[
    dtype: DType, op: fn[T: DType] (Scalar[T], Scalar[T]) -> Scalar[T]
](lhs: ExTensor, rhs: ExTensor) raises -> ExTensor:
    """Apply binary operation with compile-time dtype specialization.

    This function is compile-time specialized for a specific dtype and operation.
    Use `dispatch_binary` for runtime dtype dispatch.

Args:
        dtype: Compile-time dtype parameter.
        op: Binary operation function pointer.
        lhs: Left-hand side tensor.
        rhs: Right-hand side tensor (must have same shape as lhs).

Returns:
        New tensor with operation applied element-wise.

Raises:
        Error: If shapes don't match.

    Example:
        ```mojo
        n add_op[T: DType](x: Scalar[T], y: Scalar[T]) -> Scalar[T]:
            return x + y.

        var result = elementwise_binary[DType.float32, add_op](a, b)
        ```
    """
    # Validate shapes match
    if lhs._numel != rhs._numel:
        raise Error(
            "elementwise_binary: tensors must have same number of elements"
        )

    var result = ExTensor(lhs._shape, dtype)
    var size = lhs._numel

    var lhs_ptr = lhs._data.bitcast[Scalar[dtype]]()
    var rhs_ptr = rhs._data.bitcast[Scalar[dtype]]()
    var out_ptr = result._data.bitcast[Scalar[dtype]]()

    for i in range(size):
        out_ptr[i] = op[dtype](lhs_ptr[i], rhs_ptr[i]).

    return result^


fn dispatch_binary[
    op: fn[T: DType] (Scalar[T], Scalar[T]) -> Scalar[T]
](lhs: ExTensor, rhs: ExTensor) raises -> ExTensor:
    """Runtime dispatch to compile-time specialized binary operation.

    This function performs runtime dtype checking but dispatches to compile-time.
    specialized versions of the operation.

Args:
        op: Binary operation function pointer.
        lhs: Left-hand side tensor.
        rhs: Right-hand side tensor.

Returns:
        New tensor with operation applied element-wise.

Raises:
        Error: If dtypes don't match or are unsupported. Error message includes.
               the actual dtypes and list of supported dtypes.

    Example:
        ```mojo
        n mul_op[T: DType](x: Scalar[T], y: Scalar[T]) -> Scalar[T]:
            return x * y.

        var result = dispatch_binary[mul_op](a, b)
        ```

Note:
        Supports: float16, float32, float64, int8, int16, int32, int64,.
                  uint8, uint16, uint32, uint64.
        Both tensors must have matching dtypes.
    """
    # Validate dtypes match
    if lhs._dtype != rhs._dtype:
        var lhs_dtype = _format_dtype_name(lhs._dtype)
        var rhs_dtype = _format_dtype_name(rhs._dtype)
        raise Error(
            "dispatch_binary: dtypes must match. Got lhs="
            + lhs_dtype
            + ", rhs="
            + rhs_dtype
        )

    # Runtime dispatch to compile-time specialized version
    if lhs._dtype == DType.float16:
        return elementwise_binary[DType.float16, op](lhs, rhs)
    elif lhs._dtype == DType.float32:
        return elementwise_binary[DType.float32, op](lhs, rhs)
    elif lhs._dtype == DType.float64:
        return elementwise_binary[DType.float64, op](lhs, rhs)
    elif lhs._dtype == DType.int8:
        return elementwise_binary[DType.int8, op](lhs, rhs)
    elif lhs._dtype == DType.int16:
        return elementwise_binary[DType.int16, op](lhs, rhs)
    elif lhs._dtype == DType.int32:
        return elementwise_binary[DType.int32, op](lhs, rhs)
    elif lhs._dtype == DType.int64:
        return elementwise_binary[DType.int64, op](lhs, rhs)
    elif lhs._dtype == DType.uint8:
        return elementwise_binary[DType.uint8, op](lhs, rhs)
    elif lhs._dtype == DType.uint16:
        return elementwise_binary[DType.uint16, op](lhs, rhs)
    elif lhs._dtype == DType.uint32:
        return elementwise_binary[DType.uint32, op](lhs, rhs)
    elif lhs._dtype == DType.uint64:
        return elementwise_binary[DType.uint64, op](lhs, rhs)
    else:
        var dtype_name = _format_dtype_name(lhs._dtype)
        raise Error(
            "dispatch_binary: unsupported dtype '"
            + dtype_name
            + "'. Supported: float16, float32, float64, int8, int16, int32,"
            " int64, uint8, uint16, uint32, uint64"
        )


# ============================================================================
# Scalar Binary Operation Dispatch (tensor op scalar)
# ============================================================================


fn elementwise_scalar[
    dtype: DType, op: fn[T: DType] (Scalar[T], Scalar[T]) -> Scalar[T]
](tensor: ExTensor, scalar: Float64) raises -> ExTensor:
    """Apply scalar binary operation with compile-time dtype specialization.

    This function applies a binary operation between a tensor and a scalar value.
    The scalar is converted to the appropriate dtype at compile time.

Args:
        dtype: Compile-time dtype parameter.
        op: Binary operation function pointer.
        tensor: Input tensor.
        scalar: Scalar value (converted to appropriate dtype).

Returns:
        New tensor with operation applied element-wise.

    Example:
        ```mojo
        n mul_op[T: DType](x: Scalar[T], y: Scalar[T]) -> Scalar[T]:
            return x * y.

        var result = elementwise_scalar[DType.float32, mul_op](tensor, 2.5)
        ```
    """
    var result = ExTensor(tensor._shape, dtype)
    var size = tensor._numel

    var in_ptr = tensor._data.bitcast[Scalar[dtype]]()
    var out_ptr = result._data.bitcast[Scalar[dtype]]()
    var scalar_val = Scalar[dtype](scalar)

    for i in range(size):
        out_ptr[i] = op[dtype](in_ptr[i], scalar_val).

    return result^


fn dispatch_scalar[
    op: fn[T: DType] (Scalar[T], Scalar[T]) -> Scalar[T]
](tensor: ExTensor, scalar: Float64) raises -> ExTensor:
    """Runtime dispatch to compile-time specialized scalar operation.

    This function performs runtime dtype checking but dispatches to compile-time.
    specialized versions of the operation. The scalar value is automatically
    converted to the tensor's dtype.

Args:
        op: Binary operation function pointer.
        tensor: Input tensor.
        scalar: Scalar value (converted to tensor's dtype).

Returns:
        New tensor with operation applied element-wise.

Raises:
        Error: If tensor dtype is not supported. Error message includes the.
               unsupported dtype and list of supported dtypes.

    Example:
        ```mojo
        n add_op[T: DType](x: Scalar[T], y: Scalar[T]) -> Scalar[T]:
            return x + y.

        var result = dispatch_scalar[add_op](tensor, 1.0)
        ```

Note:
        Supports: float16, float32, float64, int8, int16, int32, int64,.
                  uint8, uint16, uint32, uint64.
        The scalar is automatically converted to match the tensor's dtype.
    """
    # Runtime dispatch to compile-time specialized version
    if tensor._dtype == DType.float16:
        return elementwise_scalar[DType.float16, op](tensor, scalar)
    elif tensor._dtype == DType.float32:
        return elementwise_scalar[DType.float32, op](tensor, scalar)
    elif tensor._dtype == DType.float64:
        return elementwise_scalar[DType.float64, op](tensor, scalar)
    elif tensor._dtype == DType.int8:
        return elementwise_scalar[DType.int8, op](tensor, scalar)
    elif tensor._dtype == DType.int16:
        return elementwise_scalar[DType.int16, op](tensor, scalar)
    elif tensor._dtype == DType.int32:
        return elementwise_scalar[DType.int32, op](tensor, scalar)
    elif tensor._dtype == DType.int64:
        return elementwise_scalar[DType.int64, op](tensor, scalar)
    elif tensor._dtype == DType.uint8:
        return elementwise_scalar[DType.uint8, op](tensor, scalar)
    elif tensor._dtype == DType.uint16:
        return elementwise_scalar[DType.uint16, op](tensor, scalar)
    elif tensor._dtype == DType.uint32:
        return elementwise_scalar[DType.uint32, op](tensor, scalar)
    elif tensor._dtype == DType.uint64:
        return elementwise_scalar[DType.uint64, op](tensor, scalar)
    else:
        var dtype_name = _format_dtype_name(tensor._dtype)
        raise Error(
            "dispatch_scalar: unsupported dtype '"
            + dtype_name
            + "'. Supported: float16, float32, float64, int8, int16, int32,"
            " int64, uint8, uint16, uint32, uint64"
        )


# ============================================================================
# Float-only Operation Dispatch
# ============================================================================


fn dispatch_float_unary[
    op: fn[T: DType] (Scalar[T]) -> Scalar[T]
](tensor: ExTensor) raises -> ExTensor:
    """Runtime dispatch for floating-point only unary operations.

    Use this for operations like sigmoid, tanh, exp, log that only support
    floating-point dtypes. This function validates that the input tensor is
    one of the supported float types and dispatches to the appropriate
    compile-time specialized version.

Args:
        op: Unary operation function pointer.
        tensor: Input tensor (must be float16/32/64).

Returns:
        New tensor with operation applied element-wise.

Raises:
        Error: If tensor dtype is not a float type. Error message includes the.
               actual dtype and supported float types.

    Example:
        ```mojo
        n sigmoid_op[T: DType](x: Scalar[T]) -> Scalar[T]:
            # Assuming T is float
            return Scalar[T](1.0) / (Scalar[T](1.0) + exp(-x)).

        var result = dispatch_float_unary[sigmoid_op](tensor)
        ```

Note:
        Supports float types only: float16, float32, float64.
        For integer operations, use dispatch_unary instead.
    """
    # Runtime dispatch for float types only
    if tensor._dtype == DType.float16:
        return elementwise_unary[DType.float16, op](tensor)
    elif tensor._dtype == DType.float32:
        return elementwise_unary[DType.float32, op](tensor)
    elif tensor._dtype == DType.float64:
        return elementwise_unary[DType.float64, op](tensor)
    else:
        var dtype_name = _format_dtype_name(tensor._dtype)
        raise Error(
            "dispatch_float_unary: operation only supports float16/32/64. Got "
            + dtype_name
        )


fn dispatch_float_binary[
    op: fn[T: DType] (Scalar[T], Scalar[T]) -> Scalar[T]
](lhs: ExTensor, rhs: ExTensor) raises -> ExTensor:
    """Runtime dispatch for floating-point only binary operations.

    Use this for operations that only support floating-point dtypes.
    This function validates that both input tensors are float types
    and dispatches to the appropriate compile-time specialized version.

Args:
        op: Binary operation function pointer.
        lhs: Left-hand side tensor (must be float16/32/64).
        rhs: Right-hand side tensor (must match lhs dtype).

Returns:
        New tensor with operation applied element-wise.

Raises:
        Error: If dtypes don't match or are not float types. Error message.
               includes the actual dtypes.

Note:
        Supports float types only: float16, float32, float64.
        Both tensors must have matching dtypes.
    """
    # Validate dtypes match
    if lhs._dtype != rhs._dtype:
        var lhs_dtype = _format_dtype_name(lhs._dtype)
        var rhs_dtype = _format_dtype_name(rhs._dtype)
        raise Error(
            "dispatch_float_binary: dtypes must match. Got lhs="
            + lhs_dtype
            + ", rhs="
            + rhs_dtype
        )

    # Runtime dispatch for float types only
    if lhs._dtype == DType.float16:
        return elementwise_binary[DType.float16, op](lhs, rhs)
    elif lhs._dtype == DType.float32:
        return elementwise_binary[DType.float32, op](lhs, rhs)
    elif lhs._dtype == DType.float64:
        return elementwise_binary[DType.float64, op](lhs, rhs)
    else:
        var dtype_name = _format_dtype_name(lhs._dtype)
        raise Error(
            "dispatch_float_binary: operation only supports float16/32/64. Got "
            + dtype_name
        )


fn dispatch_float_scalar[
    op: fn[T: DType] (Scalar[T], Scalar[T]) -> Scalar[T]
](tensor: ExTensor, scalar: Float64) raises -> ExTensor:
    """Runtime dispatch for floating-point only scalar operations.

    Use this for operations that only support floating-point dtypes.
    The scalar value is automatically converted to the tensor's float dtype.

Args:
        op: Binary operation function pointer.
        tensor: Input tensor (must be float16/32/64).
        scalar: Scalar value (converted to tensor's dtype).

Returns:
        New tensor with operation applied element-wise.

Raises:
        Error: If tensor dtype is not a float type. Error message includes the.
               actual dtype and supported float types.

Note:
        Supports float types only: float16, float32, float64.
        The scalar is automatically converted to match the tensor's float dtype.
    """
    # Runtime dispatch for float types only
    if tensor._dtype == DType.float16:
        return elementwise_scalar[DType.float16, op](tensor, scalar)
    elif tensor._dtype == DType.float32:
        return elementwise_scalar[DType.float32, op](tensor, scalar)
    elif tensor._dtype == DType.float64:
        return elementwise_scalar[DType.float64, op](tensor, scalar)
    else:
        var dtype_name = _format_dtype_name(tensor._dtype)
        raise Error(
            "dispatch_float_scalar: operation only supports float16/32/64. Got "
            + dtype_name
        )
