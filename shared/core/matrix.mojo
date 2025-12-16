"""Matrix operations for ExTensor.

Implements linear algebra operations like matrix multiplication and transpose

FIXME(#2715): Placeholder tests in tests/shared/core/legacy/test_matrix.mojo (lines 493-560) require:
- inner() function (test_inner_1d, test_inner_2d at lines 493-515) - TODO(#2717)
- tensordot() function (test_tensordot_basic, test_tensordot_multiple_axes at lines 522-560) - TODO(#2717)
Both functions are marked as "TODO: Implement" and tests pass as placeholders (line 501, 515, 537).
See Issue #49 for details
"""

from collections import List
from .extensor import ExTensor
from .gradient_types import GradientPair


# ============================================================================
# Dtype-Specialized Matmul Helpers (Eliminates Float64 Conversions)
# ============================================================================


fn _matmul_2d_2d_impl[
    dtype: DType
](result: ExTensor, a: ExTensor, b: ExTensor, a_rows: Int, a_cols: Int, b_cols: Int):
    """Compile-time specialized 2D@2D matrix multiplication.

    Uses Float64 accumulation for float16/float32 to maintain precision,
    while using typed pointers to avoid per-element conversion overhead.

    Parameters:
        dtype: Compile-time dtype parameter.

    Args:
        result: Pre-allocated result tensor.
        a: Left matrix (a_rows, a_cols).
        b: Right matrix (a_cols, b_cols).
        a_rows: Number of rows in a.
        a_cols: Number of columns in a (= rows in b).
        b_cols: Number of columns in b.
    """
    var a_ptr = a._data.bitcast[Scalar[dtype]]()
    var b_ptr = b._data.bitcast[Scalar[dtype]]()
    var result_ptr = result._data.bitcast[Scalar[dtype]]()

    # Use Float64 accumulation for float16/float32 to preserve precision
    @parameter
    if dtype == DType.float16 or dtype == DType.float32:
        for i in range(a_rows):
            for j in range(b_cols):
                var sum_val: Float64 = 0.0
                for k in range(a_cols):
                    sum_val += Float64(a_ptr[i * a_cols + k]) * Float64(b_ptr[k * b_cols + j])
                result_ptr[i * b_cols + j] = Scalar[dtype](sum_val)
    else:
        for i in range(a_rows):
            for j in range(b_cols):
                var sum_val = Scalar[dtype](0)
                for k in range(a_cols):
                    sum_val += a_ptr[i * a_cols + k] * b_ptr[k * b_cols + j]
                result_ptr[i * b_cols + j] = sum_val


fn _matmul_2d_1d_impl[
    dtype: DType
](result: ExTensor, a: ExTensor, b: ExTensor, m: Int, k: Int):
    """Compile-time specialized 2D@1D matrix-vector multiplication.

    Uses Float64 accumulation for float16/float32 to maintain precision.

    Parameters:
        dtype: Compile-time dtype parameter.

    Args:
        result: Pre-allocated result tensor (m,).
        a: Matrix (m, k).
        b: Vector (k,).
        m: Number of rows.
        k: Number of columns (= vector length).
    """
    var a_ptr = a._data.bitcast[Scalar[dtype]]()
    var b_ptr = b._data.bitcast[Scalar[dtype]]()
    var result_ptr = result._data.bitcast[Scalar[dtype]]()

    # Use Float64 accumulation for float16/float32 to preserve precision
    @parameter
    if dtype == DType.float16 or dtype == DType.float32:
        for i in range(m):
            var sum_val: Float64 = 0.0
            for j in range(k):
                sum_val += Float64(a_ptr[i * k + j]) * Float64(b_ptr[j])
            result_ptr[i] = Scalar[dtype](sum_val)
    else:
        for i in range(m):
            var sum_val = Scalar[dtype](0)
            for j in range(k):
                sum_val += a_ptr[i * k + j] * b_ptr[j]
            result_ptr[i] = sum_val


fn _matmul_1d_2d_impl[
    dtype: DType
](result: ExTensor, a: ExTensor, b: ExTensor, m: Int, n: Int):
    """Compile-time specialized 1D@2D vector-matrix multiplication.

    Uses Float64 accumulation for float16/float32 to maintain precision.

    Parameters:
        dtype: Compile-time dtype parameter.

    Args:
        result: Pre-allocated result tensor (n,).
        a: Vector (m,).
        b: Matrix (m, n).
        m: Vector length (= matrix rows).
        n: Matrix columns.
    """
    var a_ptr = a._data.bitcast[Scalar[dtype]]()
    var b_ptr = b._data.bitcast[Scalar[dtype]]()
    var result_ptr = result._data.bitcast[Scalar[dtype]]()

    # Use Float64 accumulation for float16/float32 to preserve precision
    @parameter
    if dtype == DType.float16 or dtype == DType.float32:
        for j in range(n):
            var sum_val: Float64 = 0.0
            for i in range(m):
                sum_val += Float64(a_ptr[i]) * Float64(b_ptr[i * n + j])
            result_ptr[j] = Scalar[dtype](sum_val)
    else:
        for j in range(n):
            var sum_val = Scalar[dtype](0)
            for i in range(m):
                sum_val += a_ptr[i] * b_ptr[i * n + j]
            result_ptr[j] = sum_val


fn _matmul_batched_impl[
    dtype: DType
](
    result: ExTensor,
    a: ExTensor,
    b: ExTensor,
    batch_size: Int,
    a_rows: Int,
    a_cols: Int,
    b_cols: Int,
    matrix_size_a: Int,
    matrix_size_b: Int,
    matrix_size_result: Int,
):
    """Compile-time specialized batched matrix multiplication.

    Uses Float64 accumulation for float16/float32 to maintain precision.

    Parameters:
        dtype: Compile-time dtype parameter.

    Args:
        result: Pre-allocated result tensor.
        a: Left batched matrix.
        b: Right batched matrix.
        batch_size: Number of batches.
        a_rows: Rows per matrix in a.
        a_cols: Columns per matrix in a.
        b_cols: Columns per matrix in b.
        matrix_size_a: Elements per matrix in a.
        matrix_size_b: Elements per matrix in b.
        matrix_size_result: Elements per result matrix.
    """
    var a_ptr = a._data.bitcast[Scalar[dtype]]()
    var b_ptr = b._data.bitcast[Scalar[dtype]]()
    var result_ptr = result._data.bitcast[Scalar[dtype]]()

    # Use Float64 accumulation for float16/float32 to preserve precision
    @parameter
    if dtype == DType.float16 or dtype == DType.float32:
        for batch in range(batch_size):
            var a_offset = batch * matrix_size_a
            var b_offset = batch * matrix_size_b
            var result_offset = batch * matrix_size_result

            for i in range(a_rows):
                for j in range(b_cols):
                    var sum_val: Float64 = 0.0
                    for k in range(a_cols):
                        var a_idx = a_offset + i * a_cols + k
                        var b_idx = b_offset + k * b_cols + j
                        sum_val += Float64(a_ptr[a_idx]) * Float64(b_ptr[b_idx])
                    var result_idx = result_offset + i * b_cols + j
                    result_ptr[result_idx] = Scalar[dtype](sum_val)
    else:
        for batch in range(batch_size):
            var a_offset = batch * matrix_size_a
            var b_offset = batch * matrix_size_b
            var result_offset = batch * matrix_size_result

            for i in range(a_rows):
                for j in range(b_cols):
                    var sum_val = Scalar[dtype](0)
                    for k in range(a_cols):
                        var a_idx = a_offset + i * a_cols + k
                        var b_idx = b_offset + k * b_cols + j
                        sum_val += a_ptr[a_idx] * b_ptr[b_idx]
                    var result_idx = result_offset + i * b_cols + j
                    result_ptr[result_idx] = sum_val


fn _dispatch_matmul_2d_2d(
    result: ExTensor, a: ExTensor, b: ExTensor, a_rows: Int, a_cols: Int, b_cols: Int
) raises:
    """Runtime dispatch for 2D@2D matmul."""
    if a._dtype == DType.float16:
        _matmul_2d_2d_impl[DType.float16](result, a, b, a_rows, a_cols, b_cols)
    elif a._dtype == DType.float32:
        _matmul_2d_2d_impl[DType.float32](result, a, b, a_rows, a_cols, b_cols)
    elif a._dtype == DType.float64:
        _matmul_2d_2d_impl[DType.float64](result, a, b, a_rows, a_cols, b_cols)
    elif a._dtype == DType.int8:
        _matmul_2d_2d_impl[DType.int8](result, a, b, a_rows, a_cols, b_cols)
    elif a._dtype == DType.int16:
        _matmul_2d_2d_impl[DType.int16](result, a, b, a_rows, a_cols, b_cols)
    elif a._dtype == DType.int32:
        _matmul_2d_2d_impl[DType.int32](result, a, b, a_rows, a_cols, b_cols)
    elif a._dtype == DType.int64:
        _matmul_2d_2d_impl[DType.int64](result, a, b, a_rows, a_cols, b_cols)
    else:
        raise Error("matmul: unsupported dtype")


fn _dispatch_matmul_2d_1d(
    result: ExTensor, a: ExTensor, b: ExTensor, m: Int, k: Int
) raises:
    """Runtime dispatch for 2D@1D matmul."""
    if a._dtype == DType.float16:
        _matmul_2d_1d_impl[DType.float16](result, a, b, m, k)
    elif a._dtype == DType.float32:
        _matmul_2d_1d_impl[DType.float32](result, a, b, m, k)
    elif a._dtype == DType.float64:
        _matmul_2d_1d_impl[DType.float64](result, a, b, m, k)
    elif a._dtype == DType.int8:
        _matmul_2d_1d_impl[DType.int8](result, a, b, m, k)
    elif a._dtype == DType.int16:
        _matmul_2d_1d_impl[DType.int16](result, a, b, m, k)
    elif a._dtype == DType.int32:
        _matmul_2d_1d_impl[DType.int32](result, a, b, m, k)
    elif a._dtype == DType.int64:
        _matmul_2d_1d_impl[DType.int64](result, a, b, m, k)
    else:
        raise Error("matmul: unsupported dtype")


fn _dispatch_matmul_1d_2d(
    result: ExTensor, a: ExTensor, b: ExTensor, m: Int, n: Int
) raises:
    """Runtime dispatch for 1D@2D matmul."""
    if a._dtype == DType.float16:
        _matmul_1d_2d_impl[DType.float16](result, a, b, m, n)
    elif a._dtype == DType.float32:
        _matmul_1d_2d_impl[DType.float32](result, a, b, m, n)
    elif a._dtype == DType.float64:
        _matmul_1d_2d_impl[DType.float64](result, a, b, m, n)
    elif a._dtype == DType.int8:
        _matmul_1d_2d_impl[DType.int8](result, a, b, m, n)
    elif a._dtype == DType.int16:
        _matmul_1d_2d_impl[DType.int16](result, a, b, m, n)
    elif a._dtype == DType.int32:
        _matmul_1d_2d_impl[DType.int32](result, a, b, m, n)
    elif a._dtype == DType.int64:
        _matmul_1d_2d_impl[DType.int64](result, a, b, m, n)
    else:
        raise Error("matmul: unsupported dtype")


fn _dot_1d_impl[dtype: DType](result: ExTensor, a: ExTensor, b: ExTensor, length: Int):
    """Compile-time specialized 1D dot product.

    Uses Float64 accumulation for float16/float32 to maintain precision.
    """
    var a_ptr = a._data.bitcast[Scalar[dtype]]()
    var b_ptr = b._data.bitcast[Scalar[dtype]]()
    var result_ptr = result._data.bitcast[Scalar[dtype]]()

    # Use Float64 accumulation for float16/float32 to preserve precision
    @parameter
    if dtype == DType.float16 or dtype == DType.float32:
        var sum_val: Float64 = 0.0
        for i in range(length):
            sum_val += Float64(a_ptr[i]) * Float64(b_ptr[i])
        result_ptr[0] = Scalar[dtype](sum_val)
    else:
        var sum_val = Scalar[dtype](0)
        for i in range(length):
            sum_val += a_ptr[i] * b_ptr[i]
        result_ptr[0] = sum_val


fn _dispatch_dot_1d(result: ExTensor, a: ExTensor, b: ExTensor, length: Int) raises:
    """Runtime dispatch for 1D dot product."""
    if a._dtype == DType.float16:
        _dot_1d_impl[DType.float16](result, a, b, length)
    elif a._dtype == DType.float32:
        _dot_1d_impl[DType.float32](result, a, b, length)
    elif a._dtype == DType.float64:
        _dot_1d_impl[DType.float64](result, a, b, length)
    elif a._dtype == DType.int8:
        _dot_1d_impl[DType.int8](result, a, b, length)
    elif a._dtype == DType.int16:
        _dot_1d_impl[DType.int16](result, a, b, length)
    elif a._dtype == DType.int32:
        _dot_1d_impl[DType.int32](result, a, b, length)
    elif a._dtype == DType.int64:
        _dot_1d_impl[DType.int64](result, a, b, length)
    else:
        raise Error("dot: unsupported dtype")


fn _outer_impl[dtype: DType](result: ExTensor, a: ExTensor, b: ExTensor, len_a: Int, len_b: Int):
    """Compile-time specialized outer product."""
    var a_ptr = a._data.bitcast[Scalar[dtype]]()
    var b_ptr = b._data.bitcast[Scalar[dtype]]()
    var result_ptr = result._data.bitcast[Scalar[dtype]]()

    for i in range(len_a):
        for j in range(len_b):
            result_ptr[i * len_b + j] = a_ptr[i] * b_ptr[j]


fn _dispatch_outer(result: ExTensor, a: ExTensor, b: ExTensor, len_a: Int, len_b: Int) raises:
    """Runtime dispatch for outer product."""
    if a._dtype == DType.float16:
        _outer_impl[DType.float16](result, a, b, len_a, len_b)
    elif a._dtype == DType.float32:
        _outer_impl[DType.float32](result, a, b, len_a, len_b)
    elif a._dtype == DType.float64:
        _outer_impl[DType.float64](result, a, b, len_a, len_b)
    elif a._dtype == DType.int8:
        _outer_impl[DType.int8](result, a, b, len_a, len_b)
    elif a._dtype == DType.int16:
        _outer_impl[DType.int16](result, a, b, len_a, len_b)
    elif a._dtype == DType.int32:
        _outer_impl[DType.int32](result, a, b, len_a, len_b)
    elif a._dtype == DType.int64:
        _outer_impl[DType.int64](result, a, b, len_a, len_b)
    else:
        raise Error("outer: unsupported dtype")


fn _transpose_indexed_copy_impl[dtype: DType](
    result: ExTensor, tensor: ExTensor, result_idx: Int, input_idx: Int
):
    """Compile-time specialized indexed copy for transpose."""
    var in_ptr = tensor._data.bitcast[Scalar[dtype]]()
    var out_ptr = result._data.bitcast[Scalar[dtype]]()
    out_ptr[result_idx] = in_ptr[input_idx]


fn _dispatch_transpose_indexed_copy(
    result: ExTensor, tensor: ExTensor, result_idx: Int, input_idx: Int
) raises:
    """Runtime dispatch for transpose indexed copy."""
    if tensor._dtype == DType.float16:
        _transpose_indexed_copy_impl[DType.float16](result, tensor, result_idx, input_idx)
    elif tensor._dtype == DType.float32:
        _transpose_indexed_copy_impl[DType.float32](result, tensor, result_idx, input_idx)
    elif tensor._dtype == DType.float64:
        _transpose_indexed_copy_impl[DType.float64](result, tensor, result_idx, input_idx)
    elif tensor._dtype == DType.int8:
        _transpose_indexed_copy_impl[DType.int8](result, tensor, result_idx, input_idx)
    elif tensor._dtype == DType.int16:
        _transpose_indexed_copy_impl[DType.int16](result, tensor, result_idx, input_idx)
    elif tensor._dtype == DType.int32:
        _transpose_indexed_copy_impl[DType.int32](result, tensor, result_idx, input_idx)
    elif tensor._dtype == DType.int64:
        _transpose_indexed_copy_impl[DType.int64](result, tensor, result_idx, input_idx)
    else:
        raise Error("transpose: unsupported dtype")


fn _dispatch_matmul_batched(
    result: ExTensor,
    a: ExTensor,
    b: ExTensor,
    batch_size: Int,
    a_rows: Int,
    a_cols: Int,
    b_cols: Int,
    matrix_size_a: Int,
    matrix_size_b: Int,
    matrix_size_result: Int,
) raises:
    """Runtime dispatch for batched matmul."""
    if a._dtype == DType.float16:
        _matmul_batched_impl[DType.float16](
            result, a, b, batch_size, a_rows, a_cols, b_cols,
            matrix_size_a, matrix_size_b, matrix_size_result
        )
    elif a._dtype == DType.float32:
        _matmul_batched_impl[DType.float32](
            result, a, b, batch_size, a_rows, a_cols, b_cols,
            matrix_size_a, matrix_size_b, matrix_size_result
        )
    elif a._dtype == DType.float64:
        _matmul_batched_impl[DType.float64](
            result, a, b, batch_size, a_rows, a_cols, b_cols,
            matrix_size_a, matrix_size_b, matrix_size_result
        )
    elif a._dtype == DType.int8:
        _matmul_batched_impl[DType.int8](
            result, a, b, batch_size, a_rows, a_cols, b_cols,
            matrix_size_a, matrix_size_b, matrix_size_result
        )
    elif a._dtype == DType.int16:
        _matmul_batched_impl[DType.int16](
            result, a, b, batch_size, a_rows, a_cols, b_cols,
            matrix_size_a, matrix_size_b, matrix_size_result
        )
    elif a._dtype == DType.int32:
        _matmul_batched_impl[DType.int32](
            result, a, b, batch_size, a_rows, a_cols, b_cols,
            matrix_size_a, matrix_size_b, matrix_size_result
        )
    elif a._dtype == DType.int64:
        _matmul_batched_impl[DType.int64](
            result, a, b, batch_size, a_rows, a_cols, b_cols,
            matrix_size_a, matrix_size_b, matrix_size_result
        )
    else:
        raise Error("matmul: unsupported dtype")


fn matmul(a: ExTensor, b: ExTensor) raises -> ExTensor:
    """Matrix multiplication.

    Args:
            a: First tensor (matrix or vector).
            b: Second tensor (matrix or vector).

    Returns:
            A new tensor containing the matrix product a @ b

    Raises:
            Error if dimensions are incompatible

        Requirements:
            - 2D @ 2D: a.shape() = (m, k), b.shape() = (k, n) -> result.shape() = (m, n)
            - 2D @ 1D: a.shape() = (m, k), b.shape() = (k,) -> result.shape() = (m,)
            - 1D @ 2D: a.shape() = (k,), b.shape() = (k, n) -> result.shape() = (n,)
            - ND tensors: batched matrix multiplication

        Preconditions:
            - a and b must have compatible dimensions
            - a and b must have the same dtype
            - Parameters are borrowed immutably - safe for aliased inputs (a and b can be the same tensor)
            - Result is always a new tensor - no in-place modification

    Examples:
            var a = zeros([3, 4], DType.float32)
            var b = zeros([4, 5], DType.float32)
            var c = matmul(a, b)  # Shape (3, 5)

            var W = zeros([10, 5], DType.float32)
            var x = zeros(List[Int](), DType.float32)
            var y = matmul(W, x)  # Shape (10,) - matrix @ vector

    Note:
            This function always allocates a new result tensor. Input tensors are only read,
            never modified, so aliasing between a and b is safe.
    """
    # Check dtype compatibility
    if a.dtype() != b.dtype():
        raise Error("Cannot multiply matrices with different dtypes")

    # Check dimension compatibility
    var a_shape = a.shape()
    var b_shape = b.shape()
    var a_ndim = len(a_shape)
    var b_ndim = len(b_shape)

    # Handle matrix @ vector (2D @ 1D)
    if a_ndim == 2 and b_ndim == 1:
        var m = a_shape[0]
        var k = a_shape[1]
        var n = b_shape[0]

        if k != n:
            raise Error(
                "Incompatible dimensions for matmul: matrix ("
                + String(m)
                + ", "
                + String(k)
                + ") @ vector ("
                + String(n)
                + ")"
            )

        # Result is a vector of shape (m,)
        var result_shape = List[Int](capacity=1)
        result_shape.append(m)
        var result = ExTensor(result_shape, a.dtype())

        # Compute using dtype-specialized implementation
        _dispatch_matmul_2d_1d(result, a, b, m, k)

        return result^

    # Handle vector @ matrix (1D @ 2D)
    if a_ndim == 1 and b_ndim == 2:
        var m = a_shape[0]
        var k = b_shape[0]
        var n = b_shape[1]

        if m != k:
            raise Error(
                "Incompatible dimensions for matmul: vector ("
                + String(m)
                + ") @ matrix ("
                + String(k)
                + ", "
                + String(n)
                + ")"
            )

        # Result is a vector of shape (n,)
        var result_shape = List[Int](capacity=1)
        result_shape.append(n)
        var result = ExTensor(result_shape, a.dtype())

        # Compute using dtype-specialized implementation
        _dispatch_matmul_1d_2d(result, a, b, m, n)

        return result^

    # For 2D and higher, require at least 2D tensors
    if a_ndim < 2 or b_ndim < 2:
        raise Error(
            "matmul requires at least 2D tensors for non-vector inputs (use"
            " dot() for 1D @ 1D)"
        )

    var a_rows = a_shape[len(a_shape) - 2]
    var a_cols = a_shape[len(a_shape) - 1]
    var b_rows = b_shape[len(b_shape) - 2]
    var b_cols = b_shape[len(b_shape) - 1]

    if a_cols != b_rows:
        raise Error(
            "Incompatible dimensions for matmul: "
            + String(a_cols)
            + " != "
            + String(b_rows)
        )

    # Compute output shape
    var result_shape = List[Int](capacity=len(a_shape))

    # Copy batch dimensions (if any)
    for i in range(len(a_shape) - 2):
        result_shape.append(a_shape[i])

    # Add matrix dimensions
    result_shape.append(a_rows)
    result_shape.append(b_cols)

    # Create result
    var result = ExTensor(result_shape, a.dtype())

    # Implement matrix multiplication
    # For 2D case: result[i, j] = sum(a[i, k] * b[k, j] for k in range(a_cols))
    # For batched case: apply same logic to each batch

    if len(a_shape) == 2:
        # Simple 2D matrix multiplication using dtype-specialized implementation
        _dispatch_matmul_2d_2d(result, a, b, a_rows, a_cols, b_cols)
    else:
        # Batched matrix multiplication (3D+)
        # Compute batch size (product of all dimensions except last 2)
        var batch_size = 1
        for i in range(len(a_shape) - 2):
            batch_size *= a_shape[i]

        var matrix_size_a = a_rows * a_cols
        var matrix_size_b = b_rows * b_cols
        var matrix_size_result = a_rows * b_cols

        # Use dtype-specialized batched implementation
        _dispatch_matmul_batched(
            result, a, b, batch_size, a_rows, a_cols, b_cols,
            matrix_size_a, matrix_size_b, matrix_size_result
        )

    return result^


fn transpose(
    tensor: ExTensor, axes: Optional[List[Int]] = None
) raises -> ExTensor:
    """Transpose tensor dimensions with optional axis permutation.

        Supports arbitrary axis permutation for N-dimensional tensors, matching NumPy semantics

    Args:
            tensor: Input tensor.
            axes: Optional permutation of axes. If None, reverses all axes (default behavior)
                    Must be a permutation of [0, 1, ..., ndim-1] with no duplicates
                    Example: axes=[2, 0, 1] permutes (N, H, W, C) -> (C, N, H, W).

    Returns:
            A new tensor with permuted dimensions according to axes

    Raises:
            Error if axes is invalid (duplicates, wrong range, or wrong length)

    Examples:
            # Default: reverse all axes
            var t = zeros([3, 4], DType.float32)
            var t_T = transpose(t)  # Shape (4, 3)

            # Custom permutation: (2, 3, 4) -> (4, 3, 2) with axes=[2, 0, 1]
            var t3d = zeros([2, 3, 4], DType.float32)
            var axes  = List[Int]()
            axes.append(2)
            axes.append(0)
            axes.append(1)
            var t3d_perm = transpose(t3d, axes)  # Shape (4, 2, 3)

    Note:
            - If axes is None, defaults to reversing all axes (same as original behavior)
            - Validates axes parameter: no duplicates, correct range, correct length
            - Matches NumPy transpose semantics for arbitrary permutations.
    """
    var ndim = tensor.dim()
    var input_shape = tensor.shape()

    # Handle default case (None or default): reverse all axes
    var perm = axes
    if perm is None:
        perm = List[Int](capacity=ndim)
        for i in range(ndim - 1, -1, -1):
            perm.value().append(i)

    # Validate axes parameter
    if len(perm.value()) != ndim:
        raise Error(
            "axes length ("
            + String(len(perm.value()))
            + ") does not match tensor dimensions ("
            + String(ndim)
            + ")"
        )

    # Check for duplicates and valid range
    var seen = List[Bool](length=ndim, fill=False)

    for axis in perm.value():
        if axis < 0 or axis >= ndim:
            raise Error(
                "axis "
                + String(axis)
                + " is out of bounds for tensor with "
                + String(ndim)
                + " dimensions"
            )
        if seen[axis]:
            raise Error("duplicate axis " + String(axis) + " in permutation")
        seen[axis] = True

    # Build result shape using permutation
    var result_shape = List[Int](capacity=ndim)
    for axis in perm.value():
        result_shape.append(input_shape[axis])

    var result = ExTensor(result_shape, tensor.dtype())

    # Compute strides for input tensor (row-major order)
    var input_strides = List[Int](capacity=ndim)
    var stride = 1
    for i in range(ndim - 1, -1, -1):
        input_strides.append(stride)
        stride *= input_shape[i]
    # Reverse to get correct indexing order
    var temp_strides = List[Int](capacity=ndim)
    for i in range(len(input_strides) - 1, -1, -1):
        temp_strides.append(input_strides[i])
    input_strides = temp_strides^

    # For each element in result, map to input position
    for result_idx in range(result.numel()):
        # Convert linear result index to coordinates
        var result_coords = List[Int](capacity=ndim)
        for _ in range(ndim):
            result_coords.append(0)
        var temp_idx = result_idx
        for i in range(ndim - 1, -1, -1):
            result_coords[i] = temp_idx % result_shape[i]
            temp_idx //= result_shape[i]

        # Map result coordinates to input coordinates using permutation
        var input_coords = List[Int](capacity=ndim)
        for _ in range(ndim):
            input_coords.append(0)
        for result_axis in range(ndim):
            var input_axis = perm.value()[result_axis]
            input_coords[input_axis] = result_coords[result_axis]

        # Convert input coordinates to linear index
        var input_idx = 0
        for i in range(ndim):
            input_idx += input_coords[i] * input_strides[i]

        # Copy value using dtype-specialized implementation
        _dispatch_transpose_indexed_copy(result, tensor, result_idx, input_idx)

    return result^


fn dot(a: ExTensor, b: ExTensor) raises -> ExTensor:
    """Dot product of tensors.

    Args:
            a: First tensor.
            b: Second tensor.

    Returns:
            Dot product (scalar for 1D, matrix product for 2D)

    Raises:
            Error: If tensor shapes are incompatible.

    Examples:
        ```
            var a = ones(List[Int](), DType.float32)
            var b = ones(List[Int](), DType.float32)
            var c = dot(a, b)  # Scalar 5.0
        ```
    """
    # Check dtype compatibility
    if a.dtype() != b.dtype():
        raise Error("Cannot compute dot product with different dtypes")

    # For 1D: dot product (inner product)
    # For 2D: matrix multiplication
    if a.dim() == 1 and b.dim() == 1:
        # Vector dot product
        if len(a.shape()) != len(b.shape()) or a.shape()[0] != b.shape()[0]:
            raise Error("Incompatible shapes for dot product")

        var result_shape = List[Int]()  # Scalar (0D)
        var result = ExTensor(result_shape, a.dtype())

        # Compute dot product using dtype-specialized implementation
        var length = a.shape()[0]
        _dispatch_dot_1d(result, a, b, length)

        return result^
    else:
        # Delegate to matmul
        return matmul(a, b)


fn outer(a: ExTensor, b: ExTensor) raises -> ExTensor:
    """Outer product of two vectors.

    Args:
            a: First 1D tensor (vector).
            b: Second 1D tensor (vector).

    Returns:
            A 2D tensor containing the outer product

    Raises:
            Error: If tensors are not 1D.

    Examples:
        ```
            var a = ones(List[Int](), DType.float32)
            var b = ones(List[Int](), DType.float32)
            var c = outer(a, b)  # Shape (3, 4), all ones
        ```
    """
    # Check that inputs are 1D
    if a.dim() != 1 or b.dim() != 1:
        raise Error("outer requires 1D tensors")

    if a.dtype() != b.dtype():
        raise Error("Cannot compute outer product with different dtypes")

    # Output shape is (len(a), len(b))
    var result_shape = List[Int](capacity=2)
    result_shape.append(a.shape()[0])
    result_shape.append(b.shape()[0])

    var result = ExTensor(result_shape, a.dtype())

    # Implement outer product using dtype-specialized implementation
    var len_a = a.shape()[0]
    var len_b = b.shape()[0]

    _dispatch_outer(result, a, b, len_a, len_b)

    return result^


# ============================================================================
# Backward Pass (Gradient Computation)
# ============================================================================


fn matmul_backward(
    grad_output: ExTensor, a: ExTensor, b: ExTensor
) raises -> GradientPair:
    """Compute gradients for matrix multiplication.

        For C = A @ B, given ∂L/∂C, computes:
            ∂L/∂A = ∂L/∂C @ B^T
            ∂L/∂B = A^T @ ∂L/∂C

        Supports all matmul cases:
            - 2D @ 2D: Standard matrix multiplication
            - 2D @ 1D: Matrix-vector multiplication
            - 1D @ 2D: Vector-matrix multiplication
            - Batched: N-D tensors with batched matmul

    Args:
            grad_output: Gradient from upstream (∂L/∂C).
            a: First input from forward pass (A).
            b: Second input from forward pass (B).

    Returns:
            GradientPair containing (grad_a, grad_b) - gradients w.r.t. inputs

    Raises:
            Error: If tensor shapes are incompatible.

    Examples:
        ```
            # Forward pass
            var a = zeros([3, 4], DType.float32)
            var b = zeros([4, 5], DType.float32)
            var c = matmul(a, b)  # Shape (3, 5)

            # Backward pass
            var grad_c = ones([3, 5], DType.float32)
            var grads = matmul_backward(grad_c, a, b)
            var grad_a = grads.grad_a  # Shape (3, 4)
            var grad_b = grads.grad_b  # Shape (4, 5)
        ```

        Mathematical Derivation:
        ```
            For element-wise: C[i,j] = Σ_k A[i,k] * B[k,j]
            ∂L/∂A[i,k] = Σ_j (∂L/∂C[i,j] * B[k,j]) = (∂L/∂C @ B^T)[i,k]
            ∂L/∂B[k,j] = Σ_i (∂L/∂C[i,j] * A[i,k]) = (A^T @ ∂L/∂C)[k,j]
        ```
    """
    var a_shape = a.shape()
    var b_shape = b.shape()
    var a_ndim = len(a_shape)
    var b_ndim = len(b_shape)

    # Handle 2D @ 1D case
    if a_ndim == 2 and b_ndim == 1:
        # Forward: C (m,) = A (m, k) @ b (k,)
        # grad_a (m, k) = grad_output (m,) @ b^T (1, k) -> outer product
        # grad_b (k,) = A^T (k, m) @ grad_output (m,)

        # grad_a: Outer product of grad_output and b
        var grad_a_shape = List[Int](capacity=2)
        grad_a_shape.append(a_shape[0])  # m
        grad_a_shape.append(a_shape[1])  # k
        var grad_a = ExTensor(grad_a_shape, a.dtype())

        var m = a_shape[0]
        var k = a_shape[1]

        # grad_a[i, j] = grad_output[i] * b[j] (outer product)
        _dispatch_outer(grad_a, grad_output, b, m, k)

        # grad_b: A^T @ grad_output
        var a_t = transpose(a)  # Transpose A to get (k, m)
        var grad_b = matmul(a_t, grad_output)  # (k, m) @ (m,) -> (k,)

        return GradientPair(grad_a, grad_b)

    # Handle 1D @ 2D case
    if a_ndim == 1 and b_ndim == 2:
        # Forward: C (n,) = a (k,) @ B (k, n)
        # grad_a (k,) = B (k, n) @ grad_output (n,)
        # grad_b (k, n) = a^T (k, 1) @ grad_output^T (1, n) -> outer product

        # grad_a: B @ grad_output
        var grad_a = matmul(b, grad_output)  # (k, n) @ (n,) -> (k,)

        # grad_b: Outer product of a and grad_output
        var grad_b_shape = List[Int](capacity=2)
        grad_b_shape.append(b_shape[0])  # k
        grad_b_shape.append(b_shape[1])  # n
        var grad_b = ExTensor(grad_b_shape, b.dtype())

        var k = b_shape[0]
        var n = b_shape[1]

        # grad_b[i, j] = a[i] * grad_output[j] (outer product)
        _dispatch_outer(grad_b, a, grad_output, k, n)

        return GradientPair(grad_a, grad_b)

    # Handle 2D @ 2D and batched cases
    # For C = A @ B, the gradients are:
    # grad_a = grad_output @ B^T
    # grad_b = A^T @ grad_output
    var b_t = transpose(b)
    var a_t = transpose(a)

    var grad_a = matmul(grad_output, b_t)
    var grad_b = matmul(a_t, grad_output)

    return GradientPair(grad_a, grad_b)


fn transpose_backward(
    grad_output: ExTensor, axes: Optional[List[Int]] = None
) raises -> ExTensor:
    """Compute gradient for transpose operation.

        For Y = transpose(X, axes), given ∂L/∂Y, computes:
            ∂L/∂X = transpose(∂L/∂Y, inverse_axes)

        The gradient of transpose is the transpose with inverse permutation

    Args:
            grad_output: Gradient from upstream (∂L/∂Y).
            axes: The axes permutation used in forward pass. If None, uses default (reverse all).

    Returns:
            Gradient w.r.t. input (∂L/∂X)

    Raises:
            Error: If operation fails.

    Examples:
        ```
            # Default case (reverse axes)
            var x = zeros([3, 4], DType.float32)
            var y = transpose(x)  # Shape (4, 3)
            var grad_y = ones([4, 3], DType.float32)
            var grad_x = transpose_backward(grad_y)  # Shape (3, 4)

            # Custom axes case
            var x3d = zeros([2, 3, 4], DType.float32)
            var axes  = List[Int]()
            axes.append(2)
            axes.append(0)
            axes.append(1)
            var y3d = transpose(x3d, axes)  # Shape (4, 2, 3)
            var grad_y3d = ones([4, 2, 3], DType.float32)
            var grad_x3d = transpose_backward(grad_y3d, axes)  # Shape (2, 3, 4)
        ```

    Note:
            Transpose is self-adjoint: the inverse permutation is used to compute gradients
            For axes=[a1, a2, ..., an], the inverse is computed by finding the permutation
            that maps back to the original order.
    """
    var ndim = grad_output.dim()

    # Handle default case (None): reverse all axes
    var perm = axes
    if perm is None:
        perm = List[Int](capacity=ndim)
        for i in range(ndim - 1, -1, -1):
            perm.value().append(i)

    # Compute inverse permutation
    # If forward permutation is [a1, a2, ..., an], inverse satisfies:
    # inverse[perm[i]] = i for all i
    var inverse_perm = List[Int](capacity=ndim)
    for _ in range(ndim):
        inverse_perm.append(0)

    for i in range(ndim):
        inverse_perm[perm.value()[i]] = i

    # Apply inverse permutation to gradient
    return transpose(grad_output, inverse_perm^)
