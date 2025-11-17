"""Matrix operations for ExTensor.

Implements linear algebra operations like matrix multiplication and transpose.
"""

from .extensor import ExTensor


fn matmul(a: ExTensor, b: ExTensor) raises -> ExTensor:
    """Matrix multiplication.

    Args:
        a: First tensor (matrix)
        b: Second tensor (matrix)

    Returns:
        A new tensor containing the matrix product a @ b

    Raises:
        Error if dimensions are incompatible

    Requirements:
        - 2D tensors: a.shape = (m, k), b.shape = (k, n) -> result.shape = (m, n)
        - ND tensors: batched matrix multiplication

    Examples:
        var a = zeros(DynamicVector[Int](3, 4), DType.float32)
        var b = zeros(DynamicVector[Int](4, 5), DType.float32)
        var c = matmul(a, b)  # Shape (3, 5)
    """
    # Check dtype compatibility
    if a.dtype() != b.dtype():
        raise Error("Cannot multiply matrices with different dtypes")

    # Check dimension compatibility
    let a_shape = a.shape()
    let b_shape = b.shape()

    if len(a_shape) < 2 or len(b_shape) < 2:
        raise Error("matmul requires at least 2D tensors")

    let a_rows = a_shape[len(a_shape) - 2]
    let a_cols = a_shape[len(a_shape) - 1]
    let b_rows = b_shape[len(b_shape) - 2]
    let b_cols = b_shape[len(b_shape) - 1]

    if a_cols != b_rows:
        raise Error(
            "Incompatible dimensions for matmul: " + str(a_cols) + " != " + str(b_rows)
        )

    # Compute output shape
    var result_shape = DynamicVector[Int]()

    # Copy batch dimensions (if any)
    for i in range(len(a_shape) - 2):
        result_shape.push_back(a_shape[i])

    # Add matrix dimensions
    result_shape.push_back(a_rows)
    result_shape.push_back(b_cols)

    # Create result
    var result = ExTensor(result_shape, a.dtype())

    # TODO: Implement actual matrix multiplication
    # For now, just fill with zeros
    result._fill_zero()

    return result^


fn transpose(tensor: ExTensor, axes: DynamicVector[Int] | None = None) raises -> ExTensor:
    """Transpose tensor dimensions.

    Args:
        tensor: Input tensor
        axes: Permutation of dimensions (default: reverse all axes)

    Returns:
        A new tensor (view) with transposed dimensions

    Examples:
        var t = zeros(DynamicVector[Int](3, 4, 5), DType.float32)
        var t_T = transpose(t)  # Shape (5, 4, 3) - reverse all axes

        var axes = DynamicVector[Int](2, 0, 1)
        var t_perm = transpose(t, axes)  # Shape (5, 3, 4) - custom permutation
    """
    # TODO: Implement transpose
    # Should create a view (zero-copy) by rearranging strides

    var result_shape = DynamicVector[Int]()

    # If no axes provided, reverse all dimensions
    for i in range(tensor.dim() - 1, -1, -1):
        result_shape.push_back(tensor.shape()[i])

    var result = ExTensor(result_shape, tensor.dtype())
    result._fill_zero()

    return result^


fn dot(a: ExTensor, b: ExTensor) raises -> ExTensor:
    """Dot product of tensors.

    Args:
        a: First tensor
        b: Second tensor

    Returns:
        Dot product (scalar for 1D, matrix product for 2D)

    Examples:
        var a = ones(DynamicVector[Int](5), DType.float32)
        var b = ones(DynamicVector[Int](5), DType.float32)
        var c = dot(a, b)  # Scalar 5.0
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

        var result_shape = DynamicVector[Int]()  # Scalar
        var result = ExTensor(result_shape, a.dtype())
        result._fill_zero()
        return result^
    else:
        # Delegate to matmul
        return matmul(a, b)


fn outer(a: ExTensor, b: ExTensor) raises -> ExTensor:
    """Outer product of two vectors.

    Args:
        a: First 1D tensor (vector)
        b: Second 1D tensor (vector)

    Returns:
        A 2D tensor containing the outer product

    Examples:
        var a = ones(DynamicVector[Int](3), DType.float32)
        var b = ones(DynamicVector[Int](4), DType.float32)
        var c = outer(a, b)  # Shape (3, 4), all ones
    """
    # Check that inputs are 1D
    if a.dim() != 1 or b.dim() != 1:
        raise Error("outer requires 1D tensors")

    if a.dtype() != b.dtype():
        raise Error("Cannot compute outer product with different dtypes")

    # Output shape is (len(a), len(b))
    var result_shape = DynamicVector[Int](2)
    result_shape[0] = a.shape()[0]
    result_shape[1] = b.shape()[0]

    var result = ExTensor(result_shape, a.dtype())

    # TODO: Implement outer product: result[i, j] = a[i] * b[j]
    result._fill_zero()

    return result^
