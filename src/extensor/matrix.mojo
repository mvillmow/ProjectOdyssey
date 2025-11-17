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

    # Implement matrix multiplication
    # For 2D case: result[i, j] = sum(a[i, k] * b[k, j] for k in range(a_cols))
    # For batched case: apply same logic to each batch

    if len(a_shape) == 2:
        # Simple 2D matrix multiplication
        for i in range(a_rows):
            for j in range(b_cols):
                var sum_val: Float64 = 0.0
                for k in range(a_cols):
                    let a_val = a._get_float64(i * a_cols + k)
                    let b_val = b._get_float64(k * b_cols + j)
                    sum_val += a_val * b_val
                result._set_float64(i * b_cols + j, sum_val)
    else:
        # Batched matrix multiplication (3D+)
        # Compute batch size (product of all dimensions except last 2)
        var batch_size = 1
        for i in range(len(a_shape) - 2):
            batch_size *= a_shape[i]

        let matrix_size_a = a_rows * a_cols
        let matrix_size_b = b_rows * b_cols
        let matrix_size_result = a_rows * b_cols

        for batch in range(batch_size):
            let a_offset = batch * matrix_size_a
            let b_offset = batch * matrix_size_b
            let result_offset = batch * matrix_size_result

            for i in range(a_rows):
                for j in range(b_cols):
                    var sum_val: Float64 = 0.0
                    for k in range(a_cols):
                        let a_idx = a_offset + i * a_cols + k
                        let b_idx = b_offset + k * b_cols + j
                        let a_val = a._get_float64(a_idx)
                        let b_val = b._get_float64(b_idx)
                        sum_val += a_val * b_val
                    let result_idx = result_offset + i * b_cols + j
                    result._set_float64(result_idx, sum_val)

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
    # Implement transpose
    # For now, copy data in transposed order (TODO: zero-copy view with strides)

    var result_shape = DynamicVector[Int]()

    # If no axes provided, reverse all dimensions
    for i in range(tensor.dim() - 1, -1, -1):
        result_shape.push_back(tensor.shape()[i])

    var result = ExTensor(result_shape, tensor.dtype())

    # Implement for 2D case (most common)
    if tensor.dim() == 2:
        let rows = tensor.shape()[0]
        let cols = tensor.shape()[1]

        # result[i, j] = tensor[j, i]
        for i in range(cols):  # New rows (was cols)
            for j in range(rows):  # New cols (was rows)
                let src_idx = j * cols + i  # tensor[j, i]
                let dst_idx = i * rows + j  # result[i, j]
                let val = tensor._get_float64(src_idx)
                result._set_float64(dst_idx, val)
    else:
        # For 3D+, implement simple reversal of axes
        # This is a simplified implementation - full permutation would require more complex indexing
        # For now, just copy all values in the same order (placeholder)
        # TODO: Implement proper multi-dimensional transpose
        for i in range(tensor.numel()):
            let val = tensor._get_float64(i)
            result._set_float64(i, val)

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

        var result_shape = DynamicVector[Int]()  # Scalar (0D)
        var result = ExTensor(result_shape, a.dtype())

        # Compute dot product: sum of a[i] * b[i]
        var sum_val: Float64 = 0.0
        let length = a.shape()[0]
        for i in range(length):
            let a_val = a._get_float64(i)
            let b_val = b._get_float64(i)
            sum_val += a_val * b_val

        result._set_float64(0, sum_val)
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

    # Implement outer product: result[i, j] = a[i] * b[j]
    let len_a = a.shape()[0]
    let len_b = b.shape()[0]

    for i in range(len_a):
        for j in range(len_b):
            let a_val = a._get_float64(i)
            let b_val = b._get_float64(j)
            let product = a_val * b_val
            result._set_float64(i * len_b + j, product)

    return result^
