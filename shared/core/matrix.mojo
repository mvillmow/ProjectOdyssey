"""Matrix operations for ExTensor.

Implements linear algebra operations like matrix multiplication and transpose.
"""

from collections.vector import DynamicVector
from shared.core.extensor import ExTensor
from shared.core.gradient_types import GradientPair


fn matmul(a: ExTensor, b: ExTensor) raises -> ExTensor:
    """Matrix multiplication.

    Args:
        a: First tensor (matrix or vector)
        b: Second tensor (matrix or vector)

    Returns:
        A new tensor containing the matrix product a @ b

    Raises:
        Error if dimensions are incompatible

    Requirements:
        - 2D @ 2D: a.shape = (m, k), b.shape = (k, n) -> result.shape = (m, n)
        - 2D @ 1D: a.shape = (m, k), b.shape = (k,) -> result.shape = (m,)
        - 1D @ 2D: a.shape = (k,), b.shape = (k, n) -> result.shape = (n,)
        - ND tensors: batched matrix multiplication

    Examples:
        var a = zeros(DynamicVector[Int](3, 4), DType.float32)
        var b = zeros(DynamicVector[Int](4, 5), DType.float32)
        var c = matmul(a, b)  # Shape (3, 5)

        var W = zeros(DynamicVector[Int](10, 5), DType.float32)
        var x = zeros(DynamicVector[Int](5), DType.float32)
        var y = matmul(W, x)  # Shape (10,) - matrix @ vector
    """
    # Check dtype compatibility
    if a.dtype() != b.dtype():
        raise Error("Cannot multiply matrices with different dtypes")

    # Check dimension compatibility
    let a_shape = a.shape()
    let b_shape = b.shape()
    let a_ndim = len(a_shape)
    let b_ndim = len(b_shape)

    # Handle matrix @ vector (2D @ 1D)
    if a_ndim == 2 and b_ndim == 1:
        let m = a_shape[0]
        let k = a_shape[1]
        let n = b_shape[0]

        if k != n:
            raise Error("Incompatible dimensions for matmul: matrix (" + str(m) + ", " + str(k) + ") @ vector (" + str(n) + ")")

        # Result is a vector of shape (m,)
        var result_shape = DynamicVector[Int](1)
        result_shape[0] = m
        var result = ExTensor(result_shape, a.dtype())

        # Compute: result[i] = sum(a[i, j] * b[j] for j in range(k))
        for i in range(m):
            var sum_val: Float64 = 0.0
            for j in range(k):
                let a_val = a._get_float64(i * k + j)
                let b_val = b._get_float64(j)
                sum_val += a_val * b_val
            result._set_float64(i, sum_val)

        return result^

    # Handle vector @ matrix (1D @ 2D)
    if a_ndim == 1 and b_ndim == 2:
        let m = a_shape[0]
        let k = b_shape[0]
        let n = b_shape[1]

        if m != k:
            raise Error("Incompatible dimensions for matmul: vector (" + str(m) + ") @ matrix (" + str(k) + ", " + str(n) + ")")

        # Result is a vector of shape (n,)
        var result_shape = DynamicVector[Int](1)
        result_shape[0] = n
        var result = ExTensor(result_shape, a.dtype())

        # Compute: result[j] = sum(a[i] * b[i, j] for i in range(m))
        for j in range(n):
            var sum_val: Float64 = 0.0
            for i in range(m):
                let a_val = a._get_float64(i)
                let b_val = b._get_float64(i * n + j)
                sum_val += a_val * b_val
            result._set_float64(j, sum_val)

        return result^

    # For 2D and higher, require at least 2D tensors
    if a_ndim < 2 or b_ndim < 2:
        raise Error("matmul requires at least 2D tensors for non-vector inputs (use dot() for 1D @ 1D)")

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


fn transpose(tensor: ExTensor) raises -> ExTensor:
    """Transpose tensor dimensions.

    Args:
        tensor: Input tensor

    Returns:
        A new tensor with transposed dimensions (reverses all axes)

    Examples:
        var t = zeros(DynamicVector[Int](3, 4), DType.float32)
        var t_T = transpose(t)  # Shape (4, 3)

        var t3d = zeros(DynamicVector[Int](2, 3, 4), DType.float32)
        var t3d_T = transpose(t3d)  # Shape (4, 3, 2) - reverse all axes

    Note:
        Currently supports reversing all axes for any dimensionality.
        TODO: Add support for custom axis permutation via axes parameter.
    """
    let ndim = tensor.dim()
    let input_shape = tensor.shape()

    # Build result shape (reverse all dimensions)
    var result_shape = DynamicVector[Int]()
    for i in range(ndim - 1, -1, -1):
        result_shape.push_back(input_shape[i])

    var result = ExTensor(result_shape, tensor.dtype())

    # Compute strides for input tensor (row-major order)
    var input_strides = DynamicVector[Int](ndim)
    var stride = 1
    for i in range(ndim - 1, -1, -1):
        input_strides[i] = stride
        stride *= input_shape[i]

    # For each element in result, map to input position
    for result_idx in range(result.numel()):
        # Convert linear result index to coordinates
        var result_coords = DynamicVector[Int](ndim)
        var temp_idx = result_idx
        for i in range(ndim - 1, -1, -1):
            result_coords[i] = temp_idx % result.shape()[i]
            temp_idx //= result.shape()[i]

        # Map result coordinates to input coordinates (reverse order)
        var input_coords = DynamicVector[Int](ndim)
        for i in range(ndim):
            input_coords[i] = result_coords[ndim - 1 - i]

        # Convert input coordinates to linear index
        var input_idx = 0
        for i in range(ndim):
            input_idx += input_coords[i] * input_strides[i]

        # Copy value
        let val = tensor._get_float64(input_idx)
        result._set_float64(result_idx, val)

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


# ============================================================================
# Backward Pass (Gradient Computation)
# ============================================================================


fn matmul_backward(grad_output: ExTensor, a: ExTensor, b: ExTensor) raises -> GradientPair:
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
        grad_output: Gradient from upstream (∂L/∂C)
        a: First input from forward pass (A)
        b: Second input from forward pass (B)

    Returns:
        GradientPair containing (grad_a, grad_b) - gradients w.r.t. inputs

    Examples:
        # Forward pass
        var a = zeros(DynamicVector[Int](3, 4), DType.float32)
        var b = zeros(DynamicVector[Int](4, 5), DType.float32)
        var c = matmul(a, b)  # Shape (3, 5)

        # Backward pass
        var grad_c = ones(DynamicVector[Int](3, 5), DType.float32)
        var grads = matmul_backward(grad_c, a, b)
        var grad_a = grads.grad_a  # Shape (3, 4)
        var grad_b = grads.grad_b  # Shape (4, 5)

    Mathematical Derivation:
        For element-wise: C[i,j] = Σ_k A[i,k] * B[k,j]
        ∂L/∂A[i,k] = Σ_j (∂L/∂C[i,j] * B[k,j]) = (∂L/∂C @ B^T)[i,k]
        ∂L/∂B[k,j] = Σ_i (∂L/∂A[i,k] * A[i,k]) = (A^T @ ∂L/∂C)[k,j]
    """
    let a_shape = a.shape()
    let b_shape = b.shape()
    let a_ndim = len(a_shape)
    let b_ndim = len(b_shape)

    # Handle 2D @ 1D case
    if a_ndim == 2 and b_ndim == 1:
        # Forward: C (m,) = A (m, k) @ b (k,)
        # grad_a (m, k) = grad_output (m,) @ b^T (1, k) -> outer product
        # grad_b (k,) = A^T (k, m) @ grad_output (m,)

        # grad_a: Outer product of grad_output and b
        var grad_a_shape = DynamicVector[Int](2)
        grad_a_shape[0] = a_shape[0]  # m
        grad_a_shape[1] = a_shape[1]  # k
        var grad_a = ExTensor(grad_a_shape, a.dtype())

        let m = a_shape[0]
        let k = a_shape[1]

        # grad_a[i, j] = grad_output[i] * b[j]
        for i in range(m):
            for j in range(k):
                let grad_val = grad_output._get_float64(i)
                let b_val = b._get_float64(j)
                grad_a._set_float64(i * k + j, grad_val * b_val)

        # grad_b: A^T @ grad_output
        var b_t = transpose(a)  # Transpose A to get (k, m)
        var grad_b = matmul(b_t, grad_output)  # (k, m) @ (m,) -> (k,)

        return GradientPair(grad_a, grad_b)

    # Handle 1D @ 2D case
    if a_ndim == 1 and b_ndim == 2:
        # Forward: C (n,) = a (k,) @ B (k, n)
        # grad_a (k,) = B (k, n) @ grad_output (n,)
        # grad_b (k, n) = a^T (k, 1) @ grad_output^T (1, n) -> outer product

        # grad_a: B @ grad_output
        var grad_a = matmul(b, grad_output)  # (k, n) @ (n,) -> (k,)

        # grad_b: Outer product of a and grad_output
        var grad_b_shape = DynamicVector[Int](2)
        grad_b_shape[0] = b_shape[0]  # k
        grad_b_shape[1] = b_shape[1]  # n
        var grad_b = ExTensor(grad_b_shape, b.dtype())

        let k = b_shape[0]
        let n = b_shape[1]

        # grad_b[i, j] = a[i] * grad_output[j]
        for i in range(k):
            for j in range(n):
                let a_val = a._get_float64(i)
                let grad_val = grad_output._get_float64(j)
                grad_b._set_float64(i * n + j, a_val * grad_val)

        return GradientPair(grad_a, grad_b)

    # Handle 2D @ 2D and batched cases
    # Standard: grad_a = grad_output @ B^T, grad_b = A^T @ grad_output
    var b_t = transpose(b)
    var a_t = transpose(a)

    var grad_a = matmul(grad_output, b_t)
    var grad_b = matmul(a_t, grad_output)

    return GradientPair(grad_a, grad_b)


fn transpose_backward(grad_output: ExTensor) raises -> ExTensor:
    """Compute gradient for transpose operation.

    For Y = transpose(X), given ∂L/∂Y, computes:
        ∂L/∂X = transpose(∂L/∂Y)

    The gradient of transpose is simply transposing the gradient back.

    Args:
        grad_output: Gradient from upstream (∂L/∂Y)

    Returns:
        Gradient w.r.t. input (∂L/∂X)

    Examples:
        var x = zeros(DynamicVector[Int](3, 4), DType.float32)
        var y = transpose(x)  # Shape (4, 3)
        var grad_y = ones(DynamicVector[Int](4, 3), DType.float32)
        var grad_x = transpose_backward(grad_y)  # Shape (3, 4)
    """
    # Transpose is self-inverse for gradients
    return transpose(grad_output)
