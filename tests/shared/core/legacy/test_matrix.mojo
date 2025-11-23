"""Tests for ExTensor matrix operations.

Tests matrix operations including matmul, transpose, dot, outer, inner,
and tensordot with various shapes and dimensions.
"""

from sys import DType

# Import ExTensor and operations
from shared.core import ExTensor, zeros, ones, full, arange, eye
from shared.core import matmul, transpose, dot, outer

# Import test helpers
from ..helpers.assertions import (
    assert_dtype,
    assert_numel,
    assert_dim,
    assert_value_at,
    assert_all_values,
    assert_all_close,
)


# ============================================================================
# Test matmul() - 2D matrix multiplication
# ============================================================================

fn test_matmul_2d_square() raises:
    """Test 2D matrix multiplication with square matrices."""
    var shape_3x3 = List[Int]()
    shape_3x3[0] = 3
    shape_3x3[1] = 3

    let a = eye(3, 3, DType.float32)  # 3x3 identity
    let b = full(shape_3x3, 2.0, DType.float32)  # 3x3 matrix of 2s
    let c = matmul(a, b)

    # Identity @ B = B, so result should be all 2s
    assert_dim(c, 2, "Result should be 2D")
    assert_numel(c, 9, "Result should be 3x3 (9 elements)")
    assert_all_values(c, 2.0, 1e-6, "Identity @ B should equal B")


fn test_matmul_2d_rectangular() raises:
    """Test 2D matrix multiplication with rectangular matrices."""
    var shape_a = List[Int]()
    shape_a[0] = 3
    shape_a[1] = 4
    var shape_b = List[Int]()
    shape_b[0] = 4
    shape_b[1] = 2

    let a = ones(shape_a, DType.float32)  # 3x4
    let b = full(shape_b, 2.0, DType.float32)  # 4x2
    let c = matmul(a, b)

    # Result should be 3x2, each element = 1*2 + 1*2 + 1*2 + 1*2 = 8
    assert_dim(c, 2, "Result should be 2D")
    assert_numel(c, 6, "Result should be 3x2 (6 elements)")
    assert_all_values(c, 8.0, 1e-6, "Each element should be 8")


fn test_matmul_2d_known_values() raises:
    """Test 2D matmul with known result."""
    # Create simple 2x2 matrices for easy verification
    # A = [[1, 2], [3, 4]]
    # B = [[2, 0], [1, 3]]
    # A @ B = [[4, 6], [10, 12]]

    # TODO: Create matrices with specific values
    # let a = ...
    # let b = ...
    # let c = matmul(a, b)

    # assert_value_at(c, 0, 4.0, 1e-6, "c[0,0] should be 4")
    # assert_value_at(c, 1, 6.0, 1e-6, "c[0,1] should be 6")
    # assert_value_at(c, 2, 10.0, 1e-6, "c[1,0] should be 10")
    # assert_value_at(c, 3, 12.0, 1e-6, "c[1,1] should be 12")
    pass  # Placeholder


# ============================================================================
# Test matmul() - Batched matrix multiplication
# ============================================================================

fn test_matmul_batched_3d() raises:
    """Test batched matrix multiplication (3D)."""
    var shape_a = List[Int]()
    shape_a[0] = 2  # batch size
    shape_a[1] = 3  # rows
    shape_a[2] = 4  # cols
    var shape_b = List[Int]()
    shape_b[0] = 2  # batch size
    shape_b[1] = 4  # rows
    shape_b[2] = 2  # cols

    let a = ones(shape_a, DType.float32)  # 2x3x4
    let b = full(shape_b, 0.5, DType.float32)  # 2x4x2
    let c = matmul(a, b)

    # Result should be 2x3x2 (batch_size x a_rows x b_cols)
    # Each element = 1*0.5 + 1*0.5 + 1*0.5 + 1*0.5 = 2
    assert_dim(c, 3, "Result should be 3D")
    assert_numel(c, 12, "Result should be 2x3x2 (12 elements)")
    assert_all_values(c, 2.0, 1e-6, "Each element should be 2")


fn test_matmul_batched_4d() raises:
    """Test batched matrix multiplication (4D)."""
    var shape_a = List[Int]()
    shape_a[0] = 2  # batch dim 1
    shape_a[1] = 3  # batch dim 2
    shape_a[2] = 4  # rows
    shape_a[3] = 5  # cols
    var shape_b = List[Int]()
    shape_b[0] = 2
    shape_b[1] = 3
    shape_b[2] = 5  # rows
    shape_b[3] = 2  # cols

    let a = ones(shape_a, DType.float32)  # 2x3x4x5
    let b = ones(shape_b, DType.float32)  # 2x3x5x2
    let c = matmul(a, b)

    # Result should be 2x3x4x2
    assert_dim(c, 4, "Result should be 4D")
    assert_numel(c, 48, "Result should be 2x3x4x2 (48 elements)")
    # TODO: Once matmul is implemented, uncomment:
    # Each element = 1*1 + ... (5 times) = 5
    # assert_all_values(c, 5.0, 1e-6, "Each element should be 5")


# ============================================================================
# Test matmul() - Error cases
# ============================================================================

fn test_matmul_incompatible_shapes() raises:
    """Test that incompatible shapes raise error."""
    var shape_a = List[Int]()
    shape_a[0] = 3
    shape_a[1] = 4
    var shape_b = List[Int]()
    shape_b[0] = 5  # Incompatible: 4 != 5
    shape_b[1] = 2

    let a = ones(shape_a, DType.float32)
    let b = ones(shape_b, DType.float32)

    # Verify error handling
    var error_raised = False
    try:
        let c = matmul(a, b)
    except:
        error_raised = True

    if not error_raised:
        raise Error("Should have raised error for incompatible matmul shapes (3,4) @ (5,2)")


fn test_matmul_dtype_mismatch() raises:
    """Test that dtype mismatch raises error."""
    var shape = List[Int]()
    shape[0] = 2
    shape[1] = 2
    let a = ones(shape, DType.float32)
    let b = ones(shape, DType.float64)  # Different dtype

    var error_raised = False
    try:
        let c = matmul(a, b)
    except:
        error_raised = True

    if not error_raised:
        raise Error("Should have raised error for dtype mismatch in matmul")


fn test_matmul_1d_error() raises:
    """Test that 1D inputs raise error."""
    var shape = List[Int]()
    shape[0] = 5
    let a = ones(shape, DType.float32)
    let b = ones(shape, DType.float32)

    var error_raised = False
    try:
        let c = matmul(a, b)  # matmul requires 2D+
    except:
        error_raised = True

    if not error_raised:
        raise Error("Should have raised error for 1D inputs to matmul")


fn test_matmul_with_zeros() raises:
    """Test matmul with zero matrices."""
    var shape = List[Int]()
    shape[0] = 3
    shape[1] = 3
    let a = zeros(shape, DType.float32)
    let b = ones(shape, DType.float32)
    let c = matmul(a, b)

    assert_dim(c, 2, "Result should be 2D")
    assert_numel(c, 9, "Result should be 3x3")
    assert_all_values(c, 0.0, 1e-6, "Zero matrix @ anything = zero matrix")


# ============================================================================
# Test transpose()
# ============================================================================

fn test_transpose_2d() raises:
    """Test transpose of 2D matrix."""
    var shape = List[Int]()
    shape[0] = 3
    shape[1] = 4
    let a = ones(shape, DType.float32)  # 3x4
    let b = transpose(a)

    # Result should be 4x3
    assert_dim(b, 2, "Transpose should be 2D")
    assert_numel(b, 12, "Transpose should have same number of elements")
    # TODO: Once transpose is implemented, verify shape is (4, 3):
    # assert_equal_int(b.shape()[0], 4, "First dim should be 4")
    # assert_equal_int(b.shape()[1], 3, "Second dim should be 3")


fn test_transpose_3d_default() raises:
    """Test transpose of 3D tensor (default permutation)."""
    var shape = List[Int]()
    shape[0] = 2
    shape[1] = 3
    shape[2] = 4
    let a = ones(shape, DType.float32)  # 2x3x4
    let b = transpose(a)

    # Default: reverse all axes -> 4x3x2
    assert_dim(b, 3, "Transpose should be 3D")
    assert_numel(b, 24, "Transpose should have same number of elements")
    # TODO: Once transpose is implemented, verify shape is (4, 3, 2):
    # assert_equal_int(b.shape()[0], 4, "First dim should be 4")
    # assert_equal_int(b.shape()[1], 3, "Second dim should be 3")
    # assert_equal_int(b.shape()[2], 2, "Third dim should be 2")


fn test_transpose_3d_permute() raises:
    """Test transpose with custom axis permutation."""
    var shape = List[Int]()
    shape[0] = 2
    shape[1] = 3
    shape[2] = 4
    let a = ones(shape, DType.float32)  # 2x3x4
    # let b = transpose(a, axes=(1, 0, 2))  # TODO: Implement axes parameter

    # Permute axes: (2,3,4) -> (3,2,4)
    # assert_numel(b, 24, "Transpose should have same number of elements")
    pass  # Placeholder


fn test_transpose_values_correctness() raises:
    """Test that transpose correctly moves values."""
    # Create matrix with known pattern
    # [[1, 2, 3],
    #  [4, 5, 6]]
    # Transpose should give:
    # [[1, 4],
    #  [2, 5],
    #  [3, 6]]

    # TODO: Create matrix with specific values and verify transpose
    pass  # Placeholder


fn test_transpose_identity() raises:
    """Test transpose of identity matrix."""
    let a = eye(4, 4, DType.float32)
    let b = transpose(a)

    # Transpose of identity should be identity
    assert_dim(b, 2, "Transpose of 2D should be 2D")
    assert_numel(b, 16, "Transpose should preserve elements")
    # TODO: Once transpose is implemented, verify values:
    # assert_all_values(b, ..., "Identity transpose = identity")


fn test_transpose_twice() raises:
    """Test that transpose(transpose(x)) == x."""
    var shape = List[Int]()
    shape[0] = 3
    shape[1] = 5
    let a = ones(shape, DType.float32)
    let b = transpose(a)
    let c = transpose(b)

    # Should return to original shape
    assert_dim(c, 2, "Double transpose should be 2D")
    assert_numel(c, 15, "Double transpose should preserve elements")
    # TODO: Once transpose is implemented, verify shape:
    # assert_equal_int(c.shape()[0], 3, "Should be back to 3")
    # assert_equal_int(c.shape()[1], 5, "Should be back to 5")


fn test_transpose_preserves_dtype() raises:
    """Test that transpose preserves dtype."""
    var shape = List[Int]()
    shape[0] = 2
    shape[1] = 3
    let a = ones(shape, DType.float64)
    let b = transpose(a)

    assert_dtype(b, DType.float64, "Transpose should preserve float64")


# ============================================================================
# Test dot()
# ============================================================================

fn test_dot_1d() raises:
    """Test dot product of two 1D vectors."""
    let a = arange(1.0, 6.0, 1.0, DType.float32)  # [1, 2, 3, 4, 5]
    let b = arange(1.0, 6.0, 1.0, DType.float32)  # [1, 2, 3, 4, 5]
    let c = dot(a, b)

    # Expected: 1*1 + 2*2 + 3*3 + 4*4 + 5*5 = 55
    assert_dim(c, 0, "Dot product should be scalar (0D)")
    assert_numel(c, 1, "Dot product should have 1 element")
    assert_value_at(c, 0, 55.0, 1e-4, "Dot product result")


fn test_dot_2d() raises:
    """Test dot product (equivalent to matmul for 2D)."""
    var shape = List[Int]()
    shape[0] = 2
    shape[1] = 3
    let a = ones(shape, DType.float32)  # 2x3
    var shape_b = List[Int]()
    shape_b[0] = 3
    shape_b[1] = 2
    let b = ones(shape_b, DType.float32)  # 3x2
    let c = dot(a, b)

    # Should behave like matmul for 2D
    assert_dim(c, 2, "Result should be 2D")
    assert_numel(c, 4, "Result should be 2x2 (4 elements)")
    assert_all_values(c, 3.0, 1e-6, "Each element should be 3")


fn test_dot_incompatible_shapes() raises:
    """Test that incompatible 1D shapes raise error."""
    var shape_a = List[Int]()
    shape_a[0] = 5
    var shape_b = List[Int]()
    shape_b[0] = 3  # Different size

    let a = ones(shape_a, DType.float32)
    let b = ones(shape_b, DType.float32)

    var error_raised = False
    try:
        let c = dot(a, b)
    except:
        error_raised = True

    if not error_raised:
        raise Error("Should have raised error for incompatible dot shapes (5,) and (3,)")


fn test_dot_dtype_mismatch() raises:
    """Test that dtype mismatch raises error."""
    var shape = List[Int]()
    shape[0] = 5
    let a = ones(shape, DType.float32)
    let b = ones(shape, DType.float64)

    var error_raised = False
    try:
        let c = dot(a, b)
    except:
        error_raised = True

    if not error_raised:
        raise Error("Should have raised error for dtype mismatch in dot")


fn test_dot_preserves_dtype() raises:
    """Test that dot preserves dtype."""
    var shape = List[Int]()
    shape[0] = 5
    let a = ones(shape, DType.float64)
    let b = ones(shape, DType.float64)
    let c = dot(a, b)

    assert_dtype(c, DType.float64, "dot should preserve float64")


# ============================================================================
# Test outer()
# ============================================================================

fn test_outer_vectors() raises:
    """Test outer product of two vectors."""
    let a = arange(1.0, 4.0, 1.0, DType.float32)  # [1, 2, 3]
    let b = arange(1.0, 3.0, 1.0, DType.float32)  # [1, 2]
    let c = outer(a, b)

    # Expected 3x2 matrix:
    # [[1, 2],
    #  [2, 4],
    #  [3, 6]]
    assert_dim(c, 2, "Outer product should be 2D")
    assert_numel(c, 6, "Outer product should be 3x2 (6 elements)")
    assert_value_at(c, 0, 1.0, 1e-6, "c[0,0] = 1*1 = 1")
    assert_value_at(c, 1, 2.0, 1e-6, "c[0,1] = 1*2 = 2")
    assert_value_at(c, 2, 2.0, 1e-6, "c[1,0] = 2*1 = 2")
    assert_value_at(c, 3, 4.0, 1e-6, "c[1,1] = 2*2 = 4")
    assert_value_at(c, 4, 3.0, 1e-6, "c[2,0] = 3*1 = 3")
    assert_value_at(c, 5, 6.0, 1e-6, "c[2,1] = 3*2 = 6")


fn test_outer_not_1d_error() raises:
    """Test that non-1D inputs raise error."""
    var shape_2d = List[Int]()
    shape_2d[0] = 2
    shape_2d[1] = 3
    var shape_1d = List[Int]()
    shape_1d[0] = 3

    let a = ones(shape_2d, DType.float32)  # 2D tensor
    let b = ones(shape_1d, DType.float32)  # 1D vector

    var error_raised = False
    try:
        let c = outer(a, b)  # Should error: outer requires 1D
    except:
        error_raised = True

    if not error_raised:
        raise Error("Should have raised error for non-1D input to outer")


fn test_outer_dtype_mismatch() raises:
    """Test that dtype mismatch raises error."""
    var shape = List[Int]()
    shape[0] = 3
    let a = ones(shape, DType.float32)
    let b = ones(shape, DType.float64)

    var error_raised = False
    try:
        let c = outer(a, b)
    except:
        error_raised = True

    if not error_raised:
        raise Error("Should have raised error for dtype mismatch in outer")


fn test_outer_with_zeros() raises:
    """Test outer product with zero vector."""
    var shape_a = List[Int]()
    shape_a[0] = 3
    var shape_b = List[Int]()
    shape_b[0] = 4

    let a = zeros(shape_a, DType.float32)
    let b = ones(shape_b, DType.float32)
    let c = outer(a, b)

    assert_dim(c, 2, "Outer product should be 2D")
    assert_numel(c, 12, "Outer product should be 3x4 (12 elements)")
    assert_all_values(c, 0.0, 1e-6, "Outer with zero vector should be all zeros")


fn test_outer_preserves_dtype() raises:
    """Test that outer preserves dtype."""
    var shape_a = List[Int]()
    shape_a[0] = 2
    var shape_b = List[Int]()
    shape_b[0] = 3

    let a = ones(shape_a, DType.float64)
    let b = ones(shape_b, DType.float64)
    let c = outer(a, b)

    assert_dtype(c, DType.float64, "outer should preserve float64")


# ============================================================================
# Test inner()
# ============================================================================

fn test_inner_1d() raises:
    """Test inner product of 1D vectors (equivalent to dot)."""
    let a = arange(1.0, 4.0, 1.0, DType.float32)  # [1, 2, 3]
    let b = arange(1.0, 4.0, 1.0, DType.float32)  # [1, 2, 3]
    # let c = inner(a, b)  # TODO: Implement inner()

    # Expected: 1*1 + 2*2 + 3*3 = 14
    # assert_value_at(c, 0, 14.0, 1e-6, "Inner product result")
    pass  # Placeholder


fn test_inner_2d() raises:
    """Test inner product with 2D arrays."""
    var shape = List[Int]()
    shape[0] = 2
    shape[1] = 3
    let a = ones(shape, DType.float32)  # 2x3
    let b = ones(shape, DType.float32)  # 2x3
    # let c = inner(a, b)  # TODO: Implement inner()

    # Result should be 2x2 (inner product along last axis)
    # assert_numel(c, 4, "Result should be 2x2")
    pass  # Placeholder


# ============================================================================
# Test tensordot()
# ============================================================================

fn test_tensordot_basic() raises:
    """Test tensordot with simple contraction."""
    var shape_a = List[Int]()
    shape_a[0] = 3
    shape_a[1] = 4
    var shape_b = List[Int]()
    shape_b[0] = 4
    shape_b[1] = 2

    let a = ones(shape_a, DType.float32)  # 3x4
    let b = ones(shape_b, DType.float32)  # 4x2
    # let c = tensordot(a, b, axes=1)  # TODO: Implement tensordot()

    # Contraction over one axis should give 3x2 result
    # assert_numel(c, 6, "Result should be 3x2 (6 elements)")
    pass  # Placeholder


fn test_tensordot_multiple_axes() raises:
    """Test tensordot with multiple axes contraction."""
    var shape_a = List[Int]()
    shape_a[0] = 2
    shape_a[1] = 3
    shape_a[2] = 4
    var shape_b = List[Int]()
    shape_b[0] = 3
    shape_b[1] = 4
    shape_b[2] = 5

    let a = ones(shape_a, DType.float32)  # 2x3x4
    let b = ones(shape_b, DType.float32)  # 3x4x5
    # let c = tensordot(a, b, axes=((1,2), (0,1)))  # TODO: Implement

    # Contract over (3,4) dims -> result should be 2x5
    # assert_numel(c, 10, "Result should be 2x5 (10 elements)")
    pass  # Placeholder


# ============================================================================
# Test dtype preservation
# ============================================================================

fn test_matmul_preserves_dtype() raises:
    """Test that matmul preserves dtype."""
    var shape = List[Int]()
    shape[0] = 3
    shape[1] = 3
    let a = ones(shape, DType.float64)
    let b = ones(shape, DType.float64)
    let c = matmul(a, b)

    assert_dtype(c, DType.float64, "matmul should preserve float64")


# ============================================================================
# Test dunder methods
# ============================================================================

fn test_dunder_matmul() raises:
    """Test __matmul__ operator overloading (a @ b)."""
    var shape = List[Int]()
    shape[0] = 2
    shape[1] = 2
    let a = ones(shape, DType.float32)
    let b = full(shape, 2.0, DType.float32)
    let c = a @ b

    # Each element should be 1*2 + 1*2 = 4
    assert_dim(c, 2, "Result should be 2D")
    assert_numel(c, 4, "Result should be 2x2 (4 elements)")
    assert_all_values(c, 4.0, 1e-6, "a @ b should work via __matmul__")


# ============================================================================
# Test Matrix @ Vector operations (Linear layers)
# ============================================================================

fn test_matmul_matrix_vector() raises:
    """Test matrix @ vector multiplication (essential for linear layers)."""
    var shape_w = List[Int]()
    shape_w[0] = 3  # out_features
    shape_w[1] = 4  # in_features
    var shape_x = List[Int]()
    shape_x[0] = 4  # in_features

    let w = ones(shape_w, DType.float32)  # 3x4 weight matrix
    let x = full(shape_x, 2.0, DType.float32)  # 4D input vector
    let y = matmul(w, x)  # Should give 3D output

    # Result: each element = 1*2 + 1*2 + 1*2 + 1*2 = 8
    assert_dim(y, 1, "Result should be 1D vector")
    assert_numel(y, 3, "Result should have 3 elements (out_features)")
    assert_value_at(y, 0, 8.0, 1e-6, "y[0] should be 8.0")
    assert_value_at(y, 1, 8.0, 1e-6, "y[1] should be 8.0")
    assert_value_at(y, 2, 8.0, 1e-6, "y[2] should be 8.0")


fn test_matmul_vector_matrix() raises:
    """Test vector @ matrix multiplication."""
    var shape_x = List[Int]()
    shape_x[0] = 3
    var shape_w = List[Int]()
    shape_w[0] = 3
    shape_w[1] = 4

    let x = full(shape_x, 2.0, DType.float32)  # 3D vector
    let w = ones(shape_w, DType.float32)  # 3x4 matrix
    let y = matmul(x, w)  # Should give 4D output

    # Result: each element = 2*1 + 2*1 + 2*1 = 6
    assert_dim(y, 1, "Result should be 1D vector")
    assert_numel(y, 4, "Result should have 4 elements")
    assert_all_values(y, 6.0, 1e-6, "All elements should be 6.0")


fn test_matmul_linear_layer_pattern() raises:
    """Test typical linear layer pattern: weight @ input."""
    # Simulate: Linear(in=5, out=10) processing single input
    var shape_w = List[Int]()
    shape_w[0] = 10  # out_features
    shape_w[1] = 5   # in_features
    var shape_x = List[Int]()
    shape_x[0] = 5   # in_features

    let weight = full(shape_w, 0.5, DType.float32)
    let input = ones(shape_x, DType.float32)
    let output = matmul(weight, input)

    # Each output element = 0.5 * 1 + ... (5 times) = 2.5
    assert_dim(output, 1, "Output should be 1D")
    assert_numel(output, 10, "Output should have 10 elements")
    assert_all_values(output, 2.5, 1e-6, "Linear output computation")


fn test_matmul_matrix_vector_error() raises:
    """Test matrix @ vector dimension mismatch error."""
    var shape_w = List[Int]()
    shape_w[0] = 3
    shape_w[1] = 4
    var shape_x = List[Int]()
    shape_x[0] = 5  # Wrong size!

    let w = ones(shape_w, DType.float32)
    let x = ones(shape_x, DType.float32)

    var error_raised = False
    try:
        let y = matmul(w, x)
    except:
        error_raised = True

    if not error_raised:
        raise Error("Should have raised error for dimension mismatch (3,4) @ (5,)")


# ============================================================================
# Test Transpose Combinations (BLAS patterns)
# ============================================================================

fn test_transpose_combination_at_b() raises:
    """Test A.T @ B (common in backprop: weight.T @ gradient)."""
    var shape_a = List[Int]()
    shape_a[0] = 3
    shape_a[1] = 4
    var shape_b = List[Int]()
    shape_b[0] = 3
    shape_b[1] = 2

    let a = ones(shape_a, DType.float32)  # 3x4
    let b = full(shape_b, 2.0, DType.float32)  # 3x2
    let a_t = transpose(a)  # 4x3
    let c = matmul(a_t, b)  # 4x3 @ 3x2 -> 4x2

    # Each element = 1*2 + 1*2 + 1*2 = 6
    assert_dim(c, 2, "Result should be 2D")
    assert_numel(c, 8, "Result should be 4x2 (8 elements)")
    assert_all_values(c, 6.0, 1e-6, "A.T @ B computation")


fn test_transpose_combination_a_bt() raises:
    """Test A @ B.T (common in attention: Q @ K.T)."""
    var shape_a = List[Int]()
    shape_a[0] = 2
    shape_a[1] = 3
    var shape_b = List[Int]()
    shape_b[0] = 4
    shape_b[1] = 3

    let a = full(shape_a, 2.0, DType.float32)  # 2x3
    let b = ones(shape_b, DType.float32)  # 4x3
    let b_t = transpose(b)  # 3x4
    let c = matmul(a, b_t)  # 2x3 @ 3x4 -> 2x4

    # Each element = 2*1 + 2*1 + 2*1 = 6
    assert_dim(c, 2, "Result should be 2D")
    assert_numel(c, 8, "Result should be 2x4 (8 elements)")
    assert_all_values(c, 6.0, 1e-6, "A @ B.T computation")


fn test_transpose_combination_at_bt() raises:
    """Test A.T @ B.T (double transpose pattern)."""
    var shape_a = List[Int]()
    shape_a[0] = 4
    shape_a[1] = 3
    var shape_b = List[Int]()
    shape_b[0] = 5
    shape_b[1] = 4

    let a = ones(shape_a, DType.float32)  # 4x3
    let b = full(shape_b, 2.0, DType.float32)  # 5x4
    let a_t = transpose(a)  # 3x4
    let b_t = transpose(b)  # 4x5
    let c = matmul(a_t, b_t)  # 3x4 @ 4x5 -> 3x5

    # Each element = 1*2 + 1*2 + 1*2 + 1*2 = 8
    assert_dim(c, 2, "Result should be 2D")
    assert_numel(c, 15, "Result should be 3x5 (15 elements)")
    assert_all_values(c, 8.0, 1e-6, "A.T @ B.T computation")


# ============================================================================
# Test Matrix Shape Variants (thin, wide, square)
# ============================================================================

fn test_matmul_thin_matrices() raises:
    """Test thin matrices (many rows, few columns)."""
    var shape_a = List[Int]()
    shape_a[0] = 100  # Many rows
    shape_a[1] = 5    # Few columns
    var shape_b = List[Int]()
    shape_b[0] = 5
    shape_b[1] = 20

    let a = ones(shape_a, DType.float32)  # 100x5
    let b = ones(shape_b, DType.float32)  # 5x20
    let c = matmul(a, b)  # 100x20

    # Each element = 1*1 + ... (5 times) = 5
    assert_dim(c, 2, "Result should be 2D")
    assert_numel(c, 2000, "Result should be 100x20 (2000 elements)")
    assert_value_at(c, 0, 5.0, 1e-6, "Thin matrix multiplication")
    assert_value_at(c, 1999, 5.0, 1e-6, "Check last element")


fn test_matmul_wide_matrices() raises:
    """Test wide matrices (few rows, many columns)."""
    var shape_a = List[Int]()
    shape_a[0] = 5     # Few rows
    shape_a[1] = 100   # Many columns
    var shape_b = List[Int]()
    shape_b[0] = 100
    shape_b[1] = 20

    let a = full(shape_a, 0.5, DType.float32)  # 5x100
    let b = ones(shape_b, DType.float32)  # 100x20
    let c = matmul(a, b)  # 5x20

    # Each element = 0.5*1 + ... (100 times) = 50
    assert_dim(c, 2, "Result should be 2D")
    assert_numel(c, 100, "Result should be 5x20 (100 elements)")
    assert_value_at(c, 0, 50.0, 1e-5, "Wide matrix multiplication")


fn test_matmul_tiny_matrices() raises:
    """Test very small matrices (1x1, 2x1, 1x2)."""
    # Test 1x1 @ 1x1
    var shape_1x1 = List[Int]()
    shape_1x1[0] = 1
    shape_1x1[1] = 1

    let a = full(shape_1x1, 3.0, DType.float32)
    let b = full(shape_1x1, 4.0, DType.float32)
    let c = matmul(a, b)

    assert_dim(c, 2, "Result should be 2D")
    assert_numel(c, 1, "Result should be 1x1 (1 element)")
    assert_value_at(c, 0, 12.0, 1e-6, "1x1 @ 1x1 = 3*4")


fn test_matmul_large_square() raises:
    """Test larger square matrix (stress test)."""
    var shape = List[Int]()
    shape[0] = 50
    shape[1] = 50

    let a = ones(shape, DType.float32)  # 50x50
    let b = ones(shape, DType.float32)  # 50x50
    let c = matmul(a, b)  # 50x50

    # Each element = 1*1 + ... (50 times) = 50
    assert_dim(c, 2, "Result should be 2D")
    assert_numel(c, 2500, "Result should be 50x50 (2500 elements)")
    assert_value_at(c, 0, 50.0, 1e-5, "Large square matrix")
    assert_value_at(c, 2499, 50.0, 1e-5, "Check last element")


# ============================================================================
# Test 3D+ Transpose (verify fix)
# ============================================================================

fn test_transpose_3d_correctness() raises:
    """Test that 3D transpose actually transposes correctly (not just copies)."""
    var shape = List[Int]()
    shape[0] = 2
    shape[1] = 3
    shape[2] = 4
    # Create a 2x3x4 tensor with distinct values
    let a = arange(0.0, 24.0, 1.0, DType.float32)  # This creates 1D

    # We need a helper to create 3D tensor - for now, verify shape is correct
    # TODO: Once we have proper 3D tensor creation, add value verification

    # For now, just test shape transformation
    var test_shape = List[Int]()
    test_shape[0] = 2
    test_shape[1] = 3
    test_shape[2] = 4
    let t = ones(test_shape, DType.float32)
    let t_T = transpose(t)

    # Verify shape is reversed
    assert_dim(t_T, 3, "Transpose should be 3D")
    # Shape should be (4, 3, 2)
    # TODO: Add shape assertions once we have a way to access shape elements


# ============================================================================
# Main test runner
# ============================================================================

fn main() raises:
    """Run all matrix operation tests."""
    print("Running ExTensor matrix operation tests...")

    # matmul() tests - 2D
    print("  Testing matmul() 2D...")
    test_matmul_2d_square()
    test_matmul_2d_rectangular()
    test_matmul_2d_known_values()

    # matmul() tests - batched
    print("  Testing matmul() batched...")
    test_matmul_batched_3d()
    test_matmul_batched_4d()

    # matmul() error cases
    print("  Testing matmul() error handling...")
    test_matmul_incompatible_shapes()
    test_matmul_dtype_mismatch()
    test_matmul_1d_error()
    test_matmul_with_zeros()

    # transpose() tests
    print("  Testing transpose()...")
    test_transpose_2d()
    test_transpose_3d_default()
    test_transpose_3d_permute()
    test_transpose_values_correctness()
    test_transpose_identity()
    test_transpose_twice()
    test_transpose_preserves_dtype()

    # dot() tests
    print("  Testing dot()...")
    test_dot_1d()
    test_dot_2d()
    test_dot_incompatible_shapes()
    test_dot_dtype_mismatch()
    test_dot_preserves_dtype()

    # outer() tests
    print("  Testing outer()...")
    test_outer_vectors()
    test_outer_not_1d_error()
    test_outer_dtype_mismatch()
    test_outer_with_zeros()
    test_outer_preserves_dtype()

    # inner() tests (placeholders for Priority 6)
    print("  Testing inner()...")
    test_inner_1d()
    test_inner_2d()

    # tensordot() tests (placeholders for Priority 6)
    print("  Testing tensordot()...")
    test_tensordot_basic()
    test_tensordot_multiple_axes()

    # Dtype preservation
    print("  Testing dtype preservation...")
    test_matmul_preserves_dtype()

    # Dunder methods
    print("  Testing matrix dunders...")
    test_dunder_matmul()

    # Matrix @ vector operations (NEW)
    print("  Testing matrix @ vector operations...")
    test_matmul_matrix_vector()
    test_matmul_vector_matrix()
    test_matmul_linear_layer_pattern()
    test_matmul_matrix_vector_error()

    # Transpose combinations (NEW)
    print("  Testing transpose combinations...")
    test_transpose_combination_at_b()
    test_transpose_combination_a_bt()
    test_transpose_combination_at_bt()

    # Matrix shape variants (NEW)
    print("  Testing matrix shape variants...")
    test_matmul_thin_matrices()
    test_matmul_wide_matrices()
    test_matmul_tiny_matrices()
    test_matmul_large_square()

    # 3D+ transpose correctness (NEW)
    print("  Testing 3D+ transpose...")
    test_transpose_3d_correctness()

    print("All matrix operation tests completed!")
