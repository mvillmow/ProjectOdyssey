"""Tests for ExTensor broadcasting operations.

Tests NumPy-style broadcasting rules for different shaped tensors,
including scalar broadcasting, vector-to-matrix, and complex multi-dimensional cases.
"""

from sys import DType

# Import ExTensor and operations
from shared.core import ExTensor, zeros, ones, full, add, multiply

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
# Test scalar broadcasting
# ============================================================================

fn test_broadcast_scalar_to_1d() raises:
    """Test broadcasting scalar to 1D tensor."""
    var shape_vec = List[Int](1)
    shape_vec[0] = 5
    var shape_scalar = List[Int](0)

    let a = full(shape_vec, 3.0, DType.float32)  # [3, 3, 3, 3, 3]
    let b = full(shape_scalar, 2.0, DType.float32)  # scalar 2
    let c = add(a, b)  # Expected: [5, 5, 5, 5, 5]

    assert_numel(c, 5, "Result should have 5 elements")
    assert_all_values(c, 5.0, 1e-6, "3 + 2 should broadcast to [5, 5, 5, 5, 5]")


fn test_broadcast_scalar_to_2d() raises:
    """Test broadcasting scalar to 2D tensor."""
    var shape_mat = List[Int](2)
    shape_mat[0] = 3
    shape_mat[1] = 4
    var shape_scalar = List[Int](0)

    let a = ones(shape_mat, DType.float32)  # 3x4 matrix of ones
    let b = full(shape_scalar, 5.0, DType.float32)  # scalar 5
    let c = multiply(a, b)  # Expected: 3x4 matrix of fives

    assert_numel(c, 12, "Result should have 12 elements")
    assert_all_values(c, 5.0, 1e-6, "1 * 5 should broadcast to all 5s")


fn test_broadcast_scalar_to_3d() raises:
    """Test broadcasting scalar to 3D tensor."""
    var shape_3d = List[Int](3)
    shape_3d[0] = 2
    shape_3d[1] = 3
    shape_3d[2] = 4
    var shape_scalar = List[Int](0)

    let a = full(shape_3d, 2.0, DType.float32)  # 2x3x4 tensor
    let b = full(shape_scalar, 3.0, DType.float32)  # scalar 3
    let c = add(a, b)  # Expected: 2x3x4 tensor of fives

    assert_numel(c, 24, "Result should have 24 elements")
    assert_all_values(c, 5.0, 1e-6, "2 + 3 should broadcast to all 5s")


# ============================================================================
# Test vector-to-matrix broadcasting
# ============================================================================

fn test_broadcast_vector_to_matrix_row() raises:
    """Test broadcasting row vector to matrix."""
    var shape_mat = List[Int](2)
    shape_mat[0] = 3
    shape_mat[1] = 4
    var shape_vec = List[Int](2)
    shape_vec[0] = 1
    shape_vec[1] = 4

    let a = ones(shape_mat, DType.float32)  # 3x4 matrix
    let b = full(shape_vec, 2.0, DType.float32)  # 1x4 vector
    let c = add(a, b)  # Expected: 3x4 matrix, each row is [3, 3, 3, 3]

    assert_numel(c, 12, "Result should have 12 elements")
    assert_all_values(c, 3.0, 1e-6, "Broadcasting 1x4 vector to 3x4 matrix")


fn test_broadcast_vector_to_matrix_column() raises:
    """Test broadcasting column vector to matrix."""
    var shape_mat = List[Int](2)
    shape_mat[0] = 3
    shape_mat[1] = 4
    var shape_vec = List[Int](2)
    shape_vec[0] = 3
    shape_vec[1] = 1

    let a = ones(shape_mat, DType.float32)  # 3x4 matrix
    let b = full(shape_vec, 2.0, DType.float32)  # 3x1 vector
    let c = multiply(a, b)  # Expected: 3x4 matrix, each column multiplied by 2

    assert_numel(c, 12, "Result should have 12 elements")
    assert_all_values(c, 2.0, 1e-6, "Broadcasting 3x1 vector to 3x4 matrix")


fn test_broadcast_1d_to_2d() raises:
    """Test broadcasting 1D vector to 2D matrix."""
    var shape_mat = List[Int](2)
    shape_mat[0] = 3
    shape_mat[1] = 4
    var shape_vec = List[Int](1)
    shape_vec[0] = 4

    let a = ones(shape_mat, DType.float32)  # 3x4 matrix
    let b = full(shape_vec, 3.0, DType.float32)  # 4-element vector
    let c = add(a, b)  # Expected: 3x4 matrix, each row is [4, 4, 4, 4]

    assert_numel(c, 12, "Result should have 12 elements")
    assert_all_values(c, 4.0, 1e-6, "Broadcasting 1D(4) to 2D(3,4)")


# ============================================================================
# Test dimension size 1 broadcasting
# ============================================================================

fn test_broadcast_size_one_dim_leading() raises:
    """Test broadcasting with leading dimension of size 1."""
    var shape_a = List[Int](3)
    shape_a[0] = 1
    shape_a[1] = 3
    shape_a[2] = 4
    var shape_b = List[Int](3)
    shape_b[0] = 2
    shape_b[1] = 3
    shape_b[2] = 4

    let a = full(shape_a, 2.0, DType.float32)  # 1x3x4
    let b = ones(shape_b, DType.float32)  # 2x3x4
    let c = add(a, b)  # Expected: 2x3x4, all 3s

    assert_numel(c, 24, "Result should have 24 elements")
    assert_all_values(c, 3.0, 1e-6, "Broadcasting 1x3x4 to 2x3x4")


fn test_broadcast_size_one_dim_middle() raises:
    """Test broadcasting with middle dimension of size 1."""
    var shape_a = List[Int](3)
    shape_a[0] = 2
    shape_a[1] = 1
    shape_a[2] = 4
    var shape_b = List[Int](3)
    shape_b[0] = 2
    shape_b[1] = 3
    shape_b[2] = 4

    let a = full(shape_a, 5.0, DType.float32)  # 2x1x4
    let b = ones(shape_b, DType.float32)  # 2x3x4
    let c = multiply(a, b)  # Expected: 2x3x4, all 5s

    assert_numel(c, 24, "Result should have 24 elements")
    assert_all_values(c, 5.0, 1e-6, "Broadcasting 2x1x4 to 2x3x4")


fn test_broadcast_size_one_dim_trailing() raises:
    """Test broadcasting with trailing dimension of size 1."""
    var shape_a = List[Int](3)
    shape_a[0] = 2
    shape_a[1] = 3
    shape_a[2] = 1
    var shape_b = List[Int](3)
    shape_b[0] = 2
    shape_b[1] = 3
    shape_b[2] = 4

    let a = full(shape_a, 3.0, DType.float32)  # 2x3x1
    let b = full(shape_b, 2.0, DType.float32)  # 2x3x4
    let c = add(a, b)  # Expected: 2x3x4, all 5s

    assert_numel(c, 24, "Result should have 24 elements")
    assert_all_values(c, 5.0, 1e-6, "Broadcasting 2x3x1 to 2x3x4")


# ============================================================================
# Test missing dimensions broadcasting
# ============================================================================

fn test_broadcast_missing_leading_dims() raises:
    """Test broadcasting when tensor has fewer dimensions (aligned to right)."""
    var shape_3d = List[Int](3)
    shape_3d[0] = 2
    shape_3d[1] = 3
    shape_3d[2] = 4
    var shape_1d = List[Int](1)
    shape_1d[0] = 4

    let a = ones(shape_3d, DType.float32)  # 2x3x4
    let b = full(shape_1d, 2.0, DType.float32)  # (4,) -> broadcasts to (1,1,4) -> (2,3,4)
    let c = multiply(a, b)  # Expected: 2x3x4, all 2s

    assert_numel(c, 24, "Result should have 24 elements")
    assert_all_values(c, 2.0, 1e-6, "Broadcasting (4,) to (2,3,4)")


fn test_broadcast_2d_to_3d() raises:
    """Test broadcasting 2D to 3D."""
    var shape_3d = List[Int](3)
    shape_3d[0] = 2
    shape_3d[1] = 3
    shape_3d[2] = 4
    var shape_2d = List[Int](2)
    shape_2d[0] = 3
    shape_2d[1] = 4

    let a = ones(shape_3d, DType.float32)  # 2x3x4
    let b = full(shape_2d, 3.0, DType.float32)  # 3x4 -> broadcasts to (1,3,4) -> (2,3,4)
    let c = add(a, b)  # Expected: 2x3x4, all 4s

    assert_numel(c, 24, "Result should have 24 elements")
    assert_all_values(c, 4.0, 1e-6, "Broadcasting (3,4) to (2,3,4)")


# ============================================================================
# Test complex multi-dimensional broadcasting
# ============================================================================

fn test_broadcast_3d_complex() raises:
    """Test complex 3D broadcasting with multiple size-1 dimensions."""
    var shape_a = List[Int](3)
    shape_a[0] = 2
    shape_a[1] = 1
    shape_a[2] = 4
    var shape_b = List[Int](3)
    shape_b[0] = 1
    shape_b[1] = 3
    shape_b[2] = 4

    let a = full(shape_a, 2.0, DType.float32)  # 2x1x4
    let b = full(shape_b, 3.0, DType.float32)  # 1x3x4
    let c = add(a, b)  # Expected: 2x3x4, all 5s

    assert_numel(c, 24, "Result should have 24 elements")
    assert_all_values(c, 5.0, 1e-6, "Broadcasting (2,1,4) + (1,3,4) to (2,3,4)")


fn test_broadcast_4d() raises:
    """Test 4D broadcasting."""
    var shape_a = List[Int](4)
    shape_a[0] = 2
    shape_a[1] = 1
    shape_a[2] = 3
    shape_a[3] = 4
    var shape_b = List[Int](4)
    shape_b[0] = 1
    shape_b[1] = 5
    shape_b[2] = 3
    shape_b[3] = 4

    let a = ones(shape_a, DType.float32)  # 2x1x3x4
    let b = full(shape_b, 2.0, DType.float32)  # 1x5x3x4
    let c = multiply(a, b)  # Expected: 2x5x3x4, all 2s

    assert_numel(c, 120, "Result should have 120 elements (2*5*3*4)")
    assert_all_values(c, 2.0, 1e-6, "Broadcasting (2,1,3,4) * (1,5,3,4) to (2,5,3,4)")


# ============================================================================
# Test incompatible shapes (should error)
# ============================================================================

fn test_broadcast_incompatible_shapes_different_sizes() raises:
    """Test that incompatible shapes raise error."""
    var shape_a = List[Int](2)
    shape_a[0] = 3
    shape_a[1] = 4
    var shape_b = List[Int](2)
    shape_b[0] = 3
    shape_b[1] = 5  # Incompatible: 4 != 5 and neither is 1

    let a = ones(shape_a, DType.float32)
    let b = ones(shape_b, DType.float32)

    # Verify this raises an error
    var error_raised = False
    try:
        let c = add(a, b)
    except:
        error_raised = True

    if not error_raised:
        raise Error("Should have raised error for incompatible broadcast shapes (3,4) and (3,5)")


fn test_broadcast_incompatible_inner_dims() raises:
    """Test that incompatible inner dimensions raise error."""
    var shape_a = List[Int](3)
    shape_a[0] = 2
    shape_a[1] = 3
    shape_a[2] = 4
    var shape_b = List[Int](3)
    shape_b[0] = 2
    shape_b[1] = 5  # Incompatible: 3 != 5 and neither is 1
    shape_b[2] = 4

    let a = ones(shape_a, DType.float32)
    let b = ones(shape_b, DType.float32)

    # Verify this raises an error
    var error_raised = False
    try:
        let c = add(a, b)
    except:
        error_raised = True

    if not error_raised:
        raise Error("Should have raised error for incompatible broadcast shapes (2,3,4) and (2,5,4)")


# ============================================================================
# Test broadcast output shape
# ============================================================================

fn test_broadcast_output_shape_scalar_1d() raises:
    """Test broadcast output shape for scalar + 1D."""
    var shape_vec = List[Int](1)
    shape_vec[0] = 5
    var shape_scalar = List[Int](0)

    let a = ones(shape_vec, DType.float32)
    let b = ones(shape_scalar, DType.float32)
    let c = add(a, b)

    assert_dim(c, 1, "Output should be 1D")
    assert_numel(c, 5, "Output should have 5 elements")


fn test_broadcast_output_shape_1d_2d() raises:
    """Test broadcast output shape for 1D + 2D."""
    var shape_2d = List[Int](2)
    shape_2d[0] = 3
    shape_2d[1] = 4
    var shape_1d = List[Int](1)
    shape_1d[0] = 4

    let a = ones(shape_2d, DType.float32)
    let b = ones(shape_1d, DType.float32)
    let c = add(a, b)

    assert_dim(c, 2, "Output should be 2D")
    assert_numel(c, 12, "Output should have 12 elements")


fn test_broadcast_output_shape_3d_complex() raises:
    """Test broadcast output shape for complex 3D case."""
    var shape_a = List[Int](3)
    shape_a[0] = 2
    shape_a[1] = 1
    shape_a[2] = 4
    var shape_b = List[Int](3)
    shape_b[0] = 1
    shape_b[1] = 3
    shape_b[2] = 4

    let a = ones(shape_a, DType.float32)
    let b = ones(shape_b, DType.float32)
    let c = add(a, b)

    assert_dim(c, 3, "Output should be 3D")
    assert_numel(c, 24, "Output should have 24 elements (2*3*4)")


# ============================================================================
# Test dtype preservation in broadcasting
# ============================================================================

fn test_broadcast_preserves_dtype() raises:
    """Test that broadcasting preserves dtype."""
    var shape_a = List[Int](2)
    shape_a[0] = 3
    shape_a[1] = 4
    var shape_b = List[Int](1)
    shape_b[0] = 4

    let a = ones(shape_a, DType.float64)
    let b = ones(shape_b, DType.float64)
    let c = add(a, b)

    assert_dtype(c, DType.float64, "Broadcast should preserve float64 dtype")


# ============================================================================
# Test broadcasting integration with comparison operations
# ============================================================================

fn test_broadcast_with_comparison_scalar() raises:
    """Test broadcasting scalar with comparison operations."""
    from shared.core import greater

    var shape_vec = List[Int](1)
    shape_vec[0] = 5
    var shape_scalar = List[Int](0)

    let a = full(shape_vec, 3.0, DType.float32)  # [3, 3, 3, 3, 3]
    let b = full(shape_scalar, 2.0, DType.float32)  # scalar 2
    let c = greater(a, b)  # Should broadcast: [True, True, True, True, True]

    assert_numel(c, 5, "Result should have 5 elements")
    assert_dtype(c, DType.bool, "Comparison should return bool dtype")
    for i in range(5):
        assert_value_at(c, i, 1.0, 1e-6, "3 > 2 should be True")


fn test_broadcast_with_comparison_vector_matrix() raises:
    """Test broadcasting vector to matrix with comparison."""
    from shared.core import less_equal

    var shape_mat = List[Int](2)
    shape_mat[0] = 3
    shape_mat[1] = 4
    var shape_vec = List[Int](1)
    shape_vec[0] = 4

    let a = ones(shape_mat, DType.float32)  # 3x4 matrix of ones
    let b = full(shape_vec, 2.0, DType.float32)  # vector [2, 2, 2, 2]
    let c = less_equal(a, b)  # 1 <= 2 broadcasts to 3x4

    assert_numel(c, 12, "Result should have 12 elements")
    assert_dtype(c, DType.bool, "Comparison should return bool dtype")
    for i in range(12):
        assert_value_at(c, i, 1.0, 1e-6, "1 <= 2 should be True")


fn test_broadcast_chained_operations() raises:
    """Test chained operations with broadcasting."""
    var shape_mat = List[Int](2)
    shape_mat[0] = 2
    shape_mat[1] = 3
    var shape_scalar = List[Int](0)

    let a = full(shape_mat, 5.0, DType.float32)  # 2x3 matrix
    let b = full(shape_scalar, 2.0, DType.float32)  # scalar
    let c = full(shape_scalar, 3.0, DType.float32)  # scalar

    # (a + b) * c = (5 + 2) * 3 = 7 * 3 = 21
    let result = multiply(add(a, b), c)

    assert_numel(result, 6, "Result should have 6 elements")
    assert_all_values(result, 21.0, 1e-6, "(5 + 2) * 3 should be 21")


fn test_broadcast_with_subtract() raises:
    """Test broadcasting with subtraction."""
    from shared.core import subtract

    var shape_2d = List[Int](2)
    shape_2d[0] = 3
    shape_2d[1] = 4
    var shape_1d = List[Int](1)
    shape_1d[0] = 4

    let a = full(shape_2d, 10.0, DType.float32)  # 3x4 matrix of 10s
    let b = full(shape_1d, 3.0, DType.float32)  # vector [3, 3, 3, 3]
    let c = subtract(a, b)  # 10 - 3 = 7, broadcast to 3x4

    assert_numel(c, 12, "Result should have 12 elements")
    assert_all_values(c, 7.0, 1e-6, "10 - 3 should broadcast to all 7s")


fn test_broadcast_with_divide() raises:
    """Test broadcasting with division."""
    from shared.core import divide

    var shape_mat = List[Int](2)
    shape_mat[0] = 2
    shape_mat[1] = 5
    var shape_scalar = List[Int](0)

    let a = full(shape_mat, 20.0, DType.float32)  # 2x5 matrix of 20s
    let b = full(shape_scalar, 4.0, DType.float32)  # scalar 4
    let c = divide(a, b)  # 20 / 4 = 5, broadcast

    assert_numel(c, 10, "Result should have 10 elements")
    assert_all_values(c, 5.0, 1e-6, "20 / 4 should broadcast to all 5s")


fn test_broadcast_complex_3d_with_multiply() raises:
    """Test complex 3D broadcasting with multiply."""
    var shape_a = List[Int](3)
    shape_a[0] = 2
    shape_a[1] = 1
    shape_a[2] = 4
    var shape_b = List[Int](3)
    shape_b[0] = 1
    shape_b[1] = 3
    shape_b[2] = 4

    let a = full(shape_a, 3.0, DType.float32)  # 2x1x4
    let b = full(shape_b, 4.0, DType.float32)  # 1x3x4
    let c = multiply(a, b)  # 3 * 4 = 12, broadcast to 2x3x4

    assert_numel(c, 24, "Result should have 24 elements")
    assert_all_values(c, 12.0, 1e-6, "3 * 4 should broadcast to all 12s")


# ============================================================================
# Main test runner
# ============================================================================

fn main() raises:
    """Run all broadcasting tests."""
    print("Running ExTensor broadcasting tests...")

    # Scalar broadcasting
    print("  Testing scalar broadcasting...")
    test_broadcast_scalar_to_1d()
    test_broadcast_scalar_to_2d()
    test_broadcast_scalar_to_3d()

    # Vector-to-matrix broadcasting
    print("  Testing vector-to-matrix broadcasting...")
    test_broadcast_vector_to_matrix_row()
    test_broadcast_vector_to_matrix_column()
    test_broadcast_1d_to_2d()

    # Size-1 dimension broadcasting
    print("  Testing size-1 dimension broadcasting...")
    test_broadcast_size_one_dim_leading()
    test_broadcast_size_one_dim_middle()
    test_broadcast_size_one_dim_trailing()

    # Missing dimensions
    print("  Testing missing dimensions broadcasting...")
    test_broadcast_missing_leading_dims()
    test_broadcast_2d_to_3d()

    # Complex multi-dimensional
    print("  Testing complex multi-dimensional broadcasting...")
    test_broadcast_3d_complex()
    test_broadcast_4d()

    # Incompatible shapes
    print("  Testing incompatible shapes...")
    test_broadcast_incompatible_shapes_different_sizes()
    test_broadcast_incompatible_inner_dims()

    # Output shape verification
    print("  Testing broadcast output shapes...")
    test_broadcast_output_shape_scalar_1d()
    test_broadcast_output_shape_1d_2d()
    test_broadcast_output_shape_3d_complex()

    # Dtype preservation
    print("  Testing dtype preservation...")
    test_broadcast_preserves_dtype()

    # Integration tests
    print("  Testing broadcasting integration with operations...")
    test_broadcast_with_comparison_scalar()
    test_broadcast_with_comparison_vector_matrix()
    test_broadcast_chained_operations()
    test_broadcast_with_subtract()
    test_broadcast_with_divide()
    test_broadcast_complex_3d_with_multiply()

    print("All broadcasting tests completed!")
