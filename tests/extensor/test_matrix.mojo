"""Tests for ExTensor matrix operations.

Tests matrix operations including matmul, transpose, dot, outer, inner,
and tensordot with various shapes and dimensions.
"""

from sys import DType

# Import ExTensor and operations
from extensor import ExTensor, zeros, ones, full, arange, eye

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
    let a = eye(3, 3, DType.float32)  # 3x3 identity
    let b = full(DynamicVector[Int](2), 2.0, DType.float32)  # Will need to set shape
    # Create 3x3 matrix filled with 2s
    # let c = matmul(a, b)  # TODO: Implement matmul()

    # Identity @ B = B, so result should be all 2s
    # assert_numel(c, 9, "Result should be 3x3 (9 elements)")
    # assert_all_values(c, 2.0, 1e-6, "Identity @ B should equal B")
    pass  # Placeholder


fn test_matmul_2d_rectangular() raises:
    """Test 2D matrix multiplication with rectangular matrices."""
    var shape_a = DynamicVector[Int](2)
    shape_a[0] = 3
    shape_a[1] = 4
    var shape_b = DynamicVector[Int](2)
    shape_b[0] = 4
    shape_b[1] = 2

    let a = ones(shape_a, DType.float32)  # 3x4
    let b = full(shape_b, 2.0, DType.float32)  # 4x2
    # let c = matmul(a, b)  # TODO: Implement matmul()

    # Result should be 3x2, each element = 1*2 + 1*2 + 1*2 + 1*2 = 8
    # assert_dim(c, 2, "Result should be 2D")
    # assert_numel(c, 6, "Result should be 3x2 (6 elements)")
    # assert_all_values(c, 8.0, 1e-6, "Each element should be 8")
    pass  # Placeholder


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
    var shape_a = DynamicVector[Int](3)
    shape_a[0] = 2  # batch size
    shape_a[1] = 3  # rows
    shape_a[2] = 4  # cols
    var shape_b = DynamicVector[Int](3)
    shape_b[0] = 2  # batch size
    shape_b[1] = 4  # rows
    shape_b[2] = 2  # cols

    let a = ones(shape_a, DType.float32)  # 2x3x4
    let b = full(shape_b, 0.5, DType.float32)  # 2x4x2
    # let c = matmul(a, b)  # TODO: Implement batched matmul

    # Result should be 2x3x2 (batch_size x a_rows x b_cols)
    # Each element = 1*0.5 + 1*0.5 + 1*0.5 + 1*0.5 = 2
    # assert_dim(c, 3, "Result should be 3D")
    # assert_numel(c, 12, "Result should be 2x3x2 (12 elements)")
    # assert_all_values(c, 2.0, 1e-6, "Each element should be 2")
    pass  # Placeholder


fn test_matmul_batched_4d() raises:
    """Test batched matrix multiplication (4D)."""
    var shape_a = DynamicVector[Int](4)
    shape_a[0] = 2  # batch dim 1
    shape_a[1] = 3  # batch dim 2
    shape_a[2] = 4  # rows
    shape_a[3] = 5  # cols
    var shape_b = DynamicVector[Int](4)
    shape_b[0] = 2
    shape_b[1] = 3
    shape_b[2] = 5  # rows
    shape_b[3] = 2  # cols

    let a = ones(shape_a, DType.float32)  # 2x3x4x5
    let b = ones(shape_b, DType.float32)  # 2x3x5x2
    # let c = matmul(a, b)  # TODO: Implement batched matmul

    # Result should be 2x3x4x2
    # assert_numel(c, 48, "Result should be 2x3x4x2 (48 elements)")
    pass  # Placeholder


# ============================================================================
# Test matmul() - Error cases
# ============================================================================

fn test_matmul_incompatible_shapes() raises:
    """Test that incompatible shapes raise error."""
    var shape_a = DynamicVector[Int](2)
    shape_a[0] = 3
    shape_a[1] = 4
    var shape_b = DynamicVector[Int](2)
    shape_b[0] = 5  # Incompatible: 4 != 5
    shape_b[1] = 2

    let a = ones(shape_a, DType.float32)
    let b = ones(shape_b, DType.float32)

    # TODO: Verify error handling
    # try:
    #     let c = matmul(a, b)
    #     raise Error("Should have raised error for incompatible shapes")
    # except:
    #     pass  # Expected
    pass  # Placeholder


# ============================================================================
# Test transpose()
# ============================================================================

fn test_transpose_2d() raises:
    """Test transpose of 2D matrix."""
    var shape = DynamicVector[Int](2)
    shape[0] = 3
    shape[1] = 4
    let a = ones(shape, DType.float32)  # 3x4
    # let b = transpose(a)  # TODO: Implement transpose()

    # Result should be 4x3
    # assert_dim(b, 2, "Transpose should be 2D")
    # assert_numel(b, 12, "Transpose should have same number of elements")
    # TODO: Check actual shape is (4, 3)
    pass  # Placeholder


fn test_transpose_3d_default() raises:
    """Test transpose of 3D tensor (default permutation)."""
    var shape = DynamicVector[Int](3)
    shape[0] = 2
    shape[1] = 3
    shape[2] = 4
    let a = ones(shape, DType.float32)  # 2x3x4
    # let b = transpose(a)  # TODO: Implement transpose()

    # Default: reverse all axes -> 4x3x2
    # assert_dim(b, 3, "Transpose should be 3D")
    # assert_numel(b, 24, "Transpose should have same number of elements")
    pass  # Placeholder


fn test_transpose_3d_permute() raises:
    """Test transpose with custom axis permutation."""
    var shape = DynamicVector[Int](3)
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


# ============================================================================
# Test dot()
# ============================================================================

fn test_dot_1d() raises:
    """Test dot product of two 1D vectors."""
    let a = arange(1.0, 6.0, 1.0, DType.float32)  # [1, 2, 3, 4, 5]
    let b = arange(1.0, 6.0, 1.0, DType.float32)  # [1, 2, 3, 4, 5]
    # let c = dot(a, b)  # TODO: Implement dot()

    # Expected: 1*1 + 2*2 + 3*3 + 4*4 + 5*5 = 55
    # assert_numel(c, 1, "Dot product should be scalar")
    # assert_value_at(c, 0, 55.0, 1e-4, "Dot product result")
    pass  # Placeholder


fn test_dot_2d() raises:
    """Test dot product (equivalent to matmul for 2D)."""
    var shape = DynamicVector[Int](2)
    shape[0] = 2
    shape[1] = 3
    let a = ones(shape, DType.float32)  # 2x3
    var shape_b = DynamicVector[Int](2)
    shape_b[0] = 3
    shape_b[1] = 2
    let b = ones(shape_b, DType.float32)  # 3x2
    # let c = dot(a, b)  # TODO: Implement dot()

    # Should behave like matmul for 2D
    # assert_numel(c, 4, "Result should be 2x2 (4 elements)")
    # assert_all_values(c, 3.0, 1e-6, "Each element should be 3")
    pass  # Placeholder


# ============================================================================
# Test outer()
# ============================================================================

fn test_outer_vectors() raises:
    """Test outer product of two vectors."""
    let a = arange(1.0, 4.0, 1.0, DType.float32)  # [1, 2, 3]
    let b = arange(1.0, 3.0, 1.0, DType.float32)  # [1, 2]
    # let c = outer(a, b)  # TODO: Implement outer()

    # Expected 3x2 matrix:
    # [[1, 2],
    #  [2, 4],
    #  [3, 6]]
    # assert_dim(c, 2, "Outer product should be 2D")
    # assert_numel(c, 6, "Outer product should be 3x2")
    # assert_value_at(c, 0, 1.0, 1e-6, "c[0,0] = 1*1")
    # assert_value_at(c, 3, 4.0, 1e-6, "c[1,1] = 2*2")
    pass  # Placeholder


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
    var shape = DynamicVector[Int](2)
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
    var shape_a = DynamicVector[Int](2)
    shape_a[0] = 3
    shape_a[1] = 4
    var shape_b = DynamicVector[Int](2)
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
    var shape_a = DynamicVector[Int](3)
    shape_a[0] = 2
    shape_a[1] = 3
    shape_a[2] = 4
    var shape_b = DynamicVector[Int](3)
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
    var shape = DynamicVector[Int](2)
    shape[0] = 3
    shape[1] = 3
    let a = ones(shape, DType.float64)
    let b = ones(shape, DType.float64)
    # let c = matmul(a, b)

    # assert_dtype(c, DType.float64, "matmul should preserve float64")
    pass  # Placeholder


# ============================================================================
# Test dunder methods
# ============================================================================

fn test_dunder_matmul() raises:
    """Test __matmul__ operator overloading (a @ b)."""
    var shape = DynamicVector[Int](2)
    shape[0] = 2
    shape[1] = 2
    let a = ones(shape, DType.float32)
    let b = full(shape, 2.0, DType.float32)
    # let c = a @ b  # TODO: Implement __matmul__

    # Each element should be 1*2 + 1*2 = 4
    # assert_all_values(c, 4.0, 1e-6, "a @ b should work via __matmul__")
    pass  # Placeholder


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

    # transpose() tests
    print("  Testing transpose()...")
    test_transpose_2d()
    test_transpose_3d_default()
    test_transpose_3d_permute()
    test_transpose_values_correctness()

    # dot() tests
    print("  Testing dot()...")
    test_dot_1d()
    test_dot_2d()

    # outer() tests
    print("  Testing outer()...")
    test_outer_vectors()

    # inner() tests
    print("  Testing inner()...")
    test_inner_1d()
    test_inner_2d()

    # tensordot() tests
    print("  Testing tensordot()...")
    test_tensordot_basic()
    test_tensordot_multiple_axes()

    # Dtype preservation
    print("  Testing dtype preservation...")
    test_matmul_preserves_dtype()

    # Dunder methods
    print("  Testing matrix dunders...")
    test_dunder_matmul()

    print("All matrix operation tests completed!")
