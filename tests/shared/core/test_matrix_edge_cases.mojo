"""Tests for matrix operation edge cases.

Tests edge cases for matmul and shape operations on matrices including:
- Matmul with different tensor sizes
- Matrix operations with special values
- Shape preservation in operations
"""

# Import ExTensor and operations
from shared.core.extensor import ExTensor, zeros, ones, full, zeros_like, eye
from shared.core.matrix import matmul

# Import test helpers
from tests.shared.conftest import (
    assert_dtype,
    assert_numel,
    assert_dim,
    assert_value_at,
    assert_all_values,
    assert_all_close,
    assert_true,
)


# ============================================================================
# Test matmul with small matrices
# ============================================================================


fn test_matmul_1x1() raises:
    """Test 1x1 matrix multiplication."""
    var shape = List[Int]()
    shape.append(1)
    shape.append(1)

    var a = full(shape, 5.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)
    var c = matmul(a, b)

    assert_numel(c, 1, "Result should have 1 element")
    assert_value_at(c, 0, 15.0, 1e-5, "5 * 3 = 15")


fn test_matmul_2x2() raises:
    """Test 2x2 matrix multiplication."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(2)

    var a = ones(shape, DType.float32)
    var b = ones(shape, DType.float32)
    var c = matmul(a, b)

    assert_dim(c, 2, "Result should be 2D")
    assert_numel(c, 4, "Result should have 4 elements")
    # Each element is sum of 2 ones = 2
    assert_all_values(c, 2.0, 1e-5, "1x1+1x1 = 2 for each element")


fn test_matmul_3x3() raises:
    """Test 3x3 matrix multiplication."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(3)

    var a = ones(shape, DType.float32)
    var b = ones(shape, DType.float32)
    var c = matmul(a, b)

    assert_dim(c, 2, "Result should be 2D")
    assert_numel(c, 9, "Result should have 9 elements")
    # Each element is sum of 3 ones = 3
    assert_all_values(c, 3.0, 1e-5, "Sum of 3 ones = 3")


fn test_matmul_2x3_3x2() raises:
    """Test [2,3] @ [3,2] matrix multiplication."""
    var shape_a = List[Int]()
    shape_a.append(2)
    shape_a.append(3)

    var shape_b = List[Int]()
    shape_b.append(3)
    shape_b.append(2)

    var a = ones(shape_a, DType.float32)
    var b = ones(shape_b, DType.float32)

    var shape_c = List[Int]()
    shape_c.append(2)
    shape_c.append(2)
    var c = matmul(a, b)

    assert_dim(c, 2, "Result should be 2D")
    assert_numel(c, 4, "Result [2,2] should have 4 elements")
    # Each element is sum of 3 ones = 3
    assert_all_values(c, 3.0, 1e-5, "1x1+1x1+1x1 = 3 for each element")


fn test_matmul_non_square() raises:
    """Test [4,3] @ [3,5] matrix multiplication."""
    var shape_a = List[Int]()
    shape_a.append(4)
    shape_a.append(3)

    var shape_b = List[Int]()
    shape_b.append(3)
    shape_b.append(5)

    var a = full(shape_a, 1.0, DType.float32)
    var b = full(shape_b, 2.0, DType.float32)

    var shape_c = List[Int]()
    shape_c.append(4)
    shape_c.append(5)
    var c = matmul(a, b)

    assert_dim(c, 2, "Result should be 2D")
    assert_numel(c, 20, "Result [4,5] should have 20 elements")
    # Each element is sum of 3 values of (1 * 2) = 6
    assert_all_values(c, 6.0, 1e-5, "1x2 summed 3 times = 6")


fn test_matmul_identity() raises:
    """Test A @ I = A with identity matrix."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(3)

    var a = full(shape, 2.0, DType.float32)
    var identity = eye(3, 3, 0, DType.float32)

    var c = matmul(a, identity)

    # Result should be approximately equal to a
    assert_all_close(c, a, 1e-5, "A @ I = A")


fn test_matmul_with_zeros() raises:
    """Test matrix multiplication with zero matrix."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(2)

    var a = full(shape, 5.0, DType.float32)
    var b = zeros(shape, DType.float32)

    var c = matmul(a, b)

    # Result should be all zeros
    assert_all_values(c, 0.0, 1e-6, "A @ 0 = 0")


fn test_matmul_with_ones() raises:
    """Test matrix multiplication with ones matrix."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(2)

    var a = full(shape, 3.0, DType.float32)
    var b = ones(shape, DType.float32)

    var c = matmul(a, b)

    # Each element is 3*1 + 3*1 = 6
    assert_all_values(c, 6.0, 1e-5, "3*1+3*1 = 6")


# ============================================================================
# Test matmul numerical stability
# ============================================================================


fn test_matmul_small_values() raises:
    """Test matmul with very small values."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(2)

    var t = full(shape, 1e-10, DType.float32)
    var c = matmul(t, t)

    # Should still compute correctly
    assert_dim(c, 2, "Result should be 2D")


fn test_matmul_large_values() raises:
    """Test matmul with large values."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(2)

    var t = full(shape, 1e6, DType.float32)
    var c = matmul(t, t)

    # Should compute (may overflow)
    assert_dim(c, 2, "Result should be 2D")


fn test_matmul_mixed_signs() raises:
    """Test matmul with mixed positive/negative values."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(2)

    var a = full(shape, 2.0, DType.float32)
    var b = full(shape, -1.0, DType.float32)

    var c = matmul(a, b)

    # 2*(-1) + 2*(-1) = -4
    assert_all_values(c, -4.0, 1e-5, "Mixed sign matmul")


# ============================================================================
# Test matmul with different dtypes
# ============================================================================


fn test_matmul_float64() raises:
    """Test matmul with float64 dtype."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(2)

    var a = ones(shape, DType.float64)
    var b = ones(shape, DType.float64)
    var c = matmul(a, b)

    assert_dtype(c, DType.float64, "Result should be float64")
    assert_all_values(c, 2.0, 1e-10, "Float64 matmul")


fn test_matmul_float32() raises:
    """Test matmul with float32 dtype."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(2)

    var a = ones(shape, DType.float32)
    var b = ones(shape, DType.float32)
    var c = matmul(a, b)

    assert_dtype(c, DType.float32, "Result should be float32")
    assert_all_values(c, 2.0, 1e-5, "Float32 matmul")


# ============================================================================
# Test matmul correctness with known values
# ============================================================================


fn test_matmul_known_result() raises:
    """Test matmul with known result values."""
    # [1, 2] @ [3, 4] = 1*3 + 2*4 = 11
    var shape_a = List[Int]()
    shape_a.append(1)
    shape_a.append(2)

    var shape_b = List[Int]()
    shape_b.append(2)
    shape_b.append(1)

    var a = ExTensor(shape_a, DType.float32)
    a._set_float32(0, 1.0)
    a._set_float32(1, 2.0)

    var b = ExTensor(shape_b, DType.float32)
    b._set_float32(0, 3.0)
    b._set_float32(1, 4.0)

    var c = matmul(a, b)

    assert_value_at(c, 0, 11.0, 1e-5, "[1,2]@[3,4]^T = 11")


# ============================================================================
# Main test runner
# ============================================================================


fn main() raises:
    """Run all matrix operation edge case tests."""
    print("Running matrix operation edge case tests...")

    # Small matrix operations
    print("  Testing small matrix operations...")
    test_matmul_1x1()
    test_matmul_2x2()
    test_matmul_3x3()
    test_matmul_2x3_3x2()
    test_matmul_non_square()

    # Special matrices
    print("  Testing special matrix operations...")
    test_matmul_identity()
    test_matmul_with_zeros()
    test_matmul_with_ones()

    # Numerical stability
    print("  Testing numerical stability...")
    test_matmul_small_values()
    test_matmul_large_values()
    test_matmul_mixed_signs()

    # Different dtypes
    print("  Testing different dtypes...")
    test_matmul_float64()
    test_matmul_float32()

    # Known results
    print("  Testing known results...")
    test_matmul_known_result()

    print("All matrix operation edge case tests completed!")
