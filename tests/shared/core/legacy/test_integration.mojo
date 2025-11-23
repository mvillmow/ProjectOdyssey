"""Integration tests for ExTensor operations.

Tests combinations of multiple operations working together, realistic usage patterns,
and end-to-end workflows using currently implemented functionality.
"""

from sys import DType

# Import ExTensor and operations
from shared.core import ExTensor, zeros, ones, full, arange, eye, linspace, add, subtract, multiply

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
# Test chained arithmetic operations
# ============================================================================

fn test_chained_add_operations() raises:
    """Test chaining multiple add operations."""
    var shape = List[Int]()
    shape.append(5)
    let a = ones(shape, DType.float32)  # [1, 1, 1, 1, 1]
    let b = full(shape, 2.0, DType.float32)  # [2, 2, 2, 2, 2]
    let c = full(shape, 3.0, DType.float32)  # [3, 3, 3, 3, 3]

    let result = add(add(a, b), c)  # (1+2)+3 = 6

    assert_all_values(result, 6.0, 1e-6, "Chained additions should work")


fn test_mixed_arithmetic_operations() raises:
    """Test mixing different arithmetic operations."""
    var shape = List[Int]()
    shape.append(5)
    let a = full(shape, 2.0, DType.float32)
    let b = full(shape, 3.0, DType.float32)
    let c = full(shape, 4.0, DType.float32)

    # (a + b) * c = (2 + 3) * 4 = 20
    let sum_ab = add(a, b)
    let result = multiply(sum_ab, c)

    assert_all_values(result, 20.0, 1e-6, "Mixed operations should work")


fn test_arithmetic_with_operator_overloading() raises:
    """Test using operator overloading for complex expressions."""
    var shape = List[Int]()
    shape.append(5)
    let a = ones(shape, DType.float32)
    let b = full(shape, 2.0, DType.float32)
    let c = full(shape, 3.0, DType.float32)

    # a + b * c = 1 + 2 * 3 = 1 + 6 = 7
    let result = a + b * c

    assert_all_values(result, 7.0, 1e-6, "Operator precedence should work")


fn test_complex_expression() raises:
    """Test complex arithmetic expression."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    let a = full(shape, 1.0, DType.float32)
    let b = full(shape, 2.0, DType.float32)
    let c = full(shape, 3.0, DType.float32)
    let d = full(shape, 4.0, DType.float32)

    # ((a + b) * c) - d = ((1 + 2) * 3) - 4 = 9 - 4 = 5
    let result = ((a + b) * c) - d

    assert_all_values(result, 5.0, 1e-6, "Complex expressions should work")


# ============================================================================
# Test creation + arithmetic patterns
# ============================================================================

fn test_identity_matrix_operations() raises:
    """Test operations with identity matrix."""
    let I = eye(3, 3, 0, DType.float32)
    let A = full(List[Int](), 2.0, DType.float32)  # Will need reshaping
    var shape = List[Int]()
    shape.append(3)
    shape.append(3)
    let B = full(shape, 2.0, DType.float32)

    # I + B should give all 3s on diagonal, 2s elsewhere
    let result = add(I, B)

    assert_numel(result, 9, "Result should be 3x3")
    # Check diagonal
    assert_value_at(result, 0, 3.0, 1e-6, "Diagonal [0,0]")
    assert_value_at(result, 4, 3.0, 1e-6, "Diagonal [1,1]")
    assert_value_at(result, 8, 3.0, 1e-6, "Diagonal [2,2]")
    # Check off-diagonal
    assert_value_at(result, 1, 2.0, 1e-6, "Off-diagonal [0,1]")
    assert_value_at(result, 3, 2.0, 1e-6, "Off-diagonal [1,0]")


fn test_arange_arithmetic() raises:
    """Test arithmetic with arange-created tensors."""
    let a = arange(0.0, 5.0, 1.0, DType.float32)  # [0, 1, 2, 3, 4]
    var shape = List[Int]()
    shape.append(5)
    let b = ones(shape, DType.float32)  # [1, 1, 1, 1, 1]

    let result = add(a, b)  # [1, 2, 3, 4, 5]

    assert_value_at(result, 0, 1.0, 1e-6, "0 + 1 = 1")
    assert_value_at(result, 2, 3.0, 1e-6, "2 + 1 = 3")
    assert_value_at(result, 4, 5.0, 1e-6, "4 + 1 = 5")


fn test_linspace_operations() raises:
    """Test operations with linspace-created tensors."""
    let a = linspace(0.0, 4.0, 5, DType.float32)  # [0, 1, 2, 3, 4]
    let b = linspace(5.0, 9.0, 5, DType.float32)  # [5, 6, 7, 8, 9]

    let result = add(a, b)  # [5, 7, 9, 11, 13]

    assert_value_at(result, 0, 5.0, 1e-6, "0 + 5 = 5")
    assert_value_at(result, 2, 9.0, 1e-6, "2 + 7 = 9")
    assert_value_at(result, 4, 13.0, 1e-6, "4 + 9 = 13")


# ============================================================================
# Test multiple dtype operations
# ============================================================================

fn test_same_dtype_consistency() raises:
    """Test that operations preserve dtype consistently."""
    var shape = List[Int]()
    shape.append(5)

    let a32 = ones(shape, DType.float32)
    let b32 = ones(shape, DType.float32)
    let result32 = add(a32, b32)
    assert_dtype(result32, DType.float32, "float32 + float32 should be float32")

    let a64 = ones(shape, DType.float64)
    let b64 = ones(shape, DType.float64)
    let result64 = add(a64, b64)
    assert_dtype(result64, DType.float64, "float64 + float64 should be float64")


fn test_int_dtype_operations() raises:
    """Test operations with integer dtypes."""
    var shape = List[Int]()
    shape.append(5)

    let a = full(shape, 3.0, DType.int32)
    let b = full(shape, 2.0, DType.int32)
    let result = add(a, b)

    assert_dtype(result, DType.int32, "int32 + int32 should be int32")
    assert_all_values(result, 5.0, 1e-6, "3 + 2 = 5")


# ============================================================================
# Test multi-dimensional operations
# ============================================================================

fn test_2d_elementwise_operations() raises:
    """Test element-wise operations on 2D tensors."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    let a = full(shape, 5.0, DType.float32)
    let b = full(shape, 3.0, DType.float32)

    let result = subtract(a, b)  # All 2s

    assert_numel(result, 12, "Result should have 12 elements")
    assert_all_values(result, 2.0, 1e-6, "5 - 3 = 2 for all elements")


fn test_3d_operations() raises:
    """Test operations on 3D tensors."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)
    let a = ones(shape, DType.float32)
    let b = full(shape, 0.5, DType.float32)

    let result = multiply(a, b)  # All 0.5s

    assert_numel(result, 24, "Result should have 24 elements")
    assert_all_values(result, 0.5, 1e-6, "1 * 0.5 = 0.5 for all elements")


# ============================================================================
# Test realistic ML-like patterns
# ============================================================================

fn test_linear_transformation_pattern() raises:
    """Test pattern similar to linear layer: W*x + b."""
    var shape = List[Int]()
    shape.append(5)

    # Simulate weights, input, and bias
    let W = full(shape, 2.0, DType.float32)
    let x = ones(shape, DType.float32)
    let b = full(shape, 0.5, DType.float32)

    # Linear transformation: W*x + b
    let Wx = multiply(W, x)  # 2 * 1 = 2
    let result = add(Wx, b)  # 2 + 0.5 = 2.5

    assert_all_values(result, 2.5, 1e-6, "Linear transformation result")


fn test_gradient_descent_update_pattern() raises:
    """Test pattern similar to gradient descent: w - lr * grad."""
    var shape = List[Int]()
    shape.append(5)

    let w = ones(shape, DType.float32)  # weights
    let grad = full(shape, 0.2, DType.float32)  # gradients
    let lr = full(shape, 0.1, DType.float32)  # learning rate

    # Update: w - lr * grad
    let lr_grad = multiply(lr, grad)  # 0.1 * 0.2 = 0.02
    let new_w = subtract(w, lr_grad)  # 1 - 0.02 = 0.98

    assert_all_values(new_w, 0.98, 1e-6, "Weight update pattern")


fn test_batch_normalization_pattern() raises:
    """Test pattern similar to batch normalization: (x - mean) * scale."""
    var shape = List[Int]()
    shape.append(5)

    let x = full(shape, 5.0, DType.float32)
    let mean = full(shape, 3.0, DType.float32)
    let scale = full(shape, 2.0, DType.float32)

    # (x - mean) * scale
    let centered = subtract(x, mean)  # 5 - 3 = 2
    let result = multiply(centered, scale)  # 2 * 2 = 4

    assert_all_values(result, 4.0, 1e-6, "Batch norm pattern")


# ============================================================================
# Test zero and identity element behavior
# ============================================================================

fn test_additive_identity() raises:
    """Test that adding zero doesn't change values."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    let a = full(shape, 7.5, DType.float32)
    let zero = zeros(shape, DType.float32)

    let result = add(a, zero)

    assert_all_values(result, 7.5, 1e-6, "x + 0 = x")


fn test_multiplicative_identity() raises:
    """Test that multiplying by one doesn't change values."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    let a = full(shape, 7.5, DType.float32)
    let one = ones(shape, DType.float32)

    let result = multiply(a, one)

    assert_all_values(result, 7.5, 1e-6, "x * 1 = x")


fn test_multiplicative_zero() raises:
    """Test that multiplying by zero gives zero."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    let a = full(shape, 99.9, DType.float32)
    let zero = zeros(shape, DType.float32)

    let result = multiply(a, zero)

    assert_all_values(result, 0.0, 1e-8, "x * 0 = 0")


# ============================================================================
# Test scalar patterns
# ============================================================================

fn test_scalar_operations() raises:
    """Test operations with scalar tensors."""
    var shape_scalar = List[Int]()
    let a = full(shape_scalar, 5.0, DType.float32)
    let b = full(shape_scalar, 3.0, DType.float32)

    let result = add(a, b)

    assert_dim(result, 0, "Result should be scalar")
    assert_value_at(result, 0, 8.0, 1e-6, "5 + 3 = 8")


# ============================================================================
# Test large tensor operations
# ============================================================================

fn test_large_tensor_operations() raises:
    """Test operations on large tensors."""
    var shape = List[Int]()
    shape.append(10000)
    let a = ones(shape, DType.float32)
    let b = full(shape, 2.0, DType.float32)

    let result = multiply(a, b)

    assert_numel(result, 10000, "Result should have 10000 elements")
    # Spot check a few values
    assert_value_at(result, 0, 2.0, 1e-6, "First element")
    assert_value_at(result, 5000, 2.0, 1e-6, "Middle element")
    assert_value_at(result, 9999, 2.0, 1e-6, "Last element")


# ============================================================================
# Main test runner
# ============================================================================

fn main() raises:
    """Run all integration tests."""
    print("Running ExTensor integration tests...")

    # Chained operations
    print("  Testing chained operations...")
    test_chained_add_operations()
    test_mixed_arithmetic_operations()
    test_arithmetic_with_operator_overloading()
    test_complex_expression()

    # Creation + arithmetic
    print("  Testing creation + arithmetic patterns...")
    test_identity_matrix_operations()
    test_arange_arithmetic()
    test_linspace_operations()

    # Multiple dtypes
    print("  Testing dtype operations...")
    test_same_dtype_consistency()
    test_int_dtype_operations()

    # Multi-dimensional
    print("  Testing multi-dimensional operations...")
    test_2d_elementwise_operations()
    test_3d_operations()

    # ML patterns
    print("  Testing ML-like patterns...")
    test_linear_transformation_pattern()
    test_gradient_descent_update_pattern()
    test_batch_normalization_pattern()

    # Identity elements
    print("  Testing identity element behavior...")
    test_additive_identity()
    test_multiplicative_identity()
    test_multiplicative_zero()

    # Scalar operations
    print("  Testing scalar operations...")
    test_scalar_operations()

    # Large tensors
    print("  Testing large tensor operations...")
    test_large_tensor_operations()

    print("All integration tests completed!")
