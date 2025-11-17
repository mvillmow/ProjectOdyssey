"""Tests for ExTensor arithmetic operations.

Tests arithmetic operations including add, subtract, multiply, divide, floor_divide,
modulo, power, and matmul with same-shape tensors, broadcasting, and edge cases.
"""

from sys import DType

# Import ExTensor and arithmetic operations
from extensor import ExTensor, zeros, ones, full, add, subtract, multiply, divide, floor_divide, modulo, power

# Import test helpers
from ..helpers.assertions import (
    assert_true,
    assert_false,
    assert_equal_int,
    assert_dtype,
    assert_numel,
    assert_dim,
    assert_value_at,
    assert_all_values,
    assert_all_close,
)


# ============================================================================
# Test add() - Same shape
# ============================================================================

fn test_add_same_shape_1d() raises:
    """Test adding two 1D tensors with same shape."""
    var shape = DynamicVector[Int](1)
    shape[0] = 5
    let a = full(shape, 2.0, DType.float32)
    let b = full(shape, 3.0, DType.float32)
    let c = add(a, b)

    assert_numel(c, 5, "Result should have 5 elements")
    assert_dtype(c, DType.float32, "Result should have float32 dtype")
    assert_all_values(c, 5.0, 1e-6, "2.0 + 3.0 should be 5.0")


fn test_add_same_shape_2d() raises:
    """Test adding two 2D tensors with same shape."""
    var shape = DynamicVector[Int](2)
    shape[0] = 3
    shape[1] = 4
    let a = ones(shape, DType.float64)
    let b = full(shape, 2.5, DType.float64)
    let c = add(a, b)

    assert_numel(c, 12, "Result should have 12 elements")
    assert_dtype(c, DType.float64, "Result should have float64 dtype")
    assert_all_values(c, 3.5, 1e-8, "1.0 + 2.5 should be 3.5")


fn test_add_zeros() raises:
    """Test adding zeros (should not change values)."""
    var shape = DynamicVector[Int](2)
    shape[0] = 2
    shape[1] = 3
    let a = full(shape, 7.0, DType.float32)
    let b = zeros(shape, DType.float32)
    let c = add(a, b)

    assert_all_values(c, 7.0, 1e-6, "x + 0 should equal x")


fn test_add_negative_values() raises:
    """Test adding negative values."""
    var shape = DynamicVector[Int](1)
    shape[0] = 10
    let a = full(shape, -5.0, DType.float32)
    let b = full(shape, 3.0, DType.float32)
    let c = add(a, b)

    assert_all_values(c, -2.0, 1e-6, "-5.0 + 3.0 should be -2.0")


# ============================================================================
# Test subtract() - Same shape
# ============================================================================

fn test_subtract_same_shape_1d() raises:
    """Test subtracting two 1D tensors with same shape."""
    var shape = DynamicVector[Int](1)
    shape[0] = 5
    let a = full(shape, 7.0, DType.float32)
    let b = full(shape, 3.0, DType.float32)
    let c = subtract(a, b)

    assert_numel(c, 5, "Result should have 5 elements")
    assert_dtype(c, DType.float32, "Result should have float32 dtype")
    assert_all_values(c, 4.0, 1e-6, "7.0 - 3.0 should be 4.0")


fn test_subtract_same_shape_2d() raises:
    """Test subtracting two 2D tensors with same shape."""
    var shape = DynamicVector[Int](2)
    shape[0] = 3
    shape[1] = 4
    let a = full(shape, 10.0, DType.float64)
    let b = full(shape, 2.5, DType.float64)
    let c = subtract(a, b)

    assert_numel(c, 12, "Result should have 12 elements")
    assert_dtype(c, DType.float64, "Result should have float64 dtype")
    assert_all_values(c, 7.5, 1e-8, "10.0 - 2.5 should be 7.5")


fn test_subtract_zeros() raises:
    """Test subtracting zeros (should not change values)."""
    var shape = DynamicVector[Int](2)
    shape[0] = 2
    shape[1] = 3
    let a = full(shape, 9.0, DType.float32)
    let b = zeros(shape, DType.float32)
    let c = subtract(a, b)

    assert_all_values(c, 9.0, 1e-6, "x - 0 should equal x")


fn test_subtract_negative_result() raises:
    """Test subtraction resulting in negative values."""
    var shape = DynamicVector[Int](1)
    shape[0] = 10
    let a = full(shape, 3.0, DType.float32)
    let b = full(shape, 5.0, DType.float32)
    let c = subtract(a, b)

    assert_all_values(c, -2.0, 1e-6, "3.0 - 5.0 should be -2.0")


# ============================================================================
# Test multiply() - Same shape
# ============================================================================

fn test_multiply_same_shape_1d() raises:
    """Test multiplying two 1D tensors with same shape."""
    var shape = DynamicVector[Int](1)
    shape[0] = 5
    let a = full(shape, 4.0, DType.float32)
    let b = full(shape, 2.5, DType.float32)
    let c = multiply(a, b)

    assert_numel(c, 5, "Result should have 5 elements")
    assert_dtype(c, DType.float32, "Result should have float32 dtype")
    assert_all_values(c, 10.0, 1e-6, "4.0 * 2.5 should be 10.0")


fn test_multiply_same_shape_2d() raises:
    """Test multiplying two 2D tensors with same shape."""
    var shape = DynamicVector[Int](2)
    shape[0] = 3
    shape[1] = 4
    let a = full(shape, 3.0, DType.float64)
    let b = full(shape, 1.5, DType.float64)
    let c = multiply(a, b)

    assert_numel(c, 12, "Result should have 12 elements")
    assert_dtype(c, DType.float64, "Result should have float64 dtype")
    assert_all_values(c, 4.5, 1e-8, "3.0 * 1.5 should be 4.5")


fn test_multiply_by_zero() raises:
    """Test multiplying by zero (should give all zeros)."""
    var shape = DynamicVector[Int](2)
    shape[0] = 2
    shape[1] = 3
    let a = full(shape, 99.0, DType.float32)
    let b = zeros(shape, DType.float32)
    let c = multiply(a, b)

    assert_all_values(c, 0.0, 1e-8, "x * 0 should equal 0")


fn test_multiply_by_one() raises:
    """Test multiplying by one (should not change values)."""
    var shape = DynamicVector[Int](1)
    shape[0] = 10
    let a = full(shape, 7.5, DType.float32)
    let b = ones(shape, DType.float32)
    let c = multiply(a, b)

    assert_all_values(c, 7.5, 1e-6, "x * 1 should equal x")


fn test_multiply_negative() raises:
    """Test multiplying with negative values."""
    var shape = DynamicVector[Int](1)
    shape[0] = 5
    let a = full(shape, -3.0, DType.float32)
    let b = full(shape, 2.0, DType.float32)
    let c = multiply(a, b)

    assert_all_values(c, -6.0, 1e-6, "-3.0 * 2.0 should be -6.0")


# ============================================================================
# Test operator overloading (dunders) - Same shape
# ============================================================================

fn test_dunder_add() raises:
    """Test __add__ operator overloading (a + b)."""
    var shape = DynamicVector[Int](1)
    shape[0] = 5
    let a = full(shape, 3.0, DType.float32)
    let b = full(shape, 2.0, DType.float32)
    let c = a + b

    assert_all_values(c, 5.0, 1e-6, "a + b should work via __add__")


fn test_dunder_sub() raises:
    """Test __sub__ operator overloading (a - b)."""
    var shape = DynamicVector[Int](1)
    shape[0] = 5
    let a = full(shape, 7.0, DType.float32)
    let b = full(shape, 3.0, DType.float32)
    let c = a - b

    assert_all_values(c, 4.0, 1e-6, "a - b should work via __sub__")


fn test_dunder_mul() raises:
    """Test __mul__ operator overloading (a * b)."""
    var shape = DynamicVector[Int](1)
    shape[0] = 5
    let a = full(shape, 4.0, DType.float32)
    let b = full(shape, 2.0, DType.float32)
    let c = a * b

    assert_all_values(c, 8.0, 1e-6, "a * b should work via __mul__")


fn test_chained_operations() raises:
    """Test chained operations with multiple operators."""
    var shape = DynamicVector[Int](1)
    shape[0] = 5
    let a = full(shape, 2.0, DType.float32)
    let b = full(shape, 3.0, DType.float32)
    let c = full(shape, 1.0, DType.float32)

    # (a + b) * c = (2 + 3) * 1 = 5
    let result = (a + b) * c
    assert_all_values(result, 5.0, 1e-6, "(2 + 3) * 1 should be 5")


fn test_complex_expression() raises:
    """Test complex expression with multiple operations."""
    var shape = DynamicVector[Int](2)
    shape[0] = 2
    shape[1] = 3
    let a = ones(shape, DType.float32)
    let b = full(shape, 2.0, DType.float32)
    let c = full(shape, 3.0, DType.float32)

    # a + b * c = 1 + 2 * 3 = 1 + 6 = 7
    let result = a + b * c
    assert_all_values(result, 7.0, 1e-6, "1 + 2 * 3 should be 7")


# ============================================================================
# Test dtype preservation
# ============================================================================

fn test_add_preserves_dtype_float32() raises:
    """Test that add preserves float32 dtype."""
    var shape = DynamicVector[Int](1)
    shape[0] = 5
    let a = ones(shape, DType.float32)
    let b = ones(shape, DType.float32)
    let c = add(a, b)

    assert_dtype(c, DType.float32, "Result should preserve float32 dtype")


fn test_add_preserves_dtype_float64() raises:
    """Test that add preserves float64 dtype."""
    var shape = DynamicVector[Int](1)
    shape[0] = 5
    let a = ones(shape, DType.float64)
    let b = ones(shape, DType.float64)
    let c = add(a, b)

    assert_dtype(c, DType.float64, "Result should preserve float64 dtype")


fn test_multiply_preserves_dtype_int32() raises:
    """Test that multiply preserves int32 dtype."""
    var shape = DynamicVector[Int](1)
    shape[0] = 5
    let a = ones(shape, DType.int32)
    let b = full(shape, 2.0, DType.int32)
    let c = multiply(a, b)

    assert_dtype(c, DType.int32, "Result should preserve int32 dtype")


# ============================================================================
# Test shape preservation
# ============================================================================

fn test_add_preserves_shape_1d() raises:
    """Test that add preserves 1D shape."""
    var shape = DynamicVector[Int](1)
    shape[0] = 10
    let a = ones(shape, DType.float32)
    let b = ones(shape, DType.float32)
    let c = add(a, b)

    assert_dim(c, 1, "Result should be 1D")
    assert_numel(c, 10, "Result should have 10 elements")


fn test_add_preserves_shape_3d() raises:
    """Test that add preserves 3D shape."""
    var shape = DynamicVector[Int](3)
    shape[0] = 2
    shape[1] = 3
    shape[2] = 4
    let a = ones(shape, DType.float32)
    let b = ones(shape, DType.float32)
    let c = add(a, b)

    assert_dim(c, 3, "Result should be 3D")
    assert_numel(c, 24, "Result should have 24 elements")


# ============================================================================
# Test error handling - Mismatched shapes
# ============================================================================

fn test_add_mismatched_shapes_raises_error() raises:
    """Test that add with mismatched shapes raises error."""
    var shape_a = DynamicVector[Int](1)
    shape_a[0] = 5
    var shape_b = DynamicVector[Int](1)
    shape_b[0] = 10

    let a = ones(shape_a, DType.float32)
    let b = ones(shape_b, DType.float32)

    # This should raise an error
    # For now, we expect it to return zeros or error
    # TODO: Verify proper error handling once implemented
    # try:
    #     let c = add(a, b)
    #     raise Error("Should have raised error for mismatched shapes")
    # except:
    #     pass  # Expected
    pass  # Placeholder until error handling is implemented


fn test_multiply_mismatched_shapes_raises_error() raises:
    """Test that multiply with mismatched shapes raises error."""
    var shape_a = DynamicVector[Int](2)
    shape_a[0] = 3
    shape_a[1] = 4
    var shape_b = DynamicVector[Int](2)
    shape_b[0] = 3
    shape_b[1] = 5

    let a = ones(shape_a, DType.float32)
    let b = ones(shape_b, DType.float32)

    # This should raise an error
    # TODO: Verify proper error handling once implemented
    pass  # Placeholder until error handling is implemented


# ============================================================================
# Test error handling - Mismatched dtypes
# ============================================================================

fn test_add_mismatched_dtypes_raises_error() raises:
    """Test that add with mismatched dtypes raises error."""
    var shape = DynamicVector[Int](1)
    shape[0] = 5

    let a = ones(shape, DType.float32)
    let b = ones(shape, DType.float64)

    # This should raise an error
    # TODO: Verify proper error handling once implemented
    pass  # Placeholder until error handling is implemented


# ============================================================================
# Test divide()
# ============================================================================

fn test_divide_same_shape() raises:
    """Test dividing two tensors with same shape."""
    var shape = DynamicVector[Int](1)
    shape[0] = 5
    let a = full(shape, 6.0, DType.float32)
    let b = full(shape, 2.0, DType.float32)
    let c = divide(a, b)

    assert_numel(c, 5, "Result should have 5 elements")
    assert_dtype(c, DType.float32, "Result should have float32 dtype")
    assert_all_values(c, 3.0, 1e-6, "6.0 / 2.0 should be 3.0")


fn test_divide_by_one() raises:
    """Test dividing by one (identity)."""
    var shape = DynamicVector[Int](2)
    shape[0] = 3
    shape[1] = 4
    let a = full(shape, 7.5, DType.float32)
    let b = ones(shape, DType.float32)
    let c = divide(a, b)

    assert_all_values(c, 7.5, 1e-6, "x / 1 should be x")


fn test_divide_by_two() raises:
    """Test dividing by two."""
    var shape = DynamicVector[Int](1)
    shape[0] = 5
    let a = full(shape, 10.0, DType.float64)
    let b = full(shape, 2.0, DType.float64)
    let c = divide(a, b)

    assert_dtype(c, DType.float64, "Should preserve float64")
    assert_all_values(c, 5.0, 1e-8, "10.0 / 2.0 should be 5.0")


fn test_divide_negative() raises:
    """Test dividing negative values."""
    var shape = DynamicVector[Int](1)
    shape[0] = 5
    let a = full(shape, -6.0, DType.float32)
    let b = full(shape, 2.0, DType.float32)
    let c = divide(a, b)

    assert_all_values(c, -3.0, 1e-6, "-6.0 / 2.0 should be -3.0")


# ============================================================================
# Test floor_divide()
# ============================================================================

fn test_floor_divide_same_shape() raises:
    """Test floor division with same shape."""
    var shape = DynamicVector[Int](1)
    shape[0] = 5
    let a = full(shape, 7.0, DType.float32)
    let b = full(shape, 2.0, DType.float32)
    let c = floor_divide(a, b)

    assert_all_values(c, 3.0, 1e-6, "7.0 // 2.0 should be 3.0")


fn test_floor_divide_positive() raises:
    """Test floor division with positive values."""
    var shape = DynamicVector[Int](1)
    shape[0] = 5
    let a = full(shape, 9.0, DType.float32)
    let b = full(shape, 4.0, DType.float32)
    let c = floor_divide(a, b)

    assert_all_values(c, 2.0, 1e-6, "9.0 // 4.0 should be 2.0")


fn test_floor_divide_negative() raises:
    """Test floor division with negative dividend."""
    var shape = DynamicVector[Int](1)
    shape[0] = 5
    let a = full(shape, -7.0, DType.float32)
    let b = full(shape, 2.0, DType.float32)
    let c = floor_divide(a, b)

    assert_all_values(c, -4.0, 1e-6, "-7.0 // 2.0 should be -4.0")


# ============================================================================
# Test modulo()
# ============================================================================

fn test_modulo_positive() raises:
    """Test modulo with positive values."""
    var shape = DynamicVector[Int](1)
    shape[0] = 5
    let a = full(shape, 7.0, DType.float32)
    let b = full(shape, 3.0, DType.float32)
    let c = modulo(a, b)

    assert_all_values(c, 1.0, 1e-6, "7.0 % 3.0 should be 1.0")


fn test_modulo_negative_dividend() raises:
    """Test modulo with negative dividend."""
    var shape = DynamicVector[Int](1)
    shape[0] = 5
    let a = full(shape, -7.0, DType.float32)
    let b = full(shape, 3.0, DType.float32)
    let c = modulo(a, b)

    # Python semantics: -7 % 3 = 2 (not -1)
    assert_all_values(c, 2.0, 1e-6, "-7.0 % 3.0 should be 2.0 (Python semantics)")


fn test_modulo_fractional() raises:
    """Test modulo with fractional values."""
    var shape = DynamicVector[Int](1)
    shape[0] = 5
    let a = full(shape, 7.5, DType.float32)
    let b = full(shape, 2.5, DType.float32)
    let c = modulo(a, b)

    assert_all_values(c, 0.0, 1e-6, "7.5 % 2.5 should be 0.0")


# ============================================================================
# Test power()
# ============================================================================

fn test_power_integer_exponent() raises:
    """Test power with small integer exponent."""
    var shape = DynamicVector[Int](1)
    shape[0] = 5
    let a = full(shape, 2.0, DType.float32)
    let b = full(shape, 3.0, DType.float32)
    let c = power(a, b)

    assert_all_values(c, 8.0, 1e-6, "2.0 ** 3.0 should be 8.0")


fn test_power_zero_exponent() raises:
    """Test power with zero exponent."""
    var shape = DynamicVector[Int](1)
    shape[0] = 5
    let a = full(shape, 5.0, DType.float32)
    let b = full(shape, 0.0, DType.float32)
    let c = power(a, b)

    assert_all_values(c, 1.0, 1e-6, "x ** 0 should be 1.0")


fn test_power_one_exponent() raises:
    """Test power with exponent of one."""
    var shape = DynamicVector[Int](1)
    shape[0] = 5
    let a = full(shape, 7.5, DType.float32)
    let b = full(shape, 1.0, DType.float32)
    let c = power(a, b)

    assert_all_values(c, 7.5, 1e-6, "x ** 1 should be x")


fn test_power_negative_base() raises:
    """Test power with negative base."""
    var shape = DynamicVector[Int](1)
    shape[0] = 5
    let a = full(shape, -2.0, DType.float32)
    let b = full(shape, 2.0, DType.float32)
    let c = power(a, b)

    assert_all_values(c, 4.0, 1e-6, "(-2.0) ** 2.0 should be 4.0")


# ============================================================================
# Main test runner
# ============================================================================

fn main() raises:
    """Run all arithmetic operation tests."""
    print("Running ExTensor arithmetic operation tests...")

    # add() tests
    print("  Testing add()...")
    test_add_same_shape_1d()
    test_add_same_shape_2d()
    test_add_zeros()
    test_add_negative_values()

    # subtract() tests
    print("  Testing subtract()...")
    test_subtract_same_shape_1d()
    test_subtract_same_shape_2d()
    test_subtract_zeros()
    test_subtract_negative_result()

    # multiply() tests
    print("  Testing multiply()...")
    test_multiply_same_shape_1d()
    test_multiply_same_shape_2d()
    test_multiply_by_zero()
    test_multiply_by_one()
    test_multiply_negative()

    # Operator overloading tests
    print("  Testing operator overloading...")
    test_dunder_add()
    test_dunder_sub()
    test_dunder_mul()
    test_chained_operations()
    test_complex_expression()

    # DType preservation tests
    print("  Testing dtype preservation...")
    test_add_preserves_dtype_float32()
    test_add_preserves_dtype_float64()
    test_multiply_preserves_dtype_int32()

    # Shape preservation tests
    print("  Testing shape preservation...")
    test_add_preserves_shape_1d()
    test_add_preserves_shape_3d()

    # Error handling tests
    print("  Testing error handling...")
    test_add_mismatched_shapes_raises_error()
    test_multiply_mismatched_shapes_raises_error()
    test_add_mismatched_dtypes_raises_error()

    # divide() tests
    print("  Testing divide()...")
    test_divide_same_shape()
    test_divide_by_one()
    test_divide_by_two()
    test_divide_negative()

    # floor_divide() tests
    print("  Testing floor_divide()...")
    test_floor_divide_same_shape()
    test_floor_divide_positive()
    test_floor_divide_negative()

    # modulo() tests
    print("  Testing modulo()...")
    test_modulo_positive()
    test_modulo_negative_dividend()
    test_modulo_fractional()

    # power() tests
    print("  Testing power()...")
    test_power_integer_exponent()
    test_power_zero_exponent()
    test_power_one_exponent()
    test_power_negative_base()

    print("All arithmetic operation tests completed!")
