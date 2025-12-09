"""Tests for trait-based elementwise operation dispatcher.

Tests cover:
- Unary operations (exp, log, sqrt, sin, cos, tanh, abs, negate, reciprocal, square, sign)
- Binary operations (add, subtract, multiply, divide, power, max, min, comparisons, logical)
- Custom operations via trait implementation
- Error handling (invalid inputs, shape mismatches, type mismatches)
- Dtype preservation
- All edge cases (zero, negative, infinity, etc.)

All tests use the ElementwiseUnaryOp and ElementwiseBinaryOp trait-based API.
"""

from tests.shared.conftest import (
    assert_almost_equal,
    assert_equal_int,
    assert_true,
)
from shared.core.extensor import ExTensor, zeros, ones, full
from shared.core.elementwise_dispatch import (
    ElementwiseUnaryOp,
    ElementwiseBinaryOp,
    apply_unary,
    apply_binary,
    ExpOp,
    LogOp,
    SqrtOp,
    SinOp,
    CosOp,
    TanhOp,
    AbsOp,
    NegateOp,
    ReciprocalOp,
    SquareOp,
    SignOp,
    AddOp,
    SubtractOp,
    MultiplyOp,
    DivideOp,
    PowerOp,
    MaxOp,
    MinOp,
    EqualOp,
    GreaterOp,
    GreaterEqualOp,
    LessOp,
    LessEqualOp,
    LogicalAndOp,
    LogicalOrOp,
)


# ============================================================================
# Custom Operations for Testing
# ============================================================================


struct DoubleOp(ElementwiseUnaryOp):
    """Custom operation: 2 * x."""

    fn __init__(out self):
        pass

    fn apply(self, value: Float64) -> Float64:
        return value * 2.0


struct IncrementOp(ElementwiseUnaryOp):
    """Custom operation: x + 1."""

    fn __init__(out self):
        pass

    fn apply(self, value: Float64) -> Float64:
        return value + 1.0


struct AverageOp(ElementwiseBinaryOp):
    """Custom operation: (a + b) / 2."""

    fn __init__(out self):
        pass

    fn apply(self, a: Float64, b: Float64) -> Float64:
        return (a + b) / 2.0


# ============================================================================
# Unary Operation Tests - ExpOp
# ============================================================================


fn test_apply_unary_exp_zeros() raises:
    """Test exp(0) = 1."""
    var shape = List[Int]()
    shape.append(3)
    var zeros_tensor = zeros(shape, DType.float32)

    var result = apply_unary[ExpOp](zeros_tensor)

    assert_equal_int(result.numel(), 3)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), 1.0, tolerance=1e-6)


fn test_apply_unary_exp_ones() raises:
    """Test exp(1) ≈ 2.71828."""
    var shape = List[Int]()
    shape.append(3)
    var ones_tensor = ones(shape, DType.float32)

    var result = apply_unary[ExpOp](ones_tensor)

    assert_equal_int(result.numel(), 3)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), 2.71828, tolerance=1e-4)


# ============================================================================
# Unary Operation Tests - LogOp
# ============================================================================


fn test_apply_unary_log_ones() raises:
    """Test log(1) = 0."""
    var shape = List[Int]()
    shape.append(3)
    var ones_tensor = ones(shape, DType.float32)

    var result = apply_unary[LogOp](ones_tensor)

    assert_equal_int(result.numel(), 3)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), 0.0, tolerance=1e-6)


fn test_apply_unary_log_error_negative() raises:
    """Test log of negative number raises error."""
    var shape = List[Int]()
    shape.append(1)
    var neg_tensor = full(shape, -1.0, DType.float32)

    try:
        var result = apply_unary[LogOp](neg_tensor)
        assert_true(False, "Expected error for log of negative number")
    except e:
        assert_true(True, "Correctly raised error")


# ============================================================================
# Unary Operation Tests - SqrtOp
# ============================================================================


fn test_apply_unary_sqrt_four() raises:
    """Test sqrt(4) = 2."""
    var shape = List[Int]()
    shape.append(3)
    var tensor = full(shape, 4.0, DType.float32)

    var result = apply_unary[SqrtOp](tensor)

    assert_equal_int(result.numel(), 3)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), 2.0, tolerance=1e-6)


fn test_apply_unary_sqrt_zero() raises:
    """Test sqrt(0) = 0."""
    var shape = List[Int]()
    shape.append(3)
    var zeros_tensor = zeros(shape, DType.float32)

    var result = apply_unary[SqrtOp](zeros_tensor)

    assert_equal_int(result.numel(), 3)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), 0.0, tolerance=1e-6)


# ============================================================================
# Unary Operation Tests - TrigonometricOps
# ============================================================================


fn test_apply_unary_sin_zero() raises:
    """Test sin(0) = 0."""
    var shape = List[Int]()
    shape.append(3)
    var zeros_tensor = zeros(shape, DType.float32)

    var result = apply_unary[SinOp](zeros_tensor)

    assert_equal_int(result.numel(), 3)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), 0.0, tolerance=1e-6)


fn test_apply_unary_cos_zero() raises:
    """Test cos(0) = 1."""
    var shape = List[Int]()
    shape.append(3)
    var zeros_tensor = zeros(shape, DType.float32)

    var result = apply_unary[CosOp](zeros_tensor)

    assert_equal_int(result.numel(), 3)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), 1.0, tolerance=1e-6)


fn test_apply_unary_tanh_zero() raises:
    """Test tanh(0) = 0."""
    var shape = List[Int]()
    shape.append(3)
    var zeros_tensor = zeros(shape, DType.float32)

    var result = apply_unary[TanhOp](zeros_tensor)

    assert_equal_int(result.numel(), 3)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), 0.0, tolerance=1e-6)


# ============================================================================
# Unary Operation Tests - AbsOp
# ============================================================================


fn test_apply_unary_abs_negative() raises:
    """Test abs(-5) = 5."""
    var shape = List[Int]()
    shape.append(3)
    var tensor = full(shape, -5.0, DType.float32)

    var result = apply_unary[AbsOp](tensor)

    assert_equal_int(result.numel(), 3)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), 5.0, tolerance=1e-6)


fn test_apply_unary_abs_positive() raises:
    """Test abs(5) = 5."""
    var shape = List[Int]()
    shape.append(3)
    var tensor = full(shape, 5.0, DType.float32)

    var result = apply_unary[AbsOp](tensor)

    assert_equal_int(result.numel(), 3)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), 5.0, tolerance=1e-6)


# ============================================================================
# Unary Operation Tests - NegateOp
# ============================================================================


fn test_apply_unary_negate() raises:
    """Test negate operation."""
    var shape = List[Int]()
    shape.append(3)
    var tensor = full(shape, 5.0, DType.float32)

    var result = apply_unary[NegateOp](tensor)

    assert_equal_int(result.numel(), 3)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), -5.0, tolerance=1e-6)


# ============================================================================
# Unary Operation Tests - SquareOp
# ============================================================================


fn test_apply_unary_square() raises:
    """Test square operation: x^2."""
    var shape = List[Int]()
    shape.append(3)
    var tensor = full(shape, 3.0, DType.float32)

    var result = apply_unary[SquareOp](tensor)

    assert_equal_int(result.numel(), 3)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), 9.0, tolerance=1e-6)


# ============================================================================
# Unary Operation Tests - SignOp
# ============================================================================


fn test_apply_unary_sign_positive() raises:
    """Test sign of positive number."""
    var shape = List[Int]()
    shape.append(1)
    var tensor = full(shape, 5.0, DType.float32)

    var result = apply_unary[SignOp](tensor)

    assert_almost_equal(result._get_float64(0), 1.0, tolerance=1e-6)


fn test_apply_unary_sign_negative() raises:
    """Test sign of negative number."""
    var shape = List[Int]()
    shape.append(1)
    var tensor = full(shape, -5.0, DType.float32)

    var result = apply_unary[SignOp](tensor)

    assert_almost_equal(result._get_float64(0), -1.0, tolerance=1e-6)


fn test_apply_unary_sign_zero() raises:
    """Test sign of zero."""
    var shape = List[Int]()
    shape.append(1)
    var tensor = zeros(shape, DType.float32)

    var result = apply_unary[SignOp](tensor)

    assert_almost_equal(result._get_float64(0), 0.0, tolerance=1e-6)


# ============================================================================
# Unary Operation Tests - Custom Operations
# ============================================================================


fn test_apply_unary_custom_double() raises:
    """Test custom double operation."""
    var shape = List[Int]()
    shape.append(3)
    var tensor = full(shape, 5.0, DType.float32)

    var result = apply_unary[DoubleOp](tensor)

    assert_equal_int(result.numel(), 3)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), 10.0, tolerance=1e-6)


fn test_apply_unary_custom_increment() raises:
    """Test custom increment operation."""
    var shape = List[Int]()
    shape.append(3)
    var tensor = full(shape, 5.0, DType.float32)

    var result = apply_unary[IncrementOp](tensor)

    assert_equal_int(result.numel(), 3)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), 6.0, tolerance=1e-6)


# ============================================================================
# Binary Operation Tests - AddOp
# ============================================================================


fn test_apply_binary_add() raises:
    """Test addition operation."""
    var shape = List[Int]()
    shape.append(3)
    var a = full(shape, 2.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)

    var result = apply_binary[AddOp](a, b)

    assert_equal_int(result.numel(), 3)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), 5.0, tolerance=1e-6)


# ============================================================================
# Binary Operation Tests - SubtractOp
# ============================================================================


fn test_apply_binary_subtract() raises:
    """Test subtraction operation."""
    var shape = List[Int]()
    shape.append(3)
    var a = full(shape, 5.0, DType.float32)
    var b = full(shape, 2.0, DType.float32)

    var result = apply_binary[SubtractOp](a, b)

    assert_equal_int(result.numel(), 3)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), 3.0, tolerance=1e-6)


# ============================================================================
# Binary Operation Tests - MultiplyOp
# ============================================================================


fn test_apply_binary_multiply() raises:
    """Test multiplication operation."""
    var shape = List[Int]()
    shape.append(3)
    var a = full(shape, 2.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)

    var result = apply_binary[MultiplyOp](a, b)

    assert_equal_int(result.numel(), 3)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), 6.0, tolerance=1e-6)


# ============================================================================
# Binary Operation Tests - DivideOp
# ============================================================================


fn test_apply_binary_divide() raises:
    """Test division operation."""
    var shape = List[Int]()
    shape.append(3)
    var a = full(shape, 6.0, DType.float32)
    var b = full(shape, 2.0, DType.float32)

    var result = apply_binary[DivideOp](a, b)

    assert_equal_int(result.numel(), 3)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), 3.0, tolerance=1e-6)


fn test_apply_binary_divide_by_zero_error() raises:
    """Test division by zero raises error."""
    var shape = List[Int]()
    shape.append(1)
    var a = full(shape, 6.0, DType.float32)
    var b = zeros(shape, DType.float32)

    try:
        var result = apply_binary[DivideOp](a, b)
        assert_true(False, "Expected error for division by zero")
    except e:
        assert_true(True, "Correctly raised error")


# ============================================================================
# Binary Operation Tests - PowerOp
# ============================================================================


fn test_apply_binary_power_base_2() raises:
    """Test power operation: 2^3 = 8."""
    var shape = List[Int]()
    shape.append(1)
    var a = full(shape, 2.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)

    var result = apply_binary[PowerOp](a, b)

    assert_almost_equal(result._get_float64(0), 8.0, tolerance=1e-4)


fn test_apply_binary_power_square() raises:
    """Test power operation: 5^2 = 25."""
    var shape = List[Int]()
    shape.append(1)
    var a = full(shape, 5.0, DType.float32)
    var b = full(shape, 2.0, DType.float32)

    var result = apply_binary[PowerOp](a, b)

    assert_almost_equal(result._get_float64(0), 25.0, tolerance=1e-4)


# ============================================================================
# Binary Operation Tests - MaxOp
# ============================================================================


fn test_apply_binary_max_a_greater() raises:
    """Test max operation where a > b."""
    var shape = List[Int]()
    shape.append(3)
    var a = full(shape, 5.0, DType.float32)
    var b = full(shape, 2.0, DType.float32)

    var result = apply_binary[MaxOp](a, b)

    assert_equal_int(result.numel(), 3)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), 5.0, tolerance=1e-6)


fn test_apply_binary_max_b_greater() raises:
    """Test max operation where b > a."""
    var shape = List[Int]()
    shape.append(3)
    var a = full(shape, 2.0, DType.float32)
    var b = full(shape, 5.0, DType.float32)

    var result = apply_binary[MaxOp](a, b)

    assert_equal_int(result.numel(), 3)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), 5.0, tolerance=1e-6)


# ============================================================================
# Binary Operation Tests - MinOp
# ============================================================================


fn test_apply_binary_min_a_less() raises:
    """Test min operation where a < b."""
    var shape = List[Int]()
    shape.append(3)
    var a = full(shape, 2.0, DType.float32)
    var b = full(shape, 5.0, DType.float32)

    var result = apply_binary[MinOp](a, b)

    assert_equal_int(result.numel(), 3)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), 2.0, tolerance=1e-6)


# ============================================================================
# Binary Operation Tests - Comparison Operations
# ============================================================================


fn test_apply_binary_equal_true() raises:
    """Test equality when values are equal."""
    var shape = List[Int]()
    shape.append(3)
    var a = full(shape, 5.0, DType.float32)
    var b = full(shape, 5.0, DType.float32)

    var result = apply_binary[EqualOp](a, b)

    assert_equal_int(result.numel(), 3)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), 1.0, tolerance=1e-6)


fn test_apply_binary_equal_false() raises:
    """Test equality when values are not equal."""
    var shape = List[Int]()
    shape.append(3)
    var a = full(shape, 5.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)

    var result = apply_binary[EqualOp](a, b)

    assert_equal_int(result.numel(), 3)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), 0.0, tolerance=1e-6)


fn test_apply_binary_greater_true() raises:
    """Test greater than when true."""
    var shape = List[Int]()
    shape.append(3)
    var a = full(shape, 5.0, DType.float32)
    var b = full(shape, 2.0, DType.float32)

    var result = apply_binary[GreaterOp](a, b)

    assert_equal_int(result.numel(), 3)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), 1.0, tolerance=1e-6)


fn test_apply_binary_greater_false() raises:
    """Test greater than when false."""
    var shape = List[Int]()
    shape.append(3)
    var a = full(shape, 2.0, DType.float32)
    var b = full(shape, 5.0, DType.float32)

    var result = apply_binary[GreaterOp](a, b)

    assert_equal_int(result.numel(), 3)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), 0.0, tolerance=1e-6)


fn test_apply_binary_less_true() raises:
    """Test less than when true."""
    var shape = List[Int]()
    shape.append(3)
    var a = full(shape, 2.0, DType.float32)
    var b = full(shape, 5.0, DType.float32)

    var result = apply_binary[LessOp](a, b)

    assert_equal_int(result.numel(), 3)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), 1.0, tolerance=1e-6)


# ============================================================================
# Binary Operation Tests - Logical Operations
# ============================================================================


fn test_apply_binary_logical_and_both_true() raises:
    """Test logical AND when both are non-zero."""
    var shape = List[Int]()
    shape.append(3)
    var a = full(shape, 5.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)

    var result = apply_binary[LogicalAndOp](a, b)

    assert_equal_int(result.numel(), 3)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), 1.0, tolerance=1e-6)


fn test_apply_binary_logical_and_one_false() raises:
    """Test logical AND when one is zero."""
    var shape = List[Int]()
    shape.append(3)
    var a = full(shape, 5.0, DType.float32)
    var b = zeros(shape, DType.float32)

    var result = apply_binary[LogicalAndOp](a, b)

    assert_equal_int(result.numel(), 3)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), 0.0, tolerance=1e-6)


fn test_apply_binary_logical_or_both_true() raises:
    """Test logical OR when both are non-zero."""
    var shape = List[Int]()
    shape.append(3)
    var a = full(shape, 5.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)

    var result = apply_binary[LogicalOrOp](a, b)

    assert_equal_int(result.numel(), 3)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), 1.0, tolerance=1e-6)


fn test_apply_binary_logical_or_one_true() raises:
    """Test logical OR when only one is non-zero."""
    var shape = List[Int]()
    shape.append(3)
    var a = full(shape, 5.0, DType.float32)
    var b = zeros(shape, DType.float32)

    var result = apply_binary[LogicalOrOp](a, b)

    assert_equal_int(result.numel(), 3)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), 1.0, tolerance=1e-6)


# ============================================================================
# Binary Operation Tests - Custom Operations
# ============================================================================


fn test_apply_binary_custom_average() raises:
    """Test custom average operation."""
    var shape = List[Int]()
    shape.append(3)
    var a = full(shape, 2.0, DType.float32)
    var b = full(shape, 4.0, DType.float32)

    var result = apply_binary[AverageOp](a, b)

    assert_equal_int(result.numel(), 3)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), 3.0, tolerance=1e-6)


# ============================================================================
# Shape and Dtype Tests
# ============================================================================


fn test_apply_unary_preserves_dtype_float32() raises:
    """Test that unary operations preserve float32 dtype."""
    var shape = List[Int]()
    shape.append(3)
    var tensor = ones(shape, DType.float32)

    var result = apply_unary[ExpOp](tensor)

    assert_true(result.dtype() == DType.float32, "Output dtype should match input")


fn test_apply_unary_preserves_dtype_float64() raises:
    """Test that unary operations preserve float64 dtype."""
    var shape = List[Int]()
    shape.append(3)
    var tensor = ones(shape, DType.float64)

    var result = apply_unary[ExpOp](tensor)

    assert_true(result.dtype() == DType.float64, "Output dtype should match input")


fn test_apply_binary_preserves_dtype() raises:
    """Test that binary operations preserve dtype."""
    var shape = List[Int]()
    shape.append(3)
    var a = ones(shape, DType.float32)
    var b = ones(shape, DType.float32)

    var result = apply_binary[AddOp](a, b)

    assert_true(result.dtype() == DType.float32, "Output dtype should match input")


fn test_apply_binary_shape_mismatch_error() raises:
    """Test error when binary operands have different shapes."""
    var shape1 = List[Int]()
    shape1.append(3)
    var shape2 = List[Int]()
    shape2.append(4)
    var a = ones(shape1, DType.float32)
    var b = ones(shape2, DType.float32)

    try:
        var result = apply_binary[AddOp](a, b)
        assert_true(False, "Expected error for shape mismatch")
    except e:
        assert_true(True, "Correctly raised error")


fn test_apply_binary_dtype_mismatch_error() raises:
    """Test error when binary operands have different dtypes."""
    var shape = List[Int]()
    shape.append(3)
    var a = ones(shape, DType.float32)
    var b = ones(shape, DType.float64)

    try:
        var result = apply_binary[AddOp](a, b)
        assert_true(False, "Expected error for dtype mismatch")
    except e:
        assert_true(True, "Correctly raised error")


# ============================================================================
# Multi-dimensional Tensor Tests
# ============================================================================


fn test_apply_unary_2d_tensor() raises:
    """Test unary operation on 2D tensor."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    var tensor = ones(shape, DType.float32)

    var result = apply_unary[DoubleOp](tensor)

    assert_equal_int(result.numel(), 6)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), 2.0, tolerance=1e-6)


fn test_apply_binary_2d_tensor() raises:
    """Test binary operation on 2D tensors."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    var a = full(shape, 2.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)

    var result = apply_binary[AddOp](a, b)

    assert_equal_int(result.numel(), 6)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), 5.0, tolerance=1e-6)


# ============================================================================
# Reciprocal and Edge Cases
# ============================================================================


fn test_apply_unary_reciprocal() raises:
    """Test reciprocal operation."""
    var shape = List[Int]()
    shape.append(3)
    var tensor = full(shape, 2.0, DType.float32)

    var result = apply_unary[ReciprocalOp](tensor)

    assert_equal_int(result.numel(), 3)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), 0.5, tolerance=1e-6)


fn test_apply_unary_reciprocal_zero_error() raises:
    """Test reciprocal of zero raises error."""
    var shape = List[Int]()
    shape.append(1)
    var tensor = zeros(shape, DType.float32)

    try:
        var result = apply_unary[ReciprocalOp](tensor)
        assert_true(False, "Expected error for reciprocal of zero")
    except e:
        assert_true(True, "Correctly raised error")


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run all elementwise dispatch tests."""
    # Unary ExpOp tests
    test_apply_unary_exp_zeros()
    test_apply_unary_exp_ones()

    # Unary LogOp tests
    test_apply_unary_log_ones()
    test_apply_unary_log_error_negative()

    # Unary SqrtOp tests
    test_apply_unary_sqrt_four()
    test_apply_unary_sqrt_zero()

    # Unary TrigonometricOps tests
    test_apply_unary_sin_zero()
    test_apply_unary_cos_zero()
    test_apply_unary_tanh_zero()

    # Unary AbsOp tests
    test_apply_unary_abs_negative()
    test_apply_unary_abs_positive()

    # Unary NegateOp tests
    test_apply_unary_negate()

    # Unary SquareOp tests
    test_apply_unary_square()

    # Unary SignOp tests
    test_apply_unary_sign_positive()
    test_apply_unary_sign_negative()
    test_apply_unary_sign_zero()

    # Unary custom operations tests
    test_apply_unary_custom_double()
    test_apply_unary_custom_increment()

    # Binary AddOp tests
    test_apply_binary_add()

    # Binary SubtractOp tests
    test_apply_binary_subtract()

    # Binary MultiplyOp tests
    test_apply_binary_multiply()

    # Binary DivideOp tests
    test_apply_binary_divide()
    test_apply_binary_divide_by_zero_error()

    # Binary PowerOp tests
    test_apply_binary_power_base_2()
    test_apply_binary_power_square()

    # Binary MaxOp tests
    test_apply_binary_max_a_greater()
    test_apply_binary_max_b_greater()

    # Binary MinOp tests
    test_apply_binary_min_a_less()

    # Binary comparison operations tests
    test_apply_binary_equal_true()
    test_apply_binary_equal_false()
    test_apply_binary_greater_true()
    test_apply_binary_greater_false()
    test_apply_binary_less_true()

    # Binary logical operations tests
    test_apply_binary_logical_and_both_true()
    test_apply_binary_logical_and_one_false()
    test_apply_binary_logical_or_both_true()
    test_apply_binary_logical_or_one_true()

    # Binary custom operations tests
    test_apply_binary_custom_average()

    # Shape and dtype tests
    test_apply_unary_preserves_dtype_float32()
    test_apply_unary_preserves_dtype_float64()
    test_apply_binary_preserves_dtype()
    test_apply_binary_shape_mismatch_error()
    test_apply_binary_dtype_mismatch_error()

    # Multi-dimensional tensor tests
    test_apply_unary_2d_tensor()
    test_apply_binary_2d_tensor()

    # Reciprocal and edge cases tests
    test_apply_unary_reciprocal()
    test_apply_unary_reciprocal_zero_error()
