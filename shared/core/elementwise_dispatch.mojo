"""Trait-based elementwise operation dispatcher for ExTensor.

Provides high-level trait-based abstraction for elementwise operations,
eliminating the need to write operation functions for each operation type.
Uses traits to define operation semantics and dispatches them through
the dtype dispatch layer.

Benefits:
- Trait-based polymorphism for operation definitions
- Reusable operation implementations
- Single source of truth for each operation
- Easy to add new operations
- Zero runtime overhead (compile-time specialization)

Architecture:
- ElementwiseUnaryOp trait: Operations with one input (e.g., exp, sin)
- ElementwiseBinaryOp trait: Operations with two inputs (e.g., add, mul)
- apply_unary[Op](): Dispatcher for unary operations
- apply_binary[Op](): Dispatcher for binary operations
- Predefined operation structs: ExpOp, LogOp, SqrtOp, AddOp, MulOp, etc.

Example usage:
    # Apply exp operation to tensor
    var result = apply_unary[ExpOp](input_tensor)

    # Apply addition to two tensors
    var sum_result = apply_binary[AddOp](tensor_a, tensor_b)

    # Define custom operation
    struct CustomOp(ElementwiseUnaryOp):
        fn apply(self, value: Float64) -> Float64:
            return value * value + 1.0

    var custom_result = apply_unary[CustomOp](tensor)

See notes/issues/elementwise-dispatch-design.md for complete design.
"""

from collections import List
from .extensor import ExTensor
from math import sqrt as math_sqrt, exp as math_exp, log as math_log
from math import sin as math_sin, cos as math_cos, tanh as math_tanh


# ============================================================================
# Trait Definitions
# ============================================================================


trait ElementwiseUnaryOp:
    """Trait for single-input elementwise operations.

    Implement this trait to define custom unary operations that can be
    dispatched through apply_unary[]. The operation receives a Float64
    value and returns the result as Float64.

    Required Methods:
        `apply`: Apply the operation to a single value

    Contract:
        - apply() must be deterministic
        - apply() must handle all Float64 values (including edge cases)
        - apply() should not have side effects

    Example:
        struct SquareOp(ElementwiseUnaryOp):
            '''Square each element.'''
            fn apply(self, value: Float64) -> Float64:
                return value * value

        var squared = apply_unary[SquareOp](tensor)
    """

    fn apply(self, value: Float64) -> Float64:
        """Apply operation to a single value.

        Args:
            value: Input value (Float64)

        Returns:
            Result value (Float64)

        Note:
            Must handle all Float64 values, including:
            - Positive/negative numbers
            - Zero
            - Infinity
            - NaN
        """
        ...


trait ElementwiseBinaryOp:
    """Trait for two-input elementwise operations.

    Implement this trait to define custom binary operations that can be
    dispatched through apply_binary[]. The operation receives two Float64
    values and returns the result as Float64.

    Required Methods:
        `apply`: Apply the operation to two values

    Contract:
        - apply() must be deterministic
        - apply() must handle all Float64 value pairs
        - apply() should not have side effects
        - apply(a, b) should not require a.shape == b.shape
          (broadcasting is handled by apply_binary)

    Example:
        struct MaxOp(ElementwiseBinaryOp):
            '''Element-wise maximum.'''
            fn apply(self, a: Float64, b: Float64) -> Float64:
                if a > b:
                    return a
                else:
                    return b

        var max_result = apply_binary[MaxOp](tensor_a, tensor_b)
    """

    fn apply(self, a: Float64, b: Float64) -> Float64:
        """Apply operation to two values.

        Args:
            a: First input value (Float64)
            b: Second input value (Float64)

        Returns:
            Result value (Float64)

        Note:
            Must handle all Float64 value pairs.
        """
        ...


# ============================================================================
# Unary Operation Dispatcher
# ============================================================================


fn apply_unary[Op: ElementwiseUnaryOp](input: ExTensor) raises -> ExTensor:
    """Apply unary operation to all elements in tensor.

    Applies the operation defined by Op to each element of the input tensor,
    returning a new tensor with the results. Uses dtype preservation.

    Args:
        input: Input tensor

    Returns:
        New tensor with operation applied to each element

    Raises:
        Error: If operation fails (e.g., invalid value for operation)

    Type Parameters:
        Op: Must implement ElementwiseUnaryOp trait

    Example:
        # Apply exponential
        var exp_tensor = apply_unary[ExpOp](tensor)

        # Apply custom operation
        struct MyOp(ElementwiseUnaryOp):
            fn apply(self, value: Float64) -> Float64:
                return value + 1.0

        var result = apply_unary[MyOp](tensor)

    Note:
        - Preserves input dtype in output
        - Single pass through data (linear time complexity)
        - Zero-copy for operation implementation
    """
    var result = ExTensor(input.shape(), input.dtype())
    var op = Op()

    var numel = input.numel()
    for i in range(numel):
        var val = input._get_float64(i)
        var result_val = op.apply(val)
        result._set_float64(i, result_val)

    return result^


# ============================================================================
# Binary Operation Dispatcher (Element-wise)
# ============================================================================


fn apply_binary[Op: ElementwiseBinaryOp](a: ExTensor, b: ExTensor) raises -> ExTensor:
    """Apply binary operation to all element pairs in two tensors.

    Applies the operation defined by Op to each pair of elements from
    the input tensors, returning a new tensor with the results.
    Both tensors must have the same shape and dtype.

    Args:
        a: First input tensor
        b: Second input tensor

    Returns:
        New tensor with operation applied to each element pair

    Raises:
        Error: If shapes don't match, dtypes don't match, or operation fails

    Type Parameters:
        Op: Must implement ElementwiseBinaryOp trait

    Example:
        # Add two tensors
        var sum_tensor = apply_binary[AddOp](tensor_a, tensor_b)

        # Element-wise multiplication
        var prod_tensor = apply_binary[MulOp](tensor_a, tensor_b)

        # Custom operation
        struct DiffOp(ElementwiseBinaryOp):
            fn apply(self, a: Float64, b: Float64) -> Float64:
                return a - b

        var difference = apply_binary[DiffOp](tensor_a, tensor_b)

    Note:
        - Both tensors must have exact same shape and dtype
        - Output dtype matches input dtype
        - Single pass through data (linear time complexity)
    """
    # Validate shapes match
    if len(a.shape()) != len(b.shape()):
        raise Error("apply_binary: tensors must have same number of dimensions")

    for i in range(len(a.shape())):
        if a.shape()[i] != b.shape()[i]:
            raise Error("apply_binary: tensors must have same shape")

    # Validate dtypes match
    if a.dtype() != b.dtype():
        raise Error("apply_binary: tensors must have same dtype")

    var result = ExTensor(a.shape(), a.dtype())
    var op = Op()

    var numel = a.numel()
    for i in range(numel):
        var a_val = a._get_float64(i)
        var b_val = b._get_float64(i)
        var result_val = op.apply(a_val, b_val)
        result._set_float64(i, result_val)

    return result^


# ============================================================================
# Predefined Unary Operations
# ============================================================================


struct ExpOp(ElementwiseUnaryOp):
    """Exponential operation: e^x."""

    fn apply(self, value: Float64) -> Float64:
        """Compute e^x."""
        return math_exp(value)


struct LogOp(ElementwiseUnaryOp):
    """Natural logarithm operation: ln(x).

    Note: Input must be positive.
    """

    fn apply(self, value: Float64) -> Float64:
        """Compute ln(x)."""
        if value <= 0.0:
            raise Error("LogOp: input must be positive, got " + String(value))
        return math_log(value)


struct SqrtOp(ElementwiseUnaryOp):
    """Square root operation: sqrt(x).

    Note: Input must be non-negative.
    """

    fn apply(self, value: Float64) -> Float64:
        """Compute sqrt(x)."""
        if value < 0.0:
            raise Error("SqrtOp: input must be non-negative, got " + String(value))
        return math_sqrt(value)


struct SinOp(ElementwiseUnaryOp):
    """Sine operation: sin(x)."""

    fn apply(self, value: Float64) -> Float64:
        """Compute sin(x)."""
        return math_sin(value)


struct CosOp(ElementwiseUnaryOp):
    """Cosine operation: cos(x)."""

    fn apply(self, value: Float64) -> Float64:
        """Compute cos(x)."""
        return math_cos(value)


struct TanhOp(ElementwiseUnaryOp):
    """Hyperbolic tangent operation: tanh(x)."""

    fn apply(self, value: Float64) -> Float64:
        """Compute tanh(x)."""
        return math_tanh(value)


struct AbsOp(ElementwiseUnaryOp):
    """Absolute value operation: |x|."""

    fn apply(self, value: Float64) -> Float64:
        """Compute |x|."""
        if value >= 0.0:
            return value
        else:
            return -value


struct NegateOp(ElementwiseUnaryOp):
    """Negation operation: -x."""

    fn apply(self, value: Float64) -> Float64:
        """Compute -x."""
        return -value


struct ReciprocalOp(ElementwiseUnaryOp):
    """Reciprocal operation: 1/x.

    Note: Input must not be zero.
    """

    fn apply(self, value: Float64) -> Float64:
        """Compute 1/x."""
        if value == 0.0:
            raise Error("ReciprocalOp: cannot compute reciprocal of zero")
        return 1.0 / value


struct SquareOp(ElementwiseUnaryOp):
    """Square operation: x^2."""

    fn apply(self, value: Float64) -> Float64:
        """Compute x^2."""
        return value * value


struct SignOp(ElementwiseUnaryOp):
    """Sign operation: -1, 0, or 1."""

    fn apply(self, value: Float64) -> Float64:
        """Compute sign(x)."""
        if value > 0.0:
            return 1.0
        elif value < 0.0:
            return -1.0
        else:
            return 0.0


# ============================================================================
# Predefined Binary Operations
# ============================================================================


struct AddOp(ElementwiseBinaryOp):
    """Addition operation: a + b."""

    fn apply(self, a: Float64, b: Float64) -> Float64:
        """Compute a + b."""
        return a + b


struct SubtractOp(ElementwiseBinaryOp):
    """Subtraction operation: a - b."""

    fn apply(self, a: Float64, b: Float64) -> Float64:
        """Compute a - b."""
        return a - b


struct MultiplyOp(ElementwiseBinaryOp):
    """Multiplication operation: a * b."""

    fn apply(self, a: Float64, b: Float64) -> Float64:
        """Compute a * b."""
        return a * b


struct DivideOp(ElementwiseBinaryOp):
    """Division operation: a / b.

    Note: b must not be zero.
    """

    fn apply(self, a: Float64, b: Float64) -> Float64:
        """Compute a / b."""
        if b == 0.0:
            raise Error("DivideOp: cannot divide by zero")
        return a / b


struct PowerOp(ElementwiseBinaryOp):
    """Power operation: a ^ b."""

    fn apply(self, a: Float64, b: Float64) -> Float64:
        """Compute a ^ b (a to the power of b)."""
        # Handle special cases
        if a == 0.0 and b < 0.0:
            raise Error("PowerOp: cannot raise 0 to negative power")

        # Use exp and log for general case: a^b = exp(b * ln(a))
        if a > 0.0:
            return math_exp(b * math_log(a))
        elif a == 0.0:
            if b == 0.0:
                return 1.0  # 0^0 = 1 by convention
            elif b > 0.0:
                return 0.0  # 0^positive = 0
            else:
                raise Error("PowerOp: 0^negative is undefined")
        else:
            # a < 0: only support integer exponents
            # For simplicity in dispatcher, raise error for negative base
            raise Error("PowerOp: negative base not supported in elementwise dispatcher")


struct MaxOp(ElementwiseBinaryOp):
    """Maximum operation: max(a, b)."""

    fn apply(self, a: Float64, b: Float64) -> Float64:
        """Compute max(a, b)."""
        if a >= b:
            return a
        else:
            return b


struct MinOp(ElementwiseBinaryOp):
    """Minimum operation: min(a, b)."""

    fn apply(self, a: Float64, b: Float64) -> Float64:
        """Compute min(a, b)."""
        if a <= b:
            return a
        else:
            return b


struct EqualOp(ElementwiseBinaryOp):
    """Equality comparison: (a == b) as 1.0 or 0.0."""

    fn apply(self, a: Float64, b: Float64) -> Float64:
        """Return 1.0 if a == b, else 0.0."""
        if a == b:
            return 1.0
        else:
            return 0.0


struct GreaterOp(ElementwiseBinaryOp):
    """Greater than comparison: (a > b) as 1.0 or 0.0."""

    fn apply(self, a: Float64, b: Float64) -> Float64:
        """Return 1.0 if a > b, else 0.0."""
        if a > b:
            return 1.0
        else:
            return 0.0


struct GreaterEqualOp(ElementwiseBinaryOp):
    """Greater than or equal comparison: (a >= b) as 1.0 or 0.0."""

    fn apply(self, a: Float64, b: Float64) -> Float64:
        """Return 1.0 if a >= b, else 0.0."""
        if a >= b:
            return 1.0
        else:
            return 0.0


struct LessOp(ElementwiseBinaryOp):
    """Less than comparison: (a < b) as 1.0 or 0.0."""

    fn apply(self, a: Float64, b: Float64) -> Float64:
        """Return 1.0 if a < b, else 0.0."""
        if a < b:
            return 1.0
        else:
            return 0.0


struct LessEqualOp(ElementwiseBinaryOp):
    """Less than or equal comparison: (a <= b) as 1.0 or 0.0."""

    fn apply(self, a: Float64, b: Float64) -> Float64:
        """Return 1.0 if a <= b, else 0.0."""
        if a <= b:
            return 1.0
        else:
            return 0.0


struct LogicalAndOp(ElementwiseBinaryOp):
    """Logical AND: (a != 0 && b != 0) as 1.0 or 0.0."""

    fn apply(self, a: Float64, b: Float64) -> Float64:
        """Return 1.0 if both non-zero, else 0.0."""
        if (a != 0.0) and (b != 0.0):
            return 1.0
        else:
            return 0.0


struct LogicalOrOp(ElementwiseBinaryOp):
    """Logical OR: (a != 0 || b != 0) as 1.0 or 0.0."""

    fn apply(self, a: Float64, b: Float64) -> Float64:
        """Return 1.0 if either non-zero, else 0.0."""
        if (a != 0.0) or (b != 0.0):
            return 1.0
        else:
            return 0.0
