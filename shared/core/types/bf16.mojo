"""BF16 (BFloat16 / Brain Floating Point) data type implementation.

This module implements BFloat16 format:
- 1 sign bit
- 8 exponent bits (bias = 127)
- 7 mantissa bits

BFloat16 has the same exponent range as Float32 but with reduced precision.
It is widely used for deep learning training due to its dynamic range matching Float32.
Supported range: approximately ±3.4e38 (same as Float32).

Key properties:
- Same exponent range as Float32 (no overflow issues during training)
- Less precision than Float16 (7 vs 10 mantissa bits)
- Commonly used in TPUs and modern GPU training
- Simple conversion: BF16 is the upper 16 bits of Float32

Example:
    ```mojo
    from shared.core.types.bf16 import BF16

    var x = BF16.from_float32(3.14159)
    var y = x.to_float32()
    print(x)  # BF16(3.140625)
    ```
"""

from memory import bitcast
from math import isnan, isinf


@register_passable("trivial")
struct BF16(Representable, Stringable):
    """16-bit brain floating point number.

    Memory layout (2 bytes):
    - Bit 15: Sign bit
    - Bits 14-7: Exponent (8 bits, bias = 127)
    - Bits 6-0: Mantissa (7 bits)

    Special values:
    - Zero: exp=0, mantissa=0
    - NaN: exp=255, mantissa!=0
    - Inf: exp=255, mantissa=0

    BFloat16 shares the same exponent range as Float32, making it ideal for
    training deep neural networks where dynamic range is more important than
    precision.

    Attributes:
        value: UInt16 storage for BF16 bit representation.
    """

    var value: UInt16

    # ========================================================================
    # Constructors
    # ========================================================================

    fn __init__(out self):
        """Initialize BF16 to zero."""
        self.value = 0

    fn __init__(out self, value: UInt16):
        """Initialize BF16 from raw UInt16 bits.

        Args:
            value: Raw 16-bit representation.
        """
        self.value = value

    # ========================================================================
    # Conversion from Float32
    # ========================================================================

    @staticmethod
    @always_inline
    fn from_float32(x: Float32) -> Self:
        """Convert Float32 to BF16 using round-to-nearest-even.

        Uses IEEE 754 round-to-nearest-even (RNE) for proper rounding.
        This is the recommended conversion method.

        Args:
            x: Float32 value to convert.

        Returns:
            BF16 representation.

        Example:
            ```mojo
            var bf16 = BF16.from_float32(3.14159)
            ```
        """
        # Handle special cases
        if isnan(x):
            return BF16._nan()
        if isinf(x):
            return BF16._inf() if x > 0 else BF16._neg_inf()

        # Get bit representation of Float32 using SIMD bitcast
        var bits32 = bitcast[DType.uint32, 1](SIMD[DType.float32, 1](x))[0]

        # Extract components from Float32 (32 bits)
        # Float32: [sign:1][exponent:8][mantissa:23]
        var sign_bit = (bits32 >> 31) & 0x1
        var exponent = (bits32 >> 23) & 0xFF
        var mantissa23 = bits32 & 0x7FFFFF

        # Round to nearest even (RNE)
        # BF16 keeps bits 31-16, so bit 15 is the rounding bit, bits 14-0 are sticky
        var rounding_bit = (bits32 >> 15) & 0x1
        var sticky_bits = bits32 & 0x7FFF

        # Extract top 7 bits of mantissa for BF16
        var mantissa7 = (mantissa23 >> 16) & 0x7F

        # Apply rounding: round up if rounding bit is 1 AND
        # (sticky bits != 0 OR mantissa7 is odd - for round-to-even)
        if rounding_bit == 1:
            if sticky_bits != 0 or (mantissa7 & 0x1) == 1:
                mantissa7 += 1
                # Handle mantissa overflow
                if mantissa7 > 0x7F:
                    mantissa7 = 0
                    exponent += 1
                    # Handle exponent overflow (infinity)
                    if exponent > 0xFF:
                        return BF16._inf() if sign_bit == 0 else BF16._neg_inf()

        # Combine into BF16 format: [sign:1][exponent:8][mantissa:7]
        var bits16 = UInt16((sign_bit << 15) | (exponent << 7) | mantissa7)

        return BF16(bits16)

    @staticmethod
    @always_inline
    fn from_float32_truncate(x: Float32) -> Self:
        """Convert Float32 to BF16 using simple truncation.

        Faster than rounding but less accurate. Use only when performance
        is critical and slight accuracy loss is acceptable.

        Args:
            x: Float32 value to convert.

        Returns:
            BF16 representation.

        Example:
            ```mojo
            var bf16 = BF16.from_float32_truncate(3.14159)
            ```
        """
        # Get bit representation using SIMD bitcast
        var bits32 = bitcast[DType.uint32, 1](SIMD[DType.float32, 1](x))[0]

        # Simply take upper 16 bits (truncate lower 16)
        var bits16 = UInt16(bits32 >> 16)

        return BF16(bits16)

    # ========================================================================
    # Conversion to Float32/Float64
    # ========================================================================

    @always_inline
    fn to_float32(self) -> Float32:
        """Convert BF16 to Float32.

        Conversion is exact (no rounding needed) since we're expanding
        mantissa bits (7 -> 23) by zero-padding.

        Returns:
            Float32 representation.

        Example:
            ```mojo
            var f32 = bf16.to_float32()
            ```
        """
        # BF16 to Float32 is simple: extend to 32 bits by shifting left 16
        # BF16: [sign:1][exponent:8][mantissa:7]
        # FP32: [sign:1][exponent:8][mantissa:23]
        #
        # We just zero-pad the lower 16 bits of mantissa
        var bits32 = UInt32(self.value) << 16

        # Convert bits to Float32 using SIMD bitcast
        var result = bitcast[DType.float32, 1](SIMD[DType.uint32, 1](bits32))[0]

        return result

    fn to_float64(self) -> Float64:
        """Convert BF16 to Float64.

        Goes through Float32 conversion first.

        Returns:
            Float64 representation.
        """
        return Float64(self.to_float32())

    # ========================================================================
    # Special Values
    # ========================================================================

    @staticmethod
    fn _zero() -> Self:
        """Create BF16 zero."""
        return BF16(0x0000)

    @staticmethod
    fn _neg_zero() -> Self:
        """Create BF16 negative zero."""
        return BF16(0x8000)

    @staticmethod
    fn _inf() -> Self:
        """Create BF16 positive infinity."""
        return BF16(0x7F80)

    @staticmethod
    fn _neg_inf() -> Self:
        """Create BF16 negative infinity."""
        return BF16(0xFF80)

    @staticmethod
    fn _nan() -> Self:
        """Create BF16 NaN."""
        return BF16(0x7FC0)

    fn is_nan(self) -> Bool:
        """Check if value is NaN.

        Returns:
            True if the value is NaN (exp=255, mantissa!=0).
        """
        var exp = (self.value >> 7) & 0xFF
        var mantissa = self.value & 0x7F
        return exp == 0xFF and mantissa != 0

    fn is_inf(self) -> Bool:
        """Check if value is infinity (positive or negative).

        Returns:
            True if the value is ±Inf (exp=255, mantissa=0).
        """
        var exp = (self.value >> 7) & 0xFF
        var mantissa = self.value & 0x7F
        return exp == 0xFF and mantissa == 0

    fn is_finite(self) -> Bool:
        """Check if value is finite (not NaN or infinity).

        Returns:
            True if finite, False otherwise.
        """
        return not (self.is_nan() or self.is_inf())

    fn is_zero(self) -> Bool:
        """Check if value is zero (positive or negative).

        Returns:
            True if the value is ±0 (exp=0, mantissa=0).
        """
        return (self.value & 0x7FFF) == 0

    fn is_subnormal(self) -> Bool:
        """Check if value is subnormal (denormalized).

        Returns:
            True if the value is subnormal (exp=0, mantissa!=0).
        """
        var exp = (self.value >> 7) & 0xFF
        var mantissa = self.value & 0x7F
        return exp == 0 and mantissa != 0

    fn sign(self) -> Int:
        """Get the sign of the value.

        Returns:
            1 for negative values, 0 for positive values (including +0).
        """
        return Int((self.value >> 15) & 1)

    # ========================================================================
    # Arithmetic Operations (via Float32)
    # ========================================================================

    fn __add__(self, other: Self) -> Self:
        """Add two BF16 values.

        Args:
            other: Value to add.

        Returns:
            Sum as BF16.
        """
        return BF16.from_float32(self.to_float32() + other.to_float32())

    fn __sub__(self, other: Self) -> Self:
        """Subtract two BF16 values.

        Args:
            other: Value to subtract.

        Returns:
            Difference as BF16.
        """
        return BF16.from_float32(self.to_float32() - other.to_float32())

    fn __mul__(self, other: Self) -> Self:
        """Multiply two BF16 values.

        Args:
            other: Value to multiply.

        Returns:
            Product as BF16.
        """
        return BF16.from_float32(self.to_float32() * other.to_float32())

    fn __truediv__(self, other: Self) -> Self:
        """Divide two BF16 values.

        Args:
            other: Divisor.

        Returns:
            Quotient as BF16.
        """
        return BF16.from_float32(self.to_float32() / other.to_float32())

    fn __neg__(self) -> Self:
        """Negate the value.

        Returns:
            BF16 with flipped sign bit.
        """
        return BF16(self.value ^ 0x8000)

    fn __abs__(self) -> Self:
        """Absolute value.

        Returns:
            BF16 with sign bit cleared.
        """
        return BF16(self.value & 0x7FFF)

    # ========================================================================
    # Comparison Operations
    # ========================================================================

    fn __eq__(self, other: Self) -> Bool:
        """Check equality.

        Note: NaN != NaN per IEEE 754.

        Args:
            other: Other BF16 value.

        Returns:
            True if equal.
        """
        if self.is_nan() or other.is_nan():
            return False
        return self.value == other.value

    fn __ne__(self, other: Self) -> Bool:
        """Check inequality.

        Args:
            other: Other BF16 value.

        Returns:
            True if not equal.
        """
        return not (self == other)

    fn __lt__(self, other: Self) -> Bool:
        """Less than comparison.

        Args:
            other: Other BF16 value.

        Returns:
            True if self < other (using Float32 comparison).
        """
        return self.to_float32() < other.to_float32()

    fn __le__(self, other: Self) -> Bool:
        """Less than or equal comparison.

        Args:
            other: Other BF16 value.

        Returns:
            True if self <= other (using Float32 comparison).
        """
        return self.to_float32() <= other.to_float32()

    fn __gt__(self, other: Self) -> Bool:
        """Greater than comparison.

        Args:
            other: Other BF16 value.

        Returns:
            True if self > other (using Float32 comparison).
        """
        return self.to_float32() > other.to_float32()

    fn __ge__(self, other: Self) -> Bool:
        """Greater than or equal comparison.

        Args:
            other: Other BF16 value.

        Returns:
            True if self >= other (using Float32 comparison).
        """
        return self.to_float32() >= other.to_float32()

    # ========================================================================
    # String Representation
    # ========================================================================

    fn __str__(self) -> String:
        """String representation showing BF16 value as Float32.

        Returns:
            String representation.
        """
        return "BF16(" + String(self.to_float32()) + ")"

    fn __repr__(self) -> String:
        """Detailed representation showing both bits and value.

        Returns:
            Detailed string representation.
        """
        return (
            "BF16(bits=0x"
            + hex(self.value)
            + ", value="
            + String(self.to_float32())
            + ")"
        )


# ============================================================================
# Utility Functions
# ============================================================================


fn print_bf16_bits(value: BF16):
    """Print BF16 value and its bit representation.

    Shows sign, exponent, and mantissa bits for debugging.

    Args:
        value: BF16 value to print.

    Example:
        ```mojo
        var bf16 = BF16.from_float32(3.14159)
        print_bf16_bits(bf16)
        # Output:
        # BF16: 3.140625
        # Bits: 0100000010010010
        # Sign: 0, Exponent: 128, Mantissa: 18
        ```
    """
    print("BF16: " + String(value.to_float32()))

    # Print binary representation
    var bits = value.value
    var binary = String("")
    for i in range(15, -1, -1):
        var bit = (bits >> i) & 0x1
        binary += String(bit)

    print("Bits: " + binary)

    # Decode components
    var sign = (bits >> 15) & 0x1
    var exponent = (bits >> 7) & 0xFF
    var mantissa = bits & 0x7F

    print(
        "Sign: "
        + String(sign)
        + ", Exponent: "
        + String(exponent)
        + ", Mantissa: "
        + String(mantissa)
    )
