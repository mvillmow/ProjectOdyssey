"""BFloat16 (Brain Floating Point) implementation.

Implements BFloat16 data type with uint16 storage and proper IEEE 754-like
bit manipulation. BFloat16 uses the same exponent bits as Float32 but with
reduced mantissa precision, making it ideal for deep learning workloads.

BFloat16 Format (16 bits):
- 1 sign bit
- 8 exponent bits (same as Float32)
- 7 mantissa bits (vs 23 in Float32, 10 in Float16)

Benefits:
- Same range as Float32 (~1e-38 to 3.4e38)
- Half the memory of Float32
- Simpler conversion to/from Float32 (truncation)
- Better for training than Float16 (wider range, less overflow)

Usage:
    var bf16 = BFloat16.from_float32(3.14159)
    var f32 = bf16.to_float32()
    print(bf16)  # BFloat16(3.140625)
"""

from memory import UnsafePointer
from math import isnan, isinf


@register_passable("trivial")
struct BFloat16:
    """BFloat16 (Brain Floating Point) data type.

    Stores value as uint16 with proper bit layout:
    - Bit 15: Sign
    - Bits 14-7: Exponent (8 bits)
    - Bits 6-0: Mantissa (7 bits)

    This matches Float32 exponent but with truncated mantissa,
    enabling simple conversion: truncate Float32's lower 16 bits.

    Attributes:
        bits: uint16 storage for BF16 representation
    """
    var bits: UInt16

    # ========================================================================
    # Constructors
    # ========================================================================

    fn __init__(mut self):
        """Initialize BFloat16 to zero."""
        self.bits = 0

    fn __init__(mut self, bits: UInt16):
        """Initialize BFloat16 from raw bits.

        Args:
            bits: Raw uint16 bit representation
        """
        self.bits = bits

    # ========================================================================
    # Conversion from Float32
    # ========================================================================

    @staticmethod
    @always_inline
    fn from_float32(value: Float32) -> BFloat16:
        """Convert Float32 to BFloat16 using rounding.

        Uses round-to-nearest-even (RNE) for proper IEEE 754 rounding.
        This is the recommended conversion method.

        Args:
            value: Float32 value to convert

        Returns:
            BFloat16 representation

        Example:
            var bf16 = BFloat16.from_float32(3.14159)
        """
        # Handle special cases
        if isnan(value):
            return BFloat16._nan()
        if isinf(value):
            return BFloat16._inf() if value > 0 else BFloat16._neg_inf()

        # Get bit representation of Float32 using stack allocation
        var f32_val = value
        var ptr_addr = UnsafePointer.address_of(f32_val)
        var bits32 = ptr_addr.bitcast[UInt32]()[0]

        # Extract components from Float32 (32 bits)
        # Float32: [sign:1][exponent:8][mantissa:23]
        var sign_bit = (bits32 >> 31) & 0x1
        var exponent = (bits32 >> 23) & 0xFF
        var mantissa23 = bits32 & 0x7FFFFF

        # Round to nearest even (RNE)
        # Check bit 16 (first truncated bit) and bits 15:0 for rounding
        var rounding_bit = (bits32 >> 16) & 0x1
        var sticky_bits = bits32 & 0xFFFF

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
                        return BFloat16._inf() if sign_bit == 0 else BFloat16._neg_inf()

        # Combine into BFloat16 format: [sign:1][exponent:8][mantissa:7]
        var bits16 = UInt16((sign_bit << 15) | (exponent << 7) | mantissa7)

        return BFloat16(bits16)

    @staticmethod
    @always_inline
    fn from_float32_truncate(value: Float32) -> BFloat16:
        """Convert Float32 to BFloat16 using simple truncation.

        Faster than rounding but less accurate. Use only when performance
        is critical and slight accuracy loss is acceptable.

        Args:
            value: Float32 value to convert

        Returns:
            BFloat16 representation

        Example:
            var bf16 = BFloat16.from_float32_truncate(3.14159)
        """
        # Get bit representation using stack allocation
        var f32_val = value
        var ptr_addr = UnsafePointer.address_of(f32_val)
        var bits32 = ptr_addr.bitcast[UInt32]()[0]

        # Simply take upper 16 bits (truncate lower 16)
        var bits16 = UInt16(bits32 >> 16)

        return BFloat16(bits16)

    # ========================================================================
    # Conversion to Float32
    # ========================================================================

    @always_inline
    fn to_float32(self) -> Float32:
        """Convert BFloat16 to Float32.

        Conversion is exact (no rounding needed) since we're expanding
        mantissa bits (7 -> 23) by zero-padding.

        Returns:
            Float32 representation

        Example:
            var f32 = bf16.to_float32()
        """
        # BFloat16 to Float32 is simple: extend to 32 bits by shifting left 16
        # BF16: [sign:1][exponent:8][mantissa:7]
        # FP32: [sign:1][exponent:8][mantissa:23]
        #
        # We just zero-pad the lower 16 bits of mantissa
        var bits32 = UInt32(self.bits) << 16

        # Convert bits to Float32 using stack allocation
        var u32_val = bits32
        var ptr_addr = UnsafePointer.address_of(u32_val)
        var result = ptr_addr.bitcast[Float32]()[0]

        return result

    fn to_float64(self) -> Float64:
        """Convert BFloat16 to Float64.

        Goes through Float32 conversion first.

        Returns:
            Float64 representation
        """
        return Float64(self.to_float32())

    # ========================================================================
    # Special Values
    # ========================================================================

    @staticmethod
    fn _zero() -> BFloat16:
        """Create BFloat16 zero."""
        return BFloat16(0x0000)

    @staticmethod
    fn _neg_zero() -> BFloat16:
        """Create BFloat16 negative zero."""
        return BFloat16(0x8000)

    @staticmethod
    fn _inf() -> BFloat16:
        """Create BFloat16 positive infinity."""
        return BFloat16(0x7F80)

    @staticmethod
    fn _neg_inf() -> BFloat16:
        """Create BFloat16 negative infinity."""
        return BFloat16(0xFF80)

    @staticmethod
    fn _nan() -> BFloat16:
        """Create BFloat16 NaN."""
        return BFloat16(0x7FC0)

    fn is_nan(self) -> Bool:
        """Check if value is NaN.

        Returns:.            True if NaN, False otherwise.
        """
        # NaN: exponent = 0xFF, mantissa != 0
        var exponent = (self.bits >> 7) & 0xFF
        var mantissa = self.bits & 0x7F
        return exponent == 0xFF and mantissa != 0

    fn is_inf(self) -> Bool:
        """Check if value is infinity (positive or negative).

        Returns:.            True if infinity, False otherwise.
        """
        # Inf: exponent = 0xFF, mantissa = 0
        var exponent = (self.bits >> 7) & 0xFF
        var mantissa = self.bits & 0x7F
        return exponent == 0xFF and mantissa == 0

    fn is_finite(self) -> Bool:
        """Check if value is finite (not NaN or infinity).

        Returns:.            True if finite, False otherwise.
        """
        return not (self.is_nan() or self.is_inf())

    # ========================================================================
    # Arithmetic Operations (via Float32)
    # ========================================================================

    fn __add__(self, other: BFloat16) -> BFloat16:
        """Add two BFloat16 values.

        Args:.            `other`: Value to add.

        Returns:.            Sum as BFloat16.
        """
        var a = self.to_float32()
        var b = other.to_float32()
        return BFloat16.from_float32(a + b)

    fn __sub__(self, other: BFloat16) -> BFloat16:
        """Subtract two BFloat16 values.

        Args:.            `other`: Value to subtract.

        Returns:.            Difference as BFloat16.
        """
        var a = self.to_float32()
        var b = other.to_float32()
        return BFloat16.from_float32(a - b)

    fn __mul__(self, other: BFloat16) -> BFloat16:
        """Multiply two BFloat16 values.

        Args:.            `other`: Value to multiply.

        Returns:.            Product as BFloat16.
        """
        var a = self.to_float32()
        var b = other.to_float32()
        return BFloat16.from_float32(a * b)

    fn __truediv__(self, other: BFloat16) -> BFloat16:
        """Divide two BFloat16 values.

        Args:.            `other`: Divisor.

        Returns:.            Quotient as BFloat16.
        """
        var a = self.to_float32()
        var b = other.to_float32()
        return BFloat16.from_float32(a / b)

    fn __neg__(self) -> BFloat16:
        """Negate BFloat16 value.

        Returns:.            Negated value.
        """
        # Flip sign bit
        return BFloat16(self.bits ^ 0x8000)

    # ========================================================================
    # Comparison Operations
    # ========================================================================

    fn __eq__(self, other: BFloat16) -> Bool:
        """Check equality.

        Args:.            `other`: Value to compare.

        Returns:.            True if equal.
        """
        # NaN != NaN
        if self.is_nan() or other.is_nan():
            return False
        return self.bits == other.bits

    fn __ne__(self, other: BFloat16) -> Bool:
        """Check inequality.

        Args:.            `other`: Value to compare.

        Returns:.            True if not equal.
        """
        return not (self == other)

    fn __lt__(self, other: BFloat16) -> Bool:
        """Check less than.

        Args:.            `other`: Value to compare.

        Returns:.            True if self < other.
        """
        return self.to_float32() < other.to_float32()

    fn __le__(self, other: BFloat16) -> Bool:
        """Check less than or equal.

        Args:.            `other`: Value to compare.

        Returns:.            True if self <= other.
        """
        return self.to_float32() <= other.to_float32()

    fn __gt__(self, other: BFloat16) -> Bool:
        """Check greater than.

        Args:.            `other`: Value to compare.

        Returns:.            True if self > other.
        """
        return self.to_float32() > other.to_float32()

    fn __ge__(self, other: BFloat16) -> Bool:
        """Check greater than or equal.

        Args:.            `other`: Value to compare.

        Returns:.            True if self >= other.
        """
        return self.to_float32() >= other.to_float32()

    # ========================================================================
    # String Representation
    # ========================================================================

    fn __str__(self) -> String:
        """Convert to string.

        Returns:.            String representation.
        """
        return "BFloat16(" + String(self.to_float32()) + ")"

    fn __repr__(self) -> String:
        """Get representation string.

        Returns:.            Representation string.
        """
        return self.__str__()


# ============================================================================
# Utility Functions
# ============================================================================

fn print_bfloat16_bits(value: BFloat16):
    """Print BFloat16 value and its bit representation.

    Shows sign, exponent, and mantissa bits for debugging.

    Args:.        `value`: BFloat16 value to print.

    Example:.        var bf16 = BFloat16.from_float32(3.14159)
        print_bfloat16_bits(bf16)
        # Output:
        # BFloat16: 3.140625
        # Bits: 0100000010010010
        # Sign: 0, Exponent: 10000000 (128), Mantissa: 1001001
    """
    print("BFloat16: " + String(value.to_float32()))

    # Print binary representation
    var bits = value.bits
    var binary = String("")
    for i in range(15, -1, -1):
        var bit = (bits >> i) & 0x1
        binary += String(bit)

    print("Bits: " + binary)

    # Decode components
    var sign = (bits >> 15) & 0x1
    var exponent = (bits >> 7) & 0xFF
    var mantissa = bits & 0x7F

    print("Sign: " + String(sign) +
          ", Exponent: " + String(exponent) +
          ", Mantissa: " + String(mantissa))
