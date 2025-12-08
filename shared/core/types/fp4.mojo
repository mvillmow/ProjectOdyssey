"""FP4 E2M1 (4-bit floating point) base format implementation.

This module implements the E2M1 format used in blocked FP4 formats (MXFP4, NVFP4):
- 1 sign bit
- 2 exponent bits (bias = 1)
- 1 mantissa bit

E2M1 is NOT used standalone - it requires a block-level scale factor.
This module provides the base encoding/decoding for MXFP4 and NVFP4.

Key characteristics:
- 4 bits per value
- Limited precision (1-bit mantissa)
- Limited range (2-bit exponent)
- Requires external scale factor for proper representation

Example:
    from shared.core.types.fp4 import FP4_E2M1

    # E2M1 is typically used within block structures
    var fp4_val = FP4_E2M1.from_float32(1.5, scale=1.0)
    var reconstructed = fp4_val.to_float32(scale=1.0)
    ```
"""

from math import isnan, isinf

# **FIXME (#2378 - TEST-010 - P0 CRITICAL)**: FP4_E2M1 base type completely untested (0% coverage)
# This entire 217-line module has ZERO test coverage. Critical untested functions:
# - Lines 56-139: from_float32() (E2M1 encoding algorithm)
# - Lines 141-178: to_float32() (E2M1 decoding algorithm)
# - Lines 180-194: String representations (__str__, __repr__)
# - Lines 196-216: Comparison operators (__eq__, __ne__, __lt__, __le__, __gt__, __ge__)
#
# Impact: Base encoding/decoding used by MXFP4 and NVFP4 is completely untested
# Action Required: Create tests/core/types/test_fp4_base.mojo with comprehensive tests
# Minimum Tests Needed:
#   1. Basic encoding/decoding (normal values, zero, special values)
#   2. Edge cases (NaN, Infinity, scale=0)
#   3. Comparison operators (6 operators)
#   4. String representations
# Severity: BLOCKING - base type must be tested before production use
# See: COMPREHENSIVE_REVIEW_FINDINGS.md (TEST-010)


struct FP4_E2M1(Copyable, Movable, Representable, Stringable):
    """4-bit floating point number in E2M1 format.

    Memory layout (4 bits stored in UInt8):
    - Bit 3: Sign bit
    - Bits 2-1: Exponent (2 bits, bias = 1)
    - Bit 0: Mantissa (1 bit)

    Special values:
    - Zero: exp=0, mantissa=0
    - Max normal: exp=2, mantissa=1 (value = 6.0 before scaling)
    - Min normal: exp=1, mantissa=0 (value = 1.0 before scaling)

Note:
        E2M1 values are meaningless without a block-level scale factor.
        Use MXFP4 or NVFP4 for complete block-based representations.
    """

    var value: UInt8  # Only lower 4 bits are used

    fn __init__(out self, value: UInt8 = 0):
        """Initialize FP4_E2M1 from raw 4-bit value.

        Args:
            value: Raw 4-bit representation (only lower 4 bits used).
        """
        self.value = value & 0xF  # Mask to 4 bits

    @staticmethod
    fn from_float32(x: Float32, scale: Float32 = 1.0) -> Self:
        """Convert Float32 to FP4 E2M1 format with given scale.

        Args:
            x: Float32 value to convert.
            scale: Block-level scale factor.

        Returns:
            FP4_E2M1 representation.

        Note:
            The value is divided by scale before encoding.
            Values outside representable range are clamped.
        """
        # Handle special cases
        if isnan(x):
            return FP4_E2M1(0b0111)  # Max value as NaN representation

        if isinf(x):
            if x > 0:
                return FP4_E2M1(0b0111)  # Max positive value
            else:
                return FP4_E2M1(0b1111)  # Max negative value

        # Scale the input
        var scaled = x / scale

        if scaled == 0.0:
            return FP4_E2M1(0)  # +0

        # Extract sign
        var sign: UInt8 = 0
        var abs_scaled = scaled
        if scaled < 0:
            sign = 1
            abs_scaled = -scaled

        # E2M1 representable values (before scaling):
        # exp=0, mantissa=0: 0
        # exp=1, mantissa=0: 1.0
        # exp=1, mantissa=1: 1.5
        # exp=2, mantissa=0: 2.0
        # exp=2, mantissa=1: 3.0
        # exp=3, mantissa=0: 4.0
        # exp=3, mantissa=1: 6.0 (max)

        # Clamp to representable range [0, 6.0]
        if abs_scaled >= 6.0:
            # Return max value: sign=s, exp=3, mantissa=1
            return FP4_E2M1((sign << 3) | 0b111)

        if abs_scaled < 0.5:
            # Return zero (subnormals not well-defined for E2M1)
            return FP4_E2M1(sign << 3)

        # Find best representation
        # Quantize to nearest representable value
        # Representable values: 0, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
        var exp: UInt8
        var mantissa: UInt8
        if abs_scaled < 1.25:
            exp = 1
            mantissa = 0  # 1.0
        elif abs_scaled < 1.75:
            exp = 1
            mantissa = 1  # 1.5
        elif abs_scaled < 2.5:
            exp = 2
            mantissa = 0  # 2.0
        elif abs_scaled < 3.5:
            exp = 2
            mantissa = 1  # 3.0
        elif abs_scaled < 5.0:
            exp = 3
            mantissa = 0  # 4.0
        else:
            exp = 3
            mantissa = 1  # 6.0

        # Combine: sign(1) | exponent(2) | mantissa(1)
        var bits = (sign << 3) | (exp << 1) | mantissa
        return FP4_E2M1(bits)

    fn to_float32(self, scale: Float32 = 1.0) -> Float32:
        """Convert FP4 E2M1 to Float32 with given scale.

        Args:
            scale: Block-level scale factor.

        Returns:
            Float32 representation of the scaled E2M1 value.
        """
        # Extract components (4 bits total)
        var sign = (self.value >> 3) & 0x1
        var exp = (self.value >> 1) & 0x3  # 2 bits
        var mantissa = self.value & 0x1  # 1 bit

        # Handle zero
        if exp == 0:
            return Float32(0.0) if sign == 0 else Float32(-0.0)

        # Compute unscaled value
        # E2M1: value = 2^(exp-1) * (1 + mantissa/2)
        # With 1-bit mantissa, the fractional part is mantissa * 0.5
        var exponent = exp.cast[DType.int32]() - 1
        var base = Float32(1.0) + Float32(
            mantissa.cast[DType.float32]()
        ) * Float32(0.5)

        # Compute 2^exponent
        var unscaled = base
        if exponent > 0:
            for _ in range(exponent):
                unscaled *= 2.0
        elif exponent < 0:
            for _ in range(-exponent):
                unscaled /= 2.0

        # Apply sign and scale
        var result = unscaled * scale
        if sign == 1:
            result = -result

        return result

    fn __str__(self) -> String:
        """String representation showing FP4 value as Float32 (unscaled).

        Returns:
            String representation.
        """
        return "FP4_E2M1(" + String(self.to_float32(scale=1.0)) + ")"

    fn __repr__(self) -> String:
        """Detailed representation showing bits and value.

        Returns:
            Detailed string representation.
        """
        return (
            "FP4_E2M1(bits=0x"
            + hex(self.value)
            + ", value="
            + String(self.to_float32(scale=1.0))
            + ")"
        )

    fn __eq__(self, other: Self) -> Bool:
        """Check equality by comparing raw bits.

        Args:
            other: Other FP4_E2M1 value.

        Returns:
            True if bit patterns match.
        """
        return self.value == other.value

    fn __ne__(self, other: Self) -> Bool:
        """Check inequality.

        Args:
            other: Other FP4_E2M1 value.

        Returns:
            True if bit patterns differ.
        """
        return self.value != other.value
