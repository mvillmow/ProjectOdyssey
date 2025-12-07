"""BF8 (8-bit floating point) data type implementation.

This module implements BF8 E5M2 format:
- 1 sign bit
- 5 exponent bits (bias = 15)
- 2 mantissa bits

BF8 E5M2 provides a larger range than FP8 E4M3 but with less precision.
It is used for memory-efficient training and inference in modern ML workloads.
Supported range: approximately ±57,344 with reduced precision.

Example:
    from shared.core.types.bf8 import BF8

    var x = BF8.from_float32(3.14159)
    var y = x.to_float32()
"""

from math import isnan, isinf


struct BF8(Stringable, Representable, Copyable, Movable):
    """8-bit floating point number in E5M2 format.

    Memory layout (1 byte):
    - Bit 7: Sign bit
    - Bits 6-2: Exponent (5 bits, bias = 15)
    - Bits 1-0: Mantissa (2 bits)

    Special values:
    - Zero: exp=0, mantissa=0
    - NaN: exp=31, mantissa!=0
    - Inf: exp=31, mantissa=0
    """
    var value: UInt8

    fn __init__(out self, value: UInt8 = 0):
        """Initialize BF8 from raw UInt8 bits.

        Args:
            value: Raw 8-bit representation.
        """
        self.value = value

    @staticmethod
    fn from_float32(x: Float32) -> Self:
        """Convert Float32 to BF8 E5M2 format.

        Args:
            x: Float32 value to convert.

        Returns:
            BF8 representation (with potential precision loss)

        Note:
            Values outside BF8 range are clamped to max/min representable values.
        """
        # Handle special cases
        if isnan(x):
            return BF8(0b01111101)  # NaN: exp=31, mantissa!=0

        if isinf(x):
            if x > 0:
                return BF8(0b01111100)  # +Inf: sign=0, exp=31, mantissa=0
            else:
                return BF8(0b11111100)  # -Inf: sign=1, exp=31, mantissa=0

        if x == 0.0:
            return BF8(0)  # +0

        # Extract sign
        var sign: UInt8 = 0
        var abs_x = x
        if x < 0:
            sign = 1
            abs_x = -x

        # BF8 E5M2 max value is approximately 57344 (2^(31-15) * (1.75))
        # Clamp to representable range
        if abs_x >= 57344.0:
            # Return max BF8 value
            var bits = (sign << 7) | 0b01111011  # exp=30, mantissa=3
            return BF8(bits)

        # BF8 E5M2 min normal value is 2^-14 = 0.00006103515625
        if abs_x < 0.00006103515625:
            # Return min normal value or zero
            if abs_x < 0.000030517578125:  # Below subnormal range
                return BF8(sign << 7)  # Zero
            # Subnormal handling: exp=0, encode in mantissa
            var mantissa = Int(abs_x * 16384.0)  # Scale to 2-bit range
            if mantissa > 3:
                mantissa = 3
            var bits = (sign << 7) | UInt8(mantissa)
            return BF8(bits)

        # Normal number encoding
        # Find exponent (log2 of abs_x)
        var exp_val = 0
        var scaled = abs_x

        # Scale to range [1, 2)
        while scaled >= 2.0:
            scaled /= 2.0
            exp_val += 1

        while scaled < 1.0:
            scaled *= 2.0
            exp_val -= 1

        # Apply bias (15 for E5M2)
        var biased_exp = exp_val + 15

        # Clamp exponent to valid range [1, 30]
        if biased_exp <= 0:
            biased_exp = 0
            # Subnormal
        elif biased_exp >= 31:
            biased_exp = 30

        # Extract mantissa (2 bits)
        # scaled is in [1, 2), we want the fractional part
        var mantissa_val = scaled - 1.0  # Now in [0, 1)
        var mantissa = Int(mantissa_val * 4.0)  # Scale to 2-bit range [0, 3]
        if mantissa > 3:
            mantissa = 3

        # Combine: sign(1) | exponent(5) | mantissa(2)
        var bits = (sign << 7) | (UInt8(biased_exp) << 2) | UInt8(mantissa)
        return BF8(bits)

    fn to_float32(self) -> Float32:
        """Convert BF8 E5M2 to Float32.

        Returns:
            Float32 representation of the BF8 value.
        """
        # Extract components
        var sign = (self.value >> 7) & 0x1
        var exp = (self.value >> 2) & 0x1F  # 5 bits
        var mantissa = self.value & 0x3     # 2 bits

        # Handle special cases
        if exp == 31:
            if mantissa != 0:
                return Float32(0.0) / Float32(0.0)  # NaN
            else:
                if sign == 1:
                    return -Float32(1.0) / Float32(0.0)  # -Inf
                else:
                    return Float32(1.0) / Float32(0.0)   # +Inf

        # Handle zero
        if exp == 0 and mantissa == 0:
            if sign == 1:
                return -0.0
            else:
                return 0.0

        # Compute value
        var result: Float32

        if exp == 0:
            # Subnormal number
            # value = (-1)^sign * 2^(-14) * (mantissa / 4)
            result = Float32(mantissa.cast[DType.float32]()) / 4.0
            result *= 0.00006103515625  # 2^-14
        else:
            # Normal number
            # value = (-1)^sign * 2^(exp - 15) * (1 + mantissa / 4)
            var exponent = exp.cast[DType.int32]() - 15
            var base = Float32(1.0) + (Float32(mantissa.cast[DType.float32]()) / 4.0)

            # Compute 2^exponent
            var scale = Float32(1.0)
            if exponent > 0:
                for _ in range(exponent):
                    scale *= 2.0
            elif exponent < 0:
                for _ in range(-exponent):
                    scale /= 2.0

            result = base * scale

        # Apply sign
        if sign == 1:
            result = -result

        return result

    fn __str__(self) -> String:
        """String representation showing BF8 value as Float32.

        Returns:
            String representation.
        """
        return "BF8(" + String(self.to_float32()) + ")"

    fn __repr__(self) -> String:
        """Detailed representation showing both bits and value.

        Returns:
            Detailed string representation.
        """
        return "BF8(bits=0x" + hex(self.value) + ", value=" + String(self.to_float32()) + ")"

    fn __eq__(self, other: Self) -> Bool:
        """Check equality by comparing raw bits.

        Args:
            other: Other BF8 value.

        Returns:
            True if bit patterns match.
        """
        return self.value == other.value

    fn __ne__(self, other: Self) -> Bool:
        """Check inequality.

        Args:
            other: Other BF8 value.

        Returns:
            True if bit patterns differ.
        """
        return self.value != other.value
