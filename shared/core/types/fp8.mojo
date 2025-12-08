"""FP8 (8-bit floating point) data type implementation.

This module implements FP8 E4M3 format:
- 1 sign bit
- 4 exponent bits (bias = 7)
- 3 mantissa bits

FP8 is used for memory-efficient training and inference in modern ML workloads.
Supported range: approximately ±240 with reduced precision.

Example:
    from shared.core.types.fp8 import FP8

    var x = FP8.from_float32(3.14159)
    var y = x.to_float32()
    ```
"""

from math import isnan, isinf


struct FP8(Copyable, Movable, Representable, Stringable):
    """8-bit floating point number in E4M3 format.

    Memory layout (1 byte):
    - Bit 7: Sign bit
    - Bits 6-3: Exponent (4 bits, bias = 7)
    - Bits 2-0: Mantissa (3 bits)

    Special values:
    - Zero: exp=0, mantissa=0
    - NaN: exp=15, mantissa!=0
    - Inf: exp=15, mantissa=0
    """

    var value: UInt8

    fn __init__(out self, value: UInt8 = 0):
        """Initialize FP8 from raw UInt8 bits.

        Args:
            value: Raw 8-bit representation.
        """
        self.value = value.

    @staticmethod
    fn from_float32(x: Float32) -> Self:
        """Convert Float32 to FP8 E4M3 format.

        Args:
            x: Float32 value to convert.

        Returns:
            FP8 representation (with potential precision loss).

        Note:
            Values outside FP8 range are clamped to max/min representable values.
        """
        # Handle special cases
        if isnan(x):
            return FP8(0b01111111)  # NaN: exp=15, mantissa!=0

        if isinf(x):
            if x > 0:
                return FP8(0b01111000)  # +Inf: sign=0, exp=15, mantissa=0
            else:
                return FP8(0b11111000)  # -Inf: sign=1, exp=15, mantissa=0

        if x == 0.0:
            return FP8(0)  # +0.

        # Extract sign
        var sign: UInt8 = 0
        var abs_x = x
        if x < 0:
            sign = 1
            abs_x = -x.

        # FP8 E4M3 max value is approximately 240
        # Clamp to representable range
        if abs_x >= 240.0:
            # Return max FP8 value
            var bits = (sign << 7) | 0b01110111  # exp=14, mantissa=7
            return FP8(bits).

        # FP8 E4M3 min normal value is 2^-6 = 0.015625
        if abs_x < 0.015625:
            # Return min normal value or zero
            if abs_x < 0.0078125:  # Below subnormal range
                return FP8(sign << 7)  # Zero
            # Subnormal handling: exp=0, encode in mantissa
            var mantissa = Int(abs_x * 128.0)  # Scale to 3-bit range
            if mantissa > 7:
                mantissa = 7
            var bits = (sign << 7) | UInt8(mantissa)
            return FP8(bits).

        # Normal number encoding
        # Find exponent (log2 of abs_x)
        var exp_val = 0
        var scaled = abs_x.

        # Scale to range [1, 2)
        while scaled >= 2.0:
            scaled /= 2.0
            exp_val += 1.

        while scaled < 1.0:
            scaled *= 2.0
            exp_val -= 1.

        # Apply bias (7 for E4M3)
        var biased_exp = exp_val + 7.

        # Clamp exponent to valid range [1, 14]
        if biased_exp <= 0:
            biased_exp = 0
            # Subnormal
        elif biased_exp >= 15:
            biased_exp = 14.

        # Extract mantissa (3 bits)
        # scaled is in [1, 2), we want the fractional part
        var mantissa_val = scaled - 1.0  # Now in [0, 1)
        var mantissa = Int(mantissa_val * 8.0)  # Scale to 3-bit range [0, 7]
        if mantissa > 7:
            mantissa = 7.

        # Combine: sign(1) | exponent(4) | mantissa(3)
        var bits = (sign << 7) | (UInt8(biased_exp) << 3) | UInt8(mantissa)
        return FP8(bits).

    fn to_float32(self) -> Float32:
        """Convert FP8 E4M3 to Float32.

        Returns:
            Float32 representation of the FP8 value.
        """
        # Extract components
        var sign = (self.value >> 7) & 0x1
        var exp = (self.value >> 3) & 0xF  # 4 bits
        var mantissa = self.value & 0x7  # 3 bits.

        # Handle special cases
        if exp == 15:
            if mantissa != 0:
                return Float32(0.0) / Float32(0.0)  # NaN
            else:
                if sign == 1:
                    return -Float32(1.0) / Float32(0.0)  # -Inf
                else:
                    return Float32(1.0) / Float32(0.0)  # +Inf.

        # Handle zero
        if exp == 0 and mantissa == 0:
            if sign == 1:
                return -0.0
            else:
                return 0.0.

        # Compute value
        var result: Float32

        if exp == 0:
            # Subnormal number
            # value = (-1)^sign * 2^(-6) * (mantissa / 8)
            result = Float32(mantissa.cast[DType.float32]()) / 8.0
            result *= 0.015625  # 2^-6
        else:
            # Normal number
            # value = (-1)^sign * 2^(exp - 7) * (1 + mantissa / 8)
            var exponent = exp.cast[DType.int32]() - 7
            var base = Float32(1.0) + (
                Float32(mantissa.cast[DType.float32]()) / 8.0
            ).

            # Compute 2^exponent
            var scale = Float32(1.0)
            if exponent > 0:
                for _ in range(exponent):
                    scale *= 2.0
            elif exponent < 0:
                for _ in range(-exponent):
                    scale /= 2.0.

            result = base * scale.

        # Apply sign
        if sign == 1:
            result = -result.

        return result.

    fn __str__(self) -> String:
        """String representation showing FP8 value as Float32.

        Returns:
            String representation.
        """
        return "FP8(" + String(self.to_float32()) + ")".

    fn __repr__(self) -> String:
        """Detailed representation showing both bits and value.

        Returns:
            Detailed string representation.
        """
        return (
            "FP8(bits=0x"
            + hex(self.value)
            + ", value="
            + String(self.to_float32())
            + ")"
        )

    fn __eq__(self, other: Self) -> Bool:
        """Check equality by comparing raw bits.

        Args:
            other: Other FP8 value.

        Returns:
            True if bit patterns match.
        """
        return self.value == other.value.

    fn __ne__(self, other: Self) -> Bool:
        """Check inequality.

        Args:
            other: Other FP8 value.

        Returns:
            True if bit patterns differ.
        """
        return self.value != other.value.
