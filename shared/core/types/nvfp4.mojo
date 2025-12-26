"""NVFP4 (NVIDIA FP4) blocked floating point format.

Implements NVFP4 format from the paper "Microscaling Data Formats for Deep Learning":
- Individual values: E2M1 (1 sign, 2 exponent, 1 mantissa)
- Block size: 16 elements
- Scale format: E4M3 (4-bit exponent, 3-bit mantissa)

NVFP4 provides finer-grained scaling than MXFP4 with smaller block size.
Each block of 16 FP4 values shares a common E4M3 scale factor.

According to the paper, E4M3 "achieves the best results" and provides
"modest improvements in accuracy" over E8M0, despite narrower exponent range.

Key characteristics:
- Memory: 8 bytes per block (16 × 4 bits = 64 bits) + 7 bits scale ≈ 9 bytes
- Dynamic range: Balanced (E4M3 scale)
- Precision: Limited (E2M1 values) but better than MXFP4 (smaller blocks)
- Use case: Memory-efficient ML training with better accuracy than MXFP4

Example:
    ```mojo
    from shared.core.types.nvfp4 import NVFP4, NVFP4Block

    # Individual value (stores both E2M1 and scale for convenience)
    var val = NVFP4.from_float32(3.14159)
    var reconstructed = val.to_float32()

    # Block storage (efficient: 16 values + 1 shared scale)
    var block = NVFP4Block.from_float32_array(data_array)
    ```

Reference:
    Tim Dettmers, Nolan Miller, Deepak Kapur, Luke Zettlemoyer.
    "Microscaling Data Formats for Deep Learning."
    arXiv preprint arXiv:2310.10537, 2023.
    https://arxiv.org/abs/2310.10537
    https://doi.org/10.48550/arXiv.2310.10537

    The paper demonstrates that microscaling (per-block scaling factors) enables
    efficient low-precision quantization (4-bit) while maintaining accuracy in
    deep learning training and inference. NVFP4 uses E4M3 scales (128 dynamic range)
    across 16-element blocks for a balance between accuracy and memory efficiency.
    The paper shows E4M3 achieves the best overall results with modest accuracy
    improvements over the wider-range E8M0 format.
"""

from math import isnan, isinf
from shared.core.types.fp4 import FP4_E2M1


struct E4M3Scale(Copyable, Movable, Representable, Stringable):
    """4-bit exponent + 3-bit mantissa scale factor for NVFP4 blocks.

    Memory layout (7 bits stored in UInt8):
    - Bits 6-3: Exponent (4 bits, bias = 7, same as FP8 E4M3)
    - Bits 2-0: Mantissa (3 bits)
    - No sign bit (always positive)

    Special values:
    - Zero: exp=0, mantissa=0
    - Max: exp=15, mantissa=7
    - No NaN/Inf (all patterns represent finite positive values)

    Valid range: Similar to FP8 E4M3 format.
    """

    var value: UInt8
    """7-bit E4M3 scale value."""

    fn __init__(out self, value: UInt8 = 0x38):
        """Initialize E4M3 scale from raw 7-bit value.

        Args:
            value: 7-bit representation (default = 0x38 = exp:7, mantissa:0 = 1.0).
        """
        self.value = value & 0x7F  # Mask to 7 bits

    @staticmethod
    fn from_float32(scale: Float32) -> Self:
        """Compute E4M3 scale from Float32 value.

        Args:
            scale: Positive Float32 scale value.

        Returns:
            E4M3 representation.

        Note:
            Scale must be positive. Uses FP8 E4M3 encoding logic.
        """
        if scale <= 0.0 or isnan(scale):
            return E4M3Scale(0)  # Zero scale

        if isinf(scale) or scale >= 240.0:
            return E4M3Scale(0x7F)  # Maximum scale (exp=15, mantissa=7)

        # Handle very small scales
        if scale < 0.015625:  # Below normal range
            if scale < 0.0078125:
                return E4M3Scale(0)  # Zero
            # Subnormal: exp=0, encode in mantissa
            var mantissa = Int(scale * 512.0)
            if mantissa > 7:
                mantissa = 7
            return E4M3Scale(UInt8(mantissa))

        # Normal number encoding (same as FP8 E4M3)
        var exp_val = 0
        var scaled = scale

        # Scale to range [1, 2)
        while scaled >= 2.0:
            scaled /= 2.0
            exp_val += 1

        while scaled < 1.0:
            scaled *= 2.0
            exp_val -= 1

        # Apply bias (7 for E4M3)
        var biased_exp = exp_val + 7

        # Clamp exponent to valid range [1, 14]
        if biased_exp <= 0:
            biased_exp = 0  # Subnormal
        elif biased_exp >= 15:
            biased_exp = 14  # Avoid overflow

        # Extract mantissa (3 bits)
        var mantissa_val = scaled - 1.0  # Now in [0, 1)
        var mantissa = Int(mantissa_val * 8.0)  # Scale to 3-bit range [0, 7]
        if mantissa > 7:
            mantissa = 7

        # Combine: exponent(4) | mantissa(3)
        var bits = (UInt8(biased_exp) << 3) | UInt8(mantissa)
        return E4M3Scale(bits)

    fn to_float32(self) -> Float32:
        """Convert E4M3 scale to Float32.

        Returns:
            Float32 representation.
        """
        # Extract components (7 bits total)
        var exp = (self.value >> 3) & 0xF  # 4 bits
        var mantissa = self.value & 0x7  # 3 bits

        # Handle zero
        if exp == 0 and mantissa == 0:
            return 0.0

        # Compute value (same logic as FP8 E4M3)
        var result: Float32

        if exp == 0:
            # Subnormal number
            # value = 2^(-6) * (mantissa / 8)
            result = Float32(mantissa.cast[DType.float32]()) / 8.0
            result *= 0.015625  # 2^-6
        else:
            # Normal number
            # value = 2^(exp - 7) * (1 + mantissa / 8)
            var exponent = exp.cast[DType.int32]() - 7
            var base = Float32(1.0) + (
                Float32(mantissa.cast[DType.float32]()) / 8.0
            )

            # Compute 2^exponent
            var scale_factor = Float32(1.0)
            if exponent > 0:
                for _ in range(exponent):
                    scale_factor *= 2.0
            elif exponent < 0:
                for _ in range(-exponent):
                    scale_factor /= 2.0

            result = base * scale_factor

        return result

    fn __str__(self) -> String:
        """String representation showing scale value.

        Returns:
            String representation.
        """
        var exp = (self.value >> 3) & 0xF
        var mantissa = self.value & 0x7
        return (
            "E4M3(exp="
            + String(exp)
            + ", mantissa="
            + String(mantissa)
            + ", scale="
            + String(self.to_float32())
            + ")"
        )

    fn __repr__(self) -> String:
        """Detailed representation.

        Returns:
            Detailed string representation.
        """
        return self.__str__()


struct NVFP4(Copyable, Movable, Representable, Stringable):
    """NVFP4 individual value (E2M1 + E4M3 scale).

    Acts like FP16 but stores internally as 4-bit E2M1 value plus 7-bit E4M3 scale.
    This representation is convenient but NOT space-efficient (11 bits total vs 4 bits in blocks).

    For efficient storage, use NVFP4Block which amortizes the scale across 16 values.

    Attributes:
        value: 4-bit E2M1 encoded value.
        scale: 7-bit E4M3 scale factor.
    """

    var value: FP4_E2M1
    """4-bit E2M1 encoded value."""
    var scale: E4M3Scale
    """7-bit E4M3 scale factor."""

    fn __init__(
        out self, value: FP4_E2M1 = FP4_E2M1(), scale: E4M3Scale = E4M3Scale()
    ):
        """Initialize NVFP4 from E2M1 value and E4M3 scale.

        Args:
            value: E2M1 encoded value.
            scale: E4M3 scale factor.
        """
        self.value = value.copy()
        self.scale = scale.copy()

    @staticmethod
    fn from_float32(x: Float32) -> Self:
        """Convert Float32 to NVFP4.

        Computes optimal scale for the single value and encodes.

        Args:
            x: Float32 value to convert.

        Returns:
            NVFP4 representation.
        """
        # Handle special cases
        if isnan(x) or isinf(x):
            return NVFP4(FP4_E2M1.from_float32(x, scale=1.0), E4M3Scale(0x38))

        if x == 0.0:
            return NVFP4(FP4_E2M1(0), E4M3Scale(0x38))

        # Compute scale: find value such that |x| / scale is in E2M1 range [0, 6]
        var abs_x = x if x > 0 else -x
        var scale_val = Float32(1.0)
        var exp_val = 0

        # Scale to fit in E2M1 range [0, 6]
        while abs_x / scale_val > 6.0:
            scale_val *= 2.0
            exp_val += 1

        while abs_x / scale_val < 0.5 and exp_val > -7:
            scale_val /= 2.0
            exp_val -= 1

        # Create E4M3 scale
        var scale = E4M3Scale.from_float32(scale_val)

        # Encode E2M1 value
        var value = FP4_E2M1.from_float32(x, scale=scale.to_float32())

        return NVFP4(value, scale)

    @staticmethod
    fn from_float32_stochastic(x: Float32, seed: UInt64) -> Self:
        """Convert Float32 to NVFP4 with stochastic rounding.

        Uses stochastic rounding which is recommended for gradient quantization.
        When a value falls between two representable FP4 values, probabilistically
        round up or down based on distance.

        Args:
            x: Float32 value to convert.
            seed: Random seed for deterministic stochastic rounding.

        Returns:
            NVFP4 representation with stochastic rounding.

        Note:
            Use this for gradients and backward computations.
            Use from_float32() for forward passes and weights.

        Example:
            ```mojo
            # Gradient value 1.25 between 1.0 and 1.5.
            # Will round to 1.5 with ~50% probability.
            var grad = NVFP4.from_float32_stochastic(1.25, seed=12345)
        ```
        """
        # Handle special cases
        if isnan(x) or isinf(x):
            return NVFP4(FP4_E2M1.from_float32(x, scale=1.0), E4M3Scale(0x38))

        if x == 0.0:
            return NVFP4(FP4_E2M1(0), E4M3Scale(0x38))

        # Compute scale same as deterministic version
        var abs_x = x if x > 0 else -x
        var scale_val = Float32(1.0)
        var exp_val = 0

        while abs_x / scale_val > 6.0:
            scale_val *= 2.0
            exp_val += 1

        while abs_x / scale_val < 0.5 and exp_val > -7:
            scale_val /= 2.0
            exp_val -= 1

        var scale = E4M3Scale.from_float32(scale_val)
        var scale_f32 = scale.to_float32()

        # Stochastic rounding for E2M1 encoding
        var value = NVFP4._fp4_stochastic_round(x, scale_f32, seed)

        return NVFP4(value, scale)

    @staticmethod
    fn _fp4_stochastic_round(
        x: Float32, scale: Float32, seed: UInt64
    ) -> FP4_E2M1:
        """Internal: Stochastic rounding helper using simple LCG.

        Args:
            x: Float32 value to encode.
            scale: Scale factor.
            seed: Random seed.

        Returns:
            FP4_E2M1 value with stochastic rounding.
        """
        var scaled = x / scale
        var sign: UInt8 = 0
        var abs_scaled = scaled

        if scaled < 0:
            sign = 1
            abs_scaled = -scaled

        # E2M1 representable values: 0, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
        # Find neighboring values and distance
        var lower: Float32
        var upper: Float32
        var lower_bits: UInt8
        var upper_bits: UInt8

        # E2M1 representable values: 0, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
        # Boundaries use actual representable values to determine bucket
        if abs_scaled < 0.5:
            # Close to zero - round to 0
            return FP4_E2M1(sign << 3)
        elif abs_scaled < 1.0:
            # Between 0 and 1.0 - stochastic round between them
            lower = 0.0
            upper = 1.0
            lower_bits = 0b000  # zero
            upper_bits = 0b010  # exp=1, mantissa=0 (1.0)
        elif abs_scaled < 1.5:
            lower = 1.0
            upper = 1.5
            lower_bits = 0b010  # exp=1, mantissa=0
            upper_bits = 0b011  # exp=1, mantissa=1
        elif abs_scaled < 2.0:
            lower = 1.5
            upper = 2.0
            lower_bits = 0b011
            upper_bits = 0b100  # exp=2, mantissa=0
        elif abs_scaled < 3.0:
            lower = 2.0
            upper = 3.0
            lower_bits = 0b100
            upper_bits = 0b101  # exp=2, mantissa=1
        elif abs_scaled < 4.0:
            lower = 3.0
            upper = 4.0
            lower_bits = 0b101
            upper_bits = 0b110  # exp=3, mantissa=0
        elif abs_scaled < 6.0:
            lower = 4.0
            upper = 6.0
            lower_bits = 0b110
            upper_bits = 0b111  # exp=3, mantissa=1
        else:
            # At or above max
            return FP4_E2M1((sign << 3) | 0b111)

        # Compute probability of rounding up
        var distance = abs_scaled - lower
        var range_val = upper - lower
        var prob_up = distance / range_val

        # Simple LCG: seed = (1103515245 * seed + 12345) mod 2^32
        var rng_state = seed.cast[DType.uint32]()
        rng_state = (1103515245 * rng_state + 12345) & 0xFFFFFFFF
        var random_val = Float32(rng_state >> 8) / Float32(16777216.0)

        # Stochastic decision
        var result_bits: UInt8
        if random_val < prob_up:
            result_bits = upper_bits
        else:
            result_bits = lower_bits

        return FP4_E2M1((sign << 3) | result_bits)

    fn to_float32(self) -> Float32:
        """Convert NVFP4 to Float32.

        Returns:
            Float32 representation.
        """
        return self.value.to_float32(scale=self.scale.to_float32())

    fn __add__(self, other: NVFP4) -> NVFP4:
        """Add two NVFP4 values (via Float32).

        Args:
            other: Value to add.

        Returns:
            Sum as NVFP4.
        """
        return NVFP4.from_float32(self.to_float32() + other.to_float32())

    fn __sub__(self, other: NVFP4) -> NVFP4:
        """Subtract two NVFP4 values (via Float32).

        Args:
            other: Value to subtract.

        Returns:
            Difference as NVFP4.
        """
        return NVFP4.from_float32(self.to_float32() - other.to_float32())

    fn __mul__(self, other: NVFP4) -> NVFP4:
        """Multiply two NVFP4 values (via Float32).

        Args:
            other: Value to multiply.

        Returns:
            Product as NVFP4.
        """
        return NVFP4.from_float32(self.to_float32() * other.to_float32())

    fn __truediv__(self, other: NVFP4) -> NVFP4:
        """Divide two NVFP4 values (via Float32).

        Args:
            other: Divisor.

        Returns:
            Quotient as NVFP4.
        """
        return NVFP4.from_float32(self.to_float32() / other.to_float32())

    fn __neg__(self) -> NVFP4:
        """Negate NVFP4 value.

        Returns:
            Negated value.
        """
        # Flip sign bit in E2M1 value
        var neg_value = FP4_E2M1(self.value.value ^ 0b1000)
        return NVFP4(neg_value, self.scale)

    fn __eq__(self, other: NVFP4) -> Bool:
        """Check equality.

        Args:
            other: Value to compare.

        Returns:
            True if equal.
        """
        return (
            self.value == other.value and self.scale.value == other.scale.value
        )

    fn __ne__(self, other: NVFP4) -> Bool:
        """Check inequality.

        Args:
            other: Value to compare.

        Returns:
            True if not equal.
        """
        return not (self == other)

    fn __lt__(self, other: NVFP4) -> Bool:
        """Check less than.

        Args:
            other: Value to compare.

        Returns:
            True if self < other.
        """
        return self.to_float32() < other.to_float32()

    fn __le__(self, other: NVFP4) -> Bool:
        """Check less than or equal.

        Args:
            other: Value to compare.

        Returns:
            True if self <= other.
        """
        return self.to_float32() <= other.to_float32()

    fn __gt__(self, other: NVFP4) -> Bool:
        """Check greater than.

        Args:
            other: Value to compare.

        Returns:
            True if self > other.
        """
        return self.to_float32() > other.to_float32()

    fn __ge__(self, other: NVFP4) -> Bool:
        """Check greater than or equal.

        Args:
            other: Value to compare.

        Returns:
            True if self >= other.
        """
        return self.to_float32() >= other.to_float32()

    fn __str__(self) -> String:
        """Convert to string.

        Returns:
            String representation.
        """
        return "NVFP4(" + String(self.to_float32()) + ")"

    fn __repr__(self) -> String:
        """Get representation string.

        Returns:
            Representation string.
        """
        return (
            "NVFP4(value="
            + repr(self.value)
            + ", scale="
            + repr(self.scale)
            + ")"
        )


struct NVFP4Block(Copyable, Movable, Representable, Stringable):
    """NVFP4 block storage: 16 E2M1 values + 1 E4M3 scale (9 bytes total).

    Memory layout:
    - Bytes 0-7: 16 E2M1 values (4 bits each, packed 2 per byte)
    - Byte 8: E4M3 scale (7 bits used, 1 bit unused)

    Bit packing:
    Each byte stores 2 E2M1 values:
    - Upper 4 bits: First E2M1 value
    - Lower 4 bits: Second E2M1 value

    This provides 14:1 compression vs Float32 (9 bytes vs 64 bytes).
    Smaller blocks (16 vs 32) provide better accuracy per the paper.

    Example:
        ```mojo
        from collections import List

        # Create block from Float32 array
        var values = List[Float32]()
        for i in range(16):
            values.append(Float32(i) * 0.1)

        var block = NVFP4Block.from_float32_array(values)
        var decoded = block.to_float32_array()
        ```
    """

    var data: SIMD[DType.uint8, 8]
    """8 bytes containing 16 packed E2M1 values (2 per byte)."""
    var scale: E4M3Scale
    """Shared E4M3 scale factor for all 16 values."""

    fn __init__(out self):
        """Initialize NVFP4Block with zeros."""
        self.data = SIMD[DType.uint8, 8](0)
        self.scale = E4M3Scale(0x38)  # Scale = 1.0

    fn __init__(out self, data: SIMD[DType.uint8, 8], scale: E4M3Scale):
        """Initialize NVFP4Block from packed data and scale.

        Args:
            data: 8 bytes containing 16 packed E2M1 values.
            scale: E4M3 scale factor for the block.
        """
        self.data = data
        self.scale = scale.copy()

    @staticmethod
    fn from_float32_array(values: List[Float32]) raises -> Self:
        """Convert 16 Float32 values to NVFP4Block.

        Args:
            values: List of exactly 16 Float32 values.

        Returns:
            NVFP4Block with optimal scale and packed E2M1 values.

        Raises:
            Error: If values list doesn't contain exactly 16 elements.

        Note:
            Computes optimal E4M3 scale as max(abs(values)) / 6.0
            to fit all values in E2M1 range [0, 6].
        """
        if len(values) != 16:
            raise Error(
                "NVFP4Block requires exactly 16 values, got "
                + String(len(values))
            )

        # Find optimal scale: max(abs(values)) / 6.0
        var max_abs = Float32(0.0)
        for i in range(16):
            var abs_val = values[i] if values[i] >= 0 else -values[i]
            if abs_val > max_abs:
                max_abs = abs_val

        # Compute scale (avoid division by zero)
        var scale_val = max_abs / 6.0
        if scale_val < 1e-10:
            scale_val = 1.0

        var scale = E4M3Scale.from_float32(scale_val)
        var scale_f32 = scale.to_float32()

        # Pack E2M1 values (2 per byte)
        var data = SIMD[DType.uint8, 8](0)
        for i in range(8):
            # First value (upper 4 bits)
            var val1 = FP4_E2M1.from_float32(values[i * 2], scale=scale_f32)
            # Second value (lower 4 bits)
            var val2 = FP4_E2M1.from_float32(values[i * 2 + 1], scale=scale_f32)

            # Pack: upper 4 bits = val1, lower 4 bits = val2
            data[i] = ((val1.value & 0xF) << 4) | (val2.value & 0xF)

        return NVFP4Block(data, scale)

    fn to_float32_array(self) -> List[Float32]:
        """Decode NVFP4Block to 16 Float32 values.

        Returns:
            List of 16 Float32 values decoded from the block.

        Note:
            Decoding is lossless given the quantization that occurred during encoding.
        """
        var result = List[Float32]()
        var scale_f32 = self.scale.to_float32()

        for i in range(8):
            var byte = self.data[i]
            # Extract upper 4 bits (first value)
            var val1 = FP4_E2M1((byte >> 4) & 0xF)
            result.append(val1.to_float32(scale=scale_f32))

            # Extract lower 4 bits (second value)
            var val2 = FP4_E2M1(byte & 0xF)
            result.append(val2.to_float32(scale=scale_f32))

        return result^

    fn get(self, index: Int) raises -> NVFP4:
        """Get NVFP4 value at index (0-15).

        Args:
            index: Index in range [0, 15].

        Returns:
            NVFP4 value at the given index.

        Raises:
            Error: If index is out of range.
        """
        if index < 0 or index >= 16:
            raise Error("Index " + String(index) + " out of range [0, 15]")

        var byte_idx = index // 2
        var is_upper = (index % 2) == 0

        var byte = self.data[byte_idx]
        var fp4_val: FP4_E2M1
        if is_upper:
            fp4_val = FP4_E2M1((byte >> 4) & 0xF)
        else:
            fp4_val = FP4_E2M1(byte & 0xF)

        return NVFP4(fp4_val, self.scale)

    fn set(mut self, index: Int, value: NVFP4) raises -> None:
        """Set NVFP4 value at index (0-15).

        Args:
            index: Index in range [0, 15].
            value: NVFP4 value to set.

        Raises:
            Error: If index is out of range.

        Note:
            This updates the E2M1 value but keeps the block's shared scale.
            The value's scale is ignored - it's re-encoded with the block's scale.
        """
        if index < 0 or index >= 16:
            raise Error("Index " + String(index) + " out of range [0, 15]")

        # Re-encode value with block's scale
        var float_val = value.to_float32()
        var fp4_val = FP4_E2M1.from_float32(
            float_val, scale=self.scale.to_float32()
        )

        var byte_idx = index // 2
        var is_upper = (index % 2) == 0

        var byte = self.data[byte_idx]
        if is_upper:
            # Update upper 4 bits
            byte = (byte & 0x0F) | ((fp4_val.value & 0xF) << 4)
        else:
            # Update lower 4 bits
            byte = (byte & 0xF0) | (fp4_val.value & 0xF)

        self.data[byte_idx] = byte

    fn __str__(self) -> String:
        """String representation showing scale and value count.

        Returns:
            String representation.
        """
        return (
            "NVFP4Block(16 values, scale="
            + String(self.scale.to_float32())
            + ")"
        )

    fn __repr__(self) -> String:
        """Detailed representation.

        Returns:
            Detailed string representation.
        """
        return "NVFP4Block(scale=" + repr(self.scale) + ", data=8 bytes)"


def main():
    """Entry point for standalone compilation.

    This function exists solely to allow standalone compilation validation.
    In normal usage, this module is imported as part of the shared.core.types
    package and this function is never called.
    """
    print("shared.core.types.nvfp4 module loaded successfully")
    print("Available types: NVFP4, E4M3Scale, NVFP4Block")
