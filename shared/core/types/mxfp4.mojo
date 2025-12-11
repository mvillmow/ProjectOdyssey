"""MXFP4 (Microscaling FP4) blocked floating point format.

Implements MXFP4 format from the paper "Microscaling Data Formats for Deep Learning":
- Individual values: E2M1 (1 sign, 2 exponent, 1 mantissa)
- Block size: 32 elements
- Scale format: E8M0 (8-bit exponent-only, no mantissa or sign)

MXFP4 provides wide dynamic range through exponent-only scaling.
Each block of 32 FP4 values shares a common E8M0 scale factor.

Key characteristics:
- Memory: 16 bytes per block (32 × 4 bits = 128 bits) + 1 byte scale = 17 bytes
- Dynamic range: Wide (E8M0 scale)
- Precision: Limited (E2M1 values)
- Use case: Memory-efficient ML training and inference

Example:
    from shared.core.types.mxfp4 import MXFP4, MXFP4Block

    # Individual value (stores both E2M1 and scale for convenience)
    var val = MXFP4.from_float32(3.14159)
    var reconstructed = val.to_float32()

    # Block storage (efficient: 32 values + 1 shared scale)
    var block = MXFP4Block.from_float32_array(data_array)
    ```

Reference:
    Tim Dettmers, Nolan Miller, Deepak Kapur, Luke Zettlemoyer
    "Microscaling Data Formats for Deep Learning."
    arXiv preprint arXiv:2310.10537, 2023
    https://arxiv.org/abs/2310.10537
    https://doi.org/10.48550/arXiv.2310.10537

    The paper demonstrates that microscaling (per-block scaling factors) enables
    efficient low-precision quantization (4-bit) while maintaining accuracy in
    deep learning training and inference. MXFP4 uses E8M0 scales (256 dynamic range)
    across 32-element blocks for maximum efficiency.
"""

from math import isnan, isinf
from .fp4 import FP4_E2M1


struct E8M0Scale(Copyable, Movable, Representable, Stringable):
    """8-bit exponent-only scale factor for MXFP4 blocks.

    Memory layout (1 byte):
    - Bits 7-0: Exponent (8 bits, bias = 127, same as Float32)
    - No sign bit (always positive)
    - No mantissa bits (implicit mantissa = 1.0)

    Represents: 2^(exponent - 127).

    Valid range: 2^-127 to 2^128.
    """

    var exponent: UInt8

    fn __init__(out self, exponent: UInt8 = 127):
        """Initialize E8M0 scale from raw exponent.

        Args:
            exponent: 8-bit exponent value (bias = 127).
        """
        self.exponent = exponent

    @staticmethod
    fn from_float32(scale: Float32) -> Self:
        """Compute E8M0 scale from Float32 value.

        Args:
            scale: Positive Float32 scale value.

        Returns:
            E8M0 representation.

        Note:
            Scale must be positive. Negative or zero values return minimum scale.
        """
        if scale <= 0.0 or isnan(scale):
            return E8M0Scale(0)  # Minimum scale (2^-127)

        if isinf(scale):
            return E8M0Scale(255)  # Maximum scale (2^128)

        # Find exponent: scale = 2^exp
        var exp_val = 0
        var s = scale

        # Scale to find the exponent
        while s >= 2.0 and exp_val < 128:
            s /= 2.0
            exp_val += 1

        while s < 1.0 and exp_val > -127:
            s *= 2.0
            exp_val -= 1

        # Apply bias (127)
        var biased_exp = exp_val + 127

        # Clamp to [0, 255]
        if biased_exp < 0:
            biased_exp = 0
        elif biased_exp > 255:
            biased_exp = 255

        return E8M0Scale(UInt8(biased_exp))

    fn to_float32(self) -> Float32:
        """Convert E8M0 scale to Float32.

        Returns:
            Float32 representation: 2^(exponent - 127).
        """
        var exponent = self.exponent.cast[DType.int32]() - 127

        # Compute 2^exponent
        var result = Float32(1.0)
        if exponent > 0:
            for _ in range(exponent):
                result *= 2.0
        elif exponent < 0:
            for _ in range(-exponent):
                result /= 2.0

        return result

    fn __str__(self) -> String:
        """String representation showing scale value.

        Returns:
            String representation.
        """
        return (
            "E8M0(exp="
            + String(self.exponent)
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


struct MXFP4(Copyable, Movable, Representable, Stringable):
    """MXFP4 individual value (E2M1 + E8M0 scale).

    Acts like FP16 but stores internally as 4-bit E2M1 value plus 8-bit E8M0 scale.
    This representation is convenient but NOT space-efficient (12 bits total vs 4 bits in blocks).

    For efficient storage, use MXFP4Block which amortizes the scale across 32 values.

    Attributes:
        value: 4-bit E2M1 encoded value.
        scale: 8-bit E8M0 scale factor.
    """

    var value: FP4_E2M1
    var scale: E8M0Scale

    fn __init__(
        out self, value: FP4_E2M1 = FP4_E2M1(), scale: E8M0Scale = E8M0Scale()
    ):
        """Initialize MXFP4 from E2M1 value and E8M0 scale.

        Args:
            value: E2M1 encoded value.
            scale: E8M0 scale factor.
        """
        self.value = value.copy()
        self.scale = scale.copy()

    @staticmethod
    fn from_float32(x: Float32) -> Self:
        """Convert Float32 to MXFP4.

        Computes optimal scale for the single value and encodes.

        Args:
            x: Float32 value to convert.

        Returns:
            MXFP4 representation.
        """
        # Handle special cases
        if isnan(x) or isinf(x):
            return MXFP4(FP4_E2M1.from_float32(x, scale=1.0), E8M0Scale(127))

        if x == 0.0:
            return MXFP4(FP4_E2M1(0), E8M0Scale(127))

        # Compute scale: find 2^k such that |x| / 2^k is in E2M1 range [0, 6]
        var abs_x = x if x > 0 else -x
        var scale_val = Float32(1.0)
        var exp_val = 0

        # Scale to fit in E2M1 range [0, 6]
        while abs_x / scale_val > 6.0:
            scale_val *= 2.0
            exp_val += 1

        while abs_x / scale_val < 0.5 and exp_val > -127:
            scale_val /= 2.0
            exp_val -= 1

        # Create E8M0 scale
        var scale = E8M0Scale.from_float32(scale_val)

        # Encode E2M1 value
        var value = FP4_E2M1.from_float32(x, scale=scale.to_float32())

        return MXFP4(value, scale)

    @staticmethod
    fn from_float32_stochastic(x: Float32, seed: UInt64) -> Self:
        """Convert Float32 to MXFP4 with stochastic rounding.

        Uses stochastic rounding which is recommended for gradient quantization.
        When a value falls between two representable FP4 values, probabilistically
        round up or down based on distance.

        Args:
            x: Float32 value to convert.
            seed: Random seed for deterministic stochastic rounding.

        Returns:
            MXFP4 representation with stochastic rounding.

        Note:
            Use this for gradients and backward computations.
            Use from_float32() for forward passes and weights.

        Example:
            ```mojo
             Gradient value 1.25 between 1.0 and 1.5.
            # Will round to 1.5 with ~50% probability.
            var grad = MXFP4.from_float32_stochastic(1.25, seed=12345)
        ```
        """
        # Handle special cases
        if isnan(x) or isinf(x):
            return MXFP4(FP4_E2M1.from_float32(x, scale=1.0), E8M0Scale(127))

        if x == 0.0:
            return MXFP4(FP4_E2M1(0), E8M0Scale(127))

        # Compute scale same as deterministic version
        var abs_x = x if x > 0 else -x
        var scale_val = Float32(1.0)
        var exp_val = 0

        while abs_x / scale_val > 6.0:
            scale_val *= 2.0
            exp_val += 1

        while abs_x / scale_val < 0.5 and exp_val > -127:
            scale_val /= 2.0
            exp_val -= 1

        var scale = E8M0Scale.from_float32(scale_val)
        var scale_f32 = scale.to_float32()

        # Stochastic rounding for E2M1 encoding
        var value = MXFP4._fp4_stochastic_round(x, scale_f32, seed)

        return MXFP4(value, scale)

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
        """Convert MXFP4 to Float32.

        Returns:
            Float32 representation.
        """
        return self.value.to_float32(scale=self.scale.to_float32())

    fn __add__(self, other: MXFP4) -> MXFP4:
        """Add two MXFP4 values (via Float32).

        Args:
            other: Value to add.

        Returns:
            Sum as MXFP4.
        """
        return MXFP4.from_float32(self.to_float32() + other.to_float32())

    fn __sub__(self, other: MXFP4) -> MXFP4:
        """Subtract two MXFP4 values (via Float32).

        Args:
            other: Value to subtract.

        Returns:
            Difference as MXFP4.
        """
        return MXFP4.from_float32(self.to_float32() - other.to_float32())

    fn __mul__(self, other: MXFP4) -> MXFP4:
        """Multiply two MXFP4 values (via Float32).

        Args:
            other: Value to multiply.

        Returns:
            Product as MXFP4.
        """
        return MXFP4.from_float32(self.to_float32() * other.to_float32())

    fn __truediv__(self, other: MXFP4) -> MXFP4:
        """Divide two MXFP4 values (via Float32).

        Args:
            other: Divisor.

        Returns:
            Quotient as MXFP4.
        """
        return MXFP4.from_float32(self.to_float32() / other.to_float32())

    fn __neg__(self) -> MXFP4:
        """Negate MXFP4 value.

        Returns:
            Negated value.
        """
        # Flip sign bit in E2M1 value
        var neg_value = FP4_E2M1(self.value.value ^ 0b1000)
        return MXFP4(neg_value, self.scale)

    fn __eq__(self, other: MXFP4) -> Bool:
        """Check equality.

        Args:
            other: Value to compare.

        Returns:
            True if equal.
        """
        return (
            self.value == other.value
            and self.scale.exponent == other.scale.exponent
        )

    fn __ne__(self, other: MXFP4) -> Bool:
        """Check inequality.

        Args:
            other: Value to compare.

        Returns:
            True if not equal.
        """
        return not (self == other)

    fn __lt__(self, other: MXFP4) -> Bool:
        """Check less than.

        Args:
            other: Value to compare.

        Returns:
            True if self < other.
        """
        return self.to_float32() < other.to_float32()

    fn __le__(self, other: MXFP4) -> Bool:
        """Check less than or equal.

        Args:
            other: Value to compare.

        Returns:
            True if self <= other.
        """
        return self.to_float32() <= other.to_float32()

    fn __gt__(self, other: MXFP4) -> Bool:
        """Check greater than.

        Args:
            other: Value to compare.

        Returns:
            True if self > other.
        """
        return self.to_float32() > other.to_float32()

    fn __ge__(self, other: MXFP4) -> Bool:
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
        return "MXFP4(" + String(self.to_float32()) + ")"

    fn __repr__(self) -> String:
        """Get representation string.

        Returns:
            Representation string.
        """
        return (
            "MXFP4(value="
            + repr(self.value)
            + ", scale="
            + repr(self.scale)
            + ")"
        )


struct MXFP4Block(Copyable, Movable, Representable, Stringable):
    """MXFP4 block storage: 32 E2M1 values + 1 E8M0 scale (17 bytes total).

    Memory layout:
    - Bytes 0-15: 32 E2M1 values (4 bits each, packed 2 per byte).
    - Byte 16: E8M0 scale (8-bit exponent).

    Bit packing:
    Each byte stores 2 E2M1 values:
    - Upper 4 bits: First E2M1 value.
    - Lower 4 bits: Second E2M1 value.

    This provides 16:1 compression vs Float32 (17 bytes vs 128 bytes).

    Example:
        ```
        from collections import List

        # Create block from Float32 array
        var values : List[Float32]()
        for i in range(32):
            values.append(Float32(i) * 0.1)

        var block = MXFP4Block.from_float32_array(values)
        var decoded = block.to_float32_array()
        ```
    """

    var data: SIMD[DType.uint8, 16]  # 32 E2M1 values (2 per byte)
    var scale: E8M0Scale  # Shared E8M0 scale

    fn __init__(out self):
        """Initialize MXFP4Block with zeros."""
        self.data = SIMD[DType.uint8, 16](0)
        self.scale = E8M0Scale(127)  # Scale = 1.0

    fn __init__(out self, data: SIMD[DType.uint8, 16], scale: E8M0Scale):
        """Initialize MXFP4Block from packed data and scale.

        Args:
            data: 16 bytes containing 32 packed E2M1 values.
            scale: E8M0 scale factor for the block.
        """
        self.data = data
        self.scale = scale.copy()

    @staticmethod
    fn from_float32_array(values: List[Float32]) raises -> Self:
        """Convert 32 Float32 values to MXFP4Block.

        Args:
            values: List of exactly 32 Float32 values.

        Returns:
            MXFP4Block with optimal scale and packed E2M1 values.

        Raises:
            Error: If values list doesn't contain exactly 32 elements.

        Note:
        ```
            Computes optimal E8M0 scale as max(abs(values)) / 6.0.
            to fit all values in E2M1 range [0, 6].
        ```
        """
        if len(values) != 32:
            raise Error(
                "MXFP4Block requires exactly 32 values, got "
                + String(len(values))
            )

        # Find optimal scale: max(abs(values)) / 6.0
        var max_abs = Float32(0.0)
        for i in range(32):
            var abs_val = values[i] if values[i] >= 0 else -values[i]
            if abs_val > max_abs:
                max_abs = abs_val

        # Compute scale (avoid division by zero)
        # **FIXME (#2379 - TEST-002 - P0 CRITICAL)**: Scale = 0 edge case untested
        # When all values in block are zero or near-zero (< 1e-10), we fallback to scale=1.0
        # This behavior is COMPLETELY UNTESTED. Missing test cases:
        #   1. Block with all zeros (should encode as scale=1.0, all E2M1 values = 0)
        #   2. Block with values < 1e-10 (should trigger fallback)
        #   3. E8M0Scale.from_float32(0.0) direct behavior
        #   4. Round-trip conversion: zeros -> MXFP4 -> zeros (verify lossless)
        # Impact: Zero blocks are common in ML (dead neurons, zero gradients)
        # Severity: BLOCKING - edge case must be tested before production use
        # See: COMPREHENSIVE_REVIEW_FINDINGS.md (TEST-002)
        var scale_val = max_abs / 6.0
        if scale_val < 1e-10:
            scale_val = 1.0

        var scale = E8M0Scale.from_float32(scale_val)
        var scale_f32 = scale.to_float32()

        # Pack E2M1 values (2 per byte)
        var data = SIMD[DType.uint8, 16](0)
        for i in range(16):
            # First value (upper 4 bits)
            var val1 = FP4_E2M1.from_float32(values[i * 2], scale=scale_f32)
            # Second value (lower 4 bits)
            var val2 = FP4_E2M1.from_float32(values[i * 2 + 1], scale=scale_f32)

            # Pack: upper 4 bits = val1, lower 4 bits = val2
            data[i] = ((val1.value & 0xF) << 4) | (val2.value & 0xF)

        return MXFP4Block(data, scale)

    fn to_float32_array(self) -> List[Float32]:
        """Decode MXFP4Block to 32 Float32 values.

        Returns:
            List of 32 Float32 values decoded from the block.

        Note:
            Decoding is lossless given the quantization that occurred during encoding.
        """
        var result = List[Float32]()
        var scale_f32 = self.scale.to_float32()

        for i in range(16):
            var byte = self.data[i]
            # Extract upper 4 bits (first value)
            var val1 = FP4_E2M1((byte >> 4) & 0xF)
            result.append(val1.to_float32(scale=scale_f32))

            # Extract lower 4 bits (second value)
            var val2 = FP4_E2M1(byte & 0xF)
            result.append(val2.to_float32(scale=scale_f32))

        return result^

    fn get(self, index: Int) raises -> MXFP4:
        """Get MXFP4 value at index (0-31).

        Args:
            index: Index in range [0, 31].

        Returns:
            MXFP4 value at the given index.

        Raises:
            Error: If index is out of range.
        """
        if index < 0 or index >= 32:
            raise Error("Index " + String(index) + " out of range [0, 31]")

        var byte_idx = index // 2
        var is_upper = (index % 2) == 0

        var byte = self.data[byte_idx]
        var fp4_val: FP4_E2M1
        if is_upper:
            fp4_val = FP4_E2M1((byte >> 4) & 0xF)
        else:
            fp4_val = FP4_E2M1(byte & 0xF)

        return MXFP4(fp4_val, self.scale)

    fn set(mut self, index: Int, value: MXFP4) raises -> None:
        """Set MXFP4 value at index (0-31).

        Args:
            index: Index in range [0, 31].
            value: MXFP4 value to set.

        Raises:
            Error: If index is out of range.

        Note:
            This updates the E2M1 value but keeps the block's shared scale.
            The value's scale is ignored - it's re-encoded with the block's scale.
        """
        if index < 0 or index >= 32:
            raise Error("Index " + String(index) + " out of range [0, 31]")

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
            "MXFP4Block(32 values, scale="
            + String(self.scale.to_float32())
            + ")"
        )

    fn __repr__(self) -> String:
        """Detailed representation.

        Returns:
            Detailed string representation.
        """
        return "MXFP4Block(scale=" + repr(self.scale) + ", data=16 bytes)"
