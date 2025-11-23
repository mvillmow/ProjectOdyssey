"""Signed integer type wrappers (Int8, Int16, Int32, Int64).

This module provides type-safe wrappers for signed integer types with comprehensive
conversion methods and operator support. These types integrate with the ExTensor system
and provide a consistent interface for integer operations.

Examples:
    ```mojo.
    from shared.core.types.integer import Int8, Int16, Int32, Int64

    # Create integer values
    var i8 = Int8(42)
    var i16 = Int16.from_int32(1000)
    var i32 = Int32(-500)
    var i64 = Int64.from_float32(3.14)

    # Convert between types
    var as_float = i8.to_float32()  # 42.0
    var as_i32 = i8.to_int32()      # 42

    # Perform operations
    var sum = i8 + Int8(10)         # 52
    var product = i16 * Int16(2)    # 2000
    ```
"""

from math import trunc


@fieldwise_init
struct Int8(Stringable, Representable, Copyable, Movable):
    """8-bit signed integer type (-128 to 127)."""

    var value: Int8  # Using builtin Int8 for storage

    # ===----------------------------------------------------------------------===#
    # Constructors
    # ===----------------------------------------------------------------------===#

    fn __init__(out self, value: Int8):
        """Initialize from Int8 value.

        Args:
            value: The 8-bit signed integer value.
        """
        self.value = value

    fn __init__(out self, value: Int):
        """Initialize from Int value with clamping.

        Args:
            value: The integer value (clamped to -128..127).
        """
        self.value = Self._clamp_int8(value)

    # ===----------------------------------------------------------------------===#
    # Conversion Methods
    # ===----------------------------------------------------------------------===#

    @staticmethod
    fn from_float32(value: Float32) -> Self:
        """Convert from Float32 to Int8 with truncation.

        Args:
            value: Float32 value to convert.

        Returns:
            Int8 with truncated and clamped value.
        """
        var int_val = Int(trunc(value))
        return Self(Self._clamp_int8(int_val))

    @staticmethod
    fn from_int16(value: Int16) -> Self:
        """Convert from Int16 to Int8 with clamping.

        Args:
            value: Int16 value to convert.

        Returns:
            Int8 with clamped value.
        """
        return Self(Int(value))

    @staticmethod
    fn from_int32(value: Int32) -> Self:
        """Convert from Int32 to Int8 with clamping.

        Args:
            value: Int32 value to convert.

        Returns:
            Int8 with clamped value.
        """
        return Self(Int(value))

    @staticmethod
    fn from_int64(value: Int64) -> Self:
        """Convert from Int64 to Int8 with clamping.

        Args:
            value: Int64 value to convert.

        Returns:
            Int8 with clamped value.
        """
        return Self(Int(value))

    fn to_float32(self) -> Float32:
        """Convert to Float32.

        Returns:
            Float32 representation of the value.
        """
        return Float32(Int(self.value))

    fn to_int16(self) -> Int16:
        """Convert to Int16 (lossless).

        Returns:
            Int16 representation of the value.
        """
        return Int16(self.value)

    fn to_int32(self) -> Int32:
        """Convert to Int32 (lossless).

        Returns:
            Int32 representation of the value.
        """
        return Int32(self.value)

    fn to_int64(self) -> Int64:
        """Convert to Int64 (lossless).

        Returns:
            Int64 representation of the value.
        """
        return Int64(self.value)

    # ===----------------------------------------------------------------------===#
    # Helper Methods
    # ===----------------------------------------------------------------------===#

    @staticmethod
    fn _clamp_int8(value: Int) -> Int8:
        """Clamp integer value to Int8 range (-128 to 127).

        Args:
            value: Integer value to clamp.

        Returns:
            Clamped Int8 value.
        """
        if value < -128:
            return Int8(-128)
        elif value > 127:
            return Int8(127)
        else:
            return Int8(value)

    # ===----------------------------------------------------------------------===#
    # String Representation
    # ===----------------------------------------------------------------------===#

    fn __str__(self) -> String:
        """String representation.

        Returns:
            String representation of the value.
        """
        return String(Int(self.value))

    fn __repr__(self) -> String:
        """Detailed representation.

        Returns:
            Detailed string representation.
        """
        return "Int8(" + String(Int(self.value)) + ")"

    # ===----------------------------------------------------------------------===#
    # Comparison Operators
    # ===----------------------------------------------------------------------===#

    fn __eq__(self, other: Self) -> Bool:
        """Equality comparison.

        Args:
            other: Value to compare.

        Returns:
            True if equal.
        """
        return self.value == other.value

    fn __ne__(self, other: Self) -> Bool:
        """Inequality comparison.

        Args:
            other: Value to compare.

        Returns:
            True if not equal.
        """
        return self.value != other.value

    fn __lt__(self, other: Self) -> Bool:
        """Less than comparison.

        Args:
            other: Value to compare.

        Returns:
            True if self < other.
        """
        return self.value < other.value

    fn __le__(self, other: Self) -> Bool:
        """Less than or equal comparison.

        Args:
            other: Value to compare.

        Returns:
            True if self <= other.
        """
        return self.value <= other.value

    fn __gt__(self, other: Self) -> Bool:
        """Greater than comparison.

        Args:
            other: Value to compare.

        Returns:
            True if self > other.
        """
        return self.value > other.value

    fn __ge__(self, other: Self) -> Bool:
        """Greater than or equal comparison.

        Args:
            other: Value to compare.

        Returns:
            True if self >= other.
        """
        return self.value >= other.value

    # ===----------------------------------------------------------------------===#
    # Arithmetic Operators
    # ===----------------------------------------------------------------------===#

    fn __add__(self, other: Self) -> Self:
        """Addition with overflow wrapping.

        Args:
            other: Value to add.

        Returns:
            Sum (with wrapping on overflow).
        """
        return Self(Int(self.value) + Int(other.value))

    fn __sub__(self, other: Self) -> Self:
        """Subtraction with overflow wrapping.

        Args:
            other: Value to subtract.

        Returns:
            Difference (with wrapping on overflow).
        """
        return Self(Int(self.value) - Int(other.value))

    fn __mul__(self, other: Self) -> Self:
        """Multiplication with overflow wrapping.

        Args:
            other: Value to multiply.

        Returns:
            Product (with wrapping on overflow).
        """
        return Self(Int(self.value) * Int(other.value))

    fn __truediv__(self, other: Self) -> Self:
        """Division (truncating).

        Args:
            other: Divisor.

        Returns:
            Quotient (truncated).
        """
        return Self(Int(self.value) // Int(other.value))

    fn __mod__(self, other: Self) -> Self:
        """Modulo operation.

        Args:
            other: Divisor.

        Returns:
            Remainder.
        """
        return Self(Int(self.value) % Int(other.value))

    fn __neg__(self) -> Self:
        """Negation.

        Returns:
            Negated value.
        """
        return Self(-Int(self.value))


@fieldwise_init
struct Int16(Stringable, Representable, Copyable, Movable):
    """16-bit signed integer type (-32768 to 32767)."""

    var value: Int16

    fn __init__(out self, value: Int16):
        """Initialize from Int16 value."""
        self.value = value

    fn __init__(out self, value: Int):
        """Initialize from Int value with clamping."""
        self.value = Self._clamp_int16(value)

    @staticmethod
    fn from_float32(value: Float32) -> Self:
        """Convert from Float32 to Int16."""
        var int_val = Int(trunc(value))
        return Self(Self._clamp_int16(int_val))

    @staticmethod
    fn from_int8(value: Int8) -> Self:
        """Convert from Int8 to Int16 (lossless)."""
        return Self(Int16(value))

    @staticmethod
    fn from_int32(value: Int32) -> Self:
        """Convert from Int32 to Int16 with clamping."""
        return Self(Int(value))

    @staticmethod
    fn from_int64(value: Int64) -> Self:
        """Convert from Int64 to Int16 with clamping."""
        return Self(Int(value))

    fn to_float32(self) -> Float32:
        """Convert to Float32."""
        return Float32(Int(self.value))

    fn to_int8(self) -> Int8:
        """Convert to Int8 with clamping."""
        return Int8._clamp_int8(Int(self.value))

    fn to_int32(self) -> Int32:
        """Convert to Int32 (lossless)."""
        return Int32(self.value)

    fn to_int64(self) -> Int64:
        """Convert to Int64 (lossless)."""
        return Int64(self.value)

    @staticmethod
    fn _clamp_int16(value: Int) -> Int16:
        """Clamp integer value to Int16 range."""
        if value < -32768:
            return Int16(-32768)
        elif value > 32767:
            return Int16(32767)
        else:
            return Int16(value)

    fn __str__(self) -> String:
        return String(Int(self.value))

    fn __repr__(self) -> String:
        return "Int16(" + String(Int(self.value)) + ")"

    fn __eq__(self, other: Self) -> Bool:
        return self.value == other.value

    fn __ne__(self, other: Self) -> Bool:
        return self.value != other.value

    fn __lt__(self, other: Self) -> Bool:
        return self.value < other.value

    fn __le__(self, other: Self) -> Bool:
        return self.value <= other.value

    fn __gt__(self, other: Self) -> Bool:
        return self.value > other.value

    fn __ge__(self, other: Self) -> Bool:
        return self.value >= other.value

    fn __add__(self, other: Self) -> Self:
        return Self(Int(self.value) + Int(other.value))

    fn __sub__(self, other: Self) -> Self:
        return Self(Int(self.value) - Int(other.value))

    fn __mul__(self, other: Self) -> Self:
        return Self(Int(self.value) * Int(other.value))

    fn __truediv__(self, other: Self) -> Self:
        return Self(Int(self.value) // Int(other.value))

    fn __mod__(self, other: Self) -> Self:
        return Self(Int(self.value) % Int(other.value))

    fn __neg__(self) -> Self:
        return Self(-Int(self.value))


@fieldwise_init
struct Int32(Stringable, Representable, Copyable, Movable):
    """32-bit signed integer type (-2147483648 to 2147483647)."""

    var value: Int32

    fn __init__(out self, value: Int32):
        """Initialize from Int32 value."""
        self.value = value

    fn __init__(out self, value: Int):
        """Initialize from Int value with clamping."""
        self.value = Self._clamp_int32(value)

    @staticmethod
    fn from_float32(value: Float32) -> Self:
        """Convert from Float32 to Int32."""
        var int_val = Int(trunc(value))
        return Self(Self._clamp_int32(int_val))

    @staticmethod
    fn from_int8(value: Int8) -> Self:
        """Convert from Int8 to Int32 (lossless)."""
        return Self(Int32(value))

    @staticmethod
    fn from_int16(value: Int16) -> Self:
        """Convert from Int16 to Int32 (lossless)."""
        return Self(Int32(value))

    @staticmethod
    fn from_int64(value: Int64) -> Self:
        """Convert from Int64 to Int32 with clamping."""
        return Self(Int(value))

    fn to_float32(self) -> Float32:
        """Convert to Float32."""
        return Float32(Int(self.value))

    fn to_int8(self) -> Int8:
        """Convert to Int8 with clamping."""
        return Int8._clamp_int8(Int(self.value))

    fn to_int16(self) -> Int16:
        """Convert to Int16 with clamping."""
        return Int16._clamp_int16(Int(self.value))

    fn to_int64(self) -> Int64:
        """Convert to Int64 (lossless)."""
        return Int64(self.value)

    @staticmethod
    fn _clamp_int32(value: Int) -> Int32:
        """Clamp integer value to Int32 range."""
        if value < -2147483648:
            return Int32(-2147483648)
        elif value > 2147483647:
            return Int32(2147483647)
        else:
            return Int32(value)

    fn __str__(self) -> String:
        return String(Int(self.value))

    fn __repr__(self) -> String:
        return "Int32(" + String(Int(self.value)) + ")"

    fn __eq__(self, other: Self) -> Bool:
        return self.value == other.value

    fn __ne__(self, other: Self) -> Bool:
        return self.value != other.value

    fn __lt__(self, other: Self) -> Bool:
        return self.value < other.value

    fn __le__(self, other: Self) -> Bool:
        return self.value <= other.value

    fn __gt__(self, other: Self) -> Bool:
        return self.value > other.value

    fn __ge__(self, other: Self) -> Bool:
        return self.value >= other.value

    fn __add__(self, other: Self) -> Self:
        return Self(Int(self.value) + Int(other.value))

    fn __sub__(self, other: Self) -> Self:
        return Self(Int(self.value) - Int(other.value))

    fn __mul__(self, other: Self) -> Self:
        return Self(Int(self.value) * Int(other.value))

    fn __truediv__(self, other: Self) -> Self:
        return Self(Int(self.value) // Int(other.value))

    fn __mod__(self, other: Self) -> Self:
        return Self(Int(self.value) % Int(other.value))

    fn __neg__(self) -> Self:
        return Self(-Int(self.value))


@fieldwise_init
struct Int64(Stringable, Representable, Copyable, Movable):
    """64-bit signed integer type (-9223372036854775808 to 9223372036854775807)."""

    var value: Int64

    fn __init__(out self, value: Int64):
        """Initialize from Int64 value."""
        self.value = value

    fn __init__(out self, value: Int):
        """Initialize from Int value."""
        self.value = Int64(value)

    @staticmethod
    fn from_float32(value: Float32) -> Self:
        """Convert from Float32 to Int64."""
        return Self(Int64(trunc(value)))

    @staticmethod
    fn from_int8(value: Int8) -> Self:
        """Convert from Int8 to Int64 (lossless)."""
        return Self(Int64(value))

    @staticmethod
    fn from_int16(value: Int16) -> Self:
        """Convert from Int16 to Int64 (lossless)."""
        return Self(Int64(value))

    @staticmethod
    fn from_int32(value: Int32) -> Self:
        """Convert from Int32 to Int64 (lossless)."""
        return Self(Int64(value))

    fn to_float32(self) -> Float32:
        """Convert to Float32 (may lose precision)."""
        return Float32(Int(self.value))

    fn to_int8(self) -> Int8:
        """Convert to Int8 with clamping."""
        return Int8._clamp_int8(Int(self.value))

    fn to_int16(self) -> Int16:
        """Convert to Int16 with clamping."""
        return Int16._clamp_int16(Int(self.value))

    fn to_int32(self) -> Int32:
        """Convert to Int32 with clamping."""
        return Int32._clamp_int32(Int(self.value))

    fn __str__(self) -> String:
        return String(Int(self.value))

    fn __repr__(self) -> String:
        return "Int64(" + String(Int(self.value)) + ")"

    fn __eq__(self, other: Self) -> Bool:
        return self.value == other.value

    fn __ne__(self, other: Self) -> Bool:
        return self.value != other.value

    fn __lt__(self, other: Self) -> Bool:
        return self.value < other.value

    fn __le__(self, other: Self) -> Bool:
        return self.value <= other.value

    fn __gt__(self, other: Self) -> Bool:
        return self.value > other.value

    fn __ge__(self, other: Self) -> Bool:
        return self.value >= other.value

    fn __add__(self, other: Self) -> Self:
        return Self(Int(self.value) + Int(other.value))

    fn __sub__(self, other: Self) -> Self:
        return Self(Int(self.value) - Int(other.value))

    fn __mul__(self, other: Self) -> Self:
        return Self(Int(self.value) * Int(other.value))

    fn __truediv__(self, other: Self) -> Self:
        return Self(Int(self.value) // Int(other.value))

    fn __mod__(self, other: Self) -> Self:
        return Self(Int(self.value) % Int(other.value))

    fn __neg__(self) -> Self:
        return Self(-Int(self.value))
