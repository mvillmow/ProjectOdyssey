"""Unsigned integer type wrappers (UInt8, UInt16, UInt32, UInt64).

This module provides type-safe wrappers for unsigned integer types with comprehensive
conversion methods and operator support. These types integrate with the ExTensor system
and provide a consistent interface for unsigned integer operations.

Examples:
    ```mojo
    from shared.core.types.unsigned import UInt8, UInt16, UInt32, UInt64

    # Create unsigned integer values
    var u8 = UInt8(255)
    var u16 = UInt16.from_int32(1000)
    var u32 = UInt32(500)
    var u64 = UInt64.from_float32(3.14)

    # Convert between types
    var as_float = u8.to_float32()  # 255.0
    var as_u32 = u8.to_uint32()     # 255

    # Perform operations
    var sum = u8 + UInt8(10)        # 9 (wrapping overflow)
    var product = u16 * UInt16(2)   # 2000
    ```
"""

from math import trunc


@value
struct UInt8(Stringable, Representable):
    """8-bit unsigned integer type (0 to 255)."""

    var value: UInt8  # Using builtin UInt8 for storage

    # ===----------------------------------------------------------------------===#
    # Constructors
    # ===----------------------------------------------------------------------===#

    fn __init__(inout self, value: UInt8):
        """Initialize from UInt8 value.

        Args:
            value: The 8-bit unsigned integer value.
        """
        self.value = value

    fn __init__(inout self, value: Int):
        """Initialize from Int value with clamping.

        Args:
            value: The integer value (clamped to 0..255).
        """
        self.value = Self._clamp_uint8(value)

    # ===----------------------------------------------------------------------===#
    # Conversion Methods
    # ===----------------------------------------------------------------------===#

    @staticmethod
    fn from_float32(value: Float32) -> Self:
        """Convert from Float32 to UInt8 with truncation.

        Args:
            value: Float32 value to convert.

        Returns:
            UInt8 with truncated and clamped value.
        """
        var int_val = Int(trunc(value))
        return Self(Self._clamp_uint8(int_val))

    @staticmethod
    fn from_uint16(value: UInt16) -> Self:
        """Convert from UInt16 to UInt8 with clamping.

        Args:
            value: UInt16 value to convert.

        Returns:
            UInt8 with clamped value.
        """
        return Self(Int(value))

    @staticmethod
    fn from_uint32(value: UInt32) -> Self:
        """Convert from UInt32 to UInt8 with clamping.

        Args:
            value: UInt32 value to convert.

        Returns:
            UInt8 with clamped value.
        """
        return Self(Int(value))

    @staticmethod
    fn from_uint64(value: UInt64) -> Self:
        """Convert from UInt64 to UInt8 with clamping.

        Args:
            value: UInt64 value to convert.

        Returns:
            UInt8 with clamped value.
        """
        return Self(Int(value))

    fn to_float32(self) -> Float32:
        """Convert to Float32.

        Returns:
            Float32 representation of the value.
        """
        return Float32(Int(self.value))

    fn to_uint16(self) -> UInt16:
        """Convert to UInt16 (lossless).

        Returns:
            UInt16 representation of the value.
        """
        return UInt16(self.value)

    fn to_uint32(self) -> UInt32:
        """Convert to UInt32 (lossless).

        Returns:
            UInt32 representation of the value.
        """
        return UInt32(self.value)

    fn to_uint64(self) -> UInt64:
        """Convert to UInt64 (lossless).

        Returns:
            UInt64 representation of the value.
        """
        return UInt64(self.value)

    # ===----------------------------------------------------------------------===#
    # Helper Methods
    # ===----------------------------------------------------------------------===#

    @staticmethod
    fn _clamp_uint8(value: Int) -> UInt8:
        """Clamp integer value to UInt8 range (0 to 255).

        Args:
            value: Integer value to clamp.

        Returns:
            Clamped UInt8 value.
        """
        if value < 0:
            return UInt8(0)
        elif value > 255:
            return UInt8(255)
        else:
            return UInt8(value)

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
        return "UInt8(" + String(Int(self.value)) + ")"

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
        """Subtraction with underflow wrapping.

        Args:
            other: Value to subtract.

        Returns:
            Difference (with wrapping on underflow).
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


@value
struct UInt16(Stringable, Representable):
    """16-bit unsigned integer type (0 to 65535)."""

    var value: UInt16

    fn __init__(inout self, value: UInt16):
        """Initialize from UInt16 value."""
        self.value = value

    fn __init__(inout self, value: Int):
        """Initialize from Int value with clamping."""
        self.value = Self._clamp_uint16(value)

    @staticmethod
    fn from_float32(value: Float32) -> Self:
        """Convert from Float32 to UInt16."""
        var int_val = Int(trunc(value))
        return Self(Self._clamp_uint16(int_val))

    @staticmethod
    fn from_uint8(value: UInt8) -> Self:
        """Convert from UInt8 to UInt16 (lossless)."""
        return Self(UInt16(value))

    @staticmethod
    fn from_uint32(value: UInt32) -> Self:
        """Convert from UInt32 to UInt16 with clamping."""
        return Self(Int(value))

    @staticmethod
    fn from_uint64(value: UInt64) -> Self:
        """Convert from UInt64 to UInt16 with clamping."""
        return Self(Int(value))

    fn to_float32(self) -> Float32:
        """Convert to Float32."""
        return Float32(Int(self.value))

    fn to_uint8(self) -> UInt8:
        """Convert to UInt8 with clamping."""
        return UInt8._clamp_uint8(Int(self.value))

    fn to_uint32(self) -> UInt32:
        """Convert to UInt32 (lossless)."""
        return UInt32(self.value)

    fn to_uint64(self) -> UInt64:
        """Convert to UInt64 (lossless)."""
        return UInt64(self.value)

    @staticmethod
    fn _clamp_uint16(value: Int) -> UInt16:
        """Clamp integer value to UInt16 range."""
        if value < 0:
            return UInt16(0)
        elif value > 65535:
            return UInt16(65535)
        else:
            return UInt16(value)

    fn __str__(self) -> String:
        return String(Int(self.value))

    fn __repr__(self) -> String:
        return "UInt16(" + String(Int(self.value)) + ")"

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


@value
struct UInt32(Stringable, Representable):
    """32-bit unsigned integer type (0 to 4294967295)."""

    var value: UInt32

    fn __init__(inout self, value: UInt32):
        """Initialize from UInt32 value."""
        self.value = value

    fn __init__(inout self, value: Int):
        """Initialize from Int value with clamping."""
        self.value = Self._clamp_uint32(value)

    @staticmethod
    fn from_float32(value: Float32) -> Self:
        """Convert from Float32 to UInt32."""
        var int_val = Int(trunc(value))
        return Self(Self._clamp_uint32(int_val))

    @staticmethod
    fn from_uint8(value: UInt8) -> Self:
        """Convert from UInt8 to UInt32 (lossless)."""
        return Self(UInt32(value))

    @staticmethod
    fn from_uint16(value: UInt16) -> Self:
        """Convert from UInt16 to UInt32 (lossless)."""
        return Self(UInt32(value))

    @staticmethod
    fn from_uint64(value: UInt64) -> Self:
        """Convert from UInt64 to UInt32 with clamping."""
        return Self(Int(value))

    fn to_float32(self) -> Float32:
        """Convert to Float32."""
        return Float32(Int(self.value))

    fn to_uint8(self) -> UInt8:
        """Convert to UInt8 with clamping."""
        return UInt8._clamp_uint8(Int(self.value))

    fn to_uint16(self) -> UInt16:
        """Convert to UInt16 with clamping."""
        return UInt16._clamp_uint16(Int(self.value))

    fn to_uint64(self) -> UInt64:
        """Convert to UInt64 (lossless)."""
        return UInt64(self.value)

    @staticmethod
    fn _clamp_uint32(value: Int) -> UInt32:
        """Clamp integer value to UInt32 range."""
        if value < 0:
            return UInt32(0)
        elif value > 4294967295:
            return UInt32(4294967295)
        else:
            return UInt32(value)

    fn __str__(self) -> String:
        return String(Int(self.value))

    fn __repr__(self) -> String:
        return "UInt32(" + String(Int(self.value)) + ")"

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


@value
struct UInt64(Stringable, Representable):
    """64-bit unsigned integer type (0 to 18446744073709551615)."""

    var value: UInt64

    fn __init__(inout self, value: UInt64):
        """Initialize from UInt64 value."""
        self.value = value

    fn __init__(inout self, value: Int):
        """Initialize from Int value with clamping."""
        self.value = Self._clamp_uint64(value)

    @staticmethod
    fn from_float32(value: Float32) -> Self:
        """Convert from Float32 to UInt64."""
        var int_val = Int(trunc(value))
        return Self(Self._clamp_uint64(int_val))

    @staticmethod
    fn from_uint8(value: UInt8) -> Self:
        """Convert from UInt8 to UInt64 (lossless)."""
        return Self(UInt64(value))

    @staticmethod
    fn from_uint16(value: UInt16) -> Self:
        """Convert from UInt16 to UInt64 (lossless)."""
        return Self(UInt64(value))

    @staticmethod
    fn from_uint32(value: UInt32) -> Self:
        """Convert from UInt32 to UInt64 (lossless)."""
        return Self(UInt64(value))

    fn to_float32(self) -> Float32:
        """Convert to Float32 (may lose precision)."""
        return Float32(Int(self.value))

    fn to_uint8(self) -> UInt8:
        """Convert to UInt8 with clamping."""
        return UInt8._clamp_uint8(Int(self.value))

    fn to_uint16(self) -> UInt16:
        """Convert to UInt16 with clamping."""
        return UInt16._clamp_uint16(Int(self.value))

    fn to_uint32(self) -> UInt32:
        """Convert to UInt32 with clamping."""
        return UInt32._clamp_uint32(Int(self.value))

    @staticmethod
    fn _clamp_uint64(value: Int) -> UInt64:
        """Clamp integer value to UInt64 range."""
        if value < 0:
            return UInt64(0)
        else:
            return UInt64(value)

    fn __str__(self) -> String:
        return String(Int(self.value))

    fn __repr__(self) -> String:
        return "UInt64(" + String(Int(self.value)) + ")"

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
