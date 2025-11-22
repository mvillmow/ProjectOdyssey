"""
Types and Data Structures Module.

This module contains custom types and data structures optimized for ML workloads.
All types are implemented using Mojo's struct system for memory safety and performance.

Components:
    - Tensor: Multi-dimensional array type with automatic memory management
    - Shape: Shape representation and utilities
    - DType: Data type definitions and conversions
    - Slice: Tensor slicing utilities
    - FP8: 8-bit floating point type (E4M3 format)
    - BF8: 8-bit floating point type (E5M2 format)
    - Int8, Int16, Int32, Int64: Signed integer types
    - UInt8, UInt16, UInt32, UInt64: Unsigned integer types

Example:
    from shared.core.types import Tensor, Shape, DType, FP8, BF8
    from shared.core.types import Int8, Int16, Int32, Int64
    from shared.core.types import UInt8, UInt16, UInt32, UInt64

    # Create a tensor with specified shape and dtype
    fn create_weights() -> Tensor:
        var shape = Shape(128, 784)
        return Tensor(shape, DType.float32)

    # Work with FP8 values
    var fp8_val = FP8.from_float32(3.14159)
    var float_val = fp8_val.to_float32()

    # Work with BF8 values
    var bf8_val = BF8.from_float32(100.0)
    var float_val2 = bf8_val.to_float32()

    # Work with integer types
    var i8 = Int8(42)
    var u32 = UInt32.from_float32(255.5)
"""

# Type exports
from .fp8 import FP8
from .bf8 import BF8
from .integer import Int8, Int16, Int32, Int64
from .unsigned import UInt8, UInt16, UInt32, UInt64

# Future exports will be added here as components are implemented
# from .tensor import Tensor
# from .shape import Shape
# from .dtype import DType
