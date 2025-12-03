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
    - FP4_E2M1: 4-bit floating point base type (E2M1 format)
    - MXFP4: Microscaling FP4 with E8M0 scaling (32-element blocks)
    - NVFP4: NVIDIA FP4 with E4M3 scaling (16-element blocks)
    - Int8, Int16, Int32, Int64: Signed integer types
    - UInt8, UInt16, UInt32, UInt64: Unsigned integer types

Example:.    from shared.core.types import Tensor, Shape, DType, FP8, BF8
    from shared.core.types import FP4_E2M1, MXFP4, NVFP4
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

    # Work with blocked FP4 values
    var mxfp4_val = MXFP4.from_float32(2.718)
    var nvfp4_val = NVFP4.from_float32(1.414)
    var reconstructed = mxfp4_val.to_float32()

    # Work with integer types
    var i8 = Int8(42)
    var u32 = UInt32.from_float32(255.5)
"""

# Type exports
from .fp8 import FP8
from .bf8 import BF8
from .fp4 import FP4_E2M1
from .mxfp4 import MXFP4, E8M0Scale
from .nvfp4 import NVFP4, E4M3Scale
# TODO: Fix integer/unsigned types - they have compilation errors
# from .integer import Int8, Int16, Int32, Int64
# from .unsigned import UInt8, UInt16, UInt32, UInt64

# Future exports will be added here as components are implemented
# from .tensor import Tensor
# from .shape() import Shape
# from .dtype import DType
