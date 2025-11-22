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

Example:
    from shared.core.types import Tensor, Shape, DType, FP8

    # Create a tensor with specified shape and dtype
    fn create_weights() -> Tensor:
        var shape = Shape(128, 784)
        return Tensor(shape, DType.float32)

    # Work with FP8 values
    var fp8_val = FP8.from_float32(3.14159)
    var float_val = fp8_val.to_float32()
"""

# Type exports
from .fp8 import FP8

# Future exports will be added here as components are implemented
# from .tensor import Tensor
# from .shape import Shape
# from .dtype import DType
