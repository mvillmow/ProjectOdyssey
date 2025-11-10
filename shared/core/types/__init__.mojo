"""
Types and Data Structures Module.

This module contains custom types and data structures optimized for ML workloads.
All types are implemented using Mojo's struct system for memory safety and performance.

Components:
    - Tensor: Multi-dimensional array type with automatic memory management
    - Shape: Shape representation and utilities
    - DType: Data type definitions and conversions
    - Slice: Tensor slicing utilities

Example:
    from shared.core.types import Tensor, Shape, DType

    # Create a tensor with specified shape and dtype
    fn create_weights() -> Tensor:
        let shape = Shape(128, 784)
        return Tensor(shape, DType.float32)
"""

# Type exports will be added here as components are implemented
# from .tensor import Tensor
# from .shape import Shape
# from .dtype import DType
