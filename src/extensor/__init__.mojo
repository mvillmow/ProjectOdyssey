"""ExTensor package - Extensible Tensors for ML Odyssey.

This package provides a comprehensive tensor implementation with:
- Dynamic shapes and arbitrary dimensions (0D to N-D)
- Multiple data types (float16/32/64, int8/16/32/64, uint8/16/32/64, bool)
- NumPy-style broadcasting
- 150+ operations from Array API Standard 2024
- SIMD-optimized element-wise operations
- Memory-safe via Mojo's ownership system

Examples:
    from extensor import ExTensor, zeros, ones

    # Create tensors
    var shape = DynamicVector[Int](3, 4)
    var a = zeros(shape, DType.float32)
    var b = ones(shape, DType.float32)

    # Arithmetic with broadcasting (TODO: implement)
    # var c = a + b

    # Matrix multiplication (TODO: implement)
    # var x = zeros(DynamicVector[Int](3, 4), DType.float32)
    # var y = zeros(DynamicVector[Int](4, 5), DType.float32)
    # var z = x @ y
"""

# Core tensor type
from .extensor import ExTensor

# Creation operations
from .extensor import zeros, ones, full, empty, arange, eye, linspace

# Broadcasting utilities
from .broadcasting import broadcast_shapes, are_shapes_broadcastable

# Arithmetic operations
from .arithmetic import add, subtract, multiply, divide, floor_divide, modulo, power

# Comparison operations
from .comparison import equal, not_equal, less, less_equal, greater, greater_equal

# Matrix operations
from .matrix import matmul, transpose, dot, outer

# Reduction operations
from .reduction import sum, mean, max_reduce, min_reduce

# TODO: Export remaining operation categories
# from .pointwise_math import sin, cos, exp, log, sqrt
# from .shape import reshape, squeeze, unsqueeze, concatenate
# from .indexing import getitem, setitem, take, gather
# etc.
