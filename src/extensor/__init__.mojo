"""ExTensor package - Extensible Tensors for ML Odyssey.

This package provides a comprehensive tensor implementation following the
Python Array API Standard (https://data-apis.org/array-api/latest/):

Architecture:
- Dynamic shapes and arbitrary dimensions (0D scalars to N-D tensors)
- Multiple data types (float16/32/64, int8/16/32/64, uint8/16/32/64, bool)
- NumPy-style broadcasting semantics
- 150+ operations from Array API Standard 2023.12
- SIMD-optimized element-wise operations
- Memory-safe via Mojo's ownership system

Array API Standard Compliance:
- Creation functions: zeros, ones, full, empty, arange, eye, linspace
- Arithmetic operations: add, subtract, multiply, divide, floor_divide, modulo, power
- Comparison operations: equal, not_equal, less, less_equal, greater, greater_equal
- Reduction operations: sum, mean, max, min
- Matrix operations: matmul, transpose (in progress)
- Broadcasting: Follows Array API broadcasting rules

Reference: https://data-apis.org/array-api/latest/API_specification/index.html

Examples:
    from extensor import ExTensor, zeros, ones, add

    # Create tensors
    var shape = DynamicVector[Int](3, 4)
    var a = zeros(shape, DType.float32)
    var b = ones(shape, DType.float32)

    # Arithmetic with operator overloading
    var c = a + b  # Element-wise addition

    # Functional interface
    var d = add(a, b)  # Same as a + b

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

# Element-wise mathematical operations
from .elementwise_math import abs, sign, exp, log, sqrt, sin, cos, tanh, clip

# Shape manipulation operations
from .shape import reshape, squeeze, unsqueeze, expand_dims, flatten, ravel, concatenate, stack

# TODO: Export remaining operation categories
# from .indexing import getitem, setitem, take, gather
# etc.
