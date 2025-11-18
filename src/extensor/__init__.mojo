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

# Creation utilities (shape-matching helpers)
from .extensor import ones_like, zeros_like, full_like

# Broadcasting utilities
from .broadcasting import broadcast_shapes, are_shapes_broadcastable

# Arithmetic operations (forward pass)
from .arithmetic import add, subtract, multiply, divide, floor_divide, modulo, power

# Arithmetic gradients (backward pass)
from .arithmetic import (
    add_backward, subtract_backward, multiply_backward, divide_backward
)

# Comparison operations
from .comparison import equal, not_equal, less, less_equal, greater, greater_equal

# Matrix operations (forward pass)
from .matrix import matmul, transpose, dot, outer

# Matrix gradients (backward pass)
from .matrix import matmul_backward, transpose_backward

# Reduction operations (forward pass)
from .reduction import sum, mean, max_reduce, min_reduce

# Reduction gradients (backward pass)
from .reduction import sum_backward, mean_backward, max_reduce_backward, min_reduce_backward

# Element-wise mathematical operations (forward pass)
from .elementwise_math import (
    abs, sign, exp, log, sqrt, sin, cos, tanh, clip,
    ceil, floor, round, trunc,
    logical_and, logical_or, logical_not, logical_xor,
    log10, log2
)

# Element-wise math gradients (backward pass)
from .elementwise_math import (
    exp_backward, log_backward, sqrt_backward, abs_backward,
    clip_backward, log10_backward, log2_backward
)

# Shape manipulation operations
from .shape import reshape, squeeze, unsqueeze, expand_dims, flatten, ravel, concatenate, stack

# Activation functions (forward pass)
from .activations import (
    relu, leaky_relu, prelu,
    sigmoid, tanh,
    softmax, gelu
)

# Activation gradients (backward pass)
from .activations import (
    relu_backward, leaky_relu_backward, prelu_backward,
    sigmoid_backward, tanh_backward,
    softmax_backward, gelu_backward
)

# Weight initializers
from .initializers import xavier_uniform, xavier_normal

# Loss functions (forward pass)
from .losses import (
    binary_cross_entropy, mean_squared_error, cross_entropy
)

# Loss gradients (backward pass)
from .losses import (
    binary_cross_entropy_backward, mean_squared_error_backward, cross_entropy_backward
)

# TODO: Export remaining operation categories
# from .indexing import getitem, setitem, take, gather
# etc.
