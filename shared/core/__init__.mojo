"""
Core Library - Fundamental building blocks for ML Odyssey.

This package contains reusable neural network components, mathematical operations,
custom types, and utilities used across all paper implementations.

Modules:
    layers: Neural network layer implementations (Linear, Conv2D, ReLU, etc.)
    ops: Low-level mathematical operations (matmul, elementwise, reduction)
    types: Custom types and data structures (Tensor, Shape, DType)
    utils: Utility functions (initialization, memory management, debugging)

Example:
    from shared.core.layers import Linear, ReLU
    from shared.core.ops import matmul
    from shared.core.types import Tensor

    # Build a simple model using core components
    struct SimpleModel:
        var fc1: Linear
        var fc2: Linear

        fn __init__(inout self):
            self.fc1 = Linear(784, 128)
            self.fc2 = Linear(128, 10)
"""

# Package version
alias VERSION = "0.1.0"

# ============================================================================
# Exports - Will be populated during implementation phase
# ============================================================================
# NOTE: These imports are commented out until implementation phase completes.

# Core exports will be added here as components are implemented
# from .layers import Linear, Conv2D, ReLU, Sigmoid, Tanh
# from .ops import matmul, transpose, sum, mean
# from .types import Tensor, Shape, DType
# from .utils import xavier_init, he_init

# ============================================================================
# Public API
# ============================================================================

__all__ = [
    # Layers
    # "Linear",
    # "Conv2D",
    # "ReLU",
    # "Sigmoid",
    # "Tanh",
    # Operations
    # "matmul",
    # "transpose",
    # "sum",
    # "mean",
    # Types
    # "Tensor",
    # "Shape",
    # "DType",
    # Utils
    # "xavier_init",
    # "he_init",
]
