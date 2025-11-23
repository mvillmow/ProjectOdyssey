"""
Utilities Module.

This module contains utility functions for initialization, memory management,
debugging, and performance profiling.

Components:
    - xavier_init: Xavier/Glorot weight initialization
    - he_init: He/Kaiming weight initialization
    - zeros_init: Zero initialization
    - ones_init: Ones initialization
    - memory: Memory management helpers
    - debug: Debugging utilities (shape checking, gradient verification)
    - profile: Performance profiling tools

Example:.    from shared.core.utils import xavier_init, he_init
    from shared.core.types import Tensor

    fn initialize_weights(mut tensor: Tensor, activation: String):
        # Choose initialization based on activation function
        if activation == "relu":
            he_init(tensor)
        else:
            xavier_init(tensor)
"""

# Utility exports will be added here as components are implemented
# from .init import xavier_init, he_init, zeros_init, ones_init
# from .memory import allocate_aligned, free_aligned
# from .debug import check_shape, verify_gradients
