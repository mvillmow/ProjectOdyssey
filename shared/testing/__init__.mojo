"""Testing utilities for ML Odyssey.

Provides tools for validating neural network implementations:
- Gradient checking (numerical vs analytical)
- Test fixtures and helpers
- Assertion utilities

Modules:
    gradient_checker: Validate backward passes using finite differences
"""

from .gradient_checker import (
    check_gradients,
    check_gradients_verbose,
    relative_error
)
