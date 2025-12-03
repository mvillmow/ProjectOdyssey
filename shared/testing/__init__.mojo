"""Testing utilities for ML Odyssey.

Provides tools for validating neural network implementations:
- Data generators for synthetic test datasets
- Gradient checking (numerical vs analytical)
- Test fixtures and helpers
- Assertion utilities

Modules:
    data_generators: Generate synthetic test data (random tensors, classification datasets)
    gradient_checker: Validate backward passes using finite differences
"""

from .gradient_checker import (
    check_gradients,
    check_gradients_verbose,
    relative_error
)

from .data_generators import (
    random_tensor,
    random_uniform,
    random_normal,
    synthetic_classification_data
)
