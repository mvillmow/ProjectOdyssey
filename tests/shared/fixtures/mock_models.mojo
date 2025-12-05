"""Mock model architectures for testing (legacy/deprecated location).

NOTE: This module is deprecated. Test models have been consolidated into
shared.testing.test_models for better organization and centralization.

For new code, import test models directly from shared.testing:
    from shared.testing import SimpleMLP, MockLayer, SimpleLinearModel, etc.

This module is maintained for backward compatibility and re-exports
the consolidated models. Existing code using imports from this module
will continue to work, but new code should use shared.testing.

Key components (now in shared.testing.test_models):
- MockLayer: Minimal layer implementation
- SimpleLinearModel: Single linear layer
- SimpleMLP: Multi-layer perceptron (2-3 layers)
- Parameter: Trainable parameter wrapper

All models use simple operations for predictable testing.
"""

# Re-export consolidated test models for backward compatibility
from shared.testing import (
    SimpleMLP,
    MockLayer,
    SimpleLinearModel,
    Parameter,
)

# Legacy imports kept for any downstream code that uses this module
from tests.shared.fixtures.mock_tensors import (
    create_random_tensor,
    create_zeros_tensor,
    create_ones_tensor,
)
from shared.core import zeros_like, zeros, ExTensor
from shared.core.traits import Model

# All struct definitions have been moved to shared.testing.test_models
# This module now serves as a compatibility shim only
