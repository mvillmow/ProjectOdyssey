"""Shared test fixtures for ML Odyssey test suite.

This module provides comprehensive test utilities including:
- Mock tensor creation and comparison utilities
- Mock datasets and data loaders
- Mock model architectures for testing
- File I/O helpers for temporary files
- Configuration fixtures for validation testing

All fixtures follow these principles:
- Simple and concrete implementations (no complex mocking)
- Deterministic behavior with fixed seeds
- Clear, documented APIs
- Minimal dependencies

Usage:
    from tests.shared.fixtures.mock_tensors import create_random_tensor
    from tests.shared.fixtures.mock_data import MockDataset
    from tests.shared.fixtures.mock_models import SimpleMLP

See individual modules for detailed documentation and examples.
"""

# Note: Mojo doesn't require explicit imports in __init__.mojo
# The module structure provides automatic namespace organization
# Users can import directly from submodules as shown above
