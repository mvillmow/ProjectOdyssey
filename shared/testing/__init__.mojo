"""Testing utilities for ML Odyssey.

Provides tools for validating neural network implementations:
- Assertion functions for test validation
- Data generators for synthetic test datasets
- Gradient checking (numerical vs analytical)
- Test fixtures and helpers

Modules:
    assertions: Comprehensive assertion functions for testing
    data_generators: Generate synthetic test data (random tensors, classification datasets)
    gradient_checker: Validate backward passes using finite differences
    fixtures: Test models, data generators, and assertion helpers
"""

from .assertions import (
    assert_true,
    assert_false,
    assert_equal,
    assert_not_equal,
    assert_not_none,
    assert_almost_equal,
    assert_dtype_equal,
    assert_equal_int,
    assert_equal_float,
    assert_close_float,
    assert_greater,
    assert_less,
    assert_greater_or_equal,
    assert_less_or_equal,
    assert_shape_equal,
    assert_not_equal_tensor,
    assert_tensor_equal,
    assert_shape,
    assert_dtype,
    assert_numel,
    assert_dim,
    assert_value_at,
    assert_all_values,
    assert_all_close,
    assert_type,
)

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

from .fixtures import (
    SimpleCNN,
    LinearModel,
    create_test_cnn,
    create_linear_model,
    create_test_input,
    create_test_targets,
    assert_tensor_shape,
    assert_tensor_dtype,
    assert_tensor_all_finite,
    assert_tensor_not_all_zeros,
)
