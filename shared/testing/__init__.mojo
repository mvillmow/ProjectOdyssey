"""Testing utilities for ML Odyssey.

Provides tools for validating neural network implementations:
- Assertion functions for test validation
- Data generators for synthetic test datasets
- Gradient checking (numerical vs analytical)
- Consolidated test models and fixtures

Modules:
    assertions: Comprehensive assertion functions for testing
    data_generators: Generate synthetic test data (random tensors, classification datasets)
    gradient_checker: Validate backward passes using finite differences
    test_models: Consolidated test model implementations (SimpleCNN, LinearModel, SimpleMLP, etc.)
    fixtures: Test model factories, tensor utilities, and assertion helpers

Test Models:
    SimpleCNN: Minimal CNN for image processing tests
    LinearModel: Single fully-connected layer for basic tests
    SimpleMLP: Multi-layer perceptron (2-3 layers) for composition tests
    SimpleLinearModel: Linear model with explicit weights and bias
    MockLayer: Minimal mock layer for testing
    Parameter: Trainable parameter with gradient tracking
"""

from .assertions import (
    TOLERANCE_DEFAULT,
    TOLERANCE_FLOAT32,
    TOLERANCE_FLOAT64,
    TOLERANCE_GRADIENT_RTOL,
    TOLERANCE_GRADIENT_ATOL,
    TOLERANCE_CONV,
    TOLERANCE_SOFTMAX,
    TOLERANCE_CROSS_ENTROPY,
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
    relative_error,
    check_gradient,
    compute_numerical_gradient,
    assert_gradients_close
)

from .data_generators import (
    random_tensor,
    random_uniform,
    random_normal,
    synthetic_classification_data
)

from .test_models import (
    SimpleCNN,
    LinearModel,
    SimpleMLP,
    MockLayer,
    SimpleLinearModel,
    Parameter,
)

from .fixtures import (
    create_test_cnn,
    create_linear_model,
    create_test_input,
    create_test_targets,
    assert_tensor_shape,
    assert_tensor_dtype,
    assert_tensor_all_finite,
    assert_tensor_not_all_zeros,
)
