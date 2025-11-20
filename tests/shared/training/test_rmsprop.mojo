"""Tests for RMSprop optimizer.

Tests cover:
- Basic parameter updates
- Running average of squared gradients
- Momentum support
- Weight decay support
- Simplified version without momentum
- Numerical correctness

All tests use pure functional API.
"""

from tests.shared.conftest import (
    assert_true,
    assert_equal,
    assert_almost_equal,
    assert_shape_equal,
    TestFixtures,
)
from shared.core.extensor import ExTensor, zeros, ones
from shared.training.optimizers.rmsprop import rmsprop_step, rmsprop_step_simple
from collections.vector import DynamicVector


# ============================================================================
# RMSprop Basic Tests
# ============================================================================


fn test_rmsprop_step_shapes() raises:
    """Test that rmsprop_step returns correct shapes."""
    var shape = DynamicVector[Int](2)
    shape[0] = 4
    shape[1] = 10

    var params = ones(shape, DType.float32)
    var gradients = ones(shape, DType.float32)
    var square_avg = zeros(shape, DType.float32)
    var buf = zeros(shape, DType.float32)

    var (new_params, new_square_avg, new_buf) = rmsprop_step(
        params, gradients, square_avg,
        t=1,
        learning_rate=0.01,
        alpha=0.99,
        epsilon=1e-8,
        weight_decay=0.0,
        momentum=0.0,
        buf=buf
    )

    # Check shapes
    assert_equal(new_params.shape()[0], 4)
    assert_equal(new_params.shape()[1], 10)
    assert_equal(new_square_avg.shape()[0], 4)
    assert_equal(new_square_avg.shape()[1], 10)


fn test_rmsprop_simple_shapes() raises:
    """Test that rmsprop_step_simple returns correct shapes."""
    var shape = DynamicVector[Int](2)
    shape[0] = 4
    shape[1] = 10

    var params = ones(shape, DType.float32)
    var gradients = ones(shape, DType.float32)
    var square_avg = zeros(shape, DType.float32)

    var (new_params, new_square_avg) = rmsprop_step_simple(
        params, gradients, square_avg,
        t=1,
        learning_rate=0.01,
        alpha=0.99,
        epsilon=1e-8
    )

    # Check shapes
    assert_equal(new_params.shape()[0], 4)
    assert_equal(new_params.shape()[1], 10)
    assert_equal(new_square_avg.shape()[0], 4)
    assert_equal(new_square_avg.shape()[1], 10)


fn test_rmsprop_step_parameter_update() raises:
    """Test that rmsprop_step updates parameters correctly."""
    var shape = DynamicVector[Int](1)
    shape[0] = 1

    var params = ones(shape, DType.float32)
    params._data.bitcast[Float32]()[0] = 1.0

    var gradients = ones(shape, DType.float32)
    gradients._data.bitcast[Float32]()[0] = 0.1

    var square_avg = zeros(shape, DType.float32)
    var buf = zeros(shape, DType.float32)

    var (new_params, new_square_avg, _) = rmsprop_step(
        params, gradients, square_avg,
        t=1,
        learning_rate=0.1,
        alpha=0.9,
        epsilon=1e-8,
        weight_decay=0.0,
        momentum=0.0,
        buf=buf
    )

    # First step:
    # square_avg = 0.9 * 0.0 + 0.1 * (0.1)^2 = 0.001
    # normalized_grad = 0.1 / (sqrt(0.001) + 1e-8) ≈ 0.1 / 0.0316 ≈ 3.16
    # new_params = 1.0 - 0.1 * 3.16 = 1.0 - 0.316 = 0.684

    assert_true(new_params._data.bitcast[Float32]()[0] < 1.0)  # Parameter should decrease
    assert_almost_equal(
        new_params._data.bitcast[Float32]()[0],
        Float32(0.684),
        tolerance=0.01
    )

    # Check that square_avg was updated
    assert_almost_equal(
        new_square_avg._data.bitcast[Float32]()[0],
        Float32(0.001),
        tolerance=1e-5
    )


fn test_rmsprop_simple_parameter_update() raises:
    """Test that rmsprop_step_simple updates parameters correctly."""
    var shape = DynamicVector[Int](1)
    shape[0] = 1

    var params = ones(shape, DType.float32)
    params._data.bitcast[Float32]()[0] = 1.0

    var gradients = ones(shape, DType.float32)
    gradients._data.bitcast[Float32]()[0] = 0.1

    var square_avg = zeros(shape, DType.float32)

    var (new_params, new_square_avg) = rmsprop_step_simple(
        params, gradients, square_avg,
        t=1,
        learning_rate=0.1,
        alpha=0.9,
        epsilon=1e-8
    )

    # Should produce same result as rmsprop_step with momentum=0.0
    assert_true(new_params._data.bitcast[Float32]()[0] < 1.0)
    assert_almost_equal(
        new_params._data.bitcast[Float32]()[0],
        Float32(0.684),
        tolerance=0.01
    )


fn test_rmsprop_square_avg_accumulation() raises:
    """Test that square_avg accumulates correctly over multiple steps."""
    var shape = DynamicVector[Int](1)
    shape[0] = 1

    var params = ones(shape, DType.float32)
    var gradients = ones(shape, DType.float32)
    gradients._data.bitcast[Float32]()[0] = 0.1

    var square_avg = zeros(shape, DType.float32)
    var buf = zeros(shape, DType.float32)

    # Step 1
    var (params1, square_avg1, buf1) = rmsprop_step(
        params, gradients, square_avg,
        t=1,
        learning_rate=0.01,
        alpha=0.9,
        epsilon=1e-8,
        weight_decay=0.0,
        momentum=0.0,
        buf=buf
    )

    # square_avg after step 1: 0.9 * 0.0 + 0.1 * 0.01 = 0.001

    # Step 2
    var (params2, square_avg2, buf2) = rmsprop_step(
        params1, gradients, square_avg1,
        t=2,
        learning_rate=0.01,
        alpha=0.9,
        epsilon=1e-8,
        weight_decay=0.0,
        momentum=0.0,
        buf=buf1
    )

    # square_avg after step 2: 0.9 * 0.001 + 0.1 * 0.01 = 0.0009 + 0.001 = 0.0019

    assert_almost_equal(
        square_avg2._data.bitcast[Float32]()[0],
        Float32(0.0019),
        tolerance=1e-5
    )

    # Square avg should be increasing
    assert_true(square_avg2._data.bitcast[Float32]()[0] > square_avg1._data.bitcast[Float32]()[0])


fn test_rmsprop_with_momentum() raises:
    """Test rmsprop with momentum."""
    var shape = DynamicVector[Int](1)
    shape[0] = 1

    var params = ones(shape, DType.float32)
    params._data.bitcast[Float32]()[0] = 1.0

    var gradients = ones(shape, DType.float32)
    gradients._data.bitcast[Float32]()[0] = 0.1

    var square_avg = zeros(shape, DType.float32)
    var buf = zeros(shape, DType.float32)

    # Step 1 with momentum
    var (params1, square_avg1, buf1) = rmsprop_step(
        params, gradients, square_avg,
        t=1,
        learning_rate=0.1,
        alpha=0.9,
        epsilon=1e-8,
        weight_decay=0.0,
        momentum=0.9,
        buf=buf
    )

    # buf should now contain momentum-weighted gradient
    assert_true(buf1._data.bitcast[Float32]()[0] != 0.0)

    # Step 2 with momentum
    var (params2, square_avg2, buf2) = rmsprop_step(
        params1, gradients, square_avg1,
        t=2,
        learning_rate=0.1,
        alpha=0.9,
        epsilon=1e-8,
        weight_decay=0.0,
        momentum=0.9,
        buf=buf1
    )

    # With momentum, buf accumulates and parameter updates should be larger
    assert_true(buf2._data.bitcast[Float32]()[0] > buf1._data.bitcast[Float32]()[0])


fn test_rmsprop_with_weight_decay() raises:
    """Test rmsprop with weight decay."""
    var shape = DynamicVector[Int](1)
    shape[0] = 1

    var params = ones(shape, DType.float32)
    params._data.bitcast[Float32]()[0] = 1.0

    var gradients = zeros(shape, DType.float32)  # Zero gradient
    var square_avg = zeros(shape, DType.float32)
    var buf = zeros(shape, DType.float32)

    var (new_params, _, _) = rmsprop_step(
        params, gradients, square_avg,
        t=1,
        learning_rate=0.1,
        alpha=0.9,
        epsilon=1e-8,
        weight_decay=0.01,
        momentum=0.0,
        buf=buf
    )

    # With weight decay, parameters should decrease even with zero gradient
    # grad_with_decay = grad + weight_decay * params = 0.0 + 0.01 * 1.0 = 0.01
    assert_true(new_params._data.bitcast[Float32]()[0] < 1.0)


fn test_rmsprop_zero_gradient() raises:
    """Test that rmsprop handles zero gradients correctly."""
    var shape = DynamicVector[Int](1)
    shape[0] = 1

    var params = ones(shape, DType.float32)
    var gradients = zeros(shape, DType.float32)
    var square_avg = zeros(shape, DType.float32)
    var buf = zeros(shape, DType.float32)

    var (new_params, new_square_avg, _) = rmsprop_step(
        params, gradients, square_avg,
        t=1,
        learning_rate=0.1,
        alpha=0.9,
        epsilon=1e-8,
        weight_decay=0.0,
        momentum=0.0,
        buf=buf
    )

    # With zero gradient and no weight decay, parameters should not change
    assert_almost_equal(
        new_params._data.bitcast[Float32]()[0],
        Float32(1.0),
        tolerance=1e-5
    )


fn test_rmsprop_alpha_parameter() raises:
    """Test that alpha parameter controls averaging."""
    var shape = DynamicVector[Int](1)
    shape[0] = 1

    var params = ones(shape, DType.float32)
    var gradients = ones(shape, DType.float32)
    gradients._data.bitcast[Float32]()[0] = 0.1

    var square_avg = zeros(shape, DType.float32)
    var buf = zeros(shape, DType.float32)

    # High alpha (0.99) - slow adaptation
    var (_, square_avg_high, _) = rmsprop_step(
        params, gradients, square_avg,
        t=1,
        learning_rate=0.01,
        alpha=0.99,
        epsilon=1e-8,
        weight_decay=0.0,
        momentum=0.0,
        buf=buf
    )

    # Low alpha (0.5) - fast adaptation
    var (_, square_avg_low, _) = rmsprop_step(
        params, gradients, square_avg,
        t=1,
        learning_rate=0.01,
        alpha=0.5,
        epsilon=1e-8,
        weight_decay=0.0,
        momentum=0.0,
        buf=buf
    )

    # Low alpha should result in larger square_avg update
    # alpha=0.99: 0.99 * 0.0 + 0.01 * 0.01 = 0.0001
    # alpha=0.5: 0.5 * 0.0 + 0.5 * 0.01 = 0.005
    assert_true(square_avg_low._data.bitcast[Float32]()[0] > square_avg_high._data.bitcast[Float32]()[0])


fn test_rmsprop_epsilon_prevents_division_by_zero() raises:
    """Test that epsilon prevents division by zero."""
    var shape = DynamicVector[Int](1)
    shape[0] = 1

    var params = ones(shape, DType.float32)
    var gradients = ones(shape, DType.float32)
    var square_avg = zeros(shape, DType.float32)  # Zero square_avg
    var buf = zeros(shape, DType.float32)

    # This should not crash despite zero square_avg
    var (new_params, _, _) = rmsprop_step(
        params, gradients, square_avg,
        t=1,
        learning_rate=0.1,
        alpha=0.9,
        epsilon=1e-8,
        weight_decay=0.0,
        momentum=0.0,
        buf=buf
    )

    # Result should be finite
    var val = new_params._data.bitcast[Float32]()[0]
    assert_true(val == val)  # Not NaN
    assert_true(val > -1e10 and val < 1e10)  # Not infinite


fn test_rmsprop_batch_update() raises:
    """Test rmsprop with batch of parameters."""
    var shape = DynamicVector[Int](2)
    shape[0] = 10
    shape[1] = 5

    var params = ones(shape, DType.float32)
    var gradients = ones(shape, DType.float32)

    # Set different gradient values
    for i in range(50):
        gradients._data.bitcast[Float32]()[i] = Float32(i) * 0.01

    var square_avg = zeros(shape, DType.float32)
    var buf = zeros(shape, DType.float32)

    var (new_params, new_square_avg, _) = rmsprop_step(
        params, gradients, square_avg,
        t=1,
        learning_rate=0.01,
        alpha=0.9,
        epsilon=1e-8,
        weight_decay=0.0,
        momentum=0.0,
        buf=buf
    )

    # All parameters should have been updated
    var all_different = True
    for i in range(50):
        if new_params._data.bitcast[Float32]()[i] == params._data.bitcast[Float32]()[i]:
            all_different = False
            break

    assert_true(all_different)


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run all RMSprop optimizer tests."""
    print("Running RMSprop optimizer tests...")

    test_rmsprop_step_shapes()
    print("✓ test_rmsprop_step_shapes")

    test_rmsprop_simple_shapes()
    print("✓ test_rmsprop_simple_shapes")

    test_rmsprop_step_parameter_update()
    print("✓ test_rmsprop_step_parameter_update")

    test_rmsprop_simple_parameter_update()
    print("✓ test_rmsprop_simple_parameter_update")

    test_rmsprop_square_avg_accumulation()
    print("✓ test_rmsprop_square_avg_accumulation")

    test_rmsprop_with_momentum()
    print("✓ test_rmsprop_with_momentum")

    test_rmsprop_with_weight_decay()
    print("✓ test_rmsprop_with_weight_decay")

    test_rmsprop_zero_gradient()
    print("✓ test_rmsprop_zero_gradient")

    test_rmsprop_alpha_parameter()
    print("✓ test_rmsprop_alpha_parameter")

    test_rmsprop_epsilon_prevents_division_by_zero()
    print("✓ test_rmsprop_epsilon_prevents_division_by_zero")

    test_rmsprop_batch_update()
    print("✓ test_rmsprop_batch_update")

    print("\nAll RMSprop optimizer tests passed!")
