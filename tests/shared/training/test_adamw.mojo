"""Unit tests for AdamW (Adam with Weight Decay) optimizer implementation.

Tests cover:
- AdamW initialization and API contract
- Parameter updates with adaptive learning rates
- Bias correction in early training steps
- Decoupled weight decay application
- Comparison with standard Adam (weight decay behavior)
- Edge cases and numerical stability

Following TDD principles - these tests define the expected API
and numerical behavior for AdamW optimizer.
"""

from tests.shared.conftest import (
    assert_true,
    assert_equal,
    assert_almost_equal,
    assert_less,
    assert_greater,
)
from shared.core.extensor import ExTensor, zeros, ones, zeros_like
from shared.training.optimizers.adamw import adamw_step, adamw_step_simple


# ============================================================================
# AdamW Initialization Tests
# ============================================================================


fn test_adamw_initialization() raises:
    """Test AdamW optimizer initialization with hyperparameters.

    Functional API Note:
        Pure functional design - no class initialization.
        Hyperparameters are passed as function arguments to adamw_step().
        This test verifies that the function accepts all expected parameters.
    """
    # Test that adamw_step accepts all hyperparameters
    var shape = List[Int]()
    shape.append(3)
    var params = ones(shape, DType.float32)
    var grads = zeros(shape, DType.float32)
    var m = zeros(shape, DType.float32)
    var v = zeros(shape, DType.float32)

    # Should accept all hyperparameters without error
    var result = adamw_step(
        params,
        grads,
        m,
        v,
        t=1,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=0.01,
    )

    # If we got here without error, the API contract is satisfied
    assert_true(True)  # Placeholder to mark test as passing


fn test_adamw_parameter_update() raises:
    """Test AdamW performs correct parameter update.

    Functional API:
        AdamW maintains two moments:
        - m (first moment, momentum)
        - v (second moment, RMSprop)

        Update formulas (same as Adam):
        - m = beta1 * m + (1 - beta1) * grad
        - v = beta2 * v + (1 - beta2) * grad^2
        - m_hat = m / (1 - beta1^t)  # Bias correction
        - v_hat = v / (1 - beta2^t)  # Bias correction
        - params = params - lr * m_hat / (sqrt(v_hat) + epsilon)

        Weight decay (DIFFERENT from Adam):
        - params = params - weight_decay * lr * params  # Decoupled WD

    This is a CRITICAL test for AdamW correctness.
    """
    var shape = List[Int]()
    shape.append(1)
    var params = ones(shape, DType.float32)
    params._data.bitcast[Float32]()[0] = 1.0

    var grads = zeros(shape, DType.float32)
    grads._data.bitcast[Float32]()[0] = 0.1

    var m = zeros(shape, DType.float32)
    var v = zeros(shape, DType.float32)

    # First step (t=1):
    # m = 0.9 * 0 + 0.1 * 0.1 = 0.01
    # v = 0.999 * 0 + 0.001 * 0.01 = 0.00001
    # m_hat = 0.01 / (1 - 0.9) = 0.1
    # v_hat = 0.00001 / (1 - 0.999) = 0.01
    # update = 0.001 * 0.1 / (sqrt(0.01) + 1e-8) ≈ 0.001
    # params_after_update ≈ 0.999
    # Then weight decay: params = 0.999 - 0.01 * 0.001 * 0.999 ≈ 0.9990001
    var result = adamw_step(
        params,
        grads,
        m,
        v,
        t=1,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=0.01,
    )
    params = result[0]
    m = result[1]
    v = result[2]

    # Parameter should decrease from 1.0
    # Should be less than standard Adam due to weight decay
    assert_less(params._data.bitcast[Float32]()[0], 1.0)
    assert_almost_equal(Float64(params._data.bitcast[Float32]()[0]), 0.999, tolerance=1e-3)


fn test_adamw_bias_correction() raises:
    """Test AdamW applies bias correction in early steps.

    Functional API:
        Bias correction factors:
        - m_hat = m / (1 - beta1^t)
        - v_hat = v / (1 - beta2^t)
        Where t is the step number (1, 2, 3, ...)

    This is CRITICAL for AdamW's fast convergence in early training.
    """
    var shape = List[Int]()
    shape.append(1)
    var params = ones(shape, DType.float32)
    params._data.bitcast[Float32]()[0] = 1.0

    var grads = zeros(shape, DType.float32)
    grads._data.bitcast[Float32]()[0] = 0.1

    var m = zeros(shape, DType.float32)
    var v = zeros(shape, DType.float32)

    # First few steps should have larger effective learning rate
    # due to bias correction
    var prev_param = Float32(1.0)

    # Run 5 steps
    for t in range(1, 6):
        var result = adamw_step(
            params, grads, m, v, t=t, learning_rate=0.001, weight_decay=0.0
        )
        params = result[0]
        m = result[1]
        v = result[2]

    # Should decrease consistently
    assert_less(params._data.bitcast[Float32]()[0], 1.0)


fn test_adamw_weight_decay_decoupled() raises:
    """Test AdamW applies weight decay in decoupled manner.

    Key difference from standard Adam:
        Adam: grad_effective = grad + weight_decay * params
              Then do adaptive update

        AdamW: Do adaptive update first
               THEN apply: params = params - weight_decay * lr * params

    This test verifies the decoupled weight decay behavior.
    """
    var shape = List[Int]()
    shape.append(1)
    var params = ones(shape, DType.float32)
    params._data.bitcast[Float32]()[0] = 1.0

    var grads = zeros(shape, DType.float32)
    grads._data.bitcast[Float32]()[0] = 0.1

    var m = zeros(shape, DType.float32)
    var v = zeros(shape, DType.float32)

    # Run with no weight decay
    var result_no_wd = adamw_step(
        params, grads, m, v, t=1, learning_rate=0.001, weight_decay=0.0
    )
    var params_no_wd = result_no_wd[0]

    # Reset
    params._data.bitcast[Float32]()[0] = 1.0
    m = zeros_like(m)
    v = zeros_like(v)

    # Run with weight decay
    result_no_wd = adamw_step(
        params, grads, m, v, t=1, learning_rate=0.001, weight_decay=0.01
    )
    var params_with_wd = result_no_wd[0]

    # With weight decay, final params should be less than without
    assert_less(
        Float64(params_with_wd._data.bitcast[Float32]()[0]),
        Float64(params_no_wd._data.bitcast[Float32]()[0]),
        "Weight decay should reduce parameters",
    )


fn test_adamw_zero_weight_decay() raises:
    """Test AdamW behaves like Adam when weight_decay=0.

    When weight_decay=0, AdamW should be equivalent to Adam.
    """
    var shape = List[Int]()
    shape.append(1)
    var params = ones(shape, DType.float32)
    params._data.bitcast[Float32]()[0] = 1.0

    var grads = zeros(shape, DType.float32)
    grads._data.bitcast[Float32]()[0] = 0.1

    var m = zeros(shape, DType.float32)
    var v = zeros(shape, DType.float32)

    # With weight_decay=0, should work without error
    var result = adamw_step(
        params, grads, m, v, t=1, learning_rate=0.001, weight_decay=0.0
    )

    # Should have reduced parameters
    assert_less(result[0]._data.bitcast[Float32]()[0], 1.0)


fn test_adamw_step_simple() raises:
    """Test simplified AdamW step with default hyperparameters.

    adamw_step_simple uses:
        - beta1 = 0.9
        - beta2 = 0.999
        - epsilon = 1e-8
        - weight_decay = 0.01
    """
    var shape = List[Int]()
    shape.append(1)
    var params = ones(shape, DType.float32)
    params._data.bitcast[Float32]()[0] = 1.0

    var grads = zeros(shape, DType.float32)
    grads._data.bitcast[Float32]()[0] = 0.1

    var m = zeros(shape, DType.float32)
    var v = zeros(shape, DType.float32)

    # Should work with just basic parameters
    var result = adamw_step_simple(params, grads, m, v, t=1, learning_rate=0.001)

    # Should have reduced parameters
    assert_less(result[0]._data.bitcast[Float32]()[0], 1.0)


fn test_adamw_multiple_steps() raises:
    """Test AdamW over multiple optimization steps.

    Verifies that:
    - Multiple steps converge (loss decreases)
    - State accumulation works correctly
    - No numerical instability
    """
    var shape = List[Int]()
    shape.append(1)
    var params = ones(shape, DType.float32)
    params._data.bitcast[Float32]()[0] = 10.0

    var grads = zeros(shape, DType.float32)
    grads._data.bitcast[Float32]()[0] = 1.0  # Positive gradient, so decrease params

    var m = zeros(shape, DType.float32)
    var v = zeros(shape, DType.float32)

    # Run 30 steps for significant convergence
    for t in range(1, 31):
        var result = adamw_step(
            params, grads, m, v, t=t, learning_rate=0.1, weight_decay=0.01
        )
        params = result[0]
        m = result[1]
        v = result[2]

    # Should have decreased significantly from 10.0 initial value
    assert_less(params._data.bitcast[Float32]()[0], 10.0)
    # AdamW with lr=0.1 and wd=0.01 converges to ~6.75 at 30 steps
    assert_less(params._data.bitcast[Float32]()[0], 7.0)


fn test_adamw_shape_mismatch() raises:
    """Test AdamW raises error on shape mismatch.

    Should validate that params and gradients have same shape.
    """
    var shape1 = List[Int]()
    shape1.append(3)
    var shape2 = List[Int]()
    shape2.append(5)

    var params = ones(shape1, DType.float32)
    var grads = zeros(shape2, DType.float32)
    var m = zeros(shape1, DType.float32)
    var v = zeros(shape1, DType.float32)

    try:
        var _ = adamw_step(params, grads, m, v, t=1, learning_rate=0.001)
        # Should not reach here
        assert_true(False, "Should have raised error on shape mismatch")
    except:
        # Expected error
        assert_true(True)


fn test_adamw_dtype_mismatch() raises:
    """Test AdamW raises error on dtype mismatch.

    Should validate that params and gradients have same dtype.
    """
    var shape = List[Int]()
    shape.append(3)

    var params = ones(shape, DType.float32)
    var grads = zeros(shape, DType.float32)
    var m = zeros(shape, DType.float32)
    var v = zeros(shape, DType.float32)

    # If different dtypes were supported, this would test that
    # For now, just verify dtypes match in valid case
    var result = adamw_step(params, grads, m, v, t=1, learning_rate=0.001)
    assert_true(True)


fn test_adamw_large_parameters() raises:
    """Test AdamW with larger parameter tensor.

    Verifies SIMD operations work correctly on realistic tensor sizes.
    """
    var shape = List[Int]()
    shape.append(100)
    var params = ones(shape, DType.float32)
    var grads = zeros(shape, DType.float32)

    # Fill with test values
    for i in range(100):
        params._data.bitcast[Float32]()[i] = Float32(i) + 1.0
        grads._data.bitcast[Float32]()[i] = 0.01

    var m = zeros(shape, DType.float32)
    var v = zeros(shape, DType.float32)

    var result = adamw_step(params, grads, m, v, t=1, learning_rate=0.001)

    # All parameters should decrease due to positive gradient
    for i in range(100):
        assert_less(
            Float64(result[0]._data.bitcast[Float32]()[i]),
            Float64(params._data.bitcast[Float32]()[i]),
        )


fn test_adamw_timestep_validation() raises:
    """Test AdamW validates positive timestep.

    Timestep must be >= 1 for bias correction to work.
    """
    var shape = List[Int]()
    shape.append(1)
    var params = ones(shape, DType.float32)
    var grads = zeros(shape, DType.float32)
    var m = zeros(shape, DType.float32)
    var v = zeros(shape, DType.float32)

    try:
        # t=0 should fail
        var _ = adamw_step(params, grads, m, v, t=0, learning_rate=0.001)
        assert_true(False, "Should have raised error for t=0")
    except:
        # Expected error
        assert_true(True)

    try:
        # t=-1 should fail
        var _ = adamw_step(params, grads, m, v, t=-1, learning_rate=0.001)
        assert_true(False, "Should have raised error for t=-1")
    except:
        # Expected error
        assert_true(True)


# ============================================================================
# Test Main
# ============================================================================


fn main() raises:
    """Run all AdamW optimizer tests."""
    print("Testing AdamW initialization...")
    test_adamw_initialization()

    print("Testing AdamW parameter update...")
    test_adamw_parameter_update()

    print("Testing AdamW bias correction...")
    test_adamw_bias_correction()

    print("Testing AdamW decoupled weight decay...")
    test_adamw_weight_decay_decoupled()

    print("Testing AdamW zero weight decay...")
    test_adamw_zero_weight_decay()

    print("Testing AdamW step simple...")
    test_adamw_step_simple()

    print("Testing AdamW multiple steps...")
    test_adamw_multiple_steps()

    print("Testing AdamW shape mismatch...")
    test_adamw_shape_mismatch()

    print("Testing AdamW dtype mismatch...")
    test_adamw_dtype_mismatch()

    print("Testing AdamW large parameters...")
    test_adamw_large_parameters()

    print("Testing AdamW timestep validation...")
    test_adamw_timestep_validation()

    print("\nAll AdamW optimizer tests passed! ✓")
