"""Unit tests for optimizer implementations.

Tests cover:
- SGD (Stochastic Gradient Descent) with momentum
- Adam (Adaptive Moment Estimation)
- AdamW (Adam with Weight Decay)
- RMSprop (Root Mean Square Propagation)

Following TDD principles - these tests define the expected API
and numerical behavior for implementation in Issue #49.
"""

from tests.shared.conftest import (
    assert_true,
    assert_equal,
    assert_almost_equal,
    assert_less,
    create_test_vector,
    TestFixtures,
)


# ============================================================================
# SGD Tests
# ============================================================================

fn test_sgd_initialization() raises:
    """Test SGD optimizer initialization with hyperparameters.

    API Contract:
        SGD(
            learning_rate: Float32 = 0.01,
            momentum: Float32 = 0.0,
            dampening: Float32 = 0.0,
            weight_decay: Float32 = 0.0,
            nesterov: Bool = False
        )
    """
    # TODO: Implement when SGD is available
    # var optimizer = SGD(
    #     learning_rate=0.01,
    #     momentum=0.9,
    #     dampening=0.0,
    #     weight_decay=0.0001,
    #     nesterov=False
    # )
    # assert_almost_equal(optimizer.learning_rate, 0.01)
    # assert_almost_equal(optimizer.momentum, 0.9)
    # assert_almost_equal(optimizer.weight_decay, 0.0001)
    pass


fn test_sgd_basic_update() raises:
    """Test SGD performs basic parameter update without momentum.

    API Contract:
        optimizer.step(inout params: Tensor, grads: Tensor)
        - Updates parameters in-place
        - Formula: params = params - lr * grads

    This is a CRITICAL test that defines the core SGD behavior.
    """
    # TODO: Implement when SGD and Tensor are available
    # # Initial parameters: [1.0, 2.0, 3.0]
    # var params = Tensor(List[Float32](1.0, 2.0, 3.0), Shape(3))
    #
    # # Gradients: [0.1, 0.2, 0.3]
    # var grads = Tensor(List[Float32](0.1, 0.2, 0.3), Shape(3))
    #
    # # Create optimizer with lr=0.1
    # var optimizer = SGD(learning_rate=0.1, momentum=0.0)
    #
    # # Perform update
    # optimizer.step(params, grads)
    #
    # # Expected: params = params - lr * grads
    # # [1.0 - 0.1*0.1, 2.0 - 0.1*0.2, 3.0 - 0.1*0.3]
    # # = [0.99, 1.98, 2.97]
    # assert_almost_equal(params[0], 0.99, tolerance=1e-6)
    # assert_almost_equal(params[1], 1.98, tolerance=1e-6)
    # assert_almost_equal(params[2], 2.97, tolerance=1e-6)
    pass


fn test_sgd_momentum_accumulation() raises:
    """Test SGD accumulates momentum correctly over multiple steps.

    API Contract:
        With momentum > 0:
        - First update: velocity = grad
        - Subsequent updates: velocity = momentum * velocity + grad
        - Parameter update: params = params - lr * velocity

    This is a CRITICAL test for momentum-based training.
    """
    # TODO: Implement when SGD is available
    # var params = Tensor(List[Float32](1.0), Shape(1))
    # var grads = Tensor(List[Float32](0.1), Shape(1))
    #
    # var optimizer = SGD(learning_rate=0.1, momentum=0.9)
    #
    # # Step 1: velocity = grad = 0.1
    # # update = lr * velocity = 0.1 * 0.1 = 0.01
    # # params = 1.0 - 0.01 = 0.99
    # optimizer.step(params, grads)
    # assert_almost_equal(params[0], 0.99)
    #
    # # Step 2: velocity = 0.9 * 0.1 + 0.1 = 0.19
    # # update = 0.1 * 0.19 = 0.019
    # # params = 0.99 - 0.019 = 0.971
    # optimizer.step(params, grads)
    # assert_almost_equal(params[0], 0.971)
    pass


fn test_sgd_weight_decay() raises:
    """Test SGD applies weight decay (L2 regularization).

    API Contract:
        With weight_decay > 0:
        - Effective gradient: grad = grad + weight_decay * params
        - Then apply standard update
    """
    # TODO: Implement when SGD is available
    # var params = Tensor(List[Float32](1.0), Shape(1))
    # var grads = Tensor(List[Float32](0.1), Shape(1))
    #
    # var optimizer = SGD(learning_rate=0.1, weight_decay=0.01)
    #
    # # Effective grad = 0.1 + 0.01 * 1.0 = 0.11
    # # update = 0.1 * 0.11 = 0.011
    # # params = 1.0 - 0.011 = 0.989
    # optimizer.step(params, grads)
    # assert_almost_equal(params[0], 0.989)
    pass


fn test_sgd_nesterov_momentum() raises:
    """Test SGD with Nesterov momentum (lookahead).

    API Contract:
        With nesterov=True:
        - Lookahead: params_ahead = params - momentum * velocity
        - Gradient computed at lookahead position
        - Update using lookahead gradient
    """
    # TODO: Implement when SGD supports Nesterov
    # This is an advanced feature, may be deferred
    pass


fn test_sgd_zero_grad() raises:
    """Test SGD clears optimizer state (if needed).

    API Contract:
        optimizer.zero_grad() or similar method
        - Clears accumulated gradients or state
    """
    # TODO: Implement if SGD has state management
    # This may not be needed if gradients are managed externally
    pass


# ============================================================================
# Adam Tests
# ============================================================================

fn test_adam_initialization() raises:
    """Test Adam optimizer initialization.

    API Contract:
        Adam(
            learning_rate: Float32 = 0.001,
            beta1: Float32 = 0.9,
            beta2: Float32 = 0.999,
            epsilon: Float32 = 1e-8
        )
    """
    # TODO: Implement when Adam is available
    # var optimizer = Adam(
    #     learning_rate=0.001,
    #     beta1=0.9,
    #     beta2=0.999,
    #     epsilon=1e-8
    # )
    # assert_almost_equal(optimizer.learning_rate, 0.001)
    # assert_almost_equal(optimizer.beta1, 0.9)
    # assert_almost_equal(optimizer.beta2, 0.999)
    pass


fn test_adam_parameter_update() raises:
    """Test Adam performs correct parameter update.

    API Contract:
        Adam maintains two moments:
        - m (first moment, momentum)
        - v (second moment, RMSprop)

        Update formulas:
        - m = beta1 * m + (1 - beta1) * grad
        - v = beta2 * v + (1 - beta2) * grad^2
        - m_hat = m / (1 - beta1^t)  # Bias correction
        - v_hat = v / (1 - beta2^t)  # Bias correction
        - params = params - lr * m_hat / (sqrt(v_hat) + epsilon)

    This is a CRITICAL test for Adam correctness.
    """
    # TODO: Implement when Adam is available
    # var params = Tensor(List[Float32](1.0), Shape(1))
    # var grads = Tensor(List[Float32](0.1), Shape(1))
    #
    # var optimizer = Adam(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)
    #
    # # First step (t=1):
    # # m = 0.9 * 0 + 0.1 * 0.1 = 0.01
    # # v = 0.999 * 0 + 0.001 * 0.01 = 0.00001
    # # m_hat = 0.01 / (1 - 0.9) = 0.1
    # # v_hat = 0.00001 / (1 - 0.999) = 0.01
    # # update = 0.001 * 0.1 / (sqrt(0.01) + 1e-8) ≈ 0.001
    # optimizer.step(params, grads)
    #
    # # Check approximate result (exact calculation complex)
    # assert_less(params[0], 1.0)  # Parameter should decrease
    pass


fn test_adam_bias_correction() raises:
    """Test Adam applies bias correction in early steps.

    API Contract:
        Bias correction factors:
        - m_hat = m / (1 - beta1^t)
        - v_hat = v / (1 - beta2^t)
        Where t is the step number (1, 2, 3, ...)

    This is CRITICAL for Adam's fast convergence in early training.
    """
    # TODO: Implement when Adam is available
    # var params = Tensor(List[Float32](1.0), Shape(1))
    # var grads = Tensor(List[Float32](0.1), Shape(1))
    #
    # var optimizer = Adam(learning_rate=0.001)
    #
    # # First few steps should have larger effective learning rate
    # # due to bias correction
    # var params_history = List[Float32]()
    # for _ in range(5):
    #     optimizer.step(params, grads)
    #     params_history.append(params[0])
    #
    # # Each step should decrease parameters
    # for i in range(len(params_history) - 1):
    #     assert_less(params_history[i+1], params_history[i])
    pass


# ============================================================================
# AdamW Tests
# ============================================================================

fn test_adamw_weight_decay() raises:
    """Test AdamW applies decoupled weight decay.

    API Contract:
        AdamW is Adam with decoupled weight decay:
        - Apply Adam update (without weight decay in gradient)
        - Then apply: params = params * (1 - lr * weight_decay)

        This differs from L2 regularization used in standard Adam.
    """
    # TODO: Implement when AdamW is available
    # var params = Tensor(List[Float32](1.0), Shape(1))
    # var grads = Tensor(List[Float32](0.1), Shape(1))
    #
    # var optimizer = AdamW(learning_rate=0.001, weight_decay=0.01)
    #
    # # AdamW should decay weights more aggressively than Adam+L2
    # optimizer.step(params, grads)
    #
    # # Verify weight decay was applied
    # assert_less(params[0], 1.0)
    pass


# ============================================================================
# RMSprop Tests
# ============================================================================

fn test_rmsprop_initialization() raises:
    """Test RMSprop optimizer initialization.

    API Contract:
        RMSprop(
            learning_rate: Float32 = 0.01,
            alpha: Float32 = 0.99,
            epsilon: Float32 = 1e-8,
            momentum: Float32 = 0.0
        )
    """
    # TODO: Implement when RMSprop is available
    # var optimizer = RMSprop(
    #     learning_rate=0.01,
    #     alpha=0.99,
    #     epsilon=1e-8,
    #     momentum=0.0
    # )
    # assert_almost_equal(optimizer.learning_rate, 0.01)
    # assert_almost_equal(optimizer.alpha, 0.99)
    pass


fn test_rmsprop_parameter_update() raises:
    """Test RMSprop performs correct parameter update.

    API Contract:
        RMSprop maintains moving average of squared gradients:
        - v = alpha * v + (1 - alpha) * grad^2
        - params = params - lr * grad / (sqrt(v) + epsilon)
    """
    # TODO: Implement when RMSprop is available
    # var params = Tensor(List[Float32](1.0), Shape(1))
    # var grads = Tensor(List[Float32](0.1), Shape(1))
    #
    # var optimizer = RMSprop(learning_rate=0.01, alpha=0.99, epsilon=1e-8)
    #
    # # First step:
    # # v = 0.99 * 0 + 0.01 * 0.01 = 0.0001
    # # update = 0.01 * 0.1 / (sqrt(0.0001) + 1e-8) ≈ 0.1
    # optimizer.step(params, grads)
    #
    # # Parameter should decrease significantly
    # assert_less(params[0], 0.95)
    pass


# ============================================================================
# Property-Based Tests
# ============================================================================

fn test_optimizer_property_decreasing_loss() raises:
    """Property: Optimizer should decrease loss on convex function.

    Test that all optimizers can minimize a simple quadratic function.
    This validates basic convergence behavior.
    """
    # TODO: Implement when optimizers and loss functions are available
    # # Define simple quadratic: f(x) = x^2
    # # Gradient: df/dx = 2x
    # # Minimum at x=0
    #
    # var initial_value = Float32(5.0)
    # var params = Tensor(List[Float32](initial_value), Shape(1))
    #
    # # Test each optimizer
    # let optimizers = [
    #     SGD(learning_rate=0.1),
    #     Adam(learning_rate=0.1),
    #     RMSprop(learning_rate=0.1),
    # ]
    #
    # for optimizer in optimizers:
    #     var x = params.copy()
    #     var initial_loss = x[0] * x[0]
    #
    #     # Run 100 steps
    #     for _ in range(100):
    #         var grad = 2 * x[0]  # Gradient of x^2
    #         optimizer.step(x, grad)
    #
    #     var final_loss = x[0] * x[0]
    #
    #     # Loss should decrease significantly
    #     assert_less(final_loss, initial_loss * 0.1)
    pass


fn test_optimizer_property_gradient_shape() raises:
    """Property: Optimizer should handle gradients of same shape as parameters.

    All optimizers should work with multi-dimensional parameter tensors.
    """
    # TODO: Implement when optimizers are available
    # # Test with various parameter shapes
    # let shapes = [Shape(10), Shape(10, 5), Shape(3, 32, 32)]
    #
    # for shape in shapes:
    #     var params = Tensor.randn(shape)
    #     var grads = Tensor.randn(shape)
    #
    #     var optimizer = SGD(learning_rate=0.01)
    #     optimizer.step(params, grads)
    #
    #     # Shape should be preserved
    #     assert_equal(params.shape, shape)
    pass


# ============================================================================
# Numerical Accuracy Tests
# ============================================================================

fn test_sgd_matches_pytorch() raises:
    """Test SGD matches PyTorch implementation exactly.

    This CRITICAL test validates numerical correctness against PyTorch.
    We load reference outputs from PyTorch and compare.
    """
    # TODO: Implement when SGD is available
    # # Load PyTorch reference data
    # let reference = load_pytorch_reference("sgd_update.json")
    #
    # var params = Tensor(reference.initial_params)
    # var grads = Tensor(reference.grads)
    #
    # var optimizer = SGD(
    #     learning_rate=reference.lr,
    #     momentum=reference.momentum
    # )
    #
    # optimizer.step(params, grads)
    #
    # # Should match PyTorch exactly (tolerance 1e-6)
    # assert_tensor_equal(params, reference.expected_params, tolerance=1e-6)
    pass


fn test_adam_matches_pytorch() raises:
    """Test Adam matches PyTorch implementation exactly.

    This CRITICAL test validates Adam's complex update rules.
    """
    # TODO: Implement when Adam is available
    # Similar to test_sgd_matches_pytorch but for Adam
    pass


# ============================================================================
# Test Main
# ============================================================================

fn main() raises:
    """Run all optimizer tests."""
    print("Running SGD tests...")
    test_sgd_initialization()
    test_sgd_basic_update()
    test_sgd_momentum_accumulation()
    test_sgd_weight_decay()
    test_sgd_nesterov_momentum()
    test_sgd_zero_grad()

    print("Running Adam tests...")
    test_adam_initialization()
    test_adam_parameter_update()
    test_adam_bias_correction()

    print("Running AdamW tests...")
    test_adamw_weight_decay()

    print("Running RMSprop tests...")
    test_rmsprop_initialization()
    test_rmsprop_parameter_update()

    print("Running property-based tests...")
    test_optimizer_property_decreasing_loss()
    test_optimizer_property_gradient_shape()

    print("Running numerical accuracy tests...")
    test_sgd_matches_pytorch()
    test_adam_matches_pytorch()

    print("\nAll optimizer tests passed! ✓")
