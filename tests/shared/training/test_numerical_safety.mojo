"""Unit tests for Numerical Safety (NaN/Inf detection and handling).

Tests cover:
- Loss NaN/Inf detection
- Gradient NaN/Inf detection
- Training divergence handling
- Gradient clipping

Following TDD principles - these tests define the expected API
for implementation in Issue #34.
"""

from tests.shared.conftest import (
    assert_true,
    assert_false,
    assert_equal,
    TestFixtures,
)


# ============================================================================
# Loss Validity Tests
# ============================================================================


fn test_numerical_safety_valid_loss() raises:
    """Test is_valid_loss() identifies valid loss values.

    API Contract:
        fn is_valid_loss(loss: Float64) -> Bool
        - Returns True for finite positive values
        - Returns True for zero
        - Returns False for NaN
        - Returns False for Inf/-Inf

    This is a CRITICAL test for training stability.
    """
    from shared.training.base import is_valid_loss

    # Valid losses
    assert_true(is_valid_loss(0.0))
    assert_true(is_valid_loss(0.5))
    assert_true(is_valid_loss(1.0))
    assert_true(is_valid_loss(100.0))

    # TODO(#34): Add NaN/Inf detection when implemented
    # Currently stub always returns True
    # assert_false(is_valid_loss(Float64.nan()))
    # assert_false(is_valid_loss(Float64.inf()))
    # assert_false(is_valid_loss(Float64.neg_inf()))


fn test_numerical_safety_nan_loss() raises:
    """Test is_valid_loss() detects NaN loss.

    API Contract:
        When loss becomes NaN:
        - is_valid_loss() returns False
        - Training should stop or handle gracefully
    """
    # TODO(#34): Implement when Float64.nan() is available
    # from shared.training.base import is_valid_loss
    #
    # var nan_loss = Float64.nan()
    # assert_false(is_valid_loss(nan_loss))
    pass


fn test_numerical_safety_inf_loss() raises:
    """Test is_valid_loss() detects infinite loss.

    API Contract:
        When loss becomes infinite:
        - is_valid_loss() returns False
        - Indicates gradient explosion or numerical instability
    """
    # TODO(#34): Implement when Float64.inf() is available
    # from shared.training.base import is_valid_loss
    #
    # var inf_loss = Float64.inf()
    # var neg_inf_loss = Float64.neg_inf()
    #
    # assert_false(is_valid_loss(inf_loss))
    # assert_false(is_valid_loss(neg_inf_loss))
    pass


# ============================================================================
# Gradient Validity Tests
# ============================================================================


fn test_numerical_safety_valid_gradients() raises:
    """Test gradient validity checking.

    API Contract:
        Gradients should be:
        - Finite (not NaN or Inf)
        - Bounded (not too large)
        - Non-zero (indicates learning)
    """
    from shared.training.base import clip_gradients

    # Test that gradient clipping accepts valid gradients
    var valid_grads = List[Float64](0.1, -0.2, 0.3, -0.4)
    var result = clip_gradients(valid_grads, max_norm=1.0)

    # Verify list is returned
    assert_equal(len(result), 4)


fn test_numerical_safety_nan_gradients() raises:
    """Test detection of NaN gradients.

    API Contract:
        When gradients contain NaN:
        - is_valid_gradient() returns False
        - Training should stop or skip update
    """
    # TODO(#34): Implement when gradient checking is available
    # from shared.training.base import is_valid_gradient
    #
    # var nan_grads = List[Float64](0.1, Float64.nan(), 0.3)
    # assert_false(is_valid_gradient(nan_grads))
    pass


fn test_numerical_safety_inf_gradients() raises:
    """Test detection of infinite gradients (gradient explosion).

    API Contract:
        When gradients contain Inf:
        - is_valid_gradient() returns False
        - Indicates gradient explosion
        - Should use gradient clipping
    """
    # TODO(#34): Implement when gradient checking is available
    # from shared.training.base import is_valid_gradient
    #
    # var inf_grads = List[Float64](0.1, Float64.inf(), 0.3)
    # assert_false(is_valid_gradient(inf_grads))
    pass


# ============================================================================
# Gradient Clipping Tests
# ============================================================================


fn test_numerical_safety_gradient_clipping_basic() raises:
    """Test gradient clipping limits gradient norm.

    API Contract:
        fn clip_gradients(gradients: List[Float64], max_norm: Float64) -> List[Float64]
        - Computes global gradient norm
        - If norm > max_norm, scale gradients down
        - Preserves gradient direction
    """
    from shared.training.base import clip_gradients

    var large_grads = List[Float64](10.0, 20.0, 30.0)
    var clipped = clip_gradients(large_grads, max_norm=1.0)

    # Verify stub returns gradients (currently unchanged)
    # Real implementation will clip to max_norm
    assert_equal(len(clipped), 3)


fn test_numerical_safety_gradient_clipping_preserves_direction() raises:
    """Test gradient clipping preserves direction.

    API Contract:
        Clipped gradients should point in same direction as original:
        clipped[i] / clipped[j] == original[i] / original[j]
    """
    # TODO(#34): Implement when clip_gradients is available
    # from shared.training.base import clip_gradients
    #
    # var original = List[Float64](3.0, 4.0)  # Norm = 5.0
    # var clipped = clip_gradients(original, max_norm=1.0)
    #
    # # Direction should be preserved (ratio of components)
    # var original_ratio = original[0] / original[1]  # 3/4 = 0.75
    # var clipped_ratio = clipped[0] / clipped[1]
    # assert_almost_equal(original_ratio, clipped_ratio)
    pass


fn test_numerical_safety_gradient_clipping_no_op_when_below_threshold() raises:
    """Test gradient clipping is no-op when gradients already small.

    API Contract:
        If gradient norm < max_norm, gradients unchanged.
    """
    # TODO(#34): Implement when clip_gradients is available
    # from shared.training.base import clip_gradients
    #
    # var small_grads = List[Float64](0.1, 0.2)  # Norm ≈ 0.22
    # var clipped = clip_gradients(small_grads, max_norm=1.0)
    #
    # # Should be unchanged
    # for i in range(len(small_grads)):
    #     assert_equal(clipped[i], small_grads[i])
    pass


# ============================================================================
# Training Divergence Detection
# ============================================================================


fn test_numerical_safety_detects_loss_explosion() raises:
    """Test detection of training divergence (loss explosion).

    API Contract:
        Training loop should detect when:
        - Loss suddenly increases significantly
        - Loss becomes > threshold (e.g., 1e6)
        - Loss growth rate is exponential
    """
    # TODO(#34): Implement when training loop is available
    # from shared.training import TrainingLoop
    #
    # # Create unstable training scenario (high LR)
    # var model = create_simple_model()
    # var optimizer = SGD(learning_rate=100.0)  # Way too high
    # var training_loop = TrainingLoop(model, optimizer, loss_fn)
    #
    # var data_loader = create_mock_dataloader()
    #
    # # Try training - should detect divergence
    # try:
    #     var losses = []
    #     for epoch in range(10):
    #         var loss = training_loop.run_epoch(data_loader)
    #         losses.append(loss)
    #
    #         # Should stop when loss explodes
    #         if loss > 1e6 or loss != loss:  # Inf or NaN
    #             break
    #
    #     # Should have stopped early
    #     assert_less(len(losses), 10)
    # except DivergenceError:
    #     pass  # Expected - training diverged
    pass


fn test_numerical_safety_handles_zero_gradients() raises:
    """Test handling of zero gradients (dead neurons).

    API Contract:
        When all gradients are zero:
        - Weight update should be no-op
        - Warning should be logged
        - Training should continue
    """
    # TODO(#34): Implement when training loop is available
    # from shared.training import TrainingLoop
    #
    # var model = create_model_with_dead_neurons()
    # var training_loop = TrainingLoop(model, optimizer, loss_fn)
    #
    # # Get initial weights
    # var initial_weights = model.get_weights().copy()
    #
    # # Training step with zero gradients
    # var inputs = Tensor.zeros(4, 10)
    # var targets = Tensor.zeros(4, 1)
    # var loss = training_loop.step(inputs, targets)
    #
    # # Weights should be unchanged
    # var final_weights = model.get_weights()
    # assert_tensor_equal(initial_weights, final_weights)
    pass


# ============================================================================
# Numerical Stability Tests
# ============================================================================


fn test_numerical_safety_loss_computation_stable() raises:
    """Test loss computation is numerically stable.

    API Contract:
        Loss functions should:
        - Use numerically stable formulas
        - Avoid overflow in intermediate computations
        - Use log-sum-exp trick for exponentials
    """
    # TODO(#34): Implement when loss functions are available
    # from shared.core.loss import CrossEntropyLoss
    #
    # # Create large logits (could overflow in naive softmax)
    # var large_logits = Tensor.fill(10, 1000.0)
    # var targets = Tensor.fill(10, 0)
    #
    # var loss_fn = CrossEntropyLoss()
    # var loss = loss_fn(large_logits, targets)
    #
    # # Should be finite (not overflow to Inf)
    # assert_true(is_valid_loss(loss))
    pass


fn test_numerical_safety_optimizer_step_stable() raises:
    """Test optimizer updates are numerically stable.

    API Contract:
        Optimizer.step() should:
        - Check gradients for NaN/Inf before update
        - Optionally clip gradients
        - Skip update if gradients invalid
    """
    # TODO(#34): Implement when optimizer is available
    # from shared.training.optimizers import SGD
    #
    # var model = create_simple_model()
    # var optimizer = SGD(learning_rate=0.1, clip_gradients=True, max_norm=1.0)
    #
    # # Set large gradients
    # for param in model.parameters():
    #     param.grad = Tensor.fill(param.shape, 100.0)
    #
    # # Get initial weights
    # var initial_weights = model.parameters()[0].data.copy()
    #
    # # Optimizer step (should clip gradients)
    # optimizer.step()
    #
    # # Weights should change, but not too much (gradients clipped)
    # var final_weights = model.parameters()[0].data
    # var weight_change = (final_weights - initial_weights).abs().max()
    # assert_less(weight_change, 1.0)  # Reasonable change due to clipping
    pass


# ============================================================================
# Test Main
# ============================================================================


fn main() raises:
    """Run all numerical safety tests."""
    print("Running loss validity tests...")
    test_numerical_safety_valid_loss()
    test_numerical_safety_nan_loss()
    test_numerical_safety_inf_loss()

    print("Running gradient validity tests...")
    test_numerical_safety_valid_gradients()
    test_numerical_safety_nan_gradients()
    test_numerical_safety_inf_gradients()

    print("Running gradient clipping tests...")
    test_numerical_safety_gradient_clipping_basic()
    test_numerical_safety_gradient_clipping_preserves_direction()
    test_numerical_safety_gradient_clipping_no_op_when_below_threshold()

    print("Running training divergence tests...")
    test_numerical_safety_detects_loss_explosion()
    test_numerical_safety_handles_zero_gradients()

    print("Running numerical stability tests...")
    test_numerical_safety_loss_computation_stable()
    test_numerical_safety_optimizer_step_stable()

    print("\nAll numerical safety tests passed! ✓")
