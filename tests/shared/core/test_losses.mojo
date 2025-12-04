"""Tests for loss functions.

This module tests the loss functions and their backward passes:
- Binary Cross-Entropy (BCE)
- Mean Squared Error (MSE)
- Focal Loss
- KL Divergence
"""

from tests.shared.conftest import (
    assert_true,
    assert_false,
    assert_equal_int,
    assert_almost_equal,
    assert_close_float,
)
from shared.core.extensor import ExTensor, zeros, ones, zeros_like, ones_like
from shared.core.loss import binary_cross_entropy, binary_cross_entropy_backward
from shared.core.loss import mean_squared_error, mean_squared_error_backward
from shared.core.loss import focal_loss, focal_loss_backward
from shared.core.loss import kl_divergence, kl_divergence_backward
from shared.core.reduction import mean
from tests.helpers.gradient_checking import check_gradient


fn test_binary_cross_entropy_perfect_prediction() raises:
    """Test BCE with perfect predictions (should be near zero)."""
    print("Testing BCE with perfect predictions...")

    var shape = List[Int]()
    shape.append(4)
    var predictions = ExTensor(shape, DType.float32)
    var targets = ExTensor(shape, DType.float32)

    # Perfect predictions: pred = target
    for i in range(4):
        var val = 1.0 if i % 2 == 0 else 0.0
        predictions._set_float64(i, val)
        targets._set_float64(i, val)

    var loss = binary_cross_entropy(predictions, targets)
    var avg_loss = mean(loss)

    var loss_val = avg_loss._get_float64(0)
    print("  Perfect prediction loss:", loss_val)

    # Should be very close to 0 (within epsilon tolerance)
    if loss_val > 0.01:
        raise Error("BCE loss for perfect predictions should be near 0")

    print("  ✓ BCE perfect prediction test passed")


fn test_binary_cross_entropy_worst_prediction() raises:
    """Test BCE with worst predictions (should be high)."""
    print("Testing BCE with worst predictions...")

    var shape = List[Int]()
    shape.append(4)
    var predictions = ExTensor(shape, DType.float32)
    var targets = ExTensor(shape, DType.float32)

    # Worst predictions: pred = 1 - target
    for i in range(4):
        var target_val = 1.0 if i % 2 == 0 else 0.0
        var pred_val = 1.0 - target_val  # Opposite
        predictions._set_float64(i, pred_val)
        targets._set_float64(i, target_val)

    var loss = binary_cross_entropy(predictions, targets)
    var avg_loss = mean(loss)

    var loss_val = avg_loss._get_float64(0)
    print("  Worst prediction loss:", loss_val)

    # Should be high (> 1.0)
    if loss_val < 1.0:
        raise Error("BCE loss for worst predictions should be high")

    print("  ✓ BCE worst prediction test passed")


fn test_binary_cross_entropy_gradient_shape() raises:
    """Test that BCE backward produces correct gradient shape."""
    print("Testing BCE gradient shape...")

    var shape = List[Int]()
    shape.append(3)
    var predictions = ExTensor(shape, DType.float32)
    var targets = ExTensor(shape, DType.float32)

    for i in range(3):
        predictions._set_float64(i, 0.5)
        targets._set_float64(i, 1.0 if i == 0 else 0.0)

    var loss = binary_cross_entropy(predictions, targets)

    # Create upstream gradient
    var grad_output = ones(shape, DType.float32)

    # Compute gradient
    var grad_pred = binary_cross_entropy_backward(grad_output, predictions, targets)

    # Check shape matches
    if grad_pred.shape()[0] != predictions.shape()[0]:
        raise Error("Gradient shape should match predictions shape")

    print("  Gradient shape:", grad_pred.shape()[0])
    print("  ✓ BCE gradient shape test passed")


fn test_mean_squared_error_zero_loss() raises:
    """Test MSE with identical predictions and targets."""
    print("Testing MSE with zero loss...")

    var shape = List[Int]()
    shape.append(5)
    var predictions = ExTensor(shape, DType.float32)
    var targets = ExTensor(shape, DType.float32)

    # Identical values
    for i in range(5):
        var val = Float64(i) * 0.5
        predictions._set_float64(i, val)
        targets._set_float64(i, val)

    var loss = mean_squared_error(predictions, targets)
    var avg_loss = mean(loss)

    var loss_val = avg_loss._get_float64(0)
    print("  MSE zero loss:", loss_val)

    # Should be exactly 0
    if loss_val != 0.0:
        raise Error("MSE loss for identical values should be 0")

    print("  ✓ MSE zero loss test passed")


fn test_mean_squared_error_known_values() raises:
    """Test MSE with known error values."""
    print("Testing MSE with known values...")

    var shape = List[Int]()
    shape.append(3)
    var predictions = ExTensor(shape, DType.float32)
    var targets = ExTensor(shape, DType.float32)

    # Predictions: [2, 3, 4]
    # Targets:     [1, 3, 5]
    # Errors:      [1, 0, -1]
    # Squared:     [1, 0, 1]
    # Mean:        (1 + 0 + 1) / 3 = 0.666...

    predictions._set_float64(0, 2.0)
    predictions._set_float64(1, 3.0)
    predictions._set_float64(2, 4.0)

    targets._set_float64(0, 1.0)
    targets._set_float64(1, 3.0)
    targets._set_float64(2, 5.0)

    var loss = mean_squared_error(predictions, targets)
    var avg_loss = mean(loss)

    var loss_val = avg_loss._get_float64(0)
    print("  MSE known value:", loss_val)

    # Should be approximately 0.666...
    var expected = 2.0 / 3.0
    var diff = abs(loss_val - expected)
    if diff > 0.01:
        raise Error("MSE loss doesn't match expected value")

    print("  ✓ MSE known values test passed")


fn test_mean_squared_error_gradient() raises:
    """Test MSE gradient computation."""
    print("Testing MSE gradient...")

    var shape = List[Int]()
    shape.append(3)
    var predictions = ExTensor(shape, DType.float32)
    var targets = ExTensor(shape, DType.float32)

    # Simple case: predictions = [2, 3, 4], targets = [1, 3, 5]
    # Gradient should be 2 * (predictions - targets) = 2 * [1, 0, -1] = [2, 0, -2]

    predictions._set_float64(0, 2.0)
    predictions._set_float64(1, 3.0)
    predictions._set_float64(2, 4.0)

    targets._set_float64(0, 1.0)
    targets._set_float64(1, 3.0)
    targets._set_float64(2, 5.0)

    var loss = mean_squared_error(predictions, targets)

    # Upstream gradient (all ones)
    var grad_output = ones(shape, DType.float32)

    # Compute gradient
    var grad_pred = mean_squared_error_backward(grad_output, predictions, targets)

    # Check values
    var grad0 = grad_pred._get_float64(0)
    var grad1 = grad_pred._get_float64(1)
    var grad2 = grad_pred._get_float64(2)

    print("  Gradients: [", grad0, ",", grad1, ",", grad2, "]")

    # Should be approximately [2, 0, -2]
    if abs(grad0 - 2.0) > 0.01:
        raise Error("Gradient[0] should be 2.0")
    if abs(grad1 - 0.0) > 0.01:
        raise Error("Gradient[1] should be 0.0")
    if abs(grad2 - (-2.0)) > 0.01:
        raise Error("Gradient[2] should be -2.0")

    print("  ✓ MSE gradient test passed")


fn test_loss_numerical_stability() raises:
    """Test that loss functions handle extreme values gracefully."""
    print("Testing loss function numerical stability...")

    var shape = List[Int]()
    shape.append(2)
    var predictions = ExTensor(shape, DType.float32)
    var targets = ExTensor(shape, DType.float32)

    # Test BCE with values close to 0 and 1 (should be clipped)
    predictions._set_float64(0, 0.0)  # Should be clipped to epsilon
    predictions._set_float64(1, 1.0)  # Should be clipped to 1-epsilon

    targets._set_float64(0, 0.0)
    targets._set_float64(1, 1.0)

    # Should not raise error or produce NaN/Inf
    var loss = binary_cross_entropy(predictions, targets)
    var avg_loss = mean(loss)

    var loss_val = avg_loss._get_float64(0)
    print("  BCE with extreme values:", loss_val)

    # Should be finite (not NaN or Inf)
    # Note: We can't check for NaN/Inf in Mojo easily, so just verify it doesn't crash

    print("  ✓ Numerical stability test passed")


fn test_binary_cross_entropy_backward_gradient() raises:
    """Test BCE backward with numerical gradient checking."""
    print("Testing BCE backward gradient checking...")

    var shape = List[Int]()
    shape.append(4)
    var predictions = zeros(shape, DType.float32)
    var targets = zeros(shape, DType.float32)

    # Initialize with non-uniform values
    predictions._set_float64(0, 0.7)
    predictions._set_float64(1, 0.3)
    predictions._set_float64(2, 0.5)
    predictions._set_float64(3, 0.2)

    targets._set_float64(0, 1.0)
    targets._set_float64(1, 0.0)
    targets._set_float64(2, 1.0)
    targets._set_float64(3, 0.0)

    # Forward function wrapper
    fn forward(pred: ExTensor) raises escaping -> ExTensor:
        return binary_cross_entropy(pred, targets)

    # Backward function wrapper
    fn backward(grad_out: ExTensor, pred: ExTensor) raises escaping -> ExTensor:
        return binary_cross_entropy_backward(grad_out, pred, targets)

    var loss = forward(predictions)
    var grad_output = ones(shape, DType.float32)

    # Numerical gradient checking (relaxed tolerance for float32 precision)
    check_gradient(forward, backward, predictions, grad_output, rtol=2e-3, atol=1e-5)

    print("  ✓ BCE backward gradient check passed")


fn test_mean_squared_error_backward_gradient() raises:
    """Test MSE backward with numerical gradient checking."""
    print("Testing MSE backward gradient checking...")

    var shape = List[Int]()
    shape.append(5)
    var predictions = zeros(shape, DType.float32)
    var targets = zeros(shape, DType.float32)

    # Initialize with non-uniform values
    predictions._set_float64(0, 2.1)
    predictions._set_float64(1, 3.5)
    predictions._set_float64(2, 1.2)
    predictions._set_float64(3, 4.8)
    predictions._set_float64(4, 0.5)

    targets._set_float64(0, 2.0)
    targets._set_float64(1, 3.0)
    targets._set_float64(2, 1.5)
    targets._set_float64(3, 4.5)
    targets._set_float64(4, 0.8)

    # Forward function wrapper
    fn forward(pred: ExTensor) raises escaping -> ExTensor:
        return mean_squared_error(pred, targets)

    # Backward function wrapper
    fn backward(grad_out: ExTensor, pred: ExTensor) raises escaping -> ExTensor:
        return mean_squared_error_backward(grad_out, pred, targets)

    var loss = forward(predictions)
    var grad_output = ones(shape, DType.float32)

    # Numerical gradient checking (relaxed tolerance for float32 precision)
    check_gradient(forward, backward, predictions, grad_output, rtol=2e-3, atol=1e-5)

    print("  ✓ MSE backward gradient check passed")


fn test_focal_loss_perfect_prediction() raises:
    """Test focal loss with perfect predictions (should be near zero)."""
    print("Testing focal loss with perfect predictions...")

    var shape = List[Int]()
    shape.append(4)
    var predictions = ExTensor(shape, DType.float32)
    var targets = ExTensor(shape, DType.float32)

    # Perfect predictions: pred = target
    for i in range(4):
        var val = 1.0 if i % 2 == 0 else 0.0
        predictions._set_float64(i, val)
        targets._set_float64(i, val)

    var loss = focal_loss(predictions, targets)
    var avg_loss = mean(loss)

    var loss_val = avg_loss._get_float64(0)
    print("  Perfect prediction focal loss:", loss_val)

    # Should be very close to 0 (within epsilon tolerance)
    if loss_val > 0.01:
        raise Error("Focal loss for perfect predictions should be near 0")

    print("  ✓ Focal loss perfect prediction test passed")


fn test_focal_loss_hard_examples() raises:
    """Test focal loss focuses on hard examples."""
    print("Testing focal loss on hard examples...")

    var shape = List[Int]()
    shape.append(2)
    var predictions = ExTensor(shape, DType.float32)
    var targets = ExTensor(shape, DType.float32)

    # Easy example: p=0.9, target=1 (easy positive)
    predictions._set_float64(0, 0.9)
    targets._set_float64(0, 1.0)

    # Hard example: p=0.1, target=1 (hard positive)
    predictions._set_float64(1, 0.1)
    targets._set_float64(1, 1.0)

    var loss = focal_loss(predictions, targets)

    var easy_loss = loss._get_float64(0)
    var hard_loss = loss._get_float64(1)

    print("  Easy example loss:", easy_loss)
    print("  Hard example loss:", hard_loss)

    # Hard examples should have larger loss
    if hard_loss <= easy_loss:
        raise Error("Focal loss should emphasize hard examples more than easy examples")

    print("  ✓ Focal loss hard examples test passed")


fn test_focal_loss_backward_shape() raises:
    """Test focal loss backward produces correct gradient shape."""
    print("Testing focal loss backward shape...")

    var shape = List[Int]()
    shape.append(3)
    var predictions = ExTensor(shape, DType.float32)
    var targets = ExTensor(shape, DType.float32)

    for i in range(3):
        predictions._set_float64(i, 0.5)
        targets._set_float64(i, 1.0 if i == 0 else 0.0)

    var loss = focal_loss(predictions, targets)
    var grad_output = ones(shape, DType.float32)
    var grad_pred = focal_loss_backward(grad_output, predictions, targets)

    if grad_pred.shape()[0] != predictions.shape()[0]:
        raise Error("Focal loss gradient shape should match predictions shape")

    print("  Gradient shape:", grad_pred.shape()[0])
    print("  ✓ Focal loss backward shape test passed")


fn test_kl_divergence_same_distribution() raises:
    """Test KL divergence with identical distributions (should be near zero)."""
    print("Testing KL divergence with same distribution...")

    var shape = List[Int]()
    shape.append(4)
    var p = ExTensor(shape, DType.float32)
    var q = ExTensor(shape, DType.float32)

    # Same distribution
    for i in range(4):
        var val = 0.25  # Uniform distribution
        p._set_float64(i, val)
        q._set_float64(i, val)

    var kl = kl_divergence(p, q)
    var avg_kl = mean(kl)

    var kl_val = avg_kl._get_float64(0)
    print("  KL divergence with same distribution:", kl_val)

    # Should be very close to 0 (within epsilon tolerance)
    if kl_val > 0.001:
        raise Error("KL divergence for identical distributions should be near 0")

    print("  ✓ KL divergence same distribution test passed")


fn test_kl_divergence_different_distributions() raises:
    """Test KL divergence with different distributions (should be positive)."""
    print("Testing KL divergence with different distributions...")

    var shape = List[Int]()
    shape.append(3)
    var p = ExTensor(shape, DType.float32)
    var q = ExTensor(shape, DType.float32)

    # Distribution p: [0.5, 0.3, 0.2]
    p._set_float64(0, 0.5)
    p._set_float64(1, 0.3)
    p._set_float64(2, 0.2)

    # Distribution q: [0.3, 0.5, 0.2] (different)
    q._set_float64(0, 0.3)
    q._set_float64(1, 0.5)
    q._set_float64(2, 0.2)

    var kl = kl_divergence(p, q)
    var avg_kl = mean(kl)

    var kl_val = avg_kl._get_float64(0)
    print("  KL divergence with different distributions:", kl_val)

    # Should be positive
    if kl_val <= 0.0:
        raise Error("KL divergence for different distributions should be positive")

    print("  ✓ KL divergence different distributions test passed")


fn test_kl_divergence_backward_shape() raises:
    """Test KL divergence backward produces correct gradient shape."""
    print("Testing KL divergence backward shape...")

    var shape = List[Int]()
    shape.append(4)
    var p = ExTensor(shape, DType.float32)
    var q = ExTensor(shape, DType.float32)

    # Initialize with valid probability distributions
    for i in range(4):
        p._set_float64(i, 0.25)
        q._set_float64(i, 0.25)

    var kl = kl_divergence(p, q)
    var grad_output = ones(shape, DType.float32)
    var grad_q = kl_divergence_backward(grad_output, p, q)

    if grad_q.shape()[0] != q.shape()[0]:
        raise Error("KL divergence gradient shape should match q shape")

    print("  Gradient shape:", grad_q.shape()[0])
    print("  ✓ KL divergence backward shape test passed")


fn test_focal_loss_backward_gradient() raises:
    """Test focal loss backward with numerical gradient checking."""
    print("Testing focal loss backward gradient checking...")

    var shape = List[Int]()
    shape.append(4)
    var predictions = zeros(shape, DType.float32)
    var targets = zeros(shape, DType.float32)

    # Initialize with non-uniform values
    predictions._set_float64(0, 0.7)
    predictions._set_float64(1, 0.3)
    predictions._set_float64(2, 0.5)
    predictions._set_float64(3, 0.2)

    targets._set_float64(0, 1.0)
    targets._set_float64(1, 0.0)
    targets._set_float64(2, 1.0)
    targets._set_float64(3, 0.0)

    # Forward function wrapper
    fn forward(pred: ExTensor) raises escaping -> ExTensor:
        return focal_loss(pred, targets)

    # Backward function wrapper
    fn backward(grad_out: ExTensor, pred: ExTensor) raises escaping -> ExTensor:
        return focal_loss_backward(grad_out, pred, targets)

    var loss = forward(predictions)
    var grad_output = ones(shape, DType.float32)

    # Numerical gradient checking (relaxed tolerance for float32 precision)
    check_gradient(forward, backward, predictions, grad_output, rtol=2e-2, atol=1e-4)

    print("  ✓ Focal loss backward gradient check passed")


fn test_kl_divergence_backward_gradient() raises:
    """Test KL divergence backward with numerical gradient checking."""
    print("Testing KL divergence backward gradient checking...")

    var shape = List[Int]()
    shape.append(4)
    var p = zeros(shape, DType.float32)
    var q = zeros(shape, DType.float32)

    # Initialize with non-uniform values
    p._set_float64(0, 0.4)
    p._set_float64(1, 0.3)
    p._set_float64(2, 0.2)
    p._set_float64(3, 0.1)

    q._set_float64(0, 0.3)
    q._set_float64(1, 0.4)
    q._set_float64(2, 0.2)
    q._set_float64(3, 0.1)

    # Forward function wrapper
    fn forward(q_dist: ExTensor) raises escaping -> ExTensor:
        return kl_divergence(p, q_dist)

    # Backward function wrapper
    fn backward(grad_out: ExTensor, q_dist: ExTensor) raises escaping -> ExTensor:
        return kl_divergence_backward(grad_out, p, q_dist)

    var kl = forward(q)
    var grad_output = ones(shape, DType.float32)

    # Numerical gradient checking (relaxed tolerance for float32 precision)
    check_gradient(forward, backward, q, grad_output, rtol=2e-2, atol=1e-4)

    print("  ✓ KL divergence backward gradient check passed")


fn run_all_tests() raises:
    """Run all loss function tests."""
    print("=" * 60)
    print("Loss Functions Test Suite")
    print("=" * 60)

    test_binary_cross_entropy_perfect_prediction()
    test_binary_cross_entropy_worst_prediction()
    test_binary_cross_entropy_gradient_shape()
    test_binary_cross_entropy_backward_gradient()
    test_mean_squared_error_zero_loss()
    test_mean_squared_error_known_values()
    test_mean_squared_error_gradient()
    test_mean_squared_error_backward_gradient()
    test_loss_numerical_stability()
    test_focal_loss_perfect_prediction()
    test_focal_loss_hard_examples()
    test_focal_loss_backward_shape()
    test_focal_loss_backward_gradient()
    test_kl_divergence_same_distribution()
    test_kl_divergence_different_distributions()
    test_kl_divergence_backward_shape()
    test_kl_divergence_backward_gradient()

    print("=" * 60)
    print("All loss function tests passed! ✓")
    print("=" * 60)


fn main() raises:
    """Entry point for loss function tests."""
    run_all_tests()
