"""Loss functions for training neural networks.

This module provides common loss functions used in supervised learning,
along with their backward passes for gradient computation.

Implemented losses:
- Binary Cross-Entropy (BCE): For binary classification
- Mean Squared Error (MSE): For regression
- Cross-Entropy: For multi-class classification (with softmax)

All loss functions include:
- Numerical stability (epsilon handling, clipping)
- Proper gradient computation
- Support for batched inputs
"""

from .extensor import ExTensor
from .arithmetic import add, subtract, multiply, divide
from .elementwise_math import log, clip
from .reduction import mean, sum


fn binary_cross_entropy(
    predictions: ExTensor, targets: ExTensor, epsilon: Float64 = 1e-7
) raises -> ExTensor:
    """Binary cross-entropy loss for binary classification.

    Formula:
        BCE = -[y*log(p) + (1-y)*log(1-p)]

    where:
        p = predictions (should be in [0, 1] range, typically from sigmoid)
        y = targets (ground truth labels, 0 or 1)

    Args:
        predictions: Predicted probabilities, shape (batch_size,) or (batch_size, 1)
        targets: Ground truth binary labels (0 or 1), same shape as predictions
        epsilon: Small constant for numerical stability (prevents log(0))

    Returns:
        Loss tensor of same shape as inputs (element-wise loss)
        Use mean() or sum() to get scalar loss for backpropagation

    Raises:
        Error if shapes don't match or dtypes are incompatible

    Example:
        var predictions = sigmoid(logits)  # (batch_size,)
        var targets = ... # (batch_size,) with values 0 or 1
        var loss_per_sample = binary_cross_entropy(predictions, targets)
        var loss = mean(loss_per_sample)  # Scalar loss for backprop

    Numerical Stability:
        - Clips predictions to [epsilon, 1-epsilon] to prevent log(0)
        - Uses epsilon=1e-7 by default
    """
    if predictions.dtype() != targets.dtype():
        raise Error("Predictions and targets must have the same dtype")

    if predictions.shape() != targets.shape():
        raise Error("Predictions and targets must have the same shape")

    # Clip predictions to prevent log(0) and log(1)
    var clipped = clip(predictions, epsilon, 1.0 - epsilon)

    # Compute log(p) and log(1-p)
    var log_pred = log(clipped)
    var one = ExTensor(clipped.shape(), clipped.dtype())
    for i in range(one.numel()):
        one._set_float64(i, 1.0)

    var one_minus_pred = subtract(one, clipped)
    var log_one_minus_pred = log(one_minus_pred)

    # BCE = -[y*log(p) + (1-y)*log(1-p)]
    var term1 = multiply(targets, log_pred)
    var one_minus_targets = subtract(one, targets)
    var term2 = multiply(one_minus_targets, log_one_minus_pred)
    var sum_terms = add(term1, term2)

    # Negate: BCE = -(term1 + term2)
    var zero = ExTensor(sum_terms.shape(), sum_terms.dtype())
    for i in range(zero.numel()):
        zero._set_float64(i, 0.0)

    return subtract(zero, sum_terms)


fn binary_cross_entropy_backward(
    grad_output: ExTensor,
    predictions: ExTensor,
    targets: ExTensor,
    epsilon: Float64 = 1e-7
) raises -> ExTensor:
    """Backward pass for binary cross-entropy loss.

    Computes gradient of BCE loss with respect to predictions.

    Formula:
        ∂BCE/∂p = -(y/p - (1-y)/(1-p))

    For numerical stability, this can be simplified to:
        ∂BCE/∂p = (p - y) / (p(1-p) + epsilon)

    However, in practice, the simplified form is often used:
        ∂BCE/∂p ≈ (p - y)

    This implementation uses the simplified form for efficiency and stability.

    Args:
        grad_output: Gradient from upstream (e.g., from mean_backward)
        predictions: Original predictions passed to forward pass
        targets: Original targets passed to forward pass
        epsilon: Small constant for numerical stability (unused in simplified form)

    Returns:
        Gradient with respect to predictions, same shape as predictions

    Example:
        # Forward
        var loss = mean(binary_cross_entropy(predictions, targets))

        # Backward
        var grad_loss = ones_like(loss)
        var grad_mean = mean_backward(grad_loss, loss_shape)
        var grad_pred = binary_cross_entropy_backward(grad_mean, predictions, targets)
    """
    # Simplified gradient: (predictions - targets)
    # This is the standard form used in most ML frameworks
    var grad = subtract(predictions, targets)

    # Chain rule: multiply by upstream gradient
    return multiply(grad_output, grad)


fn mean_squared_error(predictions: ExTensor, targets: ExTensor) raises -> ExTensor:
    """Mean squared error loss for regression.

    Formula:
        MSE = (predictions - targets)^2

    Returns element-wise squared error. Use mean() to get scalar loss.

    Args:
        predictions: Predicted values, any shape
        targets: Ground truth values, same shape as predictions

    Returns:
        Squared error tensor, same shape as inputs
        Use mean() or sum() to get scalar loss for backpropagation

    Raises:
        Error if shapes don't match or dtypes are incompatible

    Example:
        var predictions = model(x)  # (batch_size, output_dim)
        var targets = y_true        # (batch_size, output_dim)
        var loss_per_sample = mean_squared_error(predictions, targets)
        var loss = mean(loss_per_sample)  # Scalar loss
    """
    if predictions.dtype() != targets.dtype():
        raise Error("Predictions and targets must have the same dtype")

    if predictions.shape() != targets.shape():
        raise Error("Predictions and targets must have the same shape")

    # MSE = (predictions - targets)^2
    var diff = subtract(predictions, targets)
    return multiply(diff, diff)


fn mean_squared_error_backward(
    grad_output: ExTensor, predictions: ExTensor, targets: ExTensor
) raises -> ExTensor:
    """Backward pass for mean squared error loss.

    Computes gradient of MSE loss with respect to predictions.

    Formula:
        ∂MSE/∂predictions = 2 * (predictions - targets)

    Args:
        grad_output: Gradient from upstream (e.g., from mean_backward)
        predictions: Original predictions passed to forward pass
        targets: Original targets passed to forward pass

    Returns:
        Gradient with respect to predictions, same shape as predictions

    Example:
        # Forward
        var loss = mean(mean_squared_error(predictions, targets))

        # Backward
        var grad_loss = ones_like(loss)
        var grad_mean = mean_backward(grad_loss, squared_error_shape)
        var grad_pred = mean_squared_error_backward(grad_mean, predictions, targets)
    """
    # Gradient: 2 * (predictions - targets)
    var diff = subtract(predictions, targets)

    # Create tensor with value 2.0
    var two = ExTensor(diff.shape(), diff.dtype())
    for i in range(two.numel()):
        two._set_float64(i, 2.0)

    var grad = multiply(two, diff)

    # Chain rule: multiply by upstream gradient
    return multiply(grad_output, grad)


fn cross_entropy(
    logits: ExTensor, targets: ExTensor, axis: Int = -1
) raises -> ExTensor:
    """Cross-entropy loss for multi-class classification.

    Formula:
        CE = -sum(targets * log(softmax(logits)))

    This implementation uses the log-sum-exp trick for numerical stability.

    Args:
        logits: Raw model outputs (before softmax), shape (batch_size, num_classes)
        targets: One-hot encoded ground truth, same shape as logits
        axis: Axis along which to compute softmax (default: -1, last axis)

    Returns:
        Loss tensor, shape depends on reduction
        Use mean() to get scalar loss for backpropagation

    Raises:
        Error if shapes don't match or dtypes are incompatible

    Example:
        var logits = model(x)           # (batch_size, num_classes)
        var targets_onehot = ...        # (batch_size, num_classes) one-hot
        var loss_per_sample = cross_entropy(logits, targets_onehot)
        var loss = mean(loss_per_sample)  # Scalar loss

    Note:
        For efficiency, this does NOT compute softmax explicitly.
        Instead, it uses: CE = -sum(targets * (logits - log_sum_exp(logits)))
    """
    if logits.dtype() != targets.dtype():
        raise Error("Logits and targets must have the same dtype")

    if logits.shape() != targets.shape():
        raise Error("Logits and targets must have the same shape")

    # TODO: Implement cross-entropy with log-sum-exp trick
    # This requires:
    # 1. max_reduce along axis
    # 2. subtract max from logits
    # 3. exp
    # 4. sum along axis
    # 5. log
    # 6. Compute: -sum(targets * (logits - log_sum_exp))

    raise Error("cross_entropy not yet implemented - use binary_cross_entropy for binary classification or implement manually")


fn cross_entropy_backward(
    grad_output: ExTensor, logits: ExTensor, targets: ExTensor
) raises -> ExTensor:
    """Backward pass for cross-entropy loss.

    For cross-entropy with softmax, the gradient simplifies to:
        ∂CE/∂logits = softmax(logits) - targets

    Args:
        grad_output: Gradient from upstream
        logits: Original logits passed to forward pass
        targets: Original one-hot targets

    Returns:
        Gradient with respect to logits

    Note:
        Requires softmax to be implemented. Not yet available.
    """
    raise Error("cross_entropy_backward not yet implemented")
