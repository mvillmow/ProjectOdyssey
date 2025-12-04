"""Loss functions for training neural networks.

This module provides common loss functions used in supervised learning,
along with their backward passes for gradient computation.

Implemented losses:
- Binary Cross-Entropy (BCE): For binary classification
- Mean Squared Error (MSE): For regression
- Cross-Entropy: For multi-class classification (with softmax)
- Focal Loss: For addressing class imbalance
- KL Divergence: For distribution matching

All loss functions include:
- Numerical stability (epsilon handling, clipping)
- Proper gradient computation
- Support for batched inputs
"""

from .extensor import ExTensor, ones_like, zeros_like, full_like
from .arithmetic import add, subtract, multiply, divide, power
from .elementwise import log, clip, exp
from .reduction import mean, sum, max_reduce
from .activation import softmax


fn binary_cross_entropy(
    predictions: ExTensor, targets: ExTensor, epsilon: Float64 = 1e-7
) raises -> ExTensor:
    """Binary cross-entropy loss for binary classification.

    Formula:
        BCE = -[y*log(p) + (1-y)*log(1-p)]

    where:
        p = predictions (should be in [0, 1] range, typically from sigmoid)
        y = targets (ground truth labels, 0 or 1)

    Args:.        `predictions`: Predicted probabilities, shape (batch_size,) or (batch_size, 1)
        `targets`: Ground truth binary labels (0 or 1), same shape as predictions.
        `epsilon`: Small constant for numerical stability (prevents log(0))

    Returns:.        Loss tensor of same shape as inputs (element-wise loss)
        Use mean() or sum() to get scalar loss for backpropagation.

    Raises:.        Error if shapes don't match or dtypes are incompatible.

    Example:.        var predictions = sigmoid(logits)  # (batch_size,)
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
    var one = ones_like(clipped)

    var one_minus_pred = subtract(one, clipped)
    var log_one_minus_pred = log(one_minus_pred)

    # BCE = -[y*log(p) + (1-y)*log(1-p)]
    var term1 = multiply(targets, log_pred)
    var one_minus_targets = subtract(one, targets)
    var term2 = multiply(one_minus_targets, log_one_minus_pred)
    var sum_terms = add(term1, term2)

    # Negate: BCE = -(term1 + term2)
    var zero = zeros_like(sum_terms)

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
        ∂BCE/∂p = -y/p + (1-y)/(1-p)

    Which simplifies to:
        ∂BCE/∂p = (-y(1-p) + (1-y)p) / (p(1-p))
        ∂BCE/∂p = (p - y) / (p(1-p))

    For numerical stability, we add epsilon to the denominator:
        ∂BCE/∂p = (p - y) / (p(1-p) + epsilon)

    Args:
        `grad_output`: Gradient from upstream (e.g., from mean_backward)
        `predictions`: Original predictions passed to forward pass.
        `targets`: Original targets passed to forward pass.
        `epsilon`: Small constant for numerical stability (default: 1e-7)

    Returns:
        Gradient with respect to predictions, same shape as predictions.

    Example:
        # Forward
        var bce_loss = binary_cross_entropy(predictions, targets)
        var loss = mean(bce_loss)

        # Backward
        var grad_loss = ones_like(loss)
        var grad_bce = mean_backward(grad_loss, bce_loss)
        var grad_pred = binary_cross_entropy_backward(grad_bce, predictions, targets)
    """
    # Gradient formula: (p - y) / (p(1-p) + epsilon)
    var one = ones_like(predictions)

    # Numerator: (predictions - targets)
    var numerator = subtract(predictions, targets)

    # Denominator: p(1-p) + epsilon
    # First compute: (1 - predictions)
    var one_minus_pred = subtract(one, predictions)

    # Then compute: predictions * (1 - predictions)
    var pred_times_one_minus_pred = multiply(predictions, one_minus_pred)

    # Create epsilon tensor
    var epsilon_tensor = full_like(predictions, epsilon)

    # Compute: p(1-p) + epsilon
    var denominator = add(pred_times_one_minus_pred, epsilon_tensor)

    # Compute gradient: (p - y) / (p(1-p) + epsilon)
    var grad = divide(numerator, denominator)

    # Chain rule: multiply by upstream gradient
    return multiply(grad_output, grad)


fn mean_squared_error(predictions: ExTensor, targets: ExTensor) raises -> ExTensor:
    """Mean squared error loss for regression.

    Formula:
        MSE = (predictions - targets)^2.

    Returns element-wise squared error. Use mean() to get scalar loss.

    Args:.        `predictions`: Predicted values, any shape.
        `targets`: Ground truth values, same shape as predictions.

    Returns:.        Squared error tensor, same shape as inputs.
        Use mean() or sum() to get scalar loss for backpropagation.

    Raises:.        Error if shapes don't match or dtypes are incompatible.

    Example:.        var predictions = model(x)  # (batch_size, output_dim)
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

    Args:.        `grad_output`: Gradient from upstream (e.g., from mean_backward)
        `predictions`: Original predictions passed to forward pass.
        `targets`: Original targets passed to forward pass.

    Returns:.        Gradient with respect to predictions, same shape as predictions.

    Example:.        # Forward.
        var squared_error = mean_squared_error(predictions, targets)
        var loss = mean(squared_error)

        # Backward
        var grad_loss = ones_like(loss)
        var grad_squared_error = mean_backward(grad_loss, squared_error)
        var grad_pred = mean_squared_error_backward(grad_squared_error, predictions, targets)
    """
    # Gradient: 2 * (predictions - targets)
    var diff = subtract(predictions, targets)

    # Create tensor with value 2.0
    var two = full_like(diff, 2.0)

    var grad = multiply(two, diff)

    # Chain rule: multiply by upstream gradient
    return multiply(grad_output, grad)


fn cross_entropy(
    logits: ExTensor, targets: ExTensor, axis: Int = -1, epsilon: Float64 = 1e-7
) raises -> ExTensor:
    """Cross-entropy loss for multi-class classification.

    Formula:
        CE = -sum(targets * log(softmax(logits)))

    This implementation uses the log-sum-exp trick for numerical stability.

    Args:.        `logits`: Raw model outputs (before softmax), shape (batch_size, num_classes)
        `targets`: One-hot encoded ground truth, same shape as logits.
        `axis`: Axis along which to compute softmax (default: -1, last axis)
        `epsilon`: Small constant for numerical stability in log operations (default: 1e-7)

    Returns:.        Loss tensor, shape depends on reduction.
        Use mean() to get scalar loss for backpropagation.

    Raises:.        Error if shapes don't match or dtypes are incompatible.

    Example:.        var logits = model(x)           # (batch_size, num_classes)
        var targets_onehot = ...        # (batch_size, num_classes) one-hot
        var loss_per_sample = cross_entropy(logits, targets_onehot)
        var loss = mean(loss_per_sample)  # Scalar loss

    Note:
        For efficiency, this does NOT compute softmax explicitly.
        Instead, it uses: CE = -sum(targets * (logits - log_sum_exp(logits)))

    Numerical Stability:
        - Uses log-sum-exp trick to prevent overflow/underflow
        - Adds epsilon to log argument to prevent log(0)
    """
    if logits.dtype() != targets.dtype():
        raise Error("Logits and targets must have the same dtype")

    if logits.shape() != targets.shape():
        raise Error("Logits and targets must have the same shape")

    # Implement cross-entropy with log-sum-exp trick for numerical stability
    # CE = -sum(targets * (logits - log_sum_exp(logits)))
    # where log_sum_exp(x) = max(x) + log(sum(exp(x - max(x))))

    # Assume last axis is the class dimension
    var shape = logits.shape()
    var class_axis = len(shape) - 1 if axis == -1 else axis

    # Step 1: Find max along class axis for numerical stability
    var max_logits = max_reduce(logits, axis=class_axis, keepdims=True)

    # Step 2: Subtract max from logits: x_stable = x - max(x)
    var logits_stable = subtract(logits, max_logits)

    # Step 3: Compute exp(x_stable)
    var exp_logits = exp(logits_stable)

    # Step 4: Sum exp values along class axis
    var sum_exp = sum(exp_logits, axis=class_axis, keepdims=True)

    # Step 5: Compute log_sum_exp = max + log(sum_exp + epsilon)
    # Add epsilon for numerical stability to prevent log(0)
    var epsilon_tensor = full_like(sum_exp, epsilon)
    var sum_exp_stable = add(sum_exp, epsilon_tensor)
    var log_sum_exp = add(max_logits, log(sum_exp_stable))

    # Step 6: Compute log probabilities: log(p) = logits - log_sum_exp
    var log_probs = subtract(logits, log_sum_exp)

    # Step 7: Compute cross-entropy: CE = -sum(targets * log_probs)
    var ce_per_sample = multiply(targets, log_probs)
    var ce_sum = sum(ce_per_sample, axis=class_axis, keepdims=False)  # Sum over classes

    # Negate: create -1.0 scalar tensor
    var neg_one = full_like(ce_sum, -1.0)
    var ce = multiply(ce_sum, neg_one)

    # Return mean over batch (first dimension)
    return mean(ce, axis=0, keepdims=False)


fn cross_entropy_backward(
    grad_output: ExTensor, logits: ExTensor, targets: ExTensor, epsilon: Float64 = 1e-7
) raises -> ExTensor:
    """Backward pass for cross-entropy loss.

    For cross-entropy with softmax, the gradient simplifies to:
        ∂CE/∂logits = softmax(logits) - targets

    This beautiful result comes from the chain rule and the properties.
    of the softmax function.

    Args:.        `grad_output`: Gradient from upstream (scalar for mean reduction)
        `logits`: Original logits passed to forward pass, shape (batch, num_classes)
        `targets`: Original one-hot targets, shape (batch, num_classes)

    Returns:.        Gradient with respect to logits, shape (batch, num_classes)

    Example:.        ```mojo.
        from shared.core import cross_entropy, cross_entropy_backward

        # Forward pass
        var loss = cross_entropy(logits, targets)
        # Backward pass (grad_output is usually 1.0 for scalar loss)
        var grad_logits = cross_entropy_backward(grad_output, logits, targets)
        ```

    Note:
        The gradient is already averaged over the batch if the forward pass.
        used mean reduction.
    """
    # Compute softmax probabilities
    var axis = len(logits.shape()) - 1  # Last axis is classes
    var probs = softmax(logits, axis=axis)

    # Gradient: softmax(logits) - targets
    var grad = subtract(probs, targets)

    # Scale by upstream gradient and average over batch
    # NOTE: Forward pass already averages via mean(ce, axis=0), so we divide by batch_size here
    var batch_size = Float32(logits.shape()[0])
    var scale_val = 1.0 / batch_size

    # Create scale tensor with same shape as grad
    var scale_tensor = full_like(grad, Float64(scale_val))

    var grad_scaled = multiply(grad, scale_tensor)

    # Chain rule: multiply by upstream gradient
    return multiply(grad_scaled, grad_output)


fn focal_loss(
    predictions: ExTensor, targets: ExTensor, alpha: Float32 = 0.25, gamma: Float32 = 2.0
) raises -> ExTensor:
    """Focal loss for addressing class imbalance in classification.

    Formula:
        FL = -alpha * (1 - p)^gamma * target * log(p) - (1 - alpha) * p^gamma * (1 - target) * log(1 - p)

    where:
        p = predictions (probabilities, should be in [0, 1] range)
        target = ground truth labels (0 or 1)
        alpha = weighting factor (default: 0.25)
        gamma = focusing parameter (default: 2.0)

    The focal loss applies a modulating term (1 - p)^gamma to the cross entropy loss.
    This down-weights easy examples and focuses training on hard examples.
    It is particularly useful for addressing class imbalance.

    Args:
        `predictions`: Predicted probabilities, shape (batch_size,) or (batch_size, 1)
        `targets`: Ground truth binary labels (0 or 1), same shape as predictions
        `alpha`: Weighting factor for class 1 (default: 0.25)
        `gamma`: Focusing parameter (default: 2.0)

    Returns:
        Loss tensor of same shape as inputs (element-wise loss)
        Use mean() to get scalar loss for backpropagation

    Raises:
        Error if shapes don't match or dtypes are incompatible

    Example:
        var predictions = sigmoid(logits)  # (batch_size,)
        var targets = ...  # (batch_size,) with values 0 or 1
        var loss_per_sample = focal_loss(predictions, targets)
        var loss = mean(loss_per_sample)  # Scalar loss

    Numerical Stability:
        - Clips predictions to [epsilon, 1-epsilon] to prevent log(0)
        - Uses epsilon=1e-7 by default
    """
    if predictions.dtype() != targets.dtype():
        raise Error("Predictions and targets must have the same dtype")

    if predictions.shape() != targets.shape():
        raise Error("Predictions and targets must have the same shape")

    var epsilon = 1e-7

    # Clip predictions to prevent log(0) and log(1)
    var clipped = clip(predictions, epsilon, 1.0 - epsilon)

    # Compute log(p) and log(1-p)
    var log_pred = log(clipped)
    var one = ones_like(clipped)
    var one_minus_pred = subtract(one, clipped)
    var log_one_minus_pred = log(one_minus_pred)

    # Compute alpha and (1-alpha) tensors
    var alpha_tensor = full_like(clipped, Float64(alpha))
    var one_minus_alpha = subtract(one, alpha_tensor)

    # Compute (1 - p)^gamma for positive class
    var gamma_tensor = full_like(clipped, Float64(gamma))
    var one_minus_p_pow = power(one_minus_pred, gamma_tensor)

    # Compute p^gamma for negative class
    var p_pow = power(clipped, gamma_tensor)

    # Focal loss: -alpha * (1-p)^gamma * target * log(p) - (1-alpha) * p^gamma * (1-target) * log(1-p)
    var one_minus_targets = subtract(one, targets)

    # First term: -alpha * (1-p)^gamma * target * log(p)
    var term1 = multiply(alpha_tensor, one_minus_p_pow)
    term1 = multiply(term1, targets)
    term1 = multiply(term1, log_pred)

    # Second term: -(1-alpha) * p^gamma * (1-target) * log(1-p)
    var term2 = multiply(one_minus_alpha, p_pow)
    term2 = multiply(term2, one_minus_targets)
    term2 = multiply(term2, log_one_minus_pred)

    # Combine: FL = -(term1 + term2)
    var sum_terms = add(term1, term2)
    var zero = zeros_like(sum_terms)

    return subtract(zero, sum_terms)


fn focal_loss_backward(
    grad_output: ExTensor,
    predictions: ExTensor,
    targets: ExTensor,
    alpha: Float32 = 0.25,
    gamma: Float32 = 2.0
) raises -> ExTensor:
    """Backward pass for focal loss.

    Computes gradient of focal loss with respect to predictions.

    The gradient formula for focal loss is:
        ∂FL/∂p = -alpha * gamma * (1-p)^(gamma-1) * target * log(p)
                 - alpha * (1-p)^gamma * target / p
                 + (1-alpha) * gamma * p^(gamma-1) * (1-target) * log(1-p)
                 + (1-alpha) * p^gamma * (1-target) / (1-p)

    Args:
        `grad_output`: Gradient from upstream (e.g., from mean_backward)
        `predictions`: Original predictions passed to forward pass
        `targets`: Original targets passed to forward pass
        `alpha`: Weighting factor (default: 0.25)
        `gamma`: Focusing parameter (default: 2.0)

    Returns:
        Gradient with respect to predictions, same shape as predictions

    Example:
        # Forward
        var focal = focal_loss(predictions, targets, alpha, gamma)
        var loss = mean(focal)

        # Backward
        var grad_loss = ones_like(loss)
        var grad_focal = mean_backward(grad_loss, focal)
        var grad_pred = focal_loss_backward(grad_focal, predictions, targets, alpha, gamma)
    """
    var epsilon = 1e-7

    # Clip predictions to prevent division by zero
    var clipped = clip(predictions, epsilon, 1.0 - epsilon)
    var one = ones_like(clipped)
    var one_minus_pred = subtract(one, clipped)

    # Compute tensor versions of alpha and gamma
    var alpha_tensor = full_like(clipped, Float64(alpha))
    var one_minus_alpha = subtract(one, alpha_tensor)
    var gamma_tensor = full_like(clipped, Float64(gamma))
    var gamma_minus_one = subtract(gamma_tensor, ones_like(gamma_tensor))

    # Compute powers for gradient
    # (1-p)^(gamma-1)
    var one_minus_p_pow_gm1 = power(one_minus_pred, gamma_minus_one)
    # p^(gamma-1)
    var p_pow_gm1 = power(clipped, gamma_minus_one)
    # (1-p)^gamma
    var one_minus_p_pow_g = power(one_minus_pred, gamma_tensor)
    # p^gamma
    var p_pow_g = power(clipped, gamma_tensor)

    # Compute log terms
    var log_pred = log(clipped)
    var log_one_minus_pred = log(one_minus_pred)

    # Gradient computation:
    # ∂FL/∂p = -alpha * [gamma * (1-p)^(gamma-1) * target * log(p) + (1-p)^gamma * target / p]
    #          + (1-alpha) * [gamma * p^(gamma-1) * (1-target) * log(1-p) + p^gamma * (1-target) / (1-p)]

    var one_minus_targets = subtract(one, targets)

    # First component: -alpha * gamma * (1-p)^(gamma-1) * target * log(p)
    var term1_a = multiply(alpha_tensor, gamma_tensor)
    term1_a = multiply(term1_a, one_minus_p_pow_gm1)
    term1_a = multiply(term1_a, targets)
    term1_a = multiply(term1_a, log_pred)

    # Second component: -alpha * (1-p)^gamma * target / p
    var term1_b = multiply(alpha_tensor, one_minus_p_pow_g)
    term1_b = multiply(term1_b, targets)
    term1_b = divide(term1_b, clipped)

    # Combine first part: -(term1_a + term1_b)
    var term1 = add(term1_a, term1_b)
    var neg_one = full_like(term1, -1.0)
    term1 = multiply(term1, neg_one)

    # Third component: (1-alpha) * gamma * p^(gamma-1) * (1-target) * log(1-p)
    var term2_a = multiply(one_minus_alpha, gamma_tensor)
    term2_a = multiply(term2_a, p_pow_gm1)
    term2_a = multiply(term2_a, one_minus_targets)
    term2_a = multiply(term2_a, log_one_minus_pred)

    # Fourth component: (1-alpha) * p^gamma * (1-target) / (1-p)
    var term2_b = multiply(one_minus_alpha, p_pow_g)
    term2_b = multiply(term2_b, one_minus_targets)
    term2_b = divide(term2_b, one_minus_pred)

    # Combine second part
    var term2 = add(term2_a, term2_b)

    # Final gradient
    var grad = add(term1, term2)

    # Chain rule: multiply by upstream gradient
    return multiply(grad_output, grad)


fn kl_divergence(p: ExTensor, q: ExTensor, epsilon: Float64 = 1e-7) raises -> ExTensor:
    """Kullback-Leibler divergence loss for distribution matching.

    Formula:
        KL(p||q) = sum(p * log(p / q)) = sum(p * (log(p) - log(q)))

    where:
        p = reference distribution (target)
        q = approximating distribution (predicted)

    KL divergence measures how much one probability distribution differs from another.
    It is always non-negative and is zero only when p == q almost everywhere.

    Args:
        `p`: Reference distribution (target), should sum to 1 along class axis
        `q`: Approximating distribution (predicted), should sum to 1 along class axis
        `epsilon`: Small constant for numerical stability (default: 1e-7)

    Returns:
        KL divergence loss, reduced along class axis
        Use mean() to get scalar loss for backpropagation

    Raises:
        Error if shapes don't match or dtypes are incompatible

    Example:
        var p_dist = softmax(targets_logits, axis=1)  # (batch_size, num_classes)
        var q_dist = softmax(predictions_logits, axis=1)  # (batch_size, num_classes)
        var kl_per_sample = kl_divergence(p_dist, q_dist)
        var loss = mean(kl_per_sample)  # Scalar loss

    Note:
        This implementation assumes inputs are already probabilities (sum to 1).
        For raw logits, apply softmax first.

    Numerical Stability:
        - Clips both p and q to [epsilon, 1] to prevent log(0)
        - Handles zero probabilities gracefully
    """
    if p.dtype() != q.dtype():
        raise Error("p and q must have the same dtype")

    if p.shape() != q.shape():
        raise Error("p and q must have the same shape")

    # Clip both distributions to prevent log(0)
    var clipped_p = clip(p, epsilon, 1.0)
    var clipped_q = clip(q, epsilon, 1.0)

    # Compute log(p) and log(q)
    var log_p = log(clipped_p)
    var log_q = log(clipped_q)

    # Compute log(p/q) = log(p) - log(q)
    var log_ratio = subtract(log_p, log_q)

    # Compute p * log(p/q)
    var kl_per_element = multiply(p, log_ratio)

    # Sum along all axes except the first (batch) to get per-sample KL divergence
    # For 2D: (batch, classes) -> (batch,)
    # For 1D: (classes,) -> scalar
    var shape = p.shape()
    if len(shape) > 1:
        # Sum over class axis (axis 1 for 2D, or all axes except batch)
        var kl_per_sample = sum(kl_per_element, axis=-1, keepdims=False)
        return kl_per_sample
    else:
        # For 1D input, sum all elements
        var kl_total = sum(kl_per_element, axis=0, keepdims=False)
        return kl_total


fn kl_divergence_backward(
    grad_output: ExTensor, p: ExTensor, q: ExTensor, epsilon: Float64 = 1e-7
) raises -> ExTensor:
    """Backward pass for KL divergence loss.

    Computes gradient of KL divergence with respect to q (the predicted distribution).

    Formula:
        ∂KL/∂q = -p / q

    The gradient with respect to p is:
        ∂KL/∂p = log(p) - log(q) + 1 (not used in typical backprop since targets are fixed)

    Args:
        `grad_output`: Gradient from upstream (e.g., from mean_backward)
        `p`: Reference distribution passed to forward pass
        `q`: Approximating distribution passed to forward pass
        `epsilon`: Small constant for numerical stability (default: 1e-7)

    Returns:
        Gradient with respect to q, same shape as q

    Example:
        # Forward
        var kl = kl_divergence(p_dist, q_dist)
        var loss = mean(kl)

        # Backward
        var grad_loss = ones_like(loss)
        var grad_kl = mean_backward(grad_loss, kl)
        var grad_q = kl_divergence_backward(grad_kl, p_dist, q_dist)
    """
    # Clip q to prevent division by zero
    var clipped_q = clip(q, epsilon, 1.0)

    # Gradient: -p / q
    var grad = divide(p, clipped_q)
    var neg_one = full_like(grad, -1.0)
    grad = multiply(grad, neg_one)

    # Chain rule: multiply by upstream gradient
    return multiply(grad_output, grad)
