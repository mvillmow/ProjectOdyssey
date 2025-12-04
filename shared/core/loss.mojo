"""Loss functions for training neural networks.

This module provides common loss functions used in supervised learning,
along with their backward passes for gradient computation.

Implemented losses:
- Binary Cross-Entropy (BCE): For binary classification
- Mean Squared Error (MSE): For regression
- Cross-Entropy: For multi-class classification (with softmax)
- Smooth L1 Loss (Huber Loss): Robust regression loss
- Hinge Loss: For Support Vector Machines (SVMs)

All loss functions include:
- Numerical stability (epsilon handling, clipping)
- Proper gradient computation
- Support for batched inputs
"""

from .extensor import ExTensor, ones_like, zeros_like, full_like
from .arithmetic import add, subtract, multiply, divide
from .elementwise import log, clip, exp, abs
from .reduction import mean, sum, max_reduce
from .activation import softmax
from .comparison import less
from .dtype_dispatch import dispatch_binary, dispatch_scalar


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


fn smooth_l1_loss(
    predictions: ExTensor, targets: ExTensor, beta: Float32 = 1.0
) raises -> ExTensor:
    """Smooth L1 loss (Huber loss) for robust regression.

    Formula:
        If |x| < beta:
            L = 0.5 * (x)^2 / beta
        Else:
            L = |x| - 0.5 * beta

        where x = predictions - targets

    This loss is less sensitive to outliers than MSE, making it more robust
    for regression tasks with noisy data.

    Args:
        `predictions`: Predicted values, any shape.
        `targets`: Ground truth values, same shape as predictions.
        `beta`: Threshold parameter that controls the transition between L2 and L1.
                Smaller beta makes the function more similar to L1.
                Default: 1.0

    Returns:
        Loss tensor, same shape as inputs. Use mean() to get scalar loss.

    Raises:
        Error if shapes don't match or dtypes are incompatible.

    Example:
        var predictions = model(x)  # (batch_size, output_dim)
        var targets = y_true        # (batch_size, output_dim)
        var loss_per_sample = smooth_l1_loss(predictions, targets, beta=1.0)
        var loss = mean(loss_per_sample)  # Scalar loss for backprop

    Numerical Stability:
        - Uses absolute value for robust handling of differences
        - Beta parameter prevents division by zero in gradient
    """
    if predictions.dtype() != targets.dtype():
        raise Error("Predictions and targets must have the same dtype")

    if predictions.shape() != targets.shape():
        raise Error("Predictions and targets must have the same shape")

    # Compute differences: x = predictions - targets
    var diff = subtract(predictions, targets)

    # Use dispatch to compute smooth L1 element-wise
    @always_inline
    fn _smooth_l1_op[T: DType](pred: Scalar[T], targ: Scalar[T]) -> Scalar[T]:
        var diff_val = pred - targ
        # Compute absolute value
        var abs_diff = diff_val if diff_val >= Scalar[T](0) else -diff_val
        var beta_val = Scalar[T](beta)
        var half_beta = beta_val * Scalar[T](0.5)

        # if |x| < beta: return 0.5 * x^2 / beta
        # else: return |x| - 0.5 * beta
        if abs_diff < beta_val:
            return Scalar[T](0.5) * diff_val * diff_val / beta_val
        else:
            return abs_diff - half_beta

    return dispatch_binary[_smooth_l1_op](predictions, targets)


fn smooth_l1_loss_backward(
    grad_output: ExTensor,
    predictions: ExTensor,
    targets: ExTensor,
    beta: Float32 = 1.0
) raises -> ExTensor:
    """Backward pass for Smooth L1 loss (Huber loss).

    Computes gradient of Smooth L1 loss with respect to predictions.

    Formula:
        If |x| < beta:
            ∂L/∂pred = x / beta
        Else:
            ∂L/∂pred = sign(x)

        where x = predictions - targets, sign(x) = 1 if x > 0 else -1

    Args:
        `grad_output`: Gradient from upstream (e.g., from mean_backward)
        `predictions`: Original predictions passed to forward pass.
        `targets`: Original targets passed to forward pass.
        `beta`: Threshold parameter (must match forward pass).

    Returns:
        Gradient with respect to predictions, same shape as predictions.

    Example:
        # Forward
        var smoothl1_loss = smooth_l1_loss(predictions, targets, beta=1.0)
        var loss = mean(smoothl1_loss)

        # Backward
        var grad_loss = ones_like(loss)
        var grad_smoothl1 = mean_backward(grad_loss, smoothl1_loss)
        var grad_pred = smooth_l1_loss_backward(grad_smoothl1, predictions, targets, beta=1.0)
    """
    if grad_output.dtype() != predictions.dtype():
        raise Error("smooth_l1_loss_backward: grad_output and predictions must have same dtype")
    if grad_output.shape() != predictions.shape():
        raise Error("smooth_l1_loss_backward: grad_output and predictions must have same shape")

    # Use dispatch to compute gradient element-wise
    @always_inline
    fn _smooth_l1_backward_op[T: DType](grad: Scalar[T], pred: Scalar[T], targ: Scalar[T]) -> Scalar[T]:
        var diff_val = pred - targ
        # Compute absolute value
        var abs_diff = diff_val if diff_val >= Scalar[T](0) else -diff_val
        var beta_val = Scalar[T](beta)

        # if |x| < beta: return grad * (x / beta)
        # else: return grad * sign(x)
        if abs_diff < beta_val:
            # Gradient: x / beta
            return grad * diff_val / beta_val
        else:
            # Gradient: sign(x)
            var sign_val = diff_val if diff_val > Scalar[T](0) else (diff_val if diff_val < Scalar[T](0) else Scalar[T](0))
            return grad * sign_val

    # Create a 3-argument wrapper by manually iterating
    var result = ExTensor(predictions.shape(), predictions.dtype())
    var pred_ptr = predictions._data.bitcast[Scalar[predictions.dtype()]]()
    var targ_ptr = targets._data.bitcast[Scalar[targets.dtype()]]()
    var grad_ptr = grad_output._data.bitcast[Scalar[grad_output.dtype()]]()
    var result_ptr = result._data.bitcast[Scalar[predictions.dtype()]]()

    for i in range(predictions._numel):
        result_ptr[i] = _smooth_l1_backward_op[predictions.dtype()](grad_ptr[i], pred_ptr[i], targ_ptr[i])

    return result


fn hinge_loss(predictions: ExTensor, targets: ExTensor) raises -> ExTensor:
    """Hinge loss for Support Vector Machines (SVMs).

    Formula:
        L = max(0, 1 - y * pred)

    where:
        y = targets (must be -1 or 1)
        pred = predictions (model output)

    The hinge loss penalizes predictions that are not confident enough.
    A prediction is correct when y * pred >= 1 (margin of 1).

    Args:
        `predictions`: Model predictions (real-valued scores).
        `targets`: Ground truth labels, must be -1 or 1, same shape as predictions.

    Returns:
        Loss tensor, same shape as inputs. Use mean() to get scalar loss.

    Raises:
        Error if shapes don't match or dtypes are incompatible.

    Example:
        var predictions = model(x)  # (batch_size,) or (batch_size, 1)
        var targets = y_true        # (batch_size,) with values -1 or 1
        var loss_per_sample = hinge_loss(predictions, targets)
        var loss = mean(loss_per_sample)  # Scalar loss for backprop

    Note:
        Hinge loss is typically used with hard labels (-1 or 1) rather than
        probabilities. For multi-class SVM, use with one-vs-rest strategy.

    Numerical Stability:
        - Uses max(0, ...) to prevent negative losses
        - Avoids numerical issues with extreme values
    """
    if predictions.dtype() != targets.dtype():
        raise Error("Predictions and targets must have the same dtype")

    if predictions.shape() != targets.shape():
        raise Error("Predictions and targets must have the same shape")

    # Use dispatch to compute hinge loss element-wise
    @always_inline
    fn _hinge_loss_op[T: DType](pred: Scalar[T], targ: Scalar[T]) -> Scalar[T]:
        var y_pred = targ * pred
        var margin = Scalar[T](1) - y_pred

        # return max(0, margin)
        return margin if margin > Scalar[T](0) else Scalar[T](0)

    return dispatch_binary[_hinge_loss_op](predictions, targets)


fn hinge_loss_backward(
    grad_output: ExTensor,
    predictions: ExTensor,
    targets: ExTensor
) raises -> ExTensor:
    """Backward pass for hinge loss.

    Computes gradient of hinge loss with respect to predictions.

    Formula:
        If y * pred < 1 (margin violated):
            ∂L/∂pred = -y
        Else (margin satisfied):
            ∂L/∂pred = 0

    where y = targets, pred = predictions

    Args:
        `grad_output`: Gradient from upstream (e.g., from mean_backward)
        `predictions`: Original predictions passed to forward pass.
        `targets`: Original targets passed to forward pass (-1 or 1).

    Returns:
        Gradient with respect to predictions, same shape as predictions.

    Example:
        # Forward
        var hinge = hinge_loss(predictions, targets)
        var loss = mean(hinge)

        # Backward
        var grad_loss = ones_like(loss)
        var grad_hinge = mean_backward(grad_loss, hinge)
        var grad_pred = hinge_loss_backward(grad_hinge, predictions, targets)
    """
    if grad_output.dtype() != predictions.dtype():
        raise Error("hinge_loss_backward: grad_output and predictions must have same dtype")
    if grad_output.shape() != predictions.shape():
        raise Error("hinge_loss_backward: grad_output and predictions must have same shape")

    # Use dispatch to compute gradient element-wise
    @always_inline
    fn _hinge_backward_op[T: DType](grad: Scalar[T], pred: Scalar[T], targ: Scalar[T]) -> Scalar[T]:
        var y_pred = targ * pred
        # if y * pred < 1: return grad * (-y)
        # else: return 0
        if y_pred < Scalar[T](1):
            return grad * (-targ)
        else:
            return Scalar[T](0)

    # Create a 3-argument wrapper by manually iterating
    var result = ExTensor(predictions.shape(), predictions.dtype())
    var pred_ptr = predictions._data.bitcast[Scalar[predictions.dtype()]]()
    var targ_ptr = targets._data.bitcast[Scalar[targets.dtype()]]()
    var grad_ptr = grad_output._data.bitcast[Scalar[grad_output.dtype()]]()
    var result_ptr = result._data.bitcast[Scalar[predictions.dtype()]]()

    for i in range(predictions._numel):
        result_ptr[i] = _hinge_backward_op[predictions.dtype()](grad_ptr[i], pred_ptr[i], targ_ptr[i])

    return result
