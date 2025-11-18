"""Stochastic Gradient Descent (SGD) optimizer.

This module provides the SGD optimizer for updating model parameters
during training using gradient descent.

Standard SGD update rule:
    params = params - learning_rate * gradients

With momentum:
    velocity = momentum * velocity + gradients
    params = params - learning_rate * velocity

With weight decay (L2 regularization):
    params = params - learning_rate * (gradients + weight_decay * params)
"""

from extensor import ExTensor, subtract, multiply, add, full_like


fn sgd_step(
    params: ExTensor,
    gradients: ExTensor,
    learning_rate: Float64,
    momentum: Float64 = 0.0,
    weight_decay: Float64 = 0.0,
    velocity: ExTensor = ExTensor()
) raises -> ExTensor:
    """Perform a single SGD optimization step.

    Updates parameters using stochastic gradient descent with optional
    momentum and weight decay.

    Args:
        params: Model parameters to update
        gradients: Gradients of loss with respect to params
        learning_rate: Step size for parameter updates
        momentum: Momentum factor (default: 0.0, no momentum)
        weight_decay: L2 regularization factor (default: 0.0, no regularization)
        velocity: Momentum buffer (required if momentum > 0)

    Returns:
        Updated parameters

    Example (basic SGD):
        var W = xavier_uniform(784, 128, shape, DType.float32)
        var grad_W = ... # Computed via backpropagation
        W = sgd_step(W, grad_W, learning_rate=0.01)

    Example (SGD with momentum):
        # Initialize velocity buffer
        var velocity = zeros_like(W)

        # Training loop
        for epoch in range(100):
            var grad_W = ... # Compute gradients
            W = sgd_step(W, grad_W, learning_rate=0.01, momentum=0.9, velocity=velocity)

    Example (SGD with weight decay):
        var W = ... # Parameters
        var grad_W = ... # Gradients
        W = sgd_step(W, grad_W, learning_rate=0.01, weight_decay=0.0001)

    Note:
        This is a simplified implementation. For production use with momentum,
        consider creating a struct to manage optimizer state.
    """
    if params.shape() != gradients.shape():
        raise Error("Parameters and gradients must have the same shape")

    if params.dtype() != gradients.dtype():
        raise Error("Parameters and gradients must have the same dtype")

    var effective_gradients = gradients

    # Apply weight decay (L2 regularization) if specified
    if weight_decay > 0.0:
        # grad = grad + weight_decay * params
        var wd_tensor = full_like(params, weight_decay)
        var decay_term = multiply(wd_tensor, params)
        effective_gradients = add(gradients, decay_term)

    # Apply momentum if specified
    if momentum > 0.0:
        if velocity.numel() == 0:
            raise Error("Velocity buffer required when using momentum")

        # velocity = momentum * velocity + gradients
        var momentum_tensor = full_like(velocity, momentum)
        var scaled_velocity = multiply(momentum_tensor, velocity)
        var new_velocity = add(scaled_velocity, effective_gradients)

        # Use velocity for update
        effective_gradients = new_velocity

    # Standard SGD update: params = params - lr * gradients
    var lr_tensor = full_like(params, learning_rate)
    var update = multiply(lr_tensor, effective_gradients)

    return subtract(params, update)


fn sgd_step_simple(
    params: ExTensor,
    gradients: ExTensor,
    learning_rate: Float64
) raises -> ExTensor:
    """Simplified SGD step without momentum or weight decay.

    This is a convenience function for basic gradient descent.

    Formula:
        params = params - learning_rate * gradients

    Args:
        params: Model parameters to update
        gradients: Gradients of loss with respect to params
        learning_rate: Step size for parameter updates

    Returns:
        Updated parameters

    Example:
        var W1 = xavier_uniform(784, 128, shape, DType.float32)
        var grad_W1 = ... # Computed gradients
        W1 = sgd_step_simple(W1, grad_W1, 0.01)
    """
    if params.shape() != gradients.shape():
        raise Error("Parameters and gradients must have the same shape")

    if params.dtype() != gradients.dtype():
        raise Error("Parameters and gradients must have the same dtype")

    # params = params - lr * gradients
    var lr_tensor = full_like(params, learning_rate)
    var update = multiply(lr_tensor, gradients)

    return subtract(params, update)
