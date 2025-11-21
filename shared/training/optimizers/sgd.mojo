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

from shared.core.extensor import ExTensor
from shared.core.arithmetic import subtract, multiply, add
from shared.core.extensor import full_like


fn sgd_step(
    params: ExTensor,
    gradients: ExTensor,
    velocity: ExTensor,
    learning_rate: Float64,
    momentum: Float64 = 0.0,
    weight_decay: Float64 = 0.0
) raises -> (ExTensor, ExTensor):
    """Perform a single SGD optimization step - pure functional.

    Returns new parameters and new velocity. Caller manages all state.

    Args:
        params: Model parameters to update
        gradients: Gradients of loss with respect to params
        velocity: Momentum buffer (use zeros_like(params) if no momentum)
        learning_rate: Step size for parameter updates
        momentum: Momentum factor (default: 0.0, no momentum)
        weight_decay: L2 regularization factor (default: 0.0, no regularization)

    Returns:
        Tuple of (new_params, new_velocity)

    Example (basic SGD without momentum):
        ```mojo
        from shared.core import ExTensor, zeros_like
        from shared.training.optimizers import sgd_step

        var W = xavier_uniform(784, 128, DType.float32)
        var W_vel = zeros_like(W)  # Not used, but required
        var grad_W = ...  # Computed via backpropagation

        # Returns new state
        (W, W_vel) = sgd_step(W, grad_W, W_vel, learning_rate=0.01)
        ```

    Example (SGD with momentum):
        ```mojo
        var W = xavier_uniform(784, 128, DType.float32)
        var W_vel = zeros_like(W)

        # Training loop
        for epoch in range(100):
            var grad_W = ...  # Compute gradients
            # Returns updated params AND updated velocity
            (W, W_vel) = sgd_step(W, grad_W, W_vel, lr=0.01, momentum=0.9)
        ```

    Note:
        This is a pure function - it returns new state rather than mutating.
        Caller must capture both return values and update their variables.
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

    var new_velocity = velocity  # Default: no change to velocity

    # Apply momentum if specified
    if momentum > 0.0:
        if velocity.numel() == 0:
            raise Error("Velocity buffer required when using momentum (use zeros_like(params))")

        # velocity = momentum * velocity + gradients
        var momentum_tensor = full_like(velocity, momentum)
        var scaled_velocity = multiply(momentum_tensor, velocity)
        new_velocity = add(scaled_velocity, effective_gradients)

        # Use velocity for update
        effective_gradients = new_velocity

    # Standard SGD update: params = params - lr * gradients
    var lr_tensor = full_like(params, learning_rate)
    var update = multiply(lr_tensor, effective_gradients)
    var new_params = subtract(params, update)

    # Return both new params and new velocity (pure functional)
    return (new_params, new_velocity)


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


fn sgd_momentum_update_inplace(
    inout param: ExTensor,
    grad: ExTensor,
    inout velocity: ExTensor,
    lr: Float32,
    momentum: Float32
) raises:
    """SGD parameter update with momentum (in-place mutation).

    This function performs in-place updates for efficiency in training loops.
    Mutates both param and velocity tensors directly.

    Formula:
        velocity = momentum * velocity - lr * grad
        param = param + velocity

    Args:
        param: Parameter tensor to update (modified in-place)
        grad: Gradient tensor
        velocity: Momentum velocity tensor (modified in-place)
        lr: Learning rate
        momentum: Momentum coefficient (typically 0.9)

    Example:
        ```mojo
        from shared.core import ExTensor, zeros_like
        from shared.training.optimizers import sgd_momentum_update_inplace

        var W = xavier_uniform([784, 128], DType.float32)
        var W_vel = zeros_like(W)

        # Training loop
        for epoch in range(100):
            var grad_W = ...  # Compute gradients
            sgd_momentum_update_inplace(W, grad_W, W_vel, lr=0.01, momentum=0.9)
        ```

    Note:
        - In-place mutation for efficiency
        - Velocity tensor must be pre-allocated (use zeros_like)
        - Both param and velocity are modified directly
        - This is the AlexNet/ResNet standard momentum formulation
    """
    var numel = param.numel()

    if param.shape() != grad.shape():
        raise Error("Parameter and gradient must have the same shape")

    if param.shape() != velocity.shape():
        raise Error("Parameter and velocity must have the same shape")

    if param.dtype() != DType.float32:
        raise Error("sgd_momentum_update_inplace only supports float32")

    var param_data = param._data.bitcast[Float32]()
    var grad_data = grad._data.bitcast[Float32]()
    var velocity_data = velocity._data.bitcast[Float32]()

    # Update velocity and parameters in-place
    for i in range(numel):
        # velocity = momentum * velocity - lr * grad
        velocity_data[i] = momentum * velocity_data[i] - lr * grad_data[i]
        # param = param + velocity
        param_data[i] += velocity_data[i]
