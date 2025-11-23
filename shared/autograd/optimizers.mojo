"""Optimizers for gradient-based parameter updates.

Implements standard optimization algorithms used in neural network training.
Each optimizer updates model parameters based on their gradients.

Implemented optimizers:
- SGD (Stochastic Gradient Descent): Basic gradient descent with optional momentum
- Adam: Adaptive learning rate optimizer (TODO)
- RMSprop: Root Mean Square Propagation (TODO)

Usage Pattern:
    # Create optimizer
    var optimizer = SGD(learning_rate=0.01)

    # Training loop
    for epoch in range(num_epochs):
        # Forward pass
        var predictions = model(inputs)
        var loss = loss_fn(predictions, targets)

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step(model.parameters())

        # Reset gradients
        optimizer.zero_grad(model.parameters())

Design Note:
    This module operates on Variables (from autograd), not raw ExTensors.
    The optimizer updates Variable.data based on Variable.grad.
"""

from ..core.extensor import ExTensor
from ..core.arithmetic import subtract, multiply
from .variable import Variable


struct SGD:
    """Stochastic Gradient Descent optimizer.

    Implements basic gradient descent with optional momentum:
        v_t = momentum * v_{t-1} + gradient
        parameter = parameter - learning_rate * v_t

    Without momentum (momentum=0):
        parameter = parameter - learning_rate * gradient

    Attributes:
        learning_rate: Step size for parameter updates
        momentum: Momentum factor for accelerated gradient descent (default: 0.0)
        velocity: Momentum accumulation for each parameter (maintained internally)

    Examples:
        # Basic SGD
        var optimizer = SGD(learning_rate=0.01)

        # SGD with momentum
        var optimizer = SGD(learning_rate=0.01, momentum=0.9)

        # Training step
        optimizer.step(parameters)
        optimizer.zero_grad(parameters)
    """

    var learning_rate: Float64
    var momentum: Float64
    # TODO: Add velocity storage for momentum
    # var velocities: List[ExTensor]

    fn __init__(out self, learning_rate: Float64, momentum: Float64 = 0.0):
        """Initialize SGD optimizer.

        Args:
            learning_rate: Step size for gradient descent (α in literature)
            momentum: Momentum coefficient (β in literature), range [0, 1]
                     0 = no momentum (standard SGD)
                     0.9 = typical momentum value
                     Higher values give more weight to past gradients

        Examples:
            var opt = SGD(learning_rate=0.01)
            var opt_momentum = SGD(learning_rate=0.01, momentum=0.9)
        """
        self.learning_rate = learning_rate
        self.momentum = momentum

    fn step(self, inout parameters: List[Variable]) raises:
        """Update parameters using their gradients.

        Performs one step of gradient descent:
            parameter = parameter - learning_rate * gradient

        Args:
            parameters: List of Variables to update (model parameters)

        Note:
            This assumes gradients have already been computed via backward().
            Parameters without gradients (grad=None) are skipped.

        Raises:
            Error if any parameter has incompatible gradient shape

        Examples:
            # After backward pass
            loss.backward()

            # Update all parameters
            optimizer.step(model.parameters())
        """
        for i in range(len(parameters)):
            var param = parameters[i]

            # Skip parameters that don't require gradients
            if not param.requires_grad:
                continue

            # Skip parameters without computed gradients
            if param.grad is None:
                continue

            # Get gradient
            var grad = param.grad.value()

            # Verify gradient shape matches parameter shape
            if grad.shape != param.data.shape:
                raise Error("Gradient shape mismatch with parameter shape")

            # Create learning rate tensor
            var lr_tensor = ExTensor(grad.shape, grad.dtype())
            for j in range(lr_tensor.numel()):
                lr_tensor._set_float64(j, self.learning_rate)

            # Compute update: lr * gradient
            var update = multiply(lr_tensor, grad)

            # Apply update: parameter = parameter - lr * gradient
            param.data = subtract(param.data, update)

    fn zero_grad(self, inout parameters: List[Variable]):
        """Reset all parameter gradients to None.

        Should be called after each optimizer step to clear gradients before
        the next backward pass.

        Args:
            parameters: List of Variables whose gradients to reset

        Examples:
            # Clear gradients before next iteration
            optimizer.zero_grad(model.parameters())

            # Or call on individual parameters
            for param in model.parameters():
                param.zero_grad()
        """
        for i in range(len(parameters)):
            parameters[i].zero_grad()


# Placeholder for future optimizers
# TODO: Implement Adam optimizer
# TODO: Implement RMSprop optimizer
# TODO: Implement AdaGrad optimizer
