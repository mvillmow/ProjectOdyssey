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
from .tape import GradientTape
from .functional import multiply_scalar, subtract_scalar


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

    fn step(self, mut parameters: List[Variable], mut tape: GradientTape) raises:
        """Update parameters using their gradients from the tape.

        Performs one step of gradient descent:
            parameter = parameter - learning_rate * gradient

        Args:
            parameters: List of Variables to update (model parameters)
            tape: The gradient tape containing computed gradients

        Note:
            This assumes gradients have already been computed via backward().
            Parameters without gradients in the tape are skipped.

        Raises:
            Error if any parameter has incompatible gradient shape

        Examples:
            # After backward pass
            loss.backward(tape)

            # Update all parameters
            optimizer.step(model.parameters(), tape)
        """
        for i in range(len(parameters)):
            # Skip parameters that don't require gradients
            if not parameters[i].requires_grad:
                continue

            # Skip if no gradient has been computed
            var param_id = parameters[i].id
            if not tape.registry.has_gradient(param_id):
                continue

            # Get the gradient for this parameter
            var grad = tape.registry.get_grad(param_id)

            # Update: param.data = param.data - learning_rate * grad
            # scaled_grad = learning_rate * grad
            var scaled_grad = multiply_scalar(grad, self.learning_rate)
            # new_data = param.data - scaled_grad
            var new_data = subtract(parameters[i].data, scaled_grad)

            # Update the parameter's data
            parameters[i].data = new_data^

    fn zero_grad(self, mut tape: GradientTape):
        """Reset all gradients in the tape.

        Should be called after each optimizer step to clear gradients before
        the next backward pass.

        Args:
            tape: The gradient tape to clear

        Examples:
            # Clear gradients before next iteration
            optimizer.zero_grad(tape)
        """
        # Clear the gradient registry
        tape.registry.clear()


# Placeholder for future optimizers
# TODO: Implement Adam optimizer
# TODO: Implement RMSprop optimizer
# TODO: Implement AdaGrad optimizer
