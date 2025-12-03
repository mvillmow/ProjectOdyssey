"""Optimizers for gradient-based parameter updates.

Implements standard optimization algorithms used in neural network training.
Each optimizer updates model parameters based on their gradients.

Implemented optimizers:
- SGD (Stochastic Gradient Descent): Basic gradient descent with optional momentum
- Adam: Adaptive learning rate optimizer (Adaptive Moment Estimation)
- AdaGrad: Adaptive Gradient optimizer with per-parameter learning rates
- RMSprop: Root Mean Square Propagation with adaptive learning rates

Usage Pattern:
    # Create optimizer
    var optimizer = SGD(learning_rate=0.01)
    # or
    var optimizer = Adam(learning_rate=0.001)

    # Training loop
    for epoch in range(num_epochs):
        # Forward pass
        var predictions = model(inputs)
        var loss = loss_fn(predictions, targets)

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step(model.parameters(), tape)

        # Reset gradients
        optimizer.zero_grad(tape)

Design Note:
    This module operates on Variables (from autograd), not raw ExTensors.
    The optimizer updates Variable.data based on Variable.grad.
"""

from ..core.extensor import ExTensor
from ..core.arithmetic import subtract, multiply, divide, add
from ..core.elementwise import sqrt
from .variable import Variable
from .tape import GradientTape
from .functional import multiply_scalar, subtract_scalar, add_scalar


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

    fn step(
        self, mut parameters: List[Variable], mut tape: GradientTape
    ) raises:
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


struct Adam:
    """Adam (Adaptive Moment Estimation) optimizer.

    Combines momentum and RMSprop to achieve adaptive learning rates for each parameter:
        m_t = β₁ * m_{t-1} + (1 - β₁) * g_t           # First moment (momentum)
        v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²          # Second moment (variance)
        m̂_t = m_t / (1 - β₁^t)                        # Bias correction
        v̂_t = v_t / (1 - β₂^t)                        # Bias correction
        θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)         # Update

    This is one of the most popular optimizers for deep learning due to its
    adaptive learning rates and momentum properties.

    Attributes:
        learning_rate: Step size for parameter updates (default: 0.001)
        beta1: Decay rate for first moment moving average (default: 0.9)
        beta2: Decay rate for second moment moving average (default: 0.999)
        epsilon: Small constant for numerical stability (default: 1e-8)
        weight_decay: L2 regularization coefficient (default: 0.0)
        t: Current step counter (incremented on each step)
        m_buffers: First moment buffers per parameter ID
        v_buffers: Second moment buffers per parameter ID

    Examples:
        # Basic Adam optimizer
        var optimizer = Adam(learning_rate=0.001)

        # Custom parameters
        var optimizer = Adam(
            learning_rate=0.002,
            beta1=0.95,
            beta2=0.99,
            epsilon=1e-6,
            weight_decay=0.01
        )

        # Training step
        loss.backward(tape)
        optimizer.step(parameters, tape)
        optimizer.zero_grad(tape)
    """

    var learning_rate: Float64
    var beta1: Float64
    var beta2: Float64
    var epsilon: Float64
    var weight_decay: Float64
    var t: Int
    var m_buffers: List[ExTensor]
    var v_buffers: List[ExTensor]
    var has_buffer: List[Bool]

    fn __init__(
        out self,
        learning_rate: Float64 = 0.001,
        beta1: Float64 = 0.9,
        beta2: Float64 = 0.999,
        epsilon: Float64 = 1e-8,
        weight_decay: Float64 = 0.0,
    ):
        """Initialize Adam optimizer.

        Args:
            learning_rate: Step size for gradient descent (α in literature)
                          Typical range: [1e-4, 1e-2]
            beta1: Decay rate for first moment (momentum) coefficient
                  Typical value: 0.9
            beta2: Decay rate for second moment (variance) coefficient
                  Typical value: 0.999
            epsilon: Small constant for numerical stability when dividing by sqrt(v_t)
                    Typical value: 1e-8
            weight_decay: L2 regularization coefficient
                         Typical value: 0.0 (no regularization) or 1e-5 to 1e-3

        Examples:
            var opt = Adam()  # Use defaults
            var opt = Adam(learning_rate=0.002)
            var opt = Adam(learning_rate=0.001, weight_decay=1e-5)
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.t = 0
        self.m_buffers = List[ExTensor]()
        self.v_buffers = List[ExTensor]()
        self.has_buffer = List[Bool]()

    fn step(
        mut self, mut parameters: List[Variable], mut tape: GradientTape
    ) raises:
        """Update parameters using Adam algorithm.

        Performs one step of Adam optimization on all parameters with gradients.
        Internally maintains momentum (m) and variance (v) buffers per parameter.

        Algorithm:
            1. Increment step counter: t = t + 1
            2. For each parameter with gradient:
                - Get gradient g_t
                - Update first moment: m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
                - Update second moment: v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
                - Compute bias corrections: m̂_t = m_t / (1 - β₁^t), v̂_t = v_t / (1 - β₂^t)
                - Update parameter: θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)

        Args:
            parameters: List of Variables to update (model parameters)
            tape: The gradient tape containing computed gradients

        Note:
            This assumes gradients have already been computed via backward().
            Parameters without gradients in the tape are skipped.
            The optimizer automatically initializes momentum buffers on first encounter.

        Raises:
            Error if any parameter has incompatible gradient shape

        Examples:
            # After backward pass
            loss.backward(tape)

            # Update all parameters (multiple calls increment step counter)
            optimizer.step(model.parameters(), tape)
        """
        # Increment step counter
        self.t += 1
        var t_float = Float64(self.t)

        # Compute bias correction terms
        var bias_correction1 = 1.0 - pow(self.beta1, t_float)
        var bias_correction2 = 1.0 - pow(self.beta2, t_float)

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

            # Ensure buffer lists are large enough for this parameter ID
            while len(self.m_buffers) <= param_id:
                var placeholder_shape = List[Int]()
                placeholder_shape.append(1)
                self.m_buffers.append(
                    ExTensor(placeholder_shape, DType.float32)
                )
                self.v_buffers.append(
                    ExTensor(placeholder_shape, DType.float32)
                )
                self.has_buffer.append(False)

            # Initialize moment buffers if they don't exist
            if not self.has_buffer[param_id]:
                # m_t = zeros like grad
                var m = ExTensor(grad.shape(), grad.dtype())
                for j in range(m.numel()):
                    m._set_float64(j, 0.0)
                self.m_buffers[param_id] = m^

                # v_t = zeros like grad
                var v = ExTensor(grad.shape(), grad.dtype())
                for j in range(v.numel()):
                    v._set_float64(j, 0.0)
                self.v_buffers[param_id] = v^

                self.has_buffer[param_id] = True

            # Get current moment buffers
            var m = self.m_buffers[param_id]
            var v = self.v_buffers[param_id]

            # Update biased first moment estimate: m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
            var m_new = ExTensor(grad.shape(), grad.dtype())
            for j in range(grad.numel()):
                var m_prev = m._get_float64(j)
                var grad_val = grad._get_float64(j)
                var m_val = self.beta1 * m_prev + (1.0 - self.beta1) * grad_val
                m_new._set_float64(j, m_val)
            self.m_buffers[param_id] = m_new^

            # Update biased second moment estimate: v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
            var v_new = ExTensor(grad.shape(), grad.dtype())
            for j in range(grad.numel()):
                var v_prev = v._get_float64(j)
                var grad_val = grad._get_float64(j)
                var v_val = (
                    self.beta2 * v_prev
                    + (1.0 - self.beta2) * grad_val * grad_val
                )
                v_new._set_float64(j, v_val)
            self.v_buffers[param_id] = v_new^

            # Get updated moment buffers for bias correction
            var m_updated = self.m_buffers[param_id]
            var v_updated = self.v_buffers[param_id]

            # Compute bias-corrected moment estimates
            var m_hat = multiply_scalar(m_updated, 1.0 / bias_correction1)
            var v_hat = multiply_scalar(v_updated, 1.0 / bias_correction2)

            # Compute update: α * m̂_t / (√v̂_t + ε)
            # First: √v̂_t
            var v_hat_sqrt = ExTensor(v_hat.shape(), v_hat.dtype())
            for j in range(v_hat.numel()):
                var v_val = v_hat._get_float64(j)
                var sqrt_v = v_val ** 0.5
                v_hat_sqrt._set_float64(j, sqrt_v)

            # Second: √v̂_t + ε
            var denominator = add_scalar(v_hat_sqrt, self.epsilon)

            # Third: m̂_t / (√v̂_t + ε) - element-wise division
            var adaptive_grad = divide(m_hat, denominator)

            # Fourth: α * (m̂_t / (√v̂_t + ε))
            var scaled_grad = multiply_scalar(adaptive_grad, self.learning_rate)

            # Apply weight decay if specified
            var param_update = scaled_grad
            if self.weight_decay != 0.0:
                var weight_decay_update = multiply_scalar(
                    parameters[i].data, self.learning_rate * self.weight_decay
                )
                param_update = ExTensor(
                    scaled_grad.shape(), scaled_grad.dtype()
                )
                for j in range(scaled_grad.numel()):
                    var grad_val = scaled_grad._get_float64(j)
                    var decay_val = weight_decay_update._get_float64(j)
                    param_update._set_float64(j, grad_val + decay_val)

            # Update parameter: θ_t = θ_{t-1} - param_update
            var new_data = subtract(parameters[i].data, param_update)
            parameters[i].data = new_data^

    fn zero_grad(self, mut tape: GradientTape):
        """Reset all gradients in the tape.

        Should be called after each optimizer step to clear gradients before
        the next backward pass.

        Note:
            This clears gradients but preserves the internal momentum and variance
            buffers, which are maintained across steps.

        Args:
            tape: The gradient tape to clear

        Examples:
            # Clear gradients before next iteration
            optimizer.zero_grad(tape)
        """
        # Clear the gradient registry
        tape.registry.clear()


struct AdaGrad:
    """AdaGrad (Adaptive Gradient) optimizer.

    Implements adaptive learning rate optimization based on accumulated squared
    gradients:
        G_t = G_{t-1} + g_t²              # Accumulated squared gradients
        θ_t = θ_{t-1} - α * g_t / (√G_t + ε)  # Parameter update

    Attributes:
        learning_rate: Initial step size for parameter updates
        epsilon: Small constant for numerical stability (default: 1e-10)
        weight_decay: L2 regularization coefficient (default: 0.0)
        G_buffers: Accumulated squared gradients for each parameter

    Examples:
        # Basic AdaGrad
        var optimizer = AdaGrad(learning_rate=0.01)

        # AdaGrad with weight decay
        var optimizer = AdaGrad(learning_rate=0.01, weight_decay=1e-4)

        # Training step
        optimizer.step(parameters, tape)
        optimizer.zero_grad(tape)
    """

    var learning_rate: Float64
    var epsilon: Float64
    var weight_decay: Float64
    var G_buffers: Dict[Int, ExTensor]

    fn __init__(
        out self,
        learning_rate: Float64,
        epsilon: Float64 = 1e-10,
        weight_decay: Float64 = 0.0,
    ):
        """Initialize AdaGrad optimizer.

        Args:
            learning_rate: Step size for parameter updates (α in literature)
                          Typical values: 0.01, 0.001
            epsilon: Small constant added to accumulated gradient for numerical
                    stability (prevents division by zero)
                    Default: 1e-10
            weight_decay: L2 regularization coefficient
                         0.0 = no weight decay
                         1e-4 = typical value for regularization
                         Default: 0.0

        Examples:
            var opt = AdaGrad(learning_rate=0.01)
            var opt_reg = AdaGrad(learning_rate=0.01, weight_decay=1e-4)
        """
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.G_buffers = Dict[Int, ExTensor]()

    fn step(
        mut self, mut parameters: List[Variable], mut tape: GradientTape
    ) raises:
        """Update parameters using AdaGrad adaptive learning rates.

        Performs one step of AdaGrad optimization:
            G_t = G_{t-1} + g_t²
            θ_t = θ_{t-1} - α * g_t / (√G_t + ε)

        Args:
            parameters: List of Variables to update (model parameters)
            tape: The gradient tape containing computed gradients

        Note:
            This method accumulates squared gradients in G_buffers.
            Parameters without gradients in the tape are skipped.
            The G_buffers are not reset by zero_grad() - they persist
            across optimization steps.

        Raises:
            Error if any parameter has incompatible gradient shape

        Examples:
            # After backward pass
            loss.backward(tape)

            # Update all parameters with adaptive learning rates
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

            # Initialize or retrieve accumulated gradient buffer
            var G = ExTensor(
                parameters[i].data.shape(), parameters[i].data.dtype()
            )
            if i in self.G_buffers:
                G = self.G_buffers[i]

            # Accumulate squared gradient: G_t = G_{t-1} + g_t²
            # grad_squared = grad * grad
            var grad_squared = multiply(grad, grad)
            # G = G + grad_squared
            G = add(G, grad_squared)

            # Store updated G buffer
            self.G_buffers[i] = G

            # Compute adaptive learning rate: sqrt(G_t) + epsilon
            # sqrt_g = sqrt(G)
            var sqrt_g = sqrt(G)
            # denominator = sqrt(G) + epsilon
            var denominator = add_scalar(sqrt_g, self.epsilon)

            # Compute scaled gradient: g_t / (√G_t + ε)
            # adaptive_grad = grad / denominator
            var adaptive_grad = divide(grad, denominator)

            # Apply learning rate: lr * (g_t / (√G_t + ε))
            var scaled_grad = multiply_scalar(adaptive_grad, self.learning_rate)

            # Apply weight decay if specified
            if self.weight_decay > 0.0:
                var decay_term = multiply_scalar(
                    parameters[i].data, self.weight_decay
                )
                scaled_grad = add(scaled_grad, decay_term)

            # Update parameter: θ_t = θ_{t-1} - scaled_grad
            var new_data = subtract(parameters[i].data, scaled_grad)

            # Update the parameter's data
            parameters[i].data = new_data^

    fn zero_grad(self, mut tape: GradientTape):
        """Reset all gradients in the tape.

        Should be called after each optimizer step to clear gradients before
        the next backward pass.

        Note:
            This does NOT clear the G_buffers (accumulated squared gradients).
            AdaGrad maintains these accumulators across optimization steps.

        Args:
            tape: The gradient tape to clear

        Examples:
            # Clear gradients before next iteration
            optimizer.zero_grad(tape)
        """
        # Clear the gradient registry
        tape.registry.clear()

    fn reset_accumulators(mut self):
        """Reset accumulated squared gradient buffers.

        Call this to clear the accumulated gradients if needed (e.g., when
        starting a new training phase or to reduce numerical drift).

        Examples:
            # Clear accumulators before new training phase
            optimizer.reset_accumulators()
        """
        self.G_buffers.clear()


struct RMSprop:
    """Root Mean Square Propagation (RMSprop) optimizer.

    Adapts learning rate per parameter based on running average of squared gradients:
        v_t = ρ * v_{t-1} + (1 - ρ) * g_t²        # Running average of squared gradients
        θ_t = θ_{t-1} - α * g_t / (√v_t + ε)      # Update with adaptive learning rate

    Optionally applies momentum to the parameter updates:
        m_t = β * m_{t-1} + update                 # Momentum accumulation
        θ_t = θ_{t-1} - m_t                        # Apply momentum update

    Attributes:
        learning_rate: Step size for parameter updates (α, default: 0.01)
        alpha: Smoothing constant for running average (ρ, default: 0.99)
        epsilon: Small constant for numerical stability (default: 1e-8)
        weight_decay: L2 regularization coefficient (default: 0.0)
        momentum: Momentum factor for accelerated updates (default: 0.0)
        v_buffers: Running average of squared gradients per parameter (internal)
        m_buffers: Momentum accumulation per parameter (internal, if momentum > 0)
        has_buffer: Tracks initialized buffers per parameter ID

    Examples:
        # Basic RMSprop
        var optimizer = RMSprop(learning_rate=0.01)

        # RMSprop with momentum
        var optimizer = RMSprop(learning_rate=0.01, momentum=0.9)

        # RMSprop with weight decay
        var optimizer = RMSprop(learning_rate=0.01, weight_decay=1e-4)

        # Training step
        optimizer.step(parameters, tape)
        optimizer.zero_grad(tape)
    """

    var learning_rate: Float64
    var alpha: Float64
    var epsilon: Float64
    var weight_decay: Float64
    var momentum: Float64
    var v_buffers: List[ExTensor]
    var m_buffers: List[ExTensor]
    var has_buffer: List[Bool]

    fn __init__(
        out self,
        learning_rate: Float64 = 0.01,
        alpha: Float64 = 0.99,
        epsilon: Float64 = 1e-8,
        weight_decay: Float64 = 0.0,
        momentum: Float64 = 0.0,
    ):
        """Initialize RMSprop optimizer.

        Args:
            learning_rate: Step size for gradient descent.
            alpha: Smoothing constant for running average.
            epsilon: Small constant for numerical stability.
            weight_decay: L2 regularization coefficient.
            momentum: Momentum coefficient.

        Examples:
            var opt = RMSprop()
            var opt = RMSprop(learning_rate=0.001)
            var opt = RMSprop(0.01, alpha=0.999)
            var opt = RMSprop(0.01, momentum=0.9)
            var opt = RMSprop(0.01, weight_decay=1e-4)
        """
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.v_buffers = List[ExTensor]()
        self.m_buffers = List[ExTensor]()
        self.has_buffer = List[Bool]()

    fn step(
        mut self, mut parameters: List[Variable], mut tape: GradientTape
    ) raises:
        """Update parameters using RMSprop algorithm.

        Args:
            parameters: List of Variables to update.
            tape: The gradient tape containing computed gradients.

        Raises:
            Error if any parameter has incompatible gradient shape.
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

            # Ensure buffer lists are large enough for this parameter ID
            while len(self.v_buffers) <= param_id:
                var placeholder_shape = List[Int]()
                placeholder_shape.append(1)
                self.v_buffers.append(
                    ExTensor(placeholder_shape, DType.float32)
                )
                self.m_buffers.append(
                    ExTensor(placeholder_shape, DType.float32)
                )
                self.has_buffer.append(False)

            # Initialize buffers if they don't exist for this parameter
            if not self.has_buffer[param_id]:
                # v_t = zeros like grad
                var v = ExTensor(grad.shape(), grad.dtype())
                for j in range(v.numel()):
                    v._set_float64(j, 0.0)
                self.v_buffers[param_id] = v^

                # m_t = zeros like grad (for momentum)
                var m = ExTensor(grad.shape(), grad.dtype())
                for j in range(m.numel()):
                    m._set_float64(j, 0.0)
                self.m_buffers[param_id] = m^

                self.has_buffer[param_id] = True

            # Apply weight decay (L2 regularization) to gradient
            var working_grad = grad
            if self.weight_decay > 0.0:
                var regularization = multiply_scalar(
                    parameters[i].data, self.weight_decay
                )
                working_grad = add(grad, regularization)

            # Update running average of squared gradients
            var v = self.v_buffers[param_id]
            var v_new = ExTensor(grad.shape(), grad.dtype())
            for j in range(grad.numel()):
                var v_prev = v._get_float64(j)
                var grad_val = working_grad._get_float64(j)
                var v_val = self.alpha * v_prev + (1.0 - self.alpha) * grad_val * grad_val
                v_new._set_float64(j, v_val)
            self.v_buffers[param_id] = v_new^

            # Get updated v buffer
            var v_updated = self.v_buffers[param_id]

            # Compute adaptive learning rate
            var adaptive_grad = ExTensor(working_grad.shape(), working_grad.dtype())
            for j in range(working_grad.numel()):
                var g = working_grad._get_float64(j)
                var v_val = v_updated._get_float64(j)
                var denom = sqrt(v_val) + self.epsilon
                adaptive_grad._set_float64(j, g / denom)

            # Apply learning rate
            var update = multiply_scalar(adaptive_grad, self.learning_rate)

            # Apply momentum if specified
            if self.momentum > 0.0:
                var m = self.m_buffers[param_id]
                var m_new = ExTensor(update.shape(), update.dtype())
                for j in range(update.numel()):
                    var m_prev = m._get_float64(j)
                    var upd_val = update._get_float64(j)
                    m_new._set_float64(j, self.momentum * m_prev + upd_val)
                self.m_buffers[param_id] = m_new^
                update = self.m_buffers[param_id]

            # Update parameter
            var new_data = subtract(parameters[i].data, update)
            parameters[i].data = new_data^

    fn zero_grad(self, mut tape: GradientTape):
        """Reset all gradients in the tape.

        Args:
            tape: The gradient tape to clear.
        """
        tape.registry.clear()
