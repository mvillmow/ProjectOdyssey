"""Core traits for ML components.

Provides zero-cost trait-based abstractions for neural network components,
enabling polymorphism without runtime overhead. All traits compile to
static dispatch (no vtables, no dynamic dispatch).

Benefits:
- Type-safe polymorphism at compile time
- Zero runtime overhead (no virtual function calls)
- Clear interface contracts
- Composable abstractions
- Testability (mock implementations)

Trait Categories:
1. Differentiable - Components with forward/backward passes
2. Parameterized - Components with learnable parameters
3. Serializable - Components that can be saved/loaded
4. Composable - Components that can be chained

Example:
    struct MyLayer(Differentiable, Parameterized):
        fn forward(self, input: ExTensor) -> ExTensor:
            # ... implementation.

        fn backward(self, grad_output: ExTensor) -> ExTensor:
            # ... implementation.

        fn parameters(self) -> List[ExTensor]:
            return [self.weights, self.bias]
    ```
"""

from shared.core import ExTensor


trait Differentiable:
    """Components that support automatic differentiation.

    Implement this trait for all neural network layers and operations
    that participate in backpropagation.

    Required Methods:
        forward: Compute output from input (forward pass)
        backward: Compute input gradient from output gradient (backward pass)

    Contract:
        - forward and backward must be mathematical inverses
        - backward must preserve batch dimension
        - backward must handle None/zero gradients gracefully

    Example:
        ```mojo
        truct ReLULayer(Differentiable):
            var last_input: ExTensor  # Cache for backward pass

            fn forward(mut self, input: ExTensor) -> ExTensor:
                self.last_input = input.copy()
                return relu(input)

            fn backward(self, grad_output: ExTensor) -> ExTensor:
                return relu_backward(grad_output, self.last_input)
        ```
    """

    fn forward(mut self, input: ExTensor) raises -> ExTensor:
        """Compute forward pass.

        Args:
            input: Input tensor (batch_size, ...)

        Returns:
            Output tensor (batch_size, ...)

        Raises:
            Error: If input shape is invalid.

        Note:
            May cache values needed for backward pass.
        """
        ...

    fn backward(self, grad_output: ExTensor) raises -> ExTensor:
        """Compute backward pass (input gradient).

        Args:
            grad_output: Gradient w.r.t. output (∂L/∂output).

        Returns:
            Gradient w.r.t. input (∂L/∂input).

        Raises:
            Error: If backward called before forward.

        Note:
            Uses values cached during forward pass.
        """
        ...


trait Parameterized:
    """Components with learnable parameters.

    Implement this trait for layers that have weights, biases, or other
    trainable parameters that should be updated during optimization.

    Required Methods:
        parameters: Return list of all parameter tensors
        gradients: Return list of all gradient tensors
        zero_grad: Reset all gradients to zero

    Contract:
        - parameters() and gradients() must return same-length lists
        - Parameters and gradients must correspond (same order)
        - zero_grad() must clear all gradient accumulation

    Example:
        ```mojo
        truct LinearLayer(Parameterized):
            var weights: ExTensor
            var bias: ExTensor
            var grad_weights: ExTensor
            var grad_bias: ExTensor

            fn parameters(self) -> List[ExTensor]:
                return [self.weights, self.bias]

            fn gradients(self) -> List[ExTensor]:
                return [self.grad_weights, self.grad_bias]

            fn zero_grad(mut self):
                self.grad_weights.fill(0.0)
                self.grad_bias.fill(0.0)
        ```
    """

    fn parameters(self) raises -> List[ExTensor]:
        """Get all learnable parameters.

        Returns:
            List of parameter tensors.

        Note:
            Order must match gradients() return order.
            Do not include non-trainable parameters (e.g., batch norm running stats).
        """
        ...

    fn gradients(self) raises -> List[ExTensor]:
        """Get gradients for all parameters.

        Returns:
            List of gradient tensors.

        Note:
            Must correspond 1:1 with parameters().
            Gradients are accumulated across mini-batches.
        """
        ...

    fn zero_grad(mut self) raises:
        """Reset all gradients to zero.

        Called at the beginning of each mini-batch to clear
        accumulated gradients from previous iteration.

        Example:
            ```mojo
            odel.zero_grad()  # Clear gradients.
            loss = forward_pass(model, input, target)
            backward_pass(loss)  # Accumulate gradients
            optimizer.step(model.parameters(), model.gradients())
        ```
        """
        ...


trait Serializable:
    """Components that can be saved and loaded.

    Implement this trait for models and layers that need to persist
    state to disk (checkpointing, model saving).

    Required Methods:
        save: Write state to file
        load: Read state from file

    Contract:
        - save() must write all necessary state
        - load() must restore exact state
        - Round-trip (save->load) must be identity
        - File format should be documented

    Example:
        ```mojo
        truct ConvLayer(Serializable):
            var weights: ExTensor
            var bias: ExTensor

            fn save(self, path: String) raises:
                # Save weights and bias to file
                write_tensor(path + "/weights.bin", self.weights)
                write_tensor(path + "/bias.bin", self.bias)

            fn load(mut self, path: String) raises:
                # Load weights and bias from file
                self.weights = read_tensor(path + "/weights.bin")
                self.bias = read_tensor(path + "/bias.bin")
        ```
    """

    fn save(self, path: String) raises:
        """Save component state to file.

        Args:
            path: File path or directory

        Raises:
            Error: If write fails or path is invalid.

        Note:
            Should save all state needed to restore component.
            Include metadata (shapes, dtypes, version).
        """
        ...

    fn load(mut self, path: String) raises:
        """Load component state from file.

        Args:
            path: File path or directory.

        Raises:
            Error: If file doesn't exist, is corrupted, or has version mismatch.

        Note:
            Should validate loaded state (shapes, dtypes).
            Handle version migration if needed.
        """
        ...


trait Composable(Differentiable):
    """Components that can be composed into pipelines.

    Implement this trait for layers and operations that can be
    chained together (e.g., Sequential, Residual connections).

    Required Methods:
        compose: Chain this component with another

    Contract:
        - Composition must preserve differentiability
        - Output shape of self must match input shape of other
        - Associative: (A ∘ B) ∘ C = A ∘ (B ∘ C)

    Example:
        ```mojo
        truct Sequential(Composable):
            var layers: List[Composable]

            fn compose[T: Composable](self, other: T) -> ComposedOp[Self, T]:
                return ComposedOp[Self, T](self, other)

        # Usage:
        var model = Linear(784, 128).compose(ReLU()).compose(Linear(128, 10))
        ```
    """

    fn compose[T: Composable](self, other: T) raises -> ExTensor:
        """Compose this component with another.

        NOTE: Full implementation blocked by Mojo language limitation
        Generic types F and S require Movable constraint which cannot
        be expressed in the current Mojo type system. See issue #2401.

        Args:
            other: The component to compose with this one.

        Returns:
            Composed operation result.

        Raises:
            Error: This method is not yet supported due to Mojo limitation.

        Workaround:
            Instead of using compose(), manually chain operations:

            # Instead of:
            # var combined = layer1.compose(layer2)

            # Use manual composition:
            var intermediate = layer1.forward(input)
            var result = layer2.forward(intermediate)

        See Also:
            - Issue #2401: Trait compose() blocked by Movable constraint
            - https://docs.modular.com/mojo/manual/traits/
        """
        raise Error(
            "compose() not yet supported - use manual composition instead. "
            "See issue #2401 and Composable trait docstring for workaround."
        )


# TODO(#2401): ComposedOp struct blocked by Mojo type system limitation
#
# Issue: ComposedOp requires Movable constraint on generic types F and S,
# but Mojo does not support trait intersection syntax needed to express:
#   struct ComposedOp[F: (Differentiable & Movable), S: (Differentiable & Movable)](...)
#
# Current Status: Deferred until Mojo adds proper trait intersection.
#
# Workaround: Use manual composition in Composable.compose() docstring.
# Commented code below for future reference:
#
# struct ComposedOp[F: Differentiable, S: Differentiable](Differentiable, Composable):
#     """Composition of two differentiable operations."""
#     var first: F
#     var second: S
#
#     fn forward(mut self, input: ExTensor) raises -> ExTensor:
#         var intermediate = self.first.forward(input)
#         return self.second.forward(intermediate)
#
#     fn backward(self, grad_output: ExTensor) raises -> ExTensor:
#         var grad_intermediate = self.second.backward(grad_output)
#         return self.first.backward(grad_intermediate)
#
# See: https://docs.modular.com/mojo/manual/traits/
# See: GitHub issue #2401 for limitation details


trait Trainable:
    """Components that support training mode.

    Implement this trait for components that behave differently during
    training vs. inference (e.g., Dropout, BatchNorm).

    Required Methods:
        train: Set to training mode
        eval: Set to evaluation mode
        is_training: Check current mode

    Example:
        ```mojo
        truct Dropout(Trainable):
            var training: Bool
            var p: Float64

            fn train(mut self):
                self.training = True

            fn eval(mut self):
                self.training = False

            fn is_training(self) -> Bool:
                return self.training

            fn forward(self, input: ExTensor) -> ExTensor:
                if self.training:
                    # Apply dropout
                else:
                    # No dropout during inference
        ```
    """

    fn train(mut self):
        """Set component to training mode.

        Enables training-specific behavior (dropout, batch norm updates, etc.).
        """
        ...

    fn eval(mut self):
        """Set component to evaluation mode.

        Disables training-specific behavior for inference.
        """
        ...

    fn is_training(self) -> Bool:
        """Check if component is in training mode.

        Returns:
            True if training, False if evaluating.
        """
        ...


# ============================================================================
# Training Loop Traits (see #2392, #2393, #2397 for implementation)
# ============================================================================


trait Model:
    """Neural network model interface for generic TrainingLoop.

    Defines the contract for models that can be trained using TrainingLoop.
    All neural network models should implement this trait.

    Required Methods:
        forward: Execute forward pass
        parameters: Return trainable parameters
        zero_grad: Reset parameter gradients

    Example:
        ```mojo
        truct SimpleMLP(Model):
            fn forward(mut self, input: ExTensor) raises -> ExTensor:
                # ... layer computations ...
                return output^

            fn parameters(self) raises -> List[ExTensor]:
                return [self.layer1_weights, self.layer1_bias, ...]^

            fn zero_grad(mut self) raises:
                # Reset all gradient accumulators
        ```
    """

    fn forward(mut self, input: ExTensor) raises -> ExTensor:
        """Execute forward pass through the model.

        Args:
            input: Input tensor (batch_size, input_dim).

        Returns:
            Output tensor (batch_size, output_dim).

        Raises:
            Error: If input shape is invalid.
        """
        ...

    fn parameters(self) raises -> List[ExTensor]:
        """Return list of all trainable parameters.

        Returns:
            List of parameter tensors.

        Note:
            Used by optimizers to update weights.
        """
        ...

    fn zero_grad(mut self) raises:
        """Reset all parameter gradients to zero.

        Note:
            Should be called before each backward pass.
        """
        ...


trait Loss:
    """Loss function interface for generic TrainingLoop.

    Defines the contract for loss functions that measure prediction error.

    Required Methods:
        compute: Calculate loss between predictions and targets

    Example:
        ```mojo
        truct MSELoss(Loss):
            fn compute(self, pred: ExTensor, target: ExTensor) raises -> ExTensor:
                var diff = subtract(pred, target)
                return mean(multiply(diff, diff))
        ```
    """

    fn compute(self, pred: ExTensor, target: ExTensor) raises -> ExTensor:
        """Compute loss between predictions and targets.

        Args:
            pred: Model predictions (batch_size, ...).
            target: Ground truth targets (batch_size, ...).

        Returns:
            Scalar loss value.

        Raises:
            Error: If shapes are incompatible.
        """
        ...


trait Optimizer:
    """Optimizer interface for generic TrainingLoop.

    Defines the contract for optimization algorithms that update parameters.

    Required Methods:
        step: Update parameters based on gradients
        zero_grad: Reset optimizer state

    Example:
        ```mojo
        truct SGD(Optimizer):
            var learning_rate: Float32

            fn step(mut self, params: List[ExTensor]) raises:
                for param in params:
                    param -= self.learning_rate * param.grad

            fn zero_grad(mut self) raises:
                # Clear any optimizer-specific state
        ```
    """

    fn step(mut self, params: List[ExTensor]) raises:
        """Update parameters using computed gradients.

        Args:
            params: List of parameter tensors to update.

        Note:
            Assumes gradients are already computed.
        """
        ...

    fn zero_grad(mut self) raises:
        """Reset optimizer state.

        Note:
            May be called before parameter zero_grad().
        """
        ...
