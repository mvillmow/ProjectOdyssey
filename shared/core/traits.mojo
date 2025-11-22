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
            # ... implementation

        fn backward(self, grad_output: ExTensor) -> ExTensor:
            # ... implementation

        fn parameters(self) -> List[ExTensor]:
            return [self.weights, self.bias]
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
        struct ReLULayer(Differentiable):
            var last_input: ExTensor  # Cache for backward pass

            fn forward(inout self, input: ExTensor) -> ExTensor:
                self.last_input = input.copy()
                return relu(input)

            fn backward(self, grad_output: ExTensor) -> ExTensor:
                return relu_backward(grad_output, self.last_input)
    """

    fn forward(inout self, input: ExTensor) raises -> ExTensor:
        """Compute forward pass.

        Args:
            input: Input tensor (batch_size, ...)

        Returns:
            Output tensor (batch_size, ...)

        Raises:
            Error: If input shape is invalid

        Note:
            May cache values needed for backward pass.
        """
        ...

    fn backward(self, grad_output: ExTensor) raises -> ExTensor:
        """Compute backward pass (input gradient).

        Args:
            grad_output: Gradient w.r.t. output (∂L/∂output)

        Returns:
            Gradient w.r.t. input (∂L/∂input)

        Raises:
            Error: If backward called before forward

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
        struct LinearLayer(Parameterized):
            var weights: ExTensor
            var bias: ExTensor
            var grad_weights: ExTensor
            var grad_bias: ExTensor

            fn parameters(self) -> List[ExTensor]:
                return [self.weights, self.bias]

            fn gradients(self) -> List[ExTensor]:
                return [self.grad_weights, self.grad_bias]

            fn zero_grad(inout self):
                self.grad_weights.fill(0.0)
                self.grad_bias.fill(0.0)
    """

    fn parameters(self) raises -> List[ExTensor]:
        """Get all learnable parameters.

        Returns:
            List of parameter tensors

        Note:
            Order must match gradients() return order.
            Do not include non-trainable parameters (e.g., batch norm running stats).
        """
        ...

    fn gradients(self) raises -> List[ExTensor]:
        """Get gradients for all parameters.

        Returns:
            List of gradient tensors

        Note:
            Must correspond 1:1 with parameters().
            Gradients are accumulated across mini-batches.
        """
        ...

    fn zero_grad(inout self) raises:
        """Reset all gradients to zero.

        Called at the beginning of each mini-batch to clear
        accumulated gradients from previous iteration.

        Example:
            model.zero_grad()  # Clear gradients
            loss = forward_pass(model, input, target)
            backward_pass(loss)  # Accumulate gradients
            optimizer.step(model.parameters(), model.gradients())
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
        struct ConvLayer(Serializable):
            var weights: ExTensor
            var bias: ExTensor

            fn save(self, path: String) raises:
                # Save weights and bias to file
                write_tensor(path + "/weights.bin", self.weights)
                write_tensor(path + "/bias.bin", self.bias)

            fn load(inout self, path: String) raises:
                # Load weights and bias from file
                self.weights = read_tensor(path + "/weights.bin")
                self.bias = read_tensor(path + "/bias.bin")
    """

    fn save(self, path: String) raises:
        """Save component state to file.

        Args:
            path: File path or directory

        Raises:
            Error: If write fails or path is invalid

        Note:
            Should save all state needed to restore component.
            Include metadata (shapes, dtypes, version).
        """
        ...

    fn load(inout self, path: String) raises:
        """Load component state from file.

        Args:
            path: File path or directory

        Raises:
            Error: If file doesn't exist, is corrupted, or has version mismatch

        Note:
            Should validate loaded state (shapes, dtypes).
            Handle version migration if needed.
        """
        ...


trait Composable:
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
        struct Sequential(Composable):
            var layers: List[Composable]

            fn compose[T: Composable](self, other: T) -> Sequential:
                var new_layers = self.layers.copy()
                new_layers.append(other)
                return Sequential(new_layers)

        # Usage:
        var model = Linear(784, 128).compose(ReLU()).compose(Linear(128, 10))
    """

    fn compose[T: Composable](self, other: T) raises -> ComposedOp:
        """Compose this component with another.

        Args:
            other: Component to compose with

        Returns:
            Composed operation (self ∘ other)

        Raises:
            Error: If shapes are incompatible

        Example:
            var layer1 = Linear(784, 128)
            var layer2 = ReLU()
            var composed = layer1.compose(layer2)  # Linear -> ReLU
        """
        ...


struct ComposedOp(Differentiable, Composable):
    """Composition of two differentiable operations.

    Represents the composition f ∘ g where:
        forward: x -> g(f(x))
        backward: Uses chain rule

    Attributes:
        first: First operation (applied first)
        second: Second operation (applied second)
    """

    var first: Differentiable
    var second: Differentiable

    fn __init__(inout self, owned first: Differentiable, owned second: Differentiable):
        """Create composed operation.

        Args:
            first: First operation
            second: Second operation (receives output of first)
        """
        self.first = first^
        self.second = second^

    fn forward(inout self, input: ExTensor) raises -> ExTensor:
        """Forward pass through composition.

        Computes: second(first(input))
        """
        var intermediate = self.first.forward(input)
        return self.second.forward(intermediate)

    fn backward(self, grad_output: ExTensor) raises -> ExTensor:
        """Backward pass through composition.

        Uses chain rule: d(second ∘ first)/dx = d(second)/d(first) * d(first)/dx
        """
        var grad_second = self.second.backward(grad_output)
        return self.first.backward(grad_second)

    fn compose[T: Composable](self, other: T) raises -> ComposedOp:
        """Further compose with another operation.

        Returns: (self ∘ other)
        """
        return ComposedOp(self, other)


trait Trainable:
    """Components that support training mode.

    Implement this trait for components that behave differently during
    training vs. inference (e.g., Dropout, BatchNorm).

    Required Methods:
        train: Set to training mode
        eval: Set to evaluation mode
        is_training: Check current mode

    Example:
        struct Dropout(Trainable):
            var training: Bool
            var p: Float64

            fn train(inout self):
                self.training = True

            fn eval(inout self):
                self.training = False

            fn is_training(self) -> Bool:
                return self.training

            fn forward(self, input: ExTensor) -> ExTensor:
                if self.training:
                    # Apply dropout
                else:
                    # No dropout during inference
    """

    fn train(inout self):
        """Set component to training mode.

        Enables training-specific behavior (dropout, batch norm updates, etc.).
        """
        ...

    fn eval(inout self):
        """Set component to evaluation mode.

        Disables training-specific behavior for inference.
        """
        ...

    fn is_training(self) -> Bool:
        """Check if component is in training mode.

        Returns:
            True if training, False if evaluating
        """
        ...
