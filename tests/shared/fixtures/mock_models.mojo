"""Mock model architectures for testing.

This module provides simple mock model implementations for testing training
loops, forward/backward passes, and model serialization without needing
full model implementations.

Key components:
- MockLayer: Minimal layer implementation
- SimpleLinearModel: Single linear layer
- SimpleMLP: Multi-layer perceptron (2-3 layers)

All models use simple operations for predictable testing.
"""

from tests.shared.fixtures.mock_tensors import (
    create_random_tensor,
    create_zeros_tensor,
    create_ones_tensor,
)
from shared.core import zeros_like, zeros, ExTensor


# ============================================================================
# Mock Layer Implementation
# ============================================================================


struct MockLayer:
    """Minimal mock layer for testing.

    Provides a simple layer that performs identity transformation
    or scaled transformation for testing purposes.

    Attributes:
        input_dim: Input feature dimension.
        output_dim: Output feature dimension.
        scale: Scale factor applied to inputs.
    """

    var input_dim: Int
    var output_dim: Int
    var scale: Float32

    fn __init__(
        mut self, input_dim: Int, output_dim: Int, scale: Float32 = 1.0
    ):
        """Initialize mock layer.

        Args:
            input_dim: Input dimension.
            output_dim: Output dimension.
            scale: Scale factor for transformation (default: 1.0).

        Example:
            ```mojo
            var layer = MockLayer(input_dim=10, output_dim=5, scale=2.0)
            ```

        Note:
            This is a simplified layer for testing - not a real neural network layer.
            It performs a simple scaled identity or zero-padding transformation.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.scale = scale

    fn forward(self, input: List[Float32]) -> List[Float32]:
        """Forward pass through mock layer.

        Args:
            input: Input tensor (flat list of size input_dim).

        Returns:
            Output tensor (flat list of size output_dim).

        Example:
            ```mojo
            var layer = MockLayer(10, 5, scale=2.0)
            var input = create_ones_tensor([10])
            var output = layer.forward(input)
            # output contains first 5 elements scaled by 2.0
            ```

        Note:
            If output_dim < input_dim: truncates and scales
            If output_dim > input_dim: pads with zeros after scaling
            If output_dim == input_dim: scales all elements
        """
        var output = List[Float32](capacity=self.output_dim)

        if self.output_dim <= self.input_dim:
            # Truncate and scale
            for i in range(self.output_dim):
                output.append(input[i] * self.scale)
        else:
            # Scale input portion
            for i in range(self.input_dim):
                output.append(input[i] * self.scale)
            # Pad with zeros
            for _ in range(self.input_dim, self.output_dim):
                output.append(0.0)

        return output

    fn num_parameters(self) -> Int:
        """Get number of trainable parameters.

        Returns:
            Number of parameters (for testing parameter counting).

        Example:
            ```mojo
            var layer = MockLayer(10, 5)
            var n_params = layer.num_parameters()  # 50 (10*5)
            ```

        Note:
            Returns input_dim * output_dim to simulate weight matrix size.
        """
        return self.input_dim * self.output_dim


# ============================================================================
# Simple Linear Model
# ============================================================================


struct SimpleLinearModel:
    """Simple linear model (single layer).

    Provides a minimal single-layer linear model for testing basic
    forward passes and training workflows.

    Attributes:
        input_dim: Input feature dimension.
        output_dim: Output dimension.
        weights: Model weights (flattened weight matrix).
        bias: Model bias.
        use_bias: Whether to use bias.
    """

    var input_dim: Int
    var output_dim: Int
    var weights: List[Float32]
    var bias: List[Float32]
    var use_bias: Bool

    fn __init__(
        mut self,
        input_dim: Int,
        output_dim: Int,
        use_bias: Bool = True,
        init_value: Float32 = 0.1,
    ):
        """Initialize simple linear model.

        Args:
            input_dim: Input dimension.
            output_dim: Output dimension.
            use_bias: Whether to use bias (default: True).
            init_value: Initial weight/bias value (default: 0.1).

        Example:
            ```mojo
            var model = SimpleLinearModel(
                input_dim=10,
                output_dim=5,
                use_bias=True,
                init_value=0.1
            )
            ```

        Note:
            Weights are initialized to constant value for predictable testing.
            For random initialization, use create_random_weights() after creation.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias

        # Initialize weights: output_dim x input_dim (flattened)
        var n_weights = output_dim * input_dim
        self.weights = List[Float32](capacity=n_weights)
        for _ in range(n_weights):
            self.weights.append(init_value)

        # Initialize bias
        self.bias = List[Float32](capacity=output_dim)
        if use_bias:
            for _ in range(output_dim):
                self.bias.append(init_value)

    fn forward(self, input: List[Float32]) -> List[Float32]:
        """Forward pass: output = weights @ input + bias.

        Args:
            input: Input tensor (flat list of size input_dim).

        Returns:
            Output tensor (flat list of size output_dim).

        Example:
            ```mojo
            var model = SimpleLinearModel(10, 5)
            var input = create_ones_tensor([10])
            var output = model.forward(input)
            # output = weights @ input + bias
            ```

        Note:
            Performs matrix-vector multiplication: y = W @ x + b
            where W is output_dim x input_dim weight matrix.
        """
        var output = List[Float32](capacity=self.output_dim)

        # Compute W @ x
        for i in range(self.output_dim):
            var sum = Float32(0.0)
            for j in range(self.input_dim):
                var weight_idx = i * self.input_dim + j
                sum += self.weights[weight_idx] * input[j]

            # Add bias if used
            if self.use_bias:
                sum += self.bias[i]

            output.append(sum)

        return output

    fn num_parameters(self) -> Int:
        """Get total number of parameters.

        Returns:
            Total parameters (weights + bias).

        Example:
            ```mojo
            var model = SimpleLinearModel(10, 5, use_bias=True)
            var n_params = model.num_parameters()  # 55 (50 + 5)
            ```
        """
        var n_params = len(self.weights)
        if self.use_bias:
            n_params += len(self.bias)
        return n_params

    fn randomize_weights(mut self, random_seed: Int = 42):
        """Randomize model weights.

        Args:
            random_seed: Random seed for reproducibility (default: 42).

        Example:
            ```mojo
            var model = SimpleLinearModel(10, 5)
            model.randomize_weights(seed=42)
            # Model now has random weights
            ```
        """
        # Generate random weights
        var random_weights = create_random_tensor(
            [self.output_dim, self.input_dim], random_seed=random_seed
        )
        self.weights = random_weights

        # Generate random bias if used
        if self.use_bias:
            var random_bias = create_random_tensor(
                [self.output_dim], random_seed=random_seed + 1000
            )
            self.bias = random_bias


# ============================================================================
# Parameter Structure
# ============================================================================


struct Parameter(Copyable, Movable):
    """Trainable parameter with data and gradient.

    Wraps an ExTensor for the parameter data with an associated gradient tensor.
    Used in model layers to track trainable weights and biases.

    Attributes:
        data: The parameter tensor (weights or bias).
        grad: The gradient tensor for backpropagation.
    """

    var data: ExTensor
    var grad: ExTensor

    fn __init__(out self, data: ExTensor) raises:
        """Initialize parameter with data tensor and zero gradient.

        Args:
            data: The parameter tensor.

        Raises:
            Error: If gradient tensor cannot be created.

        Example:
            ```mojo
            var param = Parameter(data_tensor)
            # grad is automatically initialized to zeros_like(data_tensor)
            ```
        """
        self.data = data
        self.grad = zeros_like(data)

    fn shape(self) -> List[Int]:
        """Get the shape of the parameter.

        Returns:
            Shape of the parameter tensor.

        Example:
            ```mojo
            var param = Parameter(tensor)
            var s = param.shape()  # Same as param.data.shape()
            ```
        """
        return self.data.shape()


# ============================================================================
# Simple Multi-Layer Perceptron
# ============================================================================


struct SimpleMLP(Copyable, Movable, ImplicitlyCopyable):
    """Simple multi-layer perceptron (2-3 layers).

    Provides a minimal MLP for testing multi-layer forward passes,
    training loops, and model composition.

    Attributes:
        input_dim: Input dimension.
        hidden_dim: Hidden layer dimension.
        output_dim: Output dimension.
        num_hidden_layers: Number of hidden layers (1 or 2).
        layer1_weights: First layer weights.
        layer1_bias: First layer bias.
        layer2_weights: Second layer weights (hidden to hidden or hidden to output).
        layer2_bias: Second layer bias.
        layer3_weights: Third layer weights (only if num_hidden_layers=2).
        layer3_bias: Third layer bias (only if num_hidden_layers=2).
    """

    var input_dim: Int
    var hidden_dim: Int
    var output_dim: Int
    var num_hidden_layers: Int
    var layer1_weights: List[Float32]
    var layer1_bias: List[Float32]
    var layer2_weights: List[Float32]
    var layer2_bias: List[Float32]
    var layer3_weights: List[Float32]
    var layer3_bias: List[Float32]

    fn __init__(
        out self,
        input_dim: Int,
        hidden_dim: Int,
        output_dim: Int,
        num_hidden_layers: Int = 1,
        init_value: Float32 = 0.1,
    ):
        """Initialize simple MLP.

        Args:
            input_dim: Input dimension.
            hidden_dim: Hidden layer dimension.
            output_dim: Output dimension.
            num_hidden_layers: Number of hidden layers (1 or 2, default: 1).
            init_value: Initial weight/bias value (default: 0.1).

        Example:
            ```mojo
            # 2-layer MLP: input -> hidden -> output
            var mlp = SimpleMLP(
                input_dim=10,
                hidden_dim=20,
                output_dim=5,
                num_hidden_layers=1
            )

            # 3-layer MLP: input -> hidden -> hidden -> output
            var deep_mlp = SimpleMLP(
                input_dim=10,
                hidden_dim=20,
                output_dim=5,
                num_hidden_layers=2
            )
            ```
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_hidden_layers = num_hidden_layers

        # Layer 1: input -> hidden
        var n_weights1 = hidden_dim * input_dim
        self.layer1_weights = List[Float32](capacity=n_weights1)
        self.layer1_bias = List[Float32](capacity=hidden_dim)
        for _ in range(n_weights1):
            self.layer1_weights.append(init_value)
        for _ in range(hidden_dim):
            self.layer1_bias.append(init_value)

        # Layer 2: depends on num_hidden_layers
        if num_hidden_layers == 1:
            # hidden -> output
            var n_weights2 = output_dim * hidden_dim
            self.layer2_weights = List[Float32](capacity=n_weights2)
            self.layer2_bias = List[Float32](capacity=output_dim)
            for _ in range(n_weights2):
                self.layer2_weights.append(init_value)
            for _ in range(output_dim):
                self.layer2_bias.append(init_value)

            # Layer 3 unused
            self.layer3_weights = List[Float32]()
            self.layer3_bias = List[Float32]()
        else:
            # hidden -> hidden
            var n_weights2 = hidden_dim * hidden_dim
            self.layer2_weights = List[Float32](capacity=n_weights2)
            self.layer2_bias = List[Float32](capacity=hidden_dim)
            for _ in range(n_weights2):
                self.layer2_weights.append(init_value)
            for _ in range(hidden_dim):
                self.layer2_bias.append(init_value)

            # Layer 3: hidden -> output
            var n_weights3 = output_dim * hidden_dim
            self.layer3_weights = List[Float32](capacity=n_weights3)
            self.layer3_bias = List[Float32](capacity=output_dim)
            for _ in range(n_weights3):
                self.layer3_weights.append(init_value)
            for _ in range(output_dim):
                self.layer3_bias.append(init_value)

    fn forward(self, input: List[Float32]) -> List[Float32]:
        """Forward pass through MLP.

        Args:
            input: Input tensor.

        Returns:
            Output tensor.

        Example:
            ```mojo
            var mlp = SimpleMLP(10, 20, 5)
            var input = create_ones_tensor([10])
            var output = mlp.forward(input)
            # output has shape [5]
            ```

        Note:
            Uses ReLU activation between layers: ReLU(x) = max(0, x)
        """
        # Layer 1: input -> hidden
        var hidden1 = self._linear_forward(
            input,
            self.layer1_weights,
            self.layer1_bias,
            self.hidden_dim,
            self.input_dim,
        )
        hidden1 = self._relu(hidden1)

        # Layer 2
        if self.num_hidden_layers == 1:
            # hidden -> output
            var output = self._linear_forward(
                hidden1,
                self.layer2_weights,
                self.layer2_bias,
                self.output_dim,
                self.hidden_dim,
            )
            return output
        else:
            # hidden -> hidden
            var hidden2 = self._linear_forward(
                hidden1,
                self.layer2_weights,
                self.layer2_bias,
                self.hidden_dim,
                self.hidden_dim,
            )
            hidden2 = self._relu(hidden2)

            # Layer 3: hidden -> output
            var output = self._linear_forward(
                hidden2,
                self.layer3_weights,
                self.layer3_bias,
                self.output_dim,
                self.hidden_dim,
            )
            return output

    fn _linear_forward(
        self,
        input: List[Float32],
        weights: List[Float32],
        bias: List[Float32],
        out_dim: Int,
        in_dim: Int,
    ) -> List[Float32]:
        """Linear layer forward pass: output = weights @ input + bias."""
        var output = List[Float32](capacity=out_dim)

        for i in range(out_dim):
            var sum = Float32(0.0)
            for j in range(in_dim):
                var weight_idx = i * in_dim + j
                sum += weights[weight_idx] * input[j]
            sum += bias[i]
            output.append(sum)

        return output

    fn _relu(self, input: List[Float32]) -> List[Float32]:
        """ReLU activation: max(0, x)."""
        var output = List[Float32](capacity=len(input))
        for i in range(len(input)):
            var val = input[i]
            output.append(max(0.0, val))
        return output

    fn num_parameters(self) -> Int:
        """Get total number of parameters.

        Returns:
            Total parameters across all layers.
        """
        var total = len(self.layer1_weights) + len(self.layer1_bias)
        total += len(self.layer2_weights) + len(self.layer2_bias)
        if self.num_hidden_layers == 2:
            total += len(self.layer3_weights) + len(self.layer3_bias)
        return total

    fn get_weights(self) raises -> ExTensor:
        """Get flattened weights as ExTensor.

        Returns:
            ExTensor containing all weights flattened.

        Example:
            ```mojo
            var mlp = SimpleMLP(10, 20, 5)
            var weights = mlp.get_weights()
            ```
        """
        # Flatten all weight lists into a single list
        var all_weights = List[Float32]()
        for i in range(len(self.layer1_weights)):
            all_weights.append(self.layer1_weights[i])
        for i in range(len(self.layer2_weights)):
            all_weights.append(self.layer2_weights[i])
        for i in range(len(self.layer3_weights)):
            all_weights.append(self.layer3_weights[i])

        # Create ExTensor from flattened weights
        var shape = List[Int](len(all_weights))
        var tensor = zeros(shape, DType.float32)

        # Copy weights into tensor
        for i in range(len(all_weights)):
            tensor._set_float32(i, all_weights[i])

        return tensor

    fn parameters(self) raises -> List[Parameter]:
        """Get list of Parameter objects.

        Returns:
            List of Parameter objects wrapping weights and biases.

        Example:
            ```mojo
            var mlp = SimpleMLP(10, 20, 5)
            var params = mlp.parameters()
            for param in params:
                print(param.shape())
            ```
        """
        var param_list = List[Parameter]()

        # Layer 1 weights parameter
        var w1_shape = List[Int](self.hidden_dim, self.input_dim)
        var w1_tensor = zeros(w1_shape, DType.float32)
        for i in range(len(self.layer1_weights)):
            w1_tensor._set_float32(i, self.layer1_weights[i])
        param_list.append(Parameter(w1_tensor))

        # Layer 1 bias parameter
        var b1_shape = List[Int](self.hidden_dim)
        var b1_tensor = zeros(b1_shape, DType.float32)
        for i in range(len(self.layer1_bias)):
            b1_tensor._set_float32(i, self.layer1_bias[i])
        param_list.append(Parameter(b1_tensor))

        # Layer 2 weights parameter
        var w2_dim_in = self.hidden_dim
        var w2_dim_out = (
            self.output_dim if self.num_hidden_layers == 1 else self.hidden_dim
        )
        var w2_shape = List[Int](w2_dim_out, w2_dim_in)
        var w2_tensor = zeros(w2_shape, DType.float32)
        for i in range(len(self.layer2_weights)):
            w2_tensor._set_float32(i, self.layer2_weights[i])
        param_list.append(Parameter(w2_tensor))

        # Layer 2 bias parameter
        var b2_shape = List[Int](w2_dim_out)
        var b2_tensor = zeros(b2_shape, DType.float32)
        for i in range(len(self.layer2_bias)):
            b2_tensor._set_float32(i, self.layer2_bias[i])
        param_list.append(Parameter(b2_tensor))

        # Layer 3 parameters (only if num_hidden_layers == 2)
        if self.num_hidden_layers == 2:
            var w3_shape = List[Int](self.output_dim, self.hidden_dim)
            var w3_tensor = zeros(w3_shape, DType.float32)
            for i in range(len(self.layer3_weights)):
                w3_tensor._set_float32(i, self.layer3_weights[i])
            param_list.append(Parameter(w3_tensor))

            var b3_shape = List[Int](self.output_dim)
            var b3_tensor = zeros(b3_shape, DType.float32)
            for i in range(len(self.layer3_bias)):
                b3_tensor._set_float32(i, self.layer3_bias[i])
            param_list.append(Parameter(b3_tensor))

        return param_list^

    fn zero_grad(mut self):
        """Zero all gradients.

        Example:
            ```mojo
            var mlp = SimpleMLP(10, 20, 5)
            mlp.zero_grad()
            ```
        """
        # This is a placeholder - actual gradient zeroing would happen
        # on Parameter objects during backpropagation
        pass

    fn state_dict(self) raises -> Dict[String, ExTensor]:
        """Export weights as state dictionary.

        Returns:
            Dictionary mapping parameter names to their tensor values.

        Example:
            ```mojo
            var mlp = SimpleMLP(10, 20, 5)
            var state = mlp.state_dict()
            ```
        """
        var state = Dict[String, ExTensor]()

        # Layer 1 weights
        var w1_shape = List[Int](self.hidden_dim, self.input_dim)
        var w1_tensor = zeros(w1_shape, DType.float32)
        for i in range(len(self.layer1_weights)):
            w1_tensor._set_float32(i, self.layer1_weights[i])
        state["layer1_weights"] = w1_tensor

        # Layer 1 bias
        var b1_shape = List[Int](self.hidden_dim)
        var b1_tensor = zeros(b1_shape, DType.float32)
        for i in range(len(self.layer1_bias)):
            b1_tensor._set_float32(i, self.layer1_bias[i])
        state["layer1_bias"] = b1_tensor

        # Layer 2 weights
        var w2_dim_in = self.hidden_dim
        var w2_dim_out = (
            self.output_dim if self.num_hidden_layers == 1 else self.hidden_dim
        )
        var w2_shape = List[Int](w2_dim_out, w2_dim_in)
        var w2_tensor = zeros(w2_shape, DType.float32)
        for i in range(len(self.layer2_weights)):
            w2_tensor._set_float32(i, self.layer2_weights[i])
        state["layer2_weights"] = w2_tensor

        # Layer 2 bias
        var b2_shape = List[Int](w2_dim_out)
        var b2_tensor = zeros(b2_shape, DType.float32)
        for i in range(len(self.layer2_bias)):
            b2_tensor._set_float32(i, self.layer2_bias[i])
        state["layer2_bias"] = b2_tensor

        # Layer 3 parameters (only if num_hidden_layers == 2)
        if self.num_hidden_layers == 2:
            var w3_shape = List[Int](self.output_dim, self.hidden_dim)
            var w3_tensor = zeros(w3_shape, DType.float32)
            for i in range(len(self.layer3_weights)):
                w3_tensor._set_float32(i, self.layer3_weights[i])
            state["layer3_weights"] = w3_tensor

            var b3_shape = List[Int](self.output_dim)
            var b3_tensor = zeros(b3_shape, DType.float32)
            for i in range(len(self.layer3_bias)):
                b3_tensor._set_float32(i, self.layer3_bias[i])
            state["layer3_bias"] = b3_tensor

        return state

    fn load_state_dict(mut self, state: Dict[String, ExTensor]):
        """Load weights from state dictionary.

        Args:
            state: Dictionary mapping parameter names to their tensor values.

        Example:
            ```mojo
            var mlp = SimpleMLP(10, 20, 5)
            mlp.load_state_dict(state)
            ```

        Note:
            This is a placeholder implementation for now.
        """
        # Placeholder for weight loading
        pass
