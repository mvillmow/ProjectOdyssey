"""Linear (fully connected) layer - stub implementation for Issue #2395.

This module provides a basic Linear layer for neural networks.
The stub implementation initializes weights and biases but the forward
pass returns zeros. Full implementation is planned for a future issue.

Key components:
- Linear: Fully connected layer with learnable weights and bias
  Implements: y = xW + b
"""

from shared.core.extensor import ExTensor, zeros, randn, zeros_like


struct Linear(Copyable, Movable):
    """Linear layer: y = xW + b (stub for testing).

    A fully connected neural network layer that transforms inputs
    from in_features to out_features dimensions.

    Attributes:
        weight: Weight matrix of shape (in_features, out_features).
        bias: Bias vector of shape (out_features,).
        in_features: Input feature dimension.
        out_features: Output feature dimension.
    """

    var weight: ExTensor
    var bias: ExTensor
    var in_features: Int
    var out_features: Int

    fn __init__(out self, in_features: Int, out_features: Int) raises:
        """Initialize linear layer with random weights and zero bias.

        Uses Xavier-style initialization for weights. Bias is initialized to zero.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.

        Raises:
            Error if tensor creation fails.

        Example:
            ```mojo
            var layer = Linear(10, 5)  # 10 inputs, 5 outputs
            ```
        """
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weights with randn (standard normal distribution)
        self.weight = randn(
            List[Int](in_features, out_features),
            DType.float32
        )

        # Initialize bias to zeros
        self.bias = zeros(List[Int](out_features), DType.float32)

    fn forward(self, input: ExTensor) raises -> ExTensor:
        """Forward pass: y = xW + b (stub - returns zeros for now).

        Args:
            input: Input tensor of shape (batch_size, in_features) or (in_features,).

        Returns:
            Output tensor of shape (batch_size, out_features) or (out_features,).

        Raises:
            Error if tensor operations fail.

        Note:
            This is a stub implementation that returns zeros. Full matrix
            multiplication implementation is planned for a later issue.

        Example:
            ```mojo
            var layer = Linear(10, 5)
            var input = ones([4, 10], DType.float32)  # batch of 4 samples
            var output = layer.forward(input)  # Shape: [4, 5]
            ```
        """
        # TODO: Implement proper matrix multiplication: output = input @ weight + bias
        # For now, return zeros with correct output shape
        if len(input.shape()) == 1:
            # Single sample: (in_features,) -> (out_features,)
            return zeros(List[Int](self.out_features), DType.float32)
        else:
            # Batch: (batch_size, in_features) -> (batch_size, out_features)
            var batch_size = input.shape()[0]
            return zeros(
                List[Int](batch_size, self.out_features),
                DType.float32
            )

    fn parameters(self) raises -> List[ExTensor]:
        """Get list of trainable parameters.

        Returns:
            List containing [weight, bias] tensors.

        Raises:
            Error if tensor copying fails.

        Example:
            ```mojo
            var layer = Linear(10, 5)
            var params = layer.parameters()
            # params[0] is weight, params[1] is bias
            ```
        """
        var params = List[ExTensor]()
        # Create copies of weight and bias tensors
        var weight_copy = zeros_like(self.weight)
        var bias_copy = zeros_like(self.bias)
        var weight_size = self.weight.numel()
        var bias_size = self.bias.numel()
        for i in range(weight_size):
            weight_copy._data[i] = self.weight._data[i]
        for i in range(bias_size):
            bias_copy._data[i] = self.bias._data[i]
        params.append(weight_copy^)
        params.append(bias_copy^)
        return params^
