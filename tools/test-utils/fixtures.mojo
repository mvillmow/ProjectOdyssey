"""
Testing Utilities - Test Fixtures

Purpose: Common test fixtures for ML model testing
Language: Mojo (required for type safety and model compatibility)
"""

from tensor import Tensor, TensorShape


struct SimpleCNN:
    """
    Minimal CNN for testing purposes.

    A very simple 2-layer CNN for testing infrastructure without
    complexity of real models.
    """

    var in_channels: Int
    var out_channels: Int
    var num_classes: Int

    fn __init__(
        mut self,
        in_channels: Int = 1,
        out_channels: Int = 8,
        num_classes: Int = 10
    ):
        """
        Initialize simple CNN.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            num_classes: Number of output classes
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_classes = num_classes

    fn forward(
        self,
        borrowed input: Tensor[DType.float32]
    ) -> Tensor[DType.float32]:
        """
        Forward pass (simplified for testing).

        Args:
            input: Input tensor

        Returns:
            Output tensor
        """
        # Simplified forward pass for testing
        # In real implementation, this would include conv, pool, fc layers

        # For now, just return a tensor of appropriate shape
        let batch_size = input.shape()[0]
        var output = Tensor[DType.float32](TensorShape(batch_size, self.num_classes))

        # Fill with dummy values
        for i in range(batch_size * self.num_classes):
            output[i] = 0.1

        return output


struct LinearModel:
    """
    Simple linear model for testing.

    Just a single fully-connected layer for basic testing.
    """

    var in_features: Int
    var out_features: Int

    fn __init__(out self, in_features: Int, out_features: Int):
        """
        Initialize linear model.

        Args:
            in_features: Input dimension
            out_features: Output dimension
        """
        self.in_features = in_features
        self.out_features = out_features

    fn forward(
        self,
        borrowed input: Tensor[DType.float32]
    ) -> Tensor[DType.float32]:
        """
        Forward pass.

        Args:
            input: Input tensor (batch_size, in_features)

        Returns:
            Output tensor (batch_size, out_features)
        """
        # Simplified implementation
        let batch_size = input.shape()[0]
        var output = Tensor[DType.float32](TensorShape(batch_size, self.out_features))

        # Fill with dummy values
        for i in range(batch_size * self.out_features):
            output[i] = 0.0

        return output


fn create_test_model(model_type: String = "cnn") -> SimpleCNN:
    """
    Factory function to create test models.

    Args:
        model_type: Type of model ("cnn", "linear")

    Returns:
        Test model instance
    """
    # For now, only SimpleCNN is supported
    # TODO: Add support for other model types
    return SimpleCNN(in_channels=1, out_channels=8, num_classes=10)


fn create_linear_model(in_features: Int = 784, out_features: Int = 10) -> LinearModel:
    """
    Create a simple linear test model.

    Args:
        in_features: Input dimension
        out_features: Output dimension

    Returns:
        Linear model instance
    """
    return LinearModel(in_features, out_features)
