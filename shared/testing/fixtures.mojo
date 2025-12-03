"""Test fixtures and utilities for ML Odyssey.

Provides common test setup utilities, model fixtures, and assertion helpers
for validating neural network implementations.

Features:
    - Simple test models (CNN, Linear) for basic functionality testing
    - Model factory functions for consistent initialization
    - Common test setup patterns
    - Assertion helpers for tensor validation

Example:
    from shared.testing.fixtures import create_test_cnn, create_linear_model
    from shared.core import zeros, ones

    fn test_forward_pass():
        var model = create_test_cnn(1, 8, 10)
        var batch_size = 32
        var input_shape = List[Int](batch_size, 1, 28, 28)
        var input = ones(input_shape, DType.float32)
        # Test forward pass...

    fn test_linear_output_shape():
        var model = create_linear_model(784, 10)
        assert_equal(model.in_features, 784)
        assert_equal(model.out_features, 10)
"""

from shared.core import ExTensor, zeros, ones, full, zeros_like


struct SimpleCNN(Copyable, Movable):
    """Minimal CNN for testing purposes.

    A very simple 2-layer CNN for testing infrastructure without the
    complexity of real models. Useful for:
    - Testing training loops
    - Validating gradient computation
    - Checking data pipeline integration
    - Performance benchmarking

    Shape flow:
        Input: (batch_size, in_channels, height, width)
        Output: (batch_size, num_classes)
    """

    var in_channels: Int
    var out_channels: Int
    var num_classes: Int

    fn __init__(out self, in_channels: Int = 1, out_channels: Int = 8, num_classes: Int = 10):
        """Initialize simple CNN.

        Args:
            in_channels: Number of input channels (default: 1 for MNIST-like)
            out_channels: Number of output channels from first conv (default: 8)
            num_classes: Number of output classes (default: 10)

        Example:
            var model = SimpleCNN(1, 8, 10)
            assert_equal(model.num_classes, 10)
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_classes = num_classes

    fn get_output_shape(self, batch_size: Int) -> List[Int]:
        """Get output shape for given batch size.

        Args:
            batch_size: Number of samples in batch

        Returns:
            Output shape (batch_size, num_classes)
        """
        var shape = List[Int]()
        shape.append(batch_size)
        shape.append(self.num_classes)
        return shape

    fn forward(self, input: ExTensor) -> ExTensor:
        """Forward pass (simplified for testing).

        Creates output tensor with correct shape filled with dummy values.
        In real implementation, this would include conv, pool, fc layers.

        Args:
            input: Input tensor (batch_size, in_channels, height, width)

        Returns:
            Output tensor (batch_size, num_classes)

        Note:
            This is a placeholder for testing. Real implementations should
            contain actual neural network operations.
        """
        var batch_size = input._shape[0]
        var output_shape = self.get_output_shape(batch_size)
        var output = zeros(output_shape, input._dtype)

        # Fill with dummy values (0.1 per element)
        for i in range(output.numel()):
            output._set_float64(i, 0.1)

        return output


struct LinearModel(Copyable, Movable):
    """Simple linear model for testing.

    A single fully-connected layer for basic testing of:
    - Linear transformations
    - Batch processing
    - Gradient computation
    - Loss functions

    Shape flow:
        Input: (batch_size, in_features)
        Output: (batch_size, out_features)
    """

    var in_features: Int
    var out_features: Int

    fn __init__(out self, in_features: Int, out_features: Int):
        """Initialize linear model.

        Args:
            in_features: Input dimension
            out_features: Output dimension

        Example:
            var model = LinearModel(784, 10)
            assert_equal(model.in_features, 784)
        """
        self.in_features = in_features
        self.out_features = out_features

    fn get_output_shape(self, batch_size: Int) -> List[Int]:
        """Get output shape for given batch size.

        Args:
            batch_size: Number of samples in batch

        Returns:
            Output shape (batch_size, out_features)
        """
        var shape = List[Int]()
        shape.append(batch_size)
        shape.append(self.out_features)
        return shape

    fn forward(self, input: ExTensor) -> ExTensor:
        """Forward pass.

        Creates output tensor with correct shape filled with zeros.
        In real implementation, this would compute y = xW^T + b.

        Args:
            input: Input tensor (batch_size, in_features)

        Returns:
            Output tensor (batch_size, out_features)

        Note:
            This is a placeholder for testing. Real implementations should
            contain actual linear transformation: y = matmul(input, weights^T) + bias
        """
        var batch_size = input._shape[0]
        var output_shape = self.get_output_shape(batch_size)
        var output = zeros(output_shape, input._dtype)

        # Already zero-filled by zeros(), no need to fill again
        return output


fn create_test_cnn(
    in_channels: Int = 1,
    out_channels: Int = 8,
    num_classes: Int = 10
) -> SimpleCNN:
    """Factory function to create test CNN models.

    Useful for consistent model initialization across tests.

    Args:
        in_channels: Number of input channels (default: 1 for MNIST)
        out_channels: Number of output channels (default: 8)
        num_classes: Number of output classes (default: 10)

    Returns:
        Test CNN instance

    Example:
        var model = create_test_cnn(1, 8, 10)
        assert_equal(model.num_classes, 10)

        # Create custom channel configuration
        var custom_model = create_test_cnn(3, 32, 1000)
        assert_equal(custom_model.in_channels, 3)
    """
    return SimpleCNN(in_channels, out_channels, num_classes)


fn create_linear_model(in_features: Int = 784, out_features: Int = 10) -> LinearModel:
    """Create a simple linear test model.

    Useful for testing linear layers, loss functions, and optimization.

    Args:
        in_features: Input dimension (default: 784 for MNIST flattened)
        out_features: Output dimension (default: 10 for MNIST classes)

    Returns:
        Linear model instance

    Example:
        var model = create_linear_model(784, 10)
        assert_equal(model.out_features, 10)

        # Create custom dimensions
        var large_model = create_linear_model(2048, 1024)
        assert_equal(large_model.in_features, 2048)
    """
    return LinearModel(in_features, out_features)


fn create_test_input(
    batch_size: Int,
    in_features: Int,
    dtype: DType = DType.float32
) -> ExTensor:
    """Create a test input tensor for linear models.

    Creates a simple tensor filled with ones for testing.

    Args:
        batch_size: Number of samples
        in_features: Input dimension
        dtype: Data type (default: float32)

    Returns:
        Input tensor (batch_size, in_features) filled with 1.0

    Example:
        var input = create_test_input(32, 784)
        assert_equal(input._shape[0], 32)
        assert_equal(input._shape[1], 784)
    """
    var shape = List[Int]()
    shape.append(batch_size)
    shape.append(in_features)
    return ones(shape, dtype)


fn create_test_targets(
    batch_size: Int,
    num_classes: Int,
    dtype: DType = DType.int32
) -> ExTensor:
    """Create a test target tensor for classification.

    Creates a tensor filled with zeros for testing.

    Args:
        batch_size: Number of samples
        num_classes: Number of classes
        dtype: Data type (default: int32)

    Returns:
        Target tensor (batch_size,) filled with 0

    Example:
        var targets = create_test_targets(32, 10)
        assert_equal(targets._shape[0], 32)
    """
    var shape = List[Int]()
    shape.append(batch_size)
    return zeros(shape, dtype)


fn assert_tensor_shape(tensor: ExTensor, expected_shape: List[Int]) -> Bool:
    """Validate tensor has expected shape.

    Args:
        tensor: Tensor to check
        expected_shape: Expected shape dimensions

    Returns:
        True if shapes match, False otherwise

    Example:
        var tensor = ones(List[Int](32, 10), DType.float32)
        assert_true(assert_tensor_shape(tensor, List[Int](32, 10)))
    """
    if tensor._shape.size() != expected_shape.size():
        return False

    for i in range(expected_shape.size()):
        if tensor._shape[i] != expected_shape[i]:
            return False

    return True


fn assert_tensor_dtype(tensor: ExTensor, expected_dtype: DType) -> Bool:
    """Validate tensor has expected data type.

    Args:
        tensor: Tensor to check
        expected_dtype: Expected DType

    Returns:
        True if dtypes match, False otherwise

    Example:
        var tensor = ones(List[Int](32, 10), DType.float32)
        assert_true(assert_tensor_dtype(tensor, DType.float32))
    """
    return tensor._dtype == expected_dtype


fn assert_tensor_all_finite(tensor: ExTensor) -> Bool:
    """Check if all tensor values are finite (no NaN or Inf).

    Args:
        tensor: Tensor to check

    Returns:
        True if all values are finite, False if any NaN/Inf found

    Example:
        var tensor = ones(List[Int](32, 10), DType.float32)
        assert_true(assert_tensor_all_finite(tensor))
    """
    for i in range(tensor.numel()):
        var val = tensor._get_float64(i)
        # Check for NaN
        if val != val:  # NaN != NaN is true
            return False
        # Check for Inf
        if val > 1e308 or val < -1e308:
            return False

    return True


fn assert_tensor_not_all_zeros(tensor: ExTensor) -> Bool:
    """Check if tensor contains at least one non-zero value.

    Useful for verifying weights are initialized and gradients are flowing.

    Args:
        tensor: Tensor to check

    Returns:
        True if at least one non-zero value exists, False if all zeros

    Example:
        var tensor = ones(List[Int](32, 10), DType.float32)
        assert_true(assert_tensor_not_all_zeros(tensor))
    """
    for i in range(tensor.numel()):
        if tensor._get_float64(i) != 0.0:
            return True

    return False
