"""Unit tests for neural network layers.

Tests cover:
- Linear (fully connected) layers
- Convolutional layers (Conv2D)
- Pooling layers (MaxPool2D, AvgPool2D)
- Activation layers (ReLU, Sigmoid, Tanh)

Following TDD principles - these tests define the expected API
for implementation in Issue #49.

Note: Tests have been adapted from class-based API to pure functional API
as per architecture decision to use functional design throughout shared library.
"""

from tests.shared.conftest import (
    assert_true,
    assert_equal,
    assert_almost_equal,
    assert_shape_equal,
    TestFixtures,
)
from shared.core.extensor import ExTensor, zeros, ones
from shared.core.linear import linear, linear_no_bias
from shared.core.activation import relu, sigmoid, tanh, softmax
from collections.vector import DynamicVector


# ============================================================================
# Linear Layer Tests
# ============================================================================


fn test_linear_initialization() raises:
    """Test Linear layer parameter creation.

    Functional API Note:
        Pure functional design - no layer class initialization.
        Caller creates weight matrix (out_features, in_features) and bias vector.
        This test verifies parameters can be created with correct shapes.
    """
    # Create parameters for a linear transformation: in=10, out=5
    var in_features = 10
    var out_features = 5

    # Weights shape: (out_features, in_features) = (5, 10)
    var weight_shape = DynamicVector[Int](2)
    weight_shape[0] = out_features
    weight_shape[1] = in_features
    var weights = ones(weight_shape, DType.float32)

    # Bias shape: (out_features,) = (5,)
    var bias_shape = DynamicVector[Int](1)
    bias_shape[0] = out_features
    var bias = zeros(bias_shape, DType.float32)

    # Verify shapes
    var w_shape = weights.shape()
    var b_shape = bias.shape()
    assert_equal(w_shape[0], out_features)
    assert_equal(w_shape[1], in_features)
    assert_equal(b_shape[0], out_features)


fn test_linear_forward() raises:
    """Test Linear layer forward pass computation.

    Functional API:
        linear(x, weights, bias) -> output
        - Input shape: (batch_size, in_features)
        - Weights shape: (out_features, in_features)
        - Bias shape: (out_features,)
        - Output shape: (batch_size, out_features)
        - Computation: output = x @ weights.T + bias
    """
    # Create parameters: in=10, out=5
    var in_features = 10
    var out_features = 5
    var batch_size = 2

    # Weights: (5, 10) filled with 0.1
    var weight_shape = DynamicVector[Int](2)
    weight_shape[0] = out_features
    weight_shape[1] = in_features
    var weights = ones(weight_shape, DType.float32)
    # Fill with 0.1
    for i in range(out_features):
        for j in range(in_features):
            weights._data.bitcast[Float32]()[i * in_features + j] = 0.1

    # Bias: (5,) filled with 0.0
    var bias_shape = DynamicVector[Int](1)
    bias_shape[0] = out_features
    var bias = zeros(bias_shape, DType.float32)

    # Input: (2, 10) filled with 1.0
    var input_shape = DynamicVector[Int](2)
    input_shape[0] = batch_size
    input_shape[1] = in_features
    var input = ones(input_shape, DType.float32)

    # Forward pass
    var output = linear(input, weights, bias)

    # Check output shape
    var out_shape = output.shape()
    assert_equal(out_shape[0], batch_size)
    assert_equal(out_shape[1], out_features)

    # Check output values: sum of weights = 10 * 0.1 = 1.0
    var expected_value = Float32(10.0 * 0.1)
    assert_almost_equal(output._data.bitcast[Float32]()[0], expected_value, tolerance=1e-5)


fn test_linear_no_bias() raises:
    """Test Linear layer without bias term.

    Functional API:
        linear_no_bias(x, weights) -> output
        - No bias parameter required
        - Computation: output = x @ weights.T
    """
    # Create parameters: in=10, out=5
    var in_features = 10
    var out_features = 5

    # Weights: (5, 10) filled with 0.5
    var weight_shape = DynamicVector[Int](2)
    weight_shape[0] = out_features
    weight_shape[1] = in_features
    var weights = ones(weight_shape, DType.float32)
    for i in range(out_features * in_features):
        weights._data.bitcast[Float32]()[i] = 0.5

    # Input: (1, 10) filled with 1.0
    var input_shape = DynamicVector[Int](2)
    input_shape[0] = 1
    input_shape[1] = in_features
    var input = ones(input_shape, DType.float32)

    # Forward pass without bias
    var output = linear_no_bias(input, weights)

    # Check output shape
    var out_shape = output.shape()
    assert_equal(out_shape[0], 1)
    assert_equal(out_shape[1], out_features)

    # Check output values: sum = 10 * 0.5 = 5.0 (no bias added)
    var expected_value = Float32(10.0 * 0.5)
    assert_almost_equal(output._data.bitcast[Float32]()[0], expected_value, tolerance=1e-5)


fn test_linear_backward() raises:
    """Test Linear layer backward pass (gradient computation).

    Deferred - backward pass implementations are not yet available.
    Will be implemented when autograd/backpropagation system is added.
    """
    pass  # Deferred - backward pass not yet implemented


# ============================================================================
# Conv2D Layer Tests
# ============================================================================


fn test_conv2d_initialization() raises:
    """Test Conv2D layer initialization.

    API Contract:
        Conv2D(
            in_channels: Int,
            out_channels: Int,
            kernel_size: Int,
            stride: Int = 1,
            padding: Int = 0,
            bias: Bool = True
        )
    """
    # TODO(#1538): Implement when Conv2D is available
    # var layer = Conv2D(
    #     in_channels=3,
    #     out_channels=16,
    #     kernel_size=3,
    #     stride=1,
    #     padding=1
    # )
    # assert_equal(layer.in_channels, 3)
    # assert_equal(layer.out_channels, 16)
    # assert_equal(layer.kernel_size, 3)
    pass


fn test_conv2d_output_shape() raises:
    """Test Conv2D computes correct output shape.

    Formula: output_size = (input_size + 2*padding - kernel_size) / stride + 1

    API Contract:
        layer.forward(input: Tensor) -> Tensor
        - Input: (batch, in_channels, height, width)
        - Output: (batch, out_channels, out_height, out_width)
    """
    # TODO(#1538): Implement when Conv2D is available
    # # Input: (batch=1, channels=3, height=32, width=32)
    # # Conv2D: out_channels=16, kernel=3, stride=1, padding=1
    # # Expected output: (1, 16, 32, 32) - same spatial size due to padding
    #
    # var layer = Conv2D(3, 16, kernel_size=3, stride=1, padding=1)
    # var input = Tensor.randn(1, 3, 32, 32)
    # var output = layer.forward(input)
    # assert_shape_equal(output, Shape(1, 16, 32, 32))
    pass


fn test_conv2d_stride() raises:
    """Test Conv2D with stride > 1 downsamples correctly.

    API Contract:
        Conv2D with stride=2 should halve spatial dimensions
    """
    # TODO(#1538): Implement when Conv2D is available
    # # Input: (1, 3, 32, 32)
    # # Conv2D: kernel=3, stride=2, padding=1
    # # Expected output: (1, 16, 16, 16) - halved spatial size
    #
    # var layer = Conv2D(3, 16, kernel_size=3, stride=2, padding=1)
    # var input = Tensor.randn(1, 3, 32, 32)
    # var output = layer.forward(input)
    # assert_shape_equal(output, Shape(1, 16, 16, 16))
    pass


fn test_conv2d_valid_padding() raises:
    """Test Conv2D with no padding (valid convolution).

    API Contract:
        Conv2D with padding=0 reduces spatial dimensions
    """
    # TODO(#1538): Implement when Conv2D is available
    # # Input: (1, 3, 32, 32)
    # # Conv2D: kernel=5, stride=1, padding=0
    # # Expected output: (1, 16, 28, 28) - reduced by kernel_size-1
    #
    # var layer = Conv2D(3, 16, kernel_size=5, stride=1, padding=0)
    # var input = Tensor.randn(1, 3, 32, 32)
    # var output = layer.forward(input)
    # assert_shape_equal(output, Shape(1, 16, 28, 28))
    pass


# ============================================================================
# Activation Layer Tests
# ============================================================================


fn test_relu_activation() raises:
    """Test ReLU zeros negative values and preserves positive values.

    Functional API:
        relu(x) -> output
        - For each element: output = max(0, input)
    """
    # Test with known values: [-2.0, -1.0, 0.0, 1.0, 2.0]
    var shape = DynamicVector[Int](1)
    shape[0] = 5
    var input = zeros(shape, DType.float32)
    input._data.bitcast[Float32]()[0] = -2.0
    input._data.bitcast[Float32]()[1] = -1.0
    input._data.bitcast[Float32]()[2] = 0.0
    input._data.bitcast[Float32]()[3] = 1.0
    input._data.bitcast[Float32]()[4] = 2.0

    # Apply ReLU
    var output = relu(input)

    # Expected: [0.0, 0.0, 0.0, 1.0, 2.0]
    assert_almost_equal(output._data.bitcast[Float32]()[0], 0.0, tolerance=1e-6)
    assert_almost_equal(output._data.bitcast[Float32]()[1], 0.0, tolerance=1e-6)
    assert_almost_equal(output._data.bitcast[Float32]()[2], 0.0, tolerance=1e-6)
    assert_almost_equal(output._data.bitcast[Float32]()[3], 1.0, tolerance=1e-6)
    assert_almost_equal(output._data.bitcast[Float32]()[4], 2.0, tolerance=1e-6)


fn test_relu_in_place() raises:
    """Test ReLU can modify input in-place for memory efficiency.

    Not applicable to pure functional design - functional operations
    always return new tensors and never mutate inputs.
    """
    pass  # Not applicable - pure functional design


fn test_sigmoid_range() raises:
    """Test Sigmoid outputs values in range [0, 1].

    Functional API:
        sigmoid(x) -> output
        - For each element: output = 1 / (1 + exp(-input))
        - Output range: (0, 1)
    """
    # Test with various inputs: [-10.0, -1.0, 0.0, 1.0, 10.0]
    var shape = DynamicVector[Int](1)
    shape[0] = 5
    var input = zeros(shape, DType.float32)
    input._data.bitcast[Float32]()[0] = -10.0
    input._data.bitcast[Float32]()[1] = -1.0
    input._data.bitcast[Float32]()[2] = 0.0
    input._data.bitcast[Float32]()[3] = 1.0
    input._data.bitcast[Float32]()[4] = 10.0

    # Apply sigmoid
    var output = sigmoid(input)

    # All outputs should be in (0, 1)
    for i in range(5):
        var val = output._data.bitcast[Float32]()[i]
        assert_less(0.0, val)  # Greater than 0
        assert_less(val, 1.0)  # Less than 1

    # Check sigmoid(0) = 0.5
    assert_almost_equal(output._data.bitcast[Float32]()[2], 0.5, tolerance=1e-6)


fn test_tanh_range() raises:
    """Test Tanh outputs values in range [-1, 1].

    Functional API:
        tanh(x) -> output
        - For each element: output = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        - Output range: (-1, 1)
    """
    # Test with various inputs: [-10.0, -1.0, 0.0, 1.0, 10.0]
    var shape = DynamicVector[Int](1)
    shape[0] = 5
    var input = zeros(shape, DType.float32)
    input._data.bitcast[Float32]()[0] = -10.0
    input._data.bitcast[Float32]()[1] = -1.0
    input._data.bitcast[Float32]()[2] = 0.0
    input._data.bitcast[Float32]()[3] = 1.0
    input._data.bitcast[Float32]()[4] = 10.0

    # Apply tanh
    var output = tanh(input)

    # All outputs should be in (-1, 1)
    for i in range(5):
        var val = output._data.bitcast[Float32]()[i]
        assert_less(-1.0, val)  # Greater than -1
        assert_less(val, 1.0)  # Less than 1

    # Check tanh(0) = 0.0
    assert_almost_equal(output._data.bitcast[Float32]()[2], 0.0, tolerance=1e-6)


# ============================================================================
# Pooling Layer Tests
# ============================================================================


fn test_maxpool2d_downsampling() raises:
    """Test MaxPool2D downsamples spatial dimensions.

    API Contract:
        MaxPool2D(kernel_size: Int, stride: Int = None, padding: Int = 0)
        - Reduces spatial dimensions by kernel_size (if stride=kernel_size)
    """
    # TODO(#1538): Implement when MaxPool2D is available
    # # Input: (1, 16, 32, 32)
    # # MaxPool2D: kernel=2, stride=2
    # # Expected output: (1, 16, 16, 16)
    #
    # var pool = MaxPool2D(kernel_size=2, stride=2)
    # var input = Tensor.randn(1, 16, 32, 32)
    # var output = pool.forward(input)
    # assert_shape_equal(output, Shape(1, 16, 16, 16))
    pass


fn test_maxpool2d_max_selection() raises:
    """Test MaxPool2D selects maximum value in each window.

    API Contract:
        MaxPool2D selects max over kernel_size x kernel_size window
    """
    # TODO(#1538): Implement when MaxPool2D is available
    # var pool = MaxPool2D(kernel_size=2)
    #
    # # Create input with known values
    # # [[1, 2], [3, 4]] -> max = 4
    # var input = Tensor(
    #     List[Float32](1.0, 2.0, 3.0, 4.0),
    #     Shape(1, 1, 2, 2)
    # )
    # var output = pool.forward(input)
    #
    # # Output should be single value: 4.0
    # assert_almost_equal(output[0, 0, 0, 0], 4.0)
    pass


# ============================================================================
# Property-Based Tests
# ============================================================================


fn test_layer_property_batch_independence() raises:
    """Property: Layer output for batch should equal individual outputs.

    This tests that layers process batch elements independently.
    """
    # TODO(#1538): Implement when layers are available
    # var layer = Linear(10, 5)
    #
    # # Create batch input
    # var batch_input = Tensor.randn(4, 10)
    #
    # # Process as batch
    # var batch_output = layer.forward(batch_input)
    #
    # # Process individually
    # for i in range(4):
    #     var single_input = batch_input[i:i+1, :]
    #     var single_output = layer.forward(single_input)
    #     assert_tensor_equal(single_output, batch_output[i:i+1, :])
    pass


fn test_layer_property_deterministic() raises:
    """Property: Layer forward pass is deterministic.

    Same input should always produce same output.
    """
    # TODO(#1538): Implement when layers are available
    # var layer = Linear(10, 5)
    # var input = Tensor.randn(2, 10, seed=42)
    #
    # # Two forward passes with same input
    # var output1 = layer.forward(input)
    # var output2 = layer.forward(input)
    #
    # # Outputs should be identical
    # assert_tensor_equal(output1, output2)
    pass


# ============================================================================
# Test Main
# ============================================================================


fn main() raises:
    """Run all layer tests."""
    print("Running Linear layer tests...")
    test_linear_initialization()
    test_linear_forward()
    test_linear_no_bias()
    test_linear_backward()

    print("Running Conv2D layer tests...")
    test_conv2d_initialization()
    test_conv2d_output_shape()
    test_conv2d_stride()
    test_conv2d_valid_padding()

    print("Running activation layer tests...")
    test_relu_activation()
    test_relu_in_place()
    test_sigmoid_range()
    test_tanh_range()

    print("Running pooling layer tests...")
    test_maxpool2d_downsampling()
    test_maxpool2d_max_selection()

    print("Running property-based tests...")
    test_layer_property_batch_independence()
    test_layer_property_deterministic()

    print("\nAll layer tests passed! ✓")
