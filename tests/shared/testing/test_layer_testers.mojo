"""Tests for layer_testers module

Tests LayerTester utility functions:
- test_layer_dtype_consistency
- test_layer_no_invalid_values
- test_conv_layer_backward
- test_linear_layer_backward
- test_activation_layer_backward
- test_batchnorm_layer
- test_batchnorm_layer_backward

Note: Full layer tests (conv, linear, pooling, activation) will be tested
in model-specific test files where actual layers with weights exist.
This file focuses on testing the utility/helper functions in LayerTester.
"""

from math import isnan, isinf
from shared.testing.layer_testers import LayerTester
from shared.testing.special_values import (
    create_ones_tensor,
    create_special_value_tensor,
    create_seeded_random_tensor,
)
from shared.testing.assertions import assert_true
from shared.testing.tensor_factory import zeros_tensor
from shared.core.extensor import ExTensor


fn test_layer_dtype_consistency_passes() raises:
    """Test that dtype consistency check passes when dtypes match."""
    var input = create_ones_tensor([1, 3, 32, 32], DType.float32)
    var output = create_ones_tensor([1, 64, 30, 30], DType.float32)

    # Should not raise - dtypes match
    LayerTester.test_layer_dtype_consistency(input, output, "TestLayer")


fn test_layer_dtype_consistency_different_shapes() raises:
    """Test that dtype consistency works with different shapes."""
    var input = create_ones_tensor([1, 3], DType.float16)
    var output = create_ones_tensor([1, 10], DType.float16)

    # Should not raise - dtypes match even though shapes differ
    LayerTester.test_layer_dtype_consistency(input, output, "TestFC")


fn test_layer_no_invalid_values_passes() raises:
    """Test that invalid value check passes when no NaN/Inf."""
    var output = create_ones_tensor([2, 3, 4, 5], DType.float32)

    # Should not raise - all values are 1.0 (finite)
    LayerTester.test_layer_no_invalid_values(output, "TestLayer")


fn test_layer_no_invalid_values_with_zeros() raises:
    """Test that invalid value check passes with zero values."""
    var output = create_special_value_tensor([3, 3], DType.float32, 0.0)

    # Should not raise - zeros are valid
    LayerTester.test_layer_no_invalid_values(output, "TestLayer")


fn test_layer_no_invalid_values_with_halves() raises:
    """Test that invalid value check passes with 0.5 values."""
    var output = create_special_value_tensor([2, 2], DType.float32, 0.5)

    # Should not raise - 0.5 is valid
    LayerTester.test_layer_no_invalid_values(output, "TestLayer")


fn test_layer_tester_utility_functions() raises:
    """Test that LayerTester utility functions are accessible."""
    # This test verifies that we can call the static methods without errors

    var input = create_ones_tensor([1, 3, 32, 32], DType.float32)
    var output = create_ones_tensor([1, 64, 30, 30], DType.float32)

    # Test dtype consistency
    LayerTester.test_layer_dtype_consistency(input, output, "Conv1")

    # Test no invalid values
    LayerTester.test_layer_no_invalid_values(output, "Conv1")


fn test_activation_layer_backward_relu() raises:
    """Test ReLU activation backward pass with gradient checking."""
    # Use small shape to avoid timeout
    LayerTester.test_activation_layer_backward(
        shape=[2, 3, 4, 4],
        dtype=DType.float32,
        activation="relu"
    )


fn test_activation_layer_backward_sigmoid() raises:
    """Test Sigmoid activation backward pass with gradient checking."""
    # Use small shape to avoid timeout
    LayerTester.test_activation_layer_backward(
        shape=[2, 3, 4, 4],
        dtype=DType.float32,
        activation="sigmoid"
    )


fn test_activation_layer_backward_tanh() raises:
    """Test Tanh activation backward pass with gradient checking."""
    # Use small shape to avoid timeout
    LayerTester.test_activation_layer_backward(
        shape=[2, 3, 4, 4],
        dtype=DType.float32,
        activation="tanh"
    )


fn test_linear_layer_backward_fp32() raises:
    """Test linear layer backward pass with FP32 gradient checking."""
    # Create small weight and bias tensors
    var weights = create_ones_tensor([10, 8], DType.float32)
    var bias = create_ones_tensor([10], DType.float32)

    # Test backward pass
    LayerTester.test_linear_layer_backward(
        in_features=8,
        out_features=10,
        weights=weights,
        bias=bias,
        dtype=DType.float32
    )


fn test_conv_layer_backward_fp32() raises:
    """Test conv layer backward pass with FP32 gradient checking."""
    # Create small kernel and bias tensors (small spatial dimensions to avoid timeout)
    var weights = create_ones_tensor([8, 3, 3, 3], DType.float32)  # 8 filters, 3x3
    var bias = create_ones_tensor([8], DType.float32)

    # Test backward pass with small input size
    LayerTester.test_conv_layer_backward(
        in_channels=3,
        out_channels=8,
        kernel_size=3,
        input_h=8,
        input_w=8,
        weights=weights,
        bias=bias,
        dtype=DType.float32,
        stride=1,
        padding=1
    )


fn test_batchnorm_layer_training_mode() raises:
    """Test BatchNorm layer in training mode."""
    var num_features = 16
    var input_shape = List[Int]()
    input_shape.append(2)      # batch size
    input_shape.append(16)     # channels
    input_shape.append(4)      # height
    input_shape.append(4)      # width

    var gamma = create_ones_tensor([num_features], DType.float32)
    var beta = create_ones_tensor([num_features], DType.float32)
    var running_mean = create_ones_tensor([num_features], DType.float32)
    var running_var = create_ones_tensor([num_features], DType.float32)

    # Test training mode
    LayerTester.test_batchnorm_layer(
        num_features=num_features,
        input_shape=input_shape,
        gamma=gamma,
        beta=beta,
        running_mean=running_mean,
        running_var=running_var,
        dtype=DType.float32,
        training_mode=True
    )


fn test_batchnorm_layer_inference_mode() raises:
    """Test BatchNorm layer in inference mode."""
    var num_features = 16
    var input_shape = List[Int]()
    input_shape.append(2)      # batch size
    input_shape.append(16)     # channels
    input_shape.append(4)      # height
    input_shape.append(4)      # width

    var gamma = create_ones_tensor([num_features], DType.float32)
    var beta = create_ones_tensor([num_features], DType.float32)
    var running_mean = create_ones_tensor([num_features], DType.float32)
    var running_var = create_ones_tensor([num_features], DType.float32)

    # Test inference mode
    LayerTester.test_batchnorm_layer(
        num_features=num_features,
        input_shape=input_shape,
        gamma=gamma,
        beta=beta,
        running_mean=running_mean,
        running_var=running_var,
        dtype=DType.float32,
        training_mode=False
    )


fn test_batchnorm_layer_backward_fp32() raises:
    """Test BatchNorm backward pass with FP32."""
    var num_features = 16
    var input_shape = List[Int]()
    input_shape.append(2)      # batch size
    input_shape.append(16)     # channels
    input_shape.append(4)      # height
    input_shape.append(4)      # width

    var gamma = create_ones_tensor([num_features], DType.float32)
    var beta = create_ones_tensor([num_features], DType.float32)
    var running_mean = create_ones_tensor([num_features], DType.float32)
    var running_var = create_ones_tensor([num_features], DType.float32)

    # Test backward pass
    LayerTester.test_batchnorm_layer_backward(
        num_features=num_features,
        input_shape=input_shape,
        gamma=gamma,
        beta=beta,
        running_mean=running_mean,
        running_var=running_var,
        dtype=DType.float32
    )


fn main() raises:
    print("Testing layer_testers module...")

    # Test dtype consistency
    test_layer_dtype_consistency_passes()
    print("✓ test_layer_dtype_consistency_passes")

    test_layer_dtype_consistency_different_shapes()
    print("✓ test_layer_dtype_consistency_different_shapes")

    # Test invalid value checking
    test_layer_no_invalid_values_passes()
    print("✓ test_layer_no_invalid_values_passes")

    test_layer_no_invalid_values_with_zeros()
    print("✓ test_layer_no_invalid_values_with_zeros")

    test_layer_no_invalid_values_with_halves()
    print("✓ test_layer_no_invalid_values_with_halves")

    # Test utility functions
    test_layer_tester_utility_functions()
    print("✓ test_layer_tester_utility_functions")

    # Test backward pass methods
    test_activation_layer_backward_relu()
    print("✓ test_activation_layer_backward_relu")

    test_activation_layer_backward_sigmoid()
    print("✓ test_activation_layer_backward_sigmoid")

    test_activation_layer_backward_tanh()
    print("✓ test_activation_layer_backward_tanh")

    test_linear_layer_backward_fp32()
    print("✓ test_linear_layer_backward_fp32")

    test_conv_layer_backward_fp32()
    print("✓ test_conv_layer_backward_fp32")

    test_batchnorm_layer_training_mode()
    print("✓ test_batchnorm_layer_training_mode")

    test_batchnorm_layer_inference_mode()
    print("✓ test_batchnorm_layer_inference_mode")

    test_batchnorm_layer_backward_fp32()
    print("✓ test_batchnorm_layer_backward_fp32")

    print("\n✅ All layer_testers tests passed!")
    print("\nBackward pass testing:")
    print("- Uses seeded random tensors for reproducibility")
    print("- Validates numerical gradients via finite differences")
    print("- Checks for NaN/Inf in gradient outputs")
    print("- Supports ReLU, Sigmoid, Tanh activations")
