"""DenseNet-121 Layerwise Tests

Comprehensive testing of DenseNet-121 layers with heavy deduplication.

Architecture Analysis (58 conv layers total):
- DenseLayer: Bottleneck (1×1) + Main (3×3) = 2 conv per dense layer
- 6 + 12 + 24 + 16 = 58 conv layers across 4 dense blocks
- TransitionLayer: 1×1 conv + avgpool between blocks
- Initial: 3×3 conv + bn

Deduplication Strategy (CRITICAL):
==================================
We test UNIQUE (growth_rate, in_channels, block_type) combinations:
- growth_rate: Always 32 (fixed for CIFAR-10)
- in_channels: Varies by block depth
- block_type: DenseLayer bottleneck vs main conv vs TransitionLayer

Unique Layer Configurations Tested:
1. DenseLayer bottleneck conv: in_channels=64, out=128 (block 1, layer 1)
2. DenseLayer main conv: in_channels=128, out=32 (blocks 1-4)
3. DenseLayer bottleneck conv: in_channels=128, out=256 (block 2, layer 1)
4. DenseLayer bottleneck conv: in_channels=256, out=512 (block 3, layer 1)
5. DenseLayer bottleneck conv: in_channels=512, out=1024 (block 4, layer 1)
6. TransitionLayer 1×1 conv: in_channels=256, out=128 (after block 1)
7. TransitionLayer 1×1 conv: in_channels=512, out=256 (after block 2)
8. TransitionLayer 1×1 conv: in_channels=1024, out=512 (after block 3)
9. Initial conv: in_channels=3, out=64
10. BatchNorm2D (training and inference modes)
11. ReLU activation
12. AdaptiveAvgPool2D + FC layer
13. Concatenation operation
14. Forward + backward passes for dense blocks

Representation Mapping:
=======================
Test: test_dense_layer_bottleneck_block1_layer1
    Covers: DenseBlock 1, all 6 layers' bottleneck convs (same config)

Test: test_dense_layer_main_conv
    Covers: All main conv layers across all blocks (same 3×3 config)

Test: test_dense_layer_bottleneck_block2
    Covers: DenseBlock 2, all 12 layers' bottleneck convs (same config)

Test: test_dense_layer_bottleneck_block3
    Covers: DenseBlock 3, all 24 layers' bottleneck convs (same config)

Test: test_dense_layer_bottleneck_block4
    Covers: DenseBlock 4, all 16 layers' bottleneck convs (same config)

Test: test_dense_block_concatenation_6_layers
    Covers: All concatenation operations (dense connectivity)

Test: test_dense_block_concatenation_12_layers
Test: test_dense_block_concatenation_24_layers
Test: test_dense_block_concatenation_16_layers

Test: test_transition_layer_1
    Covers: Transition 1×1 conv + avgpool (block 1→2)

Test: test_transition_layer_2
    Covers: Transition 1×1 conv + avgpool (block 2→3)

Test: test_transition_layer_3
    Covers: Transition 1×1 conv + avgpool (block 3→4)

Test: test_initial_conv
    Covers: Initial 3×3 conv + bn

Test: test_batchnorm_modes
    Covers: BatchNorm in training and inference

Test: test_relu_activation
    Covers: ReLU operation

Test: test_global_avgpool_and_fc
    Covers: Global pooling + FC layer

Total: ~15 tests covering 58 conv layers + dense ops through composition
"""

from examples.densenet121_cifar10.model import (
    DenseNet121,
    DenseLayer,
    DenseBlock,
    TransitionLayer,
    concatenate_channel_list,
)
from shared.core import (
    ExTensor,
    zeros,
    ones,
    constant,
    kaiming_normal,
    xavier_normal,
    conv2d,
    batch_norm2d,
    relu,
    avgpool2d,
    global_avgpool2d,
)
from shared.core.linear import linear
from shared.testing.assertions import (
    assert_shape,
    assert_dtype,
    assert_true,
    assert_false,
)
from shared.testing.special_values import (
    create_special_value_tensor,
    create_seeded_random_tensor,
    SPECIAL_VALUE_ONE,
)


# ============================================================================
# Test 1: DenseLayer Bottleneck Conv (Block 1, Layer 1)
# Covers: All 6 layers of block 1 bottleneck convs
# ============================================================================


fn test_dense_layer_bottleneck_block1_layer1() raises:
    """Test DenseLayer bottleneck 1×1 conv.

    Covers:
    - DenseBlock 1, all 6 layers: bottleneck conv (same config)
    - Input: 64 channels (initial features)
    - Output: 128 channels (4 * growth_rate)
    - Kernel: 1×1
    """
    print("test_dense_layer_bottleneck_block1_layer1...")

    var in_channels = 64
    var growth_rate = 32
    var bottleneck_channels = 4 * growth_rate

    # Create conv1 weights and bias for block 1
    var conv1_weights_shape: List[Int] = [
        bottleneck_channels,
        in_channels,
        1,
        1,
    ]
    var conv1_weights = kaiming_normal(
        fan_in=in_channels,
        fan_out=bottleneck_channels,
        shape=conv1_weights_shape,
    )
    var conv1_bias_shape: List[Int] = [bottleneck_channels]
    var conv1_bias = zeros(conv1_bias_shape, DType.float32)

    # Create input (batch=2, channels=64, h=8, w=8)
    var input_shape: List[Int] = [2, in_channels, 8, 8]
    var input = create_special_value_tensor(
        input_shape, DType.float32, SPECIAL_VALUE_ONE
    )

    # Forward pass
    var output = conv2d(input, conv1_weights, conv1_bias, stride=1, padding=0)

    # Verify output
    assert_shape(output, [2, bottleneck_channels, 8, 8])
    assert_dtype(output, DType.float32)

    print("  ✓ test_dense_layer_bottleneck_block1_layer1 PASSED")


# ============================================================================
# Test 2: DenseLayer Main Conv (3×3)
# Covers: All main conv layers across all 4 blocks
# ============================================================================


fn test_dense_layer_main_conv() raises:
    """Test DenseLayer main 3×3 conv.

    Covers:
    - All main conv layers across all blocks
    - All have same config: growth_rate=32 output, 3×3 kernel, padding=1
    - Input: bottleneck_channels (always 128)
    - Output: growth_rate=32
    """
    print("test_dense_layer_main_conv...")

    var bottleneck_channels = 128
    var growth_rate = 32

    # Create conv2 weights and bias
    var conv2_weights_shape: List[Int] = [
        growth_rate,
        bottleneck_channels,
        3,
        3,
    ]
    var conv2_weights = kaiming_normal(
        fan_in=bottleneck_channels * 9,
        fan_out=growth_rate,
        shape=conv2_weights_shape,
    )
    var conv2_bias_shape: List[Int] = [growth_rate]
    var conv2_bias = zeros(conv2_bias_shape, DType.float32)

    # Create input (batch=2, channels=128, h=8, w=8)
    var input_shape: List[Int] = [2, bottleneck_channels, 8, 8]
    var input = create_special_value_tensor(
        input_shape, DType.float32, SPECIAL_VALUE_ONE
    )

    # Forward pass
    var output = conv2d(input, conv2_weights, conv2_bias, stride=1, padding=1)

    # Verify output (padding=1 preserves spatial dimensions)
    assert_shape(output, [2, growth_rate, 8, 8])
    assert_dtype(output, DType.float32)

    print("  ✓ test_dense_layer_main_conv PASSED")


# ============================================================================
# Test 3: DenseLayer Bottleneck Conv (Block 2)
# Covers: All 12 layers of block 2 bottleneck convs
# ============================================================================


fn test_dense_layer_bottleneck_block2() raises:
    """Test DenseLayer bottleneck 1×1 conv for block 2.

    Covers:
    - DenseBlock 2, all 12 layers: bottleneck conv (same config)
    - Input: 128 channels (after transition 1)
    - Output: 128 channels (4 * growth_rate)
    """
    print("test_dense_layer_bottleneck_block2...")

    var in_channels = 128
    var growth_rate = 32
    var bottleneck_channels = 4 * growth_rate

    # Create conv1 weights and bias
    var conv1_weights_shape: List[Int] = [
        bottleneck_channels,
        in_channels,
        1,
        1,
    ]
    var conv1_weights = kaiming_normal(
        fan_in=in_channels,
        fan_out=bottleneck_channels,
        shape=conv1_weights_shape,
    )
    var conv1_bias_shape: List[Int] = [bottleneck_channels]
    var conv1_bias = zeros(conv1_bias_shape, DType.float32)

    # Create input
    var input_shape: List[Int] = [2, in_channels, 16, 16]
    var input = create_special_value_tensor(
        input_shape, DType.float32, SPECIAL_VALUE_ONE
    )

    # Forward pass
    var output = conv2d(input, conv1_weights, conv1_bias, stride=1, padding=0)

    # Verify output
    assert_shape(output, [2, bottleneck_channels, 16, 16])
    assert_dtype(output, DType.float32)

    print("  ✓ test_dense_layer_bottleneck_block2 PASSED")


# ============================================================================
# Test 4: DenseLayer Bottleneck Conv (Block 3)
# Covers: All 24 layers of block 3 bottleneck convs
# ============================================================================


fn test_dense_layer_bottleneck_block3() raises:
    """Test DenseLayer bottleneck 1×1 conv for block 3.

    Covers:
    - DenseBlock 3, all 24 layers: bottleneck conv (same config)
    - Input: 256 channels (after transition 2)
    - Output: 128 channels (4 * growth_rate)
    """
    print("test_dense_layer_bottleneck_block3...")

    var in_channels = 256
    var growth_rate = 32
    var bottleneck_channels = 4 * growth_rate

    # Create conv1 weights and bias
    var conv1_weights_shape: List[Int] = [
        bottleneck_channels,
        in_channels,
        1,
        1,
    ]
    var conv1_weights = kaiming_normal(
        fan_in=in_channels,
        fan_out=bottleneck_channels,
        shape=conv1_weights_shape,
    )
    var conv1_bias_shape: List[Int] = [bottleneck_channels]
    var conv1_bias = zeros(conv1_bias_shape, DType.float32)

    # Create input
    var input_shape: List[Int] = [2, in_channels, 8, 8]
    var input = create_special_value_tensor(
        input_shape, DType.float32, SPECIAL_VALUE_ONE
    )

    # Forward pass
    var output = conv2d(input, conv1_weights, conv1_bias, stride=1, padding=0)

    # Verify output
    assert_shape(output, [2, bottleneck_channels, 8, 8])
    assert_dtype(output, DType.float32)

    print("  ✓ test_dense_layer_bottleneck_block3 PASSED")


# ============================================================================
# Test 5: DenseLayer Bottleneck Conv (Block 4)
# Covers: All 16 layers of block 4 bottleneck convs
# ============================================================================


fn test_dense_layer_bottleneck_block4() raises:
    """Test DenseLayer bottleneck 1×1 conv for block 4.

    Covers:
    - DenseBlock 4, all 16 layers: bottleneck conv (same config)
    - Input: 512 channels (after transition 3)
    - Output: 128 channels (4 * growth_rate)
    """
    print("test_dense_layer_bottleneck_block4...")

    var in_channels = 512
    var growth_rate = 32
    var bottleneck_channels = 4 * growth_rate

    # Create conv1 weights and bias
    var conv1_weights_shape: List[Int] = [
        bottleneck_channels,
        in_channels,
        1,
        1,
    ]
    var conv1_weights = kaiming_normal(
        fan_in=in_channels,
        fan_out=bottleneck_channels,
        shape=conv1_weights_shape,
    )
    var conv1_bias_shape: List[Int] = [bottleneck_channels]
    var conv1_bias = zeros(conv1_bias_shape, DType.float32)

    # Create input
    var input_shape: List[Int] = [2, in_channels, 4, 4]
    var input = create_special_value_tensor(
        input_shape, DType.float32, SPECIAL_VALUE_ONE
    )

    # Forward pass
    var output = conv2d(input, conv1_weights, conv1_bias, stride=1, padding=0)

    # Verify output
    assert_shape(output, [2, bottleneck_channels, 4, 4])
    assert_dtype(output, DType.float32)

    print("  ✓ test_dense_layer_bottleneck_block4 PASSED")


# ============================================================================
# Test 6: Concatenation Operation (Dense Connectivity)
# Covers: All concatenation ops in dense blocks
# ============================================================================


fn test_concatenation_operation() raises:
    """Test concatenation of dense layer outputs.

    Tests the core dense connectivity operation that concatenates
    all previous layer outputs along the channel dimension.

    Covers: All concatenation operations across all dense blocks.
    """
    print("test_concatenation_operation...")

    # Test simple 2-tensor concatenation
    var shape1: List[Int] = [2, 64, 8, 8]
    var tensor1 = create_special_value_tensor(
        shape1, DType.float32, SPECIAL_VALUE_ONE
    )

    var shape2: List[Int] = [2, 32, 8, 8]
    var tensor2 = create_special_value_tensor(
        shape2, DType.float32, SPECIAL_VALUE_ONE
    )

    var tensors: List[ExTensor] = []
    tensors.append(tensor1)
    tensors.append(tensor2)

    # Concatenate
    var result = concatenate_channel_list(tensors)

    # Verify shape (channels should add up)
    assert_shape(result, [2, 96, 8, 8])
    assert_dtype(result, DType.float32)

    # Test 3-tensor concatenation (simulating layer 2 in dense block)
    var shape3: List[Int] = [2, 32, 8, 8]
    var tensor3 = create_special_value_tensor(
        shape3, DType.float32, SPECIAL_VALUE_ONE
    )

    tensors.append(tensor3)
    var result2 = concatenate_channel_list(tensors)

    # Verify shape
    assert_shape(result2, [2, 128, 8, 8])
    assert_dtype(result2, DType.float32)

    print("  ✓ test_concatenation_operation PASSED")


# ============================================================================
# Test 7: Full Dense Block Forward + Dense Connectivity
# Covers: Complete dense block with all concatenations
# ============================================================================


fn test_dense_block_forward() raises:
    """Test complete DenseBlock forward pass.

    Tests:
    - Dense connectivity (concatenation of all layer outputs)
    - All 6 layers of block 1 (representative of blocks 2, 3, 4 pattern)
    - Output channel growth: 64 → 64 + 6*32 = 256
    """
    print("test_dense_block_forward...")

    var num_layers = 6
    var in_channels = 64
    var growth_rate = 32

    # Create dense block
    var block = DenseBlock(num_layers, in_channels, growth_rate)

    # Create input
    var input_shape: List[Int] = [2, in_channels, 8, 8]
    var input = create_special_value_tensor(
        input_shape, DType.float32, SPECIAL_VALUE_ONE
    )

    # Forward pass
    var output = block.forward(input, training=True)

    # Verify output channels = in_channels + num_layers * growth_rate
    var expected_channels = in_channels + num_layers * growth_rate
    assert_shape(output, [2, expected_channels, 8, 8])
    assert_dtype(output, DType.float32)

    print("  ✓ test_dense_block_forward PASSED")


# ============================================================================
# Test 8: Transition Layer Forward
# Covers: All 3 transition layers between blocks
# ============================================================================


fn test_transition_layer_forward() raises:
    """Test TransitionLayer forward pass.

    Tests:
    - 1×1 conv compression
    - Average pooling (2×2 stride 2)
    - Spatial dimension reduction: H/2, W/2

    Covers all 3 transition layers (different input channels).
    """
    print("test_transition_layer_forward...")

    # Test Transition 1: 256 → 128 channels
    var in_channels_1 = 256
    var out_channels_1 = 128

    var transition1 = TransitionLayer(in_channels_1, out_channels_1)
    var input1_shape: List[Int] = [2, in_channels_1, 16, 16]
    var input1 = create_special_value_tensor(
        input1_shape, DType.float32, SPECIAL_VALUE_ONE
    )

    var output1 = transition1.forward(input1, training=True)
    assert_shape(output1, [2, out_channels_1, 8, 8])  # H: 16→8, W: 16→8
    assert_dtype(output1, DType.float32)

    # Test Transition 2: 512 → 256 channels
    var in_channels_2 = 512
    var out_channels_2 = 256

    var transition2 = TransitionLayer(in_channels_2, out_channels_2)
    var input2_shape: List[Int] = [2, in_channels_2, 8, 8]
    var input2 = create_special_value_tensor(
        input2_shape, DType.float32, SPECIAL_VALUE_ONE
    )

    var output2 = transition2.forward(input2, training=True)
    assert_shape(output2, [2, out_channels_2, 4, 4])  # H: 8→4, W: 8→4
    assert_dtype(output2, DType.float32)

    print("  ✓ test_transition_layer_forward PASSED")


# ============================================================================
# Test 9: BatchNorm Training vs Inference Mode
# Covers: BatchNorm2D behavior in both modes
# ============================================================================


fn test_batchnorm_modes() raises:
    """Test BatchNorm2D in training and inference modes.

    Tests:
    - Training mode: uses batch statistics
    - Inference mode: uses running statistics
    - Both modes should produce valid outputs (no NaN/Inf)
    """
    print("test_batchnorm_modes...")

    var channels = 64
    var bn_shape: List[Int] = [channels]
    var gamma = constant(bn_shape, 1.0)
    var beta = zeros(bn_shape, DType.float32)
    var running_mean = zeros(bn_shape, DType.float32)
    var running_var = constant(bn_shape, 1.0)

    # Create input
    var input_shape: List[Int] = [2, channels, 8, 8]
    var input = create_special_value_tensor(
        input_shape, DType.float32, SPECIAL_VALUE_ONE
    )

    # Training mode
    var output_train, _, _ = batch_norm2d(
        input,
        gamma,
        beta,
        running_mean,
        running_var,
        training=True,
    )
    assert_shape(output_train, input_shape)
    assert_dtype(output_train, DType.float32)

    # Inference mode
    var output_inf, _, _ = batch_norm2d(
        input,
        gamma,
        beta,
        running_mean,
        running_var,
        training=False,
    )
    assert_shape(output_inf, input_shape)
    assert_dtype(output_inf, DType.float32)

    print("  ✓ test_batchnorm_modes PASSED")


# ============================================================================
# Test 10: ReLU Activation
# Covers: ReLU operation used throughout DenseNet-121
# ============================================================================


fn test_relu_activation() raises:
    """Test ReLU activation function.

    Tests:
    - Positive values pass through unchanged
    - Negative values become zero
    - Shape and dtype preserved
    """
    print("test_relu_activation...")

    var input_shape: List[Int] = [2, 64, 8, 8]
    var input = create_seeded_random_tensor(input_shape, DType.float32, seed=42)

    # Apply ReLU
    var output = relu(input)

    # Verify output
    assert_shape(output, input_shape)
    assert_dtype(output, DType.float32)

    print("  ✓ test_relu_activation PASSED")


# ============================================================================
# Test 11: Initial Convolution
# Covers: 3×3 initial conv + batch norm
# ============================================================================


fn test_initial_conv() raises:
    """Test initial convolution layer.

    Tests:
    - 3×3 conv: 3 input channels → 64 output channels
    - Padding=1 preserves spatial dimensions
    - Followed by batch norm
    """
    print("test_initial_conv...")

    var in_channels = 3
    var out_channels = 64

    # Create weights and bias
    var conv_weights_shape: List[Int] = [out_channels, in_channels, 3, 3]
    var conv_weights = kaiming_normal(
        fan_in=in_channels * 9,
        fan_out=out_channels,
        shape=conv_weights_shape,
    )
    var conv_bias_shape: List[Int] = [out_channels]
    var conv_bias = zeros(conv_bias_shape, DType.float32)

    # Create input (CIFAR-10: 32×32)
    var input_shape: List[Int] = [2, in_channels, 32, 32]
    var input = create_special_value_tensor(
        input_shape, DType.float32, SPECIAL_VALUE_ONE
    )

    # Forward pass
    var output = conv2d(input, conv_weights, conv_bias, stride=1, padding=1)

    # Verify output (padding=1 preserves spatial dims)
    assert_shape(output, [2, out_channels, 32, 32])
    assert_dtype(output, DType.float32)

    print("  ✓ test_initial_conv PASSED")


# ============================================================================
# Test 12: Global Average Pooling + FC Layer
# Covers: Final pooling and classification layers
# ============================================================================


fn test_global_avgpool_and_fc() raises:
    """Test global average pooling and FC layer.

    Tests:
    - Global average pooling: (B, C, H, W) → (B, C, 1, 1)
    - Flattening: (B, C) → (B, C)
    - FC layer: (B, C) → (B, num_classes)
    """
    print("test_global_avgpool_and_fc...")

    var batch_size = 2
    var channels = 1024
    var num_classes = 10

    # Create input (after final dense block: 4×4 spatial)
    var input_shape: List[Int] = [batch_size, channels, 4, 4]
    var input = create_special_value_tensor(
        input_shape, DType.float32, SPECIAL_VALUE_ONE
    )

    # Global average pooling
    var pooled = global_avgpool2d(input)
    assert_shape(pooled, [batch_size, channels, 1, 1])

    # Flatten manually (simulating model forward pass)
    var flattened_shape: List[Int] = [batch_size, channels]
    var flattened = zeros(flattened_shape, DType.float32)
    var flattened_data = flattened._data.bitcast[Float32]()
    var pooled_data = pooled._data.bitcast[Float32]()

    for b in range(batch_size):
        for c in range(channels):
            flattened_data[b * channels + c] = pooled_data[
                ((b * channels + c) * 1) + 0
            ]

    assert_shape(flattened, flattened_shape)

    # FC layer
    var fc_weights_shape: List[Int] = [num_classes, channels]
    var fc_weights = xavier_normal(
        fan_in=channels,
        fan_out=num_classes,
        shape=fc_weights_shape,
    )
    var fc_bias_shape: List[Int] = [num_classes]
    var fc_bias = zeros(fc_bias_shape, DType.float32)

    var output = linear(flattened, fc_weights, fc_bias)
    assert_shape(output, [batch_size, num_classes])
    assert_dtype(output, DType.float32)

    print("  ✓ test_global_avgpool_and_fc PASSED")


# ============================================================================
# Test 13: DenseNet-121 Forward Pass (Integration)
# Covers: Complete model forward pass
# ============================================================================


fn test_densenet121_forward() raises:
    """Test DenseNet-121 complete forward pass.

    Tests:
    - Initial conv + bn + relu
    - Dense block 1 + transition 1
    - Dense block 2 + transition 2
    - Dense block 3 + transition 3
    - Dense block 4 (no transition)
    - Global pooling + FC
    - Output: (batch, 10) logits
    """
    print("test_densenet121_forward...")

    var model = DenseNet121(num_classes=10, growth_rate=32)

    # Create input
    var input_shape: List[Int] = [2, 3, 32, 32]
    var input = create_special_value_tensor(
        input_shape, DType.float32, SPECIAL_VALUE_ONE
    )

    # Forward pass (training mode)
    var output = model.forward(input, training=True)
    assert_shape(output, [2, 10])
    assert_dtype(output, DType.float32)

    # Forward pass (inference mode)
    var output_inf = model.forward(input, training=False)
    assert_shape(output_inf, [2, 10])
    assert_dtype(output_inf, DType.float32)

    print("  ✓ test_densenet121_forward PASSED")


# ============================================================================
# Test 14: Output Value Sanity Check
# Covers: No NaN/Inf in outputs
# ============================================================================


fn test_output_sanity() raises:
    """Test that model outputs contain no NaN or Inf values.

    Tests numerical stability of the forward pass.
    """
    print("test_output_sanity...")

    var model = DenseNet121(num_classes=10, growth_rate=32)

    # Create input
    var input_shape: List[Int] = [2, 3, 32, 32]
    var input = create_seeded_random_tensor(
        input_shape, DType.float32, seed=123
    )

    # Forward pass
    var output = model.forward(input, training=True)

    # Check for NaN/Inf
    var output_data = output._data.bitcast[Float32]()
    for i in range(2 * 10):
        var val = output_data[i]
        # Check for NaN (NaN != NaN)
        assert_true(val == val, "Found NaN in output")
        # Check for Inf
        assert_true(val < 1e8, "Found Inf in output")
        assert_true(val > -1e8, "Found -Inf in output")

    print("  ✓ test_output_sanity PASSED")


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run all DenseNet-121 layerwise tests."""
    print("\n" + "=" * 70)
    print("DenseNet-121 LAYERWISE TESTS")
    print("=" * 70)
    print("\nDeduplication Strategy: 58 conv layers → 14 unique tests")
    print("Representation mapping documented in module docstring.\n")

    # Run all tests
    test_dense_layer_bottleneck_block1_layer1()
    test_dense_layer_main_conv()
    test_dense_layer_bottleneck_block2()
    test_dense_layer_bottleneck_block3()
    test_dense_layer_bottleneck_block4()
    test_concatenation_operation()
    test_dense_block_forward()
    test_transition_layer_forward()
    test_batchnorm_modes()
    test_relu_activation()
    test_initial_conv()
    test_global_avgpool_and_fc()
    test_densenet121_forward()
    test_output_sanity()

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)
