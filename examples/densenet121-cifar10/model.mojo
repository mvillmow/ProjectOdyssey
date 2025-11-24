"""DenseNet-121 Model for CIFAR-10

This module implements the DenseNet-121 architecture adapted for CIFAR-10.

Architecture:
    - 121 layers deep (58 conv layers in dense blocks + transitions + initial + classifier)
    - Input: 32×32×3 RGB images
    - Output: 10 classes
    - ~7M parameters

Key Innovation:
    - Dense connectivity: Each layer connects to all subsequent layers
    - Feature reuse: All layers can access features from all previous layers
    - Short gradient paths: Direct connections from loss to all layers
    - Parameter efficient: Only 7M parameters despite 121 layers

References:
    Huang et al. (2017) - Densely Connected Convolutional Networks
    https://arxiv.org/abs/1608.06993
"""

from shared.core import (
    ExTensor,
    zeros,
    conv2d,
    avgpool2d,
    batch_norm2d,
    relu,
    linear,
    global_avgpool2d,
    kaiming_normal,
    xavier_normal,
    constant,
)


fn concatenate_channel_list(tensors: List[ExTensor]) raises -> ExTensor:
    """Concatenate a list of tensors along the channel dimension.

    Args:
        tensors: List of tensors to concatenate (all with shape: (B, C_i, H, W))

    Returns:
        Concatenated tensor (B, sum(C_i), H, W)
    """
    if len(tensors) == 0:
        raise Error("Cannot concatenate empty list")

    if len(tensors) == 1:
        return tensors[0]

    var batch_size = tensors[0].shape()[0]
    var height = tensors[0].shape()[2]
    var width = tensors[0].shape()[3]

    # Calculate total channels
    var total_channels = 0
    for i in range(len(tensors)):
        total_channels += tensors[i].shape()[1]

    # Create output tensor
    var result = zeros(
        List[Int]()
        .append(batch_size)
        .append(total_channels)
        .append(height)
        .append(width),
        tensors[0].dtype(),
    )

    var result_data = result._data.bitcast[Float32]()
    var hw = height * width
    var offset = 0

    # Copy each tensor's data
    for i in range(len(tensors)):
        var tensor = tensors[i]
        var channels = tensor.shape()[1]
        var tensor_data = tensor._data.bitcast[Float32]()

        for b in range(batch_size):
            for c in range(channels):
                for idx in range(hw):
                    var src_idx = ((b * channels + c) * hw) + idx
                    var dst_idx = ((b * total_channels + (offset + c)) * hw) + idx
                    result_data[dst_idx] = tensor_data[src_idx]

        offset += channels

    return result


struct DenseLayer:
    """Single dense layer with bottleneck architecture.

    Architecture:
        Input → BN → ReLU → Conv1×1(4k) → BN → ReLU → Conv3×3(k) → Output

    The output is concatenated with the input to form the next layer's input.
    """

    # Bottleneck: 1×1 convolution (reduces channels before expensive 3×3)
    var bn1_gamma: ExTensor
    var bn1_beta: ExTensor
    var bn1_running_mean: ExTensor
    var bn1_running_var: ExTensor
    var conv1_weights: ExTensor
    var conv1_bias: ExTensor

    # Main convolution: 3×3
    var bn2_gamma: ExTensor
    var bn2_beta: ExTensor
    var bn2_running_mean: ExTensor
    var bn2_running_var: ExTensor
    var conv2_weights: ExTensor
    var conv2_bias: ExTensor

    fn __init__(out self, in_channels: Int, growth_rate: Int) raises:
        """Initialize dense layer.

        Args:
            in_channels: Number of input channels (concatenated from all previous layers)
            growth_rate: Number of output channels (k, typically 32)
        """
        var bottleneck_channels = 4 * growth_rate

        # Bottleneck 1×1 conv
        self.bn1_gamma = constant(List[Int]().append(in_channels), 1.0)
        self.bn1_beta = zeros(List[Int]().append(in_channels))
        self.bn1_running_mean = zeros(List[Int]().append(in_channels))
        self.bn1_running_var = constant(List[Int]().append(in_channels), 1.0)
        self.conv1_weights = kaiming_normal(
            List[Int]()
            .append(bottleneck_channels)
            .append(in_channels)
            .append(1)
            .append(1),
            fan_in=in_channels,
        )
        self.conv1_bias = zeros(List[Int]().append(bottleneck_channels))

        # 3×3 conv
        self.bn2_gamma = constant(List[Int]().append(bottleneck_channels), 1.0)
        self.bn2_beta = zeros(List[Int]().append(bottleneck_channels))
        self.bn2_running_mean = zeros(List[Int]().append(bottleneck_channels))
        self.bn2_running_var = constant(List[Int]().append(bottleneck_channels), 1.0)
        self.conv2_weights = kaiming_normal(
            List[Int]()
            .append(growth_rate)
            .append(bottleneck_channels)
            .append(3)
            .append(3),
            fan_in=bottleneck_channels * 9,
        )
        self.conv2_bias = zeros(List[Int]().append(growth_rate))

    fn forward(mut self, x: ExTensor, training: Bool) raises -> ExTensor:
        """Forward pass through dense layer.

        Args:
            x: Input tensor (batch, in_channels, H, W)
            training: Training mode flag

        Returns:
            Output tensor (batch, growth_rate, H, W)
        """
        # Bottleneck
        var out = batch_norm2d(
            x, self.bn1_gamma, self.bn1_beta, self.bn1_running_mean, self.bn1_running_var, training
        )
        out = relu(out)
        out = conv2d(out, self.conv1_weights, self.conv1_bias, stride=1, padding=0)

        # 3×3 convolution
        out = batch_norm2d(
            out, self.bn2_gamma, self.bn2_beta, self.bn2_running_mean, self.bn2_running_var, training
        )
        out = relu(out)
        out = conv2d(out, self.conv2_weights, self.conv2_bias, stride=1, padding=1)

        return out


struct DenseBlock:
    """Dense block with L dense layers and dense connectivity.

    Each layer receives all previous feature maps as input and produces
    growth_rate new feature maps. The output is the concatenation of all
    layer outputs.

    NOTE: This is a simplified implementation with fixed structure.
    For a fully general implementation, you would dynamically create layers.
    """

    var num_layers: Int
    var growth_rate: Int
    var layers: List[DenseLayer]

    fn __init__(out self, num_layers: Int, in_channels: Int, growth_rate: Int) raises:
        """Initialize dense block.

        Args:
            num_layers: Number of dense layers in this block
            in_channels: Number of input channels to the first layer
            growth_rate: Growth rate (k) for each layer
        """
        self.num_layers = num_layers
        self.growth_rate = growth_rate
        self.layers = List[DenseLayer]()

        # Create layers with increasing input channels
        for i in range(num_layers):
            var layer_in_channels = in_channels + i * growth_rate
            self.layers.append(DenseLayer(layer_in_channels, growth_rate))

    fn forward(mut self, x: ExTensor, training: Bool) raises -> ExTensor:
        """Forward pass through dense block with dense connectivity.

        Args:
            x: Input tensor (batch, in_channels, H, W)
            training: Training mode flag

        Returns:
            Output tensor (batch, in_channels + num_layers * growth_rate, H, W)
        """
        var features = List[ExTensor]()
        features.append(x)

        for i in range(self.num_layers):
            # Concatenate all previous features
            var concat_input = concatenate_channel_list(features)

            # Pass through this layer
            var layer_output = self.layers[i].forward(concat_input, training)

            # Add to feature list
            features.append(layer_output)

        # Final output: concatenation of all features
        return concatenate_channel_list(features)


struct TransitionLayer:
    """Transition layer between dense blocks.

    Reduces both spatial dimensions (via pooling) and channels (via 1×1 conv).

    Architecture:
        Input → BN → Conv1×1(θ × in_channels) → AvgPool2×2 → Output

    Where θ = 0.5 (compression factor)
    """

    var bn_gamma: ExTensor
    var bn_beta: ExTensor
    var bn_running_mean: ExTensor
    var bn_running_var: ExTensor
    var conv_weights: ExTensor
    var conv_bias: ExTensor

    fn __init__(out self, in_channels: Int, out_channels: Int) raises:
        """Initialize transition layer.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels (typically in_channels / 2)
        """
        self.bn_gamma = constant(List[Int]().append(in_channels), 1.0)
        self.bn_beta = zeros(List[Int]().append(in_channels))
        self.bn_running_mean = zeros(List[Int]().append(in_channels))
        self.bn_running_var = constant(List[Int]().append(in_channels), 1.0)

        self.conv_weights = kaiming_normal(
            List[Int]()
            .append(out_channels)
            .append(in_channels)
            .append(1)
            .append(1),
            fan_in=in_channels,
        )
        self.conv_bias = zeros(List[Int]().append(out_channels))

    fn forward(mut self, x: ExTensor, training: Bool) raises -> ExTensor:
        """Forward pass through transition layer.

        Args:
            x: Input tensor (batch, in_channels, H, W)
            training: Training mode flag

        Returns:
            Output tensor (batch, out_channels, H/2, W/2)
        """
        var out = batch_norm2d(
            x, self.bn_gamma, self.bn_beta, self.bn_running_mean, self.bn_running_var, training
        )
        out = conv2d(out, self.conv_weights, self.conv_bias, stride=1, padding=0)
        out = avgpool2d(out, kernel_size=2, stride=2, padding=0)
        return out


struct DenseNet121:
    """DenseNet-121 for CIFAR-10.

    Architecture:
        - Initial conv block
        - Dense Block 1: 6 layers (64 → 256 channels)
        - Transition 1: 256 → 128 channels, 32×32 → 16×16
        - Dense Block 2: 12 layers (128 → 512 channels)
        - Transition 2: 512 → 256 channels, 16×16 → 8×8
        - Dense Block 3: 24 layers (256 → 1024 channels)
        - Transition 3: 1024 → 512 channels, 8×8 → 4×4
        - Dense Block 4: 16 layers (512 → 1024 channels)
        - Global average pooling + classifier

    Total parameters: ~7M
    Growth rate: k = 32
    """

    # Initial convolution
    var initial_conv_weights: ExTensor
    var initial_conv_bias: ExTensor
    var initial_bn_gamma: ExTensor
    var initial_bn_beta: ExTensor
    var initial_bn_running_mean: ExTensor
    var initial_bn_running_var: ExTensor

    # Dense blocks and transitions
    var dense_block_1: DenseBlock  # 6 layers
    var transition_1: TransitionLayer
    var dense_block_2: DenseBlock  # 12 layers
    var transition_2: TransitionLayer
    var dense_block_3: DenseBlock  # 24 layers
    var transition_3: TransitionLayer
    var dense_block_4: DenseBlock  # 16 layers

    # Final classifier
    var fc_weights: ExTensor
    var fc_bias: ExTensor

    fn __init__(out self, num_classes: Int = 10, growth_rate: Int = 32) raises:
        """Initialize DenseNet-121 model.

        Args:
            num_classes: Number of output classes (default: 10 for CIFAR-10)
            growth_rate: Growth rate k (default: 32)
        """
        var num_init_features = 2 * growth_rate  # 64 channels

        # Initial convolution: 3×3, 64 filters
        self.initial_conv_weights = kaiming_normal(
            List[Int]()
            .append(num_init_features)
            .append(3)
            .append(3)
            .append(3),
            fan_in=3 * 9,
        )
        self.initial_conv_bias = zeros(List[Int]().append(num_init_features))
        self.initial_bn_gamma = constant(List[Int]().append(num_init_features), 1.0)
        self.initial_bn_beta = zeros(List[Int]().append(num_init_features))
        self.initial_bn_running_mean = zeros(List[Int]().append(num_init_features))
        self.initial_bn_running_var = constant(List[Int]().append(num_init_features), 1.0)

        # Dense Block 1: 6 layers, 64 → 256 channels
        self.dense_block_1 = DenseBlock(6, num_init_features, growth_rate)
        var num_features_1 = num_init_features + 6 * growth_rate  # 256

        # Transition 1: 256 → 128 channels
        self.transition_1 = TransitionLayer(num_features_1, num_features_1 // 2)
        var num_features_2 = num_features_1 // 2  # 128

        # Dense Block 2: 12 layers, 128 → 512 channels
        self.dense_block_2 = DenseBlock(12, num_features_2, growth_rate)
        var num_features_3 = num_features_2 + 12 * growth_rate  # 512

        # Transition 2: 512 → 256 channels
        self.transition_2 = TransitionLayer(num_features_3, num_features_3 // 2)
        var num_features_4 = num_features_3 // 2  # 256

        # Dense Block 3: 24 layers, 256 → 1024 channels
        self.dense_block_3 = DenseBlock(24, num_features_4, growth_rate)
        var num_features_5 = num_features_4 + 24 * growth_rate  # 1024

        # Transition 3: 1024 → 512 channels
        self.transition_3 = TransitionLayer(num_features_5, num_features_5 // 2)
        var num_features_6 = num_features_5 // 2  # 512

        # Dense Block 4: 16 layers, 512 → 1024 channels
        self.dense_block_4 = DenseBlock(16, num_features_6, growth_rate)
        var num_features_final = num_features_6 + 16 * growth_rate  # 1024

        # Final FC layer
        self.fc_weights = xavier_normal(
            List[Int]().append(num_classes).append(num_features_final),
            fan_in=num_features_final,
            fan_out=num_classes,
        )
        self.fc_bias = zeros(List[Int]().append(num_classes))

    fn forward(mut self, x: ExTensor, training: Bool = True) raises -> ExTensor:
        """Forward pass through DenseNet-121.

        Args:
            x: Input tensor (batch, 3, 32, 32)
            training: Training mode flag

        Returns:
            Logits tensor (batch, num_classes)
        """
        # Initial convolution
        var out = conv2d(x, self.initial_conv_weights, self.initial_conv_bias, stride=1, padding=1)
        out = batch_norm2d(
            out,
            self.initial_bn_gamma,
            self.initial_bn_beta,
            self.initial_bn_running_mean,
            self.initial_bn_running_var,
            training,
        )
        out = relu(out)
        # Shape: (batch, 64, 32, 32)

        # Dense Block 1 + Transition 1
        out = self.dense_block_1.forward(out, training)
        out = self.transition_1.forward(out, training)
        # Shape: (batch, 128, 16, 16)

        # Dense Block 2 + Transition 2
        out = self.dense_block_2.forward(out, training)
        out = self.transition_2.forward(out, training)
        # Shape: (batch, 256, 8, 8)

        # Dense Block 3 + Transition 3
        out = self.dense_block_3.forward(out, training)
        out = self.transition_3.forward(out, training)
        # Shape: (batch, 512, 4, 4)

        # Dense Block 4 (no transition)
        out = self.dense_block_4.forward(out, training)
        # Shape: (batch, 1024, 4, 4)

        # Global average pooling
        out = global_avgpool2d(out)
        # Shape: (batch, 1024, 1, 1)

        # Flatten
        var batch_size = out.shape()[0]
        var channels = out.shape()[1]
        var flattened = zeros(
            List[Int]().append(batch_size).append(channels),
            out.dtype(),
        )
        var flattened_data = flattened._data.bitcast[Float32]()
        var out_data = out._data.bitcast[Float32]()

        for b in range(batch_size):
            for c in range(channels):
                flattened_data[b * channels + c] = out_data[((b * channels + c) * 1) + 0]

        # Final FC layer
        var logits = linear(flattened, self.fc_weights, self.fc_bias)
        # Shape: (batch, num_classes)

        return logits

    fn load_weights(mut self, weights_dir: String) raises:
        """Load model weights from directory."""
        raise Error("Weight loading not yet implemented")

    fn save_weights(self, weights_dir: String) raises:
        """Save model weights to directory."""
        raise Error("Weight saving not yet implemented")
