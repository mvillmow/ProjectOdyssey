"""AlexNet Model for CIFAR-10 Classification

Classic AlexNet architecture adapted for CIFAR-10 dataset (32x32 RGB images).

Architecture:
    Input (32x32x3) ->
    Conv2D(96 filters, 11x11, stride=4, padding=2) -> ReLU -> MaxPool(3x3, stride=2) ->
    Conv2D(256 filters, 5x5, padding=2) -> ReLU -> MaxPool(3x3, stride=2) ->
    Conv2D(384 filters, 3x3, padding=1) -> ReLU ->
    Conv2D(384 filters, 3x3, padding=1) -> ReLU ->
    Conv2D(256 filters, 3x3, padding=1) -> ReLU -> MaxPool(3x3, stride=2) ->
    Flatten ->
    Linear(4096) -> ReLU -> Dropout(0.5) ->
    Linear(4096) -> ReLU -> Dropout(0.5) ->
    Linear(num_classes)

References:
    - Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012).
      ImageNet classification with deep convolutional neural networks.
      Advances in Neural Information Processing Systems, 25, 1097-1105.
    - CIFAR-10 Dataset: https://www.cs.toronto.edu/~kriz/cifar.html
"""

from shared.core import ExTensor, zeros
from shared.core.conv import conv2d, conv2d_backward
from shared.core.pooling import maxpool2d, maxpool2d_backward
from shared.core.linear import linear, linear_backward
from shared.core.activation import relu, relu_backward
from shared.core.dropout import dropout, dropout_backward
from shared.core.initializers import he_uniform, xavier_uniform
from shared.training.optimizers import sgd_momentum_update_inplace
from weights import save_tensor, load_tensor


struct AlexNet:
    """AlexNet model for CIFAR-10 classification.

    Attributes:
        num_classes: Number of output classes (10 for CIFAR-10)
        dropout_rate: Dropout probability for FC layers (default 0.5)
        conv1_kernel: First conv layer weights (96, 3, 11, 11)
        conv1_bias: First conv layer bias (96,)
        conv2_kernel: Second conv layer weights (256, 96, 5, 5)
        conv2_bias: Second conv layer bias (256,)
        conv3_kernel: Third conv layer weights (384, 256, 3, 3)
        conv3_bias: Third conv layer bias (384,)
        conv4_kernel: Fourth conv layer weights (384, 384, 3, 3)
        conv4_bias: Fourth conv layer bias (384,)
        conv5_kernel: Fifth conv layer weights (256, 384, 3, 3)
        conv5_bias: Fifth conv layer bias (256,)
        fc1_weights: First FC layer weights (4096, 256)
        fc1_bias: First FC layer bias (4096,)
        fc2_weights: Second FC layer weights (4096, 4096)
        fc2_bias: Second FC layer bias (4096,)
        fc3_weights: Third FC layer weights (num_classes, 4096)
        fc3_bias: Third FC layer bias (num_classes,)
    """

    var num_classes: Int
    var dropout_rate: Float32

    # Convolutional layer parameters
    var conv1_kernel: ExTensor
    var conv1_bias: ExTensor
    var conv2_kernel: ExTensor
    var conv2_bias: ExTensor
    var conv3_kernel: ExTensor
    var conv3_bias: ExTensor
    var conv4_kernel: ExTensor
    var conv4_bias: ExTensor
    var conv5_kernel: ExTensor
    var conv5_bias: ExTensor

    # Fully connected layer parameters
    var fc1_weights: ExTensor
    var fc1_bias: ExTensor
    var fc2_weights: ExTensor
    var fc2_bias: ExTensor
    var fc3_weights: ExTensor
    var fc3_bias: ExTensor

    fn __init__(out self, num_classes: Int = 10, dropout_rate: Float32 = 0.5) raises:
        """Initialize AlexNet model with random weights.

        Args:
            num_classes: Number of output classes (default: 10 for CIFAR-10)
            dropout_rate: Dropout probability for FC layers (default: 0.5)
        """
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # Conv1: 3 input channels (RGB), 96 output channels, 11x11 kernel
        var conv1_shape = List[Int]()
        conv1_shape.append(96)  # out_channels
        conv1_shape.append(3)   # in_channels (RGB)
        conv1_shape.append(11)  # kernel_height
        conv1_shape.append(11)  # kernel_width
        self.conv1_kernel = he_uniform(conv1_shape, DType.float32)
        self.conv1_bias = zeros(List[Int]().append(96), DType.float32)

        # Conv2: 96 input channels, 256 output channels, 5x5 kernel
        var conv2_shape = List[Int]()
        conv2_shape.append(256)  # out_channels
        conv2_shape.append(96)   # in_channels
        conv2_shape.append(5)    # kernel_height
        conv2_shape.append(5)    # kernel_width
        self.conv2_kernel = he_uniform(conv2_shape, DType.float32)
        self.conv2_bias = zeros(List[Int]().append(256), DType.float32)

        # Conv3: 256 input channels, 384 output channels, 3x3 kernel
        var conv3_shape = List[Int]()
        conv3_shape.append(384)  # out_channels
        conv3_shape.append(256)  # in_channels
        conv3_shape.append(3)    # kernel_height
        conv3_shape.append(3)    # kernel_width
        self.conv3_kernel = he_uniform(conv3_shape, DType.float32)
        self.conv3_bias = zeros(List[Int]().append(384), DType.float32)

        # Conv4: 384 input channels, 384 output channels, 3x3 kernel
        var conv4_shape = List[Int]()
        conv4_shape.append(384)  # out_channels
        conv4_shape.append(384)  # in_channels
        conv4_shape.append(3)    # kernel_height
        conv4_shape.append(3)    # kernel_width
        self.conv4_kernel = he_uniform(conv4_shape, DType.float32)
        self.conv4_bias = zeros(List[Int]().append(384), DType.float32)

        # Conv5: 384 input channels, 256 output channels, 3x3 kernel
        var conv5_shape = List[Int]()
        conv5_shape.append(256)  # out_channels
        conv5_shape.append(384)  # in_channels
        conv5_shape.append(3)    # kernel_height
        conv5_shape.append(3)    # kernel_width
        self.conv5_kernel = he_uniform(conv5_shape, DType.float32)
        self.conv5_bias = zeros(List[Int]().append(256), DType.float32)

        # After conv1 (32x32 -> 5x5 with stride=4, padding=2) -> pool1 (5x5 -> 2x2 with stride=2)
        # After conv2 (2x2 -> 2x2 with padding=2) -> pool2 (2x2 -> 1x1 with stride=2)
        # After conv3, conv4, conv5 (1x1 -> 1x1) -> pool3 (1x1 -> 1x1)
        # Flattened size: 256 * 1 * 1 = 256

        # FC1: 256 -> 4096
        var fc1_shape = List[Int]()
        fc1_shape.append(4096)  # out_features
        fc1_shape.append(256)   # in_features
        self.fc1_weights = xavier_uniform(fc1_shape, DType.float32)
        self.fc1_bias = zeros(List[Int]().append(4096), DType.float32)

        # FC2: 4096 -> 4096
        var fc2_shape = List[Int]()
        fc2_shape.append(4096)  # out_features
        fc2_shape.append(4096)  # in_features
        self.fc2_weights = xavier_uniform(fc2_shape, DType.float32)
        self.fc2_bias = zeros(List[Int]().append(4096), DType.float32)

        # FC3: 4096 -> num_classes
        var fc3_shape = List[Int]()
        fc3_shape.append(num_classes)  # out_features
        fc3_shape.append(4096)         # in_features
        self.fc3_weights = xavier_uniform(fc3_shape, DType.float32)
        self.fc3_bias = zeros(List[Int]().append(num_classes), DType.float32)

    fn forward(mut self, input: ExTensor, training: Bool = True) raises -> ExTensor:
        """Forward pass through AlexNet.

        Args:
            input: Input tensor of shape (batch, 3, 32, 32)
            training: Whether in training mode (applies dropout if True)

        Returns:
            Output logits of shape (batch, num_classes)
        """
        # Conv1 + ReLU + MaxPool
        var conv1_out = conv2d(input, self.conv1_kernel, self.conv1_bias, stride=4, padding=2)
        var relu1_out = relu(conv1_out)
        var pool1_out = maxpool2d(relu1_out, kernel_size=3, stride=2, padding=0)

        # Conv2 + ReLU + MaxPool
        var conv2_out = conv2d(pool1_out, self.conv2_kernel, self.conv2_bias, stride=1, padding=2)
        var relu2_out = relu(conv2_out)
        var pool2_out = maxpool2d(relu2_out, kernel_size=3, stride=2, padding=0)

        # Conv3 + ReLU
        var conv3_out = conv2d(pool2_out, self.conv3_kernel, self.conv3_bias, stride=1, padding=1)
        var relu3_out = relu(conv3_out)

        # Conv4 + ReLU
        var conv4_out = conv2d(relu3_out, self.conv4_kernel, self.conv4_bias, stride=1, padding=1)
        var relu4_out = relu(conv4_out)

        # Conv5 + ReLU + MaxPool
        var conv5_out = conv2d(relu4_out, self.conv5_kernel, self.conv5_bias, stride=1, padding=1)
        var relu5_out = relu(conv5_out)
        var pool3_out = maxpool2d(relu5_out, kernel_size=3, stride=2, padding=0)

        # Flatten: (batch, 256, 1, 1) -> (batch, 256)
        var pool3_shape = pool3_out.shape()
        var batch_size = pool3_shape[0]
        var flattened_size = pool3_shape[1] * pool3_shape[2] * pool3_shape[3]

        var flatten_shape = List[Int]()
        flatten_shape.append(batch_size)
        flatten_shape.append(flattened_size)
        var flattened = pool3_out.reshape(flatten_shape)

        # FC1 + ReLU + Dropout
        var fc1_out = linear(flattened, self.fc1_weights, self.fc1_bias)
        var relu6_out = relu(fc1_out)
        var drop1_out = relu6_out  # Will be replaced by dropout in training
        if training:
            drop1_out = dropout(relu6_out, self.dropout_rate)

        # FC2 + ReLU + Dropout
        var fc2_out = linear(drop1_out, self.fc2_weights, self.fc2_bias)
        var relu7_out = relu(fc2_out)
        var drop2_out = relu7_out  # Will be replaced by dropout in training
        if training:
            drop2_out = dropout(relu7_out, self.dropout_rate)

        # FC3 (output logits)
        var output = linear(drop2_out, self.fc3_weights, self.fc3_bias)

        return output

    fn predict(mut self, input: ExTensor) raises -> Int:
        """Predict class for a single input.

        Args:
            input: Input tensor of shape (1, 3, 32, 32)

        Returns:
            Predicted class index (0 to num_classes-1)
        """
        var logits = self.forward(input, training=False)

        # Find argmax
        var logits_shape = logits.shape()
        var max_idx = 0
        var max_val = logits[0]

        for i in range(1, logits_shape[1]):
            if logits[i] > max_val:
                max_val = logits[i]
                max_idx = i

        return max_idx

    fn save_weights(self, weights_dir: String) raises:
        """Save model weights to directory.

        Args:
            weights_dir: Directory to save weight files (one file per parameter)

        Note:
            Creates directory if it doesn't exist. Each parameter saved as:
            - conv1_kernel.weights, conv1_bias.weights, etc.
        """
        # Save each parameter to its own file
        save_tensor(self.conv1_kernel, "conv1_kernel", weights_dir + "/conv1_kernel.weights")
        save_tensor(self.conv1_bias, "conv1_bias", weights_dir + "/conv1_bias.weights")
        save_tensor(self.conv2_kernel, "conv2_kernel", weights_dir + "/conv2_kernel.weights")
        save_tensor(self.conv2_bias, "conv2_bias", weights_dir + "/conv2_bias.weights")
        save_tensor(self.conv3_kernel, "conv3_kernel", weights_dir + "/conv3_kernel.weights")
        save_tensor(self.conv3_bias, "conv3_bias", weights_dir + "/conv3_bias.weights")
        save_tensor(self.conv4_kernel, "conv4_kernel", weights_dir + "/conv4_kernel.weights")
        save_tensor(self.conv4_bias, "conv4_bias", weights_dir + "/conv4_bias.weights")
        save_tensor(self.conv5_kernel, "conv5_kernel", weights_dir + "/conv5_kernel.weights")
        save_tensor(self.conv5_bias, "conv5_bias", weights_dir + "/conv5_bias.weights")
        save_tensor(self.fc1_weights, "fc1_weights", weights_dir + "/fc1_weights.weights")
        save_tensor(self.fc1_bias, "fc1_bias", weights_dir + "/fc1_bias.weights")
        save_tensor(self.fc2_weights, "fc2_weights", weights_dir + "/fc2_weights.weights")
        save_tensor(self.fc2_bias, "fc2_bias", weights_dir + "/fc2_bias.weights")
        save_tensor(self.fc3_weights, "fc3_weights", weights_dir + "/fc3_weights.weights")
        save_tensor(self.fc3_bias, "fc3_bias", weights_dir + "/fc3_bias.weights")

    fn load_weights(mut self, weights_dir: String) raises:
        """Load model weights from directory.

        Args:
            weights_dir: Directory containing weight files

        Raises:
            Error: If weight files are missing or have incompatible shapes
        """
        # Load each parameter from its file
        var result1 = load_tensor(weights_dir + "/conv1_kernel.weights")
        self.conv1_kernel = result1[1]^

        var result2 = load_tensor(weights_dir + "/conv1_bias.weights")
        self.conv1_bias = result2[1]^

        var result3 = load_tensor(weights_dir + "/conv2_kernel.weights")
        self.conv2_kernel = result3[1]^

        var result4 = load_tensor(weights_dir + "/conv2_bias.weights")
        self.conv2_bias = result4[1]^

        var result5 = load_tensor(weights_dir + "/conv3_kernel.weights")
        self.conv3_kernel = result5[1]^

        var result6 = load_tensor(weights_dir + "/conv3_bias.weights")
        self.conv3_bias = result6[1]^

        var result7 = load_tensor(weights_dir + "/conv4_kernel.weights")
        self.conv4_kernel = result7[1]^

        var result8 = load_tensor(weights_dir + "/conv4_bias.weights")
        self.conv4_bias = result8[1]^

        var result9 = load_tensor(weights_dir + "/conv5_kernel.weights")
        self.conv5_kernel = result9[1]^

        var result10 = load_tensor(weights_dir + "/conv5_bias.weights")
        self.conv5_bias = result10[1]^

        var result11 = load_tensor(weights_dir + "/fc1_weights.weights")
        self.fc1_weights = result11[1]^

        var result12 = load_tensor(weights_dir + "/fc1_bias.weights")
        self.fc1_bias = result12[1]^

        var result13 = load_tensor(weights_dir + "/fc2_weights.weights")
        self.fc2_weights = result13[1]^

        var result14 = load_tensor(weights_dir + "/fc2_bias.weights")
        self.fc2_bias = result14[1]^

        var result15 = load_tensor(weights_dir + "/fc3_weights.weights")
        self.fc3_weights = result15[1]^

        var result16 = load_tensor(weights_dir + "/fc3_bias.weights")
        self.fc3_bias = result16[1]^

    fn update_parameters(mut self, learning_rate: Float32, momentum: Float32,
                        grad_conv1_kernel: ExTensor,
                        grad_conv1_bias: ExTensor,
                        grad_conv2_kernel: ExTensor,
                        grad_conv2_bias: ExTensor,
                        grad_conv3_kernel: ExTensor,
                        grad_conv3_bias: ExTensor,
                        grad_conv4_kernel: ExTensor,
                        grad_conv4_bias: ExTensor,
                        grad_conv5_kernel: ExTensor,
                        grad_conv5_bias: ExTensor,
                        grad_fc1_weights: ExTensor,
                        grad_fc1_bias: ExTensor,
                        grad_fc2_weights: ExTensor,
                        grad_fc2_bias: ExTensor,
                        grad_fc3_weights: ExTensor,
                        grad_fc3_bias: ExTensor,
                        inout velocity_conv1_kernel: ExTensor,
                        inout velocity_conv1_bias: ExTensor,
                        inout velocity_conv2_kernel: ExTensor,
                        inout velocity_conv2_bias: ExTensor,
                        inout velocity_conv3_kernel: ExTensor,
                        inout velocity_conv3_bias: ExTensor,
                        inout velocity_conv4_kernel: ExTensor,
                        inout velocity_conv4_bias: ExTensor,
                        inout velocity_conv5_kernel: ExTensor,
                        inout velocity_conv5_bias: ExTensor,
                        inout velocity_fc1_weights: ExTensor,
                        inout velocity_fc1_bias: ExTensor,
                        inout velocity_fc2_weights: ExTensor,
                        inout velocity_fc2_bias: ExTensor,
                        inout velocity_fc3_weights: ExTensor,
                        inout velocity_fc3_bias: ExTensor) raises:
        """Update parameters using SGD with momentum.

        Args:
            learning_rate: Learning rate for gradient descent
            momentum: Momentum factor (typically 0.9)
            grad_*: Gradients for each parameter
            velocity_*: Velocity (momentum) for each parameter
        """
        # SGD with momentum update: v = momentum * v - lr * grad; param = param + v
        # Now uses shared library implementation
        sgd_momentum_update_inplace(self.conv1_kernel, grad_conv1_kernel, velocity_conv1_kernel, learning_rate, momentum)
        sgd_momentum_update_inplace(self.conv1_bias, grad_conv1_bias, velocity_conv1_bias, learning_rate, momentum)
        sgd_momentum_update_inplace(self.conv2_kernel, grad_conv2_kernel, velocity_conv2_kernel, learning_rate, momentum)
        sgd_momentum_update_inplace(self.conv2_bias, grad_conv2_bias, velocity_conv2_bias, learning_rate, momentum)
        sgd_momentum_update_inplace(self.conv3_kernel, grad_conv3_kernel, velocity_conv3_kernel, learning_rate, momentum)
        sgd_momentum_update_inplace(self.conv3_bias, grad_conv3_bias, velocity_conv3_bias, learning_rate, momentum)
        sgd_momentum_update_inplace(self.conv4_kernel, grad_conv4_kernel, velocity_conv4_kernel, learning_rate, momentum)
        sgd_momentum_update_inplace(self.conv4_bias, grad_conv4_bias, velocity_conv4_bias, learning_rate, momentum)
        sgd_momentum_update_inplace(self.conv5_kernel, grad_conv5_kernel, velocity_conv5_kernel, learning_rate, momentum)
        sgd_momentum_update_inplace(self.conv5_bias, grad_conv5_bias, velocity_conv5_bias, learning_rate, momentum)
        sgd_momentum_update_inplace(self.fc1_weights, grad_fc1_weights, velocity_fc1_weights, learning_rate, momentum)
        sgd_momentum_update_inplace(self.fc1_bias, grad_fc1_bias, velocity_fc1_bias, learning_rate, momentum)
        sgd_momentum_update_inplace(self.fc2_weights, grad_fc2_weights, velocity_fc2_weights, learning_rate, momentum)
        sgd_momentum_update_inplace(self.fc2_bias, grad_fc2_bias, velocity_fc2_bias, learning_rate, momentum)
        sgd_momentum_update_inplace(self.fc3_weights, grad_fc3_weights, velocity_fc3_weights, learning_rate, momentum)
        sgd_momentum_update_inplace(self.fc3_bias, grad_fc3_bias, velocity_fc3_bias, learning_rate, momentum)
