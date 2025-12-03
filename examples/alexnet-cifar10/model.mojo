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
from shared.core.shape import conv2d_output_shape, pool_output_shape
from shared.training.optimizers import sgd_momentum_update_inplace
from shared.utils.serialization import save_tensor, load_tensor


# ============================================================================
# Architecture Hyperparameters
# ============================================================================
# All architecture dimensions are defined here for easy modification.
# Change these values to experiment with different model sizes.

# Input dimensions (CIFAR-10)
alias INPUT_HEIGHT = 32
alias INPUT_WIDTH = 32
alias INPUT_CHANNELS = 3

# Conv layer 1 hyperparameters
alias CONV1_OUT_CHANNELS = 96
alias CONV1_KERNEL_SIZE = 11
alias CONV1_STRIDE = 4
alias CONV1_PADDING = 2

# Pool layer 1 hyperparameters
alias POOL1_KERNEL_SIZE = 3
alias POOL1_STRIDE = 2
alias POOL1_PADDING = 0

# Conv layer 2 hyperparameters
alias CONV2_OUT_CHANNELS = 256
alias CONV2_KERNEL_SIZE = 5
alias CONV2_STRIDE = 1
alias CONV2_PADDING = 2

# Pool layer 2 hyperparameters
alias POOL2_KERNEL_SIZE = 3
alias POOL2_STRIDE = 2
alias POOL2_PADDING = 0

# Conv layer 3 hyperparameters
alias CONV3_OUT_CHANNELS = 384
alias CONV3_KERNEL_SIZE = 3
alias CONV3_STRIDE = 1
alias CONV3_PADDING = 1

# Conv layer 4 hyperparameters
alias CONV4_OUT_CHANNELS = 384
alias CONV4_KERNEL_SIZE = 3
alias CONV4_STRIDE = 1
alias CONV4_PADDING = 1

# Conv layer 5 hyperparameters
alias CONV5_OUT_CHANNELS = 256
alias CONV5_KERNEL_SIZE = 3
alias CONV5_STRIDE = 1
alias CONV5_PADDING = 1

# Pool layer 3 hyperparameters
alias POOL3_KERNEL_SIZE = 3
alias POOL3_STRIDE = 2
alias POOL3_PADDING = 0

# Fully connected layer sizes
alias FC1_OUT_FEATURES = 4096
alias FC2_OUT_FEATURES = 4096


fn compute_flattened_size() -> Int:
    """Compute the flattened feature size after all conv/pool layers.

    This derives the FC1 input dimension from the architecture hyperparameters.

    Returns:
        Number of features after flattening (channels * height * width)
    """
    # After conv1: Use shared conv2d_output_shape
    var h1, w1 = conv2d_output_shape(
        INPUT_HEIGHT,
        INPUT_WIDTH,
        CONV1_KERNEL_SIZE,
        CONV1_KERNEL_SIZE,
        CONV1_STRIDE,
        CONV1_PADDING,
    )

    # After pool1: Use shared pool_output_shape
    var h2, w2 = pool_output_shape(
        h1, w1, POOL1_KERNEL_SIZE, POOL1_STRIDE, POOL1_PADDING
    )

    # After conv2
    var h3, w3 = conv2d_output_shape(
        h2,
        w2,
        CONV2_KERNEL_SIZE,
        CONV2_KERNEL_SIZE,
        CONV2_STRIDE,
        CONV2_PADDING,
    )

    # After pool2
    var h4, w4 = pool_output_shape(
        h3, w3, POOL2_KERNEL_SIZE, POOL2_STRIDE, POOL2_PADDING
    )

    # After conv3
    var h5, w5 = conv2d_output_shape(
        h4,
        w4,
        CONV3_KERNEL_SIZE,
        CONV3_KERNEL_SIZE,
        CONV3_STRIDE,
        CONV3_PADDING,
    )

    # After conv4
    var h6, w6 = conv2d_output_shape(
        h5,
        w5,
        CONV4_KERNEL_SIZE,
        CONV4_KERNEL_SIZE,
        CONV4_STRIDE,
        CONV4_PADDING,
    )

    # After conv5
    var h7, w7 = conv2d_output_shape(
        h6,
        w6,
        CONV5_KERNEL_SIZE,
        CONV5_KERNEL_SIZE,
        CONV5_STRIDE,
        CONV5_PADDING,
    )

    # After pool3
    var h8, w8 = pool_output_shape(
        h7, w7, POOL3_KERNEL_SIZE, POOL3_STRIDE, POOL3_PADDING
    )

    return CONV5_OUT_CHANNELS * h8 * w8


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

    fn __init__(
        out self, num_classes: Int = 10, dropout_rate: Float32 = 0.5
    ) raises:
        """Initialize AlexNet model with random weights.

        Args:
            num_classes: Number of output classes (default: 10 for CIFAR-10)
            dropout_rate: Dropout probability for FC layers (default: 0.5)
        """
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # Compute derived dimensions using shared shape utilities
        var flattened_size = compute_flattened_size()

        # Conv1: INPUT_CHANNELS -> CONV1_OUT_CHANNELS
        var conv1_shape = List[Int](
            CONV1_OUT_CHANNELS,
            INPUT_CHANNELS,
            CONV1_KERNEL_SIZE,
            CONV1_KERNEL_SIZE,
        )
        self.conv1_kernel = he_uniform(conv1_shape, DType.float32)
        self.conv1_bias = zeros(List[Int](CONV1_OUT_CHANNELS), DType.float32)

        # Conv2: CONV1_OUT_CHANNELS -> CONV2_OUT_CHANNELS
        var conv2_shape = List[Int](
            CONV2_OUT_CHANNELS,
            CONV1_OUT_CHANNELS,
            CONV2_KERNEL_SIZE,
            CONV2_KERNEL_SIZE,
        )
        self.conv2_kernel = he_uniform(conv2_shape, DType.float32)
        self.conv2_bias = zeros(List[Int](CONV2_OUT_CHANNELS), DType.float32)

        # Conv3: CONV2_OUT_CHANNELS -> CONV3_OUT_CHANNELS
        var conv3_shape = List[Int](
            CONV3_OUT_CHANNELS,
            CONV2_OUT_CHANNELS,
            CONV3_KERNEL_SIZE,
            CONV3_KERNEL_SIZE,
        )
        self.conv3_kernel = he_uniform(conv3_shape, DType.float32)
        self.conv3_bias = zeros(List[Int](CONV3_OUT_CHANNELS), DType.float32)

        # Conv4: CONV3_OUT_CHANNELS -> CONV4_OUT_CHANNELS
        var conv4_shape = List[Int](
            CONV4_OUT_CHANNELS,
            CONV3_OUT_CHANNELS,
            CONV4_KERNEL_SIZE,
            CONV4_KERNEL_SIZE,
        )
        self.conv4_kernel = he_uniform(conv4_shape, DType.float32)
        self.conv4_bias = zeros(List[Int](CONV4_OUT_CHANNELS), DType.float32)

        # Conv5: CONV4_OUT_CHANNELS -> CONV5_OUT_CHANNELS
        var conv5_shape = List[Int](
            CONV5_OUT_CHANNELS,
            CONV4_OUT_CHANNELS,
            CONV5_KERNEL_SIZE,
            CONV5_KERNEL_SIZE,
        )
        self.conv5_kernel = he_uniform(conv5_shape, DType.float32)
        self.conv5_bias = zeros(List[Int](CONV5_OUT_CHANNELS), DType.float32)

        # FC1: flattened_size -> FC1_OUT_FEATURES (derived from conv/pool layers)
        var fc1_shape = List[Int](FC1_OUT_FEATURES, flattened_size)
        self.fc1_weights = xavier_uniform(fc1_shape, DType.float32)
        self.fc1_bias = zeros(List[Int](FC1_OUT_FEATURES), DType.float32)

        # FC2: FC1_OUT_FEATURES -> FC2_OUT_FEATURES
        var fc2_shape = List[Int](FC2_OUT_FEATURES, FC1_OUT_FEATURES)
        self.fc2_weights = xavier_uniform(fc2_shape, DType.float32)
        self.fc2_bias = zeros(List[Int](FC2_OUT_FEATURES), DType.float32)

        # FC3: FC2_OUT_FEATURES -> num_classes
        var fc3_shape = List[Int](num_classes, FC2_OUT_FEATURES)
        self.fc3_weights = xavier_uniform(fc3_shape, DType.float32)
        self.fc3_bias = zeros(List[Int](num_classes), DType.float32)

    fn forward(
        mut self, input: ExTensor, training: Bool = True
    ) raises -> ExTensor:
        """Forward pass through AlexNet.

        Args:
            input: Input tensor of shape (batch, 3, 32, 32)
            training: Whether in training mode (applies dropout if True)

        Returns:
            Output logits of shape (batch, num_classes)
        """
        # Conv1 + ReLU + MaxPool
        var conv1_out = conv2d(
            input,
            self.conv1_kernel,
            self.conv1_bias,
            stride=CONV1_STRIDE,
            padding=CONV1_PADDING,
        )
        var relu1_out = relu(conv1_out)
        var pool1_out = maxpool2d(
            relu1_out,
            kernel_size=POOL1_KERNEL_SIZE,
            stride=POOL1_STRIDE,
            padding=POOL1_PADDING,
        )

        # Conv2 + ReLU + MaxPool
        var conv2_out = conv2d(
            pool1_out,
            self.conv2_kernel,
            self.conv2_bias,
            stride=CONV2_STRIDE,
            padding=CONV2_PADDING,
        )
        var relu2_out = relu(conv2_out)
        var pool2_out = maxpool2d(
            relu2_out,
            kernel_size=POOL2_KERNEL_SIZE,
            stride=POOL2_STRIDE,
            padding=POOL2_PADDING,
        )

        # Conv3 + ReLU
        var conv3_out = conv2d(
            pool2_out,
            self.conv3_kernel,
            self.conv3_bias,
            stride=CONV3_STRIDE,
            padding=CONV3_PADDING,
        )
        var relu3_out = relu(conv3_out)

        # Conv4 + ReLU
        var conv4_out = conv2d(
            relu3_out,
            self.conv4_kernel,
            self.conv4_bias,
            stride=CONV4_STRIDE,
            padding=CONV4_PADDING,
        )
        var relu4_out = relu(conv4_out)

        # Conv5 + ReLU + MaxPool
        var conv5_out = conv2d(
            relu4_out,
            self.conv5_kernel,
            self.conv5_bias,
            stride=CONV5_STRIDE,
            padding=CONV5_PADDING,
        )
        var relu5_out = relu(conv5_out)
        var pool3_out = maxpool2d(
            relu5_out,
            kernel_size=POOL3_KERNEL_SIZE,
            stride=POOL3_STRIDE,
            padding=POOL3_PADDING,
        )

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
        var drop1_result = dropout(
            relu6_out, Float64(self.dropout_rate), training
        )
        var drop1_out = drop1_result[0]

        # FC2 + ReLU + Dropout
        var fc2_out = linear(drop1_out, self.fc2_weights, self.fc2_bias)
        var relu7_out = relu(fc2_out)
        var drop2_result = dropout(
            relu7_out, Float64(self.dropout_rate), training
        )
        var drop2_out = drop2_result[0]

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
        save_tensor(
            self.conv1_kernel,
            weights_dir + "/conv1_kernel.weights",
            "conv1_kernel",
        )
        save_tensor(
            self.conv1_bias, weights_dir + "/conv1_bias.weights", "conv1_bias"
        )
        save_tensor(
            self.conv2_kernel,
            weights_dir + "/conv2_kernel.weights",
            "conv2_kernel",
        )
        save_tensor(
            self.conv2_bias, weights_dir + "/conv2_bias.weights", "conv2_bias"
        )
        save_tensor(
            self.conv3_kernel,
            weights_dir + "/conv3_kernel.weights",
            "conv3_kernel",
        )
        save_tensor(
            self.conv3_bias, weights_dir + "/conv3_bias.weights", "conv3_bias"
        )
        save_tensor(
            self.conv4_kernel,
            weights_dir + "/conv4_kernel.weights",
            "conv4_kernel",
        )
        save_tensor(
            self.conv4_bias, weights_dir + "/conv4_bias.weights", "conv4_bias"
        )
        save_tensor(
            self.conv5_kernel,
            weights_dir + "/conv5_kernel.weights",
            "conv5_kernel",
        )
        save_tensor(
            self.conv5_bias, weights_dir + "/conv5_bias.weights", "conv5_bias"
        )
        save_tensor(
            self.fc1_weights,
            weights_dir + "/fc1_weights.weights",
            "fc1_weights",
        )
        save_tensor(
            self.fc1_bias, weights_dir + "/fc1_bias.weights", "fc1_bias"
        )
        save_tensor(
            self.fc2_weights,
            weights_dir + "/fc2_weights.weights",
            "fc2_weights",
        )
        save_tensor(
            self.fc2_bias, weights_dir + "/fc2_bias.weights", "fc2_bias"
        )
        save_tensor(
            self.fc3_weights,
            weights_dir + "/fc3_weights.weights",
            "fc3_weights",
        )
        save_tensor(
            self.fc3_bias, weights_dir + "/fc3_bias.weights", "fc3_bias"
        )

    fn load_weights(mut self, weights_dir: String) raises:
        """Load model weights from directory.

        Args:
            weights_dir: Directory containing weight files.

        Raises:
            Error: If weight files are missing or have incompatible shapes.
        """
        # Load each parameter from its file
        self.conv1_kernel = load_tensor(weights_dir + "/conv1_kernel.weights")
        self.conv1_bias = load_tensor(weights_dir + "/conv1_bias.weights")
        self.conv2_kernel = load_tensor(weights_dir + "/conv2_kernel.weights")
        self.conv2_bias = load_tensor(weights_dir + "/conv2_bias.weights")
        self.conv3_kernel = load_tensor(weights_dir + "/conv3_kernel.weights")
        self.conv3_bias = load_tensor(weights_dir + "/conv3_bias.weights")
        self.conv4_kernel = load_tensor(weights_dir + "/conv4_kernel.weights")
        self.conv4_bias = load_tensor(weights_dir + "/conv4_bias.weights")
        self.conv5_kernel = load_tensor(weights_dir + "/conv5_kernel.weights")
        self.conv5_bias = load_tensor(weights_dir + "/conv5_bias.weights")
        self.fc1_weights = load_tensor(weights_dir + "/fc1_weights.weights")
        self.fc1_bias = load_tensor(weights_dir + "/fc1_bias.weights")
        self.fc2_weights = load_tensor(weights_dir + "/fc2_weights.weights")
        self.fc2_bias = load_tensor(weights_dir + "/fc2_bias.weights")
        self.fc3_weights = load_tensor(weights_dir + "/fc3_weights.weights")
        self.fc3_bias = load_tensor(weights_dir + "/fc3_bias.weights")

    fn update_parameters(
        mut self,
        learning_rate: Float32,
        momentum: Float32,
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
        mut velocity_conv1_kernel: ExTensor,
        mut velocity_conv1_bias: ExTensor,
        mut velocity_conv2_kernel: ExTensor,
        mut velocity_conv2_bias: ExTensor,
        mut velocity_conv3_kernel: ExTensor,
        mut velocity_conv3_bias: ExTensor,
        mut velocity_conv4_kernel: ExTensor,
        mut velocity_conv4_bias: ExTensor,
        mut velocity_conv5_kernel: ExTensor,
        mut velocity_conv5_bias: ExTensor,
        mut velocity_fc1_weights: ExTensor,
        mut velocity_fc1_bias: ExTensor,
        mut velocity_fc2_weights: ExTensor,
        mut velocity_fc2_bias: ExTensor,
        mut velocity_fc3_weights: ExTensor,
        mut velocity_fc3_bias: ExTensor,
    ) raises:
        """Update parameters using SGD with momentum.

        Args:
            learning_rate: Learning rate for gradient descent
            momentum: Momentum factor (typically 0.9)
            grad_*: Gradients for each parameter
            velocity_*: Velocity (momentum) for each parameter
        """
        # SGD with momentum update: v = momentum * v - lr * grad; param = param + v
        # Convert Float32 parameters to Float64 for shared library
        var lr = Float64(learning_rate)
        var mom = Float64(momentum)
        sgd_momentum_update_inplace(
            self.conv1_kernel, grad_conv1_kernel, velocity_conv1_kernel, lr, mom
        )
        sgd_momentum_update_inplace(
            self.conv1_bias, grad_conv1_bias, velocity_conv1_bias, lr, mom
        )
        sgd_momentum_update_inplace(
            self.conv2_kernel, grad_conv2_kernel, velocity_conv2_kernel, lr, mom
        )
        sgd_momentum_update_inplace(
            self.conv2_bias, grad_conv2_bias, velocity_conv2_bias, lr, mom
        )
        sgd_momentum_update_inplace(
            self.conv3_kernel, grad_conv3_kernel, velocity_conv3_kernel, lr, mom
        )
        sgd_momentum_update_inplace(
            self.conv3_bias, grad_conv3_bias, velocity_conv3_bias, lr, mom
        )
        sgd_momentum_update_inplace(
            self.conv4_kernel, grad_conv4_kernel, velocity_conv4_kernel, lr, mom
        )
        sgd_momentum_update_inplace(
            self.conv4_bias, grad_conv4_bias, velocity_conv4_bias, lr, mom
        )
        sgd_momentum_update_inplace(
            self.conv5_kernel, grad_conv5_kernel, velocity_conv5_kernel, lr, mom
        )
        sgd_momentum_update_inplace(
            self.conv5_bias, grad_conv5_bias, velocity_conv5_bias, lr, mom
        )
        sgd_momentum_update_inplace(
            self.fc1_weights, grad_fc1_weights, velocity_fc1_weights, lr, mom
        )
        sgd_momentum_update_inplace(
            self.fc1_bias, grad_fc1_bias, velocity_fc1_bias, lr, mom
        )
        sgd_momentum_update_inplace(
            self.fc2_weights, grad_fc2_weights, velocity_fc2_weights, lr, mom
        )
        sgd_momentum_update_inplace(
            self.fc2_bias, grad_fc2_bias, velocity_fc2_bias, lr, mom
        )
        sgd_momentum_update_inplace(
            self.fc3_weights, grad_fc3_weights, velocity_fc3_weights, lr, mom
        )
        sgd_momentum_update_inplace(
            self.fc3_bias, grad_fc3_bias, velocity_fc3_bias, lr, mom
        )
