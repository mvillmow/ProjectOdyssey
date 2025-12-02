"""LeNet-5 Model for EMNIST Classification

Classic LeNet-5 architecture adapted for EMNIST dataset (28x28 grayscale images).

Architecture:
    Input (28x28x1) ->
    Conv2D(6 filters, 5x5) -> ReLU -> MaxPool(2x2, stride=2) ->
    Conv2D(16 filters, 5x5) -> ReLU -> MaxPool(2x2, stride=2) ->
    Flatten ->
    Linear(120) -> ReLU ->
    Linear(84) -> ReLU ->
    Linear(num_classes)

References:
    - LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998).
      Gradient-based learning applied to document recognition.
      Proceedings of the IEEE, 86(11), 2278-2324.
    - EMNIST Dataset: https://www.nist.gov/itl/products-and-services/emnist-dataset
    - Reference Implementation: https://github.com/mattwang44/LeNet-from-Scratch
"""

from shared.core import ExTensor, zeros
from shared.core.conv import conv2d, conv2d_backward
from shared.core.pooling import maxpool2d, maxpool2d_backward
from shared.core.linear import linear, linear_backward
from shared.core.activation import relu, relu_backward
from shared.core.initializers import kaiming_uniform, xavier_uniform
from shared.core.shape import conv2d_output_shape, pool_output_shape
from shared.core.traits import Model
from collections import List
from weights import save_tensor, load_tensor


# ============================================================================
# Architecture Hyperparameters
# ============================================================================
# All architecture dimensions are defined here for easy modification.
# Change these values to experiment with different model sizes.

# Input dimensions
alias INPUT_HEIGHT = 28
alias INPUT_WIDTH = 28
alias INPUT_CHANNELS = 1

# Conv layer 1 hyperparameters
alias CONV1_OUT_CHANNELS = 6
alias CONV1_KERNEL_SIZE = 5
alias CONV1_STRIDE = 1
alias CONV1_PADDING = 0

# Pool layer 1 hyperparameters
alias POOL1_KERNEL_SIZE = 2
alias POOL1_STRIDE = 2
alias POOL1_PADDING = 0

# Conv layer 2 hyperparameters
alias CONV2_OUT_CHANNELS = 16
alias CONV2_KERNEL_SIZE = 5
alias CONV2_STRIDE = 1
alias CONV2_PADDING = 0

# Pool layer 2 hyperparameters
alias POOL2_KERNEL_SIZE = 2
alias POOL2_STRIDE = 2
alias POOL2_PADDING = 0

# Fully connected layer sizes
alias FC1_OUT_FEATURES = 120
alias FC2_OUT_FEATURES = 84


fn compute_flattened_size() -> Int:
    """Compute the flattened feature size after all conv/pool layers.

    This derives the FC1 input dimension from the architecture hyperparameters.

    Returns:
        Number of features after flattening (channels * height * width)
    """
    # After conv1: Use shared conv2d_output_shape
    var h1, w1 = conv2d_output_shape(INPUT_HEIGHT, INPUT_WIDTH, CONV1_KERNEL_SIZE, CONV1_KERNEL_SIZE,
                                      CONV1_STRIDE, CONV1_PADDING)

    # After pool1: Use shared pool_output_shape
    var h2, w2 = pool_output_shape(h1, w1, POOL1_KERNEL_SIZE, POOL1_STRIDE, POOL1_PADDING)

    # After conv2
    var h3, w3 = conv2d_output_shape(h2, w2, CONV2_KERNEL_SIZE, CONV2_KERNEL_SIZE,
                                      CONV2_STRIDE, CONV2_PADDING)

    # After pool2
    var h4, w4 = pool_output_shape(h3, w3, POOL2_KERNEL_SIZE, POOL2_STRIDE, POOL2_PADDING)

    return CONV2_OUT_CHANNELS * h4 * w4


struct LeNet5(Model, Movable):
    """LeNet-5 model for EMNIST classification.

    Architecture is defined by module-level constants (CONV1_*, POOL1_*, etc.)
    for easy experimentation. FC layer input sizes are automatically derived
    from the conv/pool layer dimensions.

    Attributes:
        num_classes: Number of output classes (47 for EMNIST Balanced)
        conv1_kernel: First conv layer weights (CONV1_OUT_CHANNELS, INPUT_CHANNELS, k, k)
        conv1_bias: First conv layer bias (CONV1_OUT_CHANNELS,)
        conv2_kernel: Second conv layer weights (CONV2_OUT_CHANNELS, CONV1_OUT_CHANNELS, k, k)
        conv2_bias: Second conv layer bias (CONV2_OUT_CHANNELS,)
        fc1_weights: First FC layer weights (FC1_OUT_FEATURES, flattened_size)
        fc1_bias: First FC layer bias (FC1_OUT_FEATURES,)
        fc2_weights: Second FC layer weights (FC2_OUT_FEATURES, FC1_OUT_FEATURES)
        fc2_bias: Second FC layer bias (FC2_OUT_FEATURES,)
        fc3_weights: Third FC layer weights (num_classes, FC2_OUT_FEATURES)
        fc3_bias: Third FC layer bias (num_classes,)
    """

    var num_classes: Int

    # Layer parameters
    var conv1_kernel: ExTensor
    var conv1_bias: ExTensor
    var conv2_kernel: ExTensor
    var conv2_bias: ExTensor
    var fc1_weights: ExTensor
    var fc1_bias: ExTensor
    var fc2_weights: ExTensor
    var fc2_bias: ExTensor
    var fc3_weights: ExTensor
    var fc3_bias: ExTensor

    fn __init__(out self, num_classes: Int = 47) raises:
        """Initialize LeNet-5 model with random weights.

        Args:
            num_classes: Number of output classes (default: 47 for EMNIST Balanced)
        """
        self.num_classes = num_classes

        # Compute derived dimensions
        var flattened_size = compute_flattened_size()

        # Conv1: INPUT_CHANNELS -> CONV1_OUT_CHANNELS, kernel CONV1_KERNEL_SIZE x CONV1_KERNEL_SIZE
        var conv1_shape = List[Int](CONV1_OUT_CHANNELS, INPUT_CHANNELS, CONV1_KERNEL_SIZE, CONV1_KERNEL_SIZE)
        var conv1_fan_in = INPUT_CHANNELS * CONV1_KERNEL_SIZE * CONV1_KERNEL_SIZE
        var conv1_fan_out = CONV1_OUT_CHANNELS * CONV1_KERNEL_SIZE * CONV1_KERNEL_SIZE
        self.conv1_kernel = kaiming_uniform(conv1_fan_in, conv1_fan_out, conv1_shape, dtype=DType.float32)
        var conv1_bias_shape = List[Int](CONV1_OUT_CHANNELS)
        self.conv1_bias = zeros(conv1_bias_shape, DType.float32)

        # Conv2: CONV1_OUT_CHANNELS -> CONV2_OUT_CHANNELS, kernel CONV2_KERNEL_SIZE x CONV2_KERNEL_SIZE
        var conv2_shape = List[Int](CONV2_OUT_CHANNELS, CONV1_OUT_CHANNELS, CONV2_KERNEL_SIZE, CONV2_KERNEL_SIZE)
        var conv2_fan_in = CONV1_OUT_CHANNELS * CONV2_KERNEL_SIZE * CONV2_KERNEL_SIZE
        var conv2_fan_out = CONV2_OUT_CHANNELS * CONV2_KERNEL_SIZE * CONV2_KERNEL_SIZE
        self.conv2_kernel = kaiming_uniform(conv2_fan_in, conv2_fan_out, conv2_shape, dtype=DType.float32)
        var conv2_bias_shape = List[Int](CONV2_OUT_CHANNELS)
        self.conv2_bias = zeros(conv2_bias_shape, DType.float32)

        # FC1: flattened_size -> FC1_OUT_FEATURES (derived from conv/pool layers)
        var fc1_shape = List[Int](FC1_OUT_FEATURES, flattened_size)
        self.fc1_weights = kaiming_uniform(flattened_size, FC1_OUT_FEATURES, fc1_shape, dtype=DType.float32)
        var fc1_bias_shape = List[Int](FC1_OUT_FEATURES)
        self.fc1_bias = zeros(fc1_bias_shape, DType.float32)

        # FC2: FC1_OUT_FEATURES -> FC2_OUT_FEATURES
        var fc2_shape = List[Int](FC2_OUT_FEATURES, FC1_OUT_FEATURES)
        self.fc2_weights = kaiming_uniform(FC1_OUT_FEATURES, FC2_OUT_FEATURES, fc2_shape, dtype=DType.float32)
        var fc2_bias_shape = List[Int](FC2_OUT_FEATURES)
        self.fc2_bias = zeros(fc2_bias_shape, DType.float32)

        # FC3: FC2_OUT_FEATURES -> num_classes
        var fc3_shape = List[Int](num_classes, FC2_OUT_FEATURES)
        self.fc3_weights = kaiming_uniform(FC2_OUT_FEATURES, num_classes, fc3_shape, dtype=DType.float32)
        var fc3_bias_shape = List[Int](num_classes)
        self.fc3_bias = zeros(fc3_bias_shape, DType.float32)

    fn forward(mut self, input: ExTensor) raises -> ExTensor:
        """Forward pass through LeNet-5.

        Args:
            input: Input tensor of shape (batch, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)

        Returns:
            Output logits of shape (batch, num_classes)
        """
        # Conv1 + ReLU + MaxPool
        var conv1_out = conv2d(input, self.conv1_kernel, self.conv1_bias,
                               stride=CONV1_STRIDE, padding=CONV1_PADDING)
        var relu1_out = relu(conv1_out)
        var pool1_out = maxpool2d(relu1_out,
                                  kernel_size=POOL1_KERNEL_SIZE,
                                  stride=POOL1_STRIDE,
                                  padding=POOL1_PADDING)

        # Conv2 + ReLU + MaxPool
        var conv2_out = conv2d(pool1_out, self.conv2_kernel, self.conv2_bias,
                               stride=CONV2_STRIDE, padding=CONV2_PADDING)
        var relu2_out = relu(conv2_out)
        var pool2_out = maxpool2d(relu2_out,
                                  kernel_size=POOL2_KERNEL_SIZE,
                                  stride=POOL2_STRIDE,
                                  padding=POOL2_PADDING)

        # Flatten: (batch, CONV2_OUT_CHANNELS, h, w) -> (batch, flattened_size)
        var pool2_shape = pool2_out.shape()
        var batch_size = pool2_shape[0]
        var flattened_size = pool2_shape[1] * pool2_shape[2] * pool2_shape[3]

        var flatten_shape = List[Int](batch_size, flattened_size)
        var flattened = pool2_out.reshape(flatten_shape)

        # FC1 + ReLU
        var fc1_out = linear(flattened, self.fc1_weights, self.fc1_bias)
        var relu3_out = relu(fc1_out)

        # FC2 + ReLU
        var fc2_out = linear(relu3_out, self.fc2_weights, self.fc2_bias)
        var relu4_out = relu(fc2_out)

        # FC3 (output logits)
        var output = linear(relu4_out, self.fc3_weights, self.fc3_bias)

        return output^

    fn predict(mut self, input: ExTensor) raises -> Int:
        """Predict class for a single input.

        Args:
            input: Input tensor of shape (1, 1, 28, 28)

        Returns:
            Predicted class index (0 to num_classes-1)
        """
        var logits = self.forward(input)

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
            - conv1_kernel.weights
            - conv1_bias.weights
            - etc.
        """
        # Save each parameter to its own file
        save_tensor(self.conv1_kernel, "conv1_kernel", weights_dir + "/conv1_kernel.weights")
        save_tensor(self.conv1_bias, "conv1_bias", weights_dir + "/conv1_bias.weights")
        save_tensor(self.conv2_kernel, "conv2_kernel", weights_dir + "/conv2_kernel.weights")
        save_tensor(self.conv2_bias, "conv2_bias", weights_dir + "/conv2_bias.weights")
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
        self.conv1_kernel = load_tensor(weights_dir + "/conv1_kernel.weights")
        self.conv1_bias = load_tensor(weights_dir + "/conv1_bias.weights")
        self.conv2_kernel = load_tensor(weights_dir + "/conv2_kernel.weights")
        self.conv2_bias = load_tensor(weights_dir + "/conv2_bias.weights")
        self.fc1_weights = load_tensor(weights_dir + "/fc1_weights.weights")
        self.fc1_bias = load_tensor(weights_dir + "/fc1_bias.weights")
        self.fc2_weights = load_tensor(weights_dir + "/fc2_weights.weights")
        self.fc2_bias = load_tensor(weights_dir + "/fc2_bias.weights")
        self.fc3_weights = load_tensor(weights_dir + "/fc3_weights.weights")
        self.fc3_bias = load_tensor(weights_dir + "/fc3_bias.weights")

    fn update_parameters(mut self, learning_rate: Float32,
                        grad_conv1_kernel: ExTensor,
                        grad_conv1_bias: ExTensor,
                        grad_conv2_kernel: ExTensor,
                        grad_conv2_bias: ExTensor,
                        grad_fc1_weights: ExTensor,
                        grad_fc1_bias: ExTensor,
                        grad_fc2_weights: ExTensor,
                        grad_fc2_bias: ExTensor,
                        grad_fc3_weights: ExTensor,
                        grad_fc3_bias: ExTensor) raises:
        """Update parameters using SGD.

        Args:
            learning_rate: Learning rate for gradient descent
            grad_*: Gradients for each parameter
        """
        # SGD update: param = param - lr * grad
        _sgd_update(self.conv1_kernel, grad_conv1_kernel, learning_rate)
        _sgd_update(self.conv1_bias, grad_conv1_bias, learning_rate)
        _sgd_update(self.conv2_kernel, grad_conv2_kernel, learning_rate)
        _sgd_update(self.conv2_bias, grad_conv2_bias, learning_rate)
        _sgd_update(self.fc1_weights, grad_fc1_weights, learning_rate)
        _sgd_update(self.fc1_bias, grad_fc1_bias, learning_rate)
        _sgd_update(self.fc2_weights, grad_fc2_weights, learning_rate)
        _sgd_update(self.fc2_bias, grad_fc2_bias, learning_rate)
        _sgd_update(self.fc3_weights, grad_fc3_weights, learning_rate)
        _sgd_update(self.fc3_bias, grad_fc3_bias, learning_rate)


    fn parameters(self) raises -> List[ExTensor]:
        """Return all trainable parameters.

        Returns:
            List of parameter tensors (10 total: conv1/2 kernel/bias, fc1/2/3 weights/bias)

        Note:
            This method copies the parameter tensors. For in-place updates,
            use update_parameters() instead.
        """
        var params = List[ExTensor]()
        params.append(self.conv1_kernel)
        params.append(self.conv1_bias)
        params.append(self.conv2_kernel)
        params.append(self.conv2_bias)
        params.append(self.fc1_weights)
        params.append(self.fc1_bias)
        params.append(self.fc2_weights)
        params.append(self.fc2_bias)
        params.append(self.fc3_weights)
        params.append(self.fc3_bias)
        return params^

    fn zero_grad(mut self) raises:
        """Reset all parameter gradients to zero.

        Note:
            LeNet5 uses manual gradient computation in compute_gradients().
            Gradients are computed fresh each forward/backward pass, so this
            is a no-op. Included for Model trait conformance.
        """
        # Gradients are computed fresh each batch in compute_gradients()
        # No persistent gradient accumulators to reset
        pass


fn _sgd_update(mut param: ExTensor, grad: ExTensor, lr: Float32) raises:
    """SGD parameter update: param = param - lr * grad"""
    var numel = param.numel()
    var param_data = param._data.bitcast[Float32]()
    var grad_data = grad._data.bitcast[Float32]()

    for i in range(numel):
        param_data[i] -= lr * grad_data[i]
