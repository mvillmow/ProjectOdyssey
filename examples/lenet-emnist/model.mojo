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
from collections import List
from weights import save_tensor, load_tensor


struct LeNet5:
    """LeNet-5 model for EMNIST classification.

    Attributes:
        num_classes: Number of output classes (47 for EMNIST Balanced)
        conv1_kernel: First conv layer weights (6, 1, 5, 5)
        conv1_bias: First conv layer bias (6,)
        conv2_kernel: Second conv layer weights (16, 6, 5, 5)
        conv2_bias: Second conv layer bias (16,)
        fc1_weights: First FC layer weights (120, 256)
        fc1_bias: First FC layer bias (120,)
        fc2_weights: Second FC layer weights (84, 120)
        fc2_bias: Second FC layer bias (84,)
        fc3_weights: Third FC layer weights (num_classes, 84)
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

        # Conv1: 1 input channel, 6 output channels, 5x5 kernel
        var conv1_shape = List[Int]()
        conv1_shape.append(6)
        conv1_shape.append(1)
        conv1_shape.append(5)
        conv1_shape.append(5)
        var conv1_fan_in = 1 * 5 * 5  # in_channels * kernel_h * kernel_w = 25
        var conv1_fan_out = 6 * 5 * 5  # out_channels * kernel_h * kernel_w = 150
        self.conv1_kernel = kaiming_uniform(conv1_fan_in, conv1_fan_out, conv1_shape, dtype=DType.float32)
        var conv1_bias_shape = List[Int]()
        conv1_bias_shape.append(6)
        self.conv1_bias = zeros(conv1_bias_shape, DType.float32)

        # Conv2: 6 input channels, 16 output channels, 5x5 kernel
        var conv2_shape = List[Int]()
        conv2_shape.append(16)
        conv2_shape.append(6)
        conv2_shape.append(5)
        conv2_shape.append(5)
        var conv2_fan_in = 6 * 5 * 5  # in_channels * kernel_h * kernel_w = 150
        var conv2_fan_out = 16 * 5 * 5  # out_channels * kernel_h * kernel_w = 400
        self.conv2_kernel = kaiming_uniform(conv2_fan_in, conv2_fan_out, conv2_shape, dtype=DType.float32)
        var conv2_bias_shape = List[Int]()
        conv2_bias_shape.append(16)
        self.conv2_bias = zeros(conv2_bias_shape, DType.float32)

        # After conv1 (28x28 -> 24x24) -> pool1 (24x24 -> 12x12)
        # After conv2 (12x12 -> 8x8) -> pool2 (8x8 -> 4x4)
        # Flattened size: 16 * 4 * 4 = 256

        # FC1: 256 -> 120
        var fc1_shape = List[Int]()
        fc1_shape.append(120)
        fc1_shape.append(256)
        var fc1_fan_in = 256
        var fc1_fan_out = 120
        self.fc1_weights = kaiming_uniform(fc1_fan_in, fc1_fan_out, fc1_shape, dtype=DType.float32)
        var fc1_bias_shape = List[Int]()
        fc1_bias_shape.append(120)
        self.fc1_bias = zeros(fc1_bias_shape, DType.float32)

        # FC2: 120 -> 84
        var fc2_shape = List[Int]()
        fc2_shape.append(84)
        fc2_shape.append(120)
        var fc2_fan_in = 120
        var fc2_fan_out = 84
        self.fc2_weights = kaiming_uniform(fc2_fan_in, fc2_fan_out, fc2_shape, dtype=DType.float32)
        var fc2_bias_shape = List[Int]()
        fc2_bias_shape.append(84)
        self.fc2_bias = zeros(fc2_bias_shape, DType.float32)

        # FC3: 84 -> num_classes
        var fc3_shape = List[Int]()
        fc3_shape.append(num_classes)
        fc3_shape.append(84)
        var fc3_fan_in = 84
        var fc3_fan_out = num_classes
        self.fc3_weights = kaiming_uniform(fc3_fan_in, fc3_fan_out, fc3_shape, dtype=DType.float32)
        var fc3_bias_shape = List[Int]()
        fc3_bias_shape.append(num_classes)
        self.fc3_bias = zeros(fc3_bias_shape, DType.float32)

    fn forward(mut self, input: ExTensor) raises -> ExTensor:
        """Forward pass through LeNet-5.

        Args:
            input: Input tensor of shape (batch, 1, 28, 28)

        Returns:
            Output logits of shape (batch, num_classes)
        """
        # Conv1 + ReLU + MaxPool
        var conv1_out = conv2d(input, self.conv1_kernel, self.conv1_bias, stride=1, padding=0)
        var relu1_out = relu(conv1_out)
        var pool1_out = maxpool2d(relu1_out, kernel_size=2, stride=2, padding=0)

        # Conv2 + ReLU + MaxPool
        var conv2_out = conv2d(pool1_out, self.conv2_kernel, self.conv2_bias, stride=1, padding=0)
        var relu2_out = relu(conv2_out)
        var pool2_out = maxpool2d(relu2_out, kernel_size=2, stride=2, padding=0)

        # Flatten: (batch, 16, 4, 4) -> (batch, 256)
        var pool2_shape = pool2_out.shape()
        var batch_size = pool2_shape[0]
        var flattened_size = pool2_shape[1] * pool2_shape[2] * pool2_shape[3]

        var flatten_shape = List[Int]()
        flatten_shape.append(batch_size)
        flatten_shape.append(flattened_size)
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


fn _sgd_update(mut param: ExTensor, grad: ExTensor, lr: Float32) raises:
    """SGD parameter update: param = param - lr * grad"""
    var numel = param.numel()
    var param_data = param._data.bitcast[Float32]()
    var grad_data = grad._data.bitcast[Float32]()

    for i in range(numel):
        param_data[i] -= lr * grad_data[i]
