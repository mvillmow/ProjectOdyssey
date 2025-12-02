"""VGG-16 Model for CIFAR-10 Classification

Classic VGG-16 architecture adapted for CIFAR-10 dataset (32x32 RGB images).

Architecture:
    VGG-16 consists of 13 convolutional layers and 3 fully connected layers,
    organized into 5 blocks. Each block uses 3x3 convolutions followed by max pooling.

    Input (32x32x3) ->

    Block 1:
        Conv2D(64, 3x3, pad=1) -> ReLU ->
        Conv2D(64, 3x3, pad=1) -> ReLU ->
        MaxPool(2x2, stride=2) -> (16x16x64)

    Block 2:
        Conv2D(128, 3x3, pad=1) -> ReLU ->
        Conv2D(128, 3x3, pad=1) -> ReLU ->
        MaxPool(2x2, stride=2) -> (8x8x128)

    Block 3:
        Conv2D(256, 3x3, pad=1) -> ReLU ->
        Conv2D(256, 3x3, pad=1) -> ReLU ->
        Conv2D(256, 3x3, pad=1) -> ReLU ->
        MaxPool(2x2, stride=2) -> (4x4x256)

    Block 4:
        Conv2D(512, 3x3, pad=1) -> ReLU ->
        Conv2D(512, 3x3, pad=1) -> ReLU ->
        Conv2D(512, 3x3, pad=1) -> ReLU ->
        MaxPool(2x2, stride=2) -> (2x2x512)

    Block 5:
        Conv2D(512, 3x3, pad=1) -> ReLU ->
        Conv2D(512, 3x3, pad=1) -> ReLU ->
        Conv2D(512, 3x3, pad=1) -> ReLU ->
        MaxPool(2x2, stride=2) -> (1x1x512)

    Flatten (512x1x1 = 512) ->
    Linear(512 -> 512) -> ReLU -> Dropout(0.5) ->
    Linear(512 -> 512) -> ReLU -> Dropout(0.5) ->
    Linear(512 -> 10)

Key Innovation:
    - Depth: 16 layers (very deep for 2014)
    - Uniform 3x3 convolutions throughout
    - Simplicity: repetitive architecture, easy to implement
    - Small receptive fields with more non-linearities

References:
    - Simonyan, K., & Zisserman, A. (2014).
      Very deep convolutional networks for large-scale image recognition.
      arXiv preprint arXiv:1409.1556.
    - CIFAR-10 Dataset: https://www.cs.toronto.edu/~kriz/cifar.html
"""

from shared.core import ExTensor, zeros
from shared.core.conv import conv2d, conv2d_backward
from shared.core.pooling import maxpool2d, maxpool2d_backward
from shared.core.linear import linear, linear_backward
from shared.core.activation import relu, relu_backward
from shared.core.dropout import dropout, dropout_backward
from shared.core.initializers import he_uniform
from shared.training.optimizers import sgd_momentum_update_inplace
from shared.utils.serialization import save_tensor, load_tensor


struct VGG16:
    """VGG-16 model for CIFAR-10 classification.

    Attributes:
        num_classes: Number of output classes (10 for CIFAR-10)
        dropout_rate: Dropout probability for FC layers (default 0.5)

        # Block 1 (2 conv layers)
        conv1_1_kernel, conv1_1_bias: First conv layer (64, 3, 3, 3)
        conv1_2_kernel, conv1_2_bias: Second conv layer (64, 64, 3, 3)

        # Block 2 (2 conv layers)
        conv2_1_kernel, conv2_1_bias: Third conv layer (128, 64, 3, 3)
        conv2_2_kernel, conv2_2_bias: Fourth conv layer (128, 128, 3, 3)

        # Block 3 (3 conv layers)
        conv3_1_kernel, conv3_1_bias: Fifth conv layer (256, 128, 3, 3)
        conv3_2_kernel, conv3_2_bias: Sixth conv layer (256, 256, 3, 3)
        conv3_3_kernel, conv3_3_bias: Seventh conv layer (256, 256, 3, 3)

        # Block 4 (3 conv layers)
        conv4_1_kernel, conv4_1_bias: Eighth conv layer (512, 256, 3, 3)
        conv4_2_kernel, conv4_2_bias: Ninth conv layer (512, 512, 3, 3)
        conv4_3_kernel, conv4_3_bias: Tenth conv layer (512, 512, 3, 3)

        # Block 5 (3 conv layers)
        conv5_1_kernel, conv5_1_bias: Eleventh conv layer (512, 512, 3, 3)
        conv5_2_kernel, conv5_2_bias: Twelfth conv layer (512, 512, 3, 3)
        conv5_3_kernel, conv5_3_bias: Thirteenth conv layer (512, 512, 3, 3)

        # Fully connected layers
        fc1_weights, fc1_bias: First FC layer (512, 512)
        fc2_weights, fc2_bias: Second FC layer (512, 512)
        fc3_weights, fc3_bias: Third FC layer (num_classes, 512)
    """

    var num_classes: Int
    var dropout_rate: Float32

    # Block 1 parameters
    var conv1_1_kernel: ExTensor
    var conv1_1_bias: ExTensor
    var conv1_2_kernel: ExTensor
    var conv1_2_bias: ExTensor

    # Block 2 parameters
    var conv2_1_kernel: ExTensor
    var conv2_1_bias: ExTensor
    var conv2_2_kernel: ExTensor
    var conv2_2_bias: ExTensor

    # Block 3 parameters
    var conv3_1_kernel: ExTensor
    var conv3_1_bias: ExTensor
    var conv3_2_kernel: ExTensor
    var conv3_2_bias: ExTensor
    var conv3_3_kernel: ExTensor
    var conv3_3_bias: ExTensor

    # Block 4 parameters
    var conv4_1_kernel: ExTensor
    var conv4_1_bias: ExTensor
    var conv4_2_kernel: ExTensor
    var conv4_2_bias: ExTensor
    var conv4_3_kernel: ExTensor
    var conv4_3_bias: ExTensor

    # Block 5 parameters
    var conv5_1_kernel: ExTensor
    var conv5_1_bias: ExTensor
    var conv5_2_kernel: ExTensor
    var conv5_2_bias: ExTensor
    var conv5_3_kernel: ExTensor
    var conv5_3_bias: ExTensor

    # Fully connected layer parameters
    var fc1_weights: ExTensor
    var fc1_bias: ExTensor
    var fc2_weights: ExTensor
    var fc2_bias: ExTensor
    var fc3_weights: ExTensor
    var fc3_bias: ExTensor

    fn __init__(out self, num_classes: Int = 10, dropout_rate: Float32 = 0.5) raises:
        """Initialize VGG-16 model with random weights.

        Args:
            num_classes: Number of output classes (default: 10 for CIFAR-10)
            dropout_rate: Dropout probability for FC layers (default: 0.5)
        """
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # Block 1: Input channels=3 (RGB), output channels=64
        # Conv1_1: (64, 3, 3, 3)
        var conv1_1_shape = List[Int]()
        conv1_1_shape.append(64)   # out_channels
        conv1_1_shape.append(3)    # in_channels (RGB)
        conv1_1_shape.append(3)    # kernel_height
        conv1_1_shape.append(3)    # kernel_width
        self.conv1_1_kernel = he_uniform(conv1_1_shape, DType.float32)
        self.conv1_1_bias = zeros(List[Int]().append(64), DType.float32)

        # Conv1_2: (64, 64, 3, 3)
        var conv1_2_shape = List[Int]()
        conv1_2_shape.append(64)   # out_channels
        conv1_2_shape.append(64)   # in_channels
        conv1_2_shape.append(3)    # kernel_height
        conv1_2_shape.append(3)    # kernel_width
        self.conv1_2_kernel = he_uniform(conv1_2_shape, DType.float32)
        self.conv1_2_bias = zeros(List[Int]().append(64), DType.float32)

        # Block 2: Input channels=64, output channels=128
        # Conv2_1: (128, 64, 3, 3)
        var conv2_1_shape = List[Int]()
        conv2_1_shape.append(128)
        conv2_1_shape.append(64)
        conv2_1_shape.append(3)
        conv2_1_shape.append(3)
        self.conv2_1_kernel = he_uniform(conv2_1_shape, DType.float32)
        self.conv2_1_bias = zeros(List[Int]().append(128), DType.float32)

        # Conv2_2: (128, 128, 3, 3)
        var conv2_2_shape = List[Int]()
        conv2_2_shape.append(128)
        conv2_2_shape.append(128)
        conv2_2_shape.append(3)
        conv2_2_shape.append(3)
        self.conv2_2_kernel = he_uniform(conv2_2_shape, DType.float32)
        self.conv2_2_bias = zeros(List[Int]().append(128), DType.float32)

        # Block 3: Input channels=128, output channels=256
        # Conv3_1: (256, 128, 3, 3)
        var conv3_1_shape = List[Int]()
        conv3_1_shape.append(256)
        conv3_1_shape.append(128)
        conv3_1_shape.append(3)
        conv3_1_shape.append(3)
        self.conv3_1_kernel = he_uniform(conv3_1_shape, DType.float32)
        self.conv3_1_bias = zeros(List[Int]().append(256), DType.float32)

        # Conv3_2: (256, 256, 3, 3)
        var conv3_2_shape = List[Int]()
        conv3_2_shape.append(256)
        conv3_2_shape.append(256)
        conv3_2_shape.append(3)
        conv3_2_shape.append(3)
        self.conv3_2_kernel = he_uniform(conv3_2_shape, DType.float32)
        self.conv3_2_bias = zeros(List[Int]().append(256), DType.float32)

        # Conv3_3: (256, 256, 3, 3)
        var conv3_3_shape = List[Int]()
        conv3_3_shape.append(256)
        conv3_3_shape.append(256)
        conv3_3_shape.append(3)
        conv3_3_shape.append(3)
        self.conv3_3_kernel = he_uniform(conv3_3_shape, DType.float32)
        self.conv3_3_bias = zeros(List[Int]().append(256), DType.float32)

        # Block 4: Input channels=256, output channels=512
        # Conv4_1: (512, 256, 3, 3)
        var conv4_1_shape = List[Int]()
        conv4_1_shape.append(512)
        conv4_1_shape.append(256)
        conv4_1_shape.append(3)
        conv4_1_shape.append(3)
        self.conv4_1_kernel = he_uniform(conv4_1_shape, DType.float32)
        self.conv4_1_bias = zeros(List[Int]().append(512), DType.float32)

        # Conv4_2: (512, 512, 3, 3)
        var conv4_2_shape = List[Int]()
        conv4_2_shape.append(512)
        conv4_2_shape.append(512)
        conv4_2_shape.append(3)
        conv4_2_shape.append(3)
        self.conv4_2_kernel = he_uniform(conv4_2_shape, DType.float32)
        self.conv4_2_bias = zeros(List[Int]().append(512), DType.float32)

        # Conv4_3: (512, 512, 3, 3)
        var conv4_3_shape = List[Int]()
        conv4_3_shape.append(512)
        conv4_3_shape.append(512)
        conv4_3_shape.append(3)
        conv4_3_shape.append(3)
        self.conv4_3_kernel = he_uniform(conv4_3_shape, DType.float32)
        self.conv4_3_bias = zeros(List[Int]().append(512), DType.float32)

        # Block 5: Input channels=512, output channels=512
        # Conv5_1: (512, 512, 3, 3)
        var conv5_1_shape = List[Int]()
        conv5_1_shape.append(512)
        conv5_1_shape.append(512)
        conv5_1_shape.append(3)
        conv5_1_shape.append(3)
        self.conv5_1_kernel = he_uniform(conv5_1_shape, DType.float32)
        self.conv5_1_bias = zeros(List[Int]().append(512), DType.float32)

        # Conv5_2: (512, 512, 3, 3)
        var conv5_2_shape = List[Int]()
        conv5_2_shape.append(512)
        conv5_2_shape.append(512)
        conv5_2_shape.append(3)
        conv5_2_shape.append(3)
        self.conv5_2_kernel = he_uniform(conv5_2_shape, DType.float32)
        self.conv5_2_bias = zeros(List[Int]().append(512), DType.float32)

        # Conv5_3: (512, 512, 3, 3)
        var conv5_3_shape = List[Int]()
        conv5_3_shape.append(512)
        conv5_3_shape.append(512)
        conv5_3_shape.append(3)
        conv5_3_shape.append(3)
        self.conv5_3_kernel = he_uniform(conv5_3_shape, DType.float32)
        self.conv5_3_bias = zeros(List[Int]().append(512), DType.float32)

        # After all conv blocks + pools: 32 -> 16 -> 8 -> 4 -> 2 -> 1
        # Flattened size: 512 * 1 * 1 = 512

        # FC1: 512 -> 512
        var fc1_shape = List[Int]()
        fc1_shape.append(512)   # out_features
        fc1_shape.append(512)   # in_features
        self.fc1_weights = he_uniform(fc1_shape, DType.float32)
        self.fc1_bias = zeros(List[Int]().append(512), DType.float32)

        # FC2: 512 -> 512
        var fc2_shape = List[Int]()
        fc2_shape.append(512)   # out_features
        fc2_shape.append(512)   # in_features
        self.fc2_weights = he_uniform(fc2_shape, DType.float32)
        self.fc2_bias = zeros(List[Int]().append(512), DType.float32)

        # FC3: 512 -> num_classes
        var fc3_shape = List[Int]()
        fc3_shape.append(num_classes)  # out_features
        fc3_shape.append(512)          # in_features
        self.fc3_weights = he_uniform(fc3_shape, DType.float32)
        self.fc3_bias = zeros(List[Int]().append(num_classes), DType.float32)

    fn forward(mut self, input: ExTensor, training: Bool = True) raises -> ExTensor:
        """Forward pass through VGG-16.

        Args:
            input: Input tensor of shape (batch, 3, 32, 32)
            training: Whether in training mode (applies dropout if True)

        Returns:
            Output logits of shape (batch, num_classes)
        """
        # Block 1: Conv -> ReLU -> Conv -> ReLU -> MaxPool
        var conv1_1 = conv2d(input, self.conv1_1_kernel, self.conv1_1_bias, stride=1, padding=1)
        var relu1_1 = relu(conv1_1)
        var conv1_2 = conv2d(relu1_1, self.conv1_2_kernel, self.conv1_2_bias, stride=1, padding=1)
        var relu1_2 = relu(conv1_2)
        var pool1 = maxpool2d(relu1_2, kernel_size=2, stride=2, padding=0)  # 32x32 -> 16x16

        # Block 2: Conv -> ReLU -> Conv -> ReLU -> MaxPool
        var conv2_1 = conv2d(pool1, self.conv2_1_kernel, self.conv2_1_bias, stride=1, padding=1)
        var relu2_1 = relu(conv2_1)
        var conv2_2 = conv2d(relu2_1, self.conv2_2_kernel, self.conv2_2_bias, stride=1, padding=1)
        var relu2_2 = relu(conv2_2)
        var pool2 = maxpool2d(relu2_2, kernel_size=2, stride=2, padding=0)  # 16x16 -> 8x8

        # Block 3: Conv -> ReLU -> Conv -> ReLU -> Conv -> ReLU -> MaxPool
        var conv3_1 = conv2d(pool2, self.conv3_1_kernel, self.conv3_1_bias, stride=1, padding=1)
        var relu3_1 = relu(conv3_1)
        var conv3_2 = conv2d(relu3_1, self.conv3_2_kernel, self.conv3_2_bias, stride=1, padding=1)
        var relu3_2 = relu(conv3_2)
        var conv3_3 = conv2d(relu3_2, self.conv3_3_kernel, self.conv3_3_bias, stride=1, padding=1)
        var relu3_3 = relu(conv3_3)
        var pool3 = maxpool2d(relu3_3, kernel_size=2, stride=2, padding=0)  # 8x8 -> 4x4

        # Block 4: Conv -> ReLU -> Conv -> ReLU -> Conv -> ReLU -> MaxPool
        var conv4_1 = conv2d(pool3, self.conv4_1_kernel, self.conv4_1_bias, stride=1, padding=1)
        var relu4_1 = relu(conv4_1)
        var conv4_2 = conv2d(relu4_1, self.conv4_2_kernel, self.conv4_2_bias, stride=1, padding=1)
        var relu4_2 = relu(conv4_2)
        var conv4_3 = conv2d(relu4_2, self.conv4_3_kernel, self.conv4_3_bias, stride=1, padding=1)
        var relu4_3 = relu(conv4_3)
        var pool4 = maxpool2d(relu4_3, kernel_size=2, stride=2, padding=0)  # 4x4 -> 2x2

        # Block 5: Conv -> ReLU -> Conv -> ReLU -> Conv -> ReLU -> MaxPool
        var conv5_1 = conv2d(pool4, self.conv5_1_kernel, self.conv5_1_bias, stride=1, padding=1)
        var relu5_1 = relu(conv5_1)
        var conv5_2 = conv2d(relu5_1, self.conv5_2_kernel, self.conv5_2_bias, stride=1, padding=1)
        var relu5_2 = relu(conv5_2)
        var conv5_3 = conv2d(relu5_2, self.conv5_3_kernel, self.conv5_3_bias, stride=1, padding=1)
        var relu5_3 = relu(conv5_3)
        var pool5 = maxpool2d(relu5_3, kernel_size=2, stride=2, padding=0)  # 2x2 -> 1x1

        # Flatten: (batch, 512, 1, 1) -> (batch, 512)
        var pool5_shape = pool5.shape()
        var batch_size = pool5_shape[0]
        var flattened_size = pool5_shape[1] * pool5_shape[2] * pool5_shape[3]

        var flatten_shape = List[Int]()
        flatten_shape.append(batch_size)
        flatten_shape.append(flattened_size)
        var flattened = pool5.reshape(flatten_shape)

        # FC1 + ReLU + Dropout
        var fc1 = linear(flattened, self.fc1_weights, self.fc1_bias)
        var relu_fc1 = relu(fc1)
        var drop1 = relu_fc1  # Will be replaced by dropout in training
        if training:
            drop1 = dropout(relu_fc1, self.dropout_rate)

        # FC2 + ReLU + Dropout
        var fc2 = linear(drop1, self.fc2_weights, self.fc2_bias)
        var relu_fc2 = relu(fc2)
        var drop2 = relu_fc2  # Will be replaced by dropout in training
        if training:
            drop2 = dropout(relu_fc2, self.dropout_rate)

        # FC3 (output logits)
        var output = linear(drop2, self.fc3_weights, self.fc3_bias)

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
            - conv1_1_kernel.weights, conv1_1_bias.weights, etc.
        """
        # Save Block 1
        save_tensor(self.conv1_1_kernel, weights_dir + "/conv1_1_kernel.weights", "conv1_1_kernel")
        save_tensor(self.conv1_1_bias, weights_dir + "/conv1_1_bias.weights", "conv1_1_bias")
        save_tensor(self.conv1_2_kernel, weights_dir + "/conv1_2_kernel.weights", "conv1_2_kernel")
        save_tensor(self.conv1_2_bias, weights_dir + "/conv1_2_bias.weights", "conv1_2_bias")

        # Save Block 2
        save_tensor(self.conv2_1_kernel, weights_dir + "/conv2_1_kernel.weights", "conv2_1_kernel")
        save_tensor(self.conv2_1_bias, weights_dir + "/conv2_1_bias.weights", "conv2_1_bias")
        save_tensor(self.conv2_2_kernel, weights_dir + "/conv2_2_kernel.weights", "conv2_2_kernel")
        save_tensor(self.conv2_2_bias, weights_dir + "/conv2_2_bias.weights", "conv2_2_bias")

        # Save Block 3
        save_tensor(self.conv3_1_kernel, weights_dir + "/conv3_1_kernel.weights", "conv3_1_kernel")
        save_tensor(self.conv3_1_bias, weights_dir + "/conv3_1_bias.weights", "conv3_1_bias")
        save_tensor(self.conv3_2_kernel, weights_dir + "/conv3_2_kernel.weights", "conv3_2_kernel")
        save_tensor(self.conv3_2_bias, weights_dir + "/conv3_2_bias.weights", "conv3_2_bias")
        save_tensor(self.conv3_3_kernel, weights_dir + "/conv3_3_kernel.weights", "conv3_3_kernel")
        save_tensor(self.conv3_3_bias, weights_dir + "/conv3_3_bias.weights", "conv3_3_bias")

        # Save Block 4
        save_tensor(self.conv4_1_kernel, weights_dir + "/conv4_1_kernel.weights", "conv4_1_kernel")
        save_tensor(self.conv4_1_bias, weights_dir + "/conv4_1_bias.weights", "conv4_1_bias")
        save_tensor(self.conv4_2_kernel, weights_dir + "/conv4_2_kernel.weights", "conv4_2_kernel")
        save_tensor(self.conv4_2_bias, weights_dir + "/conv4_2_bias.weights", "conv4_2_bias")
        save_tensor(self.conv4_3_kernel, weights_dir + "/conv4_3_kernel.weights", "conv4_3_kernel")
        save_tensor(self.conv4_3_bias, weights_dir + "/conv4_3_bias.weights", "conv4_3_bias")

        # Save Block 5
        save_tensor(self.conv5_1_kernel, weights_dir + "/conv5_1_kernel.weights", "conv5_1_kernel")
        save_tensor(self.conv5_1_bias, weights_dir + "/conv5_1_bias.weights", "conv5_1_bias")
        save_tensor(self.conv5_2_kernel, weights_dir + "/conv5_2_kernel.weights", "conv5_2_kernel")
        save_tensor(self.conv5_2_bias, weights_dir + "/conv5_2_bias.weights", "conv5_2_bias")
        save_tensor(self.conv5_3_kernel, weights_dir + "/conv5_3_kernel.weights", "conv5_3_kernel")
        save_tensor(self.conv5_3_bias, weights_dir + "/conv5_3_bias.weights", "conv5_3_bias")

        # Save FC layers
        save_tensor(self.fc1_weights, weights_dir + "/fc1_weights.weights", "fc1_weights")
        save_tensor(self.fc1_bias, weights_dir + "/fc1_bias.weights", "fc1_bias")
        save_tensor(self.fc2_weights, weights_dir + "/fc2_weights.weights", "fc2_weights")
        save_tensor(self.fc2_bias, weights_dir + "/fc2_bias.weights", "fc2_bias")
        save_tensor(self.fc3_weights, weights_dir + "/fc3_weights.weights", "fc3_weights")
        save_tensor(self.fc3_bias, weights_dir + "/fc3_bias.weights", "fc3_bias")

    fn load_weights(mut self, weights_dir: String) raises:
        """Load model weights from directory.

        Args:
            weights_dir: Directory containing weight files

        Raises:
            Error: If weight files are missing or have incompatible shapes
        """
        # Load Block 1
        var r1 = load_tensor(weights_dir + "/conv1_1_kernel.weights")
        self.conv1_1_kernel = r1[1]^
        var r2 = load_tensor(weights_dir + "/conv1_1_bias.weights")
        self.conv1_1_bias = r2[1]^
        var r3 = load_tensor(weights_dir + "/conv1_2_kernel.weights")
        self.conv1_2_kernel = r3[1]^
        var r4 = load_tensor(weights_dir + "/conv1_2_bias.weights")
        self.conv1_2_bias = r4[1]^

        # Load Block 2
        var r5 = load_tensor(weights_dir + "/conv2_1_kernel.weights")
        self.conv2_1_kernel = r5[1]^
        var r6 = load_tensor(weights_dir + "/conv2_1_bias.weights")
        self.conv2_1_bias = r6[1]^
        var r7 = load_tensor(weights_dir + "/conv2_2_kernel.weights")
        self.conv2_2_kernel = r7[1]^
        var r8 = load_tensor(weights_dir + "/conv2_2_bias.weights")
        self.conv2_2_bias = r8[1]^

        # Load Block 3
        var r9 = load_tensor(weights_dir + "/conv3_1_kernel.weights")
        self.conv3_1_kernel = r9[1]^
        var r10 = load_tensor(weights_dir + "/conv3_1_bias.weights")
        self.conv3_1_bias = r10[1]^
        var r11 = load_tensor(weights_dir + "/conv3_2_kernel.weights")
        self.conv3_2_kernel = r11[1]^
        var r12 = load_tensor(weights_dir + "/conv3_2_bias.weights")
        self.conv3_2_bias = r12[1]^
        var r13 = load_tensor(weights_dir + "/conv3_3_kernel.weights")
        self.conv3_3_kernel = r13[1]^
        var r14 = load_tensor(weights_dir + "/conv3_3_bias.weights")
        self.conv3_3_bias = r14[1]^

        # Load Block 4
        var r15 = load_tensor(weights_dir + "/conv4_1_kernel.weights")
        self.conv4_1_kernel = r15[1]^
        var r16 = load_tensor(weights_dir + "/conv4_1_bias.weights")
        self.conv4_1_bias = r16[1]^
        var r17 = load_tensor(weights_dir + "/conv4_2_kernel.weights")
        self.conv4_2_kernel = r17[1]^
        var r18 = load_tensor(weights_dir + "/conv4_2_bias.weights")
        self.conv4_2_bias = r18[1]^
        var r19 = load_tensor(weights_dir + "/conv4_3_kernel.weights")
        self.conv4_3_kernel = r19[1]^
        var r20 = load_tensor(weights_dir + "/conv4_3_bias.weights")
        self.conv4_3_bias = r20[1]^

        # Load Block 5
        var r21 = load_tensor(weights_dir + "/conv5_1_kernel.weights")
        self.conv5_1_kernel = r21[1]^
        var r22 = load_tensor(weights_dir + "/conv5_1_bias.weights")
        self.conv5_1_bias = r22[1]^
        var r23 = load_tensor(weights_dir + "/conv5_2_kernel.weights")
        self.conv5_2_kernel = r23[1]^
        var r24 = load_tensor(weights_dir + "/conv5_2_bias.weights")
        self.conv5_2_bias = r24[1]^
        var r25 = load_tensor(weights_dir + "/conv5_3_kernel.weights")
        self.conv5_3_kernel = r25[1]^
        var r26 = load_tensor(weights_dir + "/conv5_3_bias.weights")
        self.conv5_3_bias = r26[1]^

        # Load FC layers
        var r27 = load_tensor(weights_dir + "/fc1_weights.weights")
        self.fc1_weights = r27[1]^
        var r28 = load_tensor(weights_dir + "/fc1_bias.weights")
        self.fc1_bias = r28[1]^
        var r29 = load_tensor(weights_dir + "/fc2_weights.weights")
        self.fc2_weights = r29[1]^
        var r30 = load_tensor(weights_dir + "/fc2_bias.weights")
        self.fc2_bias = r30[1]^
        var r31 = load_tensor(weights_dir + "/fc3_weights.weights")
        self.fc3_weights = r31[1]^
        var r32 = load_tensor(weights_dir + "/fc3_bias.weights")
        self.fc3_bias = r32[1]^
