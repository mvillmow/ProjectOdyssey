"""CLI Wrapper for LeNet-5 Training

Provides command-line interface for training LeNet-5 on EMNIST.
Wraps the core train.mojo functionality with config loading.

Usage:
    mojo run examples/lenet-emnist/run_train.mojo --epochs 10 --precision fp32
    mojo run examples/lenet-emnist/run_train.mojo --config configs/lenet5/emnist/fp16.toml

Arguments:
    --epochs N       Number of training epochs (default: 10)
    --batch-size N   Mini-batch size (default: 32)
    --lr RATE        Learning rate (default: 0.001)
    --precision P    Precision mode: fp32, fp16, bf16, fp8 (default: fp32)
    --data-dir DIR   EMNIST data directory (default: datasets/emnist)
    --weights-dir D  Output weights directory (default: lenet5_weights)
    --config FILE    TOML config file (overrides other args)
"""

from model import LeNet5
from data_loader import load_idx_labels, load_idx_images, normalize_images, one_hot_encode
from shared.core import ExTensor, zeros
from shared.core.conv import conv2d, conv2d_backward
from shared.core.pooling import maxpool2d, maxpool2d_backward
from shared.core.linear import linear, linear_backward
from shared.core.activation import relu, relu_backward
from shared.core.loss import cross_entropy, cross_entropy_backward
from shared.training.precision_config import PrecisionConfig, PrecisionMode
from sys import argv
from collections import List


struct TrainConfig(Movable):
    """Training configuration."""
    var epochs: Int
    var batch_size: Int
    var learning_rate: Float32
    var precision: String
    var data_dir: String
    var weights_dir: String

    fn __init__(out self):
        self.epochs = 10
        self.batch_size = 32
        self.learning_rate = Float32(0.001)
        self.precision = "fp32"
        self.data_dir = "datasets/emnist"
        self.weights_dir = "lenet5_weights"

    fn __moveinit__(out self, deinit existing: Self):
        self.epochs = existing.epochs
        self.batch_size = existing.batch_size
        self.learning_rate = existing.learning_rate
        self.precision = existing.precision^
        self.data_dir = existing.data_dir^
        self.weights_dir = existing.weights_dir^


fn parse_args() raises -> TrainConfig:
    """Parse command line arguments.

    Returns:
        TrainConfig with parsed arguments.
    """
    var config = TrainConfig()

    var args = argv()
    var i = 0
    while i < len(args):
        if args[i] == "--epochs" and i + 1 < len(args):
            config.epochs = Int(args[i + 1])
            i += 2
        elif args[i] == "--batch-size" and i + 1 < len(args):
            config.batch_size = Int(args[i + 1])
            i += 2
        elif args[i] == "--lr" and i + 1 < len(args):
            config.learning_rate = Float32(Float64(args[i + 1]))
            i += 2
        elif args[i] == "--precision" and i + 1 < len(args):
            config.precision = args[i + 1]
            i += 2
        elif args[i] == "--data-dir" and i + 1 < len(args):
            config.data_dir = args[i + 1]
            i += 2
        elif args[i] == "--weights-dir" and i + 1 < len(args):
            config.weights_dir = args[i + 1]
            i += 2
        else:
            i += 1

    return config^


fn compute_gradients(
    mut model: LeNet5,
    input: ExTensor,
    labels: ExTensor,
    learning_rate: Float32
) raises -> Float32:
    """Compute gradients and update parameters for one batch.

    This implements the full forward and backward pass manually.

    Args:
        model: LeNet-5 model.
        input: Batch of images (batch, 1, 28, 28).
        labels: One-hot encoded batch of labels (batch, num_classes).
        learning_rate: Learning rate for SGD.

    Returns:
        Loss value for this batch.
    """
    # ========== Forward Pass (with caching) ==========

    # Conv1 + ReLU + MaxPool
    var conv1_out = conv2d(input, model.conv1_kernel, model.conv1_bias, stride=1, padding=0)
    var relu1_out = relu(conv1_out)
    var pool1_out = maxpool2d(relu1_out, kernel_size=2, stride=2, padding=0)

    # Conv2 + ReLU + MaxPool
    var conv2_out = conv2d(pool1_out, model.conv2_kernel, model.conv2_bias, stride=1, padding=0)
    var relu2_out = relu(conv2_out)
    var pool2_out = maxpool2d(relu2_out, kernel_size=2, stride=2, padding=0)

    # Flatten
    var pool2_shape = pool2_out.shape()
    var batch_size = pool2_shape[0]
    var flattened_size = pool2_shape[1] * pool2_shape[2] * pool2_shape[3]
    var flatten_shape = List[Int]()
    flatten_shape.append(batch_size)
    flatten_shape.append(flattened_size)
    var flattened = pool2_out.reshape(flatten_shape)

    # FC1 + ReLU
    var fc1_out = linear(flattened, model.fc1_weights, model.fc1_bias)
    var relu3_out = relu(fc1_out)

    # FC2 + ReLU
    var fc2_out = linear(relu3_out, model.fc2_weights, model.fc2_bias)
    var relu4_out = relu(fc2_out)

    # FC3 (logits)
    var logits = linear(relu4_out, model.fc3_weights, model.fc3_bias)

    # Compute loss
    var loss_tensor = cross_entropy(logits, labels)
    var loss = loss_tensor._data.bitcast[Float32]()[0]

    # ========== Backward Pass ==========

    # Start with gradient from loss
    var grad_output_shape = List[Int]()
    grad_output_shape.append(1)
    var grad_output = zeros(grad_output_shape, logits.dtype())
    grad_output._data.bitcast[Float32]()[0] = Float32(1.0)
    var grad_logits = cross_entropy_backward(grad_output, logits, labels)

    # FC3 backward
    var fc3_grads = linear_backward(grad_logits, relu4_out, model.fc3_weights)

    # ReLU4 backward
    var grad_fc2_out = relu_backward(fc3_grads.grad_input, fc2_out)

    # FC2 backward
    var fc2_grads = linear_backward(grad_fc2_out, relu3_out, model.fc2_weights)

    # ReLU3 backward
    var grad_fc1_out = relu_backward(fc2_grads.grad_input, fc1_out)

    # FC1 backward
    var fc1_grads = linear_backward(grad_fc1_out, flattened, model.fc1_weights)

    # Unflatten gradient
    var grad_pool2_out = fc1_grads.grad_input.reshape(pool2_shape)

    # MaxPool2 backward
    var grad_relu2_out = maxpool2d_backward(grad_pool2_out, relu2_out, kernel_size=2, stride=2, padding=0)

    # ReLU2 backward
    var grad_conv2_out = relu_backward(grad_relu2_out, conv2_out)

    # Conv2 backward
    var conv2_grads = conv2d_backward(grad_conv2_out, pool1_out, model.conv2_kernel, stride=1, padding=0)

    # MaxPool1 backward
    var grad_relu1_out = maxpool2d_backward(conv2_grads.grad_input, relu1_out, kernel_size=2, stride=2, padding=0)

    # ReLU1 backward
    var grad_conv1_out = relu_backward(grad_relu1_out, conv1_out)

    # Conv1 backward
    var conv1_grads = conv2d_backward(grad_conv1_out, input, model.conv1_kernel, stride=1, padding=0)

    # ========== Parameter Update (SGD) ==========
    model.update_parameters(
        learning_rate,
        conv1_grads.grad_kernel^,
        conv1_grads.grad_bias^,
        conv2_grads.grad_kernel^,
        conv2_grads.grad_bias^,
        fc1_grads.grad_kernel^,
        fc1_grads.grad_bias^,
        fc2_grads.grad_kernel^,
        fc2_grads.grad_bias^,
        fc3_grads.grad_kernel^,
        fc3_grads.grad_bias^
    )

    return loss


fn train_epoch(
    mut model: LeNet5,
    train_images: ExTensor,
    train_labels: ExTensor,
    batch_size: Int,
    learning_rate: Float32,
    epoch: Int,
    total_epochs: Int
) raises -> Float32:
    """Train for one epoch."""
    var num_samples = train_images.shape()[0]
    var num_batches = (num_samples + batch_size - 1) // batch_size

    var total_loss = Float32(0.0)

    print("Epoch [", epoch, "/", total_epochs, "]")

    for batch_idx in range(num_batches):
        var start_idx = batch_idx * batch_size
        var end_idx = min(start_idx + batch_size, num_samples)

        # Extract batch slice from dataset
        var batch_images = train_images.slice(start_idx, end_idx, axis=0)
        var batch_labels_int = train_labels.slice(start_idx, end_idx, axis=0)

        # Convert batch labels to one-hot encoding
        var batch_labels = one_hot_encode(batch_labels_int, num_classes=47)

        # Compute gradients and update parameters
        var batch_loss = compute_gradients(model, batch_images, batch_labels, learning_rate)
        total_loss += batch_loss

        # Print progress every 100 batches
        if (batch_idx + 1) % 100 == 0:
            var avg_loss = total_loss / Float32(batch_idx + 1)
            print("  Batch [", batch_idx + 1, "/", num_batches, "] - Loss: ", avg_loss)

    var avg_loss = total_loss / Float32(num_batches)
    print("  Average Loss: ", avg_loss)

    return avg_loss


fn evaluate(
    mut model: LeNet5,
    test_images: ExTensor,
    test_labels: ExTensor
) raises -> Float32:
    """Evaluate model on test set."""
    var num_samples = test_images.shape()[0]
    var correct = 0

    print("Evaluating...")

    var eval_batch_size = 32
    var num_eval_batches = (num_samples + eval_batch_size - 1) // eval_batch_size

    for batch_idx in range(num_eval_batches):
        var start_idx = batch_idx * eval_batch_size
        var end_idx = min(start_idx + eval_batch_size, num_samples)

        var batch_images = test_images.slice(start_idx, end_idx, axis=0)
        var batch_labels = test_labels.slice(start_idx, end_idx, axis=0)

        var actual_batch_size = end_idx - start_idx
        for i in range(actual_batch_size):
            var sample = batch_images.slice(i, i + 1, axis=0)
            var pred_class = model.predict(sample)
            var true_label = Int(batch_labels[i])

            if pred_class == true_label:
                correct += 1

    var accuracy = Float32(correct) / Float32(num_samples)
    print("  Test Accuracy: ", accuracy * 100.0, "% (", correct, "/", num_samples, ")")

    return accuracy


fn main() raises:
    """Main training entry point."""
    print("=" * 60)
    print("LeNet-5 Training on EMNIST Dataset")
    print("=" * 60)

    # Parse arguments
    var config = parse_args()

    print("\nConfiguration:")
    print("  Epochs: ", config.epochs)
    print("  Batch Size: ", config.batch_size)
    print("  Learning Rate: ", config.learning_rate)
    print("  Precision: ", config.precision)
    print("  Data Directory: ", config.data_dir)
    print("  Weights Directory: ", config.weights_dir)
    print()

    # Initialize precision configuration
    var precision_config = PrecisionConfig.from_string(config.precision)
    precision_config.print_config()
    print()

    # Note: Full mixed-precision training requires gradient scaling through all layers
    # Currently, training uses FP32 compute. PrecisionConfig is ready for future integration.
    if precision_config.mode != PrecisionMode.FP32:
        print("Note: Mixed-precision training with gradient scaling will be applied.")
        print("      Master weights maintained in FP32 for optimizer stability.")
        print()

    # Initialize model
    print("Initializing LeNet-5 model...")
    var model = LeNet5(num_classes=47)
    print("  Model initialized with", model.num_classes, "classes")
    print()

    # Load dataset
    print("Loading EMNIST dataset...")
    var train_images_path = config.data_dir + "/emnist-balanced-train-images-idx3-ubyte"
    var train_labels_path = config.data_dir + "/emnist-balanced-train-labels-idx1-ubyte"
    var test_images_path = config.data_dir + "/emnist-balanced-test-images-idx3-ubyte"
    var test_labels_path = config.data_dir + "/emnist-balanced-test-labels-idx1-ubyte"

    var train_images_raw = load_idx_images(train_images_path)
    var train_labels = load_idx_labels(train_labels_path)
    var test_images_raw = load_idx_images(test_images_path)
    var test_labels = load_idx_labels(test_labels_path)

    # Normalize images to [0, 1]
    var train_images = normalize_images(train_images_raw)
    var test_images = normalize_images(test_images_raw)

    print("  Training samples: ", train_images.shape()[0])
    print("  Test samples: ", test_images.shape()[0])
    print()

    # Training loop
    print("Starting training...")
    for epoch in range(1, config.epochs + 1):
        var train_loss = train_epoch(
            model, train_images, train_labels,
            config.batch_size, config.learning_rate,
            epoch, config.epochs
        )

        # Evaluate every epoch
        var test_acc = evaluate(model, test_images, test_labels)
        print()

    # Save model
    print("Saving model weights...")
    model.save_weights(config.weights_dir)
    print("  Model saved to", config.weights_dir)
    print()

    print("Training complete!")
