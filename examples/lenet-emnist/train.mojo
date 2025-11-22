"""Training Script for LeNet-5 on EMNIST

Implements training with manual backward passes (no autograd).
Uses SGD optimizer with simple training loop.

Usage:
    mojo run examples/lenet-emnist/train.mojo --epochs 10 --batch-size 32 --lr 0.01

Requirements:
    - EMNIST dataset downloaded (run: python scripts/download_emnist.py)
    - Dataset location: datasets/emnist/

References:
    - LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998).
      Gradient-based learning applied to document recognition.
    - Reference Implementation: https://github.com/mattwang44/LeNet-from-Scratch
"""

from model import LeNet5
from data_loader import load_idx_labels, load_idx_images, normalize_images
from shared.core import ExTensor, zeros
from shared.core.conv import conv2d, conv2d_backward
from shared.core.pooling import maxpool2d, maxpool2d_backward
from shared.core.linear import linear, linear_backward
from shared.core.activation import relu, relu_backward
from shared.core.loss import cross_entropy, cross_entropy_backward
from sys import argv
from collections import List


struct TrainConfig:
    """Training configuration from command line arguments."""
    var epochs: Int
    var batch_size: Int
    var learning_rate: Float32
    var data_dir: String
    var weights_dir: String

    fn __init__(out self, epochs: Int, batch_size: Int, learning_rate: Float32, data_dir: String, weights_dir: String):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.data_dir = data_dir
        self.weights_dir = weights_dir


fn parse_args() raises -> TrainConfig:
    """Parse command line arguments.

    Returns:
        TrainConfig with parsed arguments
    """
    var epochs = 10
    var batch_size = 32
    var learning_rate = Float32(0.01)
    var data_dir = String("datasets/emnist")
    var weights_dir = String("lenet5_weights")

    var args = argv()
    for i in range(len(args)):
        if args[i] == "--epochs" and i + 1 < len(args):
            epochs = Int(args[i + 1])
        elif args[i] == "--batch-size" and i + 1 < len(args):
            batch_size = Int(args[i + 1])
        elif args[i] == "--lr" and i + 1 < len(args):
            learning_rate = Float32(Float64(args[i + 1]))
        elif args[i] == "--data-dir" and i + 1 < len(args):
            data_dir = args[i + 1]
        elif args[i] == "--weights-dir" and i + 1 < len(args):
            weights_dir = args[i + 1]

    return TrainConfig(epochs, batch_size, learning_rate, data_dir, weights_dir)


fn compute_gradients(
    mut model: LeNet5,
    input: ExTensor,
    labels: ExTensor,
    learning_rate: Float32
) raises -> Float32:
    """Compute gradients and update parameters for one batch.

    This implements the full forward and backward pass manually.

    Args:
        model: LeNet-5 model
        input: Batch of images (batch, 1, 28, 28)
        labels: Batch of labels (batch,)
        learning_rate: Learning rate for SGD

    Returns:
        Loss value for this batch
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
    # For cross-entropy with mean reduction, the initial gradient is 1.0 / batch_size
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
        fc1_grads.grad_weights^,
        fc1_grads.grad_bias^,
        fc2_grads.grad_weights^,
        fc2_grads.grad_bias^,
        fc3_grads.grad_weights^,
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
    """Train for one epoch.

    Args:
        model: LeNet-5 model
        train_images: Training images (num_samples, 1, 28, 28)
        train_labels: Training labels (num_samples,)
        batch_size: Mini-batch size
        learning_rate: Learning rate for SGD
        epoch: Current epoch number (1-indexed)
        total_epochs: Total number of epochs

    Returns:
        Average loss for the epoch
    """
    var num_samples = train_images.shape()[0]
    var num_batches = (num_samples + batch_size - 1) // batch_size

    var total_loss = Float32(0.0)

    print("Epoch [", epoch, "/", total_epochs, "]")

    for batch_idx in range(num_batches):
        var start_idx = batch_idx * batch_size
        var end_idx = min(start_idx + batch_size, num_samples)

        # Extract batch slice from dataset (zero-copy view)
        var batch_images = train_images.slice(start_idx, end_idx, axis=0)
        var batch_labels = train_labels.slice(start_idx, end_idx, axis=0)

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
    """Evaluate model on test set.

    Args:
        model: LeNet-5 model
        test_images: Test images (num_samples, 1, 28, 28)
        test_labels: Test labels (num_samples,)

    Returns:
        Test accuracy (0.0 to 1.0)
    """
    var num_samples = test_images.shape()[0]
    var correct = 0

    print("Evaluating...")

    # Evaluate in batches to avoid memory issues
    var eval_batch_size = 32
    var num_eval_batches = (num_samples + eval_batch_size - 1) // eval_batch_size

    for batch_idx in range(num_eval_batches):
        var start_idx = batch_idx * eval_batch_size
        var end_idx = min(start_idx + eval_batch_size, num_samples)

        # Extract batch slice
        var batch_images = test_images.slice(start_idx, end_idx, axis=0)
        var batch_labels = test_labels.slice(start_idx, end_idx, axis=0)

        # Process each sample in the batch
        var actual_batch_size = end_idx - start_idx
        for i in range(actual_batch_size):
            # Extract single sample from batch
            var sample = batch_images.slice(i, i + 1, axis=0)
            var pred_class = model.predict(sample)
            var true_label = Int(batch_labels[i])

            if pred_class == true_label:
                correct += 1

    var accuracy = Float32(correct) / Float32(num_samples)
    print("  Test Accuracy: ", accuracy * 100.0, "% (", correct, "/", num_samples, ")")

    return accuracy


fn main() raises:
    """Main training loop."""
    print("=" * 60)
    print("LeNet-5 Training on EMNIST Dataset")
    print("=" * 60)

    # Parse arguments
    var config = parse_args()
    var epochs = config.epochs
    var batch_size = config.batch_size
    var learning_rate = config.learning_rate
    var data_dir = config.data_dir
    var weights_dir = config.weights_dir

    print("\nConfiguration:")
    print("  Epochs: ", epochs)
    print("  Batch Size: ", batch_size)
    print("  Learning Rate: ", learning_rate)
    print("  Data Directory: ", data_dir)
    print("  Weights Directory: ", weights_dir)
    print()

    # Initialize model
    print("Initializing LeNet-5 model...")
    var model = LeNet5(num_classes=47)
    print("  Model initialized with", model.num_classes, "classes")
    print()

    # Load dataset
    print("Loading EMNIST dataset...")
    var train_images_path = data_dir + "/emnist-balanced-train-images-idx3-ubyte"
    var train_labels_path = data_dir + "/emnist-balanced-train-labels-idx1-ubyte"
    var test_images_path = data_dir + "/emnist-balanced-test-images-idx3-ubyte"
    var test_labels_path = data_dir + "/emnist-balanced-test-labels-idx1-ubyte"

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
    for epoch in range(1, epochs + 1):
        var train_loss = train_epoch(model, train_images, train_labels, batch_size, learning_rate, epoch, epochs)

        # Evaluate every epoch
        var test_acc = evaluate(model, test_images, test_labels)
        print()

    # Save model
    print("Saving model weights...")
    model.save_weights(weights_dir)
    print("  Model saved to", weights_dir)
    print()

    print("Training complete!")
    print("\nNote: This implementation demonstrates the full training structure.")
    print("Batch processing will be more efficient when tensor slicing is optimized.")
