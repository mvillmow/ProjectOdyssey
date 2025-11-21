"""Training Script for LeNet-5 on EMNIST

Simple training script following KISS principles. Trains LeNet-5 model on EMNIST dataset.

Usage:
    mojo run examples/lenet-emnist/train.mojo [--epochs 10] [--batch-size 32]

Requirements:
    - EMNIST dataset downloaded (run: python scripts/download_emnist.py)
    - Dataset location: datasets/emnist/

References:
    - LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998).
      Gradient-based learning applied to document recognition.
    - Reference Implementation: https://github.com/mattwang44/LeNet-from-Scratch
"""

from model import LeNet5
from shared.core import ExTensor, zeros
from shared.core.loss import cross_entropy_loss
from shared.data import ExTensorDataset, BatchLoader, Normalize, ToExTensor, Compose
from sys import argv
from collections.vector import DynamicVector
from pathlib import Path


fn load_idx_labels(filepath: String) raises -> ExTensor:
    """Load labels from IDX file format.

    Args:
        filepath: Path to IDX labels file

    Returns:
        ExTensor of shape (num_samples,) with label values

    Note:
        IDX format: magic (4 bytes) | num_items (4 bytes) | data
    """
    # TODO: Implement when Mojo file I/O is stable
    raise Error("IDX loading not yet implemented - waiting for stable Mojo file I/O")


fn load_idx_images(filepath: String) raises -> ExTensor:
    """Load images from IDX file format.

    Args:
        filepath: Path to IDX images file

    Returns:
        ExTensor of shape (num_samples, height, width) with pixel values

    Note:
        IDX format: magic (4 bytes) | num_items (4 bytes) | rows (4 bytes) | cols (4 bytes) | data
    """
    # TODO: Implement when Mojo file I/O is stable
    raise Error("IDX loading not yet implemented - waiting for stable Mojo file I/O")


fn train_epoch(
    inout model: LeNet5,
    borrowed train_loader: BatchLoader,
    learning_rate: Float32,
    epoch: Int,
    total_epochs: Int
) raises -> Float32:
    """Train for one epoch.

    Args:
        model: LeNet-5 model to train
        train_loader: Data loader for training data
        learning_rate: Learning rate for SGD
        epoch: Current epoch number (1-indexed)
        total_epochs: Total number of epochs

    Returns:
        Average loss for the epoch
    """
    var total_loss = Float32(0.0)
    var num_batches = 0

    print("Epoch [", epoch, "/", total_epochs, "]")

    # Iterate over batches
    for batch in train_loader:
        # Forward pass
        var logits = model.forward(batch.data)

        # Compute loss
        var loss = cross_entropy_loss(logits, batch.labels)
        total_loss += loss

        # Backward pass and parameter update
        # TODO: Implement backward pass when autograd is ready
        # For now, this is a placeholder showing the structure

        num_batches += 1

        # Print progress every 100 batches
        if num_batches % 100 == 0:
            var avg_loss = total_loss / Float32(num_batches)
            print("  Batch [", num_batches, "] - Loss: ", avg_loss)

    var avg_loss = total_loss / Float32(num_batches)
    print("  Average Loss: ", avg_loss)

    return avg_loss


fn evaluate(
    inout model: LeNet5,
    borrowed test_loader: BatchLoader
) raises -> Float32:
    """Evaluate model on test set.

    Args:
        model: LeNet-5 model to evaluate
        test_loader: Data loader for test data

    Returns:
        Test accuracy (0.0 to 1.0)
    """
    var correct = 0
    var total = 0

    print("Evaluating...")

    for batch in test_loader:
        var logits = model.forward(batch.data)

        # Get predictions (argmax)
        var batch_size = logits.shape()[0]
        var num_classes = logits.shape()[1]

        for i in range(batch_size):
            var max_idx = 0
            var max_val = logits[i * num_classes + 0]

            for j in range(1, num_classes):
                var val = logits[i * num_classes + j]
                if val > max_val:
                    max_val = val
                    max_idx = j

            var label = Int(batch.labels[i])
            if max_idx == label:
                correct += 1

            total += 1

    var accuracy = Float32(correct) / Float32(total)
    print("  Test Accuracy: ", accuracy * 100.0, "%")

    return accuracy


fn parse_args() raises -> (Int, Int, Float32, String):
    """Parse command line arguments.

    Returns:
        Tuple of (epochs, batch_size, learning_rate, data_dir)
    """
    var epochs = 10
    var batch_size = 32
    var learning_rate = Float32(0.01)
    var data_dir = "datasets/emnist"

    # Simple argument parsing (can be improved)
    var args = argv()
    for i in range(len(args)):
        if args[i] == "--epochs" and i + 1 < len(args):
            epochs = int(args[i + 1])
        elif args[i] == "--batch-size" and i + 1 < len(args):
            batch_size = int(args[i + 1])
        elif args[i] == "--lr" and i + 1 < len(args):
            learning_rate = Float32(float(args[i + 1]))
        elif args[i] == "--data-dir" and i + 1 < len(args):
            data_dir = args[i + 1]

    return (epochs, batch_size, learning_rate, data_dir)


fn main() raises:
    """Main training loop."""
    print("=" * 60)
    print("LeNet-5 Training on EMNIST Dataset")
    print("=" * 60)

    # Parse arguments
    var config = parse_args()
    var epochs = config[0]
    var batch_size = config[1]
    var learning_rate = config[2]
    var data_dir = config[3]

    print("Configuration:")
    print("  Epochs: ", epochs)
    print("  Batch Size: ", batch_size)
    print("  Learning Rate: ", learning_rate)
    print("  Data Directory: ", data_dir)
    print()

    # Initialize model (47 classes for EMNIST Balanced)
    print("Initializing LeNet-5 model...")
    var model = LeNet5(num_classes=47)
    print("  Model initialized with", model.num_classes, "classes")
    print()

    # Load dataset
    print("Loading EMNIST dataset...")
    print("  Note: Dataset loading from IDX files not yet implemented")
    print("  Waiting for stable Mojo file I/O")
    print()

    # TODO: Load dataset when file I/O is stable
    # var train_images = load_idx_images(data_dir + "/emnist-balanced-train-images-idx3-ubyte")
    # var train_labels = load_idx_labels(data_dir + "/emnist-balanced-train-labels-idx1-ubyte")
    # var test_images = load_idx_images(data_dir + "/emnist-balanced-test-images-idx3-ubyte")
    # var test_labels = load_idx_labels(data_dir + "/emnist-balanced-test-labels-idx1-ubyte")

    # Create data loaders
    # var train_dataset = ExTensorDataset(train_images, train_labels)
    # var train_loader = BatchLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # var test_dataset = ExTensorDataset(test_images, test_labels)
    # var test_loader = BatchLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Training loop
    # for epoch in range(1, epochs + 1):
    #     var train_loss = train_epoch(model, train_loader, learning_rate, epoch, epochs)
    #     var test_acc = evaluate(model, test_loader)
    #     print()

    # Save model
    # print("Saving model...")
    # model.save_weights("lenet5_emnist.weights")
    # print("  Model saved to lenet5_emnist.weights")

    print("Training complete!")
    print()
    print("Note: This is a skeleton implementation demonstrating the structure.")
    print("Full training will be available when Mojo file I/O and autograd are stable.")
