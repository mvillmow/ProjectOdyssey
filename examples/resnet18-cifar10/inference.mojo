"""Inference Script for ResNet-18 on CIFAR-10

This script loads a trained ResNet-18 model and evaluates it on the CIFAR-10 test set.

Usage:
    # Evaluate on test set
    mojo run examples/resnet18-cifar10/inference.mojo --weights-dir resnet18_weights

    # Evaluate on specific samples
    mojo run examples/resnet18-cifar10/inference.mojo --weights-dir resnet18_weights --samples 100

Features:
    - Loads saved model weights
    - Evaluates accuracy on CIFAR-10 test set
    - Reports per-class accuracy
    - Inference mode (no training, no batch norm running stats updates)
"""

from shared.core import ExTensor, zeros
from shared.data import extract_batch_pair, compute_num_batches
from model import ResNet18
from data_loader import load_cifar10_test


# CIFAR-10 class names
alias CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


fn evaluate_model(
    inout model: ResNet18,
    images: ExTensor,
    labels: ExTensor,
    batch_size: Int = 100,
    verbose: Bool = True
) raises -> Tuple[Float32, List[Int], List[Int]]:
    """Evaluate model on a dataset.

    Args:
        model: ResNet-18 model
        images: Input images (N, 3, 32, 32)
        labels: Ground truth labels (N,)
        batch_size: Batch size for evaluation
        verbose: Print progress during evaluation

    Returns:
        Tuple of (accuracy, correct_per_class, total_per_class)
        - accuracy: Overall accuracy as percentage (0-100)
        - correct_per_class: Correct predictions per class (10,)
        - total_per_class: Total samples per class (10,)
    """
    var num_samples = images.shape()[0]
    var num_batches = compute_num_batches(num_samples, batch_size)

    var total_correct = 0
    var correct_per_class = List[Int](capacity=10)
    var total_per_class = List[Int](capacity=10)

    # Initialize counters
    for i in range(10):
        correct_per_class.append(0)
        total_per_class.append(0)

    if verbose:
        print("Evaluating on " + str(num_samples) + " samples (" + str(num_batches) + " batches)...")

    # Evaluate in batches
    for batch_idx in range(num_batches):
        var start_idx = batch_idx * batch_size

        # Extract mini-batch
        var batch_pair = extract_batch_pair(images, labels, start_idx, batch_size)
        var batch_images = batch_pair[0]
        var batch_labels = batch_pair[1]
        var current_batch_size = batch_images.shape()[0]

        # Forward pass (inference mode - no BN running stats updates)
        var logits = model.forward(batch_images, training=False)

        # Compute predictions for each sample in batch
        for i in range(current_batch_size):
            # Extract logits for this sample (shape: (10,))
            var shape = List[Int](capacity=1)
            shape.append(10)
            var sample_logits = zeros(shape, DType.float32)
            var sample_logits_data = sample_logits._data.bitcast[Float32]()
            var logits_data = logits._data.bitcast[Float32]()

            # Copy logits for this sample
            var offset = i * 10
            for j in range(10):
                sample_logits_data[j] = logits_data[offset + j]

            # Find argmax (predicted class)
            var pred_class = 0
            var max_logit = sample_logits_data[0]
            for j in range(1, 10):
                if sample_logits_data[j] > max_logit:
                    max_logit = sample_logits_data[j]
                    pred_class = j

            # Get true label
            var true_class = int(batch_labels[i])

            # Update counters
            total_per_class[true_class] += 1
            if pred_class == true_class:
                total_correct += 1
                correct_per_class[true_class] += 1

        if verbose and (batch_idx + 1) % 20 == 0:
            var progress = Float32(batch_idx + 1) / Float32(num_batches) * 100.0
            var current_acc = Float32(total_correct) / Float32((batch_idx + 1) * batch_size) * 100.0
            print("  Progress: " + str(progress) + "% - Current Acc: " + str(current_acc) + "%")

    var overall_accuracy = Float32(total_correct) / Float32(num_samples) * 100.0

    if verbose:
        print("Evaluation complete!")
        print()

    return (overall_accuracy, correct_per_class, total_per_class)


fn print_detailed_results(
    accuracy: Float32,
    correct_per_class: List[Int],
    total_per_class: List[Int]
):
    """Print detailed evaluation results.

    Args:
        accuracy: Overall accuracy percentage
        correct_per_class: Correct predictions per class
        total_per_class: Total samples per class
    """
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print()

    print("Overall Accuracy: " + str(accuracy) + "%")
    print()

    print("Per-Class Accuracy:")
    print("-" * 60)
    print(str('Class') + " " + str('Correct') + " " + str('Total') + " " + str('Accuracy'))
    print("-" * 60)

    for i in range(10):
        var class_name = CLASS_NAMES[i]
        var correct = correct_per_class[i]
        var total = total_per_class[i]
        var class_acc = Float32(correct) / Float32(total) * 100.0 if total > 0 else Float32(0.0)

        print(str(class_name) + " " + str(correct) + " " + str(total) + " " + str(class_acc) + "%")

    print("-" * 60)
    print()


fn main() raises:
    """Main inference entry point."""
    print("=" * 60)
    print("ResNet-18 Inference on CIFAR-10")
    print("=" * 60)
    print()

    # Configuration
    var weights_dir = "resnet18_weights"  # Default weights directory
    var batch_size = 100  # Batch size for evaluation
    var evaluate_all = True  # Evaluate on full test set

    print("Configuration:")
    print("  Weights directory: " + str(weights_dir))
    print("  Batch size: " + str(batch_size))
    print("  Evaluate full test set: " + str(evaluate_all))
    print()

    # Load CIFAR-10 test set
    print("Loading CIFAR-10 test set...")
    var test_data = load_cifar10_test("datasets/cifar10")
    var test_images = test_data[0]
    var test_labels = test_data[1]

    var num_samples = test_images.shape()[0]
    print("  Test samples: " + str(num_samples))
    print("  Image shape: (3, 32, 32)")
    print("  Number of classes: 10")
    print()

    # Initialize model
    print("Initializing ResNet-18 model...")
    var model = ResNet18(num_classes=10)
    print("  Model architecture: ResNet-18")
    print("  Parameters: ~11M")
    print()

    # Load weights
    print("Loading weights from " + str(weights_dir) + "/...")
    try:
        model.load_weights(weights_dir)
        print("  ✓ Weights loaded successfully")
        print()
    except e:
        print("  ✗ Failed to load weights: " + str(e))
        print()
        print("ERROR: Cannot proceed without trained weights.")
        print()
        print("To train a model, run:")
        print("  python examples/resnet18-cifar10/download_cifar10.py")
        print("  mojo run examples/resnet18-cifar10/train.mojo --epochs 200")
        print()
        print("Note: Training requires batch_norm2d_backward implementation.")
        print("See GAP_ANALYSIS.md for details.")
        return

    # Run inference
    print("Running inference on test set...")
    print()

    var results = evaluate_model(model, test_images, test_labels, batch_size, verbose=True)
    var accuracy = results[0]
    var correct_per_class = results[1]
    var total_per_class = results[2]

    # Print detailed results
    print_detailed_results(accuracy, correct_per_class, total_per_class)

    # Performance context
    print("Performance Context:")
    print("-" * 60)
    print("Expected ResNet-18 accuracy on CIFAR-10:")
    print("  - Without data augmentation: 91-93%")
    print("  - With data augmentation: 93-95%")
    print("  - State-of-the-art: 96-97%")
    print()
    print("Training details:")
    print("  - 200 epochs recommended")
    print("  - SGD with momentum (0.9)")
    print("  - Learning rate: 0.01 with step decay")
    print("  - Batch size: 128")
    print("-" * 60)
    print()

    print("Inference complete!")
