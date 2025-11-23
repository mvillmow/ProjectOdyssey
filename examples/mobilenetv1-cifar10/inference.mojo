"""Inference Script for MobileNetV1 on CIFAR-10

This script loads a trained MobileNetV1 model and evaluates it on the CIFAR-10 test set.

Usage:
    mojo run examples/mobilenetv1-cifar10/inference.mojo --weights-dir mobilenetv1_weights

Features:
    - Loads saved model weights
    - Evaluates accuracy on CIFAR-10 test set
    - Reports per-class accuracy
    - Inference mode (no training, no batch norm running stats updates)
"""

from shared.core import ExTensor, zeros
from shared.data import extract_batch_pair, compute_num_batches
from model import MobileNetV1
from data_loader import load_cifar10_test


# CIFAR-10 class names
alias CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


fn evaluate_model(
    inout model: MobileNetV1,
    images: ExTensor,
    labels: ExTensor,
    batch_size: Int = 100,
    verbose: Bool = True
) raises -> (Float32, List[Int], List[Int]):
    """Evaluate model on a dataset."""
    var num_samples = images.shape[0]
    var num_batches = compute_num_batches(num_samples, batch_size)
    var total_correct = 0
    var correct_per_class = List[Int]()
    var total_per_class = List[Int]()

    for i in range(10):
        correct_per_class[i] = 0
        total_per_class[i] = 0

    if verbose:
        print("Evaluating on " + str(num_samples) + " samples (" + str(num_batches) + " batches)...")

    for batch_idx in range(num_batches):
        var start_idx = batch_idx * batch_size
        var batch_pair = extract_batch_pair(images, labels, start_idx, batch_size)
        var batch_images = batch_pair[0]
        var batch_labels = batch_pair[1]
        var current_batch_size = batch_images.shape[0]

        var logits = model.forward(batch_images, training=False)

        for i in range(current_batch_size):
            var sample_logits = zeros(List[Int]().append(10), DType.float32)
            var sample_logits_data = sample_logits._data.bitcast[Float32]()
            var logits_data = logits._data.bitcast[Float32]()

            var offset = i * 10
            for j in range(10):
                sample_logits_data[j] = logits_data[offset + j]

            var pred_class = 0
            var max_logit = sample_logits_data[0]
            for j in range(1, 10):
                if sample_logits_data[j] > max_logit:
                    max_logit = sample_logits_data[j]
                    pred_class = j

            var true_class = int(batch_labels[i])
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
    """Print detailed evaluation results."""
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
    print("MobileNetV1 Inference on CIFAR-10")
    print("=" * 60)
    print()

    var weights_dir = "mobilenetv1_weights"
    var batch_size = 100

    print("Configuration:")
    print("  Weights directory: " + str(weights_dir))
    print("  Batch size: " + str(batch_size))
    print()

    print("Loading CIFAR-10 test set...")
    var test_data = load_cifar10_test("datasets/cifar10")
    var test_images = test_data[0]
    var test_labels = test_data[1]
    print("  Test samples: " + str(test_images.shape[0]))
    print()

    print("Initializing MobileNetV1 model...")
    var model = MobileNetV1(num_classes=10)
    print("  Model architecture: MobileNetV1")
    print("  Parameters: ~4.2M")
    print("  Key feature: Depthwise separable convolutions")
    print()

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
        print("  python examples/mobilenetv1-cifar10/download_cifar10.py")
        print("  mojo run examples/mobilenetv1-cifar10/train.mojo --epochs 200")
        print()
        return

    print("Running inference on test set...")
    print()

    var results = evaluate_model(model, test_images, test_labels, batch_size, verbose=True)
    var accuracy = results[0]
    var correct_per_class = results[1]
    var total_per_class = results[2]

    print_detailed_results(accuracy, correct_per_class, total_per_class)

    print("Performance Context:")
    print("-" * 60)
    print("Expected MobileNetV1 accuracy on CIFAR-10:")
    print("  - Without data augmentation: 88-90%")
    print("  - With data augmentation: 90-92%")
    print("  - State-of-the-art: 92-94%")
    print()
    print("Efficiency highlights:")
    print("  - 8-9× fewer operations than standard convolutions")
    print("  - Smallest model: 4.2M parameters")
    print("  - Fastest training: ~25-35 hours (200 epochs)")
    print("-" * 60)
    print()

    print("Inference complete!")
