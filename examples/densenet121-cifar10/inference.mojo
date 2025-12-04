"""Inference Script for DenseNet-121 on CIFAR-10"""

from shared.core import ExTensor, zeros
from shared.data import extract_batch_pair, compute_num_batches
from shared.data.datasets import load_cifar10_test
from shared.training.metrics import evaluate_logits_batch
from model import DenseNet121

alias CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


fn evaluate_model(
    mut model: DenseNet121,
    images: ExTensor,
    labels: ExTensor,
    batch_size: Int = 100,
    verbose: Bool = True
) raises -> Tuple[Float32, List[Int], List[Int]]:
    """Evaluate model on a dataset.

    Args:
        model: DenseNet-121 model
        images: Input images (N, 3, 32, 32)
        labels: Ground truth labels (N,)
        batch_size: Batch size for evaluation
        verbose: Print progress during evaluation

    Returns:
        Tuple of (accuracy, correct_per_class, total_per_class)
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

        # Forward pass (inference mode)
        var logits = model.forward(batch_images, training=False)

        # Compute batch accuracy using shared function
        var batch_acc_fraction = evaluate_logits_batch(logits, batch_labels)
        var batch_correct = Int(batch_acc_fraction * Float32(current_batch_size))
        total_correct += batch_correct

        # Update per-class counters
        var logits_data = logits._data.bitcast[Float32]()
        for i in range(current_batch_size):
            # Find argmax (predicted class)
            var pred_class = 0
            var max_logit = logits_data[i * 10]
            for j in range(1, 10):
                if logits_data[i * 10 + j] > max_logit:
                    max_logit = logits_data[i * 10 + j]
                    pred_class = j

            # Get true label
            var true_class = int(batch_labels[i])

            # Update counters
            total_per_class[true_class] += 1
            if pred_class == true_class:
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
    print("=" * 60)
    print("DenseNet-121 Inference on CIFAR-10")
    print("=" * 60)
    print()

    var weights_dir = "densenet121_weights"
    var batch_size = 100

    print("Configuration:")
    print("  Weights directory: " + str(weights_dir))
    print("  Batch size: " + str(batch_size))
    print()

    print("Loading CIFAR-10 test set...")
    var test_data = load_cifar10_test("datasets/cifar10")
    var test_images = test_data[0]
    var test_labels = test_data[1]
    print("  Test samples: " + str(test_images.shape()[0]))
    print()

    print("Initializing DenseNet-121 model...")
    var model = DenseNet121(num_classes=10)
    print("  Model: DenseNet-121 (121 layers, dense connectivity)")
    print("  Parameters: ~7M")
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
        return

    print("Running inference on test set...")
    print()

    var results = evaluate_model(model, test_images, test_labels, batch_size, verbose=True)
    var accuracy = results[0]
    var correct_per_class = results[1]
    var total_per_class = results[2]

    print_detailed_results(accuracy, correct_per_class, total_per_class)

    print("Expected accuracy: 94-95% on CIFAR-10")
    print("Key feature: Dense connections ensure gradient flow")
    print()
