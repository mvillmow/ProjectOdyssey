"""Inference Script for LeNet-5 on EMNIST

Runs inference with trained LeNet-5 model on EMNIST test set.

Usage:
    # Run on test set
    mojo run examples/lenet-emnist/inference.mojo --weights-dir lenet5_weights

Requirements:
    - Trained model weights (from train.mojo)
    - EMNIST dataset (for test set evaluation)

References:
    - LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998).
      Gradient-based learning applied to document recognition.
    - Reference Implementation: https://github.com/mattwang44/LeNet-from-Scratch
"""

from model import LeNet5
from data_loader import load_idx_labels, load_idx_images, normalize_images
from shared.core import ExTensor, zeros
from sys import argv
from collections.vector import DynamicVector
from math import exp


# EMNIST Balanced class mapping (47 classes)
# 0-9: digits, 10-35: uppercase letters, 36-46: lowercase letters (select)
var CLASS_NAMES = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",  # 0-9
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",  # 10-19
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",  # 20-29
    "U", "V", "W", "X", "Y", "Z",                      # 30-35
    "a", "b", "d", "e", "f", "g", "h", "n", "q", "r", "t"  # 36-46 (select lowercase)
]


fn parse_args() raises -> (String, String):
    """Parse command line arguments.

    Returns:
        Tuple of (weights_dir, data_dir)
    """
    var weights_dir = "lenet5_weights"
    var data_dir = "datasets/emnist"

    var args = argv()
    for i in range(len(args)):
        if args[i] == "--weights-dir" and i + 1 < len(args):
            weights_dir = args[i + 1]
        elif args[i] == "--data-dir" and i + 1 < len(args):
            data_dir = args[i + 1]

    return (weights_dir, data_dir)


fn infer_single(inout model: LeNet5, borrowed image: ExTensor) raises -> (Int, Float32):
    """Run inference on a single image.

    Args:
        model: Trained LeNet-5 model
        image: Input image of shape (1, 1, 28, 28)

    Returns:
        Tuple of (predicted_class, confidence)
    """
    var logits = model.forward(image)

    # Find argmax and max value
    var num_classes = logits.shape()[1]
    var max_idx = 0
    var max_val_data = logits._data.bitcast[Float32]()
    var max_val = max_val_data[0]

    for i in range(1, num_classes):
        var val = max_val_data[i]
        if val > max_val:
            max_val = val
            max_idx = i

    # Compute softmax for confidence (simplified - just exp(max) / sum(exp))
    var exp_sum = Float32(0.0)
    for i in range(num_classes):
        exp_sum += exp(max_val_data[i])

    var confidence = exp(max_val) / exp_sum

    return (max_idx, confidence)


fn evaluate_test_set(
    inout model: LeNet5,
    borrowed images: ExTensor,
    borrowed labels: ExTensor
) raises -> (Float32, Int, Int):
    """Evaluate model on entire test set.

    Args:
        model: Trained LeNet-5 model
        images: Test images of shape (num_samples, 1, 28, 28)
        labels: Test labels of shape (num_samples,)

    Returns:
        Tuple of (accuracy, num_correct, num_total)
    """
    var num_samples = images.shape()[0]
    var correct = 0

    print("Evaluating on", num_samples, "test samples...")

    # TODO: Process in batches when slicing is fully supported
    # For now, evaluate on first 1000 samples
    var eval_samples = min(1000, num_samples)

    for i in range(eval_samples):
        # TODO: Extract single sample when slicing works
        # For now, we'll use the entire dataset (inefficient but demonstrates structure)
        var pred_class = model.predict(images)
        var labels_data = labels._data  # uint8
        var true_label = Int(labels_data[i])

        if pred_class == true_label:
            correct += 1

        # Print progress every 100 samples
        if (i + 1) % 100 == 0:
            var current_acc = Float32(correct) / Float32(i + 1)
            print("  Processed [", i + 1, "/", eval_samples, "] - Accuracy: ", current_acc * 100.0, "%")

        # Break after first iteration for demonstration
        # Remove when slicing is implemented
        break

    var accuracy = Float32(correct) / Float32(eval_samples)

    return (accuracy, correct, eval_samples)


fn main() raises:
    """Main inference entry point."""
    print("=" * 60)
    print("LeNet-5 Inference on EMNIST Dataset")
    print("=" * 60)

    # Parse arguments
    var config = parse_args()
    var weights_dir = config[0]
    var data_dir = config[1]

    print("\nConfiguration:")
    print("  Weights Directory: ", weights_dir)
    print("  Data Directory: ", data_dir)
    print()

    # Initialize model
    print("Initializing LeNet-5 model...")
    var model = LeNet5(num_classes=47)
    print("  Model initialized with", model.num_classes, "classes")
    print()

    # Load weights
    print("Loading model weights...")
    model.load_weights(weights_dir)
    print("  Weights loaded from", weights_dir)
    print()

    # Load test dataset
    print("Loading EMNIST test set...")
    var test_images_path = data_dir + "/emnist-balanced-test-images-idx3-ubyte"
    var test_labels_path = data_dir + "/emnist-balanced-test-labels-idx1-ubyte"

    var test_images_raw = load_idx_images(test_images_path)
    var test_labels = load_idx_labels(test_labels_path)

    # Normalize images
    var test_images = normalize_images(test_images_raw)

    print("  Test samples: ", test_images.shape()[0])
    print()

    # Run inference on test set
    print("Running inference on test set...")
    var result = evaluate_test_set(model, test_images, test_labels)
    var accuracy = result[0]
    var correct = result[1]
    var total = result[2]

    print()
    print("Results:")
    print("  Correct:", correct, "/", total)
    print("  Accuracy:", accuracy * 100.0, "%")
    print()

    print("Inference complete!")
    print("\nNote: This implementation demonstrates the inference structure.")
    print("Batch processing will be more efficient when tensor slicing is optimized.")
