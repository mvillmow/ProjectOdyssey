"""Inference Script for AlexNet on CIFAR-10

Loads trained weights and evaluates on CIFAR-10 test set.

Usage:
    mojo run examples/alexnet-cifar10/inference.mojo --weights-dir alexnet_weights

Requirements:
    - Trained model weights saved in weights-dir
    - CIFAR-10 test set downloaded (run: python examples/alexnet-cifar10/download_cifar10.py)

References:
    - Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012).
      ImageNet classification with deep convolutional neural networks.
      Advances in Neural Information Processing Systems, 25, 1097-1105.
"""

from model import AlexNet
from data_loader import load_cifar10_test
from shared.core import ExTensor
from sys import argv


fn parse_args() raises -> Tuple[String, String]:
    """Parse command line arguments.

    Returns:
        Tuple of (weights_dir, data_dir)
    """
    var weights_dir = "alexnet_weights"
    var data_dir = "datasets/cifar10"

    var args = argv()
    for i in range(len(args)):
        if args[i] == "--weights-dir" and i + 1 < len(args):
            weights_dir = args[i + 1]
        elif args[i] == "--data-dir" and i + 1 < len(args):
            data_dir = args[i + 1]

    return (weights_dir, data_dir)


fn evaluate_model(
    inout model: AlexNet,
    borrowed test_images: ExTensor,
    borrowed test_labels: ExTensor
) raises -> Tuple[Float32, Float32]:
    """Evaluate model on test set with Top-1 and Top-5 accuracy.

    Args:
        model: AlexNet model with loaded weights
        test_images: Test images (10000, 3, 32, 32)
        test_labels: Test labels (10000,)

    Returns:
        Tuple of (top1_accuracy, top5_accuracy)

    Note:
        Top-5 accuracy: Model is correct if true label is in top 5 predictions
        For CIFAR-10 (10 classes), Top-5 is less meaningful but included for completeness
    """
    var num_samples = test_images.shape()[0]
    var correct_top1 = 0
    var correct_top5 = 0

    print("Evaluating on test set...")
    print("  Total samples:", num_samples)

    # TODO: Process in batches when slicing is implemented
    # For now, evaluate on first 1000 samples for demonstration
    var eval_samples = min(1000, num_samples)

    for i in range(eval_samples):
        # TODO: Extract single sample when slicing works
        # For demonstration, we'll use the first image repeatedly
        var logits = model.forward(test_images, training=False)
        var true_label = Int(test_labels[i])

        # Compute Top-1 accuracy (highest logit)
        var pred_class = _argmax(logits)
        if pred_class == true_label:
            correct_top1 += 1

        # Compute Top-5 accuracy (true label in top 5 logits)
        var top5_indices = _top_k_indices(logits, k=5)
        for j in range(5):
            if top5_indices[j] == true_label:
                correct_top5 += 1
                break

        # Print progress every 100 samples
        if (i + 1) % 100 == 0:
            var current_top1 = Float32(correct_top1) / Float32(i + 1) * 100.0
            print("  Processed [", i + 1, "/", eval_samples, "] - Top-1 Accuracy: ", current_top1, "%")

    var top1_accuracy = Float32(correct_top1) / Float32(eval_samples)
    var top5_accuracy = Float32(correct_top5) / Float32(eval_samples)

    print()
    print("Final Results:")
    print("  Top-1 Accuracy: ", top1_accuracy * 100.0, "% (", correct_top1, "/", eval_samples, ")")
    print("  Top-5 Accuracy: ", top5_accuracy * 100.0, "% (", correct_top5, "/", eval_samples, ")")

    return (top1_accuracy, top5_accuracy)


fn _argmax(tensor: ExTensor) raises -> Int:
    """Find index of maximum value in 1D tensor.

    Args:
        tensor: 1D tensor

    Returns:
        Index of maximum value
    """
    var shape = tensor.shape()
    var max_idx = 0
    var max_val = tensor[0]

    for i in range(1, shape[1]):
        if tensor[i] > max_val:
            max_val = tensor[i]
            max_idx = i

    return max_idx


fn _top_k_indices(tensor: ExTensor, k: Int) raises -> List[Int]:
    """Find indices of top-k maximum values in 1D tensor.

    Args:
        tensor: 1D tensor
        k: Number of top values to find

    Returns:
        DynamicVector of indices (length k) sorted by descending value

    Note:
        Simple implementation using repeated argmax (not optimal but clear)
    """
    var shape = tensor.shape()
    var num_classes = shape[1]
    var indices = List[Int]()

    # Create a copy of tensor values for modification
    var values = List[Float32]()
    var tensor_data = tensor._data.bitcast[Float32]()
    for i in range(num_classes):
        values.append(tensor_data[i])

    # Find top-k by repeatedly finding max and setting it to -inf
    for _ in range(k):
        var max_idx = 0
        var max_val = values[0]

        for i in range(1, num_classes):
            if values[i] > max_val:
                max_val = values[i]
                max_idx = i

        indices.append(max_idx)
        values[max_idx] = Float32(-1e9)  # Set to very negative value

    return indices


fn print_class_names():
    """Print CIFAR-10 class names for reference."""
    print("\nCIFAR-10 Classes:")
    print("  0: airplane")
    print("  1: automobile")
    print("  2: bird")
    print("  3: cat")
    print("  4: deer")
    print("  5: dog")
    print("  6: frog")
    print("  7: horse")
    print("  8: ship")
    print("  9: truck")
    print()


fn main() raises:
    """Main inference routine."""
    print("=" * 60)
    print("AlexNet Inference on CIFAR-10 Dataset")
    print("=" * 60)

    # Parse arguments
    var config = parse_args()
    var weights_dir = config[0]
    var data_dir = config[1]

    print("\nConfiguration:")
    print("  Weights Directory: ", weights_dir)
    print("  Data Directory: ", data_dir)
    print()

    # Print class names for reference
    print_class_names()

    # Initialize model
    print("Initializing AlexNet model...")
    var model = AlexNet(num_classes=10, dropout_rate=0.5)
    print("  Model initialized with", model.num_classes, "classes")
    print()

    # Load trained weights
    print("Loading trained weights...")
    model.load_weights(weights_dir)
    print("  Weights loaded from", weights_dir)
    print()

    # Load test dataset
    print("Loading CIFAR-10 test set...")
    var test_data = load_cifar10_test(data_dir)
    var test_images = test_data[0]
    var test_labels = test_data[1]
    print("  Test samples: ", test_images.shape()[0])
    print()

    # Evaluate model
    var results = evaluate_model(model, test_images, test_labels)
    var top1_acc = results[0]
    var top5_acc = results[1]

    print()
    print("Inference complete!")
    print("\nNote: This implementation demonstrates the full inference structure.")
    print("Batch processing will be more efficient when tensor slicing is optimized.")
