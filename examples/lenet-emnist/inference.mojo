"""Inference Script for LeNet-5 on EMNIST

Simple inference script following KISS principles. Runs inference with trained LeNet-5 model.

Usage:
    # Run on entire test set
    mojo run examples/lenet-emnist/inference.mojo --weights lenet5_emnist.weights

    # Run on single image
    mojo run examples/lenet-emnist/inference.mojo --weights lenet5_emnist.weights --image path/to/image.png

Requirements:
    - Trained model weights (from train.mojo)
    - EMNIST dataset (for test set evaluation)

References:
    - LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998).
      Gradient-based learning applied to document recognition.
    - Reference Implementation: https://github.com/mattwang44/LeNet-from-Scratch
"""

from model import LeNet5
from shared.core import ExTensor, zeros
from shared.data import ExTensorDataset, BatchLoader
from sys import argv
from collections.vector import DynamicVector


# EMNIST Balanced class mapping (47 classes)
# 0-9: digits, 10-35: uppercase letters, 36-46: lowercase letters (select)
alias CLASS_NAMES = List[String](
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",  # 0-9
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",  # 10-19
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",  # 20-29
    "U", "V", "W", "X", "Y", "Z",                      # 30-35
    "a", "b", "d", "e", "f", "g", "h", "n", "q", "r", "t"  # 36-46 (select lowercase)
)


fn load_image(filepath: String) raises -> ExTensor:
    """Load single image for inference.

    Args:
        filepath: Path to image file (28x28 grayscale)

    Returns:
        ExTensor of shape (1, 1, 28, 28) normalized to [0, 1]

    Note:
        TODO: Implement when Mojo image I/O is stable
    """
    raise Error("Image loading not yet implemented - waiting for stable Mojo file I/O")


fn load_idx_images(filepath: String) raises -> ExTensor:
    """Load images from IDX file format.

    Args:
        filepath: Path to IDX images file

    Returns:
        ExTensor of shape (num_samples, 1, 28, 28) with pixel values

    Note:
        IDX format: magic (4 bytes) | num_items (4 bytes) | rows (4 bytes) | cols (4 bytes) | data
    """
    raise Error("IDX loading not yet implemented - waiting for stable Mojo file I/O")


fn load_idx_labels(filepath: String) raises -> ExTensor:
    """Load labels from IDX file format.

    Args:
        filepath: Path to IDX labels file

    Returns:
        ExTensor of shape (num_samples,) with label values

    Note:
        IDX format: magic (4 bytes) | num_items (4 bytes) | data
    """
    raise Error("IDX loading not yet implemented - waiting for stable Mojo file I/O")


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
    var max_val = logits[0]

    for i in range(1, num_classes):
        if logits[i] > max_val:
            max_val = logits[i]
            max_idx = i

    # Compute softmax for confidence (simplified)
    var exp_sum = Float32(0.0)
    for i in range(num_classes):
        exp_sum += exp(logits[i])

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

    for i in range(num_samples):
        # Extract single image (create batch of 1)
        var img_shape = DynamicVector[Int](4)
        img_shape.push_back(1)   # batch=1
        img_shape.push_back(1)   # channels=1
        img_shape.push_back(28)  # height
        img_shape.push_back(28)  # width

        # TODO: Extract slice when slicing is stable
        # var single_image = images[i:i+1]

        # For now, create placeholder
        var single_image = zeros(img_shape, images.dtype())

        # Run inference
        var pred_class = model.predict(single_image)
        var true_label = Int(labels[i])

        if pred_class == true_label:
            correct += 1

        # Print progress every 1000 samples
        if (i + 1) % 1000 == 0:
            var current_acc = Float32(correct) / Float32(i + 1)
            print("  Processed [", i + 1, "/", num_samples, "] - Accuracy: ", current_acc * 100.0, "%")

    var accuracy = Float32(correct) / Float32(num_samples)

    return (accuracy, correct, num_samples)


fn parse_args() raises -> (String, String, String):
    """Parse command line arguments.

    Returns:
        Tuple of (weights_path, mode, image_or_data_path)
        mode is either "single" or "test"
    """
    var weights_path = "lenet5_emnist.weights"
    var mode = "test"
    var path = "datasets/emnist"

    var args = argv()
    for i in range(len(args)):
        if args[i] == "--weights" and i + 1 < len(args):
            weights_path = args[i + 1]
        elif args[i] == "--image" and i + 1 < len(args):
            mode = "single"
            path = args[i + 1]
        elif args[i] == "--data-dir" and i + 1 < len(args):
            path = args[i + 1]

    return (weights_path, mode, path)


fn main() raises:
    """Main inference entry point."""
    print("=" * 60)
    print("LeNet-5 Inference on EMNIST Dataset")
    print("=" * 60)

    # Parse arguments
    var config = parse_args()
    var weights_path = config[0]
    var mode = config[1]
    var path = config[2]

    print("Configuration:")
    print("  Weights: ", weights_path)
    print("  Mode: ", mode)
    print("  Path: ", path)
    print()

    # Initialize model
    print("Initializing LeNet-5 model...")
    var model = LeNet5(num_classes=47)
    print("  Model initialized with", model.num_classes, "classes")
    print()

    # Load weights
    print("Loading model weights...")
    print("  Note: Weight loading not yet implemented")
    print("  Waiting for stable Mojo file I/O")
    # model.load_weights(weights_path)
    # print("  Weights loaded from", weights_path)
    print()

    if mode == "single":
        # Single image inference
        print("Running inference on single image:", path)
        print("  Note: Image loading not yet implemented")
        # var image = load_image(path)
        # var result = infer_single(model, image)
        # var pred_class = result[0]
        # var confidence = result[1]
        # print("  Predicted class:", pred_class, "(" + CLASS_NAMES[pred_class] + ")")
        # print("  Confidence:", confidence * 100.0, "%")
    else:
        # Test set evaluation
        print("Running inference on test set...")
        print("  Note: Dataset loading not yet implemented")
        # var test_images = load_idx_images(path + "/emnist-balanced-test-images-idx3-ubyte")
        # var test_labels = load_idx_labels(path + "/emnist-balanced-test-labels-idx1-ubyte")
        # var result = evaluate_test_set(model, test_images, test_labels)
        # var accuracy = result[0]
        # var correct = result[1]
        # var total = result[2]
        # print()
        # print("Results:")
        # print("  Correct:", correct, "/", total)
        # print("  Accuracy:", accuracy * 100.0, "%")

    print()
    print("Inference complete!")
    print()
    print("Note: This is a skeleton implementation demonstrating the structure.")
    print("Full inference will be available when Mojo file I/O is stable.")
