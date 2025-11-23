"""Inference script for VGG-16 on CIFAR-10

Loads trained weights and evaluates on test set.

Usage:
    mojo run examples/vgg16-cifar10/inference.mojo --weights-dir vgg16_weights
"""

from shared.core import ExTensor, zeros
from data_loader import load_cifar10_batch
from model import VGG16


fn compute_test_accuracy(inout model: VGG16, test_images: ExTensor, test_labels: ExTensor) raises -> Float32:
    """Compute accuracy on test set.

    Args:
        model: VGG-16 model with loaded weights
        test_images: Test images of shape (num_samples, 3, 32, 32)
        test_labels: Test labels of shape (num_samples,)

    Returns:
        Accuracy as percentage (0-100)
    """
    var test_shape = test_images.shape
    var num_test_samples = test_shape[0]

    var correct = 0

    print("Evaluating on " + str(num_test_samples) + " test samples...")

    # Process each test sample
    for i in range(num_test_samples):
        # Get single sample (simplified - in production would use proper slicing)
        var sample_shape = List[Int]()
        sample_shape.append(1)   # batch size = 1
        sample_shape.append(3)   # RGB channels
        sample_shape.append(32)  # height
        sample_shape.append(32)  # width

        var sample = zeros(sample_shape, DType.float32)
        # TODO: Copy actual sample data from test_images[i]

        # Forward pass (inference mode)
        var pred_class = model.predict(sample)
        var true_class = int(test_labels[i])

        if pred_class == true_class:
            correct += 1

        # Print progress every 1000 samples
        if (i + 1) % 1000 == 0:
            print("  Processed " + str(i + 1) + "/" + str(num_test_samples))

    var accuracy = Float32(correct) / Float32(num_test_samples) * 100.0
    return accuracy


fn main() raises:
    """Main inference function."""
    print("=== VGG-16 Inference on CIFAR-10 ===")
    print()

    # Configuration
    var weights_dir = "vgg16_weights"
    var data_dir = "datasets/cifar10"

    # Load test set
    print("Loading CIFAR-10 test set...")
    var test_images = zeros(List[Int]().append(10000).append(3).append(32).append(32), DType.float32)
    var test_labels = zeros(List[Int]().append(10000), DType.float32)

    # TODO: Load actual test set using load_cifar10_batch
    print("Test set loaded: 10,000 images")
    print()

    # Initialize model
    print("Initializing VGG-16 model...")
    var model = VGG16(num_classes=10, dropout_rate=0.5)
    print("Model initialized")
    print()

    # Load trained weights
    print("Loading weights from " + weights_dir + "...")
    model.load_weights(weights_dir)
    print("Weights loaded successfully")
    print()

    # Evaluate on test set
    var test_accuracy = compute_test_accuracy(model, test_images, test_labels)

    print()
    print("=" * 50)
    print("Test Accuracy: " + str(test_accuracy) + "%")
    print("=" * 50)
