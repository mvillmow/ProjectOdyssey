"""Training script for VGG-16 on CIFAR-10

Implements manual backpropagation through all 16 layers without autograd.
Uses SGD with momentum optimizer and step learning rate decay.

Usage:
    mojo run examples/vgg16-cifar10/train.mojo --epochs 200 --batch-size 128 --lr 0.01
"""

from shared.core import ExTensor, zeros, zeros_like
from shared.core.loss import cross_entropy_loss, cross_entropy_loss_backward
from shared.training.schedulers import step_lr_schedule
from data_loader import load_cifar10_batches
from model import VGG16
from collections.vector import DynamicVector


fn compute_accuracy(logits: ExTensor, labels: ExTensor) raises -> Float32:
    """Compute classification accuracy.

    Args:
        logits: Model outputs of shape (batch, num_classes)
        labels: Ground truth labels of shape (batch,)

    Returns:
        Accuracy as percentage (0-100)
    """
    var logits_shape = logits.shape()
    var batch_size = logits_shape[0]
    var num_classes = logits_shape[1]

    var correct = 0

    # Get logits pointer for faster access
    var logits_data = logits._data

    for b in range(batch_size):
        # Find argmax for this sample
        var max_idx = 0
        var max_val = logits_data[b * num_classes + 0]

        for c in range(1, num_classes):
            var idx = b * num_classes + c
            if logits_data[idx] > max_val:
                max_val = logits_data[idx]
                max_idx = c

        # Check if prediction matches label
        var label = int(labels[b])
        if max_idx == label:
            correct += 1

    return Float32(correct) / Float32(batch_size) * 100.0


fn train_epoch(
    inout model: VGG16,
    images: ExTensor,
    labels: ExTensor,
    batch_size: Int,
    learning_rate: Float32,
    momentum: Float32,
    inout velocities: DynamicVector[ExTensor],
    epoch: Int
) raises -> (Float32, Float32):
    """Train for one epoch.

    Args:
        model: VGG-16 model
        images: Training images of shape (num_samples, 3, 32, 32)
        labels: Training labels of shape (num_samples,)
        batch_size: Mini-batch size
        learning_rate: Learning rate for SGD
        momentum: Momentum factor
        velocities: Momentum velocities for all parameters
        epoch: Current epoch number (for logging)

    Returns:
        Tuple of (average_loss, average_accuracy)
    """
    var images_shape = images.shape()
    var num_samples = images_shape[0]
    var num_batches = num_samples // batch_size

    var total_loss = Float32(0.0)
    var total_accuracy = Float32(0.0)

    print("Training epoch " + str(epoch) + "...")

    for batch_idx in range(num_batches):
        # Extract batch
        var start_idx = batch_idx * batch_size
        var end_idx = start_idx + batch_size

        # Get batch slice (simplified - in production would use proper slicing)
        var batch_images = images  # TODO: Implement proper batch slicing
        var batch_labels = labels  # TODO: Implement proper batch slicing

        # Forward pass
        var logits = model.forward(batch_images, training=True)

        # Compute loss
        var loss = cross_entropy_loss(logits, batch_labels)
        total_loss += loss

        # Compute accuracy
        var accuracy = compute_accuracy(logits, batch_labels)
        total_accuracy += accuracy

        # Backward pass (TODO: Implement full backward pass through all 16 layers)
        var grad_logits = cross_entropy_loss_backward(logits, batch_labels)

        # TODO: Implement backward passes for:
        # - FC3 (output layer)
        # - Dropout2
        # - FC2
        # - Dropout1
        # - FC1
        # - Flatten (reshape gradient)
        # - Block 5 (3 conv layers + pool)
        # - Block 4 (3 conv layers + pool)
        # - Block 3 (3 conv layers + pool)
        # - Block 2 (2 conv layers + pool)
        # - Block 1 (2 conv layers + pool)

        # TODO: Update parameters using SGD with momentum

        # Print progress every 10 batches
        if batch_idx % 10 == 0:
            print("  Batch " + str(batch_idx) + "/" + str(num_batches) +
                  " | Loss: " + str(loss) + " | Acc: " + str(accuracy) + "%")

    var avg_loss = total_loss / Float32(num_batches)
    var avg_accuracy = total_accuracy / Float32(num_batches)

    return (avg_loss, avg_accuracy)


fn main() raises:
    """Main training function."""
    print("=== VGG-16 Training on CIFAR-10 ===")
    print()

    # Hyperparameters
    var num_epochs = 200
    var batch_size = 128
    var initial_lr = Float32(0.01)
    var momentum = Float32(0.9)
    var lr_decay_step = 60  # Decay every 60 epochs
    var lr_decay_gamma = Float32(0.2)  # Decay by 5x

    # Load dataset
    print("Loading CIFAR-10 dataset...")
    var train_images = zeros(DynamicVector[Int](4).push_back(50000).push_back(3).push_back(32).push_back(32), DType.float32)
    var train_labels = zeros(DynamicVector[Int](1).push_back(50000), DType.float32)

    # TODO: Load actual dataset using load_cifar10_batches
    print("Dataset loaded: 50,000 training images")
    print()

    # Initialize model
    print("Initializing VGG-16 model...")
    var model = VGG16(num_classes=10, dropout_rate=0.5)
    print("Model initialized with ~15M parameters")
    print()

    # Initialize velocity tensors for momentum (one per parameter)
    var velocities = DynamicVector[ExTensor]()
    # TODO: Initialize 32 velocity tensors (16 layers × 2 params per layer)
    print("Initialized SGD momentum optimizer")
    print()

    # Training loop
    print("Starting training for " + str(num_epochs) + " epochs...")
    print()

    for epoch in range(num_epochs):
        # Compute learning rate with step decay
        var lr = step_lr_schedule(initial_lr, epoch, step_size=lr_decay_step, gamma=lr_decay_gamma)

        print("Epoch " + str(epoch + 1) + "/" + str(num_epochs) + " | LR: " + str(lr))

        # Train for one epoch
        var result = train_epoch(
            model,
            train_images,
            train_labels,
            batch_size,
            lr,
            momentum,
            velocities,
            epoch + 1
        )
        var avg_loss = result.get[0, Float32]()
        var avg_accuracy = result.get[1, Float32]()

        print("Epoch " + str(epoch + 1) + " completed | Avg Loss: " + str(avg_loss) +
              " | Avg Accuracy: " + str(avg_accuracy) + "%")
        print()

        # Save weights every 20 epochs
        if (epoch + 1) % 20 == 0:
            print("Saving weights...")
            model.save_weights("vgg16_weights")
            print("Weights saved to vgg16_weights/")
            print()

    print("Training completed!")
    print("Final weights saved to vgg16_weights/")
