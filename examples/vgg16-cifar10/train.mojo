"""Training Script for VGG-16 on CIFAR-10

Implements training with manual backward passes through all 16 layers (no autograd).
Uses SGD optimizer with momentum and dropout regularization.

Usage:
    mojo run examples/vgg16-cifar10/train.mojo --epochs 200 --batch-size 128 --lr 0.01 --momentum 0.9

Requirements:
    - CIFAR-10 dataset downloaded (run: python examples/vgg16-cifar10/download_cifar10.py)
    - Dataset location: datasets/cifar10/

References:
    - Simonyan, K., & Zisserman, A. (2014).
      Very deep convolutional networks for large-scale image recognition.
      arXiv preprint arXiv:1409.1556.
"""

from model import VGG16
from shared.data.datasets import load_cifar10_train, load_cifar10_test
from shared.core import ExTensor, zeros
from shared.core.conv import conv2d, conv2d_backward
from shared.core.pooling import maxpool2d, maxpool2d_backward
from shared.core.linear import linear, linear_backward
from shared.core.activation import relu, relu_backward
from shared.core.dropout import dropout, dropout_backward
from shared.core.loss import cross_entropy_loss, cross_entropy_loss_backward
from shared.training.schedulers import step_lr
from shared.data import extract_batch_pair, compute_num_batches, get_batch_indices
from sys import argv


fn parse_args() raises -> (Int, Int, Float32, Float32, String, String):
    """Parse command line arguments.

    Returns:
        Tuple of (epochs, batch_size, learning_rate, momentum, data_dir, weights_dir)
    """
    var epochs = 200
    var batch_size = 128
    var learning_rate = Float32(0.01)
    var momentum = Float32(0.9)
    var data_dir = "datasets/cifar10"
    var weights_dir = "vgg16_weights"

    var args = argv()
    for i in range(len(args)):
        if args[i] == "--epochs" and i + 1 < len(args):
            epochs = int(args[i + 1])
        elif args[i] == "--batch-size" and i + 1 < len(args):
            batch_size = int(args[i + 1])
        elif args[i] == "--lr" and i + 1 < len(args):
            learning_rate = Float32(float(args[i + 1]))
        elif args[i] == "--momentum" and i + 1 < len(args):
            momentum = Float32(float(args[i + 1]))
        elif args[i] == "--data-dir" and i + 1 < len(args):
            data_dir = args[i + 1]
        elif args[i] == "--weights-dir" and i + 1 < len(args):
            weights_dir = args[i + 1]

    return (epochs, batch_size, learning_rate, momentum, data_dir, weights_dir)


fn compute_gradients(
    mut model: VGG16,
    borrowed input: ExTensor,
    borrowed labels: ExTensor,
    learning_rate: Float32,
    momentum: Float32,
    mut velocities: List[ExTensor]
) raises -> Float32:
    """Compute gradients and update parameters for one batch.

    This implements the full forward and backward pass manually through all 16 layers.

    Args:
        model: VGG16 model
        input: Batch of images (batch, 3, 32, 32)
        labels: Batch of labels (batch,)
        learning_rate: Learning rate for SGD
        momentum: Momentum factor for SGD
        velocities: Momentum velocities for each parameter (32 tensors)

    Returns:
        Loss value for this batch
    """
    # ========== Forward Pass (with caching for backward) ==========

    # Block 1: Conv1_1 -> ReLU -> Conv1_2 -> ReLU -> MaxPool
    var conv1_1_out = conv2d(input, model.conv1_1_kernel, model.conv1_1_bias, stride=1, padding=1)
    var relu1_1_out = relu(conv1_1_out)
    var conv1_2_out = conv2d(relu1_1_out, model.conv1_2_kernel, model.conv1_2_bias, stride=1, padding=1)
    var relu1_2_out = relu(conv1_2_out)
    var pool1_out = maxpool2d(relu1_2_out, kernel_size=2, stride=2, padding=0)

    # Block 2: Conv2_1 -> ReLU -> Conv2_2 -> ReLU -> MaxPool
    var conv2_1_out = conv2d(pool1_out, model.conv2_1_kernel, model.conv2_1_bias, stride=1, padding=1)
    var relu2_1_out = relu(conv2_1_out)
    var conv2_2_out = conv2d(relu2_1_out, model.conv2_2_kernel, model.conv2_2_bias, stride=1, padding=1)
    var relu2_2_out = relu(conv2_2_out)
    var pool2_out = maxpool2d(relu2_2_out, kernel_size=2, stride=2, padding=0)

    # Block 3: Conv3_1 -> ReLU -> Conv3_2 -> ReLU -> Conv3_3 -> ReLU -> MaxPool
    var conv3_1_out = conv2d(pool2_out, model.conv3_1_kernel, model.conv3_1_bias, stride=1, padding=1)
    var relu3_1_out = relu(conv3_1_out)
    var conv3_2_out = conv2d(relu3_1_out, model.conv3_2_kernel, model.conv3_2_bias, stride=1, padding=1)
    var relu3_2_out = relu(conv3_2_out)
    var conv3_3_out = conv2d(relu3_2_out, model.conv3_3_kernel, model.conv3_3_bias, stride=1, padding=1)
    var relu3_3_out = relu(conv3_3_out)
    var pool3_out = maxpool2d(relu3_3_out, kernel_size=2, stride=2, padding=0)

    # Block 4: Conv4_1 -> ReLU -> Conv4_2 -> ReLU -> Conv4_3 -> ReLU -> MaxPool
    var conv4_1_out = conv2d(pool3_out, model.conv4_1_kernel, model.conv4_1_bias, stride=1, padding=1)
    var relu4_1_out = relu(conv4_1_out)
    var conv4_2_out = conv2d(relu4_1_out, model.conv4_2_kernel, model.conv4_2_bias, stride=1, padding=1)
    var relu4_2_out = relu(conv4_2_out)
    var conv4_3_out = conv2d(relu4_2_out, model.conv4_3_kernel, model.conv4_3_bias, stride=1, padding=1)
    var relu4_3_out = relu(conv4_3_out)
    var pool4_out = maxpool2d(relu4_3_out, kernel_size=2, stride=2, padding=0)

    # Block 5: Conv5_1 -> ReLU -> Conv5_2 -> ReLU -> Conv5_3 -> ReLU -> MaxPool
    var conv5_1_out = conv2d(pool4_out, model.conv5_1_kernel, model.conv5_1_bias, stride=1, padding=1)
    var relu5_1_out = relu(conv5_1_out)
    var conv5_2_out = conv2d(relu5_1_out, model.conv5_2_kernel, model.conv5_2_bias, stride=1, padding=1)
    var relu5_2_out = relu(conv5_2_out)
    var conv5_3_out = conv2d(relu5_2_out, model.conv5_3_kernel, model.conv5_3_bias, stride=1, padding=1)
    var relu5_3_out = relu(conv5_3_out)
    var pool5_out = maxpool2d(relu5_3_out, kernel_size=2, stride=2, padding=0)

    # Flatten
    var pool5_shape = pool5_out.shape()
    var batch_size = pool5_shape[0]
    var flattened_size = pool5_shape[1] * pool5_shape[2] * pool5_shape[3]
    var flatten_shape = List[Int]()
    flatten_shape.append(batch_size)
    flatten_shape.append(flattened_size)
    var flattened = pool5_out.reshape(flatten_shape)

    # FC1 + ReLU + Dropout
    var fc1_out = linear(flattened, model.fc1_weights, model.fc1_bias)
    var relu_fc1_out = relu(fc1_out)
    var drop1_result = dropout(relu_fc1_out, model.dropout_rate)
    var drop1_out = drop1_result[0]      # Dropout output
    var drop1_mask = drop1_result[1]     # Dropout mask for backward

    # FC2 + ReLU + Dropout
    var fc2_out = linear(drop1_out, model.fc2_weights, model.fc2_bias)
    var relu_fc2_out = relu(fc2_out)
    var drop2_result = dropout(relu_fc2_out, model.dropout_rate)
    var drop2_out = drop2_result[0]      # Dropout output
    var drop2_mask = drop2_result[1]     # Dropout mask for backward

    # FC3 (logits)
    var logits = linear(drop2_out, model.fc3_weights, model.fc3_bias)

    # Compute loss
    var loss = cross_entropy_loss(logits, labels)

    # ========== Backward Pass (through all 16 layers) ==========

    # Start with gradient from loss
    var grad_logits = cross_entropy_loss_backward(logits, labels)

    # FC3 backward
    var fc3_grads = linear_backward(grad_logits, drop2_out, model.fc3_weights)
    var grad_drop2_out = fc3_grads[0]
    var grad_fc3_weights = fc3_grads[1]
    var grad_fc3_bias = fc3_grads[2]

    # Dropout2 backward
    var grad_relu_fc2_out = dropout_backward(grad_drop2_out, drop2_mask, model.dropout_rate)

    # ReLU FC2 backward
    var grad_fc2_out = relu_backward(grad_relu_fc2_out, fc2_out)

    # FC2 backward
    var fc2_grads = linear_backward(grad_fc2_out, drop1_out, model.fc2_weights)
    var grad_drop1_out = fc2_grads[0]
    var grad_fc2_weights = fc2_grads[1]
    var grad_fc2_bias = fc2_grads[2]

    # Dropout1 backward
    var grad_relu_fc1_out = dropout_backward(grad_drop1_out, drop1_mask, model.dropout_rate)

    # ReLU FC1 backward
    var grad_fc1_out = relu_backward(grad_relu_fc1_out, fc1_out)

    # FC1 backward
    var fc1_grads = linear_backward(grad_fc1_out, flattened, model.fc1_weights)
    var grad_flattened = fc1_grads[0]
    var grad_fc1_weights = fc1_grads[1]
    var grad_fc1_bias = fc1_grads[2]

    # Unflatten gradient
    var grad_pool5_out = grad_flattened.reshape(pool5_shape)

    # ===== Block 5 Backward =====

    # MaxPool5 backward
    var grad_relu5_3_out = maxpool2d_backward(grad_pool5_out, relu5_3_out, pool5_out, kernel_size=2, stride=2, padding=0)

    # ReLU5_3 backward
    var grad_conv5_3_out = relu_backward(grad_relu5_3_out, conv5_3_out)

    # Conv5_3 backward
    var conv5_3_grads = conv2d_backward(grad_conv5_3_out, relu5_2_out, model.conv5_3_kernel, stride=1, padding=1)
    var grad_relu5_2_out = conv5_3_grads[0]
    var grad_conv5_3_kernel = conv5_3_grads[1]
    var grad_conv5_3_bias = conv5_3_grads[2]

    # ReLU5_2 backward
    var grad_conv5_2_out = relu_backward(grad_relu5_2_out, conv5_2_out)

    # Conv5_2 backward
    var conv5_2_grads = conv2d_backward(grad_conv5_2_out, relu5_1_out, model.conv5_2_kernel, stride=1, padding=1)
    var grad_relu5_1_out = conv5_2_grads[0]
    var grad_conv5_2_kernel = conv5_2_grads[1]
    var grad_conv5_2_bias = conv5_2_grads[2]

    # ReLU5_1 backward
    var grad_conv5_1_out = relu_backward(grad_relu5_1_out, conv5_1_out)

    # Conv5_1 backward
    var conv5_1_grads = conv2d_backward(grad_conv5_1_out, pool4_out, model.conv5_1_kernel, stride=1, padding=1)
    var grad_pool4_out = conv5_1_grads[0]
    var grad_conv5_1_kernel = conv5_1_grads[1]
    var grad_conv5_1_bias = conv5_1_grads[2]

    # ===== Block 4 Backward =====

    # MaxPool4 backward
    var grad_relu4_3_out = maxpool2d_backward(grad_pool4_out, relu4_3_out, pool4_out, kernel_size=2, stride=2, padding=0)

    # ReLU4_3 backward
    var grad_conv4_3_out = relu_backward(grad_relu4_3_out, conv4_3_out)

    # Conv4_3 backward
    var conv4_3_grads = conv2d_backward(grad_conv4_3_out, relu4_2_out, model.conv4_3_kernel, stride=1, padding=1)
    var grad_relu4_2_out = conv4_3_grads[0]
    var grad_conv4_3_kernel = conv4_3_grads[1]
    var grad_conv4_3_bias = conv4_3_grads[2]

    # ReLU4_2 backward
    var grad_conv4_2_out = relu_backward(grad_relu4_2_out, conv4_2_out)

    # Conv4_2 backward
    var conv4_2_grads = conv2d_backward(grad_conv4_2_out, relu4_1_out, model.conv4_2_kernel, stride=1, padding=1)
    var grad_relu4_1_out = conv4_2_grads[0]
    var grad_conv4_2_kernel = conv4_2_grads[1]
    var grad_conv4_2_bias = conv4_2_grads[2]

    # ReLU4_1 backward
    var grad_conv4_1_out = relu_backward(grad_relu4_1_out, conv4_1_out)

    # Conv4_1 backward
    var conv4_1_grads = conv2d_backward(grad_conv4_1_out, pool3_out, model.conv4_1_kernel, stride=1, padding=1)
    var grad_pool3_out = conv4_1_grads[0]
    var grad_conv4_1_kernel = conv4_1_grads[1]
    var grad_conv4_1_bias = conv4_1_grads[2]

    # ===== Block 3 Backward =====

    # MaxPool3 backward
    var grad_relu3_3_out = maxpool2d_backward(grad_pool3_out, relu3_3_out, pool3_out, kernel_size=2, stride=2, padding=0)

    # ReLU3_3 backward
    var grad_conv3_3_out = relu_backward(grad_relu3_3_out, conv3_3_out)

    # Conv3_3 backward
    var conv3_3_grads = conv2d_backward(grad_conv3_3_out, relu3_2_out, model.conv3_3_kernel, stride=1, padding=1)
    var grad_relu3_2_out = conv3_3_grads[0]
    var grad_conv3_3_kernel = conv3_3_grads[1]
    var grad_conv3_3_bias = conv3_3_grads[2]

    # ReLU3_2 backward
    var grad_conv3_2_out = relu_backward(grad_relu3_2_out, conv3_2_out)

    # Conv3_2 backward
    var conv3_2_grads = conv2d_backward(grad_conv3_2_out, relu3_1_out, model.conv3_2_kernel, stride=1, padding=1)
    var grad_relu3_1_out = conv3_2_grads[0]
    var grad_conv3_2_kernel = conv3_2_grads[1]
    var grad_conv3_2_bias = conv3_2_grads[2]

    # ReLU3_1 backward
    var grad_conv3_1_out = relu_backward(grad_relu3_1_out, conv3_1_out)

    # Conv3_1 backward
    var conv3_1_grads = conv2d_backward(grad_conv3_1_out, pool2_out, model.conv3_1_kernel, stride=1, padding=1)
    var grad_pool2_out = conv3_1_grads[0]
    var grad_conv3_1_kernel = conv3_1_grads[1]
    var grad_conv3_1_bias = conv3_1_grads[2]

    # ===== Block 2 Backward =====

    # MaxPool2 backward
    var grad_relu2_2_out = maxpool2d_backward(grad_pool2_out, relu2_2_out, pool2_out, kernel_size=2, stride=2, padding=0)

    # ReLU2_2 backward
    var grad_conv2_2_out = relu_backward(grad_relu2_2_out, conv2_2_out)

    # Conv2_2 backward
    var conv2_2_grads = conv2d_backward(grad_conv2_2_out, relu2_1_out, model.conv2_2_kernel, stride=1, padding=1)
    var grad_relu2_1_out = conv2_2_grads[0]
    var grad_conv2_2_kernel = conv2_2_grads[1]
    var grad_conv2_2_bias = conv2_2_grads[2]

    # ReLU2_1 backward
    var grad_conv2_1_out = relu_backward(grad_relu2_1_out, conv2_1_out)

    # Conv2_1 backward
    var conv2_1_grads = conv2d_backward(grad_conv2_1_out, pool1_out, model.conv2_1_kernel, stride=1, padding=1)
    var grad_pool1_out = conv2_1_grads[0]
    var grad_conv2_1_kernel = conv2_1_grads[1]
    var grad_conv2_1_bias = conv2_1_grads[2]

    # ===== Block 1 Backward =====

    # MaxPool1 backward
    var grad_relu1_2_out = maxpool2d_backward(grad_pool1_out, relu1_2_out, pool1_out, kernel_size=2, stride=2, padding=0)

    # ReLU1_2 backward
    var grad_conv1_2_out = relu_backward(grad_relu1_2_out, conv1_2_out)

    # Conv1_2 backward
    var conv1_2_grads = conv2d_backward(grad_conv1_2_out, relu1_1_out, model.conv1_2_kernel, stride=1, padding=1)
    var grad_relu1_1_out = conv1_2_grads[0]
    var grad_conv1_2_kernel = conv1_2_grads[1]
    var grad_conv1_2_bias = conv1_2_grads[2]

    # ReLU1_1 backward
    var grad_conv1_1_out = relu_backward(grad_relu1_1_out, conv1_1_out)

    # Conv1_1 backward
    var conv1_1_grads = conv2d_backward(grad_conv1_1_out, input, model.conv1_1_kernel, stride=1, padding=1)
    var grad_input = conv1_1_grads[0]  # Not used (no input gradient needed)
    var grad_conv1_1_kernel = conv1_1_grads[1]
    var grad_conv1_1_bias = conv1_1_grads[2]

    # ========== Parameter Update (SGD with Momentum) ==========
    # Update all 32 parameters (16 layers × 2 params per layer)

    from shared.training.optimizers import sgd_momentum_update_inplace

    # Block 1 updates
    sgd_momentum_update_inplace(model.conv1_1_kernel, grad_conv1_1_kernel, velocities[0], learning_rate, momentum)
    sgd_momentum_update_inplace(model.conv1_1_bias, grad_conv1_1_bias, velocities[1], learning_rate, momentum)
    sgd_momentum_update_inplace(model.conv1_2_kernel, grad_conv1_2_kernel, velocities[2], learning_rate, momentum)
    sgd_momentum_update_inplace(model.conv1_2_bias, grad_conv1_2_bias, velocities[3], learning_rate, momentum)

    # Block 2 updates
    sgd_momentum_update_inplace(model.conv2_1_kernel, grad_conv2_1_kernel, velocities[4], learning_rate, momentum)
    sgd_momentum_update_inplace(model.conv2_1_bias, grad_conv2_1_bias, velocities[5], learning_rate, momentum)
    sgd_momentum_update_inplace(model.conv2_2_kernel, grad_conv2_2_kernel, velocities[6], learning_rate, momentum)
    sgd_momentum_update_inplace(model.conv2_2_bias, grad_conv2_2_bias, velocities[7], learning_rate, momentum)

    # Block 3 updates
    sgd_momentum_update_inplace(model.conv3_1_kernel, grad_conv3_1_kernel, velocities[8], learning_rate, momentum)
    sgd_momentum_update_inplace(model.conv3_1_bias, grad_conv3_1_bias, velocities[9], learning_rate, momentum)
    sgd_momentum_update_inplace(model.conv3_2_kernel, grad_conv3_2_kernel, velocities[10], learning_rate, momentum)
    sgd_momentum_update_inplace(model.conv3_2_bias, grad_conv3_2_bias, velocities[11], learning_rate, momentum)
    sgd_momentum_update_inplace(model.conv3_3_kernel, grad_conv3_3_kernel, velocities[12], learning_rate, momentum)
    sgd_momentum_update_inplace(model.conv3_3_bias, grad_conv3_3_bias, velocities[13], learning_rate, momentum)

    # Block 4 updates
    sgd_momentum_update_inplace(model.conv4_1_kernel, grad_conv4_1_kernel, velocities[14], learning_rate, momentum)
    sgd_momentum_update_inplace(model.conv4_1_bias, grad_conv4_1_bias, velocities[15], learning_rate, momentum)
    sgd_momentum_update_inplace(model.conv4_2_kernel, grad_conv4_2_kernel, velocities[16], learning_rate, momentum)
    sgd_momentum_update_inplace(model.conv4_2_bias, grad_conv4_2_bias, velocities[17], learning_rate, momentum)
    sgd_momentum_update_inplace(model.conv4_3_kernel, grad_conv4_3_kernel, velocities[18], learning_rate, momentum)
    sgd_momentum_update_inplace(model.conv4_3_bias, grad_conv4_3_bias, velocities[19], learning_rate, momentum)

    # Block 5 updates
    sgd_momentum_update_inplace(model.conv5_1_kernel, grad_conv5_1_kernel, velocities[20], learning_rate, momentum)
    sgd_momentum_update_inplace(model.conv5_1_bias, grad_conv5_1_bias, velocities[21], learning_rate, momentum)
    sgd_momentum_update_inplace(model.conv5_2_kernel, grad_conv5_2_kernel, velocities[22], learning_rate, momentum)
    sgd_momentum_update_inplace(model.conv5_2_bias, grad_conv5_2_bias, velocities[23], learning_rate, momentum)
    sgd_momentum_update_inplace(model.conv5_3_kernel, grad_conv5_3_kernel, velocities[24], learning_rate, momentum)
    sgd_momentum_update_inplace(model.conv5_3_bias, grad_conv5_3_bias, velocities[25], learning_rate, momentum)

    # FC layer updates
    sgd_momentum_update_inplace(model.fc1_weights, grad_fc1_weights, velocities[26], learning_rate, momentum)
    sgd_momentum_update_inplace(model.fc1_bias, grad_fc1_bias, velocities[27], learning_rate, momentum)
    sgd_momentum_update_inplace(model.fc2_weights, grad_fc2_weights, velocities[28], learning_rate, momentum)
    sgd_momentum_update_inplace(model.fc2_bias, grad_fc2_bias, velocities[29], learning_rate, momentum)
    sgd_momentum_update_inplace(model.fc3_weights, grad_fc3_weights, velocities[30], learning_rate, momentum)
    sgd_momentum_update_inplace(model.fc3_bias, grad_fc3_bias, velocities[31], learning_rate, momentum)

    return loss


fn train_epoch(
    mut model: VGG16,
    borrowed train_images: ExTensor,
    borrowed train_labels: ExTensor,
    batch_size: Int,
    learning_rate: Float32,
    momentum: Float32,
    epoch: Int,
    total_epochs: Int,
    mut velocities: List[ExTensor]
) raises -> Float32:
    """Train for one epoch.

    Args:
        model: VGG16 model
        train_images: Training images (50000, 3, 32, 32)
        train_labels: Training labels (50000,)
        batch_size: Mini-batch size
        learning_rate: Learning rate for SGD
        momentum: Momentum factor
        epoch: Current epoch number (1-indexed)
        total_epochs: Total number of epochs
        velocities: Momentum velocities for each parameter

    Returns:
        Average loss for the epoch
    """
    var num_samples = train_images.shape()[0]
    var num_batches = compute_num_batches(num_samples, batch_size)

    var total_loss = Float32(0.0)

    print("Epoch [", epoch, "/", total_epochs, "]")

    for batch_idx in range(num_batches):
        # Get batch indices
        var indices = get_batch_indices(batch_idx, batch_size, num_samples)
        var start_idx = indices.get[0, Int]()
        var actual_batch_size = indices.get[2, Int]()

        # Extract batch using shared library utility
        var batch_pair = extract_batch_pair(train_images, train_labels, start_idx, batch_size)
        var batch_images = batch_pair[0]
        var batch_labels = batch_pair[1]

        # Compute gradients and update parameters
        var batch_loss = compute_gradients(model, batch_images, batch_labels, learning_rate, momentum, velocities)
        total_loss += batch_loss

        # Print progress every 100 batches
        if (batch_idx + 1) % 100 == 0:
            var avg_loss = total_loss / Float32(batch_idx + 1)
            print("  Batch [", batch_idx + 1, "/", num_batches, "] - Loss: ", avg_loss)

    var avg_loss = total_loss / Float32(num_batches)
    print("  Average Loss: ", avg_loss)

    return avg_loss


fn evaluate(
    mut model: VGG16,
    borrowed test_images: ExTensor,
    borrowed test_labels: ExTensor
) raises -> Float32:
    """Evaluate model on test set.

    Args:
        model: VGG16 model
        test_images: Test images (10000, 3, 32, 32)
        test_labels: Test labels (10000,)

    Returns:
        Test accuracy (0.0 to 1.0)
    """
    var num_samples = test_images.shape()[0]
    var correct = 0

    print("Evaluating on ", num_samples, " samples...")

    # Evaluate all test samples
    for i in range(num_samples):
        # Extract single sample using batch utilities
        var batch_pair = extract_batch_pair(test_images, test_labels, i, 1)
        var sample_image = batch_pair[0]
        var sample_label = batch_pair[1]

        # Forward pass (inference mode)
        var pred_class = model.predict(sample_image)
        var true_label = Int(sample_label[0])

        if pred_class == true_label:
            correct += 1

        # Print progress every 1000 samples
        if (i + 1) % 1000 == 0:
            print("  Processed ", i + 1, "/", num_samples)

    var accuracy = Float32(correct) / Float32(num_samples)
    print("  Test Accuracy: ", accuracy * 100.0, "% (", correct, "/", num_samples, ")")

    return accuracy


fn initialize_velocities(model: VGG16) raises -> List[ExTensor]:
    """Initialize momentum velocities for all parameters (32 tensors).

    Args:
        model: VGG16 model

    Returns:
        DynamicVector of zero-initialized velocity tensors matching parameter shapes
    """
    var velocities = List[ExTensor]()

    # Initialize velocities for all 32 parameters (conv1-5 blocks + fc1-3)
    # Block 1 (4 params)
    velocities.append(zeros(model.conv1_1_kernel.shape(), DType.float32))
    velocities.append(zeros(model.conv1_1_bias.shape(), DType.float32))
    velocities.append(zeros(model.conv1_2_kernel.shape(), DType.float32))
    velocities.append(zeros(model.conv1_2_bias.shape(), DType.float32))

    # Block 2 (4 params)
    velocities.append(zeros(model.conv2_1_kernel.shape(), DType.float32))
    velocities.append(zeros(model.conv2_1_bias.shape(), DType.float32))
    velocities.append(zeros(model.conv2_2_kernel.shape(), DType.float32))
    velocities.append(zeros(model.conv2_2_bias.shape(), DType.float32))

    # Block 3 (6 params)
    velocities.append(zeros(model.conv3_1_kernel.shape(), DType.float32))
    velocities.append(zeros(model.conv3_1_bias.shape(), DType.float32))
    velocities.append(zeros(model.conv3_2_kernel.shape(), DType.float32))
    velocities.append(zeros(model.conv3_2_bias.shape(), DType.float32))
    velocities.append(zeros(model.conv3_3_kernel.shape(), DType.float32))
    velocities.append(zeros(model.conv3_3_bias.shape(), DType.float32))

    # Block 4 (6 params)
    velocities.append(zeros(model.conv4_1_kernel.shape(), DType.float32))
    velocities.append(zeros(model.conv4_1_bias.shape(), DType.float32))
    velocities.append(zeros(model.conv4_2_kernel.shape(), DType.float32))
    velocities.append(zeros(model.conv4_2_bias.shape(), DType.float32))
    velocities.append(zeros(model.conv4_3_kernel.shape(), DType.float32))
    velocities.append(zeros(model.conv4_3_bias.shape(), DType.float32))

    # Block 5 (6 params)
    velocities.append(zeros(model.conv5_1_kernel.shape(), DType.float32))
    velocities.append(zeros(model.conv5_1_bias.shape(), DType.float32))
    velocities.append(zeros(model.conv5_2_kernel.shape(), DType.float32))
    velocities.append(zeros(model.conv5_2_bias.shape(), DType.float32))
    velocities.append(zeros(model.conv5_3_kernel.shape(), DType.float32))
    velocities.append(zeros(model.conv5_3_bias.shape(), DType.float32))

    # FC layers (6 params)
    velocities.append(zeros(model.fc1_weights.shape(), DType.float32))
    velocities.append(zeros(model.fc1_bias.shape(), DType.float32))
    velocities.append(zeros(model.fc2_weights.shape(), DType.float32))
    velocities.append(zeros(model.fc2_bias.shape(), DType.float32))
    velocities.append(zeros(model.fc3_weights.shape(), DType.float32))
    velocities.append(zeros(model.fc3_bias.shape(), DType.float32))

    return velocities


fn main() raises:
    """Main training loop."""
    print("=" * 60)
    print("VGG-16 Training on CIFAR-10 Dataset")
    print("=" * 60)

    # Parse arguments
    var config = parse_args()
    var epochs = config[0]
    var batch_size = config[1]
    var learning_rate = config[2]
    var momentum = config[3]
    var data_dir = config[4]
    var weights_dir = config[5]

    print("\nConfiguration:")
    print("  Epochs: ", epochs)
    print("  Batch Size: ", batch_size)
    print("  Learning Rate: ", learning_rate)
    print("  Momentum: ", momentum)
    print("  Data Directory: ", data_dir)
    print("  Weights Directory: ", weights_dir)
    print()

    # Initialize model
    print("Initializing VGG-16 model...")
    var model = VGG16(num_classes=10, dropout_rate=0.5)
    print("  Model initialized with", model.num_classes, "classes")
    print("  Dropout rate:", model.dropout_rate)
    print()

    # Initialize momentum velocities
    print("Initializing momentum velocities...")
    var velocities = initialize_velocities(model)
    print("  Velocities initialized for 32 parameters")
    print()

    # Load dataset
    print("Loading CIFAR-10 dataset...")
    var train_data = load_cifar10_train(data_dir)
    var train_images = train_data[0]
    var train_labels = train_data[1]

    var test_data = load_cifar10_test(data_dir)
    var test_images = test_data[0]
    var test_labels = test_data[1]

    print("  Training samples: ", train_images.shape()[0])
    print("  Test samples: ", test_images.shape()[0])
    print()

    # Training loop with learning rate decay
    print("Starting training...")
    print("Learning rate schedule: step decay every 60 epochs by 0.2x")
    print()

    for epoch in range(1, epochs + 1):
        # Apply learning rate decay (step every 60 epochs, gamma=0.2)
        var current_lr = step_lr(learning_rate, epoch - 1, step_size=60, gamma=Float32(0.2))

        if epoch == 1 or epoch % 60 == 1:
            print("Epoch", epoch, "- Learning rate:", current_lr)

        var train_loss = train_epoch(model, train_images, train_labels, batch_size, current_lr, momentum, epoch, epochs, velocities)

        # Evaluate every epoch
        var test_acc = evaluate(model, test_images, test_labels)
        print()

        # Save model every 20 epochs
        if epoch % 20 == 0:
            print("Saving checkpoint at epoch", epoch, "...")
            model.save_weights(weights_dir)
            print("  Checkpoint saved to", weights_dir)
            print()

    # Save final model
    print("Saving final model weights...")
    model.save_weights(weights_dir)
    print("  Model saved to", weights_dir)
    print()

    print("Training complete!")
    print("\nNote: This implementation demonstrates the full training structure with complete manual backpropagation.")
    print("Batch processing will be more efficient when tensor slicing is optimized.")
