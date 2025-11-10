# Usage Examples - ML Odyssey Shared Library

This document provides comprehensive usage examples for the ML Odyssey shared library.

**Note**: Examples use commented imports for components not yet implemented. Uncomment imports as Issue #49 completes implementation.

## Table of Contents

1. [Basic Neural Network](#basic-neural-network)
2. [Convolutional Neural Network](#convolutional-neural-network)
3. [Training with Validation](#training-with-validation)
4. [Custom Training Loop](#custom-training-loop)
5. [Data Loading](#data-loading)
6. [Learning Rate Scheduling](#learning-rate-scheduling)
7. [Callbacks and Monitoring](#callbacks-and-monitoring)
8. [Model Checkpointing](#model-checkpointing)
9. [Multiple Metrics](#multiple-metrics)
10. [Complete Example: MNIST Classifier](#complete-example-mnist-classifier)

## Basic Neural Network

Simple feedforward network for classification:

```mojo
from shared import Linear, ReLU, Sequential, SGD, Tensor

fn basic_network_example():
    """Create and use a basic neural network."""

    # Define model architecture
    var model = Sequential([
        Linear(784, 256),
        ReLU(),
        Linear(256, 128),
        ReLU(),
        Linear(128, 10),
    ])

    # Create optimizer
    var optimizer = SGD(
        learning_rate=0.01,
        momentum=0.9,
        weight_decay=1e-4
    )

    # Training loop (simplified)
    for epoch in range(10):
        # Forward pass
        var outputs = model.forward(inputs)

        # Compute loss
        var loss = cross_entropy_loss(outputs, targets)

        # Backward pass
        var grads = compute_gradients(loss, model)

        # Update parameters
        optimizer.step(model.parameters(), grads)

        print("Epoch", epoch, "Loss:", loss.item())
```

## Convolutional Neural Network

CNN for image classification:

```mojo
from shared import (
    Conv2D, ReLU, MaxPool2D, Flatten, Linear, Sequential,
    Adam, Tensor
)

fn cnn_example():
    """Create a convolutional neural network."""

    # Define CNN architecture
    var model = Sequential([
        # Convolutional layers
        Conv2D(in_channels=1, out_channels=32, kernel_size=3, padding=1),
        ReLU(),
        MaxPool2D(kernel_size=2, stride=2),

        Conv2D(32, 64, kernel_size=3, padding=1),
        ReLU(),
        MaxPool2D(kernel_size=2, stride=2),

        # Fully connected layers
        Flatten(),
        Linear(64 * 7 * 7, 128),
        ReLU(),
        Linear(128, 10),
    ])

    # Create Adam optimizer
    var optimizer = Adam(
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8
    )

    print("CNN created with", model.num_parameters(), "parameters")
```

## Training with Validation

Training loop with validation:

```mojo
from shared import (
    Linear, ReLU, Sequential,
    Adam, CosineAnnealingLR,
    Accuracy, LossTracker,
    DataLoader,
)

fn training_with_validation():
    """Train model with validation."""

    # Create model
    var model = Sequential([
        Linear(784, 256),
        ReLU(),
        Linear(256, 10),
    ])

    # Create optimizer and scheduler
    var optimizer = Adam(learning_rate=0.001)
    var scheduler = CosineAnnealingLR(
        optimizer=optimizer,
        T_max=100,
        eta_min=1e-6
    )

    # Create metrics
    var train_loss = LossTracker()
    var train_acc = Accuracy()
    var val_loss = LossTracker()
    var val_acc = Accuracy()

    # Training loop
    for epoch in range(100):
        # Training phase
        model.train()
        train_loss.reset()
        train_acc.reset()

        for batch in train_loader:
            # Forward pass
            var outputs = model.forward(batch.inputs)
            var loss = criterion(outputs, batch.targets)

            # Backward pass
            var grads = compute_gradients(loss, model)
            optimizer.step(model.parameters(), grads)

            # Update metrics
            train_loss.update(loss.item())
            train_acc.update(outputs, batch.targets)

        # Validation phase
        model.eval()
        val_loss.reset()
        val_acc.reset()

        for batch in val_loader:
            var outputs = model.forward(batch.inputs)
            var loss = criterion(outputs, batch.targets)

            val_loss.update(loss.item())
            val_acc.update(outputs, batch.targets)

        # Update learning rate
        scheduler.step()

        # Print metrics
        print("Epoch", epoch)
        print("  Train Loss:", train_loss.compute(), "Acc:", train_acc.compute())
        print("  Val Loss:", val_loss.compute(), "Acc:", val_acc.compute())
        print("  LR:", optimizer.get_lr())
```

## Custom Training Loop

Custom training loop with paper-specific logic:

```mojo
from shared import SGD, Accuracy, Tensor
from shared.core.layers import Linear, ReLU

fn custom_training_loop(
    model: Model,
    train_data: DataLoader,
    val_data: DataLoader,
    epochs: Int
):
    """Custom training loop with special logic."""

    # Create optimizer with custom settings
    var optimizer = SGD(
        learning_rate=0.01,
        momentum=0.9,
        dampening=0.0,
        weight_decay=1e-4,
        nesterov=True
    )

    # Metrics
    var best_val_acc: Float32 = 0.0
    var patience_counter: Int = 0
    var patience: Int = 10

    for epoch in range(epochs):
        # Training phase
        var total_loss: Float32 = 0.0
        var train_acc = Accuracy()

        for batch in train_data:
            # Forward pass
            var outputs = model.forward(batch.inputs)
            var loss = compute_loss(outputs, batch.targets)

            # Add custom regularization (paper-specific)
            loss = loss + custom_regularization(model)

            # Backward pass
            var grads = compute_gradients(loss, model)

            # Gradient clipping (if needed)
            grads = clip_gradients(grads, max_norm=1.0)

            # Optimizer step
            optimizer.step(model.parameters(), grads)

            # Track metrics
            total_loss += loss.item()
            train_acc.update(outputs, batch.targets)

        # Validation phase
        var val_acc = Accuracy()

        for batch in val_data:
            var outputs = model.forward(batch.inputs)
            val_acc.update(outputs, batch.targets)

        var val_accuracy = val_acc.compute()

        # Early stopping logic
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            patience_counter = 0
            save_checkpoint(model, "best_model.mojo")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping at epoch", epoch)
            break

        # Custom learning rate adjustment
        if epoch % 30 == 0 and epoch > 0:
            optimizer.set_lr(optimizer.get_lr() * 0.1)

        print("Epoch", epoch, "Train Acc:", train_acc.compute(), "Val Acc:", val_accuracy)
```

## Data Loading

Data loading with transforms:

```mojo
from shared import (
    TensorDataset, DataLoader,
    Compose, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip
)

fn data_loading_example():
    """Create data loaders with transforms."""

    # Training transforms (with augmentation)
    var train_transform = Compose([
        RandomCrop(28, padding=4),
        RandomHorizontalFlip(probability=0.5),
        ToTensor(),
        Normalize(mean=0.1307, std=0.3081),
    ])

    # Validation transforms (no augmentation)
    var val_transform = Compose([
        ToTensor(),
        Normalize(mean=0.1307, std=0.3081),
    ])

    # Create datasets
    var train_dataset = TensorDataset(
        train_data,
        train_labels,
        transform=train_transform
    )

    var val_dataset = TensorDataset(
        val_data,
        val_labels,
        transform=val_transform
    )

    # Create data loaders
    var train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        drop_last=True,
        num_workers=4
    )

    var val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
        drop_last=False,
        num_workers=2
    )

    # Iterate over batches
    for batch in train_loader:
        # batch.inputs: Tensor of shape (batch_size, channels, height, width)
        # batch.targets: Tensor of shape (batch_size,)
        print("Batch shape:", batch.inputs.shape())
```

## Learning Rate Scheduling

Using learning rate schedulers:

```mojo
from shared import (
    SGD, Adam,
    StepLR, CosineAnnealingLR, ExponentialLR, WarmupLR
)

fn lr_scheduling_example():
    """Examples of learning rate scheduling."""

    # Example 1: Step decay
    var optimizer1 = SGD(learning_rate=0.1, momentum=0.9)
    var scheduler1 = StepLR(
        optimizer=optimizer1,
        step_size=30,    # Decay every 30 epochs
        gamma=0.1        # Multiply by 0.1
    )

    # Example 2: Cosine annealing
    var optimizer2 = Adam(learning_rate=0.001)
    var scheduler2 = CosineAnnealingLR(
        optimizer=optimizer2,
        T_max=100,       # Period of cosine cycle
        eta_min=1e-6     # Minimum learning rate
    )

    # Example 3: Exponential decay
    var optimizer3 = SGD(learning_rate=0.01)
    var scheduler3 = ExponentialLR(
        optimizer=optimizer3,
        gamma=0.95       # Decay rate per epoch
    )

    # Example 4: Warmup then decay
    var optimizer4 = Adam(learning_rate=0.001)
    var warmup = WarmupLR(
        optimizer=optimizer4,
        warmup_epochs=10,
        start_lr=1e-6
    )
    var main_scheduler = CosineAnnealingLR(
        optimizer=optimizer4,
        T_max=90
    )

    # Training loop with scheduling
    for epoch in range(100):
        # Training code...

        # Update learning rate
        if epoch < 10:
            warmup.step()
        else:
            main_scheduler.step()

        print("Epoch", epoch, "LR:", optimizer4.get_lr())
```

## Callbacks and Monitoring

Using callbacks for training monitoring:

```mojo
from shared import (
    EarlyStopping, ModelCheckpoint, Logger,
    LRSchedulerCallback
)

fn callbacks_example():
    """Using callbacks for training monitoring."""

    # Create callbacks
    var early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",            # "min" for loss, "max" for accuracy
        min_delta=0.001,       # Minimum change to qualify as improvement
        verbose=True
    )

    var checkpoint = ModelCheckpoint(
        filepath="checkpoints/model_{epoch:03d}_{val_acc:.3f}.mojo",
        monitor="val_acc",
        mode="max",
        save_best_only=True,
        save_freq=1            # Save every epoch
    )

    var logger = Logger(
        log_file="training.log",
        log_level="INFO",
        metrics=["loss", "acc", "val_loss", "val_acc"]
    )

    var lr_callback = LRSchedulerCallback(scheduler)

    # Training loop with callbacks
    for epoch in range(100):
        # On epoch start
        early_stop.on_epoch_begin(epoch)
        checkpoint.on_epoch_begin(epoch)
        logger.on_epoch_begin(epoch)

        # Training...
        train_loss = train_epoch(model, optimizer, train_loader)
        val_loss = validate_epoch(model, val_loader)

        # Create logs dictionary
        var logs = {
            "loss": train_loss,
            "acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        }

        # On epoch end
        early_stop.on_epoch_end(epoch, logs)
        checkpoint.on_epoch_end(epoch, logs)
        logger.on_epoch_end(epoch, logs)
        lr_callback.on_epoch_end(epoch, logs)

        # Check if should stop
        if early_stop.should_stop:
            print("Early stopping triggered at epoch", epoch)
            break
```

## Model Checkpointing

Saving and loading models:

```mojo
from shared import ModelCheckpoint, Logger

fn checkpointing_example():
    """Save and load model checkpoints."""

    # Save checkpoint manually
    fn save_checkpoint(model, optimizer, epoch, filepath):
        var checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss": current_loss,
        }
        save(checkpoint, filepath)
        print("Saved checkpoint to", filepath)

    # Load checkpoint
    fn load_checkpoint(filepath) -> Checkpoint:
        var checkpoint = load(filepath)
        return checkpoint

    # Using checkpoint callback
    var checkpoint_callback = ModelCheckpoint(
        filepath="checkpoints/best_model.mojo",
        monitor="val_acc",
        mode="max",
        save_best_only=True,
        save_weights_only=False  # Save full model + optimizer state
    )

    # In training loop
    for epoch in range(epochs):
        # Training...

        var logs = {"val_acc": val_accuracy}
        checkpoint_callback.on_epoch_end(epoch, logs)

    # Resume from checkpoint
    fn resume_training(checkpoint_path):
        var checkpoint = load_checkpoint(checkpoint_path)

        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        var start_epoch = checkpoint["epoch"] + 1

        print("Resuming from epoch", start_epoch)

        # Continue training
        for epoch in range(start_epoch, total_epochs):
            # Training...
            pass
```

## Multiple Metrics

Tracking multiple metrics:

```mojo
from shared import (
    Accuracy, LossTracker, Precision, Recall, ConfusionMatrix
)

fn multiple_metrics_example():
    """Track multiple evaluation metrics."""

    # Create metrics
    var train_loss = LossTracker()
    var train_acc = Accuracy()

    var val_loss = LossTracker()
    var val_acc = Accuracy()
    var val_precision = Precision(num_classes=10)
    var val_recall = Recall(num_classes=10)
    var val_confusion = ConfusionMatrix(num_classes=10)

    # Training loop
    for epoch in range(epochs):
        # Reset metrics
        train_loss.reset()
        train_acc.reset()

        # Training phase
        for batch in train_loader:
            var outputs = model.forward(batch.inputs)
            var loss = criterion(outputs, batch.targets)

            # Update training metrics
            train_loss.update(loss.item())
            train_acc.update(outputs, batch.targets)

            # Backprop and optimization...

        # Reset validation metrics
        val_loss.reset()
        val_acc.reset()
        val_precision.reset()
        val_recall.reset()
        val_confusion.reset()

        # Validation phase
        for batch in val_loader:
            var outputs = model.forward(batch.inputs)
            var loss = criterion(outputs, batch.targets)

            # Update validation metrics
            val_loss.update(loss.item())
            val_acc.update(outputs, batch.targets)
            val_precision.update(outputs, batch.targets)
            val_recall.update(outputs, batch.targets)
            val_confusion.update(outputs, batch.targets)

        # Print all metrics
        print("Epoch", epoch)
        print("  Train Loss:", train_loss.compute(), "Acc:", train_acc.compute())
        print("  Val Loss:", val_loss.compute(), "Acc:", val_acc.compute())
        print("  Precision:", val_precision.compute())
        print("  Recall:", val_recall.compute())

        # Print confusion matrix
        print("  Confusion Matrix:")
        print(val_confusion.compute())
```

## Complete Example: MNIST Classifier

Full example bringing it all together:

```mojo
from shared import (
    # Core
    Linear, ReLU, Dropout, Sequential,
    # Training
    Adam, CosineAnnealingLR,
    Accuracy, LossTracker,
    EarlyStopping, ModelCheckpoint, Logger,
    # Data
    TensorDataset, DataLoader, Compose, ToTensor, Normalize,
)

fn mnist_classifier_complete():
    """Complete MNIST classifier example."""

    # ========================================================================
    # 1. Data Preparation
    # ========================================================================

    # Load MNIST data (assuming loaded from somewhere)
    var train_images = load_mnist_images("train")
    var train_labels = load_mnist_labels("train")
    var test_images = load_mnist_images("test")
    var test_labels = load_mnist_labels("test")

    # Create transforms
    var transform = Compose([
        ToTensor(),
        Normalize(mean=0.1307, std=0.3081),
    ])

    # Create datasets
    var train_dataset = TensorDataset(train_images, train_labels, transform)
    var test_dataset = TensorDataset(test_images, test_labels, transform)

    # Create data loaders
    var train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        drop_last=True
    )

    var test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False
    )

    # ========================================================================
    # 2. Model Definition
    # ========================================================================

    var model = Sequential([
        Linear(784, 512),
        ReLU(),
        Dropout(0.2),

        Linear(512, 256),
        ReLU(),
        Dropout(0.2),

        Linear(256, 128),
        ReLU(),
        Dropout(0.1),

        Linear(128, 10),
    ])

    print("Model created with", model.num_parameters(), "parameters")

    # ========================================================================
    # 3. Training Setup
    # ========================================================================

    # Optimizer
    var optimizer = Adam(
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        weight_decay=1e-4
    )

    # Learning rate scheduler
    var scheduler = CosineAnnealingLR(
        optimizer=optimizer,
        T_max=50,
        eta_min=1e-6
    )

    # Metrics
    var train_loss = LossTracker()
    var train_acc = Accuracy()
    var test_loss = LossTracker()
    var test_acc = Accuracy()

    # Callbacks
    var early_stop = EarlyStopping(
        monitor="test_loss",
        patience=10,
        mode="min"
    )

    var checkpoint = ModelCheckpoint(
        filepath="checkpoints/mnist_best.mojo",
        monitor="test_acc",
        mode="max",
        save_best_only=True
    )

    var logger = Logger(
        log_file="mnist_training.log",
        metrics=["loss", "acc", "test_loss", "test_acc", "lr"]
    )

    # ========================================================================
    # 4. Training Loop
    # ========================================================================

    var num_epochs = 50

    for epoch in range(num_epochs):
        print("\nEpoch", epoch + 1, "/", num_epochs)

        # ====================================================================
        # Training Phase
        # ====================================================================

        model.train()
        train_loss.reset()
        train_acc.reset()

        for batch_idx, batch in enumerate(train_loader):
            # Forward pass
            var outputs = model.forward(batch.inputs)
            var loss = cross_entropy_loss(outputs, batch.targets)

            # Backward pass
            var grads = compute_gradients(loss, model)
            optimizer.step(model.parameters(), grads)
            optimizer.zero_grad()

            # Update metrics
            train_loss.update(loss.item())
            train_acc.update(outputs, batch.targets)

            # Print progress
            if batch_idx % 100 == 0:
                print("  Train Batch", batch_idx, "/", len(train_loader),
                      "Loss:", loss.item())

        # ====================================================================
        # Test Phase
        # ====================================================================

        model.eval()
        test_loss.reset()
        test_acc.reset()

        for batch in test_loader:
            var outputs = model.forward(batch.inputs)
            var loss = cross_entropy_loss(outputs, batch.targets)

            test_loss.update(loss.item())
            test_acc.update(outputs, batch.targets)

        # ====================================================================
        # Logging and Callbacks
        # ====================================================================

        var logs = {
            "loss": train_loss.compute(),
            "acc": train_acc.compute(),
            "test_loss": test_loss.compute(),
            "test_acc": test_acc.compute(),
            "lr": optimizer.get_lr()
        }

        # Print epoch summary
        print("\nEpoch", epoch + 1, "Summary:")
        print("  Train Loss:", logs["loss"], "Acc:", logs["acc"])
        print("  Test Loss:", logs["test_loss"], "Acc:", logs["test_acc"])
        print("  Learning Rate:", logs["lr"])

        # Update callbacks
        logger.on_epoch_end(epoch, logs)
        checkpoint.on_epoch_end(epoch, logs)
        early_stop.on_epoch_end(epoch, logs)

        # Update learning rate
        scheduler.step()

        # Check early stopping
        if early_stop.should_stop:
            print("\nEarly stopping triggered!")
            print("Best test accuracy:", early_stop.best_score)
            break

    # ========================================================================
    # 5. Final Evaluation
    # ========================================================================

    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)

    # Load best model
    model.load_state_dict(load("checkpoints/mnist_best.mojo"))

    # Final evaluation
    test_acc.reset()
    for batch in test_loader:
        var outputs = model.forward(batch.inputs)
        test_acc.update(outputs, batch.targets)

    print("Final Test Accuracy:", test_acc.compute())
```

## Next Steps

- Review [API Documentation](docs/api/) for detailed API reference
- Check [INSTALL.md](INSTALL.md) for installation instructions
- See [README.md](README.md) for library overview
- Browse tests in `tests/shared/` for more examples

## Notes

- Examples marked with comments indicate components not yet implemented
- Uncomment imports as implementation progresses (Issue #49)
- All examples follow Mojo best practices (fn, struct, SIMD where applicable)
- See individual module documentation for more details
