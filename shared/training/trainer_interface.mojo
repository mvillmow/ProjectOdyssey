"""Trainer interface and contract definition.

Defines the abstract interface that all trainers must implement,
establishing a consistent API for training, validation, and testing.

Trainer Interface (#303-307):
- #304: Trainer trait definition
- #305: Configuration structures
- #306: State management interfaces

Design principles:
- Trait-based polymorphism for flexible implementations
- Composition over inheritance
- Explicit configuration objects
- Clear separation of concerns (train, validate, test)
"""

from collections.vector import DynamicVector
from extensor import ExTensor


struct TrainerConfig:
    """Configuration for trainer behavior.

    Centralizes all training hyperparameters and settings.
    """
    var num_epochs: Int
    var batch_size: Int
    var learning_rate: Float64
    var log_interval: Int  # Log metrics every N batches
    var validate_interval: Int  # Validate every N epochs (0 = every epoch)
    var save_checkpoints: Bool
    var checkpoint_interval: Int  # Save checkpoint every N epochs

    fn __init__(
        inout self,
        num_epochs: Int = 10,
        batch_size: Int = 32,
        learning_rate: Float64 = 0.001,
        log_interval: Int = 10,
        validate_interval: Int = 1,
        save_checkpoints: Bool = False,
        checkpoint_interval: Int = 5
    ):
        """Initialize trainer configuration with defaults."""
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.log_interval = log_interval
        self.validate_interval = validate_interval
        self.save_checkpoints = save_checkpoints
        self.checkpoint_interval = checkpoint_interval


struct TrainingMetrics:
    """Metrics collected during training.

    Stores current and historical metrics for analysis.
    """
    var current_epoch: Int
    var current_batch: Int
    var total_batches: Int
    var train_loss: Float64
    var train_accuracy: Float64
    var val_loss: Float64
    var val_accuracy: Float64
    var best_val_loss: Float64
    var best_val_accuracy: Float64
    var best_epoch: Int

    fn __init__(inout self):
        """Initialize training metrics with defaults."""
        self.current_epoch = 0
        self.current_batch = 0
        self.total_batches = 0
        self.train_loss = 0.0
        self.train_accuracy = 0.0
        self.val_loss = 0.0
        self.val_accuracy = 0.0
        self.best_val_loss = 1e10  # Initialize to very large value
        self.best_val_accuracy = 0.0
        self.best_epoch = 0

    fn update_train_metrics(inout self, loss: Float64, accuracy: Float64):
        """Update training metrics for current batch.

        Args:
            loss: Current batch loss
            accuracy: Current batch accuracy
        """
        self.train_loss = loss
        self.train_accuracy = accuracy

    fn update_val_metrics(inout self, loss: Float64, accuracy: Float64):
        """Update validation metrics and track best results.

        Args:
            loss: Validation loss
            accuracy: Validation accuracy
        """
        self.val_loss = loss
        self.val_accuracy = accuracy

        # Track best validation results
        if loss < self.best_val_loss:
            self.best_val_loss = loss
            self.best_val_accuracy = accuracy
            self.best_epoch = self.current_epoch

    fn reset_epoch(inout self):
        """Reset epoch-level metrics."""
        self.current_batch = 0
        self.train_loss = 0.0
        self.train_accuracy = 0.0

    fn print_summary(self):
        """Print training metrics summary."""
        print("\nTraining Metrics Summary:")
        print("-" * 50)
        print("Current Epoch: " + String(self.current_epoch))
        print("Train Loss: " + String(self.train_loss))
        print("Train Accuracy: " + String(self.train_accuracy))
        print("Val Loss: " + String(self.val_loss))
        print("Val Accuracy: " + String(self.val_accuracy))
        print("Best Val Loss: " + String(self.best_val_loss) + " (epoch " + String(self.best_epoch) + ")")
        print("Best Val Accuracy: " + String(self.best_val_accuracy) + " (epoch " + String(self.best_epoch) + ")")
        print("-" * 50)


trait Trainer:
    """Abstract interface for all trainer implementations.

    Defines the contract that all trainers must follow, ensuring
    consistent API across different training strategies.

    Key methods:
    - train(): Execute training loop
    - validate(): Evaluate on validation set
    - test(): Evaluate on test set
    - fit(): Convenience method combining train+validate

    Lifecycle hooks (via callbacks):
    - on_train_begin()
    - on_train_end()
    - on_epoch_begin()
    - on_epoch_end()
    - on_batch_begin()
    - on_batch_end()
    - on_validation_begin()
    - on_validation_end()
    """

    fn train(inout self, num_epochs: Int) raises:
        """Execute training loop for specified number of epochs.

        Args:
            num_epochs: Number of epochs to train

        Raises:
            Error if training fails
        """
        ...

    fn validate(inout self) raises -> Float64:
        """Evaluate model on validation set.

        Returns:
            Validation loss

        Raises:
            Error if validation fails
        """
        ...

    fn fit(inout self, num_epochs: Int, validate_every: Int = 1) raises:
        """Train model with periodic validation.

        Args:
            num_epochs: Number of epochs to train
            validate_every: Validate every N epochs (default=1)

        Raises:
            Error if training or validation fails
        """
        ...


struct DataBatch:
    """Single batch of training/validation data.

    Represents a mini-batch with input features and labels.
    """
    var data: ExTensor  # Input features [batch_size, feature_dim]
    var labels: ExTensor  # Labels [batch_size] or [batch_size, num_classes]
    var batch_size: Int

    fn __init__(inout self, data: ExTensor, labels: ExTensor):
        """Initialize data batch.

        Args:
            data: Input features tensor
            labels: Labels tensor
        """
        self.data = data
        self.labels = labels
        self.batch_size = data.shape[0]


struct DataLoader:
    """Simple data loader for batching.

    Provides iteration over dataset in batches.
    NOTE: This is a minimal implementation for testing.
    Production code should use proper data loading infrastructure.
    """
    var data: ExTensor
    var labels: ExTensor
    var batch_size: Int
    var num_samples: Int
    var num_batches: Int
    var current_batch: Int

    fn __init__(inout self, data: ExTensor, labels: ExTensor, batch_size: Int):
        """Initialize data loader.

        Args:
            data: Full dataset features
            labels: Full dataset labels
            batch_size: Batch size
        """
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.num_samples = data.shape[0]
        self.num_batches = (self.num_samples + batch_size - 1) // batch_size
        self.current_batch = 0

    fn reset(inout self):
        """Reset loader to beginning."""
        self.current_batch = 0

    fn has_next(self) -> Bool:
        """Check if more batches available.

        Returns:
            True if more batches available
        """
        return self.current_batch < self.num_batches

    fn next(inout self) raises -> DataBatch:
        """Get next batch.

        Returns:
            Next data batch

        Raises:
            Error if no more batches
        """
        if not self.has_next():
            raise Error("No more batches available")

        var start_idx = self.current_batch * self.batch_size
        var end_idx = min(start_idx + self.batch_size, self.num_samples)
        var actual_batch_size = end_idx - start_idx

        # Extract batch slice
        var batch_data_shape = DynamicVector[Int](actual_batch_size, self.data.shape[1])
        var batch_data = ExTensor(batch_data_shape, self.data.dtype)

        var batch_labels_shape = DynamicVector[Int](actual_batch_size)
        var batch_labels = ExTensor(batch_labels_shape, self.labels.dtype)

        # Copy data (simplified - real implementation would use slicing)
        # For now, we'll just create placeholders
        # TODO: Implement proper tensor slicing in ExTensor

        self.current_batch += 1

        return DataBatch(batch_data, batch_labels)


fn create_simple_dataloader(data: ExTensor, labels: ExTensor, batch_size: Int) -> DataLoader:
    """Create a simple dataloader for training.

    Args:
        data: Full dataset features
        labels: Full dataset labels
        batch_size: Batch size

    Returns:
        DataLoader instance
    """
    return DataLoader(data, labels, batch_size)
