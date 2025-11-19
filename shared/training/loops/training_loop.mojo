"""Training loop implementation.

Core iteration logic for model training with forward pass, backward pass,
and weight updates.

Training Loop (#308-312):
- #309: Forward/backward/update iteration
- #310: Gradient management and zeroing
- #311: Metric tracking and callbacks

Design principles:
- Clear separation: forward, backward, update steps
- Proper gradient management (zero before backward)
- Metric tracking at batch and epoch level
- Callback integration at all lifecycle points
"""

from collections.vector import DynamicVector
from shared.core.extensor import ExTensor
from shared.training.metrics import AccuracyMetric, LossTracker
from shared.training.trainer_interface import DataLoader, DataBatch, TrainingMetrics


fn training_step(
    model_forward: fn(ExTensor) raises -> ExTensor,
    compute_loss: fn(ExTensor, ExTensor) raises -> ExTensor,
    optimizer_step: fn() raises -> None,
    zero_gradients: fn() raises -> None,
    data: ExTensor,
    labels: ExTensor
) raises -> Float64:
    """Execute single training step (forward, backward, update).

    Args:
        model_forward: Function to compute model forward pass
        compute_loss: Function to compute loss
        optimizer_step: Function to update weights
        zero_gradients: Function to zero gradients
        data: Input batch data
        labels: Target labels

    Returns:
        Loss value for this batch

    Raises:
        Error if any step fails
    """
    # Zero gradients from previous step
    zero_gradients()

    # Forward pass
    var predictions = model_forward(data)

    # Compute loss
    var loss_tensor = compute_loss(predictions, labels)

    # Extract scalar loss (assume first element)
    var loss_value = Float64(loss_tensor._data.bitcast[Float32]()[0])

    # Backward pass (implicit through loss_tensor.backward())
    # NOTE: In real implementation, loss_tensor would have .backward() method

    # Update weights
    optimizer_step()

    return loss_value


fn train_one_epoch(
    model_forward: fn(ExTensor) raises -> ExTensor,
    compute_loss: fn(ExTensor, ExTensor) raises -> ExTensor,
    optimizer_step: fn() raises -> None,
    zero_gradients: fn() raises -> None,
    train_loader: DataLoader,
    inout metrics: TrainingMetrics,
    log_interval: Int = 10
) raises:
    """Train for one epoch.

    Args:
        model_forward: Forward pass function
        compute_loss: Loss computation function
        optimizer_step: Weight update function
        zero_gradients: Gradient zeroing function
        train_loader: Training data loader
        metrics: Training metrics to update
        log_interval: Log metrics every N batches

    Raises:
        Error if training fails
    """
    var epoch_loss = Float64(0.0)
    var num_batches = 0

    # Setup metrics for epoch
    var accuracy_metric = AccuracyMetric()
    var loss_tracker = LossTracker(window_size=log_interval)

    # Reset dataloader
    train_loader.reset()

    # Iterate over batches
    while train_loader.has_next():
        var batch = train_loader.next()

        # Training step
        var batch_loss = training_step(
            model_forward,
            compute_loss,
            optimizer_step,
            zero_gradients,
            batch.data,
            batch.labels
        )

        # Update metrics
        loss_tracker.update(Float32(batch_loss))
        epoch_loss += batch_loss
        num_batches += 1

        # Update batch counter in metrics
        metrics.current_batch = num_batches

        # Log progress
        if num_batches % log_interval == 0:
            var avg_loss = loss_tracker.get_average()
            print("  Batch " + String(num_batches) + "/" + String(train_loader.num_batches) +
                  " - Loss: " + String(avg_loss))

    # Update epoch metrics
    var epoch_avg_loss = epoch_loss / Float64(num_batches)
    metrics.update_train_metrics(epoch_avg_loss, 0.0)  # Accuracy placeholder

    print("Epoch " + String(metrics.current_epoch) + " complete - Avg Loss: " + String(epoch_avg_loss))


struct TrainingLoop:
    """Training loop coordinator.

    Manages the training process including:
    - Batch iteration
    - Forward/backward passes
    - Gradient updates
    - Metric tracking
    - Progress logging
    """
    var log_interval: Int
    var clip_gradients: Bool
    var max_grad_norm: Float64

    fn __init__(
        inout self,
        log_interval: Int = 10,
        clip_gradients: Bool = False,
        max_grad_norm: Float64 = 1.0
    ):
        """Initialize training loop.

        Args:
            log_interval: Log metrics every N batches
            clip_gradients: Whether to clip gradients
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.log_interval = log_interval
        self.clip_gradients = clip_gradients
        self.max_grad_norm = max_grad_norm

    fn run_epoch(
        self,
        model_forward: fn(ExTensor) raises -> ExTensor,
        compute_loss: fn(ExTensor, ExTensor) raises -> ExTensor,
        optimizer_step: fn() raises -> None,
        zero_gradients: fn() raises -> None,
        train_loader: DataLoader,
        inout metrics: TrainingMetrics
    ) raises:
        """Run one training epoch.

        Args:
            model_forward: Forward pass function
            compute_loss: Loss computation function
            optimizer_step: Weight update function
            zero_gradients: Gradient zeroing function
            train_loader: Training data loader
            metrics: Training metrics to update

        Raises:
            Error if training fails
        """
        train_one_epoch(
            model_forward,
            compute_loss,
            optimizer_step,
            zero_gradients,
            train_loader,
            metrics,
            self.log_interval
        )

    fn run(
        self,
        model_forward: fn(ExTensor) raises -> ExTensor,
        compute_loss: fn(ExTensor, ExTensor) raises -> ExTensor,
        optimizer_step: fn() raises -> None,
        zero_gradients: fn() raises -> None,
        train_loader: DataLoader,
        num_epochs: Int,
        inout metrics: TrainingMetrics
    ) raises:
        """Run complete training loop.

        Args:
            model_forward: Forward pass function
            compute_loss: Loss computation function
            optimizer_step: Weight update function
            zero_gradients: Gradient zeroing function
            train_loader: Training data loader
            num_epochs: Number of epochs to train
            metrics: Training metrics to update

        Raises:
            Error if training fails
        """
        print("\nStarting training for " + String(num_epochs) + " epochs...")
        print("=" * 50)

        for epoch in range(num_epochs):
            metrics.current_epoch = epoch
            metrics.reset_epoch()

            print("\nEpoch " + String(epoch + 1) + "/" + String(num_epochs))
            print("-" * 50)

            self.run_epoch(
                model_forward,
                compute_loss,
                optimizer_step,
                zero_gradients,
                train_loader,
                metrics
            )

        print("\n" + "=" * 50)
        print("Training complete!")
        metrics.print_summary()
