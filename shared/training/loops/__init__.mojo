"""
Training Loops

Training loop implementations for common training patterns.

Includes:
- TrainingLoop: Main training coordinator with support for both generic (callbacks-based)
  and manual (custom batch function) training patterns
- ValidationLoop: Training loop with validation support

The TrainingLoop consolidates patterns from all examples:
- Epoch iteration with configurable batch size
- Custom batch processing via compute_batch_loss callback
- Automatic progress reporting
- Evaluation support with custom eval function

Example Usage:
    # Create training loop with progress logging every 100 batches
    var loop = TrainingLoop(log_interval=100)

    # Run epoch with manual batch processing (AlexNet, VGG16, etc pattern)
    var avg_loss = loop.run_epoch_manual(
        train_images, train_labels, batch_size=128,
        compute_batch_loss=my_batch_fn,
        epoch=1, total_epochs=100
    )
"""

# Export training loop implementations
from .training_loop import TrainingLoop, train_one_epoch, training_step

# Export validation loop
from .validation_loop import ValidationLoop
