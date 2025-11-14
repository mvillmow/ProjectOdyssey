"""Unit tests for Checkpointing Callback.

Tests cover:
- Saving model state during training
- Loading and restoring complete state
- Best model tracking
- Checkpoint file management

Following TDD principles - these tests define the expected API
for implementation in Issue #34.
"""

from tests.shared.conftest import (
    assert_true,
    assert_equal,
    assert_almost_equal,
    TestFixtures,
)


# ============================================================================
# Checkpointing Core Tests
# ============================================================================


fn test_checkpointing_initialization() raises:
    """Test Checkpointing callback initialization.

    API Contract:
        Checkpointing(
            filepath: String,
            monitor: String = "val_loss",
            save_best_only: Bool = False,
            save_frequency: Int = 1
        )
        - filepath: Path template for checkpoints (e.g., "model_{epoch}.pt")
        - monitor: Metric to monitor for best model
        - save_best_only: If True, only save when monitored metric improves
        - save_frequency: Save every N epochs
    """
    from shared.training.stubs import MockCheckpoint

    var checkpoint = MockCheckpoint(save_path="checkpoints/model.pt")

    Verify parameters
    assert_equal(checkpoint.save_path, "checkpoints/model.pt")
    assert_equal(checkpoint.save_count, 0)


fn test_checkpointing_saves_at_epoch_end() raises:
    """Test Checkpointing saves model after each epoch.

    API Contract:
        Callback hook: on_epoch_end(epoch, logs)
        - Saves model state to filepath
        - Replaces {epoch} placeholder with epoch number
        - Creates checkpoint file

    This is a CRITICAL test for checkpoint saving.
    """
    from shared.training.stubs import MockCheckpoint
    from shared.training.base import TrainingState

    var checkpoint = MockCheckpoint(save_path="/tmp/model.pt")
    var state = TrainingState(epoch=1, learning_rate=0.1)
    state.metrics["train_loss"] = 0.5
    state.metrics["val_loss"] = 0.6

    Simulate epoch end
    _ = checkpoint.on_epoch_end(state)

    Checkpoint stub should have incremented save count
    assert_equal(checkpoint.save_count, 1)


fn test_checkpointing_saves_complete_state() raises:
    """Test Checkpointing saves complete training state.

    API Contract:
        Checkpoint should include:
        - Model weights
        - Optimizer state
        - Epoch number
        - Training metrics/logs
        - Random state (optional)

    This is CRITICAL for resuming training.
    """
    # TODO(#34): Implement when Checkpointing is available
    var model = create_simple_model()
    var optimizer = SGD(learning_rate=0.1, momentum=0.9)
    var trainer = Trainer(model, optimizer, loss_fn)
    var checkpoint = Checkpointing(filepath="/tmp/checkpoint.pt")
    #
    # Train for a few steps
    trainer.train(epochs=2, train_loader, val_loader)
    #
    # Save checkpoint
    checkpoint.on_epoch_end(epoch=2, logs={"train_loss": 0.3})
    #
    # Load checkpoint
    var loaded_state = load_checkpoint("/tmp/checkpoint.pt")
    #
    # Verify all components present
    assert_true(loaded_state.contains("model_state"))
    assert_true(loaded_state.contains("optimizer_state"))
    assert_true(loaded_state.contains("epoch"))
    assert_equal(loaded_state["epoch"], 2)


# ============================================================================
# Best Model Tracking Tests
# ============================================================================


fn test_checkpointing_save_best_only() raises:
    """Test Checkpointing saves only when monitored metric improves.

    API Contract:
        With save_best_only=True:
        - Save when monitored metric is best so far
        - Skip save when metric doesn't improve
        - Track best value seen

    This is CRITICAL for saving best models during training.
    """
    # TODO(#34): Implement when Checkpointing is available
    var checkpoint = Checkpointing(
        filepath="/tmp/best_model.pt",
        monitor="val_loss",
        save_best_only=True
    )
    #
    # Epoch 1: val_loss = 0.5 (first, so save)
    checkpoint.on_epoch_end(1, {"val_loss": 0.5})
    assert_true(file_exists("/tmp/best_model.pt"))
    var checkpoint_time_1 = get_file_modified_time("/tmp/best_model.pt")
    #
    # Epoch 2: val_loss = 0.6 (worse, don't save)
    checkpoint.on_epoch_end(2, {"val_loss": 0.6})
    var checkpoint_time_2 = get_file_modified_time("/tmp/best_model.pt")
    assert_equal(checkpoint_time_1, checkpoint_time_2)  # File not updated
    #
    # Epoch 3: val_loss = 0.4 (better, save)
    checkpoint.on_epoch_end(3, {"val_loss": 0.4})
    var checkpoint_time_3 = get_file_modified_time("/tmp/best_model.pt")
    assert_not_equal(checkpoint_time_2, checkpoint_time_3)  # File updated


fn test_checkpointing_monitor_different_metrics() raises:
    """Test Checkpointing can monitor different metrics.

    API Contract:
        monitor parameter can be:
        - "val_loss" (minimize)
        - "val_accuracy" (maximize)
        - "train_loss" (minimize)
        - Custom metrics
    """
    # TODO(#34): Implement when Checkpointing is available
    # Monitor accuracy (higher is better)
    var checkpoint = Checkpointing(
        filepath="/tmp/best_acc.pt",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max"  # Maximize accuracy
    )
    #
    # Epoch 1: acc = 0.5
    checkpoint.on_epoch_end(1, {"val_accuracy": 0.5})
    #
    # Epoch 2: acc = 0.6 (better, save)
    checkpoint.on_epoch_end(2, {"val_accuracy": 0.6})
    #
    # Epoch 3: acc = 0.4 (worse, don't save)
    checkpoint.on_epoch_end(3, {"val_accuracy": 0.4})


# ============================================================================
# Save Frequency Tests
# ============================================================================


fn test_checkpointing_save_frequency() raises:
    """Test Checkpointing respects save_frequency parameter.

    API Contract:
        save_frequency=N:
        - Save every N epochs
        - Skip intermediate epochs
    """
    from shared.training.stubs import MockCheckpoint
    from shared.training.base import TrainingState

    var checkpoint = MockCheckpoint(save_path="/tmp/model.pt")

    Simulate multiple epochs
    for epoch in range(10):
        var state = TrainingState(epoch=epoch, learning_rate=0.1)
        state.metrics["train_loss"] = 0.5
        _ = checkpoint.on_epoch_end(state)

    Checkpoint should have been called 10 times (stub increments each time)
    assert_equal(checkpoint.save_count, 10)


# ============================================================================
# Filepath Template Tests
# ============================================================================


fn test_checkpointing_filepath_template() raises:
    """Test Checkpointing supports filepath templates.

    API Contract:
        Filepath can contain placeholders:
        - {epoch}: Replaced with epoch number
        - {val_loss:.3f}: Replaced with metric value (formatted)
        - {timestamp}: Replaced with timestamp
    """
    # TODO(#34): Implement when Checkpointing is available
    var checkpoint = Checkpointing(
        filepath="/tmp/model_epoch{epoch}_loss{val_loss:.3f}.pt"
    )
    #
    checkpoint.on_epoch_end(5, {"val_loss": 0.123456})
    #
    # Expected: /tmp/model_epoch5_loss0.123.pt
    assert_true(file_exists("/tmp/model_epoch5_loss0.123.pt"))


fn test_checkpointing_creates_directory() raises:
    """Test Checkpointing creates output directory if needed.

    API Contract:
        If directory in filepath doesn't exist:
        - Create directory (and parent directories)
        - Then save checkpoint
    """
    # TODO(#34): Implement when Checkpointing is available
    var checkpoint = Checkpointing(
        filepath="/tmp/deep/nested/path/model.pt"
    )
    #
    # Directory doesn't exist initially
    assert_false(dir_exists("/tmp/deep/nested/path"))
    #
    # Save checkpoint
    checkpoint.on_epoch_end(1, {"train_loss": 0.5})
    #
    # Directory and file should be created
    assert_true(dir_exists("/tmp/deep/nested/path"))
    assert_true(file_exists("/tmp/deep/nested/path/model.pt"))


# ============================================================================
# Load Checkpoint Tests
# ============================================================================


fn test_checkpointing_restore_training() raises:
    """Test loading checkpoint restores complete training state.

    API Contract:
        load_checkpoint(filepath) should:
        - Restore model weights
        - Restore optimizer state
        - Return epoch number
        - Allow seamless training continuation

    This is CRITICAL for training resumption.
    """
    # TODO(#34): Implement when Checkpointing is available
    # Train and save
    var model1 = create_simple_model()
    var optimizer1 = SGD(learning_rate=0.1, momentum=0.9)
    var trainer1 = Trainer(model1, optimizer1, loss_fn)
    #
    trainer1.train(epochs=5, train_loader, val_loader)
    trainer1.save_checkpoint("/tmp/checkpoint.pt")
    #
    # Create new trainer and load
    var model2 = create_simple_model()
    var optimizer2 = SGD(learning_rate=0.1, momentum=0.9)
    var trainer2 = Trainer(model2, optimizer2, loss_fn)
    #
    var start_epoch = trainer2.load_checkpoint("/tmp/checkpoint.pt")
    #
    # Verify epoch
    assert_equal(start_epoch, 5)
    #
    # Verify model weights identical
    var test_input = Tensor.randn(1, 10)
    var output1 = model1.forward(test_input)
    var output2 = model2.forward(test_input)
    assert_tensor_equal(output1, output2)
    #
    # Continue training from checkpoint
    trainer2.train(epochs=5, train_loader, val_loader, initial_epoch=start_epoch)


# ============================================================================
# Edge Cases
# ============================================================================


fn test_checkpointing_overwrite_existing() raises:
    """Test Checkpointing overwrites existing checkpoint files.

    API Contract:
        When saving to existing filepath:
        - Overwrite file (don't append)
        - No error raised
    """
    # TODO(#34): Implement when Checkpointing is available
    var checkpoint = Checkpointing(filepath="/tmp/model.pt")
    #
    # Save twice
    checkpoint.on_epoch_end(1, {"train_loss": 0.5})
    checkpoint.on_epoch_end(2, {"train_loss": 0.3})
    #
    # File should exist and contain epoch 2 state
    var loaded_state = load_checkpoint("/tmp/model.pt")
    assert_equal(loaded_state["epoch"], 2)


fn test_checkpointing_missing_monitored_metric() raises:
    """Test Checkpointing handles missing monitored metric gracefully.

    API Contract:
        If monitored metric not in logs:
        - Raise error OR
        - Skip saving with warning
    """
    # TODO(#34): Implement error handling when Checkpointing is available
    var checkpoint = Checkpointing(
        filepath="/tmp/model.pt",
        monitor="val_loss",
        save_best_only=True
    )
    #
    # Logs missing val_loss
    try:
        checkpoint.on_epoch_end(1, {"train_loss": 0.5})
        # Should either raise error or skip gracefully
    except Error as e:
        # Expected error
        pass


# ============================================================================
# Test Main
# ============================================================================


fn main() raises:
    """Run all checkpointing callback tests."""
    print("Running checkpointing core tests...")
    test_checkpointing_initialization()
    test_checkpointing_saves_at_epoch_end()
    test_checkpointing_saves_complete_state()

    print("Running best model tracking tests...")
    test_checkpointing_save_best_only()
    test_checkpointing_monitor_different_metrics()

    print("Running save frequency tests...")
    test_checkpointing_save_frequency()

    print("Running filepath template tests...")
    test_checkpointing_filepath_template()
    test_checkpointing_creates_directory()

    print("Running load checkpoint tests...")
    test_checkpointing_restore_training()

    print("Running edge cases...")
    test_checkpointing_overwrite_existing()
    test_checkpointing_missing_monitored_metric()

    print("\nAll checkpointing callback tests passed! ✓")
