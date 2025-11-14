"""Unit tests for Validation Loop (evaluation without weight updates).

Tests cover:
- Forward-only pass (no gradients)
- Loss computation without backpropagation
- Metrics tracking (accuracy, etc.)
- No weight updates during validation

Following TDD principles - these tests define the expected API
for implementation in Issue #34.
"""

from tests.shared.conftest import (
    assert_true,
    assert_equal,
    assert_almost_equal,
    assert_greater,
    TestFixtures,
)


# ============================================================================
# Validation Loop Core Tests
# ============================================================================


fn test_validation_loop_single_batch() raises:
    """Test validation loop processes a single batch without gradients.

    API Contract:
        Validation step should:
        1. Get batch from data loader
        2. Forward pass: output = model(input)
        3. Compute loss: loss = loss_fn(output, target)
        4. NO backward pass
        5. NO optimizer step
        6. Return loss value

    This is a CRITICAL test - validation must not update weights.
    """
    # TODO(#34): Implement when ValidationLoop is available
    var model = create_simple_model()
    var loss_fn = MSELoss()
    var validation_loop = ValidationLoop(model, loss_fn)
    #
    # Create single batch
    var inputs = Tensor.ones(4, 10)
    var targets = Tensor.zeros(4, 1)
    #
    # Get initial weights
    var initial_weights = model.get_weights().copy()
    #
    # Run validation step
    var loss = validation_loop.step(inputs, targets)
    #
    # Verify loss is computed
    assert_greater(loss, 0.0)
    #
    # Verify weights UNCHANGED
    var final_weights = model.get_weights()
    assert_tensor_equal(initial_weights, final_weights)


fn test_validation_loop_full_epoch() raises:
    """Test validation loop completes a full epoch over dataset.

    API Contract:
        fn run_epoch(self, data_loader: DataLoader) -> Dict
        - Iterates through all batches in data loader
        - Computes validation metrics for each batch
        - Returns average loss and other metrics
    """
    # TODO(#34): Implement when ValidationLoop is available
    var model = create_simple_model()
    var loss_fn = MSELoss()
    var validation_loop = ValidationLoop(model, loss_fn)
    #
    # Create data loader with 10 batches
    var data_loader = create_mock_dataloader(n_batches=10)
    #
    # Run one validation epoch
    var results = validation_loop.run_epoch(data_loader)
    #
    # Should return average loss
    assert_true(results.contains("loss"))
    assert_greater(results["loss"], 0.0)


fn test_validation_loop_no_weight_updates() raises:
    """Test validation loop never modifies model weights.

    API Contract:
        After validation (single step or full epoch):
        - All model parameters unchanged
        - No optimizer state modified

    This is a CRITICAL test for validation correctness.
    """
    # TODO(#34): Implement when ValidationLoop is available
    var model = create_simple_model()
    var validation_loop = ValidationLoop(model, loss_fn)
    var data_loader = create_mock_dataloader(n_batches=100)
    #
    # Get initial weights
    var initial_weights = [param.data.copy() for param in model.parameters()]
    #
    # Run multiple validation epochs
    for _ in range(10):
        validation_loop.run_epoch(data_loader)
    #
    # Verify all weights unchanged
    for i, param in enumerate(model.parameters()):
        assert_tensor_equal(param.data, initial_weights[i])


# ============================================================================
# No Gradient Computation Tests
# ============================================================================


fn test_validation_loop_no_gradient_computation() raises:
    """Test validation loop does not compute gradients.

    API Contract:
        During validation:
        - Gradients not computed
        - param.grad remains None or unchanged
        - Memory efficient (no gradient storage)

    This is CRITICAL for memory efficiency during validation.
    """
    # TODO(#34): Implement when ValidationLoop is available
    var model = create_simple_model()
    var validation_loop = ValidationLoop(model, loss_fn)
    #
    var inputs = Tensor.randn(4, 10)
    var targets = Tensor.randn(4, 1)
    #
    # Ensure gradients initially None or zero
    model.zero_grad()
    #
    # Run validation step
    var loss = validation_loop.step(inputs, targets)
    #
    # Gradients should still be None or zero
    for param in model.parameters():
        if param.grad is not None:
            assert_tensor_equal(param.grad, Tensor.zeros_like(param.data))


fn test_validation_loop_forward_only_mode() raises:
    """Test validation loop puts model in evaluation mode.

    API Contract (if model has train/eval modes):
        validation_loop should:
        - Set model to eval mode (model.eval())
        - Disable dropout, batch norm updates, etc.
        - Restore original mode after validation
    """
    # TODO(#34): Implement if model has eval mode
    var model = create_model_with_dropout()
    var validation_loop = ValidationLoop(model, loss_fn)
    #
    # Model starts in training mode
    model.train()
    assert_true(model.training)
    #
    # Run validation
    validation_loop.run_epoch(data_loader)
    #
    # Model should be back in training mode (or explicitly in eval mode)
    # Depends on design choice


# ============================================================================
# Metrics Computation Tests
# ============================================================================


fn test_validation_loop_computes_loss() raises:
    """Test validation loop computes loss correctly.

    API Contract:
        Validation loss should be:
        - Average loss over all batches
        - Computed with same loss function as training
    """
    # TODO(#34): Implement when ValidationLoop is available
    var model = create_simple_model()
    var loss_fn = MSELoss()
    var validation_loop = ValidationLoop(model, loss_fn)
    #
    # Known outputs and targets
    var outputs = Tensor(List[Float32](1.0, 2.0, 3.0), Shape(3, 1))
    var targets = Tensor(List[Float32](0.0, 0.0, 0.0), Shape(3, 1))
    #
    # Compute loss
    var loss = validation_loop.compute_loss(outputs, targets)
    #
    # MSE = mean((outputs - targets)^2) = mean([1, 4, 9]) = 14/3 ≈ 4.67
    assert_almost_equal(loss, 4.6667, tolerance=1e-3)


fn test_validation_loop_computes_accuracy() raises:
    """Test validation loop computes classification accuracy (if applicable).

    API Contract:
        For classification tasks:
        fn run_epoch(self, data_loader) -> Dict
        - Returns dict with "loss" and "accuracy"
        - Accuracy = correct_predictions / total_predictions
    """
    # TODO(#34): Implement when ValidationLoop supports metrics
    var model = create_classifier()
    var validation_loop = ValidationLoop(model, CrossEntropyLoss(), metrics=["accuracy"])
    #
    # Create classification dataset
    var data_loader = create_classification_data()
    #
    # Run validation
    var results = validation_loop.run_epoch(data_loader)
    #
    # Should return accuracy
    assert_true(results.contains("accuracy"))
    assert_greater(results["accuracy"], 0.0)
    assert_less(results["accuracy"], 1.0)


fn test_validation_loop_custom_metrics() raises:
    """Test validation loop supports custom metric functions.

    API Contract (optional):
        ValidationLoop(model, loss_fn, metrics=[custom_metric_fn])
        - Allows adding custom metrics (F1, precision, recall, etc.)
        - Returns all metrics in results dict
    """
    # TODO(#34): Implement if custom metrics are supported
    This is a nice-to-have feature


# ============================================================================
# Batch Processing Tests
# ============================================================================


fn test_validation_loop_processes_variable_batch_sizes() raises:
    """Test validation loop handles different batch sizes.

    API Contract:
        Validation should work with any batch size, including:
        - Variable batch sizes within same dataset
        - Last batch smaller than others (common case)
    """
    # TODO(#34): Implement when ValidationLoop is available
    var model = create_simple_model()
    var validation_loop = ValidationLoop(model, loss_fn)
    #
    # Test different batch sizes
    for batch_size in [1, 4, 16, 64, 128]:
        var inputs = Tensor.randn(batch_size, 10)
        var targets = Tensor.randn(batch_size, 1)
    #
        # Should process without error
        var loss = validation_loop.step(inputs, targets)
        assert_greater(loss, 0.0)


fn test_validation_loop_handles_incomplete_batch() raises:
    """Test validation loop handles last incomplete batch correctly.

    API Contract:
        When dataset size not divisible by batch size:
        - Last batch has fewer samples
        - Loss computed correctly (average over actual batch size)
    """
    # TODO(#34): Implement when ValidationLoop is available
    var model = create_simple_model()
    var validation_loop = ValidationLoop(model, loss_fn)
    #
    # Dataset with 105 samples, batch size 32
    # Last batch has 9 samples
    var data_loader = create_dataloader(n_samples=105, batch_size=32)
    #
    # Run validation
    var results = validation_loop.run_epoch(data_loader)
    #
    # Should handle incomplete batch gracefully
    assert_greater(results["loss"], 0.0)


# ============================================================================
# Determinism Tests
# ============================================================================


fn test_validation_loop_deterministic_same_data() raises:
    """Test validation produces identical results for same data.

    API Contract:
        Multiple validation runs on same data should produce
        identical results (assuming deterministic model).
    """
    # TODO(#34): Implement when ValidationLoop is available
    var model = create_simple_model()
    var validation_loop = ValidationLoop(model, loss_fn)
    var data_loader = create_mock_dataloader(seed=42)
    #
    # Run validation twice
    var results1 = validation_loop.run_epoch(data_loader)
    var results2 = validation_loop.run_epoch(data_loader)
    #
    # Results should be identical
    assert_almost_equal(results1["loss"], results2["loss"])


fn test_validation_loop_independent_of_training() raises:
    """Test validation results independent of training state.

    API Contract:
        Validation should produce same results regardless of:
        - Training mode vs eval mode (if modes exist)
        - Optimizer state
        - Gradient computation state
    """
    # TODO(#34): Implement when ValidationLoop is available
    var model = create_simple_model()
    var validation_loop = ValidationLoop(model, loss_fn)
    var data_loader = create_mock_dataloader()
    #
    # Validate before any training
    var results_before = validation_loop.run_epoch(data_loader)
    #
    # Train for a few steps (but don't change weights for this test)
    # Just to potentially modify internal state
    var optimizer = SGD(learning_rate=0.0)  # LR=0 means no weight change
    var training_loop = TrainingLoop(model, optimizer, loss_fn)
    training_loop.run_epoch(create_mock_dataloader())
    #
    # Validate after training
    var results_after = validation_loop.run_epoch(data_loader)
    #
    # Results should be identical (weights didn't change)
    assert_almost_equal(results_before["loss"], results_after["loss"])


# ============================================================================
# Efficiency Tests
# ============================================================================


fn test_validation_loop_memory_efficient() raises:
    """Test validation loop is memory efficient (no gradient storage).

    This is a performance test - validation should use less memory
    than training since gradients not stored.
    """
    # TODO(#34): Implement memory measurement when available
    This tests memory usage, not just correctness


fn test_validation_loop_faster_than_training() raises:
    """Test validation is faster than training (no backward pass).

    Validation should be roughly 2-3x faster than training step
    since it skips backward pass and optimizer step.
    """
    # TODO(#34): Implement timing comparison when available
    This is a performance property test


# ============================================================================
# Property-Based Tests
# ============================================================================


fn test_validation_loop_property_loss_matches_training() raises:
    """Property: Validation loss formula should match training loss.

    When computed on same batch, validation and training should
    give same loss value (before backward pass).
    """
    # TODO(#34): Implement when both loops available
    var model = create_simple_model()
    var loss_fn = MSELoss()
    var training_loop = TrainingLoop(model, optimizer, loss_fn)
    var validation_loop = ValidationLoop(model, loss_fn)
    #
    var inputs = Tensor.randn(4, 10, seed=42)
    var targets = Tensor.randn(4, 1, seed=43)
    #
    # Compute loss via validation
    var val_loss = validation_loop.step(inputs, targets)
    #
    # Compute loss via training (but check before weight update)
    var outputs = model.forward(inputs)
    var train_loss = loss_fn(outputs, targets)
    #
    # Should be identical
    assert_almost_equal(val_loss, train_loss)


# ============================================================================
# Test Main
# ============================================================================


fn main() raises:
    """Run all validation loop tests."""
    print("Running validation loop core tests...")
    test_validation_loop_single_batch()
    test_validation_loop_full_epoch()
    test_validation_loop_no_weight_updates()

    print("Running no gradient computation tests...")
    test_validation_loop_no_gradient_computation()
    test_validation_loop_forward_only_mode()

    print("Running metrics computation tests...")
    test_validation_loop_computes_loss()
    test_validation_loop_computes_accuracy()
    test_validation_loop_custom_metrics()

    print("Running batch processing tests...")
    test_validation_loop_processes_variable_batch_sizes()
    test_validation_loop_handles_incomplete_batch()

    print("Running determinism tests...")
    test_validation_loop_deterministic_same_data()
    test_validation_loop_independent_of_training()

    print("Running efficiency tests...")
    test_validation_loop_memory_efficient()
    test_validation_loop_faster_than_training()

    print("Running property-based tests...")
    test_validation_loop_property_loss_matches_training()

    print("\nAll validation loop tests passed! ✓")
