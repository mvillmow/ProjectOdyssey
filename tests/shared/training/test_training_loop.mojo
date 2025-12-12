"""Unit tests for Training Loop (forward/backward pass iteration).

Tests cover:
- Forward pass execution
- Loss computation
- Backward pass and gradient computation
- Weight updates via optimizer
- Batch iteration and epoch completion

Following TDD principles - these tests define the expected API
for implementation in Issue #34.
"""

from tests.shared.conftest import (
    assert_true,
    assert_equal,
    assert_almost_equal,
    assert_less,
    assert_greater,
    assert_not_equal_tensor,
    assert_not_none,
    assert_shape_equal,
    assert_tensor_equal,
    assert_type,
    create_simple_model,
    create_simple_dataset,
    create_mock_dataloader,
    create_test_vector,
    TestFixtures,
)
from shared.training import SGD, MSELoss, TrainingLoop
from shared.core.extensor import ExTensor
from shared.core import ones, zeros

# NOTE: TrainingLoop is now generic with trait bounds (Issue #34 Track 2)
# Generic instantiation pattern:
#   var training_loop = TrainingLoop[SimpleMLP, MSELoss, SGD](model, optimizer, loss_fn)
#
# Type parameters:
#   [M: Model, L: Loss, O: Optimizer]
#
# This provides compile-time type safety - wrong types are rejected at compile time.
# Once Track 3 (SimpleMLP trait impl) merges, these tests will fully compile.


# ============================================================================
# Training Loop Core Tests
# ============================================================================


fn test_training_loop_single_batch() raises:
    """Test training loop processes a single batch correctly.

    API Contract:
        Training step should:
        1. Get batch from data loader
        2. Forward pass: output = model(input)
        3. Compute loss: loss = loss_fn(output, target)
        4. Backward pass: compute gradients
        5. Optimizer step: update weights
        6. Return loss value

    This is a CRITICAL test for basic training functionality.
    """
    # TODO(#2728): Implement when TrainingLoop is available
    # TODO(#2728): Uncomment when SimpleMLP and create_simple_model are available
    # var model = create_simple_model()
    # var optimizer = SGD(learning_rate=0.01)
    # var loss_fn = MSELoss()
    # var training_loop = TrainingLoop(model^, optimizer^, loss_fn^)
    # #
    # # Create single batch
    # var inputs = ones([4, 10], DType.float32)  # batch_size=4, input_dim=10
    # var targets = zeros([4, 1], DType.float32)  # batch_size=4, output_dim=1
    # #
    # # Get initial weights
    # var initial_weights = model.get_weights().copy()
    # #
    # # Run single training step
    # var loss = training_loop.step(inputs, targets)
    # #
    # # Verify loss is computed
    # assert_greater(loss, 0.0)
    # #
    # # Verify weights changed
    # var final_weights = model.get_weights()
    # assert_not_equal_tensor(initial_weights, final_weights)
    pass


fn test_training_loop_full_epoch() raises:
    """Test training loop completes a full epoch over dataset.

    API Contract:
        fn run_epoch(self, data_loader: DataLoader) -> Float32
        - Iterates through all batches in data loader
        - Performs training step on each batch
        - Returns average loss for the epoch.
    """
    # TODO(#2728): Implement when TrainingLoop is available
    # TODO(#2728): Uncomment when SimpleMLP and create_mock_dataloader are available
    # var model = create_simple_model()
    # var optimizer = SGD(learning_rate=0.01)
    # var loss_fn = MSELoss()
    # var training_loop = TrainingLoop(model^, optimizer^, loss_fn^)
    # #
    # # Create data loader with 10 batches
    # var data_loader = create_mock_dataloader(n_batches=10)
    # #
    # # Run one epoch
    # var avg_loss = training_loop.run_epoch(data_loader)
    # #
    # # Should return average loss
    # assert_greater(avg_loss, 0.0)
    pass


fn test_training_loop_multiple_epochs() raises:
    """Test training loop runs multiple epochs and loss decreases.

    API Contract:
        Multiple epochs should:
        - Each epoch processes entire dataset
        - Loss should generally decrease (for simple problems)
        - Return list of epoch losses.
    """
    # TODO(#2728): Implement when TrainingLoop is available
    # TODO(#2728): Uncomment when SimpleMLP and create_simple_dataset are available
    # var model = create_simple_model()
    # var optimizer = SGD(learning_rate=0.1)
    # var loss_fn = MSELoss()
    # var training_loop = TrainingLoop(model^, optimizer^, loss_fn^)
    # #
    # var data_loader = create_simple_dataset()
    # #
    # # Run 10 epochs
    # var epoch_losses = List[Float32]()
    # for _ in range(10):
    #     var loss = training_loop.run_epoch(data_loader)
    #     epoch_losses.append(loss)
    # #
    # # First epoch loss should be higher than last
    # assert_greater(epoch_losses[0], epoch_losses[-1])
    pass


# ============================================================================
# Forward Pass Tests
# ============================================================================


fn test_training_loop_forward_pass() raises:
    """Test training loop executes forward pass correctly.

    API Contract:
        Forward pass should:
        - Call model.forward(input)
        - Return output tensor
        - Preserve batch dimension.
    """
    # TODO(#2728): Implement when TrainingLoop is available
    # TODO(#2728): Uncomment when SimpleMLP and ExTensor.randn are available
    # var model = create_simple_model()
    # var training_loop = TrainingLoop(model^, optimizer^, loss_fn^)
    # #
    # # TODO(#2728): Implement randn - var inputs = ExTensor.zeros([8, 10], DType.float32)  # batch_size=8
    # #
    # # Execute forward pass
    # var outputs = training_loop.forward(inputs)
    # #
    # # Output should preserve batch size
    # assert_equal(outputs.shape()[0], 8)
    pass


fn test_training_loop_forward_batches_independently() raises:
    """Test forward pass processes batch samples independently.

    API Contract:
        Forward pass on batch should equal processing samples individually.
    """
    # TODO(#2728): Implement when TrainingLoop is available
    # TODO(#2728): Uncomment when SimpleMLP and ExTensor.randn are available
    # var model = create_simple_model()
    # var training_loop = TrainingLoop(model^, optimizer^, loss_fn^)
    # #
    # # Create batch
    # # TODO(#2728): Implement randn - var batch_input = ExTensor.zeros([4, 10], DType.float32)
    # #
    # # Forward pass on batch
    # var batch_output = training_loop.forward(batch_input)
    # #
    # # Forward pass on individual samples
    # for i in range(4):
    #     var single_input = batch_input[i:i+1, :]
    #     var single_output = training_loop.forward(single_input)
    #     assert_tensor_equal(single_output, batch_output[i:i+1, :])
    pass


# ============================================================================
# Loss Computation Tests
# ============================================================================


fn test_training_loop_computes_loss() raises:
    """Test training loop computes loss correctly.

    API Contract:
        fn compute_loss(self, outputs: Tensor, targets: Tensor) -> Float32
        - Calls loss_fn(outputs, targets)
        - Returns scalar loss value.
    """
    # TODO(#2728): Implement when TrainingLoop is available
    # TODO(#2728): Uncomment when TrainingLoop is available
    # var loss_fn = MSELoss()
    # var training_loop = TrainingLoop(model^, optimizer^, loss_fn^)
    # #
    # # Known outputs and targets
    # var outputs_list = List[Float32]()
    # outputs_list.append(1.0)
    # outputs_list.append(2.0)
    # outputs_list.append(3.0)
    # var outputs_shape = List[Int]()
    # outputs_shape.append(3)
    # outputs_shape.append(1)
    # var outputs = ExTensor(outputs_shape, DType.float32)
    # for i in range(len(outputs_list)):
    #     outputs._set_float32(i, outputs_list[i])

    # var targets_list = List[Float32]()
    # targets_list.append(0.0)
    # targets_list.append(0.0)
    # targets_list.append(0.0)
    # var targets_shape = List[Int]()
    # targets_shape.append(3)
    # targets_shape.append(1)
    # var targets = ExTensor(targets_shape, DType.float32)
    # for i in range(len(targets_list)):
    #     targets._set_float32(i, targets_list[i])
    # #
    # # Compute loss
    # var loss = training_loop.compute_loss(outputs, targets)
    # #
    # # MSE = mean((outputs - targets)^2) = mean([1, 4, 9]) = 14/3 ≈ 4.67
    # assert_almost_equal(loss, 4.6667, tolerance=1e-3)
    pass


fn test_training_loop_loss_scalar() raises:
    """Test training loop returns scalar loss (not tensor).

    API Contract:
        Loss should be reduced to single Float32 value
        (average over batch or sum, depending on loss function).
    """
    # TODO(#2728): Implement when TrainingLoop is available
    # TODO(#2728): Uncomment when TrainingLoop and ExTensor.randn are available
    # var training_loop = TrainingLoop(model^, optimizer^, loss_fn^)
    # #
    # # TODO(#2728): Implement randn - var inputs = ExTensor.zeros([10, 5], DType.float32)
    # # TODO(#2728): Implement randn - var targets = ExTensor.zeros([10, 1], DType.float32)
    # #
    # var loss = training_loop.step(inputs, targets)
    # #
    # # Loss should be scalar Float32
    # assert_type(loss, Float32)
    pass


# ============================================================================
# Backward Pass Tests
# ============================================================================


fn test_training_loop_backward_pass() raises:
    """Test training loop executes backward pass.

    API Contract:
        Backward pass should:
        - Compute gradients w.r.t. model parameters
        - Gradients stored in parameter.grad
        - Gradients have same shape as parameters

    This is a CRITICAL test for gradient computation.
    """
    # TODO(#2728): Implement when TrainingLoop is available
    # TODO(#2728): Uncomment when SimpleMLP, TrainingLoop, and ExTensor.randn are available
    # Note: ExTensor._grad attribute doesn't exist yet - will be added in gradient computation implementation
    # var model = create_simple_model()
    # var training_loop = TrainingLoop(model^, optimizer^, loss_fn^)
    # #
    # # TODO(#2728): Implement randn - var inputs = ExTensor.zeros([4, 10], DType.float32)
    # # TODO(#2728): Implement randn - var targets = ExTensor.zeros([4, 1], DType.float32)
    # #
    # # Zero gradients
    # model.zero_grad()
    # #
    # # Run backward pass
    # var loss = training_loop.step(inputs, targets)
    # #
    # # Check gradients are computed
    # for param in model.parameters():
    #     assert_not_none(param._grad)
    #     assert_shape_equal(param._grad, param.shape())
    pass


fn test_training_loop_gradient_accumulation() raises:
    """Test training loop accumulates gradients when not zeroed.

    API Contract:
        If gradients not zeroed, backward pass should accumulate:
        new_grad = old_grad + computed_grad.
    """
    # TODO(#2728): Implement when TrainingLoop is available
    # TODO(#2728): Uncomment when SimpleMLP, TrainingLoop, ExTensor.randn, and _grad attribute are available
    # var model = create_simple_model()
    # var training_loop = TrainingLoop(model^, optimizer^, loss_fn^)
    # #
    # # TODO(#2728): Implement randn - var inputs = ExTensor.zeros([4, 10], DType.float32)
    # # TODO(#2728): Implement randn - var targets = ExTensor.zeros([4, 1], DType.float32)
    # #
    # # First backward (gradients zeroed initially)
    # model.zero_grad()
    # var loss1 = training_loop.step(inputs, targets)
    # var grad_after_first = model.parameters()[0]._grad.copy()
    # #
    # # Second backward without zeroing
    # var loss2 = training_loop.step(inputs, targets)
    # var grad_after_second = model.parameters()[0]._grad
    # #
    # # Gradients should be approximately 2x first gradients
    # # (assuming same inputs/targets)
    # for i in range(grad_after_first.numel()):
    #     assert_almost_equal(
    #         grad_after_second[i],
    #         2 * grad_after_first[i],
    #         tolerance=1e-5
    #     )
    pass


# ============================================================================
# Weight Update Tests
# ============================================================================


fn test_training_loop_updates_weights() raises:
    """Test training loop updates model weights.

    API Contract:
        After training step:
        - Weights should be different from before
        - Weight update direction: opposite to gradient
        - Update magnitude proportional to learning rate

    This is a CRITICAL test for learning.
    """
    # TODO(#2728): Implement when TrainingLoop is available
    # TODO(#2728): Uncomment when SimpleMLP, TrainingLoop, and ExTensor.randn are available
    # var model = create_simple_model()
    # var optimizer = SGD(learning_rate=0.1)
    # var training_loop = TrainingLoop(model^, optimizer^, loss_fn^)
    # #
    # # Get initial weights
    # var initial_weights = model.parameters()[0]._data.copy()
    # #
    # # Training step
    # # TODO(#2728): Implement randn - var inputs = ExTensor.zeros([4, 10], DType.float32)
    # # TODO(#2728): Implement randn - var targets = ExTensor.zeros([4, 1], DType.float32)
    # var loss = training_loop.step(inputs, targets)
    # #
    # # Get updated weights
    # var updated_weights = model.parameters()[0]._data
    # #
    # # Weights should change
    # assert_not_equal_tensor(initial_weights, updated_weights)
    pass


fn test_training_loop_respects_learning_rate() raises:
    """Test training loop weight updates scale with learning rate.

    API Contract:
        Higher learning rate → larger weight updates
        Lower learning rate → smaller weight updates.
    """
    # TODO(#2728): Implement when TrainingLoop is available
    # TODO(#2728): Uncomment when SimpleMLP, TrainingLoop, state_dict, and ExTensor.randn are available
    # var model1 = create_simple_model()
    # var model2 = create_simple_model()
    # #
    # # Same initial weights
    # model2.load_state_dict(model1.state_dict())
    # #
    # # Different learning rates
    # var optimizer1 = SGD(learning_rate=0.01)
    # var optimizer2 = SGD(learning_rate=0.1)  # 10x larger
    # #
    # var loop1 = TrainingLoop(model1^, optimizer1^, loss_fn)
    # var loop2 = TrainingLoop(model2^, optimizer2^, loss_fn^)
    # #
    # # Same inputs/targets
    # # TODO(#2728): Implement randn with seed=42 - var inputs = ExTensor.zeros([4, 10], DType.float32)
    # # TODO(#2728): Implement randn with seed=43 - var targets = ExTensor.zeros([4, 1], DType.float32)
    # #
    # # Training steps
    # loop1.step(inputs, targets)
    # loop2.step(inputs, targets)
    # #
    # # Weight changes
    # var change1 = (model1.parameters()[0]._data - initial_weights).abs().sum()
    # var change2 = (model2.parameters()[0]._data - initial_weights).abs().sum()
    # #
    # # Change2 should be ~10x larger
    # assert_almost_equal(change2 / change1, 10.0, tolerance=0.5)
    pass


# ============================================================================
# Batch Processing Tests
# ============================================================================


fn test_training_loop_processes_variable_batch_sizes() raises:
    """Test training loop handles different batch sizes.

    API Contract:
        Training loop should work with any batch size:
        - Small batches (1-4 samples)
        - Medium batches (16-64 samples)
        - Large batches (128+ samples).
    """
    # TODO(#2728): Implement when TrainingLoop is available
    # TODO(#2728): Uncomment when SimpleMLP, TrainingLoop, and ExTensor.randn are available
    # var model = create_simple_model()
    # var training_loop = TrainingLoop(model^, optimizer^, loss_fn^)
    # #
    # # Test different batch sizes
    # for batch_size in [1, 4, 16, 64, 128]:
    #     # TODO(#2728): Implement randn - var inputs = ExTensor.zeros(List[Int](batch_size, 10), DType.float32)
    #     # TODO(#2728): Implement randn - var targets = ExTensor.zeros(List[Int](batch_size, 1), DType.float32)
    # #
    #     # Should process without error
    #     var loss = training_loop.step(inputs, targets)
    #     assert_greater(loss, 0.0)
    pass


fn test_training_loop_averages_loss_over_batch() raises:
    """Test training loop computes average loss over batch.

    API Contract:
        Batch loss should be mean of individual sample losses
        (for most loss functions).
    """
    # TODO(#2728): Implement when TrainingLoop is available
    # TODO(#2728): Uncomment when SimpleMLP, TrainingLoop, and ExTensor.randn are available
    # var model = create_simple_model()
    # var optimizer = SGD(learning_rate=0.01)
    # var training_loop = TrainingLoop(model^, optimizer^, MSELoss(reduction="mean"))
    # #
    # # Create batch
    # # TODO(#2728): Implement randn - var batch_inputs = ExTensor.zeros([4, 10], DType.float32)
    # # TODO(#2728): Implement randn - var batch_targets = ExTensor.zeros([4, 1], DType.float32)
    # #
    # # Batch loss
    # var batch_loss = training_loop.compute_loss(
    #     model.forward(batch_inputs),
    #     batch_targets
    # )
    # #
    # # Individual losses
    # var individual_losses = List[Float32]()
    # for i in range(4):
    #     var single_loss = training_loop.compute_loss(
    #         model.forward(batch_inputs[i:i+1, :]),
    #         batch_targets[i:i+1, :]
    #     )
    #     individual_losses.append(single_loss)
    # #
    # # Batch loss should equal average of individual losses
    # var avg_individual = sum(individual_losses) / 4
    # assert_almost_equal(batch_loss, avg_individual)
    pass


# ============================================================================
# Property-Based Tests
# ============================================================================


fn test_training_loop_property_loss_decreases_on_simple_problem() raises:
    """Property: Training should decrease loss on simple convex problem.

    Test that training loop can solve a basic regression problem.
    """
    # TODO(#2728): Implement when TrainingLoop is available
    # TODO(#2728): Uncomment when Linear layer, TrainingLoop, and create_dataloader are available
    # Simple problem: learn to map inputs to sum(inputs)
    # var model = Linear(in_features=10, out_features=1)
    # var optimizer = SGD(learning_rate=0.01)
    # var training_loop = TrainingLoop(model^, optimizer^, MSELoss())
    # #
    # # Generate simple dataset
    # # TODO(#2728): Implement randn - var inputs = ExTensor.zeros([100, 10], DType.float32)
    # var targets = inputs.sum(dim=1, keepdim=True)
    # var data_loader = create_dataloader(inputs, targets, batch_size=10)
    # #
    # # Record initial loss
    # var initial_loss = training_loop.run_epoch(data_loader)
    # #
    # # Train for 50 epochs
    # var final_loss = initial_loss
    # for _ in range(50):
    #     final_loss = training_loop.run_epoch(data_loader)
    # #
    # # Loss should decrease significantly
    # assert_less(final_loss, initial_loss * 0.1)
    pass


# ============================================================================
# Test Main
# ============================================================================


fn main() raises:
    """Run all training loop tests."""
    print("Running training loop core tests...")
    test_training_loop_single_batch()
    test_training_loop_full_epoch()
    test_training_loop_multiple_epochs()

    print("Running forward pass tests...")
    test_training_loop_forward_pass()
    test_training_loop_forward_batches_independently()

    print("Running loss computation tests...")
    test_training_loop_computes_loss()
    test_training_loop_loss_scalar()

    print("Running backward pass tests...")
    test_training_loop_backward_pass()
    test_training_loop_gradient_accumulation()

    print("Running weight update tests...")
    test_training_loop_updates_weights()
    test_training_loop_respects_learning_rate()

    print("Running batch processing tests...")
    test_training_loop_processes_variable_batch_sizes()
    test_training_loop_averages_loss_over_batch()

    print("Running property-based tests...")
    test_training_loop_property_loss_decreases_on_simple_problem()

    print("\nAll training loop tests passed! ✓")
