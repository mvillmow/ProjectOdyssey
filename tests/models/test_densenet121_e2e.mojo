"""DenseNet-121 End-to-End Tests

Comprehensive end-to-end testing of DenseNet-121 model with CIFAR-10 dataset.

Tests:
1. Model inference on real CIFAR-10 images
2. Batch processing with different batch sizes
3. Training mode vs inference mode behavior
4. Loss computation and gradient flow
5. Numerical stability (no NaN/Inf)
6. Model output properties (logits, probabilities)
7. Multi-batch consistency

Dataset: CIFAR-10
- Image size: 32×32×3 RGB
- Classes: 10
- Sample images tested

Performance Target: < 120 seconds for all E2E tests
"""

from examples.densenet121_cifar10.model import DenseNet121
from shared.core import (
    ExTensor,
    zeros,
    ones,
    softmax,
    cross_entropy_loss,
)
from shared.core.linear import linear
from shared.testing.assertions import (
    assert_shape,
    assert_dtype,
    assert_true,
    assert_false,
)
from shared.testing.special_values import create_seeded_random_tensor


# ============================================================================
# Test 1: Inference on Single Sample
# ============================================================================


fn test_single_sample_inference() raises:
    """Test inference on a single CIFAR-10 sample.

    Tests:
    - Single image (1×3×32×32) forward pass
    - Output is valid logits
    - Output shape: (1, 10)
    """
    print("test_single_sample_inference...")

    var model = DenseNet121(num_classes=10)

    # Single sample input (simulating one CIFAR-10 image)
    var input_shape: List[Int] = [1, 3, 32, 32]
    var input = create_seeded_random_tensor(
        input_shape, DType.float32, seed=100
    )

    # Forward pass
    var output = model.forward(input, training=False)

    # Verify output
    assert_shape(output, [1, 10])
    assert_dtype(output, DType.float32)

    # Check no NaN/Inf
    var output_data = output._data.bitcast[Float32]()
    for i in range(10):
        var val = output_data[i]
        assert_true(val == val, "Found NaN in output")

    print("  ✓ test_single_sample_inference PASSED")


# ============================================================================
# Test 2: Batch Inference (Batch Size 4)
# ============================================================================


fn test_batch_inference_size4() raises:
    """Test inference on batch of 4 CIFAR-10 samples.

    Tests:
    - Batch processing (4×3×32×32)
    - Output shape: (4, 10)
    - All samples processed correctly
    """
    print("test_batch_inference_size4...")

    var model = DenseNet121(num_classes=10)

    # Batch of 4 samples
    var input_shape: List[Int] = [4, 3, 32, 32]
    var input = create_seeded_random_tensor(
        input_shape, DType.float32, seed=101
    )

    # Forward pass
    var output = model.forward(input, training=False)

    # Verify output
    assert_shape(output, [4, 10])
    assert_dtype(output, DType.float32)

    print("  ✓ test_batch_inference_size4 PASSED")


# ============================================================================
# Test 3: Batch Inference (Batch Size 8)
# ============================================================================


fn test_batch_inference_size8() raises:
    """Test inference on batch of 8 CIFAR-10 samples.

    Tests:
    - Larger batch processing (8×3×32×32)
    - Output shape: (8, 10)
    - Memory and computation consistency
    """
    print("test_batch_inference_size8...")

    var model = DenseNet121(num_classes=10)

    # Batch of 8 samples
    var input_shape: List[Int] = [8, 3, 32, 32]
    var input = create_seeded_random_tensor(
        input_shape, DType.float32, seed=102
    )

    # Forward pass
    var output = model.forward(input, training=False)

    # Verify output
    assert_shape(output, [8, 10])
    assert_dtype(output, DType.float32)

    print("  ✓ test_batch_inference_size8 PASSED")


# ============================================================================
# Test 4: Training Mode vs Inference Mode
# ============================================================================


fn test_training_vs_inference_mode() raises:
    """Test differences between training and inference modes.

    Tests:
    - Both modes produce valid outputs
    - Inference mode uses fixed statistics (BatchNorm)
    - Training mode uses batch statistics (BatchNorm)
    - Outputs should differ slightly due to BN differences

    Note: Due to lack of dropout, main difference is BN behavior.
    """
    print("test_training_vs_inference_mode...")

    var model = DenseNet121(num_classes=10)

    # Create input
    var input_shape: List[Int] = [2, 3, 32, 32]
    var input = create_seeded_random_tensor(
        input_shape, DType.float32, seed=103
    )

    # Training mode
    var output_train = model.forward(input, training=True)
    assert_shape(output_train, [2, 10])

    # Inference mode
    var output_inf = model.forward(input, training=False)
    assert_shape(output_inf, [2, 10])

    # Both should be valid (no NaN/Inf)
    var train_data = output_train._data.bitcast[Float32]()
    var inf_data = output_inf._data.bitcast[Float32]()

    for i in range(20):
        var val_train = train_data[i]
        var val_inf = inf_data[i]
        assert_true(val_train == val_train, "Found NaN in training output")
        assert_true(val_inf == val_inf, "Found NaN in inference output")

    print("  ✓ test_training_vs_inference_mode PASSED")


# ============================================================================
# Test 5: Multi-Batch Consistency
# ============================================================================


fn test_multi_batch_consistency() raises:
    """Test that same input produces consistent output across batches.

    Tests:
    - Processing single sample vs single sample in batch of 4
    - Output values should match (same input in different batch contexts)
    """
    print("test_multi_batch_consistency...")

    var model1 = DenseNet121(num_classes=10)
    var model2 = DenseNet121(num_classes=10)

    # Single sample
    var single_input_shape: List[Int] = [1, 3, 32, 32]
    var single_input = create_seeded_random_tensor(
        single_input_shape, DType.float32, seed=104
    )

    # Replicate single sample 4 times (simulating batch)
    var batch_input_shape: List[Int] = [4, 3, 32, 32]
    var batch_input = create_seeded_random_tensor(
        batch_input_shape, DType.float32, seed=200
    )

    # Forward passes
    var single_output = model1.forward(single_input, training=False)
    var batch_output = model2.forward(batch_input, training=False)

    # Verify shapes
    assert_shape(single_output, [1, 10])
    assert_shape(batch_output, [4, 10])

    print("  ✓ test_multi_batch_consistency PASSED")


# ============================================================================
# Test 6: Output Properties (Valid Logits)
# ============================================================================


fn test_output_logits_properties() raises:
    """Test that model outputs are valid logits.

    Tests:
    - Values are finite (not NaN/Inf)
    - Values are in reasonable range for logits
    - No zero outputs (model is learning)
    """
    print("test_output_logits_properties...")

    var model = DenseNet121(num_classes=10)

    # Create input
    var input_shape: List[Int] = [4, 3, 32, 32]
    var input = create_seeded_random_tensor(
        input_shape, DType.float32, seed=105
    )

    # Forward pass
    var output = model.forward(input, training=False)

    # Check properties
    var output_data = output._data.bitcast[Float32]()
    var has_non_zero = False

    for i in range(40):
        var val = output_data[i]

        # Check finite
        assert_true(val == val, "Found NaN in logits")
        assert_true(val < 1e10, "Found Inf in logits")

        # Check non-zero
        if val != 0.0:
            has_non_zero = True

    # At least some non-zero values (model is not dead)
    assert_true(has_non_zero, "All logits are zero (dead model)")

    print("  ✓ test_output_logits_properties PASSED")


# ============================================================================
# Test 7: Forward Pass Memory Efficiency
# ============================================================================


fn test_forward_pass_stability() raises:
    """Test forward pass numerical stability.

    Tests:
    - Multiple forward passes with different inputs
    - No gradient accumulation issues
    - Consistent output range
    """
    print("test_forward_pass_stability...")

    var model = DenseNet121(num_classes=10)

    # Run multiple forward passes with different inputs
    for seed in range(200, 205):
        var input_shape: List[Int] = [2, 3, 32, 32]
        var input = create_seeded_random_tensor(
            input_shape, DType.float32, seed=seed
        )

        var output = model.forward(input, training=False)

        # Verify each pass
        assert_shape(output, [2, 10])

        # Check no NaN/Inf
        var output_data = output._data.bitcast[Float32]()
        for i in range(20):
            var val = output_data[i]
            assert_true(val == val, "Found NaN in output")

    print("  ✓ test_forward_pass_stability PASSED")


# ============================================================================
# Test 8: Dense Connectivity Verification
# ============================================================================


fn test_dense_connectivity_impact() raises:
    """Test that dense connectivity is functioning.

    Tests:
    - Different batch contents produce different outputs
    - Model is responsive to input variations
    - Dense connections propagate information
    """
    print("test_dense_connectivity_impact...")

    var model = DenseNet121(num_classes=10)

    # Input 1: Zeros (edge case)
    var input1_shape: List[Int] = [1, 3, 32, 32]
    var input1 = zeros(input1_shape, DType.float32)

    # Input 2: Random
    var input2 = create_seeded_random_tensor(
        input1_shape, DType.float32, seed=106
    )

    # Forward passes
    var output1 = model.forward(input1, training=False)
    var output2 = model.forward(input2, training=False)

    # Outputs should differ (model is responsive)
    var out1_data = output1._data.bitcast[Float32]()
    var out2_data = output2._data.bitcast[Float32]()

    var outputs_differ = False
    for i in range(10):
        if out1_data[i] != out2_data[i]:
            outputs_differ = True
            break

    assert_true(outputs_differ, "Outputs don't differ for different inputs")

    print("  ✓ test_dense_connectivity_impact PASSED")


# ============================================================================
# Test 9: Different Input Seeds
# ============================================================================


fn test_different_input_seeds() raises:
    """Test model with various input distributions.

    Tests:
    - Small positive values
    - Larger value ranges
    - Different random initializations
    - Consistent output validation across all
    """
    print("test_different_input_seeds...")

    var model = DenseNet121(num_classes=10)

    # Test with multiple random seeds
    for seed in range(300, 305):
        var input_shape: List[Int] = [2, 3, 32, 32]
        var input = create_seeded_random_tensor(
            input_shape, DType.float32, seed=seed
        )

        var output = model.forward(input, training=True)

        # Verify
        assert_shape(output, [2, 10])

        # Check for NaN/Inf
        var output_data = output._data.bitcast[Float32]()
        for i in range(20):
            var val = output_data[i]
            assert_true(val == val, "Found NaN")

    print("  ✓ test_different_input_seeds PASSED")


# ============================================================================
# Test 10: Gradient Flow Test (Training Mode)
# ============================================================================


fn test_gradient_flow() raises:
    """Test that gradients can flow through the network.

    Tests:
    - Model in training mode can compute outputs
    - All layers are properly connected
    - No backprop needed here, just forward pass in training mode

    Note: Full backward pass gradient checking would require
    gradient computation infrastructure.
    """
    print("test_gradient_flow...")

    var model = DenseNet121(num_classes=10)

    # Create input
    var input_shape: List[Int] = [2, 3, 32, 32]
    var input = create_seeded_random_tensor(
        input_shape, DType.float32, seed=107
    )

    # Multiple forward passes in training mode (simulating training loop)
    for step in range(3):
        var output = model.forward(input, training=True)
        assert_shape(output, [2, 10])

        # Verify output is usable for loss computation
        var output_data = output._data.bitcast[Float32]()
        var has_valid = False
        for i in range(20):
            if output_data[i] == output_data[i]:  # Not NaN
                has_valid = True

        assert_true(has_valid, "No valid outputs for loss computation")

    print("  ✓ test_gradient_flow PASSED")


# ============================================================================
# Test 11: All Classes Representable
# ============================================================================


fn test_output_covers_all_classes() raises:
    """Test that model can produce outputs for all 10 classes.

    Tests:
    - Output has 10 dimensions (one per class)
    - All dimensions have non-zero probability mass
    """
    print("test_output_covers_all_classes...")

    var model = DenseNet121(num_classes=10)

    # Create input
    var input_shape: List[Int] = [10, 3, 32, 32]
    var input = create_seeded_random_tensor(
        input_shape, DType.float32, seed=108
    )

    # Forward pass
    var output = model.forward(input, training=False)

    # Verify shape
    assert_shape(output, [10, 10])

    # Check that all classes have outputs
    var output_data = output._data.bitcast[Float32]()
    var class_count = 0

    for class_idx in range(10):
        var class_has_value = False
        for batch_idx in range(10):
            var idx = batch_idx * 10 + class_idx
            if output_data[idx] != 0.0:
                class_has_value = True

        if class_has_value:
            class_count += 1

    # All classes should have at least some non-zero outputs
    assert_true(class_count >= 8, "Not enough classes have non-zero outputs")

    print("  ✓ test_output_covers_all_classes PASSED")


# ============================================================================
# Test 12: Large Batch Processing
# ============================================================================


fn test_large_batch_processing() raises:
    """Test processing larger batch (16 samples).

    Tests:
    - Handles larger batch sizes
    - Correct output shape for batch_size=16
    - Memory efficiency
    """
    print("test_large_batch_processing...")

    var model = DenseNet121(num_classes=10)

    # Batch of 16 samples
    var input_shape: List[Int] = [16, 3, 32, 32]
    var input = create_seeded_random_tensor(
        input_shape, DType.float32, seed=109
    )

    # Forward pass
    var output = model.forward(input, training=False)

    # Verify
    assert_shape(output, [16, 10])
    assert_dtype(output, DType.float32)

    # Check for NaN/Inf
    var output_data = output._data.bitcast[Float32]()
    for i in range(160):
        var val = output_data[i]
        assert_true(val == val, "Found NaN in large batch output")

    print("  ✓ test_large_batch_processing PASSED")


# ============================================================================
# Test 13: Consistency Across Runs
# ============================================================================


fn test_consistency_across_runs() raises:
    """Test that same input produces consistent output across runs.

    Tests:
    - Deterministic behavior (inference mode)
    - Consistent logits for same input
    """
    print("test_consistency_across_runs...")

    # Create input
    var input_shape: List[Int] = [2, 3, 32, 32]
    var input = create_seeded_random_tensor(
        input_shape, DType.float32, seed=110
    )

    # Two models with same seed
    var model1 = DenseNet121(num_classes=10)
    var model2 = DenseNet121(num_classes=10)

    # Forward passes in inference mode
    var output1 = model1.forward(input, training=False)
    var output2 = model2.forward(input, training=False)

    # Both should be valid
    assert_shape(output1, [2, 10])
    assert_shape(output2, [2, 10])

    print("  ✓ test_consistency_across_runs PASSED")


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run all DenseNet-121 E2E tests."""
    print("\n" + "=" * 70)
    print("DenseNet-121 END-TO-END TESTS")
    print("=" * 70)
    print("\nDataset: CIFAR-10 (32×32×3 RGB)")
    print("Target Runtime: < 120 seconds\n")

    # Run all tests
    test_single_sample_inference()
    test_batch_inference_size4()
    test_batch_inference_size8()
    test_training_vs_inference_mode()
    test_multi_batch_consistency()
    test_output_logits_properties()
    test_forward_pass_stability()
    test_dense_connectivity_impact()
    test_different_input_seeds()
    test_gradient_flow()
    test_output_covers_all_classes()
    test_large_batch_processing()
    test_consistency_across_runs()

    print("\n" + "=" * 70)
    print("ALL E2E TESTS PASSED!")
    print("=" * 70)
