"""Tests for batch normalization and layer normalization.

Tests cover:
- Batch normalization (training and inference modes)
- Layer normalization (2D and 4D inputs)
- Running statistics updates
- Numerical correctness
- Shape validation
- Backward passes with gradient checking (COMPREHENSIVE)

All tests use pure functional API.
"""

from tests.shared.conftest import (
    assert_true,
    assert_equal,
    assert_almost_equal,
    assert_shape_equal,
    TestFixtures,
)
from tests.helpers.gradient_checking import (
    compute_numerical_gradient,
    assert_gradients_close,
)
from shared.core.extensor import ExTensor, zeros, ones, zeros_like, ones_like
from shared.core.normalization import batch_norm2d, batch_norm2d_backward, layer_norm
from shared.core.arithmetic import add, subtract, multiply
from shared.core.reduction import sum as reduce_sum


# ============================================================================
# Batch Normalization Tests
# ============================================================================


fn test_batch_norm2d_shapes() raises:
    """Test that batch_norm2d returns correct output shape."""
    var shape = List[Int]()
    shape[0] = 2  # batch
    shape[1] = 3  # channels
    shape[2] = 4  # height
    shape[3] = 4  # width
    var x = ones(shape, DType.float32)

    # Create gamma, beta, running_mean, running_var for 3 channels
    var param_shape = List[Int]()
    param_shape[0] = 3
    var gamma = ones(param_shape, DType.float32)
    var beta = zeros(param_shape, DType.float32)
    var running_mean = zeros(param_shape, DType.float32)
    var running_var = ones(param_shape, DType.float32)

    # Training mode
    var (output, new_mean, new_var) = batch_norm2d(
        x, gamma, beta, running_mean, running_var,
        training=True, momentum=0.1, epsilon=1e-5
    )

    # Check output shape
    assert_equal(output.shape()[0], 2)
    assert_equal(output.shape()[1], 3)
    assert_equal(output.shape()[2], 4)
    assert_equal(output.shape()[3], 4)

    # Check statistics shapes
    assert_equal(new_mean.shape()[0], 3)
    assert_equal(new_var.shape()[0], 3)


fn test_batch_norm2d_training_mode() raises:
    """Test that batch_norm2d computes batch statistics in training mode."""
    var shape = List[Int]()
    shape[0] = 2  # batch
    shape[1] = 1  # channels
    shape[2] = 2  # height
    shape[3] = 2  # width
    var x = zeros(shape, DType.float32)

    # Set specific values: [0, 1, 2, 3, 4, 5, 6, 7]
    for i in range(8):
        x._data.bitcast[Float32]()[i] = Float32(i)

    # Mean should be 3.5, variance should be computed from data
    var param_shape = List[Int]()
    param_shape[0] = 1
    var gamma = ones(param_shape, DType.float32)
    var beta = zeros(param_shape, DType.float32)
    var running_mean = zeros(param_shape, DType.float32)
    var running_var = ones(param_shape, DType.float32)

    var (output, new_mean, new_var) = batch_norm2d(
        x, gamma, beta, running_mean, running_var,
        training=True, momentum=0.1, epsilon=1e-5
    )

    # In training mode, running stats should be updated
    # new_running_mean = (1 - momentum) * old + momentum * batch_mean
    # = 0.9 * 0.0 + 0.1 * 3.5 = 0.35
    assert_almost_equal(
        new_mean._data.bitcast[Float32]()[0],
        Float32(0.35),
        tolerance=1e-4
    )


fn test_batch_norm2d_inference_mode() raises:
    """Test that batch_norm2d uses running statistics in inference mode."""
    var shape = List[Int]()
    shape[0] = 2  # batch
    shape[1] = 1  # channels
    shape[2] = 2  # height
    shape[3] = 2  # width
    var x = ones(shape, DType.float32)

    var param_shape = List[Int]()
    param_shape[0] = 1
    var gamma = ones(param_shape, DType.float32)
    var beta = zeros(param_shape, DType.float32)

    # Set running statistics
    var running_mean = zeros(param_shape, DType.float32)
    running_mean._data.bitcast[Float32]()[0] = 0.5

    var running_var = ones(param_shape, DType.float32)
    running_var._data.bitcast[Float32]()[0] = 0.25

    # Inference mode
    var (output, new_mean, new_var) = batch_norm2d(
        x, gamma, beta, running_mean, running_var,
        training=False, momentum=0.1, epsilon=1e-5
    )

    # Running statistics should be unchanged in inference mode
    assert_almost_equal(
        new_mean._data.bitcast[Float32]()[0],
        Float32(0.5),
        tolerance=1e-5
    )
    assert_almost_equal(
        new_var._data.bitcast[Float32]()[0],
        Float32(0.25),
        tolerance=1e-5
    )

    # Output should use running statistics for normalization
    # normalized = (x - running_mean) / sqrt(running_var + eps)
    # = (1.0 - 0.5) / sqrt(0.25 + 1e-5) ≈ 0.5 / 0.5 = 1.0
    # output = gamma * normalized + beta = 1.0 * 1.0 + 0.0 = 1.0
    for i in range(8):
        assert_almost_equal(
            output._data.bitcast[Float32]()[i],
            Float32(1.0),
            tolerance=1e-3
        )


fn test_batch_norm2d_scale_shift() raises:
    """Test that batch_norm2d applies gamma and beta correctly."""
    var shape = List[Int]()
    shape[0] = 1  # batch
    shape[1] = 2  # channels
    shape[2] = 2  # height
    shape[3] = 2  # width
    var x = zeros(shape, DType.float32)

    var param_shape = List[Int]()
    param_shape[0] = 2

    # Set gamma = [2.0, 3.0], beta = [1.0, -1.0]
    var gamma = zeros(param_shape, DType.float32)
    gamma._data.bitcast[Float32]()[0] = 2.0
    gamma._data.bitcast[Float32]()[1] = 3.0

    var beta = zeros(param_shape, DType.float32)
    beta._data.bitcast[Float32]()[0] = 1.0
    beta._data.bitcast[Float32]()[1] = -1.0

    var running_mean = zeros(param_shape, DType.float32)
    var running_var = ones(param_shape, DType.float32)

    # Inference mode with zero input and zero mean
    var (output, _, _) = batch_norm2d(
        x, gamma, beta, running_mean, running_var,
        training=False, momentum=0.1, epsilon=1e-5
    )

    # For zero input with zero mean: normalized = 0
    # output = gamma * 0 + beta = beta
    # Channel 0: beta[0] = 1.0
    # Channel 1: beta[1] = -1.0

    # Check channel 0 values (indices 0-3)
    for i in range(4):
        assert_almost_equal(
            output._data.bitcast[Float32]()[i],
            Float32(1.0),
            tolerance=1e-4
        )

    # Check channel 1 values (indices 4-7)
    for i in range(4, 8):
        assert_almost_equal(
            output._data.bitcast[Float32]()[i],
            Float32(-1.0),
            tolerance=1e-4
        )


fn test_batch_norm2d_zero_variance() raises:
    """Test that batch_norm2d handles zero variance with epsilon."""
    var shape = List[Int]()
    shape[0] = 2  # batch
    shape[1] = 1  # channels
    shape[2] = 1  # height
    shape[3] = 1  # width
    var x = ones(shape, DType.float32)  # All values are 1.0 - zero variance

    var param_shape = List[Int]()
    param_shape[0] = 1
    var gamma = ones(param_shape, DType.float32)
    var beta = zeros(param_shape, DType.float32)
    var running_mean = zeros(param_shape, DType.float32)
    var running_var = ones(param_shape, DType.float32)

    # This should not crash due to division by zero
    var (output, _, _) = batch_norm2d(
        x, gamma, beta, running_mean, running_var,
        training=True, momentum=0.1, epsilon=1e-5
    )

    # All outputs should be finite
    for i in range(2):
        var val = output._data.bitcast[Float32]()[i]
        assert_true(val == val)  # Not NaN
        assert_true(val > -1e10 and val < 1e10)  # Not infinite


# ============================================================================
# Batch Normalization Backward Pass Tests (GRADIENT CHECKING)
# ============================================================================


fn test_batch_norm2d_backward_gradient_input() raises:
    """Test batch_norm2d_backward gradient w.r.t. input using numerical validation.

    CRITICAL TEST: Validates mathematical correctness of batch norm backpropagation.
    Uses central finite differences for gold-standard gradient validation.
    """
    # Small tensor for gradient checking (computational cost is O(n²))
    var shape = List[Int]()
    shape[0] = 2  # batch
    shape[1] = 2  # channels
    shape[2] = 2  # height
    shape[3] = 2  # width

    # Create test input with varying values
    var x = zeros(shape, DType.float32)
    for i in range(16):
        x._data.bitcast[Float32]()[i] = Float32(i) * 0.1

    # Parameters
    var param_shape = List[Int]()
    param_shape[0] = 2
    var gamma = ones(param_shape, DType.float32)
    gamma._data.bitcast[Float32]()[0] = 1.5
    gamma._data.bitcast[Float32]()[1] = 2.0

    var beta = zeros(param_shape, DType.float32)
    var running_mean = zeros(param_shape, DType.float32)
    var running_var = ones(param_shape, DType.float32)

    # Forward pass
    var (output, _, _) = batch_norm2d(
        x, gamma, beta, running_mean, running_var,
        training=True, epsilon=1e-5
    )

    # Upstream gradient (typically from loss)
    var grad_output = ones_like(output)

    # Backward pass
    var (grad_input, grad_gamma, grad_beta) = batch_norm2d_backward(
        grad_output, x, gamma, running_mean, running_var,
        training=True, epsilon=1e-5
    )

    # Numerical gradient via finite differences
    fn forward_for_grad(inp: ExTensor) raises -> ExTensor:
        var (out, _, _) = batch_norm2d(
            inp, gamma, beta, running_mean, running_var,
            training=True, epsilon=1e-5
        )
        # Sum output to get scalar (gradient checking requires scalar loss)
        return reduce_sum(out, axis=-1, keepdims=False)

    var numerical_grad = compute_numerical_gradient(forward_for_grad, x, epsilon=1e-4)

    # Validate analytical gradient matches numerical gradient
    # Looser tolerance for batch norm (complex operation with many intermediate steps)
    assert_gradients_close(grad_input, numerical_grad, rtol=1e-2, atol=1e-5,
                          message="Batch norm gradient w.r.t. input")

    print("✓ Batch norm backward gradient (input) validated numerically")


fn test_batch_norm2d_backward_training_vs_inference() raises:
    """Test that batch_norm2d_backward behaves differently in training vs inference.

    Training mode: Gradients flow through batch statistics
    Inference mode: Gradients bypass statistics (use running stats)
    """
    var shape = List[Int]()
    shape[0] = 2  # batch
    shape[1] = 1  # channels
    shape[2] = 2  # height
    shape[3] = 2  # width

    var x = zeros(shape, DType.float32)
    for i in range(8):
        x._data.bitcast[Float32]()[i] = Float32(i)

    var param_shape = List[Int]()
    param_shape[0] = 1
    var gamma = ones(param_shape, DType.float32)
    gamma._data.bitcast[Float32]()[0] = 2.0

    var beta = zeros(param_shape, DType.float32)
    var running_mean = zeros(param_shape, DType.float32)
    var running_var = ones(param_shape, DType.float32)

    # Forward passes
    var (out_train, _, _) = batch_norm2d(x, gamma, beta, running_mean, running_var, training=True)
    var (out_infer, _, _) = batch_norm2d(x, gamma, beta, running_mean, running_var, training=False)

    var grad_output = ones_like(out_train)

    # Backward passes
    var (grad_train, _, _) = batch_norm2d_backward(
        grad_output, x, gamma, running_mean, running_var, training=True
    )
    var (grad_infer, _, _) = batch_norm2d_backward(
        grad_output, x, gamma, running_mean, running_var, training=False
    )

    # Gradients should differ between training and inference modes
    var diff_found = False
    for i in range(8):
        var diff = abs(grad_train._data.bitcast[Float32]()[i] -
                      grad_infer._data.bitcast[Float32]()[i])
        if diff > 1e-5:
            diff_found = True
            break

    assert_true(diff_found, "Training and inference gradients should differ")
    print("✓ Batch norm backward: training vs inference modes differ correctly")


fn test_batch_norm2d_backward_shapes() raises:
    """Test that batch_norm2d_backward returns correct gradient shapes."""
    var shape = List[Int]()
    shape[0] = 3  # batch
    shape[1] = 4  # channels
    shape[2] = 5  # height
    shape[3] = 5  # width

    var x = ones(shape, DType.float32)
    var grad_output = ones(shape, DType.float32)

    var param_shape = List[Int]()
    param_shape[0] = 4
    var gamma = ones(param_shape, DType.float32)
    var running_mean = zeros(param_shape, DType.float32)
    var running_var = ones(param_shape, DType.float32)

    var (grad_input, grad_gamma, grad_beta) = batch_norm2d_backward(
        grad_output, x, gamma, running_mean, running_var, training=True
    )

    # Validate shapes
    assert_shape_equal(grad_input, x, "grad_input should match input shape")
    assert_shape_equal(grad_gamma, gamma, "grad_gamma should match gamma shape")
    assert_shape_equal(grad_beta, gamma, "grad_beta should match beta shape")

    # Check specific dimensions
    assert_equal(grad_input.shape()[0], 3, "batch dimension")
    assert_equal(grad_input.shape()[1], 4, "channels dimension")
    assert_equal(grad_gamma.shape()[0], 4, "gamma channels")
    assert_equal(grad_beta.shape()[0], 4, "beta channels")


# ============================================================================
# Layer Normalization Tests
# ============================================================================


fn test_layer_norm_shapes_2d() raises:
    """Test that layer_norm returns correct shape for 2D input."""
    var shape = List[Int]()
    shape[0] = 4  # batch
    shape[1] = 10  # features
    var x = ones(shape, DType.float32)

    var param_shape = List[Int]()
    param_shape[0] = 10
    var gamma = ones(param_shape, DType.float32)
    var beta = zeros(param_shape, DType.float32)

    var output = layer_norm(x, gamma, beta, epsilon=1e-5)

    # Check output shape
    assert_equal(output.shape()[0], 4)
    assert_equal(output.shape()[1], 10)


fn test_layer_norm_shapes_4d() raises:
    """Test that layer_norm returns correct shape for 4D input."""
    var shape = List[Int]()
    shape[0] = 2  # batch
    shape[1] = 3  # channels
    shape[2] = 4  # height
    shape[3] = 4  # width
    var x = ones(shape, DType.float32)

    # For 4D input, normalize over C*H*W
    var normalized_shape = 3 * 4 * 4  # 48
    var param_shape = List[Int]()
    param_shape[0] = normalized_shape
    var gamma = ones(param_shape, DType.float32)
    var beta = zeros(param_shape, DType.float32)

    var output = layer_norm(x, gamma, beta, epsilon=1e-5)

    # Check output shape
    assert_equal(output.shape()[0], 2)
    assert_equal(output.shape()[1], 3)
    assert_equal(output.shape()[2], 4)
    assert_equal(output.shape()[3], 4)


fn test_layer_norm_normalization_2d() raises:
    """Test that layer_norm normalizes each sample independently."""
    var shape = List[Int]()
    shape[0] = 2  # batch
    shape[1] = 4  # features
    var x = zeros(shape, DType.float32)

    # Sample 1: [0, 1, 2, 3]
    x._data.bitcast[Float32]()[0] = 0.0
    x._data.bitcast[Float32]()[1] = 1.0
    x._data.bitcast[Float32]()[2] = 2.0
    x._data.bitcast[Float32]()[3] = 3.0

    # Sample 2: [4, 5, 6, 7]
    x._data.bitcast[Float32]()[4] = 4.0
    x._data.bitcast[Float32]()[5] = 5.0
    x._data.bitcast[Float32]()[6] = 6.0
    x._data.bitcast[Float32]()[7] = 7.0

    var param_shape = List[Int]()
    param_shape[0] = 4
    var gamma = ones(param_shape, DType.float32)
    var beta = zeros(param_shape, DType.float32)

    var output = layer_norm(x, gamma, beta, epsilon=1e-5)

    # For each sample, mean should be 0 and variance should be 1 (approximately)
    # Sample 1 mean: (0+1+2+3)/4 = 1.5
    # Sample 1 std: sqrt(((0-1.5)^2 + (1-1.5)^2 + (2-1.5)^2 + (3-1.5)^2)/4) ≈ 1.118

    # After normalization: (x - mean) / std
    # Sample 1[0]: (0 - 1.5) / 1.118 ≈ -1.34
    # Sample 1[1]: (1 - 1.5) / 1.118 ≈ -0.45
    # Sample 1[2]: (2 - 1.5) / 1.118 ≈ 0.45
    # Sample 1[3]: (3 - 1.5) / 1.118 ≈ 1.34

    # Check that first sample has approximately zero mean
    var sum1 = Float32(0.0)
    for i in range(4):
        sum1 += output._data.bitcast[Float32]()[i]
    var mean1 = sum1 / 4.0
    assert_almost_equal(mean1, Float32(0.0), tolerance=1e-5)

    # Check that second sample has approximately zero mean
    var sum2 = Float32(0.0)
    for i in range(4, 8):
        sum2 += output._data.bitcast[Float32]()[i]
    var mean2 = sum2 / 4.0
    assert_almost_equal(mean2, Float32(0.0), tolerance=1e-5)


fn test_layer_norm_scale_shift() raises:
    """Test that layer_norm applies gamma and beta correctly."""
    var shape = List[Int]()
    shape[0] = 1  # batch
    shape[1] = 3  # features
    var x = zeros(shape, DType.float32)

    var param_shape = List[Int]()
    param_shape[0] = 3

    # Set gamma = [2.0, 3.0, 4.0], beta = [1.0, 0.0, -1.0]
    var gamma = zeros(param_shape, DType.float32)
    gamma._data.bitcast[Float32]()[0] = 2.0
    gamma._data.bitcast[Float32]()[1] = 3.0
    gamma._data.bitcast[Float32]()[2] = 4.0

    var beta = zeros(param_shape, DType.float32)
    beta._data.bitcast[Float32]()[0] = 1.0
    beta._data.bitcast[Float32]()[1] = 0.0
    beta._data.bitcast[Float32]()[2] = -1.0

    var output = layer_norm(x, gamma, beta, epsilon=1e-5)

    # For zero input with zero mean: normalized = 0
    # output = gamma * 0 + beta = beta
    assert_almost_equal(
        output._data.bitcast[Float32]()[0],
        Float32(1.0),
        tolerance=1e-4
    )
    assert_almost_equal(
        output._data.bitcast[Float32]()[1],
        Float32(0.0),
        tolerance=1e-4
    )
    assert_almost_equal(
        output._data.bitcast[Float32]()[2],
        Float32(-1.0),
        tolerance=1e-4
    )


fn test_layer_norm_zero_variance() raises:
    """Test that layer_norm handles zero variance with epsilon."""
    var shape = List[Int]()
    shape[0] = 2  # batch
    shape[1] = 3  # features
    var x = ones(shape, DType.float32)  # All values are 1.0 - zero variance

    var param_shape = List[Int]()
    param_shape[0] = 3
    var gamma = ones(param_shape, DType.float32)
    var beta = zeros(param_shape, DType.float32)

    # This should not crash due to division by zero
    var output = layer_norm(x, gamma, beta, epsilon=1e-5)

    # All outputs should be finite
    for i in range(6):
        var val = output._data.bitcast[Float32]()[i]
        assert_true(val == val)  # Not NaN
        assert_true(val > -1e10 and val < 1e10)  # Not infinite


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run all normalization tests."""
    print("Running normalization tests...")

    # Batch normalization tests
    test_batch_norm2d_shapes()
    print("✓ test_batch_norm2d_shapes")

    test_batch_norm2d_training_mode()
    print("✓ test_batch_norm2d_training_mode")

    test_batch_norm2d_inference_mode()
    print("✓ test_batch_norm2d_inference_mode")

    test_batch_norm2d_scale_shift()
    print("✓ test_batch_norm2d_scale_shift")

    test_batch_norm2d_zero_variance()
    print("✓ test_batch_norm2d_zero_variance")

    # Batch normalization backward pass tests (gradient checking)
    test_batch_norm2d_backward_gradient_input()
    print("✓ test_batch_norm2d_backward_gradient_input")

    test_batch_norm2d_backward_training_vs_inference()
    print("✓ test_batch_norm2d_backward_training_vs_inference")

    test_batch_norm2d_backward_shapes()
    print("✓ test_batch_norm2d_backward_shapes")

    # Layer normalization tests
    test_layer_norm_shapes_2d()
    print("✓ test_layer_norm_shapes_2d")

    test_layer_norm_shapes_4d()
    print("✓ test_layer_norm_shapes_4d")

    test_layer_norm_normalization_2d()
    print("✓ test_layer_norm_normalization_2d")

    test_layer_norm_scale_shift()
    print("✓ test_layer_norm_scale_shift")

    test_layer_norm_zero_variance()
    print("✓ test_layer_norm_zero_variance")

    print("\nAll normalization tests passed!")
