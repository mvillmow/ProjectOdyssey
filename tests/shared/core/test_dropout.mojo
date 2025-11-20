"""Tests for dropout regularization.

Tests cover:
- Standard dropout (element-wise)
- Spatial dropout (channel-wise for CNNs)
- Training vs inference mode
- Mask generation and backward pass
- Reproducibility with seed

All tests use pure functional API.
"""

from tests.shared.conftest import (
    assert_true,
    assert_equal,
    assert_almost_equal,
    assert_shape_equal,
    TestFixtures,
)
from shared.core.extensor import ExTensor, zeros, ones
from shared.core.dropout import dropout, dropout2d, dropout_backward, dropout2d_backward
from collections.vector import DynamicVector


# ============================================================================
# Standard Dropout Tests
# ============================================================================


fn test_dropout_shapes() raises:
    """Test that dropout returns correct output and mask shapes."""
    var shape = DynamicVector[Int](2)
    shape[0] = 4
    shape[1] = 10
    var x = ones(shape, DType.float32)

    # Training mode
    var (output, mask) = dropout(x, p=0.5, training=True, seed=42)

    # Check shapes
    assert_equal(output.shape()[0], 4)
    assert_equal(output.shape()[1], 10)
    assert_equal(mask.shape()[0], 4)
    assert_equal(mask.shape()[1], 10)


fn test_dropout_inference_mode() raises:
    """Test that dropout passes input unchanged in inference mode."""
    var shape = DynamicVector[Int](2)
    shape[0] = 3
    shape[1] = 5
    var x = ones(shape, DType.float32)

    # Inference mode
    var (output, mask) = dropout(x, p=0.5, training=False)

    # Output should be unchanged
    var size = x.numel()
    for i in range(size):
        assert_almost_equal(
            output._data.bitcast[Float32]()[i],
            x._data.bitcast[Float32]()[i],
            tolerance=1e-5
        )

    # Mask should be all ones
    for i in range(size):
        assert_almost_equal(
            mask._data.bitcast[Float32]()[i],
            Float32(1.0),
            tolerance=1e-5
        )


fn test_dropout_probability() raises:
    """Test that dropout approximately drops p% of elements."""
    var shape = DynamicVector[Int](2)
    shape[0] = 100
    shape[1] = 100
    var x = ones(shape, DType.float32)

    var p = 0.5
    var (output, mask) = dropout(x, p=p, training=True, seed=42)

    # Count dropped elements (where mask is 0)
    var total = x.numel()
    var dropped = 0

    for i in range(total):
        if mask._data.bitcast[Float32]()[i] == 0.0:
            dropped += 1

    # Should be approximately 50% dropped (within 10% tolerance for randomness)
    var drop_rate = Float64(dropped) / Float64(total)
    var expected = p
    var tolerance = 0.1  # 10% tolerance

    assert_true(drop_rate > expected - tolerance)
    assert_true(drop_rate < expected + tolerance)


fn test_dropout_scaling() raises:
    """Test that kept elements are scaled by 1/(1-p)."""
    var shape = DynamicVector[Int](2)
    shape[0] = 10
    shape[1] = 10
    var x = ones(shape, DType.float32)

    var p = 0.5
    var (output, mask) = dropout(x, p=p, training=True, seed=42)

    # Elements that weren't dropped should be scaled by 1/(1-p) = 2.0
    var expected_scale = Float32(1.0 / (1.0 - p))

    for i in range(x.numel()):
        var mask_val = mask._data.bitcast[Float32]()[i]
        var out_val = output._data.bitcast[Float32]()[i]

        if mask_val > 0:
            # Not dropped - should be scaled
            assert_almost_equal(out_val, expected_scale, tolerance=1e-5)
        else:
            # Dropped - should be zero
            assert_almost_equal(out_val, Float32(0.0), tolerance=1e-5)


fn test_dropout_reproducibility() raises:
    """Test that dropout with same seed produces same mask."""
    var shape = DynamicVector[Int](2)
    shape[0] = 5
    shape[1] = 5
    var x = ones(shape, DType.float32)

    # Same seed should produce same mask
    var (output1, mask1) = dropout(x, p=0.5, training=True, seed=42)
    var (output2, mask2) = dropout(x, p=0.5, training=True, seed=42)

    # Masks should be identical
    for i in range(x.numel()):
        assert_almost_equal(
            mask1._data.bitcast[Float32]()[i],
            mask2._data.bitcast[Float32]()[i],
            tolerance=1e-5
        )


fn test_dropout_backward_shapes() raises:
    """Test that dropout_backward returns correct gradient shape."""
    var shape = DynamicVector[Int](2)
    shape[0] = 4
    shape[1] = 8
    var x = ones(shape, DType.float32)

    # Forward pass
    var (output, mask) = dropout(x, p=0.5, training=True, seed=42)

    # Backward pass
    var grad_output = ones(shape, DType.float32)
    var grad_input = dropout_backward(grad_output, mask, p=0.5)

    # Check shape
    assert_equal(grad_input.shape()[0], 4)
    assert_equal(grad_input.shape()[1], 8)


fn test_dropout_backward_gradient_flow() raises:
    """Test that dropout_backward only passes gradients through non-dropped elements."""
    var shape = DynamicVector[Int](2)
    shape[0] = 3
    shape[1] = 3
    var x = ones(shape, DType.float32)

    # Forward pass
    var (output, mask) = dropout(x, p=0.5, training=True, seed=42)

    # Backward pass with all-ones gradient
    var grad_output = ones(shape, DType.float32)
    var grad_input = dropout_backward(grad_output, mask, p=0.5)

    # Gradient should only flow where mask is non-zero
    var scale = Float32(1.0 / (1.0 - 0.5))
    for i in range(x.numel()):
        var mask_val = mask._data.bitcast[Float32]()[i]
        var grad_val = grad_input._data.bitcast[Float32]()[i]

        if mask_val > 0:
            # Gradient should be scaled
            assert_almost_equal(grad_val, scale, tolerance=1e-5)
        else:
            # Gradient should be zero
            assert_almost_equal(grad_val, Float32(0.0), tolerance=1e-5)


# ============================================================================
# Spatial Dropout (Dropout2D) Tests
# ============================================================================


fn test_dropout2d_shapes() raises:
    """Test that dropout2d returns correct output and mask shapes."""
    var shape = DynamicVector[Int](4)
    shape[0] = 2  # batch
    shape[1] = 3  # channels
    shape[2] = 4  # height
    shape[3] = 4  # width
    var x = ones(shape, DType.float32)

    # Training mode
    var (output, mask) = dropout2d(x, p=0.2, training=True, seed=42)

    # Check shapes match input
    assert_equal(output.shape()[0], 2)
    assert_equal(output.shape()[1], 3)
    assert_equal(output.shape()[2], 4)
    assert_equal(output.shape()[3], 4)


fn test_dropout2d_channel_level() raises:
    """Test that dropout2d drops entire channels (all spatial positions)."""
    var shape = DynamicVector[Int](4)
    shape[0] = 1  # batch
    shape[1] = 4  # channels
    shape[2] = 3  # height
    shape[3] = 3  # width
    var x = ones(shape, DType.float32)

    var (output, mask) = dropout2d(x, p=0.5, training=True, seed=42)

    # Check that entire channels are either all kept or all dropped
    var channels = 4
    var height = 3
    var width = 3
    var spatial_size = height * width

    for c in range(channels):
        # Get first pixel value in channel
        var first_idx = c * spatial_size
        var first_val = mask._data.bitcast[Float32]()[first_idx]

        # All pixels in this channel should have same mask value
        for h in range(height):
            for w in range(width):
                var idx = c * spatial_size + h * width + w
                var val = mask._data.bitcast[Float32]()[idx]
                assert_almost_equal(val, first_val, tolerance=1e-5)


fn test_dropout2d_inference_mode() raises:
    """Test that dropout2d passes input unchanged in inference mode."""
    var shape = DynamicVector[Int](4)
    shape[0] = 2
    shape[1] = 3
    shape[2] = 4
    shape[3] = 4
    var x = ones(shape, DType.float32)

    # Inference mode
    var (output, mask) = dropout2d(x, p=0.5, training=False)

    # Output should be unchanged
    var size = x.numel()
    for i in range(size):
        assert_almost_equal(
            output._data.bitcast[Float32]()[i],
            x._data.bitcast[Float32]()[i],
            tolerance=1e-5
        )


fn test_dropout2d_backward_shapes() raises:
    """Test that dropout2d_backward returns correct gradient shape."""
    var shape = DynamicVector[Int](4)
    shape[0] = 2
    shape[1] = 4
    shape[2] = 8
    shape[3] = 8
    var x = ones(shape, DType.float32)

    # Forward pass
    var (output, mask) = dropout2d(x, p=0.2, training=True, seed=42)

    # Backward pass
    var grad_output = ones(shape, DType.float32)
    var grad_input = dropout2d_backward(grad_output, mask, p=0.2)

    # Check shape
    assert_equal(grad_input.shape()[0], 2)
    assert_equal(grad_input.shape()[1], 4)
    assert_equal(grad_input.shape()[2], 8)
    assert_equal(grad_input.shape()[3], 8)


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run all dropout tests."""
    print("Running dropout tests...")

    # Standard dropout tests
    test_dropout_shapes()
    print("✓ test_dropout_shapes")

    test_dropout_inference_mode()
    print("✓ test_dropout_inference_mode")

    test_dropout_probability()
    print("✓ test_dropout_probability")

    test_dropout_scaling()
    print("✓ test_dropout_scaling")

    test_dropout_reproducibility()
    print("✓ test_dropout_reproducibility")

    test_dropout_backward_shapes()
    print("✓ test_dropout_backward_shapes")

    test_dropout_backward_gradient_flow()
    print("✓ test_dropout_backward_gradient_flow")

    # Spatial dropout (dropout2d) tests
    test_dropout2d_shapes()
    print("✓ test_dropout2d_shapes")

    test_dropout2d_channel_level()
    print("✓ test_dropout2d_channel_level")

    test_dropout2d_inference_mode()
    print("✓ test_dropout2d_inference_mode")

    test_dropout2d_backward_shapes()
    print("✓ test_dropout2d_backward_shapes")

    print("\nAll dropout tests passed!")
