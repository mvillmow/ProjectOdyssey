"""Unit tests for 2D pooling operations.

Tests cover:
- maxpool2d: Max pooling with selectable implementation
- avgpool2d: Average pooling with selectable implementation
- global_avgpool2d: Global average pooling
- Padding and stride behavior
- Shape computations
- Numerical correctness

All tests use pure functional API - no internal state.
"""

from tests.shared.conftest import (
    assert_true,
    assert_equal,
    assert_almost_equal,
    assert_shape_equal,
    TestFixtures,
)
from shared.core.extensor import ExTensor, zeros, ones
from shared.core.pooling import maxpool2d, avgpool2d, global_avgpool2d


# ============================================================================
# MaxPool2D Tests
# ============================================================================


fn test_maxpool2d_output_shape() raises:
    """Test maxpool2d output shape computation.

    Formula: out_size = (in_size + 2*padding - kernel_size) // stride + 1

    Test case: 8x8 input, kernel=2, stride=2 (default), padding=0
    Expected: 4x4 output (8 - 2) // 2 + 1 = 4
    """
    var batch = 1
    var channels = 1
    var in_h = 8
    var in_w = 8

    # Create input: (1, 1, 8, 8)
    var input_shape = List[Int]()
    input_shape[0] = batch
    input_shape[1] = channels
    input_shape[2] = in_h
    input_shape[3] = in_w
    var input = ones(input_shape, DType.float32)

    # Compute maxpool: kernel=2, stride=2 (default)
    var output = maxpool2d(input, kernel_size=2, stride=2, padding=0)

    # Check output shape: (1, 1, 4, 4)
    var out_shape = output.shape()
    assert_equal(out_shape[0], 1)
    assert_equal(out_shape[1], 1)
    assert_equal(out_shape[2], 4)
    assert_equal(out_shape[3], 4)


fn test_maxpool2d_stride_default() raises:
    """Test maxpool2d with default stride (uses kernel_size).

    When stride=0, it should default to kernel_size.
    """
    var batch = 1
    var channels = 1

    # Create input: (1, 1, 6, 6)
    var input_shape = List[Int]()
    input_shape[0] = batch
    input_shape[1] = channels
    input_shape[2] = 6
    input_shape[3] = 6
    var input = ones(input_shape, DType.float32)

    # Compute maxpool: kernel=3, stride=0 (default to kernel_size=3)
    var output = maxpool2d(input, kernel_size=3, stride=0, padding=0)

    # Check output shape: (1, 1, 2, 2)
    # (6 - 3) // 3 + 1 = 2
    var out_shape = output.shape()
    assert_equal(out_shape[0], 1)
    assert_equal(out_shape[1], 1)
    assert_equal(out_shape[2], 2)
    assert_equal(out_shape[3], 2)


fn test_maxpool2d_numerical_correctness() raises:
    """Test maxpool2d produces correct numerical output.

    Input: [[1, 2], [3, 4]] (one 2x2 window)
    Kernel: 2x2
    Expected max: 4.0
    """
    # Create input: (1, 1, 2, 2) with values [1, 2, 3, 4]
    var input_shape = List[Int]()
    input_shape[0] = 1
    input_shape[1] = 1
    input_shape[2] = 2
    input_shape[3] = 2
    var input = ones(input_shape, DType.float32)
    input._data.bitcast[Float32]()[0] = 1.0
    input._data.bitcast[Float32]()[1] = 2.0
    input._data.bitcast[Float32]()[2] = 3.0
    input._data.bitcast[Float32]()[3] = 4.0

    # Compute maxpool: kernel=2, stride=2
    var output = maxpool2d(input, kernel_size=2, stride=2, padding=0)

    # Check output shape: (1, 1, 1, 1)
    var out_shape = output.shape()
    assert_equal(out_shape[0], 1)
    assert_equal(out_shape[1], 1)
    assert_equal(out_shape[2], 1)
    assert_equal(out_shape[3], 1)

    # Check output value: max([1, 2, 3, 4]) = 4.0
    var result = output._data.bitcast[Float32]()[0]
    assert_almost_equal(result, Float32(4.0), tolerance=1e-5)


fn test_maxpool2d_multi_channel() raises:
    """Test maxpool2d with multiple channels.

    Pooling should be applied independently to each channel.
    """
    var batch = 1
    var channels = 3

    # Create input: (1, 3, 4, 4)
    var input_shape = List[Int]()
    input_shape[0] = batch
    input_shape[1] = channels
    input_shape[2] = 4
    input_shape[3] = 4
    var input = ones(input_shape, DType.float32)

    # Compute maxpool: kernel=2, stride=2
    var output = maxpool2d(input, kernel_size=2, stride=2, padding=0)

    # Check output shape: (1, 3, 2, 2) - channels preserved
    var out_shape = output.shape()
    assert_equal(out_shape[0], 1)
    assert_equal(out_shape[1], 3)  # Channels preserved
    assert_equal(out_shape[2], 2)
    assert_equal(out_shape[3], 2)


fn test_maxpool2d_batched() raises:
    """Test maxpool2d with batch size > 1.

    Pooling should be applied independently to each batch element.
    """
    var batch = 4
    var channels = 1

    # Create input: (4, 1, 6, 6)
    var input_shape = List[Int]()
    input_shape[0] = batch
    input_shape[1] = channels
    input_shape[2] = 6
    input_shape[3] = 6
    var input = ones(input_shape, DType.float32)

    # Compute maxpool: kernel=2, stride=2
    var output = maxpool2d(input, kernel_size=2, stride=2, padding=0)

    # Check output shape: (4, 1, 3, 3) - batch preserved
    var out_shape = output.shape()
    assert_equal(out_shape[0], 4)  # Batch preserved
    assert_equal(out_shape[1], 1)
    assert_equal(out_shape[2], 3)
    assert_equal(out_shape[3], 3)


fn test_maxpool2d_method_selection() raises:
    """Test maxpool2d method parameter.

    Currently only 'direct' method is supported.
    """
    var input_shape = List[Int]()
    input_shape[0] = 1
    input_shape[1] = 1
    input_shape[2] = 4
    input_shape[3] = 4
    var input = ones(input_shape, DType.float32)

    # Test with explicit method="direct"
    var output = maxpool2d(input, kernel_size=2, stride=2, padding=0, method="direct")

    # Should succeed
    var out_shape = output.shape()
    assert_equal(out_shape[2], 2)


# ============================================================================
# AvgPool2D Tests
# ============================================================================


fn test_avgpool2d_output_shape() raises:
    """Test avgpool2d output shape computation."""
    var batch = 1
    var channels = 1

    # Create input: (1, 1, 8, 8)
    var input_shape = List[Int]()
    input_shape[0] = batch
    input_shape[1] = channels
    input_shape[2] = 8
    input_shape[3] = 8
    var input = ones(input_shape, DType.float32)

    # Compute avgpool: kernel=2, stride=2
    var output = avgpool2d(input, kernel_size=2, stride=2, padding=0)

    # Check output shape: (1, 1, 4, 4)
    var out_shape = output.shape()
    assert_equal(out_shape[0], 1)
    assert_equal(out_shape[1], 1)
    assert_equal(out_shape[2], 4)
    assert_equal(out_shape[3], 4)


fn test_avgpool2d_numerical_correctness() raises:
    """Test avgpool2d produces correct numerical output.

    Input: [[1, 2], [3, 4]] (one 2x2 window)
    Kernel: 2x2
    Expected average: (1+2+3+4) / 4 = 2.5
    """
    # Create input: (1, 1, 2, 2) with values [1, 2, 3, 4]
    var input_shape = List[Int]()
    input_shape[0] = 1
    input_shape[1] = 1
    input_shape[2] = 2
    input_shape[3] = 2
    var input = ones(input_shape, DType.float32)
    input._data.bitcast[Float32]()[0] = 1.0
    input._data.bitcast[Float32]()[1] = 2.0
    input._data.bitcast[Float32]()[2] = 3.0
    input._data.bitcast[Float32]()[3] = 4.0

    # Compute avgpool: kernel=2, stride=2
    var output = avgpool2d(input, kernel_size=2, stride=2, padding=0)

    # Check output value: avg([1, 2, 3, 4]) = 2.5
    var result = output._data.bitcast[Float32]()[0]
    assert_almost_equal(result, Float32(2.5), tolerance=1e-5)


fn test_avgpool2d_multi_channel() raises:
    """Test avgpool2d with multiple channels."""
    var batch = 1
    var channels = 4

    # Create input: (1, 4, 6, 6)
    var input_shape = List[Int]()
    input_shape[0] = batch
    input_shape[1] = channels
    input_shape[2] = 6
    input_shape[3] = 6
    var input = ones(input_shape, DType.float32)

    # Compute avgpool: kernel=3, stride=3
    var output = avgpool2d(input, kernel_size=3, stride=3, padding=0)

    # Check output shape: (1, 4, 2, 2) - channels preserved
    var out_shape = output.shape()
    assert_equal(out_shape[0], 1)
    assert_equal(out_shape[1], 4)  # Channels preserved
    assert_equal(out_shape[2], 2)
    assert_equal(out_shape[3], 2)


fn test_avgpool2d_method_selection() raises:
    """Test avgpool2d method parameter."""
    var input_shape = List[Int]()
    input_shape[0] = 1
    input_shape[1] = 1
    input_shape[2] = 4
    input_shape[3] = 4
    var input = ones(input_shape, DType.float32)

    # Test with explicit method="direct"
    var output = avgpool2d(input, kernel_size=2, stride=2, padding=0, method="direct")

    # Should succeed
    var out_shape = output.shape()
    assert_equal(out_shape[2], 2)


# ============================================================================
# Global AvgPool2D Tests
# ============================================================================


fn test_global_avgpool2d_output_shape() raises:
    """Test global_avgpool2d reduces spatial dims to 1x1."""
    var batch = 2
    var channels = 3
    var height = 8
    var width = 8

    # Create input: (2, 3, 8, 8)
    var input_shape = List[Int]()
    input_shape[0] = batch
    input_shape[1] = channels
    input_shape[2] = height
    input_shape[3] = width
    var input = ones(input_shape, DType.float32)

    # Compute global avgpool
    var output = global_avgpool2d(input)

    # Check output shape: (2, 3, 1, 1)
    var out_shape = output.shape()
    assert_equal(out_shape[0], 2)  # Batch preserved
    assert_equal(out_shape[1], 3)  # Channels preserved
    assert_equal(out_shape[2], 1)  # Height reduced to 1
    assert_equal(out_shape[3], 1)  # Width reduced to 1


fn test_global_avgpool2d_numerical_correctness() raises:
    """Test global_avgpool2d computes correct average.

    Input: (1, 1, 2, 2) with values [1, 2, 3, 4]
    Expected average: (1+2+3+4) / 4 = 2.5
    """
    # Create input: (1, 1, 2, 2) with values [1, 2, 3, 4]
    var input_shape = List[Int]()
    input_shape[0] = 1
    input_shape[1] = 1
    input_shape[2] = 2
    input_shape[3] = 2
    var input = ones(input_shape, DType.float32)
    input._data.bitcast[Float32]()[0] = 1.0
    input._data.bitcast[Float32]()[1] = 2.0
    input._data.bitcast[Float32]()[2] = 3.0
    input._data.bitcast[Float32]()[3] = 4.0

    # Compute global avgpool
    var output = global_avgpool2d(input)

    # Check output shape: (1, 1, 1, 1)
    var out_shape = output.shape()
    assert_equal(out_shape[0], 1)
    assert_equal(out_shape[1], 1)
    assert_equal(out_shape[2], 1)
    assert_equal(out_shape[3], 1)

    # Check output value: avg([1, 2, 3, 4]) = 2.5
    var result = output._data.bitcast[Float32]()[0]
    assert_almost_equal(result, Float32(2.5), tolerance=1e-5)


fn test_global_avgpool2d_multi_channel() raises:
    """Test global_avgpool2d with multiple channels.

    Each channel should be averaged independently.
    """
    var batch = 1
    var channels = 2

    # Create input: (1, 2, 3, 3)
    var input_shape = List[Int]()
    input_shape[0] = batch
    input_shape[1] = channels
    input_shape[2] = 3
    input_shape[3] = 3
    var input = ones(input_shape, DType.float32)

    # Channel 0: all 1.0, Channel 1: all 2.0
    for c in range(channels):
        for h in range(3):
            for w in range(3):
                var idx = c * 9 + h * 3 + w
                input._data.bitcast[Float32]()[idx] = Float32(c + 1)

    # Compute global avgpool
    var output = global_avgpool2d(input)

    # Check channel 0: avg = 1.0
    var ch0_result = output._data.bitcast[Float32]()[0]
    assert_almost_equal(ch0_result, Float32(1.0), tolerance=1e-5)

    # Check channel 1: avg = 2.0
    var ch1_result = output._data.bitcast[Float32]()[1]
    assert_almost_equal(ch1_result, Float32(2.0), tolerance=1e-5)


fn test_global_avgpool2d_method_selection() raises:
    """Test global_avgpool2d method parameter."""
    var input_shape = List[Int]()
    input_shape[0] = 1
    input_shape[1] = 1
    input_shape[2] = 4
    input_shape[3] = 4
    var input = ones(input_shape, DType.float32)

    # Test with explicit method="direct"
    var output = global_avgpool2d(input, method="direct")

    # Should succeed
    var out_shape = output.shape()
    assert_equal(out_shape[2], 1)
    assert_equal(out_shape[3], 1)


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run all pooling tests."""
    print("Running pooling tests...")

    # MaxPool2D tests
    test_maxpool2d_output_shape()
    print("✓ test_maxpool2d_output_shape")

    test_maxpool2d_stride_default()
    print("✓ test_maxpool2d_stride_default")

    test_maxpool2d_numerical_correctness()
    print("✓ test_maxpool2d_numerical_correctness")

    test_maxpool2d_multi_channel()
    print("✓ test_maxpool2d_multi_channel")

    test_maxpool2d_batched()
    print("✓ test_maxpool2d_batched")

    test_maxpool2d_method_selection()
    print("✓ test_maxpool2d_method_selection")

    # AvgPool2D tests
    test_avgpool2d_output_shape()
    print("✓ test_avgpool2d_output_shape")

    test_avgpool2d_numerical_correctness()
    print("✓ test_avgpool2d_numerical_correctness")

    test_avgpool2d_multi_channel()
    print("✓ test_avgpool2d_multi_channel")

    test_avgpool2d_method_selection()
    print("✓ test_avgpool2d_method_selection")

    # Global AvgPool2D tests
    test_global_avgpool2d_output_shape()
    print("✓ test_global_avgpool2d_output_shape")

    test_global_avgpool2d_numerical_correctness()
    print("✓ test_global_avgpool2d_numerical_correctness")

    test_global_avgpool2d_multi_channel()
    print("✓ test_global_avgpool2d_multi_channel")

    test_global_avgpool2d_method_selection()
    print("✓ test_global_avgpool2d_method_selection")

    print("\nAll pooling tests passed!")
