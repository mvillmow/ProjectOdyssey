"""Unit tests for 2D convolution operations.

Tests cover:
- conv2d: Direct 2D convolution with bias
- conv2d_no_bias: Direct 2D convolution without bias
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
from shared.core.conv import conv2d, conv2d_no_bias


# ============================================================================
# Conv2D Basic Tests
# ============================================================================


fn test_conv2d_initialization() raises:
    """Test that conv2d parameters can be created with correct shapes.

    Functional API Note:
        Caller creates kernel (out_ch, in_ch, kH, kW) and bias (out_ch,).
        This test verifies parameters can be created.
    """
    var out_channels = 16
    var in_channels = 3
    var kernel_h = 3
    var kernel_w = 3

    # Kernel shape: (out_channels, in_channels, kH, kW)
    var kernel_shape = List[Int](4)
    kernel_shape[0] = out_channels
    kernel_shape[1] = in_channels
    kernel_shape[2] = kernel_h
    kernel_shape[3] = kernel_w
    var kernel = ones(kernel_shape, DType.float32)

    # Bias shape: (out_channels,)
    var bias_shape = List[Int](1)
    bias_shape[0] = out_channels
    var bias = zeros(bias_shape, DType.float32)

    # Verify shapes
    var k_shape = kernel.shape()
    var b_shape = bias.shape()
    assert_equal(k_shape[0], out_channels)
    assert_equal(k_shape[1], in_channels)
    assert_equal(k_shape[2], kernel_h)
    assert_equal(k_shape[3], kernel_w)
    assert_equal(b_shape[0], out_channels)


fn test_conv2d_output_shape() raises:
    """Test conv2d output shape computation.

    Formula: out_size = (in_size + 2*padding - kernel_size) // stride + 1

    Test case: 8x8 input, 3x3 kernel, stride=1, padding=0
    Expected: 6x6 output (8 - 3 + 1 = 6)
    """
    var batch = 1
    var in_channels = 1
    var out_channels = 1
    var in_h = 8
    var in_w = 8
    var kh = 3
    var kw = 3

    # Create input: (1, 1, 8, 8)
    var input_shape = List[Int](4)
    input_shape[0] = batch
    input_shape[1] = in_channels
    input_shape[2] = in_h
    input_shape[3] = in_w
    var input = ones(input_shape, DType.float32)

    # Create kernel: (1, 1, 3, 3)
    var kernel_shape = List[Int](4)
    kernel_shape[0] = out_channels
    kernel_shape[1] = in_channels
    kernel_shape[2] = kh
    kernel_shape[3] = kw
    var kernel = ones(kernel_shape, DType.float32)

    # Create bias: (1,)
    var bias_shape = List[Int](1)
    bias_shape[0] = out_channels
    var bias = zeros(bias_shape, DType.float32)

    # Compute convolution: stride=1, padding=0
    var output = conv2d(input, kernel, bias, stride=1, padding=0)

    # Check output shape: (1, 1, 6, 6)
    var out_shape = output.shape()
    assert_equal(out_shape[0], 1)
    assert_equal(out_shape[1], 1)
    assert_equal(out_shape[2], 6)  # (8 - 3) / 1 + 1 = 6
    assert_equal(out_shape[3], 6)


fn test_conv2d_with_padding() raises:
    """Test conv2d with padding.

    With padding=1, a 6x6 input with 3x3 kernel should produce 6x6 output.
    Formula: (6 + 2*1 - 3) // 1 + 1 = 6
    """
    var batch = 1
    var in_channels = 1
    var out_channels = 1
    var in_h = 6
    var in_w = 6
    var kh = 3
    var kw = 3

    # Create input: (1, 1, 6, 6)
    var input_shape = List[Int](4)
    input_shape[0] = batch
    input_shape[1] = in_channels
    input_shape[2] = in_h
    input_shape[3] = in_w
    var input = ones(input_shape, DType.float32)

    # Create kernel: (1, 1, 3, 3)
    var kernel_shape = List[Int](4)
    kernel_shape[0] = out_channels
    kernel_shape[1] = in_channels
    kernel_shape[2] = kh
    kernel_shape[3] = kw
    var kernel = ones(kernel_shape, DType.float32)

    # Create bias: (1,)
    var bias_shape = List[Int](1)
    bias_shape[0] = out_channels
    var bias = zeros(bias_shape, DType.float32)

    # Compute with padding=1
    var output = conv2d(input, kernel, bias, stride=1, padding=1)

    # Check output shape: (1, 1, 6, 6) - same as input due to padding
    var out_shape = output.shape()
    assert_equal(out_shape[0], 1)
    assert_equal(out_shape[1], 1)
    assert_equal(out_shape[2], 6)
    assert_equal(out_shape[3], 6)


fn test_conv2d_with_stride() raises:
    """Test conv2d with stride=2.

    8x8 input, 3x3 kernel, stride=2, padding=0
    Formula: (8 - 3) // 2 + 1 = 3
    Expected: 3x3 output
    """
    var batch = 1
    var in_channels = 1
    var out_channels = 1

    # Create input: (1, 1, 8, 8)
    var input_shape = List[Int](4)
    input_shape[0] = batch
    input_shape[1] = in_channels
    input_shape[2] = 8
    input_shape[3] = 8
    var input = ones(input_shape, DType.float32)

    # Create kernel: (1, 1, 3, 3)
    var kernel_shape = List[Int](4)
    kernel_shape[0] = out_channels
    kernel_shape[1] = in_channels
    kernel_shape[2] = 3
    kernel_shape[3] = 3
    var kernel = ones(kernel_shape, DType.float32)

    # Create bias: (1,)
    var bias_shape = List[Int](1)
    bias_shape[0] = out_channels
    var bias = zeros(bias_shape, DType.float32)

    # Compute with stride=2
    var output = conv2d(input, kernel, bias, stride=2, padding=0)

    # Check output shape: (1, 1, 3, 3)
    var out_shape = output.shape()
    assert_equal(out_shape[0], 1)
    assert_equal(out_shape[1], 1)
    assert_equal(out_shape[2], 3)
    assert_equal(out_shape[3], 3)


fn test_conv2d_numerical_correctness() raises:
    """Test conv2d produces correct numerical output.

    Simple case:
    - 1x1 input: [[1.0]]
    - 1x1 kernel: [[2.0]]
    - bias: [0.5]
    - Expected output: 1.0 * 2.0 + 0.5 = 2.5
    """
    # Create input: (1, 1, 1, 1) with value 1.0
    var input_shape = List[Int](4)
    input_shape[0] = 1
    input_shape[1] = 1
    input_shape[2] = 1
    input_shape[3] = 1
    var input = ones(input_shape, DType.float32)

    # Create kernel: (1, 1, 1, 1) with value 2.0
    var kernel_shape = List[Int](4)
    kernel_shape[0] = 1
    kernel_shape[1] = 1
    kernel_shape[2] = 1
    kernel_shape[3] = 1
    var kernel = ones(kernel_shape, DType.float32)
    kernel._data.bitcast[Float32]()[0] = 2.0

    # Create bias: (1,) with value 0.5
    var bias_shape = List[Int](1)
    bias_shape[0] = 1
    var bias = zeros(bias_shape, DType.float32)
    bias._data.bitcast[Float32]()[0] = 0.5

    # Compute convolution
    var output = conv2d(input, kernel, bias, stride=1, padding=0)

    # Check output: 1.0 * 2.0 + 0.5 = 2.5
    var result = output._data.bitcast[Float32]()[0]
    assert_almost_equal(result, Float32(2.5), tolerance=1e-5)


fn test_conv2d_multi_channel() raises:
    """Test conv2d with multiple input and output channels.

    Input: (1, 2, 3, 3) - 2 input channels
    Kernel: (3, 2, 2, 2) - 3 output channels, 2 input channels
    Output: (1, 3, 2, 2) - 3 output channels
    """
    var batch = 1
    var in_channels = 2
    var out_channels = 3
    var in_h = 3
    var in_w = 3

    # Create input: (1, 2, 3, 3)
    var input_shape = List[Int](4)
    input_shape[0] = batch
    input_shape[1] = in_channels
    input_shape[2] = in_h
    input_shape[3] = in_w
    var input = ones(input_shape, DType.float32)

    # Create kernel: (3, 2, 2, 2)
    var kernel_shape = List[Int](4)
    kernel_shape[0] = out_channels
    kernel_shape[1] = in_channels
    kernel_shape[2] = 2
    kernel_shape[3] = 2
    var kernel = ones(kernel_shape, DType.float32)

    # Create bias: (3,)
    var bias_shape = List[Int](1)
    bias_shape[0] = out_channels
    var bias = zeros(bias_shape, DType.float32)

    # Compute convolution
    var output = conv2d(input, kernel, bias, stride=1, padding=0)

    # Check output shape: (1, 3, 2, 2)
    var out_shape = output.shape()
    assert_equal(out_shape[0], 1)
    assert_equal(out_shape[1], 3)  # 3 output channels
    assert_equal(out_shape[2], 2)  # (3 - 2) + 1 = 2
    assert_equal(out_shape[3], 2)


fn test_conv2d_no_bias() raises:
    """Test conv2d_no_bias produces correct output without bias.

    Should be equivalent to conv2d with zero bias.
    """
    # Create input: (1, 1, 3, 3)
    var input_shape = List[Int](4)
    input_shape[0] = 1
    input_shape[1] = 1
    input_shape[2] = 3
    input_shape[3] = 3
    var input = ones(input_shape, DType.float32)

    # Create kernel: (1, 1, 2, 2)
    var kernel_shape = List[Int](4)
    kernel_shape[0] = 1
    kernel_shape[1] = 1
    kernel_shape[2] = 2
    kernel_shape[3] = 2
    var kernel = ones(kernel_shape, DType.float32)

    # Compute without bias
    var output = conv2d_no_bias(input, kernel, stride=1, padding=0)

    # Check output shape: (1, 1, 2, 2)
    var out_shape = output.shape()
    assert_equal(out_shape[0], 1)
    assert_equal(out_shape[1], 1)
    assert_equal(out_shape[2], 2)
    assert_equal(out_shape[3], 2)

    # Check numerical value: 4 ones summed = 4.0
    var result = output._data.bitcast[Float32]()[0]
    assert_almost_equal(result, Float32(4.0), tolerance=1e-5)


fn test_conv2d_batched() raises:
    """Test conv2d with batch size > 1.

    Verify that convolution is applied independently to each batch element.
    """
    var batch = 4
    var in_channels = 1
    var out_channels = 1

    # Create input: (4, 1, 4, 4)
    var input_shape = List[Int](4)
    input_shape[0] = batch
    input_shape[1] = in_channels
    input_shape[2] = 4
    input_shape[3] = 4
    var input = ones(input_shape, DType.float32)

    # Create kernel: (1, 1, 2, 2)
    var kernel_shape = List[Int](4)
    kernel_shape[0] = out_channels
    kernel_shape[1] = in_channels
    kernel_shape[2] = 2
    kernel_shape[3] = 2
    var kernel = ones(kernel_shape, DType.float32)

    # Create bias: (1,)
    var bias_shape = List[Int](1)
    bias_shape[0] = out_channels
    var bias = zeros(bias_shape, DType.float32)

    # Compute convolution
    var output = conv2d(input, kernel, bias, stride=1, padding=0)

    # Check output shape: (4, 1, 3, 3) - batch size preserved
    var out_shape = output.shape()
    assert_equal(out_shape[0], 4)
    assert_equal(out_shape[1], 1)
    assert_equal(out_shape[2], 3)
    assert_equal(out_shape[3], 3)


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run all conv2d tests."""
    print("Running conv2d tests...")

    test_conv2d_initialization()
    print("✓ test_conv2d_initialization")

    test_conv2d_output_shape()
    print("✓ test_conv2d_output_shape")

    test_conv2d_with_padding()
    print("✓ test_conv2d_with_padding")

    test_conv2d_with_stride()
    print("✓ test_conv2d_with_stride")

    test_conv2d_numerical_correctness()
    print("✓ test_conv2d_numerical_correctness")

    test_conv2d_multi_channel()
    print("✓ test_conv2d_multi_channel")

    test_conv2d_no_bias()
    print("✓ test_conv2d_no_bias")

    test_conv2d_batched()
    print("✓ test_conv2d_batched")

    print("\nAll conv2d tests passed!")
