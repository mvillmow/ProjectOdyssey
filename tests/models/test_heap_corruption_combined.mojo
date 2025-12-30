"""Combined test file to reproduce heap corruption bug (#2942).

This file runs all LeNet-5 tests in a single file to test if heap corruption
is still occurring. It combines:
- 6 conv tests (test_lenet5_conv_layers.mojo)
- 3 activation tests (test_lenet5_activation_layers.mojo)
- 4 pooling tests (test_lenet5_pooling_layers.mojo)
- 9 fc tests (test_lenet5_fc_layers.mojo)
- 2 reshape tests (test_lenet5_reshape_layers.mojo)

Total: 24 tests in one file.

If this file runs without crashing, the heap corruption bug may be resolved.
"""

from shared.core.extensor import ExTensor
from shared.core.conv import conv2d
from shared.core.pooling import maxpool2d
from shared.core.linear import linear
from shared.core.activation import relu
from shared.testing.layer_params import ConvFixture, LinearFixture
from shared.testing.assertions import (
    assert_shape,
    assert_dtype,
    assert_true,
    assert_false,
)
from shared.testing.special_values import (
    create_special_value_tensor,
    SPECIAL_VALUE_ONE,
)
from shared.testing.layer_testers import LayerTester
from math import isnan, isinf


# ============================================================================
# Fixtures
# ============================================================================


fn create_conv1_parameters(dtype: DType) raises -> Tuple[ExTensor, ExTensor]:
    """Create Conv1 layer parameters (1→6, 5x5 kernel)."""
    var fixture = ConvFixture(
        in_channels=1, out_channels=6, kernel_size=5, dtype=dtype
    )
    return fixture.kernel, fixture.bias


fn create_conv2_parameters(dtype: DType) raises -> Tuple[ExTensor, ExTensor]:
    """Create Conv2 layer parameters (6→16, 5x5 kernel)."""
    var fixture = ConvFixture(
        in_channels=6, out_channels=16, kernel_size=5, dtype=dtype
    )
    return fixture.kernel, fixture.bias


fn create_fc1_parameters(dtype: DType) raises -> Tuple[ExTensor, ExTensor]:
    """Create FC1 layer parameters (400→120)."""
    var fixture = LinearFixture(in_features=400, out_features=120, dtype=dtype)
    return fixture.weights, fixture.bias


fn create_fc2_parameters(dtype: DType) raises -> Tuple[ExTensor, ExTensor]:
    """Create FC2 layer parameters (120→84)."""
    var fixture = LinearFixture(in_features=120, out_features=84, dtype=dtype)
    return fixture.weights, fixture.bias


fn create_fc3_parameters(dtype: DType) raises -> Tuple[ExTensor, ExTensor]:
    """Create FC3 layer parameters (84→10)."""
    var fixture = LinearFixture(in_features=84, out_features=10, dtype=dtype)
    return fixture.weights, fixture.bias


# ============================================================================
# Conv1 Tests
# ============================================================================


fn test_conv1_forward_float32() raises:
    """Test Conv1 forward pass float32."""
    var dtype = DType.float32
    var _result = create_conv1_parameters(dtype)
    var kernel = _result[0]
    var bias = _result[1]
    LayerTester.test_conv_layer(
        in_channels=1, out_channels=6, kernel_size=5,
        input_h=28, input_w=28, weights=kernel, bias=bias, dtype=dtype
    )


fn test_conv1_forward_float16() raises:
    """Test Conv1 forward pass float16."""
    var dtype = DType.float16
    var _result = create_conv1_parameters(dtype)
    var kernel = _result[0]
    var bias = _result[1]
    LayerTester.test_conv_layer(
        in_channels=1, out_channels=6, kernel_size=5,
        input_h=28, input_w=28, weights=kernel, bias=bias, dtype=dtype
    )


fn test_conv1_backward_float32() raises:
    """Test Conv1 backward pass."""
    var dtype = DType.float32
    var _result = create_conv1_parameters(dtype)
    var kernel = _result[0]
    var bias = _result[1]
    LayerTester.test_conv_layer_backward(
        in_channels=1, out_channels=6, kernel_size=5,
        input_h=8, input_w=8, weights=kernel, bias=bias, dtype=dtype
    )


# ============================================================================
# Conv2 Tests
# ============================================================================


fn test_conv2_forward_float32() raises:
    """Test Conv2 forward pass float32."""
    var dtype = DType.float32
    var _result = create_conv2_parameters(dtype)
    var kernel = _result[0]
    var bias = _result[1]
    LayerTester.test_conv_layer(
        in_channels=6, out_channels=16, kernel_size=5,
        input_h=14, input_w=14, weights=kernel, bias=bias, dtype=dtype
    )


fn test_conv2_forward_float16() raises:
    """Test Conv2 forward pass float16."""
    var dtype = DType.float16
    var _result = create_conv2_parameters(dtype)
    var kernel = _result[0]
    var bias = _result[1]
    LayerTester.test_conv_layer(
        in_channels=6, out_channels=16, kernel_size=5,
        input_h=14, input_w=14, weights=kernel, bias=bias, dtype=dtype
    )


fn test_conv2_backward_float32() raises:
    """Test Conv2 backward pass."""
    var dtype = DType.float32
    var _result = create_conv2_parameters(dtype)
    var kernel = _result[0]
    var bias = _result[1]
    LayerTester.test_conv_layer_backward(
        in_channels=6, out_channels=16, kernel_size=5,
        input_h=8, input_w=8, weights=kernel, bias=bias, dtype=dtype
    )


# ============================================================================
# ReLU Tests
# ============================================================================


fn test_relu_forward_float32() raises:
    """Test ReLU forward pass float32."""
    var shape: List[Int] = [1, 6, 24, 24]
    LayerTester.test_activation_layer(shape, DType.float32, activation="relu")


fn test_relu_forward_float16() raises:
    """Test ReLU forward pass float16."""
    var shape: List[Int] = [1, 6, 24, 24]
    LayerTester.test_activation_layer(shape, DType.float16, activation="relu")


fn test_relu_backward_float32() raises:
    """Test ReLU backward pass."""
    var shape: List[Int] = [1, 6, 8, 8]
    LayerTester.test_activation_layer_backward(shape, DType.float32, activation="relu")


# ============================================================================
# MaxPool Tests
# ============================================================================


fn test_maxpool1_forward_float32() raises:
    """Test MaxPool1 forward pass float32."""
    LayerTester.test_pooling_layer(
        channels=6, input_h=24, input_w=24, pool_size=2, stride=2,
        dtype=DType.float32, pool_type="max", padding=0
    )


fn test_maxpool1_forward_float16() raises:
    """Test MaxPool1 forward pass float16."""
    LayerTester.test_pooling_layer(
        channels=6, input_h=24, input_w=24, pool_size=2, stride=2,
        dtype=DType.float16, pool_type="max", padding=0
    )


fn test_maxpool2_forward_float32() raises:
    """Test MaxPool2 forward pass float32."""
    LayerTester.test_pooling_layer(
        channels=16, input_h=10, input_w=10, pool_size=2, stride=2,
        dtype=DType.float32, pool_type="max", padding=0
    )


fn test_maxpool2_forward_float16() raises:
    """Test MaxPool2 forward pass float16."""
    LayerTester.test_pooling_layer(
        channels=16, input_h=10, input_w=10, pool_size=2, stride=2,
        dtype=DType.float16, pool_type="max", padding=0
    )


# ============================================================================
# FC Tests
# ============================================================================


fn test_fc1_forward_float32() raises:
    """Test FC1 forward pass float32."""
    var dtype = DType.float32
    var _result = create_fc1_parameters(dtype)
    var weights = _result[0]
    var bias = _result[1]
    LayerTester.test_linear_layer(
        in_features=400, out_features=120, weights=weights, bias=bias, dtype=dtype
    )


fn test_fc1_forward_float16() raises:
    """Test FC1 forward pass float16."""
    var dtype = DType.float16
    var _result = create_fc1_parameters(dtype)
    var weights = _result[0]
    var bias = _result[1]
    LayerTester.test_linear_layer(
        in_features=400, out_features=120, weights=weights, bias=bias, dtype=dtype
    )


fn test_fc1_backward_float32() raises:
    """Test FC1 backward pass."""
    var dtype = DType.float32
    var _result = create_fc1_parameters(dtype)
    var weights = _result[0]
    var bias = _result[1]
    LayerTester.test_linear_layer_backward(
        in_features=400, out_features=120, weights=weights, bias=bias, dtype=dtype
    )


fn test_fc2_forward_float32() raises:
    """Test FC2 forward pass float32."""
    var dtype = DType.float32
    var _result = create_fc2_parameters(dtype)
    var weights = _result[0]
    var bias = _result[1]
    LayerTester.test_linear_layer(
        in_features=120, out_features=84, weights=weights, bias=bias, dtype=dtype
    )


fn test_fc2_forward_float16() raises:
    """Test FC2 forward pass float16."""
    var dtype = DType.float16
    var _result = create_fc2_parameters(dtype)
    var weights = _result[0]
    var bias = _result[1]
    LayerTester.test_linear_layer(
        in_features=120, out_features=84, weights=weights, bias=bias, dtype=dtype
    )


fn test_fc2_backward_float32() raises:
    """Test FC2 backward pass."""
    var dtype = DType.float32
    var _result = create_fc2_parameters(dtype)
    var weights = _result[0]
    var bias = _result[1]
    LayerTester.test_linear_layer_backward(
        in_features=120, out_features=84, weights=weights, bias=bias, dtype=dtype
    )


fn test_fc3_forward_float32() raises:
    """Test FC3 forward pass float32."""
    var dtype = DType.float32
    var _result = create_fc3_parameters(dtype)
    var weights = _result[0]
    var bias = _result[1]
    LayerTester.test_linear_layer(
        in_features=84, out_features=10, weights=weights, bias=bias, dtype=dtype
    )


fn test_fc3_forward_float16() raises:
    """Test FC3 forward pass float16."""
    var dtype = DType.float16
    var _result = create_fc3_parameters(dtype)
    var weights = _result[0]
    var bias = _result[1]
    LayerTester.test_linear_layer(
        in_features=84, out_features=10, weights=weights, bias=bias, dtype=dtype
    )


fn test_fc3_backward_float32() raises:
    """Test FC3 backward pass."""
    var dtype = DType.float32
    var _result = create_fc3_parameters(dtype)
    var weights = _result[0]
    var bias = _result[1]
    LayerTester.test_linear_layer_backward(
        in_features=84, out_features=10, weights=weights, bias=bias, dtype=dtype
    )


# ============================================================================
# Flatten/Reshape Tests
# ============================================================================


fn test_flatten_operation_float32() raises:
    """Test reshape/flatten operation (16, 5, 5) -> (400,)."""
    var dtype = DType.float32
    var input = create_special_value_tensor([1, 16, 5, 5], dtype, SPECIAL_VALUE_ONE)
    var flattened = input.reshape([1, 400])
    assert_shape(flattened, [1, 400], "Flatten shape mismatch")
    assert_dtype(flattened, dtype, "Flatten dtype mismatch")
    var expected_value = 1.0
    for i in range(flattened.numel()):
        var val = flattened._get_float64(i)
        assert_false(isnan(val), "Flatten produced NaN")
        assert_false(isinf(val), "Flatten produced Inf")


fn test_flatten_operation_float16() raises:
    """Test flatten with float16."""
    var dtype = DType.float16
    var input = create_special_value_tensor([1, 16, 5, 5], dtype, SPECIAL_VALUE_ONE)
    var flattened = input.reshape([1, 400])
    assert_shape(flattened, [1, 400], "Flatten shape mismatch (float16)")
    assert_dtype(flattened, dtype, "Flatten dtype mismatch (float16)")


fn main() raises:
    """Run all 24 tests to check for heap corruption."""
    print("Heap Corruption Test - Running 24 Tests in One File")
    print("=" * 60)
    print("If this crashes, heap corruption bug (#2942) is still present.")
    print("")

    var test_count = 0

    # Conv1 tests (3)
    print("[1/24] test_conv1_forward_float32...", end="")
    test_conv1_forward_float32()
    print(" OK")
    test_count += 1

    print("[2/24] test_conv1_forward_float16...", end="")
    test_conv1_forward_float16()
    print(" OK")
    test_count += 1

    print("[3/24] test_conv1_backward_float32...", end="")
    test_conv1_backward_float32()
    print(" OK")
    test_count += 1

    # Conv2 tests (3)
    print("[4/24] test_conv2_forward_float32...", end="")
    test_conv2_forward_float32()
    print(" OK")
    test_count += 1

    print("[5/24] test_conv2_forward_float16...", end="")
    test_conv2_forward_float16()
    print(" OK")
    test_count += 1

    print("[6/24] test_conv2_backward_float32...", end="")
    test_conv2_backward_float32()
    print(" OK")
    test_count += 1

    # ReLU tests (3)
    print("[7/24] test_relu_forward_float32...", end="")
    test_relu_forward_float32()
    print(" OK")
    test_count += 1

    print("[8/24] test_relu_forward_float16...", end="")
    test_relu_forward_float16()
    print(" OK")
    test_count += 1

    print("[9/24] test_relu_backward_float32...", end="")
    test_relu_backward_float32()
    print(" OK")
    test_count += 1

    # MaxPool tests (4)
    print("[10/24] test_maxpool1_forward_float32...", end="")
    test_maxpool1_forward_float32()
    print(" OK")
    test_count += 1

    print("[11/24] test_maxpool1_forward_float16...", end="")
    test_maxpool1_forward_float16()
    print(" OK")
    test_count += 1

    print("[12/24] test_maxpool2_forward_float32...", end="")
    test_maxpool2_forward_float32()
    print(" OK")
    test_count += 1

    print("[13/24] test_maxpool2_forward_float16...", end="")
    test_maxpool2_forward_float16()
    print(" OK")
    test_count += 1

    # FC1 tests (3)
    print("[14/24] test_fc1_forward_float32...", end="")
    test_fc1_forward_float32()
    print(" OK")
    test_count += 1

    print("[15/24] test_fc1_forward_float16...", end="")
    test_fc1_forward_float16()
    print(" OK")
    test_count += 1

    print("[16/24] test_fc1_backward_float32...", end="")
    test_fc1_backward_float32()
    print(" OK")
    test_count += 1

    # FC2 tests (3)
    print("[17/24] test_fc2_forward_float32...", end="")
    test_fc2_forward_float32()
    print(" OK")
    test_count += 1

    print("[18/24] test_fc2_forward_float16...", end="")
    test_fc2_forward_float16()
    print(" OK")
    test_count += 1

    print("[19/24] test_fc2_backward_float32...", end="")
    test_fc2_backward_float32()
    print(" OK")
    test_count += 1

    # FC3 tests (3)
    print("[20/24] test_fc3_forward_float32...", end="")
    test_fc3_forward_float32()
    print(" OK")
    test_count += 1

    print("[21/24] test_fc3_forward_float16...", end="")
    test_fc3_forward_float16()
    print(" OK")
    test_count += 1

    print("[22/24] test_fc3_backward_float32...", end="")
    test_fc3_backward_float32()
    print(" OK")
    test_count += 1

    # Flatten tests (2)
    print("[23/24] test_flatten_operation_float32...", end="")
    test_flatten_operation_float32()
    print(" OK")
    test_count += 1

    print("[24/24] test_flatten_operation_float16...", end="")
    test_flatten_operation_float16()
    print(" OK")
    test_count += 1

    print("")
    print("=" * 60)
    print("✅ ALL 24 TESTS PASSED - NO HEAP CORRUPTION DETECTED!")
    print("")
    print("The heap corruption bug (#2942) may be fixed or no longer reproducible.")
    print("Mojo version: 0.26.1.0.dev2025122805")
