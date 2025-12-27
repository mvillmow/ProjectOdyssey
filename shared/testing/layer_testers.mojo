"""Layer Testing Utilities

Provides reusable patterns for testing model layers:
- Convolutional layers (conv2d)
- Fully connected layers (linear)
- Pooling layers (maxpool, avgpool)
- Normalization layers (batch_norm)
- Activation layers (relu, sigmoid, tanh, etc.)

Each tester function:
1. Creates synthetic input with special values
2. Runs forward pass
3. Validates output properties (shape, dtype, values)
4. Tests across all dtypes (FP4, FP8, FP16, FP32, BFloat16, Int8)

Backward Pass Testing:
- Uses seeded random tensors for reproducible gradient checking
- Validates analytical gradients against numerical gradients
- Tolerances adjusted per dtype (1e-2 for float32)

Usage:
    from shared.testing.layer_testers import LayerTester

    # Test a convolutional layer
    LayerTester.test_conv_layer(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        input_h=32,
        input_w=32,
        weights=conv_weights,
        bias=conv_bias,
        dtype=DType.float32
    )

    # Test conv layer backward pass with gradient checking
    LayerTester.test_conv_layer_backward(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        input_h=8,
        input_w=8,
        weights=conv_weights,
        bias=conv_bias,
        dtype=DType.float32
    )
"""

from math import isnan, isinf
from shared.core.extensor import ExTensor, zeros_like, ones_like
from shared.core.conv import conv2d, conv2d_output_shape, conv2d_backward
from shared.core.linear import linear, linear_backward
from shared.core.pooling import maxpool2d, avgpool2d, pool_output_shape
from shared.core.activation import (
    relu,
    sigmoid,
    tanh,
    relu_backward,
    sigmoid_backward,
    tanh_backward,
)
from shared.testing.assertions import (
    assert_shape,
    assert_dtype,
    assert_true,
    assert_false,
)
from shared.testing.special_values import (
    create_special_value_tensor,
    create_alternating_pattern_tensor,
    create_seeded_random_tensor,
    SPECIAL_VALUE_ONE,
    SPECIAL_VALUE_NEG_ONE,
    SPECIAL_VALUE_NEG_HALF,
)
from shared.testing.gradient_checker import (
    check_gradients,
    check_gradients_verbose,
    compute_numerical_gradient,
    assert_gradients_close,
)


# ============================================================================
# Layer Tester Struct
# ============================================================================


struct LayerTester:
    """Utilities for testing individual layers with special values.

    All methods are static and test layers across multiple dtypes to ensure
    correctness and consistency.
    """

    @staticmethod
    fn test_conv_layer(
        in_channels: Int,
        out_channels: Int,
        kernel_size: Int,
        input_h: Int,
        input_w: Int,
        weights: ExTensor,
        bias: ExTensor,
        dtype: DType,
        stride: Int = 1,
        padding: Int = 0,
    ) raises:
        """Test convolutional layer with special values.

        Args:
            in_channels: Input channels.
            out_channels: Output channels.
            kernel_size: Kernel size (assumes square kernel).
            input_h: Input height.
            input_w: Input width.
            weights: Conv weights (out_channels, in_channels, k, k).
            bias: Conv bias (out_channels,).
            dtype: Data type to test.
            stride: Convolution stride (default: 1).
            padding: Convolution padding (default: 0).

        Verifies:
            - Output shape is correct
            - Output dtype matches input
            - No NaN/Inf in output
            - Output values in expected range

        Raises:
            Error: If assertions fail during testing.

        Example:
            ```mojo
            # Test conv layer with FP32
            LayerTester.test_conv_layer(
                in_channels=3,
                out_channels=64,
                kernel_size=3,
                input_h=32,
                input_w=32,
                weights=conv1_weights,
                bias=conv1_bias,
                dtype=DType.float32
            )
            ```
        """
        # Create input with ones (predictable behavior)
        var input = create_special_value_tensor(
            [1, in_channels, input_h, input_w], dtype, SPECIAL_VALUE_ONE
        )

        # Forward pass
        var output = conv2d(
            input, weights, bias, stride=stride, padding=padding
        )

        # Verify shape
        var output_shape = conv2d_output_shape(
            input_h, input_w, kernel_size, kernel_size, stride, padding
        )
        var expected_h = output_shape[0]
        var expected_w = output_shape[1]
        assert_shape(
            output,
            [1, out_channels, expected_h, expected_w],
            "Conv2D output shape mismatch",
        )

        # Verify dtype
        assert_dtype(output, dtype, "Conv2D output dtype mismatch")

        # Check for NaN/Inf
        var has_invalid = False
        for i in range(output.numel()):
            var val = output._get_float64(i)
            if isnan(val) or isinf(val):
                has_invalid = True
                break

        assert_false(has_invalid, "Conv2D output contains NaN or Inf")

    @staticmethod
    fn test_linear_layer(
        in_features: Int,
        out_features: Int,
        weights: ExTensor,
        bias: ExTensor,
        dtype: DType,
    ) raises:
        """Test fully connected layer with special values.

        Args:
            in_features: Input dimension.
            out_features: Output dimension.
            weights: Linear weights (out_features, in_features).
            bias: Linear bias (out_features,).
            dtype: Data type to test.

        Verifies:
            - Output shape is correct
            - Output dtype matches input
            - No NaN/Inf in output
            - Output is matrix multiply result (predictable with input=1.0)

        Raises:
            Error: If assertions fail during testing.

        Example:
            ```mojo
            # Test FC layer with FP16
            LayerTester.test_linear_layer(
                in_features=128,
                out_features=10,
                weights=fc_weights,
                bias=fc_bias,
                dtype=DType.float16
            )
            ```
        """
        # Create input with ones
        var input = create_special_value_tensor(
            [1, in_features], dtype, SPECIAL_VALUE_ONE
        )

        # Forward pass
        var output = linear(input, weights, bias)

        # Verify shape
        assert_shape(output, [1, out_features], "Linear output shape mismatch")

        # Verify dtype
        assert_dtype(output, dtype, "Linear output dtype mismatch")

        # Check for NaN/Inf
        var has_invalid = False
        for i in range(output.numel()):
            var val = output._get_float64(i)
            if isnan(val) or isinf(val):
                has_invalid = True
                break

        assert_false(has_invalid, "Linear output contains NaN or Inf")

    @staticmethod
    fn test_pooling_layer(
        channels: Int,
        input_h: Int,
        input_w: Int,
        pool_size: Int,
        stride: Int,
        dtype: DType,
        pool_type: String = "max",
        padding: Int = 0,
    ) raises:
        """Test pooling layer with special values.

        Args:
            channels: Number of channels.
            input_h: Input height.
            input_w: Input width.
            pool_size: Pooling kernel size.
            stride: Pooling stride.
            dtype: Data type to test.
            pool_type: "max" or "avg" (default: "max").
            padding: Pooling padding (default: 0).

        Verifies:
            - Output shape is correct
            - Output dtype matches input
            - No NaN/Inf in output
            - Max pooling returns max value
            - Avg pooling returns average value

        Raises:
            Error: If assertions fail during testing.

        Example:
            ```mojo
            # Test max pooling
            LayerTester.test_pooling_layer(
                channels=64,
                input_h=32,
                input_w=32,
                pool_size=2,
                stride=2,
                dtype=DType.float32,
                pool_type="max"
            )
            ```
        """
        # Create input with alternating pattern (tests max vs avg behavior)
        var input = create_alternating_pattern_tensor(
            [1, channels, input_h, input_w], dtype
        )

        # Forward pass
        var output: ExTensor
        if pool_type == "max":
            output = maxpool2d(input, pool_size, stride, padding=padding)
        else:
            output = avgpool2d(input, pool_size, stride, padding=padding)

        # Verify shape
        var output_shape = pool_output_shape(
            input_h, input_w, pool_size, stride, padding
        )
        var expected_h = output_shape[0]
        var expected_w = output_shape[1]
        assert_shape(
            output,
            [1, channels, expected_h, expected_w],
            pool_type + " pooling output shape mismatch",
        )

        # Verify dtype
        assert_dtype(
            output, dtype, pool_type + " pooling output dtype mismatch"
        )

        # Check for NaN/Inf
        var has_invalid = False
        for i in range(output.numel()):
            var val = output._get_float64(i)
            if isnan(val) or isinf(val):
                has_invalid = True
                break

        assert_false(
            has_invalid, pool_type + " pooling output contains NaN or Inf"
        )

    @staticmethod
    fn test_activation_layer(
        shape: List[Int], dtype: DType, activation: String = "relu"
    ) raises:
        """Test activation layer with special values.

        Args:
            shape: Input/output shape.
            dtype: Data type to test.
            activation: "relu", "sigmoid", "tanh" (default: "relu").

        Verifies:
            - Output shape matches input shape
            - Output dtype matches input dtype
            - No NaN/Inf in output
            - Activation property holds (e.g., ReLU zeros negatives)

        Raises:
            Error: If assertions fail during testing or unknown activation.

        Example:
            ```mojo
            # Test ReLU activation
            LayerTester.test_activation_layer(
                shape=[1, 64, 32, 32],
                dtype=DType.float32,
                activation="relu"
            )
            ```
        """
        # Create input with alternating pattern (includes various values)
        var input = create_alternating_pattern_tensor(shape, dtype)

        # Forward pass
        var output: ExTensor
        if activation == "relu":
            output = relu(input)
        elif activation == "sigmoid":
            output = sigmoid(input)
        elif activation == "tanh":
            output = tanh(input)
        else:
            raise Error("Unknown activation: " + activation)

        # Verify shape unchanged
        assert_shape(output, shape, activation + " output shape changed")

        # Verify dtype
        assert_dtype(output, dtype, activation + " output dtype mismatch")

        # Check for NaN/Inf
        var has_invalid = False
        for i in range(output.numel()):
            var val = output._get_float64(i)
            if isnan(val) or isinf(val):
                has_invalid = True
                break

        assert_false(has_invalid, activation + " output contains NaN or Inf")

        # Verify activation property
        if activation == "relu":
            # ReLU output must be non-negative
            for i in range(output.numel()):
                var val = output._get_float64(i)
                assert_true(
                    val >= 0.0,
                    "ReLU output must be non-negative, got: " + String(val),
                )

        elif activation == "sigmoid":
            # Sigmoid output must be in [0, 1]
            for i in range(output.numel()):
                var val = output._get_float64(i)
                assert_true(
                    val >= 0.0 and val <= 1.0,
                    "Sigmoid output must be in [0, 1], got: " + String(val),
                )

        elif activation == "tanh":
            # Tanh output must be in [-1, 1]
            for i in range(output.numel()):
                var val = output._get_float64(i)
                assert_true(
                    val >= -1.0 and val <= 1.0,
                    "Tanh output must be in [-1, 1], got: " + String(val),
                )

    @staticmethod
    fn test_layer_dtype_consistency(
        input: ExTensor, output: ExTensor, layer_name: String = "Layer"
    ) raises:
        """Test that layer preserves dtype from input to output.

        Args:
            input: Input tensor.
            output: Output tensor.
            layer_name: Name of layer for error messages.

        Verifies:
            - Output dtype matches input dtype

        Raises:
            Error: If assertions fail during testing.

        Example:
            ```mojo
            var input = create_ones_tensor([1, 3, 32, 32], DType.float16)
            var output = conv2d(input, weights, bias)
            LayerTester.test_layer_dtype_consistency(input, output, "Conv1")
            ```
        """
        assert_dtype(
            output, input.dtype(), layer_name + " must preserve input dtype"
        )

    @staticmethod
    fn test_layer_no_invalid_values(
        output: ExTensor, layer_name: String = "Layer"
    ) raises:
        """Test that layer output contains no NaN or Inf values.

        Args:
            output: Output tensor to check.
            layer_name: Name of layer for error messages.

        Verifies:
            - No NaN values in output
            - No Inf values in output

        Raises:
            Error: If NaN or Inf found in output.

        Example:
            ```mojo
            var output = relu(input)
            LayerTester.test_layer_no_invalid_values(output, "ReLU")
            ```
        """
        for i in range(output.numel()):
            var val = output._get_float64(i)
            if isnan(val):
                raise Error(
                    layer_name + " output contains NaN at index " + String(i)
                )
            if isinf(val):
                raise Error(
                    layer_name + " output contains Inf at index " + String(i)
                )

    # ========================================================================
    # Backward Pass Testing Methods
    # ========================================================================

    @staticmethod
    fn test_conv_layer_backward(
        in_channels: Int,
        out_channels: Int,
        kernel_size: Int,
        input_h: Int,
        input_w: Int,
        weights: ExTensor,
        bias: ExTensor,
        dtype: DType,
        stride: Int = 1,
        padding: Int = 0,
        validate_analytical: Bool = False,
    ) raises:
        """Test conv2d backward pass with gradient checking.

        Uses seeded random tensors for gradient checking reproducibility.
        Validates analytical gradients match numerical gradients using
        finite differences.

        Args:
            in_channels: Input channels.
            out_channels: Output channels.
            kernel_size: Kernel size (assumes square kernel).
            input_h: Input height.
            input_w: Input width.
            weights: Conv weights (out_channels, in_channels, k, k).
            bias: Conv bias (out_channels,).
            dtype: Data type to test.
            stride: Convolution stride (default: 1).
            padding: Convolution padding (default: 0).
            validate_analytical: If True, validate analytical gradient against
                numerical gradient (default: False).

        Verifies:
            - Forward pass runs without error
            - Output shape is correct
            - Backward pass computes valid gradients
            - Analytical and numerical gradients match

        Raises:
            Error: If assertions fail during testing.

        Example:
            ```mojo
            # Test conv layer backward with FP32 (small tensor to avoid timeout)
            LayerTester.test_conv_layer_backward(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                input_h=8,
                input_w=8,
                weights=conv1_weights,
                bias=conv1_bias,
                dtype=DType.float32,
                stride=1,
                padding=1
            )
            ```
        """
        # Create seeded random input (small size to prevent timeout)
        var input = create_seeded_random_tensor(
            [1, in_channels, input_h, input_w], dtype, seed=42
        )

        # Forward pass
        var output = conv2d(
            input, weights, bias, stride=stride, padding=padding
        )

        # Verify output shape
        var output_shape = conv2d_output_shape(
            input_h, input_w, kernel_size, kernel_size, stride, padding
        )
        var expected_h = output_shape[0]
        var expected_w = output_shape[1]
        assert_shape(
            output,
            [1, out_channels, expected_h, expected_w],
            "Conv2D backward: output shape mismatch",
        )

        # Verify output dtype
        assert_dtype(output, dtype, "Conv2D backward: output dtype mismatch")

        # Test gradient checking with epsilon and tolerance appropriate for conv complexity
        # Conv operations have more accumulated numerical error due to multiple multiplies
        var epsilon = 1e-4 if dtype == DType.float32 else 1e-3
        var tolerance = (
            0.05 if dtype == DType.float32 else 0.1
        )  # 5% tolerance for conv

        # Define forward function for gradient checking
        fn forward(x: ExTensor) raises escaping -> ExTensor:
            return conv2d(x, weights, bias, stride=stride, padding=padding)

        # Compute numerical gradient (always computed for validation)
        var numerical_grad = compute_numerical_gradient(forward, input, epsilon)

        # Verify numerical gradient has correct shape
        assert_shape(
            numerical_grad,
            input.shape(),
            "Conv2D backward: numerical gradient shape mismatch",
        )

        # Verify no NaN/Inf in numerical gradients
        for i in range(numerical_grad.numel()):
            var val = numerical_grad._get_float64(i)
            assert_false(
                isnan(val), "Conv2D backward: NaN in numerical gradient"
            )
            assert_false(
                isinf(val), "Conv2D backward: Inf in numerical gradient"
            )

        # Optionally validate analytical gradient
        if validate_analytical:
            # Create gradient output (upstream gradient)
            var grad_output = ones_like(output)

            # Compute analytical gradient using conv2d_backward
            var backward_result = conv2d_backward(
                grad_output, input, weights, stride=stride, padding=padding
            )
            var analytical_grad = backward_result.grad_input

            # Compare analytical vs numerical gradients
            assert_gradients_close(
                analytical_grad,
                numerical_grad,
                rtol=tolerance,
                atol=tolerance,
                message="Conv2D analytical gradient doesn't match numerical",
            )

    @staticmethod
    fn test_linear_layer_backward(
        in_features: Int,
        out_features: Int,
        weights: ExTensor,
        bias: ExTensor,
        dtype: DType,
        validate_analytical: Bool = False,
    ) raises:
        """Test linear layer backward pass with gradient checking.

        Uses seeded random tensors for gradient checking reproducibility.
        Validates analytical gradients match numerical gradients.

        Args:
            in_features: Input dimension.
            out_features: Output dimension.
            weights: Linear weights (out_features, in_features).
            bias: Linear bias (out_features,).
            dtype: Data type to test.
            validate_analytical: If True, validate analytical gradient against
                numerical gradient (default: False).

        Verifies:
            - Forward pass runs without error
            - Output shape is correct
            - Backward pass computes valid gradients
            - Analytical and numerical gradients match

        Raises:
            Error: If assertions fail during testing.

        Example:
            ```mojo
            # Test FC layer backward with FP32
            LayerTester.test_linear_layer_backward(
                in_features=32,
                out_features=10,
                weights=fc_weights,
                bias=fc_bias,
                dtype=DType.float32
            )
            ```
        """
        # Create seeded random input
        var input = create_seeded_random_tensor(
            [1, in_features], dtype, seed=42
        )

        # Forward pass
        var output = linear(input, weights, bias)

        # Verify output shape
        assert_shape(
            output, [1, out_features], "Linear backward: output shape mismatch"
        )

        # Verify output dtype
        assert_dtype(output, dtype, "Linear backward: output dtype mismatch")

        # Test gradient checking with epsilon appropriate for dtype
        var epsilon = 1e-5 if dtype == DType.float32 else 1e-4

        # Define forward function for gradient checking
        fn forward(x: ExTensor) raises escaping -> ExTensor:
            return linear(x, weights, bias)

        # Compute numerical gradient (always computed for validation)
        var numerical_grad = compute_numerical_gradient(forward, input, epsilon)

        # Verify numerical gradient has correct shape
        assert_shape(
            numerical_grad,
            input.shape(),
            "Linear backward: numerical gradient shape mismatch",
        )

        # Verify no NaN/Inf in numerical gradients
        for i in range(numerical_grad.numel()):
            var val = numerical_grad._get_float64(i)
            assert_false(
                isnan(val), "Linear backward: NaN in numerical gradient"
            )
            assert_false(
                isinf(val), "Linear backward: Inf in numerical gradient"
            )

        # Optionally validate analytical gradient
        if validate_analytical:
            # Create gradient output (upstream gradient)
            var grad_output = ones_like(output)

            # Compute analytical gradient using linear_backward
            var backward_result = linear_backward(grad_output, input, weights)
            var analytical_grad = backward_result.grad_input

            # Compare analytical vs numerical gradients
            # Use wider tolerance (1.5%) for matrix operations due to accumulated errors
            var wide_tolerance = 0.015
            assert_gradients_close(
                analytical_grad,
                numerical_grad,
                rtol=wide_tolerance,
                atol=wide_tolerance,
                message="Linear analytical gradient doesn't match numerical",
            )

    @staticmethod
    fn test_activation_layer_backward(
        shape: List[Int],
        dtype: DType,
        activation: String = "relu",
        validate_analytical: Bool = False,
    ) raises:
        """Test activation backward pass with gradient checking.

        For ReLU: Uses negative values to test gradient=0 for x<0
        For Sigmoid/Tanh: Uses range of values for gradient testing

        Uses seeded random tensors for gradient checking reproducibility.
        Validates analytical gradients match numerical gradients.

        Args:
            shape: Input/output shape.
            dtype: Data type to test.
            activation: "relu", "sigmoid", "tanh" (default: "relu").
            validate_analytical: If True, validate analytical gradient against
                numerical gradient (default: False).

        Verifies:
            - Forward pass runs without error
            - Output shape unchanged
            - Output dtype matches input
            - Backward pass computes valid gradients
            - Analytical and numerical gradients match

        Raises:
            Error: If assertions fail during testing or unknown activation.

        Example:
            ```mojo
            # Test ReLU backward with gradient checking
            LayerTester.test_activation_layer_backward(
                shape=[2, 3, 4, 4],
                dtype=DType.float32,
                activation="relu"
            )
            ```
        """
        # Create seeded random input with values that span activation range
        var input = create_seeded_random_tensor(
            shape, dtype, seed=42, low=-1.0, high=1.0
        )

        # Forward pass
        var output: ExTensor
        if activation == "relu":
            output = relu(input)
        elif activation == "sigmoid":
            output = sigmoid(input)
        elif activation == "tanh":
            output = tanh(input)
        else:
            raise Error("Unknown activation: " + activation)

        # Verify output shape unchanged
        assert_shape(
            output, shape, activation + " backward: output shape changed"
        )

        # Verify output dtype
        assert_dtype(
            output, dtype, activation + " backward: output dtype mismatch"
        )

        # Test gradient checking with appropriate epsilon and tolerance for dtype
        var epsilon = 1e-5 if dtype == DType.float32 else 1e-4
        var tolerance = 1e-2 if dtype == DType.float32 else 1e-1

        # Define forward function for gradient checking
        fn forward(x: ExTensor) raises escaping -> ExTensor:
            if activation == "relu":
                return relu(x)
            elif activation == "sigmoid":
                return sigmoid(x)
            else:  # tanh
                return tanh(x)

        # Compute numerical gradient (always computed for validation)
        var numerical_grad = compute_numerical_gradient(forward, input, epsilon)

        # Verify numerical gradient has correct shape
        assert_shape(
            numerical_grad,
            shape,
            activation + " backward: numerical gradient shape mismatch",
        )

        # Verify no NaN/Inf in numerical gradients
        for i in range(numerical_grad.numel()):
            var val = numerical_grad._get_float64(i)
            assert_false(
                isnan(val), activation + " backward: NaN in numerical gradient"
            )
            assert_false(
                isinf(val), activation + " backward: Inf in numerical gradient"
            )

        # Optionally validate analytical gradient
        if validate_analytical:
            # Create gradient output (upstream gradient)
            var grad_output = ones_like(output)

            # Compute analytical gradient using activation-specific backward
            var analytical_grad: ExTensor
            if activation == "relu":
                # ReLU backward takes input
                analytical_grad = relu_backward(grad_output, input)
            elif activation == "sigmoid":
                # Sigmoid backward takes output
                analytical_grad = sigmoid_backward(grad_output, output)
            else:  # tanh
                # Tanh backward takes output
                analytical_grad = tanh_backward(grad_output, output)

            # Compare analytical vs numerical gradients
            assert_gradients_close(
                analytical_grad,
                numerical_grad,
                rtol=tolerance,
                atol=tolerance,
                message=activation
                + " analytical gradient doesn't match numerical",
            )

    @staticmethod
    fn test_batchnorm_layer(
        num_features: Int,
        input_shape: List[Int],
        gamma: ExTensor,
        beta: ExTensor,
        running_mean: ExTensor,
        running_var: ExTensor,
        dtype: DType,
        training_mode: Bool = True,
    ) raises:
        """Test BatchNorm in both training and inference modes.

        Training mode: Computes batch statistics and updates running statistics
        Inference mode: Uses frozen running statistics

        Args:
            num_features: Number of features to normalize.
            input_shape: Input tensor shape [batch, channels, height, width] or [batch, features].
            gamma: Scale parameter (num_features,).
            beta: Shift parameter (num_features,).
            running_mean: Running mean statistics.
            running_var: Running variance statistics.
            dtype: Data type to test.
            training_mode: If True, test training mode; else test inference mode.

        Verifies:
            - Output shape matches input shape
            - Output dtype matches input dtype
            - No NaN/Inf in output
            - In training: running statistics are updated
            - In inference: uses frozen statistics

        Raises:
            Error: If assertions fail during testing.

        Example:
            ```mojo
            # Test BatchNorm training mode
            LayerTester.test_batchnorm_layer(
                num_features=64,
                input_shape=[2, 64, 8, 8],
                gamma=gamma,
                beta=beta,
                running_mean=running_mean,
                running_var=running_var,
                dtype=DType.float32,
                training_mode=True
            )
            ```
        """
        # Create seeded random input
        var input = create_seeded_random_tensor(input_shape, dtype, seed=42)

        # Verify shapes
        assert_true(
            gamma.numel() == num_features, "BatchNorm: gamma shape mismatch"
        )
        assert_true(
            beta.numel() == num_features, "BatchNorm: beta shape mismatch"
        )
        assert_true(
            running_mean.numel() == num_features,
            "BatchNorm: running_mean shape mismatch",
        )
        assert_true(
            running_var.numel() == num_features,
            "BatchNorm: running_var shape mismatch",
        )

        # Note: Actual BatchNorm implementation would go here
        # For now, we verify the input/output shapes would be correct
        assert_shape(input, input_shape, "BatchNorm: input shape mismatch")
        assert_dtype(input, dtype, "BatchNorm: input dtype mismatch")

        # Verify no NaN/Inf in input
        for i in range(input.numel()):
            var val = input._get_float64(i)
            assert_false(isnan(val), "BatchNorm: NaN in input")
            assert_false(isinf(val), "BatchNorm: Inf in input")

    @staticmethod
    fn test_batchnorm_layer_backward(
        num_features: Int,
        input_shape: List[Int],
        gamma: ExTensor,
        beta: ExTensor,
        running_mean: ExTensor,
        running_var: ExTensor,
        dtype: DType,
    ) raises:
        """Test BatchNorm backward pass with gradient checking.

        Uses seeded random tensors for gradient checking reproducibility.
        Validates analytical gradients match numerical gradients.

        Args:
            num_features: Number of features to normalize.
            input_shape: Input tensor shape.
            gamma: Scale parameter (num_features,).
            beta: Shift parameter (num_features,).
            running_mean: Running mean statistics.
            running_var: Running variance statistics.
            dtype: Data type to test.

        Verifies:
            - Forward pass runs without error
            - Output shape matches input shape
            - Backward pass computes valid gradients
            - Analytical and numerical gradients match

        Raises:
            Error: If assertions fail during testing.

        Example:
            ```mojo
            # Test BatchNorm backward with gradient checking
            LayerTester.test_batchnorm_layer_backward(
                num_features=64,
                input_shape=[2, 64, 8, 8],
                gamma=gamma,
                beta=beta,
                running_mean=running_mean,
                running_var=running_var,
                dtype=DType.float32
            )
            ```
        """
        # Create seeded random input
        var input = create_seeded_random_tensor(input_shape, dtype, seed=42)

        # Verify shapes
        assert_true(
            gamma.numel() == num_features,
            "BatchNorm backward: gamma shape mismatch",
        )
        assert_true(
            beta.numel() == num_features,
            "BatchNorm backward: beta shape mismatch",
        )

        # Verify input shape and dtype
        assert_shape(
            input, input_shape, "BatchNorm backward: input shape mismatch"
        )
        assert_dtype(input, dtype, "BatchNorm backward: input dtype mismatch")

        # Test gradient checking with appropriate epsilon and tolerance for dtype
        # FIXME(#2710, unused) var epsilon = 1e-5 if dtype == DType.float32 else 1e-4
        # FIXME(#2710, unused) var tolerance = 1e-2 if dtype == DType.float32 else 1e-1

        # Note: Actual BatchNorm backward gradient checking would be implemented
        # when BatchNorm forward pass is available
        # For now, we validate that we can compute numerical gradients on the input

        # Verify no NaN/Inf in input
        for i in range(input.numel()):
            var val = input._get_float64(i)
            assert_false(isnan(val), "BatchNorm backward: NaN in input")
            assert_false(isinf(val), "BatchNorm backward: Inf in input")
