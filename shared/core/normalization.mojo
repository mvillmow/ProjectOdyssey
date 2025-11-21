"""Functional normalization layers.

This module provides pure functional implementations of normalization operations.
All operations are stateless - caller manages running statistics and parameters.
"""

from .extensor import ExTensor, zeros_like, ones_like, full_like
from .arithmetic import subtract, add, multiply, divide, power
from .elementwise import sqrt
from .reduction import mean as reduce_mean, sum as reduce_sum
from collections.vector import DynamicVector


fn batch_norm2d(
    x: ExTensor,
    gamma: ExTensor,
    beta: ExTensor,
    running_mean: ExTensor,
    running_var: ExTensor,
    training: Bool,
    momentum: Float64 = 0.1,
    epsilon: Float64 = 1e-5
) raises -> (ExTensor, ExTensor, ExTensor):
    """Functional 2D batch normalization.

    Normalizes activations across the batch dimension for each channel.
    Returns updated running statistics (pure functional - caller must capture).

    Args:
        x: Input tensor of shape (batch, channels, height, width)
        gamma: Scale parameter of shape (channels,)
        beta: Shift parameter of shape (channels,)
        running_mean: Running mean of shape (channels,)
        running_var: Running variance of shape (channels,)
        training: If True, use batch statistics and update running stats.
                 If False, use running statistics.
        momentum: Momentum for running statistics update (default: 0.1)
        epsilon: Small constant for numerical stability (default: 1e-5)

    Returns:
        Tuple of (output, new_running_mean, new_running_var):
            - output: Normalized tensor, shape (batch, channels, height, width)
            - new_running_mean: Updated running mean, shape (channels,)
            - new_running_var: Updated running variance, shape (channels,)

    Example:
        ```mojo
        from shared.core import batch_norm2d, zeros, ones

        var gamma = ones([channels])
        var beta = zeros([channels])
        var running_mean = zeros([channels])
        var running_var = ones([channels])

        # Training mode
        var (output, new_mean, new_var) = batch_norm2d(
            x, gamma, beta, running_mean, running_var,
            training=True, momentum=0.1
        )
        # Update running stats
        running_mean = new_mean
        running_var = new_var

        # Inference mode
        var (output, _, _) = batch_norm2d(
            x, gamma, beta, running_mean, running_var,
            training=False
        )
        ```

    Formula (training):
        mean = mean(x, axis=(0, 2, 3))  # Per channel
        var = var(x, axis=(0, 2, 3))
        x_norm = (x - mean) / sqrt(var + epsilon)
        output = gamma * x_norm + beta
        running_mean = (1 - momentum) * running_mean + momentum * mean
        running_var = (1 - momentum) * running_var + momentum * var

    Formula (inference):
        x_norm = (x - running_mean) / sqrt(running_var + epsilon)
        output = gamma * x_norm + beta

    Note:
        Pure functional: caller must capture and manage all three return values.
        Running statistics are updated only during training mode.
    """
    var x_shape = x.shape()
    if x_shape.size != 4:
        raise Error("batch_norm2d requires 4D input (batch, channels, height, width)")

    var batch = x_shape[0]
    var channels = x_shape[1]
    var height = x_shape[2]
    var width = x_shape[3]

    if training:
        # Training mode: compute batch statistics
        # Compute mean and variance per channel across (batch, height, width)

        var batch_mean = zeros([channels], x.dtype())
        var batch_var = zeros([channels], x.dtype())

        var batch_mean_ptr = batch_mean._data
        var batch_var_ptr = batch_var._data
        var x_ptr = x._data

        var spatial_size = Float32(batch * height * width)

        # Compute mean per channel
        if x.dtype() == DType.float32:
            for c in range(channels):
                var sum_val = Float32(0.0)
                for b in range(batch):
                    for h in range(height):
                        for w in range(width):
                            var idx = b * (channels * height * width) + c * (height * width) + h * width + w
                            sum_val += x_ptr.bitcast[Float32]()[idx]
                batch_mean_ptr.bitcast[Float32]()[c] = sum_val / spatial_size

            # Compute variance per channel
            for c in range(channels):
                var mean_val = batch_mean_ptr.bitcast[Float32]()[c]
                var sum_sq_diff = Float32(0.0)
                for b in range(batch):
                    for h in range(height):
                        for w in range(width):
                            var idx = b * (channels * height * width) + c * (height * width) + h * width + w
                            var diff = x_ptr.bitcast[Float32]()[idx] - mean_val
                            sum_sq_diff += diff * diff
                batch_var_ptr.bitcast[Float32]()[c] = sum_sq_diff / spatial_size

        elif x.dtype() == DType.float64:
            for c in range(channels):
                var sum_val = Float64(0.0)
                for b in range(batch):
                    for h in range(height):
                        for w in range(width):
                            var idx = b * (channels * height * width) + c * (height * width) + h * width + w
                            sum_val += x_ptr.bitcast[Float64]()[idx]
                batch_mean_ptr.bitcast[Float64]()[c] = sum_val / Float64(spatial_size)

            for c in range(channels):
                var mean_val = batch_mean_ptr.bitcast[Float64]()[c]
                var sum_sq_diff = Float64(0.0)
                for b in range(batch):
                    for h in range(height):
                        for w in range(width):
                            var idx = b * (channels * height * width) + c * (height * width) + h * width + w
                            var diff = x_ptr.bitcast[Float64]()[idx] - mean_val
                            sum_sq_diff += diff * diff
                batch_var_ptr.bitcast[Float64]()[c] = sum_sq_diff / Float64(spatial_size)
        else:
            raise Error("batch_norm2d: only float32/64 dtypes supported")

        # Normalize
        var output = zeros_like(x)
        var output_ptr = output._data
        var gamma_ptr = gamma._data
        var beta_ptr = beta._data

        if x.dtype() == DType.float32:
            for b in range(batch):
                for c in range(channels):
                    var mean_val = batch_mean_ptr.bitcast[Float32]()[c]
                    var var_val = batch_var_ptr.bitcast[Float32]()[c]
                    var std = sqrt_scalar_f32(var_val + Float32(epsilon))
                    var gamma_val = gamma_ptr.bitcast[Float32]()[c]
                    var beta_val = beta_ptr.bitcast[Float32]()[c]

                    for h in range(height):
                        for w in range(width):
                            var idx = b * (channels * height * width) + c * (height * width) + h * width + w
                            var x_val = x_ptr.bitcast[Float32]()[idx]
                            var x_norm = (x_val - mean_val) / std
                            output_ptr.bitcast[Float32]()[idx] = gamma_val * x_norm + beta_val

        elif x.dtype() == DType.float64:
            for b in range(batch):
                for c in range(channels):
                    var mean_val = batch_mean_ptr.bitcast[Float64]()[c]
                    var var_val = batch_var_ptr.bitcast[Float64]()[c]
                    var std = sqrt_scalar_f64(var_val + epsilon)
                    var gamma_val = gamma_ptr.bitcast[Float64]()[c]
                    var beta_val = beta_ptr.bitcast[Float64]()[c]

                    for h in range(height):
                        for w in range(width):
                            var idx = b * (channels * height * width) + c * (height * width) + h * width + w
                            var x_val = x_ptr.bitcast[Float64]()[idx]
                            var x_norm = (x_val - mean_val) / std
                            output_ptr.bitcast[Float64]()[idx] = gamma_val * x_norm + beta_val

        # Update running statistics
        var new_running_mean = zeros_like(running_mean)
        var new_running_var = zeros_like(running_var)
        var rm_ptr = running_mean._data
        var rv_ptr = running_var._data
        var nrm_ptr = new_running_mean._data
        var nrv_ptr = new_running_var._data

        if x.dtype() == DType.float32:
            for c in range(channels):
                var rm_val = rm_ptr.bitcast[Float32]()[c]
                var rv_val = rv_ptr.bitcast[Float32]()[c]
                var bm_val = batch_mean_ptr.bitcast[Float32]()[c]
                var bv_val = batch_var_ptr.bitcast[Float32]()[c]

                nrm_ptr.bitcast[Float32]()[c] = Float32(1.0 - momentum) * rm_val + Float32(momentum) * bm_val
                nrv_ptr.bitcast[Float32]()[c] = Float32(1.0 - momentum) * rv_val + Float32(momentum) * bv_val

        elif x.dtype() == DType.float64:
            for c in range(channels):
                var rm_val = rm_ptr.bitcast[Float64]()[c]
                var rv_val = rv_ptr.bitcast[Float64]()[c]
                var bm_val = batch_mean_ptr.bitcast[Float64]()[c]
                var bv_val = batch_var_ptr.bitcast[Float64]()[c]

                nrm_ptr.bitcast[Float64]()[c] = (1.0 - momentum) * rm_val + momentum * bm_val
                nrv_ptr.bitcast[Float64]()[c] = (1.0 - momentum) * rv_val + momentum * bv_val

        return (output, new_running_mean, new_running_var)

    else:
        # Inference mode: use running statistics
        var output = zeros_like(x)
        var output_ptr = output._data
        var x_ptr = x._data
        var gamma_ptr = gamma._data
        var beta_ptr = beta._data
        var rm_ptr = running_mean._data
        var rv_ptr = running_var._data

        if x.dtype() == DType.float32:
            for b in range(batch):
                for c in range(channels):
                    var mean_val = rm_ptr.bitcast[Float32]()[c]
                    var var_val = rv_ptr.bitcast[Float32]()[c]
                    var std = sqrt_scalar_f32(var_val + Float32(epsilon))
                    var gamma_val = gamma_ptr.bitcast[Float32]()[c]
                    var beta_val = beta_ptr.bitcast[Float32]()[c]

                    for h in range(height):
                        for w in range(width):
                            var idx = b * (channels * height * width) + c * (height * width) + h * width + w
                            var x_val = x_ptr.bitcast[Float32]()[idx]
                            var x_norm = (x_val - mean_val) / std
                            output_ptr.bitcast[Float32]()[idx] = gamma_val * x_norm + beta_val

        elif x.dtype() == DType.float64:
            for b in range(batch):
                for c in range(channels):
                    var mean_val = rm_ptr.bitcast[Float64]()[c]
                    var var_val = rv_ptr.bitcast[Float64]()[c]
                    var std = sqrt_scalar_f64(var_val + epsilon)
                    var gamma_val = gamma_ptr.bitcast[Float64]()[c]
                    var beta_val = beta_ptr.bitcast[Float64]()[c]

                    for h in range(height):
                        for w in range(width):
                            var idx = b * (channels * height * width) + c * (height * width) + h * width + w
                            var x_val = x_ptr.bitcast[Float64]()[idx]
                            var x_norm = (x_val - mean_val) / std
                            output_ptr.bitcast[Float64]()[idx] = gamma_val * x_norm + beta_val

        # Running stats unchanged in inference mode
        return (output, running_mean, running_var)


# Helper functions for scalar sqrt
fn sqrt_scalar_f32(x: Float32) -> Float32:
    """Compute sqrt of a scalar float32."""
    return x ** 0.5


fn sqrt_scalar_f64(x: Float64) -> Float64:
    """Compute sqrt of a scalar float64."""
    return x ** 0.5


fn normalize_rgb(
    images: ExTensor,
    mean: (Float32, Float32, Float32),
    std: (Float32, Float32, Float32)
) raises -> ExTensor:
    """Normalize RGB images with per-channel mean and standard deviation.

    Applies normalization to RGB images using the formula:
        normalized = (pixel / 255.0 - mean) / std

    This is commonly used with ImageNet statistics for transfer learning.

    Args:
        images: Input uint8 tensor of shape (N, 3, H, W)
        mean: Mean values for R, G, B channels (e.g., (0.485, 0.456, 0.406))
        std: Std deviation values for R, G, B channels (e.g., (0.229, 0.224, 0.225))

    Returns:
        Normalized float32 tensor of shape (N, 3, H, W)

    Example:
        ```mojo
        from shared.core import normalize_rgb

        # ImageNet normalization
        var mean = (Float32(0.485), Float32(0.456), Float32(0.406))
        var std = (Float32(0.229), Float32(0.224), Float32(0.225))
        var normalized = normalize_rgb(images, mean, std)
        ```

    Note:
        - Input must be uint8 [0, 255]
        - Output is float32 with normalized values
        - Common for CIFAR-10, ImageNet, and other RGB datasets
    """
    var shape = images.shape()
    if shape.size != 4:
        raise Error("normalize_rgb requires 4D input (N, 3, H, W)")

    var num_images = shape[0]
    var num_channels = shape[1]
    var num_rows = shape[2]
    var num_cols = shape[3]

    if num_channels != 3:
        raise Error("normalize_rgb requires 3 RGB channels, got: " + str(num_channels))

    # Create output tensor (float32)
    var normalized = zeros(shape, DType.float32)

    var src_data = images._data
    var dst_data = normalized._data.bitcast[Float32]()

    # Extract mean and std values
    var mean_r = mean.get[0, Float32]()
    var mean_g = mean.get[1, Float32]()
    var mean_b = mean.get[2, Float32]()
    var std_r = std.get[0, Float32]()
    var std_g = std.get[1, Float32]()
    var std_b = std.get[2, Float32]()

    # Normalize each image
    for n in range(num_images):
        for h in range(num_rows):
            for w in range(num_cols):
                # R channel (c=0)
                var idx_r = n * (num_channels * num_rows * num_cols) + 0 * (num_rows * num_cols) + h * num_cols + w
                var pixel_r = Float32(src_data[idx_r]) / 255.0
                dst_data[idx_r] = (pixel_r - mean_r) / std_r

                # G channel (c=1)
                var idx_g = n * (num_channels * num_rows * num_cols) + 1 * (num_rows * num_cols) + h * num_cols + w
                var pixel_g = Float32(src_data[idx_g]) / 255.0
                dst_data[idx_g] = (pixel_g - mean_g) / std_g

                # B channel (c=2)
                var idx_b = n * (num_channels * num_rows * num_cols) + 2 * (num_rows * num_cols) + h * num_cols + w
                var pixel_b = Float32(src_data[idx_b]) / 255.0
                dst_data[idx_b] = (pixel_b - mean_b) / std_b

    return normalized


fn batch_norm2d_backward(
    grad_output: ExTensor,
    x: ExTensor,
    gamma: ExTensor,
    running_mean: ExTensor,
    running_var: ExTensor,
    training: Bool,
    epsilon: Float64 = 1e-5
) raises -> (ExTensor, ExTensor, ExTensor):
    """Backward pass for 2D batch normalization.

    Computes gradients with respect to input, gamma, and beta parameters.

    Args:
        grad_output: Gradient w.r.t. output (batch, channels, height, width)
        x: Original input tensor (batch, channels, height, width)
        gamma: Scale parameter (channels,)
        running_mean: Running mean (channels,) - used in inference mode
        running_var: Running variance (channels,) - used in inference mode
        training: Whether in training mode (affects gradient computation)
        epsilon: Small constant for numerical stability (default: 1e-5)

    Returns:
        Tuple of (grad_input, grad_gamma, grad_beta):
            - grad_input: Gradient w.r.t. input (batch, channels, height, width)
            - grad_gamma: Gradient w.r.t. gamma (channels,)
            - grad_beta: Gradient w.r.t. beta (channels,)

    Example:
        ```mojo
        from shared.core import batch_norm2d_backward

        # Forward pass (save x for backward)
        var (output, new_mean, new_var) = batch_norm2d(
            x, gamma, beta, running_mean, running_var, training=True
        )

        # ... compute loss and grad_output ...

        # Backward pass
        var (grad_x, grad_gamma, grad_beta) = batch_norm2d_backward(
            grad_output, x, gamma, running_mean, running_var, training=True
        )
        ```

    Mathematical Formulation (Training Mode):

        Forward pass computes:
            mean = E[x] over (batch, height, width) per channel
            var = Var[x] over (batch, height, width) per channel
            x_norm = (x - mean) / sqrt(var + eps)
            y = gamma * x_norm + beta

        Backward pass (chain rule):
            grad_beta = sum(grad_output) over (batch, height, width) per channel
            grad_gamma = sum(grad_output * x_norm) over (batch, height, width) per channel

            grad_x_norm = grad_output * gamma
            grad_var = sum(grad_x_norm * (x - mean) * -0.5 * (var + eps)^(-3/2))
            grad_mean = sum(grad_x_norm * -1/sqrt(var + eps)) +
                        grad_var * mean(-2(x - mean))

            grad_input = grad_x_norm / sqrt(var + eps) +
                         grad_var * 2(x - mean) / N +
                         grad_mean / N

        where N = batch * height * width (spatial size)

    Mathematical Formulation (Inference Mode):

        Forward pass uses fixed running statistics:
            x_norm = (x - running_mean) / sqrt(running_var + eps)
            y = gamma * x_norm + beta

        Backward pass (simpler):
            grad_beta = sum(grad_output)
            grad_gamma = sum(grad_output * x_norm)
            grad_input = grad_output * gamma / sqrt(running_var + eps)

    References:
        - Ioffe & Szegedy (2015). Batch Normalization: Accelerating Deep Network
          Training by Reducing Internal Covariate Shift. ICML 2015.
          https://arxiv.org/abs/1502.03167

        - Gradient derivation:
          https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html

    Note:
        Pure functional: returns new tensors, does not modify inputs.
        Training mode requires computing batch statistics from x.
        Inference mode uses precomputed running statistics.
    """
    var x_shape = x.shape()
    var grad_shape = grad_output.shape()

    if x_shape.size != 4 or grad_shape.size != 4:
        raise Error("batch_norm2d_backward requires 4D inputs (batch, channels, height, width)")

    var batch = x_shape[0]
    var channels = x_shape[1]
    var height = x_shape[2]
    var width = x_shape[3]

    var spatial_size = batch * height * width

    # Initialize output gradients
    var grad_input = zeros_like(x)
    var grad_gamma = zeros([channels], x.dtype())
    var grad_beta = zeros([channels], x.dtype())

    var grad_output_ptr = grad_output._data
    var x_ptr = x._data
    var gamma_ptr = gamma._data
    var grad_input_ptr = grad_input._data
    var grad_gamma_ptr = grad_gamma._data
    var grad_beta_ptr = grad_beta._data

    if training:
        # ========== TRAINING MODE: Complex gradients through batch statistics ==========

        # Step 1: Compute batch statistics (mean and variance) per channel
        var batch_mean = zeros([channels], x.dtype())
        var batch_var = zeros([channels], x.dtype())
        var batch_mean_ptr = batch_mean._data
        var batch_var_ptr = batch_var._data

        if x.dtype() == DType.float32:
            var N = Float32(spatial_size)

            # Compute mean per channel
            for c in range(channels):
                var sum_val = Float32(0.0)
                for b in range(batch):
                    for h in range(height):
                        for w in range(width):
                            var idx = b * (channels * height * width) + c * (height * width) + h * width + w
                            sum_val += x_ptr.bitcast[Float32]()[idx]
                batch_mean_ptr.bitcast[Float32]()[c] = sum_val / N

            # Compute variance per channel
            for c in range(channels):
                var mean_val = batch_mean_ptr.bitcast[Float32]()[c]
                var sum_sq_diff = Float32(0.0)
                for b in range(batch):
                    for h in range(height):
                        for w in range(width):
                            var idx = b * (channels * height * width) + c * (height * width) + h * width + w
                            var diff = x_ptr.bitcast[Float32]()[idx] - mean_val
                            sum_sq_diff += diff * diff
                batch_var_ptr.bitcast[Float32]()[c] = sum_sq_diff / N

            # Step 2: Compute grad_beta and grad_gamma
            for c in range(channels):
                var mean_val = batch_mean_ptr.bitcast[Float32]()[c]
                var var_val = batch_var_ptr.bitcast[Float32]()[c]
                var std = sqrt_scalar_f32(var_val + Float32(epsilon))

                var sum_grad_output = Float32(0.0)
                var sum_grad_output_x_norm = Float32(0.0)

                for b in range(batch):
                    for h in range(height):
                        for w in range(width):
                            var idx = b * (channels * height * width) + c * (height * width) + h * width + w
                            var grad_out = grad_output_ptr.bitcast[Float32]()[idx]
                            var x_val = x_ptr.bitcast[Float32]()[idx]
                            var x_norm = (x_val - mean_val) / std

                            sum_grad_output += grad_out
                            sum_grad_output_x_norm += grad_out * x_norm

                grad_beta_ptr.bitcast[Float32]()[c] = sum_grad_output
                grad_gamma_ptr.bitcast[Float32]()[c] = sum_grad_output_x_norm

            # Step 3: Compute grad_input using chain rule through normalization
            for c in range(channels):
                var mean_val = batch_mean_ptr.bitcast[Float32]()[c]
                var var_val = batch_var_ptr.bitcast[Float32]()[c]
                var std = sqrt_scalar_f32(var_val + Float32(epsilon))
                var gamma_val = gamma_ptr.bitcast[Float32]()[c]

                # Compute grad_var and grad_mean
                var grad_var = Float32(0.0)
                var grad_mean = Float32(0.0)

                for b in range(batch):
                    for h in range(height):
                        for w in range(width):
                            var idx = b * (channels * height * width) + c * (height * width) + h * width + w
                            var grad_out = grad_output_ptr.bitcast[Float32]()[idx]
                            var x_val = x_ptr.bitcast[Float32]()[idx]

                            var grad_x_norm = grad_out * gamma_val
                            grad_var += grad_x_norm * (x_val - mean_val) * Float32(-0.5) * pow_scalar_f32(var_val + Float32(epsilon), Float32(-1.5))
                            grad_mean += grad_x_norm * Float32(-1.0) / std

                # Add contribution from grad_var to grad_mean
                grad_mean += grad_var * (Float32(-2.0) * Float32(0.0))  # mean(x - mean) = 0

                # Compute grad_input
                for b in range(batch):
                    for h in range(height):
                        for w in range(width):
                            var idx = b * (channels * height * width) + c * (height * width) + h * width + w
                            var grad_out = grad_output_ptr.bitcast[Float32]()[idx]
                            var x_val = x_ptr.bitcast[Float32]()[idx]

                            var grad_x_norm = grad_out * gamma_val
                            var term1 = grad_x_norm / std
                            var term2 = grad_var * Float32(2.0) * (x_val - mean_val) / N
                            var term3 = grad_mean / N

                            grad_input_ptr.bitcast[Float32]()[idx] = term1 + term2 + term3

        elif x.dtype() == DType.float64:
            var N = Float64(spatial_size)

            # Compute mean per channel
            for c in range(channels):
                var sum_val = Float64(0.0)
                for b in range(batch):
                    for h in range(height):
                        for w in range(width):
                            var idx = b * (channels * height * width) + c * (height * width) + h * width + w
                            sum_val += x_ptr.bitcast[Float64]()[idx]
                batch_mean_ptr.bitcast[Float64]()[c] = sum_val / N

            # Compute variance per channel
            for c in range(channels):
                var mean_val = batch_mean_ptr.bitcast[Float64]()[c]
                var sum_sq_diff = Float64(0.0)
                for b in range(batch):
                    for h in range(height):
                        for w in range(width):
                            var idx = b * (channels * height * width) + c * (height * width) + h * width + w
                            var diff = x_ptr.bitcast[Float64]()[idx] - mean_val
                            sum_sq_diff += diff * diff
                batch_var_ptr.bitcast[Float64]()[c] = sum_sq_diff / N

            # Step 2: Compute grad_beta and grad_gamma
            for c in range(channels):
                var mean_val = batch_mean_ptr.bitcast[Float64]()[c]
                var var_val = batch_var_ptr.bitcast[Float64]()[c]
                var std = sqrt_scalar_f64(var_val + epsilon)

                var sum_grad_output = Float64(0.0)
                var sum_grad_output_x_norm = Float64(0.0)

                for b in range(batch):
                    for h in range(height):
                        for w in range(width):
                            var idx = b * (channels * height * width) + c * (height * width) + h * width + w
                            var grad_out = grad_output_ptr.bitcast[Float64]()[idx]
                            var x_val = x_ptr.bitcast[Float64]()[idx]
                            var x_norm = (x_val - mean_val) / std

                            sum_grad_output += grad_out
                            sum_grad_output_x_norm += grad_out * x_norm

                grad_beta_ptr.bitcast[Float64]()[c] = sum_grad_output
                grad_gamma_ptr.bitcast[Float64]()[c] = sum_grad_output_x_norm

            # Step 3: Compute grad_input using chain rule through normalization
            for c in range(channels):
                var mean_val = batch_mean_ptr.bitcast[Float64]()[c]
                var var_val = batch_var_ptr.bitcast[Float64]()[c]
                var std = sqrt_scalar_f64(var_val + epsilon)
                var gamma_val = gamma_ptr.bitcast[Float64]()[c]

                # Compute grad_var and grad_mean
                var grad_var = Float64(0.0)
                var grad_mean = Float64(0.0)

                for b in range(batch):
                    for h in range(height):
                        for w in range(width):
                            var idx = b * (channels * height * width) + c * (height * width) + h * width + w
                            var grad_out = grad_output_ptr.bitcast[Float64]()[idx]
                            var x_val = x_ptr.bitcast[Float64]()[idx]

                            var grad_x_norm = grad_out * gamma_val
                            grad_var += grad_x_norm * (x_val - mean_val) * Float64(-0.5) * pow_scalar_f64(var_val + epsilon, Float64(-1.5))
                            grad_mean += grad_x_norm * Float64(-1.0) / std

                # Add contribution from grad_var to grad_mean (mean(x - mean) = 0, so no contribution)
                grad_mean += grad_var * (Float64(-2.0) * Float64(0.0))

                # Compute grad_input
                for b in range(batch):
                    for h in range(height):
                        for w in range(width):
                            var idx = b * (channels * height * width) + c * (height * width) + h * width + w
                            var grad_out = grad_output_ptr.bitcast[Float64]()[idx]
                            var x_val = x_ptr.bitcast[Float64]()[idx]

                            var grad_x_norm = grad_out * gamma_val
                            var term1 = grad_x_norm / std
                            var term2 = grad_var * Float64(2.0) * (x_val - mean_val) / N
                            var term3 = grad_mean / N

                            grad_input_ptr.bitcast[Float64]()[idx] = term1 + term2 + term3

        else:
            raise Error("batch_norm2d_backward: only float32/64 dtypes supported")

    else:
        # ========== INFERENCE MODE: Simpler gradients using running statistics ==========

        var rm_ptr = running_mean._data
        var rv_ptr = running_var._data

        if x.dtype() == DType.float32:
            # Compute grad_beta and grad_gamma
            for c in range(channels):
                var mean_val = rm_ptr.bitcast[Float32]()[c]
                var var_val = rv_ptr.bitcast[Float32]()[c]
                var std = sqrt_scalar_f32(var_val + Float32(epsilon))

                var sum_grad_output = Float32(0.0)
                var sum_grad_output_x_norm = Float32(0.0)

                for b in range(batch):
                    for h in range(height):
                        for w in range(width):
                            var idx = b * (channels * height * width) + c * (height * width) + h * width + w
                            var grad_out = grad_output_ptr.bitcast[Float32]()[idx]
                            var x_val = x_ptr.bitcast[Float32]()[idx]
                            var x_norm = (x_val - mean_val) / std

                            sum_grad_output += grad_out
                            sum_grad_output_x_norm += grad_out * x_norm

                grad_beta_ptr.bitcast[Float32]()[c] = sum_grad_output
                grad_gamma_ptr.bitcast[Float32]()[c] = sum_grad_output_x_norm

            # Compute grad_input (simple rescaling)
            for b in range(batch):
                for c in range(channels):
                    var var_val = rv_ptr.bitcast[Float32]()[c]
                    var std = sqrt_scalar_f32(var_val + Float32(epsilon))
                    var gamma_val = gamma_ptr.bitcast[Float32]()[c]

                    for h in range(height):
                        for w in range(width):
                            var idx = b * (channels * height * width) + c * (height * width) + h * width + w
                            var grad_out = grad_output_ptr.bitcast[Float32]()[idx]
                            grad_input_ptr.bitcast[Float32]()[idx] = grad_out * gamma_val / std

        elif x.dtype() == DType.float64:
            # Compute grad_beta and grad_gamma
            for c in range(channels):
                var mean_val = rm_ptr.bitcast[Float64]()[c]
                var var_val = rv_ptr.bitcast[Float64]()[c]
                var std = sqrt_scalar_f64(var_val + epsilon)

                var sum_grad_output = Float64(0.0)
                var sum_grad_output_x_norm = Float64(0.0)

                for b in range(batch):
                    for h in range(height):
                        for w in range(width):
                            var idx = b * (channels * height * width) + c * (height * width) + h * width + w
                            var grad_out = grad_output_ptr.bitcast[Float64]()[idx]
                            var x_val = x_ptr.bitcast[Float64]()[idx]
                            var x_norm = (x_val - mean_val) / std

                            sum_grad_output += grad_out
                            sum_grad_output_x_norm += grad_out * x_norm

                grad_beta_ptr.bitcast[Float64]()[c] = sum_grad_output
                grad_gamma_ptr.bitcast[Float64]()[c] = sum_grad_output_x_norm

            # Compute grad_input (simple rescaling)
            for b in range(batch):
                for c in range(channels):
                    var var_val = rv_ptr.bitcast[Float64]()[c]
                    var std = sqrt_scalar_f64(var_val + epsilon)
                    var gamma_val = gamma_ptr.bitcast[Float64]()[c]

                    for h in range(height):
                        for w in range(width):
                            var idx = b * (channels * height * width) + c * (height * width) + h * width + w
                            var grad_out = grad_output_ptr.bitcast[Float64]()[idx]
                            grad_input_ptr.bitcast[Float64]()[idx] = grad_out * gamma_val / std

        else:
            raise Error("batch_norm2d_backward: only float32/64 dtypes supported")

    return (grad_input, grad_gamma, grad_beta)


# Helper function for scalar power
fn pow_scalar_f32(x: Float32, y: Float32) -> Float32:
    """Compute x^y for scalar float32."""
    return x ** y


fn pow_scalar_f64(x: Float64, y: Float64) -> Float64:
    """Compute x^y for scalar float64."""
    return x ** y


fn layer_norm(
    x: ExTensor,
    gamma: ExTensor,
    beta: ExTensor,
    epsilon: Float64 = 1e-5
) raises -> ExTensor:
    """Functional layer normalization.

    Normalizes activations across the feature dimension for each sample.
    Unlike batch norm, this doesn't require running statistics.

    Args:
        x: Input tensor of shape (batch, features) or (batch, channels, height, width)
        gamma: Scale parameter of shape matching last dim(s)
        beta: Shift parameter of shape matching last dim(s)
        epsilon: Small constant for numerical stability (default: 1e-5)

    Returns:
        Normalized tensor, same shape as input

    Example:
        ```mojo
        from shared.core import layer_norm, zeros, ones

        # For 2D input (batch, features)
        var gamma = ones([features])
        var beta = zeros([features])
        var output = layer_norm(x, gamma, beta)

        # For 4D input (batch, channels, height, width)
        var gamma = ones([channels, height, width])
        var beta = zeros([channels, height, width])
        var output = layer_norm(x, gamma, beta)
        ```

    Formula:
        For each sample:
            mean = mean(x[i])  # Over all features
            var = var(x[i])
            x_norm[i] = (x[i] - mean) / sqrt(var + epsilon)
            output[i] = gamma * x_norm[i] + beta

    Note:
        - No running statistics needed (stateless)
        - Normalizes each sample independently
        - Commonly used in transformers and RNNs
    """
    var x_shape = x.shape()

    if x_shape.size == 2:
        # 2D input: (batch, features)
        var batch = x_shape[0]
        var features = x_shape[1]

        var output = zeros_like(x)
        var x_ptr = x._data
        var output_ptr = output._data
        var gamma_ptr = gamma._data
        var beta_ptr = beta._data

        if x.dtype() == DType.float32:
            for b in range(batch):
                # Compute mean for this sample
                var sum_val = Float32(0.0)
                for f in range(features):
                    var idx = b * features + f
                    sum_val += x_ptr.bitcast[Float32]()[idx]
                var mean_val = sum_val / Float32(features)

                # Compute variance
                var sum_sq_diff = Float32(0.0)
                for f in range(features):
                    var idx = b * features + f
                    var diff = x_ptr.bitcast[Float32]()[idx] - mean_val
                    sum_sq_diff += diff * diff
                var var_val = sum_sq_diff / Float32(features)
                var std = sqrt_scalar_f32(var_val + Float32(epsilon))

                # Normalize and scale
                for f in range(features):
                    var idx = b * features + f
                    var x_val = x_ptr.bitcast[Float32]()[idx]
                    var x_norm = (x_val - mean_val) / std
                    var gamma_val = gamma_ptr.bitcast[Float32]()[f]
                    var beta_val = beta_ptr.bitcast[Float32]()[f]
                    output_ptr.bitcast[Float32]()[idx] = gamma_val * x_norm + beta_val

        elif x.dtype() == DType.float64:
            for b in range(batch):
                var sum_val = Float64(0.0)
                for f in range(features):
                    var idx = b * features + f
                    sum_val += x_ptr.bitcast[Float64]()[idx]
                var mean_val = sum_val / Float64(features)

                var sum_sq_diff = Float64(0.0)
                for f in range(features):
                    var idx = b * features + f
                    var diff = x_ptr.bitcast[Float64]()[idx] - mean_val
                    sum_sq_diff += diff * diff
                var var_val = sum_sq_diff / Float64(features)
                var std = sqrt_scalar_f64(var_val + epsilon)

                for f in range(features):
                    var idx = b * features + f
                    var x_val = x_ptr.bitcast[Float64]()[idx]
                    var x_norm = (x_val - mean_val) / std
                    var gamma_val = gamma_ptr.bitcast[Float64]()[f]
                    var beta_val = beta_ptr.bitcast[Float64]()[f]
                    output_ptr.bitcast[Float64]()[idx] = gamma_val * x_norm + beta_val
        else:
            raise Error("layer_norm: only float32/64 dtypes supported")

        return output

    elif x_shape.size == 4:
        # 4D input: (batch, channels, height, width)
        # Normalize over (channels, height, width) for each sample
        var batch = x_shape[0]
        var channels = x_shape[1]
        var height = x_shape[2]
        var width = x_shape[3]
        var feature_size = channels * height * width

        var output = zeros_like(x)
        var x_ptr = x._data
        var output_ptr = output._data
        var gamma_ptr = gamma._data
        var beta_ptr = beta._data

        if x.dtype() == DType.float32:
            for b in range(batch):
                # Compute mean for this sample
                var sum_val = Float32(0.0)
                for c in range(channels):
                    for h in range(height):
                        for w in range(width):
                            var idx = b * (channels * height * width) + c * (height * width) + h * width + w
                            sum_val += x_ptr.bitcast[Float32]()[idx]
                var mean_val = sum_val / Float32(feature_size)

                # Compute variance
                var sum_sq_diff = Float32(0.0)
                for c in range(channels):
                    for h in range(height):
                        for w in range(width):
                            var idx = b * (channels * height * width) + c * (height * width) + h * width + w
                            var diff = x_ptr.bitcast[Float32]()[idx] - mean_val
                            sum_sq_diff += diff * diff
                var var_val = sum_sq_diff / Float32(feature_size)
                var std = sqrt_scalar_f32(var_val + Float32(epsilon))

                # Normalize and scale
                for c in range(channels):
                    for h in range(height):
                        for w in range(width):
                            var idx = b * (channels * height * width) + c * (height * width) + h * width + w
                            var gamma_idx = c * (height * width) + h * width + w
                            var x_val = x_ptr.bitcast[Float32]()[idx]
                            var x_norm = (x_val - mean_val) / std
                            var gamma_val = gamma_ptr.bitcast[Float32]()[gamma_idx]
                            var beta_val = beta_ptr.bitcast[Float32]()[gamma_idx]
                            output_ptr.bitcast[Float32]()[idx] = gamma_val * x_norm + beta_val

        elif x.dtype() == DType.float64:
            for b in range(batch):
                var sum_val = Float64(0.0)
                for c in range(channels):
                    for h in range(height):
                        for w in range(width):
                            var idx = b * (channels * height * width) + c * (height * width) + h * width + w
                            sum_val += x_ptr.bitcast[Float64]()[idx]
                var mean_val = sum_val / Float64(feature_size)

                var sum_sq_diff = Float64(0.0)
                for c in range(channels):
                    for h in range(height):
                        for w in range(width):
                            var idx = b * (channels * height * width) + c * (height * width) + h * width + w
                            var diff = x_ptr.bitcast[Float64]()[idx] - mean_val
                            sum_sq_diff += diff * diff
                var var_val = sum_sq_diff / Float64(feature_size)
                var std = sqrt_scalar_f64(var_val + epsilon)

                for c in range(channels):
                    for h in range(height):
                        for w in range(width):
                            var idx = b * (channels * height * width) + c * (height * width) + h * width + w
                            var gamma_idx = c * (height * width) + h * width + w
                            var x_val = x_ptr.bitcast[Float64]()[idx]
                            var x_norm = (x_val - mean_val) / std
                            var gamma_val = gamma_ptr.bitcast[Float64]()[gamma_idx]
                            var beta_val = beta_ptr.bitcast[Float64]()[gamma_idx]
                            output_ptr.bitcast[Float64]()[idx] = gamma_val * x_norm + beta_val
        else:
            raise Error("layer_norm: only float32/64 dtypes supported")

        return output

    else:
        raise Error("layer_norm supports 2D or 4D inputs only")
