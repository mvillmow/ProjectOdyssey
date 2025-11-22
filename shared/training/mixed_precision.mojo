"""Mixed precision training utilities for FP16 and BF16.

This module provides infrastructure for training neural networks with reduced
precision (FP16, BF16) while maintaining numerical stability through gradient
scaling, loss scaling, and master weight management.

Key Features:
- Gradient scaling to prevent underflow in FP16
- Dynamic loss scaling with automatic adjustment
- Master weights in FP32 for optimizer state
- Numerical safety checks for NaN/Inf detection

Usage:
    # Create gradient scaler
    var scaler = GradientScaler(initial_scale=65536.0)

    # In training loop
    var scaled_loss = scaler.scale(loss)
    # Compute gradients...
    var unscaled_grads = scaler.unscale(gradients)
    scaler.step()  # Update scale factor

See mixed precision training examples in examples/mixed_precision/
"""

from ..core.extensor import ExTensor
from ..core.numerical_safety import has_nan, has_inf
from math import log2, pow


struct GradientScaler:
    """Manages gradient scaling for mixed precision training.

    Prevents gradient underflow in FP16 by scaling loss and gradients.
    Dynamically adjusts scale factor based on gradient overflow/underflow.

    Attributes:
        scale: Current loss scale factor (default: 65536.0 = 2^16)
        growth_factor: Factor to increase scale (default: 2.0)
        backoff_factor: Factor to decrease scale (default: 0.5)
        growth_interval: Steps between scale increases (default: 2000)
        min_scale: Minimum allowed scale (default: 1.0)
        max_scale: Maximum allowed scale (default: 65536.0)

    Example:
        var scaler = GradientScaler()

        # In training loop:
        var scaled_loss = scaler.scale(loss)
        # Backward pass computes scaled gradients
        var grads = compute_gradients(scaled_loss)
        var unscaled_grads = scaler.unscale(grads)

        if not has_nan(unscaled_grads) and not has_inf(unscaled_grads):
            # Apply optimizer step with unscaled gradients
            optimizer_step(unscaled_grads)
            scaler.step()  # Successful step, update scale
        else:
            # Skip step and reduce scale
            scaler.backoff()
    """
    var scale: Float32
    var growth_factor: Float32
    var backoff_factor: Float32
    var growth_interval: Int
    var _steps_since_growth: Int
    var _num_steps: Int
    var min_scale: Float32
    var max_scale: Float32

    fn __init__(inout self,
                initial_scale: Float32 = 65536.0,
                growth_factor: Float32 = 2.0,
                backoff_factor: Float32 = 0.5,
                growth_interval: Int = 2000,
                min_scale: Float32 = 1.0,
                max_scale: Float32 = 65536.0):
        """Initialize gradient scaler.

        Args:
            initial_scale: Initial loss scale (power of 2 recommended)
            growth_factor: Multiplicative factor for scale growth
            backoff_factor: Multiplicative factor for scale reduction
            growth_interval: Steps between automatic scale increases
            min_scale: Minimum allowed scale factor
            max_scale: Maximum allowed scale factor
        """
        self.scale = initial_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self._steps_since_growth = 0
        self._num_steps = 0
        self.min_scale = min_scale
        self.max_scale = max_scale

    fn scale_loss(self, loss: ExTensor) raises -> ExTensor:
        """Scale loss by current scale factor.

        Args:
            loss: Unscaled loss tensor (typically scalar)

        Returns:
            Scaled loss tensor

        Example:
            var loss = compute_loss(predictions, targets)
            var scaled_loss = scaler.scale_loss(loss)
        """
        # Create a scalar tensor with the scale value and multiply
        var scale_tensor = ExTensor.full(loss.shape(), Float64(self.scale), loss.dtype())
        return loss * scale_tensor

    fn unscale_gradients(self, gradients: ExTensor) raises -> ExTensor:
        """Unscale gradients by dividing by scale factor.

        Args:
            gradients: Scaled gradients from backward pass

        Returns:
            Unscaled gradients ready for optimizer

        Example:
            var scaled_grads = backward(scaled_loss)
            var grads = scaler.unscale_gradients(scaled_grads)
        """
        # Create a scalar tensor with the scale value and divide
        var scale_tensor = ExTensor.full(gradients.shape(), Float64(self.scale), gradients.dtype())
        return gradients / scale_tensor

    fn step(inout self):
        """Update scale factor after successful optimizer step.

        Increases scale if no overflow occurred for growth_interval steps.
        Should be called after each successful optimizer update.

        Example:
            if optimizer_step_successful:
                scaler.step()
        """
        self._num_steps += 1
        self._steps_since_growth += 1

        # Increase scale if growth interval reached
        if self._steps_since_growth >= self.growth_interval:
            var new_scale = self.scale * self.growth_factor
            if new_scale <= self.max_scale:
                self.scale = new_scale
            self._steps_since_growth = 0

    fn backoff(inout self):
        """Reduce scale factor after gradient overflow.

        Called when NaN or Inf detected in gradients.
        Reduces scale and resets growth counter.

        Example:
            if has_nan(gradients) or has_inf(gradients):
                scaler.backoff()
                continue  # Skip optimizer step
        """
        var new_scale = self.scale * self.backoff_factor
        if new_scale >= self.min_scale:
            self.scale = new_scale
        self._steps_since_growth = 0

    fn get_scale(self) -> Float32:
        """Get current scale factor.

        Returns:
            Current loss scale value
        """
        return self.scale

    fn get_num_steps(self) -> Int:
        """Get total number of steps taken.

        Returns:
            Number of successful optimizer steps
        """
        return self._num_steps


fn convert_to_fp32_master(params: ExTensor) raises -> ExTensor:
    """Convert model parameters to FP32 master weights.

    Creates FP32 copy of parameters for optimizer state management.
    Use when training with FP16/BF16 but need FP32 precision for updates.

    Args:
        params: Model parameters (any dtype)

    Returns:
        FP32 copy of parameters

    Example:
        # Model params in FP16
        var fp16_params = ExTensor.zeros((1000, 1000), DType.float16)

        # Create FP32 master weights for optimizer
        var master_params = convert_to_fp32_master(fp16_params)
    """
    # Create FP32 tensor with same shape
    var result = ExTensor(params.shape(), DType.float32)

    # Copy and convert each element
    var size = params._numel
    for i in range(size):
        var val = params._get_float64(i)
        result._set_float64(i, val)

    return result


fn update_model_from_master(inout model_params: ExTensor,
                            master_params: ExTensor) raises:
    """Update model parameters from FP32 master weights.

    Copies FP32 master weights back to model parameters with dtype conversion.
    Call after optimizer updates master weights.

    Args:
        model_params: Model parameters to update (FP16/BF16)
        master_params: Updated master weights (FP32)

    Example:
        # Optimizer updates master weights in FP32
        optimizer_step(master_params, gradients)

        # Copy back to FP16 model params
        update_model_from_master(fp16_params, master_params)
    """
    # Copy and convert each element in-place
    var size = model_params._numel
    for i in range(size):
        var val = master_params._get_float64(i)
        model_params._set_float64(i, val)


fn check_gradients_finite(gradients: ExTensor) raises -> Bool:
    """Check if gradients contain only finite values.

    Returns True if gradients are all finite (no NaN or Inf).
    Use to validate gradients before optimizer step.

    Args:
        gradients: Gradient tensor to check

    Returns:
        True if all gradients are finite, False otherwise

    Example:
        if check_gradients_finite(grads):
            optimizer_step(grads)
        else:
            scaler.backoff()
            continue  # Skip this step
    """
    return not (has_nan(gradients) or has_inf(gradients))


fn clip_gradients_by_norm(gradients: ExTensor, max_norm: Float32) raises -> ExTensor:
    """Clip gradients by global norm.

    Scales gradients if their L2 norm exceeds max_norm.
    Useful for preventing gradient explosion in mixed precision.

    Args:
        gradients: Gradient tensor
        max_norm: Maximum allowed gradient norm

    Returns:
        Clipped gradients

    Example:
        # Clip to prevent explosion in FP16
        var clipped_grads = clip_gradients_by_norm(grads, 1.0)
    """
    from ..core.reduction import sum as tensor_sum
    from math import sqrt as math_sqrt

    # Compute L2 norm: sqrt(sum(grad^2))
    var grad_squared = gradients * gradients
    var sum_squared = tensor_sum(grad_squared)

    # Get scalar value from tensor
    var sum_val = sum_squared.item()
    var grad_norm = math_sqrt(sum_val)

    # Clip if norm exceeds max_norm
    if grad_norm > Float64(max_norm):
        var scale_factor = Float64(max_norm) / grad_norm
        var scale_tensor = ExTensor.full(gradients.shape(), scale_factor, gradients.dtype())
        return gradients * scale_tensor
    else:
        return gradients


fn clip_gradients_by_value(gradients: ExTensor,
                           min_value: Float32,
                           max_value: Float32) raises -> ExTensor:
    """Clip gradients by value range.

    Clamps each gradient value to [min_value, max_value].
    Simpler than norm clipping but less theoretically motivated.

    Args:
        gradients: Gradient tensor
        min_value: Minimum allowed gradient value
        max_value: Maximum allowed gradient value

    Returns:
        Clipped gradients

    Example:
        # Clip each gradient to [-1, 1]
        var clipped = clip_gradients_by_value(grads, -1.0, 1.0)
    """
    # Clamp each element to [min_value, max_value]
    var result = ExTensor(gradients.shape(), gradients.dtype())
    var size = gradients._numel

    for i in range(size):
        var val = gradients._get_float64(i)
        if val < Float64(min_value):
            result._set_float64(i, Float64(min_value))
        elif val > Float64(max_value):
            result._set_float64(i, Float64(max_value))
        else:
            result._set_float64(i, val)

    return result
