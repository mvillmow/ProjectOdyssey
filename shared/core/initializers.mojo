"""Weight initialization methods for neural networks.

Provides common initialization strategies including Xavier/Glorot for sigmoid/tanh
networks, Kaiming/He for ReLU networks, and basic uniform/normal distributions.
Proper initialization is crucial for avoiding vanishing/exploding gradients.

Mathematical foundations:
- Xavier: Var(W) = 2/(fan_in + fan_out) for symmetric activations
- Kaiming: Var(W) = 2/fan_in for ReLU activations
- Uniform/Normal: Basic distributions with configurable parameters

Type support:
- All initializers: float16, float32, float64

Issues covered:
- #258-260: Xavier/Glorot initialization (uniform and normal variants)
- #268-272: Uniform/Normal basic distributions
"""

from random import random_float64, random_si64, seed as random_seed
from math import sqrt, log, cos, sin
from collections.vector import DynamicVector
from .extensor import ExTensor


# ============================================================================
# Internal Helper Functions (Dtype-Generic)
# ============================================================================


@always_inline
fn _fill_uniform_scaled[dtype: DType](result: ExTensor, scale: Float64, offset: Float64) raises:
    """Fill tensor with scaled uniform random values: offset + random() * scale.

    This is a dtype-generic helper that eliminates dtype branching.
    random_float64() returns [0, 1), which is transformed to [offset, offset+scale).

    Args:
        result: Tensor to fill (must be pre-allocated)
        scale: Scale factor for random values
        offset: Offset to add to scaled values
    """
    var ptr = result._data.bitcast[Scalar[dtype]]()
    for i in range(result._numel):
        var rand_val = random_float64()
        ptr[i] = Scalar[dtype](offset + rand_val * scale)


@always_inline
fn _fill_normal_boxmuller[dtype: DType](result: ExTensor, mean: Float64, std: Float64) raises:
    """Fill tensor with normal random values using Box-Muller transform.

    This is a dtype-generic helper that eliminates dtype branching.
    Generates pairs of normal random values using Box-Muller transform.

    Args:
        result: Tensor to fill (must be pre-allocated)
        mean: Mean of normal distribution
        std: Standard deviation of normal distribution
    """
    var ptr = result._data.bitcast[Scalar[dtype]]()
    var i = 0
    alias PI = 3.14159265359

    while i < result._numel:
        # Box-Muller transform to generate normal distribution
        var u1 = random_float64()
        var u2 = random_float64()

        # Avoid log(0)
        if u1 < 1e-10:
            u1 = 1e-10

        var mag = std * sqrt(-2.0 * log(u1))
        var z0 = mean + mag * cos(2.0 * PI * u2)
        var z1 = mean + mag * sin(2.0 * PI * u2)

        ptr[i] = Scalar[dtype](z0)
        i += 1

        if i < result._numel:
            ptr[i] = Scalar[dtype](z1)
            i += 1


@always_inline
fn _fill_constant[dtype: DType](result: ExTensor, value: Float64) raises:
    """Fill tensor with constant value.

    This is a dtype-generic helper that eliminates dtype branching.

    Args:
        result: Tensor to fill (must be pre-allocated)
        value: Constant value to fill with
    """
    var ptr = result._data.bitcast[Scalar[dtype]]()
    var val = Scalar[dtype](value)
    for i in range(result._numel):
        ptr[i] = val


# ============================================================================
# Xavier/Glorot Initialization (#258-260)
# ============================================================================


fn xavier_uniform(fan_in: Int, fan_out: Int, shape: DynamicVector[Int], dtype: DType = DType.float32, seed_val: Int = -1) raises -> ExTensor:
    """Initialize weights using Xavier/Glorot uniform distribution.

    Draws samples from uniform distribution U(-a, a) where:
        a = sqrt(6 / (fan_in + fan_out))

    This initialization maintains variance of activations and gradients across
    layers for networks using sigmoid or tanh activations.

    Mathematical derivation:
    - For uniform U(-a, a), variance is a²/3
    - Target variance: 2/(fan_in + fan_out)
    - Therefore: a²/3 = 2/(fan_in + fan_out)
    - Solving: a = sqrt(6/(fan_in + fan_out))

    Supported dtypes: float16, float32, float64

    Args:
        fan_in: Number of input units to the layer
        fan_out: Number of output units from the layer
        shape: Shape of weight tensor to initialize
        dtype: Data type (default: float32)
        seed_val: Random seed for reproducibility (-1 for random seed)

    Returns:
        Initialized weight tensor with Xavier uniform distribution

    Raises:
        Error: If fan_in or fan_out are not positive

    Examples:
        # Fully connected layer: 784 inputs -> 128 outputs
        var weights = xavier_uniform(784, 128, DynamicVector[Int](784, 128))

        # With fixed seed for reproducibility
        var w = xavier_uniform(100, 50, DynamicVector[Int](100, 50), seed_val=42)

    References:
        Glorot & Bengio (2010): "Understanding the difficulty of training
        deep feedforward neural networks"
    """
    if fan_in <= 0 or fan_out <= 0:
        raise Error("xavier_uniform: fan_in and fan_out must be positive")

    # Set random seed if provided
    if seed_val >= 0:
        random_seed(seed_val)

    # Calculate bound: sqrt(6 / (fan_in + fan_out))
    var bound = sqrt(6.0 / Float64(fan_in + fan_out))

    # Create tensor
    var result = ExTensor(shape, dtype)

    # Fill with uniform random values in [-bound, bound]
    # Use dtype dispatch pattern to eliminate branching
    if dtype == DType.float16:
        _fill_uniform_scaled[DType.float16](result, 2.0 * bound, -bound)
    elif dtype == DType.float32:
        _fill_uniform_scaled[DType.float32](result, 2.0 * bound, -bound)
    elif dtype == DType.float64:
        _fill_uniform_scaled[DType.float64](result, 2.0 * bound, -bound)
    else:
        raise Error("xavier_uniform: only float16, float32, and float64 dtypes supported")

    return result


fn xavier_normal(fan_in: Int, fan_out: Int, shape: DynamicVector[Int], dtype: DType = DType.float32, seed_val: Int = -1) raises -> ExTensor:
    """Initialize weights using Xavier/Glorot normal distribution.

    Draws samples from normal distribution N(0, std²) where:
        std = sqrt(2 / (fan_in + fan_out))

    This initialization maintains variance of activations and gradients across
    layers for networks using sigmoid or tanh activations.

    Mathematical derivation:
    - For normal N(0, σ²), variance is σ²
    - Target variance: 2/(fan_in + fan_out)
    - Therefore: σ² = 2/(fan_in + fan_out)
    - Standard deviation: σ = sqrt(2/(fan_in + fan_out))

    Supported dtypes: float16, float32, float64

    Args:
        fan_in: Number of input units to the layer
        fan_out: Number of output units from the layer
        shape: Shape of weight tensor to initialize
        dtype: Data type (default: float32)
        seed_val: Random seed for reproducibility (-1 for random seed)

    Returns:
        Initialized weight tensor with Xavier normal distribution

    Raises:
        Error: If fan_in or fan_out are not positive

    Examples:
        # Fully connected layer: 784 inputs -> 128 outputs
        var weights = xavier_normal(784, 128, DynamicVector[Int](784, 128))

        # With fixed seed for reproducibility
        var w = xavier_normal(100, 50, DynamicVector[Int](100, 50), seed_val=42)

    References:
        Glorot & Bengio (2010): "Understanding the difficulty of training
        deep feedforward neural networks"
    """
    if fan_in <= 0 or fan_out <= 0:
        raise Error("xavier_normal: fan_in and fan_out must be positive")

    # Set random seed if provided
    if seed_val >= 0:
        random_seed(seed_val)

    # Calculate standard deviation: sqrt(2 / (fan_in + fan_out))
    var std = sqrt(2.0 / Float64(fan_in + fan_out))

    # Create tensor
    var result = ExTensor(shape, dtype)

    # Fill with normal random values using Box-Muller transform
    # Use dtype dispatch pattern to eliminate branching
    if dtype == DType.float16:
        _fill_normal_boxmuller[DType.float16](result, 0.0, std)
    elif dtype == DType.float32:
        _fill_normal_boxmuller[DType.float32](result, 0.0, std)
    elif dtype == DType.float64:
        _fill_normal_boxmuller[DType.float64](result, 0.0, std)
    else:
        raise Error("xavier_normal: only float16, float32, and float64 dtypes supported")

    return result


# Helper functions for Box-Muller transform
fn log(x: Float64) -> Float64:
    """Natural logarithm (wrapper for clarity)."""
    from math import log as math_log
    return math_log(x)


fn cos(x: Float64) -> Float64:
    """Cosine function (wrapper for clarity)."""
    from math import cos as math_cos
    return math_cos(x)


fn sin(x: Float64) -> Float64:
    """Sine function (wrapper for clarity)."""
    from math import sin as math_sin
    return math_sin(x)


# ============================================================================
# Kaiming/He Initialization (#263-267)
# ============================================================================


fn kaiming_uniform(fan_in: Int, fan_out: Int, shape: DynamicVector[Int], fan_mode: String = "fan_in", dtype: DType = DType.float32, seed_val: Int = -1) raises -> ExTensor:
    """Initialize weights using Kaiming/He uniform distribution.

    Draws samples from uniform distribution U(-a, a) where:
        a = sqrt(6 / fan)  (fan depends on fan_mode)

    This initialization is designed for networks with ReLU activations, which
    kill half the activations (output 0 for negative inputs). The gain factor
    accounts for this to maintain variance.

    Mathematical derivation:
    - For uniform U(-a, a), variance is a²/3
    - Target variance for ReLU: 2/fan (not 2/(fan_in + fan_out) like Xavier)
    - Therefore: a²/3 = 2/fan
    - Solving: a = sqrt(6/fan)

    Supported dtypes: float16, float32, float64

    Args:
        fan_in: Number of input units to the layer
        fan_out: Number of output units from the layer
        shape: Shape of weight tensor to initialize
        fan_mode: "fan_in" (default) or "fan_out" for fan calculation
        dtype: Data type (default: float32)
        seed_val: Random seed for reproducibility (-1 for random seed)

    Returns:
        Initialized weight tensor with Kaiming uniform distribution

    Raises:
        Error: If fan_in or fan_out are not positive
        Error: If fan_mode is not "fan_in" or "fan_out"

    Examples:
        # Fully connected layer: 784 inputs -> 128 outputs (using fan_in)
        var weights = kaiming_uniform(784, 128, DynamicVector[Int](784, 128))

        # Using fan_out mode
        var w = kaiming_uniform(784, 128, DynamicVector[Int](784, 128), fan_mode="fan_out")

        # With fixed seed for reproducibility
        var w_repro = kaiming_uniform(100, 50, DynamicVector[Int](100, 50), seed_val=42)

    References:
        He et al. (2015): "Delving Deep into Rectifiers: Surpassing Human-Level
        Performance on ImageNet Classification"

    Issue: #263-267 - Kaiming/He initialization
    """
    if fan_in <= 0 or fan_out <= 0:
        raise Error("kaiming_uniform: fan_in and fan_out must be positive")

    # Determine which fan to use
    var fan: Int
    if fan_mode == "fan_in":
        fan = fan_in
    elif fan_mode == "fan_out":
        fan = fan_out
    else:
        raise Error("kaiming_uniform: fan_mode must be 'fan_in' or 'fan_out'")

    # Set random seed if provided
    if seed_val >= 0:
        random_seed(seed_val)

    # Calculate bound: sqrt(6 / fan)
    var bound = sqrt(6.0 / Float64(fan))

    # Create tensor
    var result = ExTensor(shape, dtype)

    # Fill with uniform random values in [-bound, bound]
    # Use dtype dispatch pattern to eliminate branching
    if dtype == DType.float16:
        _fill_uniform_scaled[DType.float16](result, 2.0 * bound, -bound)
    elif dtype == DType.float32:
        _fill_uniform_scaled[DType.float32](result, 2.0 * bound, -bound)
    elif dtype == DType.float64:
        _fill_uniform_scaled[DType.float64](result, 2.0 * bound, -bound)
    else:
        raise Error("kaiming_uniform: only float16, float32, and float64 dtypes supported")

    return result


fn kaiming_normal(fan_in: Int, fan_out: Int, shape: DynamicVector[Int], fan_mode: String = "fan_in", dtype: DType = DType.float32, seed_val: Int = -1) raises -> ExTensor:
    """Initialize weights using Kaiming/He normal distribution.

    Draws samples from normal distribution N(0, std²) where:
        std = sqrt(2 / fan)  (fan depends on fan_mode)

    This initialization is designed for networks with ReLU activations, which
    kill half the activations. The variance scaling accounts for this effect.

    Mathematical derivation:
    - For normal N(0, σ²), variance is σ²
    - Target variance for ReLU: 2/fan
    - Therefore: σ² = 2/fan
    - Standard deviation: σ = sqrt(2/fan)

    Supported dtypes: float16, float32, float64

    Args:
        fan_in: Number of input units to the layer
        fan_out: Number of output units from the layer
        shape: Shape of weight tensor to initialize
        fan_mode: "fan_in" (default) or "fan_out" for fan calculation
        dtype: Data type (default: float32)
        seed_val: Random seed for reproducibility (-1 for random seed)

    Returns:
        Initialized weight tensor with Kaiming normal distribution

    Raises:
        Error: If fan_in or fan_out are not positive
        Error: If fan_mode is not "fan_in" or "fan_out"

    Examples:
        # Fully connected layer: 784 inputs -> 128 outputs
        var weights = kaiming_normal(784, 128, DynamicVector[Int](784, 128))

        # Using fan_out mode
        var w = kaiming_normal(784, 128, DynamicVector[Int](784, 128), fan_mode="fan_out")

        # With fixed seed for reproducibility
        var w_repro = kaiming_normal(100, 50, DynamicVector[Int](100, 50), seed_val=42)

    References:
        He et al. (2015): "Delving Deep into Rectifiers: Surpassing Human-Level
        Performance on ImageNet Classification"

    Issue: #263-267 - Kaiming/He initialization
    """
    if fan_in <= 0 or fan_out <= 0:
        raise Error("kaiming_normal: fan_in and fan_out must be positive")

    # Determine which fan to use
    var fan: Int
    if fan_mode == "fan_in":
        fan = fan_in
    elif fan_mode == "fan_out":
        fan = fan_out
    else:
        raise Error("kaiming_normal: fan_mode must be 'fan_in' or 'fan_out'")

    # Set random seed if provided
    if seed_val >= 0:
        random_seed(seed_val)

    # Calculate standard deviation: sqrt(2 / fan)
    var std = sqrt(2.0 / Float64(fan))

    # Create tensor
    var result = ExTensor(shape, dtype)

    # Fill with normal random values using Box-Muller transform
    # Use dtype dispatch pattern to eliminate branching
    if dtype == DType.float16:
        _fill_normal_boxmuller[DType.float16](result, 0.0, std)
    elif dtype == DType.float32:
        _fill_normal_boxmuller[DType.float32](result, 0.0, std)
    elif dtype == DType.float64:
        _fill_normal_boxmuller[DType.float64](result, 0.0, std)
    else:
        raise Error("kaiming_normal: only float16, float32, and float64 dtypes supported")

    return result


# ============================================================================
# Uniform/Normal Initialization (#268-272)
# ============================================================================


fn uniform(shape: DynamicVector[Int], low: Float64 = -0.1, high: Float64 = 0.1, dtype: DType = DType.float32, seed_val: Int = -1) raises -> ExTensor:
    """Initialize weights using uniform distribution.

    Draws samples from uniform distribution U(low, high) with configurable bounds.
    This is a basic initializer useful for biases, embeddings, or custom schemes.

    Args:
        shape: Shape of tensor to initialize
        low: Lower bound of uniform distribution (default: -0.1)
        high: Upper bound of uniform distribution (default: 0.1)
        dtype: Data type (default: float32)
        seed_val: Random seed for reproducibility (-1 for random seed)

    Returns:
        Initialized tensor with uniform distribution

    Raises:
        Error: If low >= high

    Examples:
        # Default range [-0.1, 0.1]
        var weights = uniform(DynamicVector[Int](100, 50))

        # Custom range [0, 1]
        var w = uniform(DynamicVector[Int](10, 10), low=0.0, high=1.0)

        # With fixed seed
        var w_repro = uniform(DynamicVector[Int](50, 50), seed_val=42)

    Issue: #268-272 - Uniform/Normal basic distributions
    """
    if low >= high:
        raise Error("uniform: low must be less than high")

    # Set random seed if provided
    if seed_val >= 0:
        random_seed(seed_val)

    # Create tensor
    var result = ExTensor(shape, dtype)

    # Fill with uniform random values in [low, high]
    # Use dtype dispatch pattern to eliminate branching
    var range_val = high - low
    if dtype == DType.float16:
        _fill_uniform_scaled[DType.float16](result, range_val, low)
    elif dtype == DType.float32:
        _fill_uniform_scaled[DType.float32](result, range_val, low)
    elif dtype == DType.float64:
        _fill_uniform_scaled[DType.float64](result, range_val, low)
    else:
        raise Error("uniform: only float16, float32, and float64 dtypes supported")

    return result


fn normal(shape: DynamicVector[Int], mean: Float64 = 0.0, std: Float64 = 0.01, dtype: DType = DType.float32, seed_val: Int = -1) raises -> ExTensor:
    """Initialize weights using normal (Gaussian) distribution.

    Draws samples from normal distribution N(mean, std²) with configurable parameters.
    This is a basic initializer useful for biases, embeddings, or custom schemes.

    Uses Box-Muller transform to generate normal distribution from uniform samples.

    Args:
        shape: Shape of tensor to initialize
        mean: Mean of normal distribution (default: 0.0)
        std: Standard deviation of normal distribution (default: 0.01)
        dtype: Data type (default: float32)
        seed_val: Random seed for reproducibility (-1 for random seed)

    Returns:
        Initialized tensor with normal distribution

    Raises:
        Error: If std <= 0

    Examples:
        # Default: N(0, 0.01)
        var weights = normal(DynamicVector[Int](100, 50))

        # Custom: N(0.5, 0.1)
        var w = normal(DynamicVector[Int](10, 10), mean=0.5, std=0.1)

        # With fixed seed
        var w_repro = normal(DynamicVector[Int](50, 50), seed_val=42)

    Issue: #268-272 - Uniform/Normal basic distributions
    """
    if std <= 0.0:
        raise Error("normal: standard deviation must be positive")

    # Set random seed if provided
    if seed_val >= 0:
        random_seed(seed_val)

    # Create tensor
    var result = ExTensor(shape, dtype)

    # Fill with normal random values using Box-Muller transform
    # Use dtype dispatch pattern to eliminate branching
    if dtype == DType.float16:
        _fill_normal_boxmuller[DType.float16](result, mean, std)
    elif dtype == DType.float32:
        _fill_normal_boxmuller[DType.float32](result, mean, std)
    elif dtype == DType.float64:
        _fill_normal_boxmuller[DType.float64](result, mean, std)
    else:
        raise Error("normal: only float16, float32, and float64 dtypes supported")

    return result


fn constant(shape: DynamicVector[Int], value: Float64, dtype: DType = DType.float32) raises -> ExTensor:
    """Initialize tensor with constant value.

    Fills all elements with the specified constant value.
    Useful for specific initialization strategies (ones, custom bias values, etc.).

    Args:
        shape: Shape of tensor to initialize
        value: Constant value to fill tensor with
        dtype: Data type (default: float32)

    Returns:
        Tensor filled with constant value

    Examples:
        # Initialize with ones
        var ones = constant(DynamicVector[Int](10, 10), 1.0)

        # Initialize with custom value
        var custom = constant(DynamicVector[Int](5, 5), 0.5)

        # Initialize bias with 0.01
        var bias = constant(DynamicVector[Int](100), 0.01)

    Issue: #268-272 - Uniform/Normal basic distributions
    """
    var result = ExTensor(shape, dtype)

    # Use dtype dispatch pattern to eliminate branching
    if dtype == DType.float16:
        _fill_constant[DType.float16](result, value)
    elif dtype == DType.float32:
        _fill_constant[DType.float32](result, value)
    elif dtype == DType.float64:
        _fill_constant[DType.float64](result, value)
    else:
        raise Error("constant: only float16, float32, and float64 dtypes supported")

    return result
