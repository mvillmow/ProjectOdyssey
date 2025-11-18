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
    if dtype == DType.float16:
        for i in range(result._numel):
            # random_float64() returns [0, 1), transform to [-bound, bound]
            var rand_val = random_float64()
            var scaled_val = Float16((2.0 * rand_val - 1.0) * bound)
            result._data.bitcast[Float16]()[i] = scaled_val
    elif dtype == DType.float32:
        for i in range(result._numel):
            var rand_val = random_float64()
            var scaled_val = Float32((2.0 * rand_val - 1.0) * bound)
            result._data.bitcast[Float32]()[i] = scaled_val
    elif dtype == DType.float64:
        for i in range(result._numel):
            var rand_val = random_float64()
            var scaled_val = (2.0 * rand_val - 1.0) * bound
            result._data.bitcast[Float64]()[i] = scaled_val
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
    if dtype == DType.float16:
        var i = 0
        while i < result._numel:
            # Box-Muller transform to generate normal distribution
            var u1 = random_float64()
            var u2 = random_float64()

            # Avoid log(0)
            if u1 < 1e-10:
                u1 = 1e-10

            var mag = std * sqrt(-2.0 * log(u1))
            var z0 = mag * cos(2.0 * 3.14159265359 * u2)
            var z1 = mag * sin(2.0 * 3.14159265359 * u2)

            result._data.bitcast[Float16]()[i] = Float16(z0)
            i += 1

            if i < result._numel:
                result._data.bitcast[Float16]()[i] = Float16(z1)
                i += 1

    elif dtype == DType.float32:
        var i = 0
        while i < result._numel:
            # Box-Muller transform to generate normal distribution
            var u1 = random_float64()
            var u2 = random_float64()

            # Avoid log(0)
            if u1 < 1e-10:
                u1 = 1e-10

            var mag = std * sqrt(-2.0 * log(u1))
            var z0 = mag * cos(2.0 * 3.14159265359 * u2)
            var z1 = mag * sin(2.0 * 3.14159265359 * u2)

            result._data.bitcast[Float32]()[i] = Float32(z0)
            i += 1

            if i < result._numel:
                result._data.bitcast[Float32]()[i] = Float32(z1)
                i += 1

    elif dtype == DType.float64:
        var i = 0
        while i < result._numel:
            # Box-Muller transform
            var u1 = random_float64()
            var u2 = random_float64()

            if u1 < 1e-10:
                u1 = 1e-10

            var mag = std * sqrt(-2.0 * log(u1))
            var z0 = mag * cos(2.0 * 3.14159265359 * u2)
            var z1 = mag * sin(2.0 * 3.14159265359 * u2)

            result._data.bitcast[Float64]()[i] = z0
            i += 1

            if i < result._numel:
                result._data.bitcast[Float64]()[i] = z1
                i += 1
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

    # Calculate range
    var range_val = high - low

    # Fill with uniform random values in [low, high]
    if dtype == DType.float16:
        for i in range(result._numel):
            # random_float64() returns [0, 1), transform to [low, high]
            var rand_val = random_float64()
            var scaled_val = Float16(low + rand_val * range_val)
            result._data.bitcast[Float16]()[i] = scaled_val
    elif dtype == DType.float32:
        for i in range(result._numel):
            var rand_val = random_float64()
            var scaled_val = Float32(low + rand_val * range_val)
            result._data.bitcast[Float32]()[i] = scaled_val
    elif dtype == DType.float64:
        for i in range(result._numel):
            var rand_val = random_float64()
            var scaled_val = low + rand_val * range_val
            result._data.bitcast[Float64]()[i] = scaled_val
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
    if dtype == DType.float16:
        var i = 0
        while i < result._numel:
            # Box-Muller transform to generate normal distribution
            var u1 = random_float64()
            var u2 = random_float64()

            # Avoid log(0)
            if u1 < 1e-10:
                u1 = 1e-10

            var mag = std * sqrt(-2.0 * log(u1))
            var z0 = mean + mag * cos(2.0 * 3.14159265359 * u2)
            var z1 = mean + mag * sin(2.0 * 3.14159265359 * u2)

            result._data.bitcast[Float16]()[i] = Float16(z0)
            i += 1

            if i < result._numel:
                result._data.bitcast[Float16]()[i] = Float16(z1)
                i += 1

    elif dtype == DType.float32:
        var i = 0
        while i < result._numel:
            # Box-Muller transform
            var u1 = random_float64()
            var u2 = random_float64()

            if u1 < 1e-10:
                u1 = 1e-10

            var mag = std * sqrt(-2.0 * log(u1))
            var z0 = mean + mag * cos(2.0 * 3.14159265359 * u2)
            var z1 = mean + mag * sin(2.0 * 3.14159265359 * u2)

            result._data.bitcast[Float32]()[i] = Float32(z0)
            i += 1

            if i < result._numel:
                result._data.bitcast[Float32]()[i] = Float32(z1)
                i += 1

    elif dtype == DType.float64:
        var i = 0
        while i < result._numel:
            # Box-Muller transform
            var u1 = random_float64()
            var u2 = random_float64()

            if u1 < 1e-10:
                u1 = 1e-10

            var mag = std * sqrt(-2.0 * log(u1))
            var z0 = mean + mag * cos(2.0 * 3.14159265359 * u2)
            var z1 = mean + mag * sin(2.0 * 3.14159265359 * u2)

            result._data.bitcast[Float64]()[i] = z0
            i += 1

            if i < result._numel:
                result._data.bitcast[Float64]()[i] = z1
                i += 1
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

    if dtype == DType.float16:
        var val = Float16(value)
        for i in range(result._numel):
            result._data.bitcast[Float16]()[i] = val
    elif dtype == DType.float32:
        var val = Float32(value)
        for i in range(result._numel):
            result._data.bitcast[Float32]()[i] = val
    elif dtype == DType.float64:
        for i in range(result._numel):
            result._data.bitcast[Float64]()[i] = value
    else:
        raise Error("constant: only float16, float32, and float64 dtypes supported")

    return result
