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
"""

from random import random_float64, random_si64, seed as random_seed
from math import sqrt
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
