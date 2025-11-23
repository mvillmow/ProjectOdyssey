"""Tests for weight initialization functions.

Tests cover:
- Xavier initialization (uniform and normal)
- Kaiming/He initialization (uniform and normal)
- Uniform and normal distributions
- Constant initialization
- Statistical properties (mean, variance, range)

All tests use pure functional API.
"""

from collections import Tuple
from tests.shared.conftest import (
    assert_true,
    assert_equal,
    assert_almost_equal,
    assert_shape_equal,
    TestFixtures,
)
from shared.core.extensor import ExTensor, zeros
from shared.core.initializers import (
    xavier_uniform,
    xavier_normal,
    kaiming_uniform,
    kaiming_normal,
    uniform,
    normal,
    constant,
)
from math import sqrt


# ============================================================================
# Helper Functions for Statistical Tests
# ============================================================================


fn compute_mean(tensor: ExTensor) -> Float64:
    """Compute mean of tensor values."""
    var sum = Float64(0.0)
    var size = tensor.numel()

    if tensor.dtype() == DType.float32:
        var ptr = tensor._data.bitcast[Float32]()
        for i in range(size):
            sum += Float64(ptr[i])
    elif tensor.dtype() == DType.float64:
        var ptr = tensor._data.bitcast[Float64]()
        for i in range(size):
            sum += ptr[i]

    return sum / Float64(size)


fn compute_variance(tensor: ExTensor, mean: Float64) -> Float64:
    """Compute variance of tensor values."""
    var sum_sq_diff = Float64(0.0)
    var size = tensor.numel()

    if tensor.dtype() == DType.float32:
        var ptr = tensor._data.bitcast[Float32]()
        for i in range(size):
            var diff = Float64(ptr[i]) - mean
            sum_sq_diff += diff * diff
    elif tensor.dtype() == DType.float64:
        var ptr = tensor._data.bitcast[Float64]()
        for i in range(size):
            var diff = ptr[i] - mean
            sum_sq_diff += diff * diff

    return sum_sq_diff / Float64(size)


fn compute_std(tensor: ExTensor, mean: Float64) -> Float64:
    """Compute standard deviation of tensor values."""
    return sqrt(compute_variance(tensor, mean))


fn compute_min_max(tensor: ExTensor) -> Tuple[Float64, Float64]:
    """Compute min and max values in tensor."""
    var size = tensor.numel()
    var min_val = Float64(1e308)
    var max_val = Float64(-1e308)

    if tensor.dtype() == DType.float32:
        var ptr = tensor._data.bitcast[Float32]()
        for i in range(size):
            var val = Float64(ptr[i])
            if val < min_val:
                min_val = val
            if val > max_val:
                max_val = val
    elif tensor.dtype() == DType.float64:
        var ptr = tensor._data.bitcast[Float64]()
        for i in range(size):
            var val = ptr[i]
            if val < min_val:
                min_val = val
            if val > max_val:
                max_val = val

    return (min_val, max_val)


# ============================================================================
# Xavier Uniform Tests
# ============================================================================


fn test_xavier_uniform_shape() raises:
    """Test Xavier uniform initialization preserves shape."""
    var fan_in = 100
    var fan_out = 50
    var W = xavier_uniform(fan_in, fan_out, DType.float32)

    assert_equal(W.shape()[0], fan_in)
    assert_equal(W.shape()[1], fan_out)
    assert_equal(W.numel(), fan_in * fan_out)


fn test_xavier_uniform_range() raises:
    """Test Xavier uniform values are within expected range."""
    var fan_in = 1000
    var fan_out = 500
    var W = xavier_uniform(fan_in, fan_out, DType.float32)

    # Xavier uniform limit: sqrt(6 / (fan_in + fan_out))
    var limit = sqrt(6.0 / Float64(fan_in + fan_out))

    var (min_val, max_val) = compute_min_max(W)

    # All values should be approximately in [-limit, limit]
    # Allow small tolerance for floating point errors
    assert_true(min_val >= -limit - 0.01)
    assert_true(max_val <= limit + 0.01)


fn test_xavier_uniform_mean() raises:
    """Test Xavier uniform has approximately zero mean."""
    var fan_in = 1000
    var fan_out = 1000
    var W = xavier_uniform(fan_in, fan_out, DType.float32)

    var mean = compute_mean(W)

    # Mean should be close to 0 (within tolerance for random sampling)
    assert_almost_equal(mean, 0.0, tolerance=0.01)


fn test_xavier_uniform_variance() raises:
    """Test Xavier uniform has approximately correct variance."""
    var fan_in = 2000
    var fan_out = 2000
    var W = xavier_uniform(fan_in, fan_out, DType.float32)

    var mean = compute_mean(W)
    var std_dev = compute_std(W, mean)

    # For uniform distribution U(-a, a): variance = a�/3
    # Xavier limit: a = sqrt(6 / (fan_in + fan_out))
    # Expected std = a / sqrt(3) = sqrt(6 / (fan_in + fan_out)) / sqrt(3)
    #                             = sqrt(2 / (fan_in + fan_out))
    var expected_std = sqrt(2.0 / Float64(fan_in + fan_out))

    # Allow 10% tolerance for statistical variation
    assert_almost_equal(std_dev, expected_std, tolerance=expected_std * 0.1)


# ============================================================================
# Xavier Normal Tests
# ============================================================================


fn test_xavier_normal_shape() raises:
    """Test Xavier normal initialization preserves shape."""
    var fan_in = 100
    var fan_out = 50
    var W = xavier_normal(fan_in, fan_out, DType.float32)

    assert_equal(W.shape()[0], fan_in)
    assert_equal(W.shape()[1], fan_out)


fn test_xavier_normal_mean() raises:
    """Test Xavier normal has approximately zero mean."""
    var fan_in = 1000
    var fan_out = 1000
    var W = xavier_normal(fan_in, fan_out, DType.float32)

    var mean = compute_mean(W)

    assert_almost_equal(mean, 0.0, tolerance=0.01)


fn test_xavier_normal_std() raises:
    """Test Xavier normal has approximately correct standard deviation."""
    var fan_in = 2000
    var fan_out = 2000
    var W = xavier_normal(fan_in, fan_out, DType.float32)

    var mean = compute_mean(W)
    var std_dev = compute_std(W, mean)

    # Xavier normal: std = sqrt(2 / (fan_in + fan_out))
    var expected_std = sqrt(2.0 / Float64(fan_in + fan_out))

    # Allow 10% tolerance
    assert_almost_equal(std_dev, expected_std, tolerance=expected_std * 0.1)


# ============================================================================
# Kaiming Uniform Tests
# ============================================================================


fn test_kaiming_uniform_shape() raises:
    """Test Kaiming uniform initialization preserves shape."""
    var fan_in = 100
    var fan_out = 50
    var W = kaiming_uniform(fan_in, fan_out, DType.float32)

    assert_equal(W.shape()[0], fan_in)
    assert_equal(W.shape()[1], fan_out)


fn test_kaiming_uniform_range() raises:
    """Test Kaiming uniform values are within expected range."""
    var fan_in = 1000
    var W = kaiming_uniform(fan_in, 500, DType.float32)

    # Kaiming uniform limit: sqrt(6 / fan_in)
    var limit = sqrt(6.0 / Float64(fan_in))

    var (min_val, max_val) = compute_min_max(W)

    # All values should be in [-limit, limit]
    assert_true(min_val >= -limit - 0.01)
    assert_true(max_val <= limit + 0.01)


fn test_kaiming_uniform_mean() raises:
    """Test Kaiming uniform has approximately zero mean."""
    var W = kaiming_uniform(1000, 1000, DType.float32)

    var mean = compute_mean(W)

    assert_almost_equal(mean, 0.0, tolerance=0.01)


fn test_kaiming_uniform_variance() raises:
    """Test Kaiming uniform has approximately correct variance."""
    var fan_in = 2000
    var W = kaiming_uniform(fan_in, 2000, DType.float32)

    var mean = compute_mean(W)
    var std_dev = compute_std(W, mean)

    # Expected std = sqrt(2 / fan_in)
    var expected_std = sqrt(2.0 / Float64(fan_in))

    # Allow 10% tolerance
    assert_almost_equal(std_dev, expected_std, tolerance=expected_std * 0.1)


# ============================================================================
# Kaiming Normal Tests
# ============================================================================


fn test_kaiming_normal_shape() raises:
    """Test Kaiming normal initialization preserves shape."""
    var fan_in = 100
    var fan_out = 50
    var W = kaiming_normal(fan_in, fan_out, DType.float32)

    assert_equal(W.shape()[0], fan_in)
    assert_equal(W.shape()[1], fan_out)


fn test_kaiming_normal_mean() raises:
    """Test Kaiming normal has approximately zero mean."""
    var W = kaiming_normal(1000, 1000, DType.float32)

    var mean = compute_mean(W)

    assert_almost_equal(mean, 0.0, tolerance=0.01)


fn test_kaiming_normal_std() raises:
    """Test Kaiming normal has approximately correct standard deviation."""
    var fan_in = 2000
    var W = kaiming_normal(fan_in, 2000, DType.float32)

    var mean = compute_mean(W)
    var std_dev = compute_std(W, mean)

    # Kaiming normal: std = sqrt(2 / fan_in)
    var expected_std = sqrt(2.0 / Float64(fan_in))

    # Allow 10% tolerance
    assert_almost_equal(std_dev, expected_std, tolerance=expected_std * 0.1)


# ============================================================================
# Uniform Distribution Tests
# ============================================================================


fn test_uniform_shape() raises:
    """Test uniform initialization with custom range."""
    var shape = List[Int](2)
    shape[0] = 50
    shape[1] = 30
    var W = uniform(shape, -0.5, 0.5, DType.float32)

    assert_equal(W.shape()[0], 50)
    assert_equal(W.shape()[1], 30)


fn test_uniform_range() raises:
    """Test uniform values are within specified range."""
    var shape = List[Int](2)
    shape[0] = 100
    shape[1] = 100
    var low = -0.7
    var high = 0.3
    var W = uniform(shape, low, high, DType.float32)

    var (min_val, max_val) = compute_min_max(W)

    # All values should be in [low, high]
    assert_true(min_val >= low - 1e-5)
    assert_true(max_val <= high + 1e-5)


fn test_uniform_mean() raises:
    """Test uniform distribution has approximately correct mean."""
    var shape = List[Int](2)
    shape[0] = 200
    shape[1] = 200
    var low = -1.0
    var high = 1.0
    var W = uniform(shape, low, high, DType.float32)

    var mean = compute_mean(W)

    # Mean of uniform distribution U(a,b) is (a+b)/2
    var expected_mean = (low + high) / 2.0
    assert_almost_equal(mean, expected_mean, tolerance=0.05)


# ============================================================================
# Normal Distribution Tests
# ============================================================================


fn test_normal_shape() raises:
    """Test normal initialization with custom parameters."""
    var shape = List[Int](2)
    shape[0] = 50
    shape[1] = 30
    var W = normal(shape, 0.0, 1.0, DType.float32)

    assert_equal(W.shape()[0], 50)
    assert_equal(W.shape()[1], 30)


fn test_normal_mean() raises:
    """Test normal distribution has approximately correct mean."""
    var shape = List[Int](2)
    shape[0] = 200
    shape[1] = 200
    var target_mean = 2.5
    var target_std = 0.5
    var W = normal(shape, target_mean, target_std, DType.float32)

    var actual_mean = compute_mean(W)

    # Allow 5% tolerance for sampling variability
    assert_almost_equal(actual_mean, target_mean, tolerance=0.1)


fn test_normal_std() raises:
    """Test normal distribution has approximately correct standard deviation."""
    var shape = List[Int](2)
    shape[0] = 300
    shape[1] = 300
    var target_mean = 0.0
    var target_std = 1.5
    var W = normal(shape, target_mean, target_std, DType.float32)

    var actual_mean = compute_mean(W)
    var actual_std = compute_std(W, actual_mean)

    # Allow 10% tolerance for sampling variability
    assert_almost_equal(actual_std, target_std, tolerance=target_std * 0.1)


# ============================================================================
# Constant Initialization Tests
# ============================================================================


fn test_constant_shape() raises:
    """Test constant initialization preserves shape."""
    var shape = List[Int](2)
    shape[0] = 50
    shape[1] = 30
    var W = constant(shape, 3.14, DType.float32)

    assert_equal(W.shape()[0], 50)
    assert_equal(W.shape()[1], 30)


fn test_constant_value() raises:
    """Test constant initialization sets all values correctly."""
    var shape = List[Int](2)
    shape[0] = 10
    shape[1] = 10
    var value = 7.5
    var W = constant(shape, value, DType.float32)

    # Check all values are exactly the constant
    for i in range(100):
        assert_almost_equal(W._data.bitcast[Float32]()[i], Float32(value), tolerance=1e-5)


fn test_constant_zero() raises:
    """Test constant initialization with zero."""
    var shape = List[Int](2)
    shape[0] = 5
    shape[1] = 5
    var W = constant(shape, 0.0, DType.float32)

    for i in range(25):
        assert_almost_equal(W._data.bitcast[Float32]()[i], Float32(0.0), tolerance=1e-10)


fn test_constant_negative() raises:
    """Test constant initialization with negative value."""
    var shape = List[Int](2)
    shape[0] = 5
    shape[1] = 5
    var value = -2.5
    var W = constant(shape, value, DType.float32)

    for i in range(25):
        assert_almost_equal(W._data.bitcast[Float32]()[i], Float32(value), tolerance=1e-5)


# ============================================================================
# Dtype Support Tests
# ============================================================================


fn test_xavier_uniform_float64() raises:
    """Test Xavier uniform with float64 dtype."""
    var W = xavier_uniform(100, 50, DType.float64)

    assert_equal(W.dtype(), DType.float64)
    assert_equal(W.shape()[0], 100)
    assert_equal(W.shape()[1], 50)


fn test_kaiming_normal_float64() raises:
    """Test Kaiming normal with float64 dtype."""
    var W = kaiming_normal(100, 50, DType.float64)

    assert_equal(W.dtype(), DType.float64)
    assert_equal(W.shape()[0], 100)
    assert_equal(W.shape()[1], 50)


fn test_constant_float64() raises:
    """Test constant initialization with float64 dtype."""
    var shape = List[Int](2)
    shape[0] = 10
    shape[1] = 10
    var W = constant(shape, 1.5, DType.float64)

    assert_equal(W.dtype(), DType.float64)

    for i in range(100):
        assert_almost_equal(W._data.bitcast[Float64]()[i], 1.5, tolerance=1e-10)


# ============================================================================
# Edge Case Tests
# ============================================================================


fn test_small_dimensions() raises:
    """Test initialization with small dimensions."""
    # Xavier with fan_in=1, fan_out=1
    var W1 = xavier_uniform(1, 1, DType.float32)
    assert_equal(W1.numel(), 1)

    # Kaiming with fan_in=2
    var W2 = kaiming_uniform(2, 3, DType.float32)
    assert_equal(W2.shape()[0], 2)
    assert_equal(W2.shape()[1], 3)


fn test_rectangular_matrices() raises:
    """Test initialization with non-square matrices."""
    # Tall matrix (more rows than columns)
    var W_tall = xavier_uniform(1000, 10, DType.float32)
    assert_equal(W_tall.shape()[0], 1000)
    assert_equal(W_tall.shape()[1], 10)

    # Wide matrix (more columns than rows)
    var W_wide = kaiming_uniform(10, 1000, DType.float32)
    assert_equal(W_wide.shape()[0], 10)
    assert_equal(W_wide.shape()[1], 1000)


fn test_large_initialization() raises:
    """Test initialization with large dimensions."""
    # Large matrix
    var W = xavier_uniform(5000, 5000, DType.float32)

    assert_equal(W.shape()[0], 5000)
    assert_equal(W.shape()[1], 5000)
    assert_equal(W.numel(), 25000000)

    # Verify statistical properties still hold
    var mean = compute_mean(W)
    assert_almost_equal(mean, 0.0, tolerance=0.01)
