"""Comprehensive test suite for weight initialization functions.

Merged from non-legacy and legacy test files to provide complete coverage of:
- Xavier/Glorot initialization (uniform and normal variants)
- Kaiming/He initialization (uniform and normal variants, fan_in/fan_out modes)
- Basic distributions (uniform, normal, constant)
- Statistical validation (mean, variance, range)
- Reproducibility with seed control
- Dtype support (float16, float32, float64)
- Edge cases (small/large dimensions, rectangular matrices)

All tests use pure functional API.
"""

from ..helpers.assertions import (
    assert_true,
    assert_equal_int,
    assert_close_float,
    assert_shape,
)
from tests.shared.conftest import TestFixtures
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
    elif tensor.dtype() == DType.float16:
        var ptr = tensor._data.bitcast[Float16]()
        for i in range(size):
            sum += Float64(ptr[i])

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
    elif tensor.dtype() == DType.float16:
        var ptr = tensor._data.bitcast[Float16]()
        for i in range(size):
            var diff = Float64(ptr[i]) - mean
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
    elif tensor.dtype() == DType.float16:
        var ptr = tensor._data.bitcast[Float16]()
        for i in range(size):
            var val = Float64(ptr[i])
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
    var shape = List[Int](fan_in, fan_out)
    var W = xavier_uniform(fan_in, fan_out, shape, DType.float32)

    assert_equal(W.shape()[0], fan_in)
    assert_equal(W.shape()[1], fan_out)
    assert_equal(W.numel(), fan_in * fan_out)


fn test_xavier_uniform_range() raises:
    """Test Xavier uniform values are within expected range."""
    var fan_in = 1000
    var fan_out = 500
    var shape = List[Int](fan_in, fan_out)
    var W = xavier_uniform(fan_in, fan_out, shape, DType.float32)

    # Xavier uniform limit: sqrt(6 / (fan_in + fan_out))
    var limit = sqrt(6.0 / Float64(fan_in + fan_out))

    var result = compute_min_max(W)
    var min_val = result[0]
    var max_val = result[1]

    # All values should be approximately in [-limit, limit]
    # Allow small tolerance for floating point errors
    assert_true(min_val >= -limit - 0.01)
    assert_true(max_val <= limit + 0.01)


fn test_xavier_uniform_mean() raises:
    """Test Xavier uniform has approximately zero mean."""
    var fan_in = 1000
    var fan_out = 1000
    var shape = List[Int](fan_in, fan_out)
    var W = xavier_uniform(fan_in, fan_out, shape, DType.float32)

    var mean = compute_mean(W)

    # Mean should be close to 0 (within tolerance for random sampling)
    assert_almost_equal(Float32(mean), Float32(0.0), tolerance=0.01)


fn test_xavier_uniform_variance() raises:
    """Test Xavier uniform has approximately correct variance."""
    var fan_in = 2000
    var fan_out = 2000
    var shape = List[Int](fan_in, fan_out)
    var W = xavier_uniform(fan_in, fan_out, shape, DType.float32)

    var mean = compute_mean(W)
    var std_dev = compute_std(W, mean)

    # For uniform distribution U(-a, a): variance = a²/3
    # Xavier limit: a = sqrt(6 / (fan_in + fan_out))
    # Expected std = a / sqrt(3) = sqrt(6 / (fan_in + fan_out)) / sqrt(3)
    #                             = sqrt(2 / (fan_in + fan_out))
    var expected_std = sqrt(2.0 / Float64(fan_in + fan_out))

    # Allow 10% tolerance for statistical variation
    assert_almost_equal(Float32(std_dev), Float32(expected_std), tolerance=Float32(expected_std) * 0.1)


fn test_xavier_uniform_reproducibility() raises:
    """Test Xavier uniform with fixed seed is reproducible."""
    var fan_in = 50
    var fan_out = 100
    var shape = List[Int](fan_in, fan_out)

    # Generate with same seed twice
    var w1 = xavier_uniform(fan_in, fan_out, shape, DType.float32, seed_val=999)
    var w2 = xavier_uniform(fan_in, fan_out, shape, DType.float32, seed_val=999)

    # Should be identical
    for i in range(w1.numel()):
        var val1 = w1._data.bitcast[Float32]()[i]
        var val2 = w2._data.bitcast[Float32]()[i]
        assert_equal(val1, val2)


fn test_xavier_uniform_different_seeds() raises:
    """Test Xavier uniform with different seeds produces different results."""
    var fan_in = 50
    var fan_out = 100
    var shape = List[Int](fan_in, fan_out)

    # Generate with different seeds
    var w1 = xavier_uniform(fan_in, fan_out, shape, DType.float32, seed_val=111)
    var w2 = xavier_uniform(fan_in, fan_out, shape, DType.float32, seed_val=222)

    # Should be different (at least some values)
    var differences = 0
    for i in range(w1.numel()):
        var val1 = w1._data.bitcast[Float32]()[i]
        var val2 = w2._data.bitcast[Float32]()[i]
        if val1 != val2:
            differences += 1

    # Expect most values to be different (allow some coincidental matches)
    assert_true(differences > w1.numel() // 2)


# ============================================================================
# Xavier Normal Tests
# ============================================================================


fn test_xavier_normal_shape() raises:
    """Test Xavier normal initialization preserves shape."""
    var fan_in = 100
    var fan_out = 50
    var shape = List[Int](fan_in, fan_out)
    var W = xavier_normal(fan_in, fan_out, shape, DType.float32)

    assert_equal(W.shape()[0], fan_in)
    assert_equal(W.shape()[1], fan_out)


fn test_xavier_normal_mean() raises:
    """Test Xavier normal has approximately zero mean."""
    var fan_in = 1000
    var fan_out = 1000
    var shape = List[Int](fan_in, fan_out)
    var W = xavier_normal(fan_in, fan_out, shape, DType.float32)

    var mean = compute_mean(W)

    assert_almost_equal(Float32(mean), Float32(0.0), tolerance=0.01)


fn test_xavier_normal_std() raises:
    """Test Xavier normal has approximately correct standard deviation."""
    var fan_in = 2000
    var fan_out = 2000
    var shape = List[Int](fan_in, fan_out)
    var W = xavier_normal(fan_in, fan_out, shape, DType.float32)

    var mean = compute_mean(W)
    var std_dev = compute_std(W, mean)

    # Xavier normal: std = sqrt(2 / (fan_in + fan_out))
    var expected_std = sqrt(2.0 / Float64(fan_in + fan_out))

    # Allow 10% tolerance
    assert_almost_equal(Float32(std_dev), Float32(expected_std), tolerance=Float32(expected_std) * 0.1)


fn test_xavier_normal_reproducibility() raises:
    """Test Xavier normal with fixed seed is reproducible."""
    var fan_in = 50
    var fan_out = 100
    var shape = List[Int](fan_in, fan_out)

    # Generate with same seed twice
    var w1 = xavier_normal(fan_in, fan_out, shape, DType.float32, seed_val=555)
    var w2 = xavier_normal(fan_in, fan_out, shape, DType.float32, seed_val=555)

    # Should be identical
    for i in range(w1.numel()):
        var val1 = w1._data.bitcast[Float32]()[i]
        var val2 = w2._data.bitcast[Float32]()[i]
        assert_equal(val1, val2)


fn test_xavier_configurations() raises:
    """Test Xavier initialization with various fan configurations."""
    # Test several configurations
    var configs = List[Tuple[Int, Int]]()
    configs.append((10, 10))     # Square
    configs.append((100, 50))    # Wide
    configs.append((50, 100))    # Tall
    configs.append((784, 128))   # Typical NN layer
    configs.append((1, 1000))    # Extreme aspect ratio

    for idx in range(len(configs)):
        var fan_in = configs[idx][0]
        var fan_out = configs[idx][1]
        var shape = List[Int](fan_in, fan_out)

        # Test uniform
        var w_uniform = xavier_uniform(fan_in, fan_out, shape, DType.float32, seed_val=42)
        var bound = sqrt(6.0 / Float64(fan_in + fan_out))

        # Check bounds
        for i in range(w_uniform.numel()):
            var val = Float64(w_uniform._data.bitcast[Float32]()[i])
            assert_true(val >= -bound and val <= bound)

        # Test normal
        var w_normal = xavier_normal(fan_in, fan_out, shape, DType.float32, seed_val=42)
        var expected_var = 2.0 / Float64(fan_in + fan_out)
        var mean = compute_mean(w_normal)
        var actual_var = compute_variance(w_normal, mean)

        # Variance should be reasonable (within 25% for smaller samples)
        var tolerance = expected_var * 0.25
        var diff = abs(actual_var - expected_var)
        assert_true(diff < tolerance)


# ============================================================================
# Kaiming Uniform Tests
# ============================================================================


fn test_kaiming_uniform_shape() raises:
    """Test Kaiming uniform initialization preserves shape."""
    var fan_in = 100
    var fan_out = 50
    var shape = List[Int](fan_in, fan_out)
    var W = kaiming_uniform(fan_in, fan_out, shape, "fan_in", DType.float32)

    assert_equal(W.shape()[0], fan_in)
    assert_equal(W.shape()[1], fan_out)


fn test_kaiming_uniform_range() raises:
    """Test Kaiming uniform values are within expected range."""
    var fan_in = 1000
    var fan_out = 500
    var shape = List[Int](fan_in, fan_out)
    var W = kaiming_uniform(fan_in, fan_out, shape, "fan_in", DType.float32)

    # Kaiming uniform limit: sqrt(6 / fan_in)
    var limit = sqrt(6.0 / Float64(fan_in))

    var result = compute_min_max(W)
    var min_val = result[0]
    var max_val = result[1]

    # All values should be in [-limit, limit]
    assert_true(min_val >= -limit - 0.01)
    assert_true(max_val <= limit + 0.01)


fn test_kaiming_uniform_mean() raises:
    """Test Kaiming uniform has approximately zero mean."""
    var fan_in = 1000
    var fan_out = 1000
    var shape = List[Int](fan_in, fan_out)
    var W = kaiming_uniform(fan_in, fan_out, shape, "fan_in", DType.float32)

    var mean = compute_mean(W)

    assert_almost_equal(Float32(mean), Float32(0.0), tolerance=0.01)


fn test_kaiming_uniform_variance_fan_in() raises:
    """Test Kaiming uniform has correct variance with fan_in mode."""
    var fan_in = 2000
    var fan_out = 2000
    var shape = List[Int](fan_in, fan_out)
    var W = kaiming_uniform(fan_in, fan_out, shape, "fan_in", DType.float32)

    var mean = compute_mean(W)
    var std_dev = compute_std(W, mean)

    # Expected std = sqrt(2 / fan_in)
    var expected_std = sqrt(2.0 / Float64(fan_in))

    # Allow 10% tolerance
    assert_almost_equal(Float32(std_dev), Float32(expected_std), tolerance=Float32(expected_std) * 0.1)


fn test_kaiming_uniform_variance_fan_out() raises:
    """Test Kaiming uniform has correct variance with fan_out mode."""
    var fan_in = 100
    var fan_out = 50
    var shape = List[Int](fan_in, fan_out)
    var weights = kaiming_uniform(fan_in, fan_out, shape, "fan_out", DType.float32, seed_val=42)

    # Expected variance: 2/fan_out = 2/50 = 0.04
    var expected_var = 2.0 / Float64(fan_out)
    var mean = compute_mean(weights)
    var actual_var = compute_variance(weights, mean)

    var tolerance = expected_var * 0.1
    var diff = abs(actual_var - expected_var)

    assert_true(diff < tolerance)


fn test_kaiming_uniform_reproducibility() raises:
    """Test Kaiming uniform with fixed seed is reproducible."""
    var fan_in = 50
    var fan_out = 100
    var shape = List[Int](fan_in, fan_out)

    # Generate with same seed twice
    var w1 = kaiming_uniform(fan_in, fan_out, shape, "fan_in", DType.float32, seed_val=999)
    var w2 = kaiming_uniform(fan_in, fan_out, shape, "fan_in", DType.float32, seed_val=999)

    # Should be identical
    for i in range(w1.numel()):
        var val1 = w1._data.bitcast[Float32]()[i]
        var val2 = w2._data.bitcast[Float32]()[i]
        assert_equal(val1, val2)


# ============================================================================
# Kaiming Normal Tests
# ============================================================================


fn test_kaiming_normal_shape() raises:
    """Test Kaiming normal initialization preserves shape."""
    var fan_in = 100
    var fan_out = 50
    var shape = List[Int](fan_in, fan_out)
    var W = kaiming_normal(fan_in, fan_out, shape, "fan_in", DType.float32)

    assert_equal(W.shape()[0], fan_in)
    assert_equal(W.shape()[1], fan_out)


fn test_kaiming_normal_mean() raises:
    """Test Kaiming normal has approximately zero mean."""
    var fan_in = 1000
    var fan_out = 1000
    var shape = List[Int](fan_in, fan_out)
    var W = kaiming_normal(fan_in, fan_out, shape, "fan_in", DType.float32)

    var mean = compute_mean(W)

    assert_almost_equal(Float32(mean), Float32(0.0), tolerance=0.01)


fn test_kaiming_normal_std() raises:
    """Test Kaiming normal has approximately correct standard deviation."""
    var fan_in = 2000
    var fan_out = 2000
    var shape = List[Int](fan_in, fan_out)
    var W = kaiming_normal(fan_in, fan_out, shape, "fan_in", DType.float32)

    var mean = compute_mean(W)
    var std_dev = compute_std(W, mean)

    # Kaiming normal: std = sqrt(2 / fan_in)
    var expected_std = sqrt(2.0 / Float64(fan_in))

    # Allow 10% tolerance
    assert_almost_equal(Float32(std_dev), Float32(expected_std), tolerance=Float32(expected_std) * 0.1)


fn test_kaiming_normal_reproducibility() raises:
    """Test Kaiming normal with fixed seed is reproducible."""
    var fan_in = 50
    var fan_out = 100
    var shape = List[Int](fan_in, fan_out)

    # Generate with same seed twice
    var w1 = kaiming_normal(fan_in, fan_out, shape, "fan_in", DType.float32, seed_val=555)
    var w2 = kaiming_normal(fan_in, fan_out, shape, "fan_in", DType.float32, seed_val=555)

    # Should be identical
    for i in range(w1.numel()):
        var val1 = w1._data.bitcast[Float32]()[i]
        var val2 = w2._data.bitcast[Float32]()[i]
        assert_equal(val1, val2)


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

    var result = compute_min_max(W)
    var min_val = result[0]
    var max_val = result[1]

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
    assert_almost_equal(Float32(mean), Float32(expected_mean), tolerance=0.05)


fn test_uniform_reproducibility() raises:
    """Test uniform with fixed seed is reproducible."""
    var shape = List[Int](50, 50)

    # Generate with same seed twice
    var w1 = uniform(shape, -0.2, 0.2, DType.float32, seed_val=999)
    var w2 = uniform(shape, -0.2, 0.2, DType.float32, seed_val=999)

    # Should be identical
    for i in range(w1.numel()):
        var val1 = w1._data.bitcast[Float32]()[i]
        var val2 = w2._data.bitcast[Float32]()[i]
        assert_equal(val1, val2)


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
    assert_almost_equal(Float32(actual_mean), Float32(target_mean), tolerance=0.1)


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
    assert_almost_equal(Float32(actual_std), Float32(target_std), tolerance=Float32(target_std) * 0.1)


fn test_normal_reproducibility() raises:
    """Test normal with fixed seed is reproducible."""
    var shape = List[Int](50, 50)

    # Generate with same seed twice
    var w1 = normal(shape, 0.0, 0.05, DType.float32, seed_val=555)
    var w2 = normal(shape, 0.0, 0.05, DType.float32, seed_val=555)

    # Should be identical
    for i in range(w1.numel()):
        var val1 = w1._data.bitcast[Float32]()[i]
        var val2 = w2._data.bitcast[Float32]()[i]
        assert_equal(val1, val2)


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


fn test_constant_ones_and_zeros() raises:
    """Test constant can create ones and zeros."""
    var shape = List[Int](5, 5)

    # Test ones
    var ones_tensor = constant(shape, 1.0, DType.float32)
    for i in range(ones_tensor.numel()):
        var val = Float64(ones_tensor._data.bitcast[Float32]()[i])
        assert_equal(val, 1.0)

    # Test zeros
    var zeros_tensor = constant(shape, 0.0, DType.float32)
    for i in range(zeros_tensor.numel()):
        var val = Float64(zeros_tensor._data.bitcast[Float32]()[i])
        assert_equal(val, 0.0)


# ============================================================================
# Dtype Support Tests
# ============================================================================


fn test_xavier_uniform_float64() raises:
    """Test Xavier uniform with float64 dtype."""
    var fan_in = 100
    var fan_out = 50
    var shape = List[Int](fan_in, fan_out)
    var W = xavier_uniform(fan_in, fan_out, shape, DType.float64)

    assert_true(W._dtype == DType.float64, "Xavier uniform should have float64 dtype")
    assert_equal(W.shape()[0], 100)
    assert_equal(W.shape()[1], 50)

    # Verify variance
    var expected_var = 2.0 / Float64(fan_in + fan_out)
    var mean = compute_mean(W)
    var actual_var = compute_variance(W, mean)
    var tolerance = expected_var * 0.1
    assert_true(abs(actual_var - expected_var) < tolerance)


fn test_xavier_normal_float16() raises:
    """Test Xavier normal with float16 dtype."""
    var fan_in = 100
    var fan_out = 50
    var shape = List[Int](fan_in, fan_out)
    var W = xavier_normal(fan_in, fan_out, shape, DType.float16, seed_val=42)

    assert_equal(W._dtype, DType.float16)

    # Check variance for float16 (with looser tolerance)
    var expected_var = 2.0 / Float64(fan_in + fan_out)
    var mean = compute_mean(W)
    var actual_var = compute_variance(W, mean)
    var tolerance = expected_var * 0.15  # Float16 has less precision
    assert_true(abs(actual_var - expected_var) < tolerance)


fn test_kaiming_normal_float64() raises:
    """Test Kaiming normal with float64 dtype."""
    var fan_in = 100
    var fan_out = 50
    var shape = List[Int](fan_in, fan_out)
    var W = kaiming_normal(fan_in, fan_out, shape, "fan_in", DType.float64)

    assert_true(W._dtype == DType.float64, "Kaiming normal should have float64 dtype")
    assert_equal(W.shape()[0], 100)
    assert_equal(W.shape()[1], 50)

    # Check variance for float64
    var expected_var = 2.0 / Float64(fan_in)
    var mean = compute_mean(W)
    var actual_var = compute_variance(W, mean)
    var tolerance = expected_var * 0.1
    assert_true(abs(actual_var - expected_var) < tolerance)


fn test_uniform_float64() raises:
    """Test uniform with float64 dtype."""
    var shape = List[Int](50, 50)
    var weights = uniform(shape, -1.0, 1.0, DType.float64, seed_val=42)

    assert_equal(weights._dtype, DType.float64)

    # Check bounds
    for i in range(weights.numel()):
        var val = weights._data.bitcast[Float64]()[i]
        assert_true(val >= -1.0 and val <= 1.0)


fn test_normal_float64() raises:
    """Test normal with float64 dtype."""
    var shape = List[Int](50, 50)
    var weights = normal(shape, 0.0, 0.1, DType.float64, seed_val=42)

    assert_equal(weights._dtype, DType.float64)


fn test_constant_float64() raises:
    """Test constant initialization with float64 dtype."""
    var shape = List[Int](2)
    shape[0] = 10
    shape[1] = 10
    var W = constant(shape, 1.5, DType.float64)

    assert_true(W._dtype == DType.float64, "Constant should have float64 dtype")

    for i in range(100):
        assert_almost_equal(Float32(W._data.bitcast[Float64]()[i]), Float32(1.5), tolerance=1e-5)


# ============================================================================
# Edge Case Tests
# ============================================================================


fn test_small_dimensions() raises:
    """Test initialization with small dimensions."""
    # Xavier with fan_in=1, fan_out=1
    var shape1 = List[Int](1, 1)
    var W1 = xavier_uniform(1, 1, shape1, DType.float32)
    assert_equal(W1.numel(), 1)

    # Kaiming with fan_in=2
    var shape2 = List[Int](2, 3)
    var W2 = kaiming_uniform(2, 3, shape2, "fan_in", DType.float32)
    assert_equal(W2.shape()[0], 2)
    assert_equal(W2.shape()[1], 3)


fn test_rectangular_matrices() raises:
    """Test initialization with non-square matrices."""
    # Tall matrix (more rows than columns)
    var shape_tall = List[Int](1000, 10)
    var W_tall = xavier_uniform(1000, 10, shape_tall, DType.float32)
    assert_equal(W_tall.shape()[0], 1000)
    assert_equal(W_tall.shape()[1], 10)

    # Wide matrix (more columns than rows)
    var shape_wide = List[Int](10, 1000)
    var W_wide = kaiming_uniform(10, 1000, shape_wide, "fan_in", DType.float32)
    assert_equal(W_wide.shape()[0], 10)
    assert_equal(W_wide.shape()[1], 1000)


fn test_large_initialization() raises:
    """Test initialization with large dimensions."""
    # Large matrix
    var fan_in = 5000
    var fan_out = 5000
    var shape = List[Int](fan_in, fan_out)
    var W = xavier_uniform(fan_in, fan_out, shape, DType.float32)

    assert_equal(W.shape()[0], 5000)
    assert_equal(W.shape()[1], 5000)
    assert_equal(W.numel(), 25000000)

    # Verify statistical properties still hold
    var mean = compute_mean(W)
    assert_almost_equal(Float32(mean), Float32(0.0), tolerance=0.01)
