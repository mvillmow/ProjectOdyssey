"""Tests for weight initialization methods.

Comprehensive test suite for Xavier/Glorot initialization and other
weight initialization strategies.

Test coverage:
- #259: Xavier/Glorot tests (uniform and normal variants)
- #264: Kaiming/He tests (uniform and normal variants)
- #269: Uniform/Normal basic distribution tests

Testing strategy:
- Statistical properties: Verify variance matches theoretical values
- Reproducibility: Fixed seed produces identical results
- Range validation: Values within expected bounds
- Different configurations: Various fan_in/fan_out combinations
"""

from testing import assert_true, assert_false, assert_equal, assert_almost_equal
from math import sqrt
from shared.core import ExTensor, xavier_uniform, xavier_normal, kaiming_uniform, kaiming_normal, uniform, normal, constant


fn compute_variance(tensor: ExTensor) raises -> Float64:
    """Compute variance of tensor values.

    Args:
        tensor: Input tensor

    Returns:
        Variance of all elements
    """
    # Compute mean
    var sum: Float64 = 0.0
    for i in range(tensor._numel):
        if tensor._dtype == DType.float32:
            sum += Float64(tensor._data.bitcast[Float32]()[i])
        else:
            sum += tensor._data.bitcast[Float64]()[i]

    var mean = sum / Float64(tensor._numel)

    # Compute variance
    var var_sum: Float64 = 0.0
    for i in range(tensor._numel):
        var val: Float64
        if tensor._dtype == DType.float32:
            val = Float64(tensor._data.bitcast[Float32]()[i])
        else:
            val = tensor._data.bitcast[Float64]()[i]

        var diff = val - mean
        var_sum += diff * diff

    return var_sum / Float64(tensor._numel)


fn test_xavier_uniform_variance() raises:
    """Test Xavier uniform produces correct variance."""
    print("Testing Xavier uniform variance...")

    # Test configuration: fan_in=100, fan_out=50
    var fan_in = 100
    var fan_out = 50
    var shape = List[Int](fan_in, fan_out)

    var weights = xavier_uniform(fan_in, fan_out, shape, DType.float32, seed_val=42)

    # Expected variance: 2/(fan_in + fan_out) = 2/150 ≈ 0.0133
    var expected_var = 2.0 / Float64(fan_in + fan_out)
    var actual_var = compute_variance(weights)

    # Allow 10% tolerance for statistical variation
    var tolerance = expected_var * 0.1
    var diff = abs(actual_var - expected_var)

    print("  Expected variance:", expected_var)
    print("  Actual variance:  ", actual_var)
    print("  Difference:       ", diff)
    print("  Tolerance:        ", tolerance)

    assert_true(diff < tolerance, "Xavier uniform variance should match theoretical value")

    print("  ✓ Xavier uniform variance test passed")


fn test_xavier_uniform_bounds() raises:
    """Test Xavier uniform values are within expected bounds."""
    print("Testing Xavier uniform bounds...")

    var fan_in = 100
    var fan_out = 50
    var shape = List[Int](fan_in, fan_out)

    var weights = xavier_uniform(fan_in, fan_out, shape, DType.float32, seed_val=123)

    # Expected bound: sqrt(6/(fan_in + fan_out)) = sqrt(6/150) ≈ 0.2
    var bound = sqrt(6.0 / Float64(fan_in + fan_out))

    # Check all values are within [-bound, bound]
    for i in range(weights._numel):
        var val = Float64(weights._data.bitcast[Float32]()[i])
        assert_true(val >= -bound and val <= bound, "Xavier uniform values should be in bounds")

    print("  Expected bound: ±", bound)
    print("  ✓ Xavier uniform bounds test passed")


fn test_xavier_uniform_reproducibility() raises:
    """Test Xavier uniform with fixed seed is reproducible."""
    print("Testing Xavier uniform reproducibility...")

    var fan_in = 50
    var fan_out = 100
    var shape = List[Int](fan_in, fan_out)

    # Generate with same seed twice
    var w1 = xavier_uniform(fan_in, fan_out, shape, DType.float32, seed_val=999)
    var w2 = xavier_uniform(fan_in, fan_out, shape, DType.float32, seed_val=999)

    # Should be identical
    for i in range(w1._numel):
        var val1 = w1._data.bitcast[Float32]()[i]
        var val2 = w2._data.bitcast[Float32]()[i]
        assert_equal(val1, val2, "Same seed should produce identical values")

    print("  ✓ Xavier uniform reproducibility test passed")


fn test_xavier_uniform_different_seeds() raises:
    """Test Xavier uniform with different seeds produces different results."""
    print("Testing Xavier uniform with different seeds...")

    var fan_in = 50
    var fan_out = 100
    var shape = List[Int](fan_in, fan_out)

    # Generate with different seeds
    var w1 = xavier_uniform(fan_in, fan_out, shape, DType.float32, seed_val=111)
    var w2 = xavier_uniform(fan_in, fan_out, shape, DType.float32, seed_val=222)

    # Should be different (at least some values)
    var differences = 0
    for i in range(w1._numel):
        var val1 = w1._data.bitcast[Float32]()[i]
        var val2 = w2._data.bitcast[Float32]()[i]
        if val1 != val2:
            differences += 1

    # Expect most values to be different (allow some coincidental matches)
    assert_true(differences > w1._numel // 2, "Different seeds should produce different values")

    print("  Differences: ", differences, " out of ", w1._numel)
    print("  ✓ Xavier uniform different seeds test passed")


fn test_xavier_normal_variance() raises:
    """Test Xavier normal produces correct variance."""
    print("Testing Xavier normal variance...")

    # Test configuration: fan_in=200, fan_out=100
    var fan_in = 200
    var fan_out = 100
    var shape = List[Int](fan_in, fan_out)

    var weights = xavier_normal(fan_in, fan_out, shape, DType.float32, seed_val=42)

    # Expected variance: 2/(fan_in + fan_out) = 2/300 ≈ 0.00667
    var expected_var = 2.0 / Float64(fan_in + fan_out)
    var actual_var = compute_variance(weights)

    # Allow 15% tolerance for statistical variation (normal has more variance)
    var tolerance = expected_var * 0.15
    var diff = abs(actual_var - expected_var)

    print("  Expected variance:", expected_var)
    print("  Actual variance:  ", actual_var)
    print("  Difference:       ", diff)
    print("  Tolerance:        ", tolerance)

    assert_true(diff < tolerance, "Xavier normal variance should match theoretical value")

    print("  ✓ Xavier normal variance test passed")


fn test_xavier_normal_mean_zero() raises:
    """Test Xavier normal has mean close to zero."""
    print("Testing Xavier normal mean...")

    var fan_in = 100
    var fan_out = 100
    var shape = List[Int](fan_in, fan_out)

    var weights = xavier_normal(fan_in, fan_out, shape, DType.float32, seed_val=789)

    # Compute mean
    var sum: Float64 = 0.0
    for i in range(weights._numel):
        sum += Float64(weights._data.bitcast[Float32]()[i])

    var mean = sum / Float64(weights._numel)

    print("  Mean: ", mean)

    # Mean should be close to 0 (within 1% of std for large sample)
    var std = sqrt(2.0 / Float64(fan_in + fan_out))
    var tolerance = std * 0.01

    assert_true(abs(mean) < tolerance, "Xavier normal should have mean ≈ 0")

    print("  ✓ Xavier normal mean test passed")


fn test_xavier_normal_reproducibility() raises:
    """Test Xavier normal with fixed seed is reproducible."""
    print("Testing Xavier normal reproducibility...")

    var fan_in = 50
    var fan_out = 100
    var shape = List[Int](fan_in, fan_out)

    # Generate with same seed twice
    var w1 = xavier_normal(fan_in, fan_out, shape, DType.float32, seed_val=555)
    var w2 = xavier_normal(fan_in, fan_out, shape, DType.float32, seed_val=555)

    # Should be identical
    for i in range(w1._numel):
        var val1 = w1._data.bitcast[Float32]()[i]
        var val2 = w2._data.bitcast[Float32]()[i]
        assert_equal(val1, val2, "Same seed should produce identical values")

    print("  ✓ Xavier normal reproducibility test passed")


fn test_xavier_configurations() raises:
    """Test Xavier initialization with various fan configurations."""
    print("Testing Xavier with various configurations...")

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
        for i in range(w_uniform._numel):
            var val = Float64(w_uniform._data.bitcast[Float32]()[i])
            assert_true(val >= -bound and val <= bound, "Values should be in bounds")

        # Test normal
        var w_normal = xavier_normal(fan_in, fan_out, shape, DType.float32, seed_val=42)
        var expected_var = 2.0 / Float64(fan_in + fan_out)
        var actual_var = compute_variance(w_normal)

        # Variance should be reasonable (within 25% for smaller samples)
        var tolerance = expected_var * 0.25
        var diff = abs(actual_var - expected_var)
        assert_true(diff < tolerance, "Variance should be reasonable")

    print("  Tested ", len(configs), " configurations")
    print("  ✓ Xavier configurations test passed")


fn test_xavier_float64() raises:
    """Test Xavier initialization with float64 dtype."""
    print("Testing Xavier with float64...")

    var fan_in = 100
    var fan_out = 50
    var shape = List[Int](fan_in, fan_out)

    # Test uniform with float64
    var w_uniform = xavier_uniform(fan_in, fan_out, shape, DType.float64, seed_val=42)
    assert_equal(w_uniform._dtype, DType.float64, "Should use float64 dtype")

    # Test normal with float64
    var w_normal = xavier_normal(fan_in, fan_out, shape, DType.float64, seed_val=42)
    assert_equal(w_normal._dtype, DType.float64, "Should use float64 dtype")

    # Check variance for uniform
    var expected_var = 2.0 / Float64(fan_in + fan_out)
    var actual_var_uniform = compute_variance(w_uniform)
    var tolerance = expected_var * 0.1
    assert_true(abs(actual_var_uniform - expected_var) < tolerance, "Float64 variance should match")

    print("  ✓ Xavier float64 test passed")


fn test_xavier_float16() raises:
    """Test Xavier initialization with float16 dtype."""
    print("Testing Xavier with float16...")

    var fan_in = 100
    var fan_out = 50
    var shape = List[Int](fan_in, fan_out)

    # Test uniform with float16
    var w_uniform = xavier_uniform(fan_in, fan_out, shape, DType.float16, seed_val=42)
    assert_equal(w_uniform._dtype, DType.float16, "Should use float16 dtype")

    # Test normal with float16
    var w_normal = xavier_normal(fan_in, fan_out, shape, DType.float16, seed_val=42)
    assert_equal(w_normal._dtype, DType.float16, "Should use float16 dtype")

    # Check variance for uniform (with looser tolerance for float16)
    var expected_var = 2.0 / Float64(fan_in + fan_out)
    var actual_var_uniform = compute_variance(w_uniform)
    var tolerance = expected_var * 0.15  # Float16 has less precision
    assert_true(abs(actual_var_uniform - expected_var) < tolerance, "Float16 variance should match")

    print("  ✓ Xavier float16 test passed")


fn test_kaiming_uniform_variance_fan_in() raises:
    """Test Kaiming uniform produces correct variance with fan_in mode."""
    print("Testing Kaiming uniform variance (fan_in)...")

    # Test configuration: fan_in=100, fan_out=50
    var fan_in = 100
    var fan_out = 50
    var shape = List[Int](fan_in, fan_out)

    var weights = kaiming_uniform(fan_in, fan_out, shape, fan_mode="fan_in", seed_val=42)

    # Expected variance: 2/fan_in = 2/100 = 0.02
    var expected_var = 2.0 / Float64(fan_in)
    var actual_var = compute_variance(weights)

    # Allow 10% tolerance for statistical variation
    var tolerance = expected_var * 0.1
    var diff = abs(actual_var - expected_var)

    print("  Expected variance:", expected_var)
    print("  Actual variance:  ", actual_var)
    print("  Difference:       ", diff)
    print("  Tolerance:        ", tolerance)

    assert_true(diff < tolerance, "Kaiming uniform variance should match theoretical value")

    print("  ✓ Kaiming uniform variance (fan_in) test passed")


fn test_kaiming_uniform_variance_fan_out() raises:
    """Test Kaiming uniform produces correct variance with fan_out mode."""
    print("Testing Kaiming uniform variance (fan_out)...")

    var fan_in = 100
    var fan_out = 50
    var shape = List[Int](fan_in, fan_out)

    var weights = kaiming_uniform(fan_in, fan_out, shape, fan_mode="fan_out", seed_val=42)

    # Expected variance: 2/fan_out = 2/50 = 0.04
    var expected_var = 2.0 / Float64(fan_out)
    var actual_var = compute_variance(weights)

    var tolerance = expected_var * 0.1
    var diff = abs(actual_var - expected_var)

    print("  Expected variance:", expected_var)
    print("  Actual variance:  ", actual_var)

    assert_true(diff < tolerance, "Kaiming uniform variance should match (fan_out)")

    print("  ✓ Kaiming uniform variance (fan_out) test passed")


fn test_kaiming_uniform_bounds() raises:
    """Test Kaiming uniform values are within expected bounds."""
    print("Testing Kaiming uniform bounds...")

    var fan_in = 100
    var fan_out = 50
    var shape = List[Int](fan_in, fan_out)

    var weights = kaiming_uniform(fan_in, fan_out, shape, fan_mode="fan_in", seed_val=123)

    # Expected bound: sqrt(6/fan_in) = sqrt(6/100) ≈ 0.245
    var bound = sqrt(6.0 / Float64(fan_in))

    # Check all values are within [-bound, bound]
    for i in range(weights._numel):
        var val = Float64(weights._data.bitcast[Float32]()[i])
        assert_true(val >= -bound and val <= bound, "Kaiming uniform values should be in bounds")

    print("  Expected bound: ±", bound)
    print("  ✓ Kaiming uniform bounds test passed")


fn test_kaiming_uniform_reproducibility() raises:
    """Test Kaiming uniform with fixed seed is reproducible."""
    print("Testing Kaiming uniform reproducibility...")

    var fan_in = 50
    var fan_out = 100
    var shape = List[Int](fan_in, fan_out)

    # Generate with same seed twice
    var w1 = kaiming_uniform(fan_in, fan_out, shape, fan_mode="fan_in", seed_val=999)
    var w2 = kaiming_uniform(fan_in, fan_out, shape, fan_mode="fan_in", seed_val=999)

    # Should be identical
    for i in range(w1._numel):
        var val1 = w1._data.bitcast[Float32]()[i]
        var val2 = w2._data.bitcast[Float32]()[i]
        assert_equal(val1, val2, "Same seed should produce identical values")

    print("  ✓ Kaiming uniform reproducibility test passed")


fn test_kaiming_normal_variance_fan_in() raises:
    """Test Kaiming normal produces correct variance with fan_in mode."""
    print("Testing Kaiming normal variance (fan_in)...")

    # Test configuration: fan_in=200, fan_out=100
    var fan_in = 200
    var fan_out = 100
    var shape = List[Int](fan_in, fan_out)

    var weights = kaiming_normal(fan_in, fan_out, shape, fan_mode="fan_in", seed_val=42)

    # Expected variance: 2/fan_in = 2/200 = 0.01
    var expected_var = 2.0 / Float64(fan_in)
    var actual_var = compute_variance(weights)

    # Allow 15% tolerance for statistical variation (normal has more variance)
    var tolerance = expected_var * 0.15
    var diff = abs(actual_var - expected_var)

    print("  Expected variance:", expected_var)
    print("  Actual variance:  ", actual_var)
    print("  Difference:       ", diff)
    print("  Tolerance:        ", tolerance)

    assert_true(diff < tolerance, "Kaiming normal variance should match theoretical value")

    print("  ✓ Kaiming normal variance (fan_in) test passed")


fn test_kaiming_normal_mean_zero() raises:
    """Test Kaiming normal has mean close to zero."""
    print("Testing Kaiming normal mean...")

    var fan_in = 100
    var fan_out = 100
    var shape = List[Int](fan_in, fan_out)

    var weights = kaiming_normal(fan_in, fan_out, shape, fan_mode="fan_in", seed_val=789)

    # Compute mean
    var sum: Float64 = 0.0
    for i in range(weights._numel):
        sum += Float64(weights._data.bitcast[Float32]()[i])

    var mean = sum / Float64(weights._numel)

    print("  Mean: ", mean)

    # Mean should be close to 0 (within 1% of std for large sample)
    var std = sqrt(2.0 / Float64(fan_in))
    var tolerance = std * 0.01

    assert_true(abs(mean) < tolerance, "Kaiming normal should have mean ≈ 0")

    print("  ✓ Kaiming normal mean test passed")


fn test_kaiming_normal_reproducibility() raises:
    """Test Kaiming normal with fixed seed is reproducible."""
    print("Testing Kaiming normal reproducibility...")

    var fan_in = 50
    var fan_out = 100
    var shape = List[Int](fan_in, fan_out)

    # Generate with same seed twice
    var w1 = kaiming_normal(fan_in, fan_out, shape, fan_mode="fan_in", seed_val=555)
    var w2 = kaiming_normal(fan_in, fan_out, shape, fan_mode="fan_in", seed_val=555)

    # Should be identical
    for i in range(w1._numel):
        var val1 = w1._data.bitcast[Float32]()[i]
        var val2 = w2._data.bitcast[Float32]()[i]
        assert_equal(val1, val2, "Same seed should produce identical values")

    print("  ✓ Kaiming normal reproducibility test passed")


fn test_kaiming_float64() raises:
    """Test Kaiming initialization with float64 dtype."""
    print("Testing Kaiming with float64...")

    var fan_in = 100
    var fan_out = 50
    var shape = List[Int](fan_in, fan_out)

    # Test uniform with float64
    var w_uniform = kaiming_uniform(fan_in, fan_out, shape, fan_mode="fan_in", dtype=DType.float64, seed_val=42)
    assert_equal(w_uniform._dtype, DType.float64, "Should use float64 dtype")

    # Test normal with float64
    var w_normal = kaiming_normal(fan_in, fan_out, shape, fan_mode="fan_in", dtype=DType.float64, seed_val=42)
    assert_equal(w_normal._dtype, DType.float64, "Should use float64 dtype")

    # Check variance for uniform
    var expected_var = 2.0 / Float64(fan_in)
    var actual_var_uniform = compute_variance(w_uniform)
    var tolerance = expected_var * 0.1
    assert_true(abs(actual_var_uniform - expected_var) < tolerance, "Float64 variance should match")

    print("  ✓ Kaiming float64 test passed")


fn test_uniform_bounds() raises:
    """Test uniform distribution stays within bounds."""
    print("Testing uniform bounds...")

    var shape = List[Int](100, 50)
    var low = -0.5
    var high = 0.5

    var weights = uniform(shape, low=low, high=high, seed_val=42)

    # Check all values are within [low, high]
    for i in range(weights._numel):
        var val = Float64(weights._data.bitcast[Float32]()[i])
        assert_true(val >= low and val <= high, "Uniform values should be in bounds")

    print("  Expected bounds: [", low, ",", high, "]")
    print("  ✓ Uniform bounds test passed")


fn test_uniform_mean() raises:
    """Test uniform distribution has correct mean."""
    print("Testing uniform mean...")

    var shape = List[Int](200, 100)
    var low = -1.0
    var high = 1.0

    var weights = uniform(shape, low=low, high=high, seed_val=123)

    # Compute mean
    var sum: Float64 = 0.0
    for i in range(weights._numel):
        sum += Float64(weights._data.bitcast[Float32]()[i])

    var mean = sum / Float64(weights._numel)
    var expected_mean = (low + high) / 2.0

    print("  Expected mean:", expected_mean)
    print("  Actual mean:  ", mean)

    # Mean should be close to (low + high) / 2
    var tolerance = (high - low) * 0.05  # 5% of range
    assert_true(abs(mean - expected_mean) < tolerance, "Uniform should have correct mean")

    print("  ✓ Uniform mean test passed")


fn test_uniform_reproducibility() raises:
    """Test uniform with fixed seed is reproducible."""
    print("Testing uniform reproducibility...")

    var shape = List[Int](50, 50)

    # Generate with same seed twice
    var w1 = uniform(shape, low=-0.2, high=0.2, seed_val=999)
    var w2 = uniform(shape, low=-0.2, high=0.2, seed_val=999)

    # Should be identical
    for i in range(w1._numel):
        var val1 = w1._data.bitcast[Float32]()[i]
        var val2 = w2._data.bitcast[Float32]()[i]
        assert_equal(val1, val2, "Same seed should produce identical values")

    print("  ✓ Uniform reproducibility test passed")


fn test_normal_mean_and_std() raises:
    """Test normal distribution has correct mean and standard deviation."""
    print("Testing normal mean and std...")

    var shape = List[Int](200, 100)
    var expected_mean = 0.5
    var expected_std = 0.1

    var weights = normal(shape, mean=expected_mean, std=expected_std, seed_val=42)

    # Compute mean
    var sum: Float64 = 0.0
    for i in range(weights._numel):
        sum += Float64(weights._data.bitcast[Float32]()[i])

    var actual_mean = sum / Float64(weights._numel)

    # Compute variance
    var var_sum: Float64 = 0.0
    for i in range(weights._numel):
        var val = Float64(weights._data.bitcast[Float32]()[i])
        var diff = val - actual_mean
        var_sum += diff * diff

    var actual_var = var_sum / Float64(weights._numel)
    var actual_std = sqrt(actual_var)

    print("  Expected mean:", expected_mean)
    print("  Actual mean:  ", actual_mean)
    print("  Expected std: ", expected_std)
    print("  Actual std:   ", actual_std)

    # Allow 10% tolerance for statistical variation
    var mean_tolerance = expected_std * 0.1
    var std_tolerance = expected_std * 0.1

    assert_true(abs(actual_mean - expected_mean) < mean_tolerance, "Normal should have correct mean")
    assert_true(abs(actual_std - expected_std) < std_tolerance, "Normal should have correct std")

    print("  ✓ Normal mean and std test passed")


fn test_normal_reproducibility() raises:
    """Test normal with fixed seed is reproducible."""
    print("Testing normal reproducibility...")

    var shape = List[Int](50, 50)

    # Generate with same seed twice
    var w1 = normal(shape, mean=0.0, std=0.05, seed_val=555)
    var w2 = normal(shape, mean=0.0, std=0.05, seed_val=555)

    # Should be identical
    for i in range(w1._numel):
        var val1 = w1._data.bitcast[Float32]()[i]
        var val2 = w2._data.bitcast[Float32]()[i]
        assert_equal(val1, val2, "Same seed should produce identical values")

    print("  ✓ Normal reproducibility test passed")


fn test_constant_values() raises:
    """Test constant initialization fills with correct value."""
    print("Testing constant values...")

    var shape = List[Int](10, 10)
    var value = 0.42

    var weights = constant(shape, value)

    # All values should equal the constant
    for i in range(weights._numel):
        var val = Float64(weights._data.bitcast[Float32]()[i])
        assert_equal(val, value, "All values should equal constant")

    print("  Constant value:", value)
    print("  ✓ Constant values test passed")


fn test_constant_ones_and_zeros() raises:
    """Test constant can create ones and zeros."""
    print("Testing constant ones and zeros...")

    var shape = List[Int](5, 5)

    # Test ones
    var ones_tensor = constant(shape, 1.0)
    for i in range(ones_tensor._numel):
        var val = Float64(ones_tensor._data.bitcast[Float32]()[i])
        assert_equal(val, 1.0, "Should be 1.0")

    # Test zeros
    var zeros_tensor = constant(shape, 0.0)
    for i in range(zeros_tensor._numel):
        var val = Float64(zeros_tensor._data.bitcast[Float32]()[i])
        assert_equal(val, 0.0, "Should be 0.0")

    print("  ✓ Constant ones and zeros test passed")


fn test_uniform_float64() raises:
    """Test uniform with float64 dtype."""
    print("Testing uniform with float64...")

    var shape = List[Int](50, 50)
    var weights = uniform(shape, low=-1.0, high=1.0, dtype=DType.float64, seed_val=42)

    assert_equal(weights._dtype, DType.float64, "Should use float64 dtype")

    # Check bounds
    for i in range(weights._numel):
        var val = weights._data.bitcast[Float64]()[i]
        assert_true(val >= -1.0 and val <= 1.0, "Values should be in bounds")

    print("  ✓ Uniform float64 test passed")


fn test_normal_float64() raises:
    """Test normal with float64 dtype."""
    print("Testing normal with float64...")

    var shape = List[Int](50, 50)
    var weights = normal(shape, mean=0.0, std=0.1, dtype=DType.float64, seed_val=42)

    assert_equal(weights._dtype, DType.float64, "Should use float64 dtype")

    print("  ✓ Normal float64 test passed")


fn test_constant_float64() raises:
    """Test constant with float64 dtype."""
    print("Testing constant with float64...")

    var shape = List[Int](5, 5)
    var weights = constant(shape, 0.5, dtype=DType.float64)

    assert_equal(weights._dtype, DType.float64, "Should use float64 dtype")

    for i in range(weights._numel):
        var val = weights._data.bitcast[Float64]()[i]
        assert_equal(val, 0.5, "Should be 0.5")

    print("  ✓ Constant float64 test passed")


fn main() raises:
    """Run all initializer tests."""
    print("\n" + "="*70)
    print("INITIALIZERS TEST SUITE")
    print("="*70 + "\n")

    print("Xavier Uniform Tests (#259)")
    print("-" * 70)
    test_xavier_uniform_variance()
    test_xavier_uniform_bounds()
    test_xavier_uniform_reproducibility()
    test_xavier_uniform_different_seeds()

    print("\nXavier Normal Tests (#259)")
    print("-" * 70)
    test_xavier_normal_variance()
    test_xavier_normal_mean_zero()
    test_xavier_normal_reproducibility()

    print("\nConfiguration Tests")
    print("-" * 70)
    test_xavier_configurations()
    test_xavier_float64()
    test_xavier_float16()

    print("\nKaiming Uniform Tests (#264)")
    print("-" * 70)
    test_kaiming_uniform_variance_fan_in()
    test_kaiming_uniform_variance_fan_out()
    test_kaiming_uniform_bounds()
    test_kaiming_uniform_reproducibility()

    print("\nKaiming Normal Tests (#264)")
    print("-" * 70)
    test_kaiming_normal_variance_fan_in()
    test_kaiming_normal_mean_zero()
    test_kaiming_normal_reproducibility()
    test_kaiming_float64()

    print("\nUniform Distribution Tests (#269)")
    print("-" * 70)
    test_uniform_bounds()
    test_uniform_mean()
    test_uniform_reproducibility()
    test_uniform_float64()

    print("\nNormal Distribution Tests (#269)")
    print("-" * 70)
    test_normal_mean_and_std()
    test_normal_reproducibility()
    test_normal_float64()

    print("\nConstant Initialization Tests (#269)")
    print("-" * 70)
    test_constant_values()
    test_constant_ones_and_zeros()
    test_constant_float64()

    print("\n" + "="*70)
    print("ALL INITIALIZER TESTS PASSED ✓")
    print("="*70 + "\n")
