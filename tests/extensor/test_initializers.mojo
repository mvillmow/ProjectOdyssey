"""Tests for weight initialization methods.

Comprehensive test suite for Xavier/Glorot initialization and other
weight initialization strategies.

Test coverage:
- #259: Xavier/Glorot tests (uniform and normal variants)

Testing strategy:
- Statistical properties: Verify variance matches theoretical values
- Reproducibility: Fixed seed produces identical results
- Range validation: Values within expected bounds
- Different configurations: Various fan_in/fan_out combinations
"""

from testing import assert_true, assert_false, assert_equal, assert_almost_equal
from collections.vector import DynamicVector
from math import abs, sqrt
from extensor import ExTensor, xavier_uniform, xavier_normal


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
    var shape = DynamicVector[Int](fan_in, fan_out)

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
    var shape = DynamicVector[Int](fan_in, fan_out)

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
    var shape = DynamicVector[Int](fan_in, fan_out)

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
    var shape = DynamicVector[Int](fan_in, fan_out)

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
    var shape = DynamicVector[Int](fan_in, fan_out)

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
    var shape = DynamicVector[Int](fan_in, fan_out)

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
    var shape = DynamicVector[Int](fan_in, fan_out)

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
        var shape = DynamicVector[Int](fan_in, fan_out)

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
    var shape = DynamicVector[Int](fan_in, fan_out)

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
    var shape = DynamicVector[Int](fan_in, fan_out)

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

    print("\n" + "="*70)
    print("ALL INITIALIZER TESTS PASSED ✓")
    print("="*70 + "\n")
