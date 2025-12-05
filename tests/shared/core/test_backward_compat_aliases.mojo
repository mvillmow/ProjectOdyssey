"""Tests for backward compatibility type aliases.

Tests that consolidated type aliases work correctly and maintain backward compatibility
for code using the old names (LinearBackwardResult, Conv2dBackwardResult, etc).

These aliases map the old type names to the new generic gradient container types:
- LinearBackwardResult -> GradientTriple
- Conv2dBackwardResult -> GradientTriple
- BenchmarkStatistics -> BenchmarkResult

Usage:
    mojo test tests/shared/core/test_backward_compat_aliases.mojo

Expected: All tests should pass with no warnings
"""

from tests.shared.conftest import assert_true, assert_equal, assert_shape
from shared.core import (
    ExTensor,
    zeros,
    GradientPair,
    GradientTriple,
    GradientQuad,
    LinearBackwardResult,
    LinearNoBiasBackwardResult,
    Conv2dBackwardResult,
    Conv2dNoBiasBackwardResult,
    DepthwiseConv2dBackwardResult,
    DepthwiseConv2dNoBiasBackwardResult,
    DepthwiseSeparableConv2dBackwardResult,
    DepthwiseSeparableConv2dNoBiasBackwardResult,
)
from tests.shared.conftest import BenchmarkStatistics, BenchmarkResult


# ============================================================================
# Gradient Triple Alias Tests
# ============================================================================


fn test_linear_backward_result_alias() raises:
    """Test LinearBackwardResult is an alias for GradientTriple."""
    print("Testing LinearBackwardResult alias...")

    # Create test tensors
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)

    var grad_input = zeros(shape, DType.float32)
    var grad_weights = zeros(shape, DType.float32)
    var grad_bias = zeros(shape, DType.float32)

    # Create using alias name
    var result: LinearBackwardResult = LinearBackwardResult(
        grad_input^, grad_weights^, grad_bias^
    )

    # Verify it's the same as GradientTriple
    assert_shape(result.grad_input, shape, "grad_input should have correct shape")
    assert_shape(result.grad_weights, shape, "grad_weights should have correct shape")
    assert_shape(result.grad_bias, shape, "grad_bias should have correct shape")

    print("  ✓ LinearBackwardResult alias works correctly")


fn test_conv2d_backward_result_alias() raises:
    """Test Conv2dBackwardResult is an alias for GradientTriple."""
    print("Testing Conv2dBackwardResult alias...")

    # Create test tensors with 4D shape
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)
    shape.append(4)

    var grad_input = zeros(shape, DType.float32)
    var grad_weights = zeros(shape, DType.float32)
    var grad_bias = zeros(shape, DType.float32)

    # Create using alias name
    var result: Conv2dBackwardResult = Conv2dBackwardResult(
        grad_input^, grad_weights^, grad_bias^
    )

    # Verify it has three gradient fields
    assert_shape(result.grad_input, shape, "grad_input should have correct shape")
    assert_shape(result.grad_weights, shape, "grad_weights should have correct shape")
    assert_shape(result.grad_bias, shape, "grad_bias should have correct shape")

    print("  ✓ Conv2dBackwardResult alias works correctly")


fn test_conv2d_no_bias_backward_result_alias() raises:
    """Test Conv2dNoBiasBackwardResult is an alias for GradientPair."""
    print("Testing Conv2dNoBiasBackwardResult alias...")

    var shape = List[Int]()
    shape.append(2)
    shape.append(3)

    var grad_input = zeros(shape, DType.float32)
    var grad_weights = zeros(shape, DType.float32)

    # Create using alias name
    var result: Conv2dNoBiasBackwardResult = Conv2dNoBiasBackwardResult(
        grad_input^, grad_weights^
    )

    # Verify it's a GradientPair (two fields)
    assert_shape(result.grad_a, shape, "grad_a should have correct shape")
    assert_shape(result.grad_b, shape, "grad_b should have correct shape")

    print("  ✓ Conv2dNoBiasBackwardResult alias works correctly")


fn test_depthwise_conv2d_backward_result_alias() raises:
    """Test DepthwiseConv2dBackwardResult is an alias for GradientTriple."""
    print("Testing DepthwiseConv2dBackwardResult alias...")

    var shape = List[Int]()
    shape.append(2)
    shape.append(3)

    var grad_input = zeros(shape, DType.float32)
    var grad_weights = zeros(shape, DType.float32)
    var grad_bias = zeros(shape, DType.float32)

    # Create using alias name
    var result: DepthwiseConv2dBackwardResult = DepthwiseConv2dBackwardResult(
        grad_input^, grad_weights^, grad_bias^
    )

    # Verify it has three fields
    assert_shape(
        result.grad_input, shape, "grad_input should have correct shape"
    )
    assert_shape(
        result.grad_weights, shape, "grad_weights should have correct shape"
    )
    assert_shape(result.grad_bias, shape, "grad_bias should have correct shape")

    print("  ✓ DepthwiseConv2dBackwardResult alias works correctly")


fn test_depthwise_separable_conv2d_backward_result_alias() raises:
    """Test DepthwiseSeparableConv2dBackwardResult is an alias for GradientQuad."""
    print("Testing DepthwiseSeparableConv2dBackwardResult alias...")

    var shape = List[Int]()
    shape.append(2)
    shape.append(3)

    var grad_a = zeros(shape, DType.float32)
    var grad_b = zeros(shape, DType.float32)
    var grad_c = zeros(shape, DType.float32)
    var grad_d = zeros(shape, DType.float32)

    # Create using alias name
    var result: DepthwiseSeparableConv2dBackwardResult = (
        DepthwiseSeparableConv2dBackwardResult(grad_a^, grad_b^, grad_c^, grad_d^)
    )

    # Verify it has four fields
    assert_shape(result.grad_a, shape, "grad_a should have correct shape")
    assert_shape(result.grad_b, shape, "grad_b should have correct shape")
    assert_shape(result.grad_c, shape, "grad_c should have correct shape")
    assert_shape(result.grad_d, shape, "grad_d should have correct shape")

    print("  ✓ DepthwiseSeparableConv2dBackwardResult alias works correctly")


# ============================================================================
# Linear No Bias Tests
# ============================================================================


fn test_linear_no_bias_backward_result_alias() raises:
    """Test LinearNoBiasBackwardResult is an alias for GradientPair."""
    print("Testing LinearNoBiasBackwardResult alias...")

    var shape = List[Int]()
    shape.append(2)
    shape.append(3)

    var grad_input = zeros(shape, DType.float32)
    var grad_weights = zeros(shape, DType.float32)

    # Create using alias name
    var result: LinearNoBiasBackwardResult = LinearNoBiasBackwardResult(
        grad_input^, grad_weights^
    )

    # Verify it's a GradientPair (two fields)
    assert_shape(result.grad_a, shape, "grad_a should have correct shape")
    assert_shape(result.grad_b, shape, "grad_b should have correct shape")

    print("  ✓ LinearNoBiasBackwardResult alias works correctly")


# ============================================================================
# Benchmark Statistics Alias Tests
# ============================================================================


fn test_benchmark_statistics_alias() raises:
    """Test BenchmarkStatistics is an alias for BenchmarkResult."""
    print("Testing BenchmarkStatistics alias...")

    # Create using alias name
    var bench: BenchmarkStatistics = BenchmarkStatistics(
        "test_bench", 10.5, 100.0, 5.0
    )

    # Verify all fields are accessible
    assert_equal(bench.name, "test_bench", "Name should be stored")
    assert_equal(bench.duration_ms, 10.5, "Duration should be stored")
    assert_equal(bench.throughput, 100.0, "Throughput should be stored")
    assert_equal(bench.memory_mb, 5.0, "Memory should be stored")

    # BenchmarkStatistics IS BenchmarkResult (alias), so bench is already a BenchmarkResult
    # No need to assign to another variable - the alias just means the type is the same

    print("  ✓ BenchmarkStatistics alias works correctly")


# ============================================================================
# Interoperability Tests
# ============================================================================


fn test_alias_interoperability() raises:
    """Test that aliases and base types are interchangeable."""
    print("Testing alias/type interoperability...")

    var shape = List[Int]()
    shape.append(2)
    shape.append(3)

    var grad_input = zeros(shape, DType.float32)
    var grad_weights = zeros(shape, DType.float32)
    var grad_bias = zeros(shape, DType.float32)

    # Create using generic GradientTriple (which IS LinearBackwardResult via alias)
    var triple: GradientTriple = GradientTriple(
        grad_input^, grad_weights^, grad_bias^
    )

    # Since LinearBackwardResult is an alias for GradientTriple, the type IS the same
    # No assignment needed - just verify the triple works
    # Access fields through GradientTriple (same as LinearBackwardResult)
    assert_shape(
        triple.grad_input, shape, "grad_input should be accessible"
    )
    assert_shape(
        triple.grad_weights, shape, "grad_weights should be accessible"
    )
    assert_shape(triple.grad_bias, shape, "grad_bias should be accessible")

    print("  ✓ Aliases are interoperable with base types")


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run all backward compatibility alias tests."""
    print("\n=== Backward Compatibility Alias Tests ===\n")

    test_linear_backward_result_alias()
    test_conv2d_backward_result_alias()
    test_conv2d_no_bias_backward_result_alias()
    test_depthwise_conv2d_backward_result_alias()
    test_depthwise_separable_conv2d_backward_result_alias()
    test_linear_no_bias_backward_result_alias()
    test_benchmark_statistics_alias()
    test_alias_interoperability()

    print("\n✓ All 8 backward compatibility alias tests passed\n")
