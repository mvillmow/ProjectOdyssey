"""Shared test fixtures and utilities for the shared library test suite.

This module provides:
- Common assertion functions for testing
- Test fixtures for creating test data
- Utilities for test setup and teardown
"""

from memory import memset_zero
from random import seed, randn, randint


# ============================================================================
# Assertion Functions
# ============================================================================


fn assert_true(condition: Bool, message: String = "Assertion failed") raises:
    """Assert that condition is true.

    Args:
        condition: The boolean condition to check.
        message: Optional error message.

    Raises:
        Error if condition is false.
    """
    if not condition:
        raise Error(message)


fn assert_false(condition: Bool, message: String = "Assertion failed") raises:
    """Assert that condition is false.

    Args:
        condition: The boolean condition to check.
        message: Optional error message.

    Raises:
        Error if condition is true.
    """
    if condition:
        raise Error(message)


fn assert_equal[T: Comparable](a: T, b: T, message: String = "") raises:
    """Assert exact equality of two values.

    Args:
        a: First value.
        b: Second value.
        message: Optional error message.

    Raises:
        Error if a != b.
    """
    if a != b:
        var error_msg = message if message else "Values are not equal"
        raise Error(error_msg)


fn assert_not_equal[T: Comparable](a: T, b: T, message: String = "") raises:
    """Assert inequality of two values.

    Args:
        a: First value.
        b: Second value.
        message: Optional error message.

    Raises:
        Error if a == b.
    """
    if a == b:
        var error_msg = (
            message if message else "Values are equal but should not be"
        )
        raise Error(error_msg)


fn assert_almost_equal(
    a: Float32, b: Float32, tolerance: Float32 = 1e-6, message: String = ""
) raises:
    """Assert floating point near-equality.

    Args:
        a: First value.
        b: Second value.
        tolerance: Maximum allowed difference.
        message: Optional error message.

    Raises:
        Error if |a - b| > tolerance.
    """
    var diff = abs(a - b)
    if diff > tolerance:
        var error_msg = message if message else (
            String(a) + " !≈ " + String(b) + " (diff: " + String(diff) + ")"
        )
        raise Error(error_msg)


fn assert_greater(a: Float32, b: Float32, message: String = "") raises:
    """Assert a > b.

    Args:
        a: First value.
        b: Second value.
        message: Optional error message.

    Raises:
        Error if a <= b.
    """
    if a <= b:
        var error_msg = message if message else String(a) + " <= " + String(b)
        raise Error(error_msg)


fn assert_less(a: Float32, b: Float32, message: String = "") raises:
    """Assert a < b.

    Args:
        a: First value.
        b: Second value.
        message: Optional error message.

    Raises:
        Error if a >= b.
    """
    if a >= b:
        var error_msg = message if message else String(a) + " >= " + String(b)
        raise Error(error_msg)


fn assert_shape_equal(shape1: List[Int], shape2: List[Int], message: String = "") raises:
    """Assert two shapes are equal.

    Args:
        shape1: First shape.
        shape2: Second shape.
        message: Optional error message.

    Raises:
        Error if shapes are not equal.
    """
    if len(shape1) != len(shape2):
        var error_msg = message if message else (
            "Shape dimensions differ: " + String(len(shape1)) + " vs " + String(len(shape2))
        )
        raise Error(error_msg)

    for i in range(len(shape1)):
        if shape1[i] != shape2[i]:
            var error_msg = message if message else (
                "Shape mismatch at dimension " + String(i) + ": " +
                String(shape1[i]) + " vs " + String(shape2[i])
            )
            raise Error(error_msg)


# ============================================================================
# Test Fixtures
# ============================================================================


struct TestFixtures:
    """Collection of reusable test fixtures and utilities.

    This struct provides static methods for creating common test data,
    models, and scenarios used across multiple tests.
    """

    @staticmethod
    fn deterministic_seed() -> Int:
        """Get deterministic random seed for reproducible tests.

        Returns:
            Fixed random seed value (42).
        """
        return 42

    @staticmethod
    fn set_seed():
        """Set random seed for deterministic test execution."""
        seed(Self.deterministic_seed())

    # FIXME: Placeholder tensor fixture methods in tests/shared/conftest.mojo
    # TODO(#1538): Add tensor fixture methods when Tensor type is implemented
    # @staticmethod
    # fn small_tensor() -> Tensor:
    #     """Create small 3x3 tensor for unit tests."""
    #     pass

    # @staticmethod
    # fn random_tensor(rows: Int, cols: Int) -> Tensor:
    #     """Create random tensor with deterministic seed."""
    #     pass

    # FIXME: Placeholder model fixture methods in tests/shared/conftest.mojo
    # TODO(#1538): Add model fixture methods when models are implemented
    # @staticmethod
    # fn simple_linear_model() -> Linear:
    #     """Create simple Linear layer with known weights."""
    #     pass

    # FIXME: Placeholder dataset fixture methods in tests/shared/conftest.mojo
    # TODO(#1538): Add dataset fixture methods when datasets are implemented
    # @staticmethod
    # fn synthetic_dataset(n_samples: Int = 100) -> TensorDataset:
    #     """Create synthetic dataset for testing."""
    #     pass


# ============================================================================
# Benchmark Utilities
# ============================================================================


struct BenchmarkResult:
    """Results from a performance benchmark.

    Attributes:
        name: Benchmark name/description.
        duration_ms: Execution duration in milliseconds.
        throughput: Operations per second.
        memory_mb: Memory used in megabytes.
    """

    var name: String
    var duration_ms: Float64
    var throughput: Float64
    var memory_mb: Float64

    fn __init__(
        out self,
        name: String,
        duration_ms: Float64,
        throughput: Float64,
        memory_mb: Float64 = 0.0,
    ):
        """Initialize benchmark result.

        Args:
            name: Benchmark name.
            duration_ms: Duration in milliseconds.
            throughput: Operations per second.
            memory_mb: Memory usage in MB.
        """
        self.name = name
        self.duration_ms = duration_ms
        self.throughput = throughput
        self.memory_mb = memory_mb

    fn print_result(self):
        """Print benchmark result in readable format."""
        print("Benchmark:", self.name)
        print("  Duration:", self.duration_ms, "ms")
        print("  Throughput:", self.throughput, "ops/s")
        if self.memory_mb > 0:
            print("  Memory:", self.memory_mb, "MB")


fn print_benchmark_results(results: List[BenchmarkResult]):
    """Print all benchmark results.

    Args:
        results: List of benchmark results to print.
    """
    print("\n=== Benchmark Results ===")
    for result in results:
        result[].print_result()
        print()


# ============================================================================
# Test Helpers
# ============================================================================


fn measure_time[func: fn () raises -> None]() raises -> Float64:
    """Measure execution time of a function.

    Parameters:
        func: Function to measure (must be fn, not def).

    Returns:
        Execution time in milliseconds.

    FIXME: Placeholder implementation in tests/shared/conftest.mojo (line 250)
    Currently returns 0.0 - needs proper time measurement using Mojo's time module.
    """
    # TODO(#1538): Implement using Mojo's time module when available
    # For now, placeholder for TDD
    return 0.0


fn measure_throughput[
    func: fn () raises -> None
](n_iterations: Int) raises -> Float64:
    """Measure throughput of a function.

    Parameters:
        func: Function to measure.

    Args:
        n_iterations: Number of iterations to run.

    Returns:
        Operations per second.

    FIXME: Placeholder implementation in tests/shared/conftest.mojo (line 264)
    Currently relies on placeholder measure_time - needs proper time measurement.
    """
    # TODO(#1538): Implement using Mojo's time module when available
    var duration_ms = measure_time[func]()
    return Float64(n_iterations) / (duration_ms / 1000.0)


# ============================================================================
# Test Data Generators
# ============================================================================


fn create_test_vector(size: Int, value: Float32 = 1.0) -> List[Float32]:
    """Create test vector filled with specific value.

    Args:
        size: Vector size.
        value: Fill value.

    Returns:
        List of Float32 values.
    """
    var vec = List[Float32](capacity=size)
    for _ in range(size):
        vec.append(value)
    return vec


fn create_test_matrix(
    rows: Int, cols: Int, value: Float32 = 1.0
) -> List[List[Float32]]:
    """Create test matrix filled with specific value.

    Args:
        rows: Number of rows.
        cols: Number of columns.
        value: Fill value.

    Returns:
        2D list of Float32 values.
    """
    var matrix = List[List[Float32]](capacity=rows)
    for _ in range(rows):
        matrix.append(create_test_vector(cols, value))
    return matrix


fn create_sequential_vector(size: Int, start: Float32 = 0.0) -> List[Float32]:
    """Create vector with sequential values [start, start+1, ...].

    Args:
        size: Vector size.
        start: Starting value.

    Returns:
        List of sequential Float32 values.
    """
    var vec = List[Float32](capacity=size)
    for i in range(size):
        vec.append(start + Float32(i))
    return vec
