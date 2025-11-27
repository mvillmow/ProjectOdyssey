"""Shared test fixtures and utilities for the shared library test suite.

This module provides:
- Common assertion functions for testing
- Test fixtures for creating test data
- Utilities for test setup and teardown
"""

from memory import memset_zero
from random import seed, randn, randint
from math import isnan, isinf
from collections.optional import Optional
from shared.core.extensor import ExTensor
from tests.shared.fixtures.mock_models import SimpleMLP


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


fn assert_not_none[T: Copyable & Movable](value: Optional[T], message: String = "") raises:
    """Assert that an Optional value is not None.

    Args:
        value: The Optional value to check.
        message: Optional error message.

    Raises:
        Error if value is None.
    """
    if not value:
        var error_msg = message if message else "Value is None but should not be"
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


fn assert_almost_equal(
    a: Float64, b: Float64, tolerance: Float64 = 1e-6, message: String = ""
) raises:
    """Assert floating point near-equality for Float64.

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


fn assert_dtype_equal(a: DType, b: DType, message: String = "") raises:
    """Assert exact equality of DType values.

    Args:
        a: First DType.
        b: Second DType.
        message: Optional error message.

    Raises:
        Error if a != b.
    """
    if a != b:
        var error_msg = message if message else "DTypes are not equal"
        raise Error(error_msg)


fn assert_equal_int(a: Int, b: Int, message: String = "") raises:
    """Assert two integers are equal.

    Args:
        a: First integer.
        b: Second integer.
        message: Optional error message.

    Raises:
        Error if integers are not equal.
    """
    if a != b:
        var error_msg = message if message else (
            "Expected " + String(a) + " == " + String(b)
        )
        raise Error(error_msg)


fn assert_close_float(
    a: Float64,
    b: Float64,
    rtol: Float64 = 1e-5,
    atol: Float64 = 1e-8,
    message: String = ""
) raises:
    """Assert two floats are numerically close.

    Uses the formula: |a - b| <= atol + rtol * |b|

    Args:
        a: First float.
        b: Second float.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        message: Optional error message.

    Raises:
        Error if floats differ beyond tolerance.
    """
    # Handle NaN and inf
    var a_is_nan = isnan(a)
    var b_is_nan = isnan(b)
    var a_is_inf = isinf(a)
    var b_is_inf = isinf(a)

    if a_is_nan and b_is_nan:
        return  # Both NaN, considered equal

    if a_is_nan or b_is_nan:
        var error_msg = message if message else (
            "NaN mismatch: " + String(a) + " vs " + String(b)
        )
        raise Error(error_msg)

    if a_is_inf or b_is_inf:
        if a != b:
            var error_msg = message if message else (
                "Infinity mismatch: " + String(a) + " vs " + String(b)
            )
            raise Error(error_msg)
        return

    # Check numeric closeness
    var diff = a - b if a >= b else b - a
    var threshold = atol + rtol * (b if b >= 0 else -b)

    if diff > threshold:
        var error_msg = message if message else (
            "Values differ: " + String(a) + " vs " + String(b) +
            " (diff=" + String(diff) + ", threshold=" + String(threshold) + ")"
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


fn assert_greater(a: Float64, b: Float64, message: String = "") raises:
    """Assert a > b for Float64 values.

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


fn assert_greater(a: Int, b: Int, message: String = "") raises:
    """Assert a > b for Int values.

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


fn assert_less(a: Float64, b: Float64, message: String = "") raises:
    """Assert a < b for Float64 values.

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


fn assert_greater_or_equal(a: Float32, b: Float32, message: String = "") raises:
    """Assert a >= b for Float32 values.

    Args:
        a: First value.
        b: Second value.
        message: Optional error message.

    Raises:
        Error if a < b.
    """
    if a < b:
        var error_msg = message if message else String(a) + " < " + String(b)
        raise Error(error_msg)


fn assert_greater_or_equal(a: Float64, b: Float64, message: String = "") raises:
    """Assert a >= b for Float64 values.

    Args:
        a: First value.
        b: Second value.
        message: Optional error message.

    Raises:
        Error if a < b.
    """
    if a < b:
        var error_msg = message if message else String(a) + " < " + String(b)
        raise Error(error_msg)


fn assert_greater_or_equal(a: Int, b: Int, message: String = "") raises:
    """Assert a >= b for Int values.

    Args:
        a: First value.
        b: Second value.
        message: Optional error message.

    Raises:
        Error if a < b.
    """
    if a < b:
        var error_msg = message if message else String(a) + " < " + String(b)
        raise Error(error_msg)


fn assert_less_or_equal(a: Float32, b: Float32, message: String = "") raises:
    """Assert a <= b for Float32 values.

    Args:
        a: First value.
        b: Second value.
        message: Optional error message.

    Raises:
        Error if a > b.
    """
    if a > b:
        var error_msg = message if message else String(a) + " > " + String(b)
        raise Error(error_msg)


fn assert_less_or_equal(a: Float64, b: Float64, message: String = "") raises:
    """Assert a <= b for Float64 values.

    Args:
        a: First value.
        b: Second value.
        message: Optional error message.

    Raises:
        Error if a > b.
    """
    if a > b:
        var error_msg = message if message else String(a) + " > " + String(b)
        raise Error(error_msg)


fn assert_less_or_equal(a: Int, b: Int, message: String = "") raises:
    """Assert a <= b for Int values.

    Args:
        a: First value.
        b: Second value.
        message: Optional error message.

    Raises:
        Error if a > b.
    """
    if a > b:
        var error_msg = message if message else String(a) + " > " + String(b)
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


fn assert_not_equal_tensor(a: ExTensor, b: ExTensor, message: String = "") raises:
    """Assert two tensors are not equal element-wise.

    Verifies that at least one element differs between the two tensors.
    Useful for tests verifying that weights have been updated during training.

    Args:
        a: First tensor.
        b: Second tensor.
        message: Optional error message.

    Raises:
        Error if all elements are equal or if shapes differ.
    """
    # Check shapes match
    var shape_a = a.shape()
    var shape_b = b.shape()

    if len(shape_a) != len(shape_b):
        raise Error("Cannot compare tensors with different dimensions")

    for i in range(len(shape_a)):
        if shape_a[i] != shape_b[i]:
            raise Error("Cannot compare tensors with different shapes")

    # Check if all elements are equal
    var numel = a.numel()
    var all_equal = True

    for i in range(numel):
        var val_a = a._get_float64(i)
        var val_b = b._get_float64(i)

        if val_a != val_b:
            all_equal = False
            break

    if all_equal:
        var error_msg = message if message else "Tensors are equal but should not be"
        raise Error(error_msg)


fn assert_tensor_equal(a: ExTensor, b: ExTensor, message: String = "") raises:
    """Assert two ExTensors are equal (shape and all elements).

    Args:
        a: First tensor.
        b: Second tensor.
        message: Optional error message.

    Raises:
        Error if shapes don't match or any elements differ.
    """
    # Check dimensions
    var a_shape = a.shape()
    var b_shape = b.shape()
    if len(a_shape) != len(b_shape):
        var msg = "Shape mismatch: " + String(len(a_shape)) + " vs " + String(len(b_shape))
        raise Error(message + ": " + msg if message else msg)

    # Check total elements
    var a_numel = a.numel()
    var b_numel = b.numel()
    if a_numel != b_numel:
        var msg = "Size mismatch: " + String(a_numel) + " vs " + String(b_numel)
        raise Error(message + ": " + msg if message else msg)

    # Check all elements
    for i in range(a_numel):
        var val_a = a._get_float64(i)
        var val_b = b._get_float64(i)
        var diff = val_a - val_b if val_a >= val_b else val_b - val_a
        if diff > 1e-10:
            var msg = "Values differ at index " + String(i) + ": " + String(val_a) + " vs " + String(val_b)
            raise Error(message + ": " + msg if message else msg)


fn assert_shape(tensor: ExTensor, expected: List[Int], message: String = "") raises:
    """Assert tensor has expected shape.

    Args:
        tensor: ExTensor to check.
        expected: Expected shape as List.
        message: Optional error message.

    Raises:
        Error if shapes don't match.
    """
    # Get actual shape
    var actual_shape = tensor.shape()

    # Check dimensions match
    if len(actual_shape) != len(expected):
        var error_msg = message if message else (
            "Shape dimension mismatch: expected " + String(len(expected)) +
            " dims, got " + String(len(actual_shape))
        )
        raise Error(error_msg)

    # Check each dimension
    for i in range(len(expected)):
        if actual_shape[i] != expected[i]:
            var error_msg = message if message else (
                "Shape mismatch at dim " + String(i) + ": expected " +
                String(expected[i]) + ", got " + String(actual_shape[i])
            )
            raise Error(error_msg)


fn assert_dtype(tensor: ExTensor, expected: DType, message: String = "") raises:
    """Assert tensor has expected dtype.

    Args:
        tensor: ExTensor to check.
        expected: Expected DType.
        message: Optional error message.

    Raises:
        Error if dtype doesn't match.
    """
    var actual = tensor.dtype()
    if actual != expected:
        var error_msg = message if message else (
            "Expected dtype " + String(expected) + ", got " + String(actual)
        )
        raise Error(error_msg)


fn assert_numel(tensor: ExTensor, expected: Int, message: String = "") raises:
    """Assert tensor has expected number of elements.

    Args:
        tensor: ExTensor to check.
        expected: Expected total element count.
        message: Optional error message.

    Raises:
        Error if numel doesn't match.
    """
    var actual = tensor.numel()
    if actual != expected:
        var error_msg = message if message else (
            "Expected numel " + String(expected) + ", got " + String(actual)
        )
        raise Error(error_msg)


fn assert_dim(tensor: ExTensor, expected: Int, message: String = "") raises:
    """Assert tensor has expected number of dimensions.

    Args:
        tensor: ExTensor to check.
        expected: Expected dimension count.
        message: Optional error message.

    Raises:
        Error if dim doesn't match.
    """
    var actual = len(tensor.shape())
    if actual != expected:
        var error_msg = message if message else (
            "Expected " + String(expected) + " dimensions, got " + String(actual)
        )
        raise Error(error_msg)


fn assert_value_at(
    tensor: ExTensor,
    index: Int,
    expected: Float64,
    tolerance: Float64 = 1e-6,
    message: String = ""
) raises:
    """Assert tensor value at flat index matches expected value.

    Args:
        tensor: ExTensor to check.
        index: Flat index to check.
        expected: Expected value.
        tolerance: Acceptable difference (default: 1e-6).
        message: Optional error message.

    Raises:
        Error if value doesn't match within tolerance.
    """
    if index < 0 or index >= tensor.numel():
        raise Error("Index out of bounds: " + String(index))

    var actual = tensor._get_float64(index)
    var diff = actual - expected if actual >= expected else expected - actual

    if diff > tolerance:
        var error_msg = message if message else (
            "Expected value " + String(expected) + " at index " + String(index) +
            ", got " + String(actual) + " (diff: " + String(diff) + ")"
        )
        raise Error(error_msg)


fn assert_all_values(
    tensor: ExTensor,
    expected: Float64,
    tolerance: Float64 = 1e-6,
    message: String = ""
) raises:
    """Assert all tensor values match expected constant.

    Args:
        tensor: ExTensor to check.
        expected: Expected constant value.
        tolerance: Acceptable difference (default: 1e-6).
        message: Optional error message.

    Raises:
        Error if any value doesn't match within tolerance.
    """
    var n = tensor.numel()
    for i in range(n):
        var actual = tensor._get_float64(i)
        var diff = actual - expected if actual >= expected else expected - actual

        if diff > tolerance:
            var error_msg = message if message else (
                "Expected all values to be " + String(expected) +
                ", but index " + String(i) + " is " + String(actual)
            )
            raise Error(error_msg)


fn assert_all_close(
    a: ExTensor,
    b: ExTensor,
    tolerance: Float64 = 1e-6,
    message: String = ""
) raises:
    """Assert two tensors are element-wise close.

    Args:
        a: First tensor.
        b: Second tensor.
        tolerance: Acceptable difference (default: 1e-6).
        message: Optional error message.

    Raises:
        Error if shapes don't match or values differ beyond tolerance.
    """
    # Check shapes match
    var shape_a = a.shape()
    var shape_b = b.shape()

    if len(shape_a) != len(shape_b):
        raise Error("Shape dimension mismatch: " + String(len(shape_a)) + " vs " + String(len(shape_b)))

    for i in range(len(shape_a)):
        if shape_a[i] != shape_b[i]:
            raise Error("Shape mismatch at dim " + String(i) + ": " + String(shape_a[i]) + " vs " + String(shape_b[i]))

    # Check all values
    var n = a.numel()
    for i in range(n):
        var val_a = a._get_float64(i)
        var val_b = b._get_float64(i)
        var diff = val_a - val_b if val_a >= val_b else val_b - val_a

        if diff > tolerance:
            var error_msg = message if message else (
                "Tensors differ at index " + String(i) + ": " +
                String(val_a) + " vs " + String(val_b) +
                " (diff: " + String(diff) + ")"
            )
            raise Error(error_msg)


fn assert_type[T: AnyType](value: T, expected_type: String) raises:
    """Assert value is of expected type (for documentation purposes).

    Note: Type checking in Mojo happens at compile time, so this function
    is primarily for test documentation and clarity.

    Args:
        value: The value to check.
        expected_type: String describing the expected type (for documentation).

    Raises:
        Never raises - type checking is done at compile time.
    """
    # Type checking in Mojo is compile-time
    # This function exists for test API clarity
    pass


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


struct BenchmarkResult(Copyable, Movable):
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
    for i in range(len(results)):
        results[i].print_result()
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
    return vec^


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
    return matrix^


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
    return vec^


# ============================================================================
# Training Test Fixtures
# ============================================================================


fn create_simple_model() -> SimpleMLP:
    """Create a simple 2-layer neural network for testing training loops.

    Returns a mock neural network with:
    - Input dimension: 10
    - Hidden dimension: 5
    - Output dimension: 1
    - Uses ReLU activation between layers

    The model uses constant initialization for predictable testing behavior.

    Returns:
        SimpleMLP instance configured for testing.

    Example:
        ```mojo
        var model = create_simple_model()
        var input = create_test_vector(10, value=1.0)
        var output = model.forward(input)
        ```
    """
    var model = SimpleMLP(
        input_dim=10,
        hidden_dim=5,
        output_dim=1,
        num_hidden_layers=1,
        init_value=0.1
    )
    return model^


fn create_simple_dataset(
    n_samples: Int = 100,
    input_dim: Int = 10,
    output_dim: Int = 1,
    seed_value: Int = 42
) -> List[Tuple[List[Float32], List[Float32]]]:
    """Create a simple dataset for testing data loading.

    Generates synthetic data with:
    - Configurable number of samples
    - Configurable input and output dimensions
    - Deterministic seeding for reproducible tests

    Returns a list of (input, output) tuples for each sample.

    Args:
        n_samples: Number of samples in dataset (default: 100).
        input_dim: Dimension of input features (default: 10).
        output_dim: Dimension of output labels (default: 1).
        seed_value: Random seed for reproducibility (default: 42).

    Returns:
        List of tuples containing (data, label) for each sample.

    Example:
        ```mojo
        var dataset = create_simple_dataset(n_samples=50, input_dim=10, output_dim=5)
        var (data, label) = dataset[0]
        ```
    """
    # Set seed for reproducibility
    seed(seed_value)

    var samples = List[Tuple[List[Float32], List[Float32]]](capacity=n_samples)

    # Generate samples
    for i in range(n_samples):
        var item_seed = seed_value + i

        # Generate input features
        var input_features = List[Float32](capacity=input_dim)
        for j in range(input_dim):
            # Use deterministic values based on seed and index
            var val = Float32((seed_value + i + j) % 100) / 100.0
            input_features.append(val)

        # Generate output labels
        var output_labels = List[Float32](capacity=output_dim)
        for j in range(output_dim):
            # Use deterministic values based on seed and index
            var val = Float32((seed_value + i + j + 1000) % 100) / 100.0
            output_labels.append(val)

        samples.append((input_features^, output_labels^))

    return samples^


struct MockDataLoader(Copyable, Movable):
    """Mock data loader for testing training loops.

    Provides a simple data loader that yields batches of data.

    Attributes:
        samples: List of (input, output) tuples.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle samples.
    """

    var samples: List[Tuple[List[Float32], List[Float32]]]
    var batch_size: Int
    var shuffle: Bool

    fn __init__(
        out self,
        var samples: List[Tuple[List[Float32], List[Float32]]],
        batch_size: Int = 32,
        shuffle: Bool = False
    ):
        """Initialize mock data loader.

        Args:
            samples: List of (input, output) tuples.
            batch_size: Samples per batch (default: 32).
            shuffle: Whether to shuffle (default: False).
        """
        self.samples = samples^
        self.batch_size = batch_size
        self.shuffle = shuffle

    fn __len__(self) -> Int:
        """Get number of batches."""
        var n_samples = len(self.samples)
        return (n_samples + self.batch_size - 1) // self.batch_size


fn create_mock_dataloader(
    n_batches: Int = 10,
    batch_size: Int = 32,
    n_samples: Int = 320,
    input_dim: Int = 10,
    output_dim: Int = 1,
    seed_value: Int = 42,
    shuffle: Bool = False
) -> MockDataLoader:
    """Create mock data loader with specified number of batches.

    Generates a synthetic dataset and wraps it in a MockDataLoader
    for testing training loops and data pipelines.

    Args:
        n_batches: Desired number of batches (default: 10).
        batch_size: Samples per batch (default: 32).
        n_samples: Total number of samples (default: 320 = 10 batches * 32).
        input_dim: Input feature dimension (default: 10).
        output_dim: Output label dimension (default: 1).
        seed_value: Random seed (default: 42).
        shuffle: Whether to shuffle data (default: False).

    Returns:
        MockDataLoader with synthetic samples.

    Example:
        ```mojo
        var loader = create_mock_dataloader(n_batches=5, batch_size=32)
        var num_batches = loader.__len__()
        ```

    Note:
        - If n_samples != n_batches * batch_size, the last batch may be partial.
        - All data is synthetically generated and deterministic.
    """
    # Create the simple dataset
    var dataset = create_simple_dataset(
        n_samples=n_samples,
        input_dim=input_dim,
        output_dim=output_dim,
        seed_value=seed_value
    )

    # Create and return mock loader
    return MockDataLoader(
        dataset^,
        batch_size=batch_size,
        shuffle=shuffle
    )
