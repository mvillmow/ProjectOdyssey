"""Performance benchmarking framework for ML Odyssey.

Provides infrastructure for measuring operation performance with statistical
confidence and comparison against baselines.

Features:
- Automatic warmup phase for JIT compilation
- Multi-iteration measurements for statistical validity
- Percentile calculations (p50, p95, p99)
- Throughput and memory tracking
- JSON export for CI/CD integration

Usage:
    var config = BenchmarkConfig(warmup=100, iterations=1000)
    var result = benchmark_operation(
        "relu_forward",
        operation=fn() raises { _ = relu(input_tensor) },
        config=config
    )
    print("Mean:", result.mean_time_us, "us")
    print("Throughput:", result.throughput_ops_per_sec, "ops/s")
"""

import time as mojo_time
from math import sqrt


struct BenchmarkConfig:
    """Configuration for benchmark execution.

    Attributes:
        warmup_iterations: Number of warmup runs for JIT compilation
        measure_iterations: Number of measured iterations
    """

    var warmup_iterations: Int
    var measure_iterations: Int

    fn __init__(
        inout self,
        warmup: Int = 100,
        iterations: Int = 1000,
    ):
        """Initialize benchmark configuration.

        Args:
            warmup: Warmup iterations (default: 100)
            iterations: Measurement iterations (default: 1000)
        """
        self.warmup_iterations = warmup
        self.measure_iterations = iterations


struct BenchmarkResult:
    """Results from a benchmark run.

    Attributes:
        name: Descriptive name for the operation
        mean_time_us: Mean execution time in microseconds
        std_dev_us: Standard deviation in microseconds
        min_time_us: Minimum execution time in microseconds
        max_time_us: Maximum execution time in microseconds
        p50_us: Median (50th percentile) in microseconds
        p95_us: 95th percentile in microseconds
        p99_us: 99th percentile in microseconds
        throughput_ops_per_sec: Operations per second
        memory_mb: Memory usage in MB
        input_shape: String representation of input shape
        dtype: Data type identifier
    """

    var name: String
    var mean_time_us: Float64
    var std_dev_us: Float64
    var min_time_us: Float64
    var max_time_us: Float64
    var p50_us: Float64
    var p95_us: Float64
    var p99_us: Float64
    var throughput_ops_per_sec: Float64
    var memory_mb: Float64
    var input_shape: String
    var dtype: String

    fn __init__(
        inout self,
        name: String,
        mean_time_us: Float64,
        std_dev_us: Float64,
        min_time_us: Float64,
        max_time_us: Float64,
        p50_us: Float64,
        p95_us: Float64,
        p99_us: Float64,
        throughput_ops_per_sec: Float64,
        memory_mb: Float64,
        input_shape: String,
        dtype: String,
    ):
        """Initialize benchmark result."""
        self.name = name
        self.mean_time_us = mean_time_us
        self.std_dev_us = std_dev_us
        self.min_time_us = min_time_us
        self.max_time_us = max_time_us
        self.p50_us = p50_us
        self.p95_us = p95_us
        self.p99_us = p99_us
        self.throughput_ops_per_sec = throughput_ops_per_sec
        self.memory_mb = memory_mb
        self.input_shape = input_shape
        self.dtype = dtype


fn benchmark_operation(
    name: String,
    operation: fn() raises -> None,
    config: BenchmarkConfig = BenchmarkConfig(),
) raises -> BenchmarkResult:
    """Run operation multiple times and collect statistics.

    Performs warmup phase for JIT compilation, then measures operation
    execution time across multiple iterations. Computes mean, standard
    deviation, percentiles, and throughput.

    Args:
        name: Descriptive name for the operation
        operation: Function to benchmark (should be self-contained)
        config: Benchmark configuration (warmup, iterations)

    Returns:
        BenchmarkResult with timing statistics

    Example:
        fn my_relu_op():
            var x = random_tensor([1024, 1024])
            _ = relu(x)

        var config = BenchmarkConfig(warmup=100, iterations=1000)
        var result = benchmark_operation("relu_1024x1024", my_relu_op, config)
        print(result.name, ":", result.mean_time_us, "us")
    """
    # Warmup phase (JIT compilation, cache warming)
    for _ in range(config.warmup_iterations):
        try:
            operation()
        except:
            pass  # Ignore warmup errors

    # Measurement phase
    var times = List[Float64](capacity=config.measure_iterations)
    for _ in range(config.measure_iterations):
        var start = mojo_time.now()
        operation()
        var end = mojo_time.now()
        var elapsed_us = Float64(end - start) / 1000.0  # nanoseconds -> microseconds
        times.append(elapsed_us)

    # Compute statistics
    from benchmarks.stats import (
        compute_mean,
        compute_std_dev,
        compute_percentile,
        compute_min,
        compute_max,
    )

    var mean = compute_mean(times)
    var std_dev = compute_std_dev(times, mean)
    var min_time = compute_min(times)
    var max_time = compute_max(times)
    var p50 = compute_percentile(times, 50)
    var p95 = compute_percentile(times, 95)
    var p99 = compute_percentile(times, 99)
    var throughput = 1_000_000.0 / mean  # ops per second

    return BenchmarkResult(
        name,
        mean,
        std_dev,
        min_time,
        max_time,
        p50,
        p95,
        p99,
        throughput,
        0.0,
        "",
        "",
    )
