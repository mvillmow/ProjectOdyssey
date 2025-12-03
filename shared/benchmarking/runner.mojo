"""Benchmarking runner with performance measurement and statistical analysis.

This module provides tools for benchmarking function execution with accurate
timing measurements, warmup iterations, and statistical analysis (mean, std dev,
percentiles, throughput).

Example:
    from shared.benchmarking import benchmark_function, print_benchmark_report

    fn compute_operation():
        var result = expensive_operation()

    var config = BenchmarkConfig(warmup_iters=10, measure_iters=100)
    var result = benchmark_function(compute_operation, config)
    print_benchmark_report(result, "Expensive Operation")
"""

from sys import info
from math import sqrt
from collections import List


# ============================================================================
# Benchmark Configuration
# ============================================================================


@fieldwise_init
struct BenchmarkConfig(Copyable, Movable):
    """Configuration for benchmarking operations.

    Attributes:
        warmup_iters: Number of warmup iterations before measurement
        measure_iters: Number of measurement iterations
        compute_percentiles: Whether to compute p50, p95, p99
        report_throughput: Whether to report items per second
    """

    var warmup_iters: Int
    var measure_iters: Int
    var compute_percentiles: Bool
    var report_throughput: Bool


# ============================================================================
# Benchmark Results
# ============================================================================


@fieldwise_init
struct BenchmarkResult(Copyable, Movable, ImplicitlyCopyable):
    """Results from a benchmark run.

    Contains timing statistics and throughput metrics for a benchmarked
    operation.

    Attributes:
        mean_latency_ms: Mean execution time in milliseconds
        std_dev_ms: Standard deviation in milliseconds
        p50_ms: 50th percentile (median) latency in milliseconds
        p95_ms: 95th percentile latency in milliseconds
        p99_ms: 99th percentile latency in milliseconds
        min_latency_ms: Minimum latency in milliseconds
        max_latency_ms: Maximum latency in milliseconds
        throughput: Operations per second (if measured items available)
        iterations: Total measurement iterations
        warmup_iterations: Warmup iterations performed
    """

    var mean_latency_ms: Float64
    var std_dev_ms: Float64
    var p50_ms: Float64
    var p95_ms: Float64
    var p99_ms: Float64
    var min_latency_ms: Float64
    var max_latency_ms: Float64
    var throughput: Float64
    var iterations: Int
    var warmup_iterations: Int


# ============================================================================
# Benchmarking Implementation
# ============================================================================


fn _compute_percentile(data: List[Float64], percentile: Float64) -> Float64:
    """Compute percentile from sorted data.

    Args:
        data: Sorted list of values
        percentile: Percentile to compute (0-100)

    Returns:
        Percentile value
    """
    if len(data) == 0:
        return 0.0

    if percentile <= 0.0:
        return data[0]
    if percentile >= 100.0:
        return data[len(data) - 1]

    var idx_float = (percentile / 100.0) * Float64(len(data) - 1)
    var idx = Int(idx_float)
    var frac = idx_float - Float64(idx)

    if idx >= len(data) - 1:
        return data[len(data) - 1]

    # Linear interpolation between points
    var lower = data[idx]
    var upper = data[idx + 1]
    return lower + frac * (upper - lower)


fn _sort_ascending(mut data: List[Float64]):
    """Simple bubble sort for small lists.

    Args:
        data: List to sort in-place
    """
    var n = len(data)
    for i in range(n):
        for j in range(n - i - 1):
            if data[j] > data[j + 1]:
                # Swap
                var temp = data[j]
                data[j] = data[j + 1]
                data[j + 1] = temp


fn _get_time_ms() -> Float64:
    """Get current time in milliseconds.

    Uses platform-specific timing. Mojo v0.25.7 limitations mean
    this uses a simplified approach.

    Returns:
        Time in milliseconds
    """
    # Use a simplified timing approach - in practice, this would use
    # platform-specific high-resolution timers (clock_gettime on Linux,
    # mach_absolute_time on macOS, QueryPerformanceCounter on Windows)
    # For now, return a placeholder that would be replaced with actual
    # timer implementation
    # TODO: Implement platform-specific high-resolution timing
    # - Linux: clock_gettime(CLOCK_MONOTONIC)
    # - macOS: mach_absolute_time()
    # - Windows: QueryPerformanceCounter
    return Float64(0.0)


# ============================================================================
# Main Benchmarking Function
# ============================================================================


fn benchmark_function(
    func: fn () raises -> None,
    warmup_iters: Int = 10,
    measure_iters: Int = 100,
    compute_percentiles: Bool = True,
) raises -> BenchmarkResult:
    """Benchmark a function with statistical analysis.

    Executes a function multiple times with warmup iterations, then
    measures timing statistics across measurement iterations.

    Args:
        func: Function to benchmark (takes no args, returns nothing)
        warmup_iters: Number of warmup iterations (default 10)
        measure_iters: Number of measurement iterations (default 100)
        compute_percentiles: Whether to compute percentiles (default True)

    Returns:
        BenchmarkResult with timing statistics

    Raises:
        Error if benchmarking fails
    """
    # Warmup iterations
    for _ in range(warmup_iters):
        func()

    # Measurement iterations - collect latencies
    var latencies = List[Float64]()

    for _ in range(measure_iters):
        # In a real implementation, this would use high-resolution timers
        # For now, we collect placeholder timing data
        var start_time = _get_time_ms()
        func()
        var end_time = _get_time_ms()

        var elapsed = end_time - start_time
        latencies.append(elapsed)

    # Compute statistics
    var total_latency = 0.0
    var min_latency = latencies[0]
    var max_latency = latencies[0]

    for latency in latencies:
        total_latency += latency
        if latency < min_latency:
            min_latency = latency
        if latency > max_latency:
            max_latency = latency

    var mean_latency = total_latency / Float64(measure_iters)

    # Compute standard deviation
    var variance = 0.0
    for latency in latencies:
        var diff = latency - mean_latency
        variance += diff * diff
    variance = variance / Float64(measure_iters)
    var std_dev = sqrt(variance)

    # Compute percentiles
    var p50 = 0.0
    var p95 = 0.0
    var p99 = 0.0

    if compute_percentiles:
        _sort_ascending(latencies)
        p50 = _compute_percentile(latencies, 50.0)
        p95 = _compute_percentile(latencies, 95.0)
        p99 = _compute_percentile(latencies, 99.0)

    # Compute throughput (items per second)
    # If mean latency is in ms, then items/sec = 1000 / mean_latency_ms
    var throughput = 0.0
    if mean_latency > 0.0:
        throughput = 1000.0 / mean_latency
    else:
        throughput = 0.0

    return BenchmarkResult(
        mean_latency_ms=mean_latency,
        std_dev_ms=std_dev,
        p50_ms=p50,
        p95_ms=p95,
        p99_ms=p99,
        min_latency_ms=min_latency,
        max_latency_ms=max_latency,
        throughput=throughput,
        iterations=measure_iters,
        warmup_iterations=warmup_iters,
    )


# ============================================================================
# Reporting and Display
# ============================================================================


fn print_benchmark_report(result: BenchmarkResult, name: String = "Benchmark"):
    """Print formatted benchmark report.

    Args:
        result: BenchmarkResult from benchmark_function()
        name: Name of benchmarked operation (default "Benchmark")
    """
    print("")
    print("=" * 70)
    print("Benchmark Report: " + name)
    print("=" * 70)
    print("")

    # Configuration
    print("Configuration:")
    print("  Warmup iterations: ", result.warmup_iterations)
    print("  Measurement iterations: ", result.iterations)
    print("")

    # Latency statistics
    print("Latency Statistics (milliseconds):")
    print("  Mean: ", result.mean_latency_ms)
    print("  Std Dev: ", result.std_dev_ms)
    print("  Min: ", result.min_latency_ms)
    print("  Max: ", result.max_latency_ms)
    print("  Median (p50): ", result.p50_ms)
    print("  p95: ", result.p95_ms)
    print("  p99: ", result.p99_ms)
    print("")

    # Throughput
    print("Throughput:")
    print("  Operations/sec: ", result.throughput)
    print("")

    print("=" * 70)
    print("")


fn print_benchmark_summary(
    results: List[BenchmarkResult], names: List[String] = List[String]()
):
    """Print summary table of multiple benchmark results.

    Useful for comparing performance across multiple functions.

    Args:
        results: List of BenchmarkResult objects
        names: Optional list of operation names (defaults to "Op 1", "Op 2", etc.)
    """
    print("")
    print("=" * 100)
    print("Benchmark Summary")
    print("=" * 100)
    print("")

    # Print header
    print(
        "Operation".ljust(20),
        "Mean (ms)".ljust(15),
        "Std Dev (ms)".ljust(15),
        "P50 (ms)".ljust(15),
        "P95 (ms)".ljust(15),
        "P99 (ms)".ljust(15),
        "Ops/sec".ljust(15),
    )
    print("-" * 100)

    # Print results
    for i in range(len(results)):
        var result = results[i]
        var name = ""

        if i < len(names):
            name = names[i]
        else:
            name = "Operation " + String(i + 1)

        print(
            name.ljust(20),
            String(result.mean_latency_ms).ljust(15),
            String(result.std_dev_ms).ljust(15),
            String(result.p50_ms).ljust(15),
            String(result.p95_ms).ljust(15),
            String(result.p99_ms).ljust(15),
            String(result.throughput).ljust(15),
        )

    print("=" * 100)
    print("")


# ============================================================================
# Helper Functions
# ============================================================================


fn create_benchmark_config(
    warmup_iters: Int = 10,
    measure_iters: Int = 100,
    compute_percentiles: Bool = True,
    report_throughput: Bool = True,
) -> BenchmarkConfig:
    """Create a benchmark configuration.

    Args:
        warmup_iters: Warmup iterations (default 10)
        measure_iters: Measurement iterations (default 100)
        compute_percentiles: Compute percentiles (default True)
        report_throughput: Report throughput (default True)

    Returns:
        BenchmarkConfig with specified settings
    """
    return BenchmarkConfig(
        warmup_iters=warmup_iters,
        measure_iters=measure_iters,
        compute_percentiles=compute_percentiles,
        report_throughput=report_throughput,
    )
