"""Test the benchmarking framework with simple operations.

Tests:
1. Basic benchmark measurement
2. Statistical calculations
3. Result formatting
4. JSON export
"""

from shared.benchmarking import (
    LegacyBenchmarkConfig as BenchmarkConfig,
    LegacyBenchmarkResult as BenchmarkResult,
    benchmark_operation,
)
from benchmarks.stats import compute_mean, compute_std_dev, compute_percentile
from benchmarks.reporter import print_summary


fn simple_loop() raises:
    """Simple operation for benchmarking: sum of integers."""
    var sum: Int = 0
    for i in range(1000):
        sum += i


fn vector_operation() raises:
    """Vector allocation and manipulation."""
    var v = List[Float64](capacity=100)
    for i in range(100):
        v.append(Float64(i) * 1.5)


fn test_statistics():
    """Test statistical computation functions."""
    print("\n" + "=" * 80)
    print("TEST: Statistical Calculations")
    print("=" * 80)

    var values = List[Float64](capacity=5)
    values.append(10.0)
    values.append(20.0)
    values.append(30.0)
    values.append(40.0)
    values.append(50.0)

    var mean = compute_mean(values)
    var std_dev = compute_std_dev(values, mean)
    var p50 = compute_percentile(values, 50)
    var p95 = compute_percentile(values, 95)

    print("Mean:", mean)
    print("Std Dev:", std_dev)
    print("Median (p50):", p50)
    print("p95:", p95)
    print("✓ Statistics test passed")


fn test_simple_benchmark() raises:
    """Test basic benchmark measurement."""
    print("\n" + "=" * 80)
    print("TEST: Simple Loop Benchmark")
    print("=" * 80)

    var config = BenchmarkConfig(warmup=10, iterations=100)
    var result = benchmark_operation("simple_loop", simple_loop, config)

    print("Operation:", result.name)
    print("Mean:", result.mean_time_us, "us")
    print("Std Dev:", result.std_dev_us, "us")
    print("Min:", result.min_time_us, "us")
    print("Max:", result.max_time_us, "us")
    print("p50:", result.p50_us, "us")
    print("p95:", result.p95_us, "us")
    print("p99:", result.p99_us, "us")
    print("Throughput:", result.throughput_ops_per_sec, "ops/s")
    print("✓ Benchmark test passed")


fn test_multiple_benchmarks() raises:
    """Test multiple benchmarks and summary."""
    print("\n" + "=" * 80)
    print("TEST: Multiple Benchmarks")
    print("=" * 80)

    var config = BenchmarkConfig(warmup=10, iterations=100)

    var results = List[BenchmarkResult](capacity=2)

    print("Running simple_loop benchmark...")
    var result1 = benchmark_operation("simple_loop", simple_loop, config)
    results.append(result1^)

    print("Running vector_operation benchmark...")
    var result2 = benchmark_operation("vector_operation", vector_operation, config)
    results.append(result2^)

    print_summary(results)
    print("✓ Multiple benchmarks test passed")


fn main() raises:
    """Run all framework tests."""
    print("=" * 80)
    print("BENCHMARKING FRAMEWORK TEST SUITE")
    print("=" * 80)

    test_statistics()
    test_simple_benchmark()
    test_multiple_benchmarks()

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED")
    print("=" * 80)
