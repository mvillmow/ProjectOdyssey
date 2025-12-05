"""ML Odyssey Benchmarking Framework.

Provides comprehensive benchmarking infrastructure for measuring operation
performance with statistical confidence.

Modules:
- stats: Statistical utilities (mean, std_dev, percentiles, min/max)
- reporter: Results formatting and export (table printing, JSON export)

Core benchmarking types are in shared.benchmarking:
- BenchmarkConfig/LegacyBenchmarkConfig: Configuration
- BenchmarkResult/LegacyBenchmarkResult: Results with timing statistics
- benchmark_function/benchmark_operation: Benchmarking functions

Usage:
    from shared.benchmarking import (
        LegacyBenchmarkConfig as BenchmarkConfig,
        LegacyBenchmarkResult as BenchmarkResult,
        benchmark_operation,
    )
    from benchmarks import stats, reporter

    var config = BenchmarkConfig(warmup=100, iterations=1000)
    var result = benchmark_operation(
        "relu_forward",
        operation=fn() raises { _ = relu(tensor) },
        config=config
    )
    reporter.print_summary(results)
    reporter.export_json_simple(results, "results.json")
"""
