"""ML Odyssey Benchmarking Framework.

Provides comprehensive benchmarking infrastructure for measuring operation
performance with statistical confidence.

Modules:
- framework: Core benchmark infrastructure (BenchmarkConfig, BenchmarkResult, benchmark_operation)
- stats: Statistical utilities (mean, std_dev, percentiles, min/max)
- reporter: Results formatting and export (table printing, JSON export)

Usage:
    from benchmarks import framework, stats, reporter

    var config = framework.BenchmarkConfig(warmup=100, iterations=1000)
    var result = framework.benchmark_operation(
        "relu_forward",
        operation=fn() raises { _ = relu(tensor) },
        config=config
    )
    reporter.print_summary(results)
    reporter.export_json_simple(results, "results.json")
"""
