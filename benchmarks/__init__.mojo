"""ML Odyssey Benchmarking Framework (DEPRECATED).

DEPRECATED - Use shared.benchmarking for all new benchmarking code.

This package contains legacy benchmark utilities that have been consolidated
into the shared.benchmarking module. The legacy modules are maintained for
backward compatibility but should not be used for new code.

For new benchmarking code, import from shared.benchmarking:

    from shared.benchmarking import (
        benchmark_function,
        BenchmarkConfig,
        BenchmarkResult,
        BenchmarkRunner,
        print_benchmark_report,
        print_benchmark_summary,
    )

Modules:
- stats: Statistical utilities (mean, std_dev, percentiles, min/max)
- reporter: Results formatting and export (table printing, JSON export)

Core benchmarking types are in shared.benchmarking:
- BenchmarkConfig/LegacyBenchmarkConfig: Configuration
- BenchmarkResult/LegacyBenchmarkResult: Results with timing statistics
- benchmark_function/benchmark_operation: Benchmarking functions

Usage:
    from shared.benchmarking import (
        LegacyBenchmarkConfig,
        LegacyBenchmarkResult,
        benchmark_operation,
    )
    from benchmarks import stats, reporter

    var config = LegacyBenchmarkConfig(warmup=100, iterations=1000)
    var result = benchmark_operation(
        "relu_forward",
        operation=fn() raises { _ = relu(tensor) },
        config=config
    )
    reporter.print_summary([result])
    reporter.export_json_simple([result], "results.json")
"""
