"""Benchmarking Framework for ML Odyssey.

Provides utilities for measuring and reporting performance characteristics
of ML operations including latency, throughput, and statistical analysis.

Modules:
    `result`: Consolidated low-level benchmark result tracking (NEW - Issue #2282)
    `runner`: High-level benchmarking runner with statistical analysis

The `result` module provides BenchmarkResult for recording individual iteration
times and computing statistics. Import directly to use:

    from shared.benchmarking.result import BenchmarkResult

The `runner` module provides benchmark_function for high-level benchmarking with
percentiles and throughput. Imported here for convenience:

    from shared.benchmarking import benchmark_function, print_benchmark_report

Example - Low-level result tracking:
    from shared.benchmarking.result import BenchmarkResult

    var result = BenchmarkResult("operation_name", iterations=1000)
    for _ in range(1000):
        var start = now()
        operation()
        var end = now()
        result.record(end - start)
    print("Mean:", result.mean(), "ns")

Example - High-level benchmarking:
    from shared.benchmarking import benchmark_function, print_benchmark_report

    fn forward_pass():
        var output = model.forward(input_batch)
        return output

    var result = benchmark_function(forward_pass, warmup_iters=10, measure_iters=100)
    print_benchmark_report(result, "Forward Pass")
"""

# Package version
alias VERSION = "0.1.0"

# ============================================================================
# Exports - Implemented modules
# ============================================================================

from .runner import (
    BenchmarkResult,  # High-level benchmark results from runner module
    benchmark_function,  # Main benchmarking function
    print_benchmark_report,  # Print formatted results
    BenchmarkConfig,  # Configuration for benchmarking
    create_benchmark_config,  # Create default config
)
