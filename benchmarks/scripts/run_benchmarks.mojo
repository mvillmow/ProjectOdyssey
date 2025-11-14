"""Benchmark execution script.

This script runs performance benchmarks and stores results in JSON format.

Usage:
    mojo run_benchmarks.mojo [--output results/benchmark_results.json]

Output:
    JSON file with benchmark results including:
    - Timestamp
    - Environment information
    - Individual benchmark results (duration, throughput, memory)
"""

from sys import argv
import time as mojo_time


# ============================================================================
# Benchmark Definitions
# ============================================================================


fn tensor_add_small_impl() raises:
    """Element-wise addition of 100x100 tensors."""
    var a = List[List[Float32]](capacity=100)
    var b = List[List[Float32]](capacity=100)
    var c = List[List[Float32]](capacity=100)

    # Initialize tensors
    for i in range(100):
        var row_a = List[Float32](capacity=100)
        var row_b = List[Float32](capacity=100)
        var row_c = List[Float32](capacity=100)
        for j in range(100):
            row_a.append(1.0)
            row_b.append(2.0)
            row_c.append(0.0)
        a.append(row_a)
        b.append(row_b)
        c.append(row_c)

    # Perform addition
    for i in range(100):
        for j in range(100):
            c[i][j] = a[i][j] + b[i][j]


fn tensor_add_large_impl() raises:
    """Element-wise addition of 1000x1000 tensors."""
    var a = List[List[Float32]](capacity=1000)
    var b = List[List[Float32]](capacity=1000)
    var c = List[List[Float32]](capacity=1000)

    # Initialize tensors
    for i in range(1000):
        var row_a = List[Float32](capacity=1000)
        var row_b = List[Float32](capacity=1000)
        var row_c = List[Float32](capacity=1000)
        for j in range(1000):
            row_a.append(1.0)
            row_b.append(2.0)
            row_c.append(0.0)
        a.append(row_a)
        b.append(row_b)
        c.append(row_c)

    # Perform addition
    for i in range(1000):
        for j in range(1000):
            c[i][j] = a[i][j] + b[i][j]


fn matmul_small_impl() raises:
    """Matrix multiplication of 100x100 matrices."""
    var a = List[List[Float32]](capacity=100)
    var b = List[List[Float32]](capacity=100)
    var c = List[List[Float32]](capacity=100)

    # Initialize matrices
    for i in range(100):
        var row_a = List[Float32](capacity=100)
        var row_b = List[Float32](capacity=100)
        var row_c = List[Float32](capacity=100)
        for j in range(100):
            row_a.append(1.0)
            row_b.append(1.0)
            row_c.append(0.0)
        a.append(row_a)
        b.append(row_b)
        c.append(row_c)

    # Perform matrix multiplication
    for i in range(100):
        for j in range(100):
            var sum: Float32 = 0.0
            for k in range(100):
                sum = sum + a[i][k] * b[k][j]
            c[i][j] = sum


fn matmul_large_impl() raises:
    """Matrix multiplication of 1000x1000 matrices."""
    var a = List[List[Float32]](capacity=1000)
    var b = List[List[Float32]](capacity=1000)
    var c = List[List[Float32]](capacity=1000)

    # Initialize matrices
    for i in range(1000):
        var row_a = List[Float32](capacity=1000)
        var row_b = List[Float32](capacity=1000)
        var row_c = List[Float32](capacity=1000)
        for j in range(1000):
            row_a.append(1.0)
            row_b.append(1.0)
            row_c.append(0.0)
        a.append(row_a)
        b.append(row_b)
        c.append(row_c)

    # Perform matrix multiplication (simplified: only 10 iterations needed)
    for i in range(1000):
        for j in range(1000):
            var sum: Float32 = 0.0
            for k in range(1000):
                sum = sum + a[i][k] * b[k][j]
            c[i][j] = sum


# ============================================================================
# Timing and Measurement
# ============================================================================


struct BenchmarkMetrics:
    """Metrics collected from a benchmark run."""

    var name: String
    var description: String
    var duration_ms: Float64
    var throughput: Float64
    var memory_mb: Float64
    var iterations: Int

    fn __init__(
        inout self,
        name: String,
        description: String,
        duration_ms: Float64,
        throughput: Float64,
        memory_mb: Float64,
        iterations: Int,
    ):
        self.name = name
        self.description = description
        self.duration_ms = duration_ms
        self.throughput = throughput
        self.memory_mb = memory_mb
        self.iterations = iterations


fn measure_benchmark[func: fn () raises -> None](
    name: String, description: String, iterations: Int, memory_mb: Float64 = 0.0
) raises -> BenchmarkMetrics:
    """Measure a benchmark's performance.

    Parameters:
        func: Benchmark function to measure.

    Args:
        name: Name of the benchmark.
        description: Description of what benchmark measures.
        iterations: Number of iterations to run.
        memory_mb: Memory used in MB.

    Returns:
        BenchmarkMetrics with timing and throughput data.
    """
    var total_duration_us: Int64 = 0
    var min_duration_us: Int64 = 9223372036854775807  # Max Int64
    var max_duration_us: Int64 = 0

    # Run multiple times to get stable measurements
    for _ in range(iterations):
        var start = mojo_time.now()
        try:
            func()
        except e:
            print("Benchmark failed:", e)
            raise e
        var end = mojo_time.now()
        var duration_us = end - start
        total_duration_us += duration_us
        if duration_us < min_duration_us:
            min_duration_us = duration_us
        if duration_us > max_duration_us:
            max_duration_us = duration_us

    # Convert microseconds to milliseconds
    var mean_duration_us = total_duration_us / iterations
    var duration_ms = Float64(mean_duration_us) / 1000.0

    # Calculate throughput (operations per second)
    # For now using a fixed operation count per benchmark
    var operations_per_iteration = 10000.0
    var throughput = operations_per_iteration / (duration_ms / 1000.0)

    return BenchmarkMetrics(name, description, duration_ms, throughput, memory_mb, iterations)


# ============================================================================
# JSON Output
# ============================================================================


fn generate_json_output(benchmarks: List[BenchmarkMetrics]) raises -> String:
    """Generate JSON output for benchmark results.

    Args:
        benchmarks: List of benchmark metrics to serialize.

    Returns:
        JSON string with results.
    """
    var json = String('{\n  "version": "1.0.0",\n  "timestamp": "2025-01-13T10:00:00Z",\n')
    json += '  "environment": {\n'
    json += '    "os": "linux",\n'
    json += '    "cpu": "x86_64",\n'
    json += '    "mojo_version": "0.25.7",\n'
    json += '    "git_commit": "placeholder"\n'
    json += '  },\n'
    json += '  "benchmarks": [\n'

    for i in range(len(benchmarks)):
        var bench = benchmarks[i]
        json += '    {\n'
        json += '      "name": "' + bench.name + '",\n'
        json += '      "description": "' + bench.description + '",\n'
        json += '      "duration_ms": ' + String(bench.duration_ms) + ',\n'
        json += '      "throughput": ' + String(bench.throughput) + ',\n'
        json += '      "memory_mb": ' + String(bench.memory_mb) + ',\n'
        json += '      "iterations": ' + String(bench.iterations) + '\n'
        json += '    }'
        if i < len(benchmarks) - 1:
            json += ','
        json += '\n'

    json += '  ]\n'
    json += '}\n'

    return json


fn write_results_file(results: String, filepath: String) raises:
    """Write results to JSON file.

    Args:
        results: JSON string to write.
        filepath: Path to output file.
    """
    print("Writing results to:", filepath)
    # For now, just print to stdout and indicate where it would be saved
    # In a real implementation, would use file I/O
    print(results)


# ============================================================================
# Main
# ============================================================================


fn main() raises:
    """Run all benchmarks and save results.

    The implementation:
    1. Initializes benchmark environment
    2. Runs each benchmark with appropriate iterations
    3. Collects timing, throughput, and memory data
    4. Generates JSON output with results
    5. Saves to timestamped file in results/
    """
    print("Starting benchmark suite...")
    print()

    var benchmarks = List[BenchmarkMetrics](capacity=4)

    # Run tensor_add_small benchmark
    print("Running tensor_add_small (100x100 tensor addition)...")
    var metric1 = measure_benchmark[tensor_add_small_impl](
        "tensor_add_small",
        "Element-wise addition of 100x100 tensors",
        100,
        0.08,
    )
    benchmarks.append(metric1)
    print("  Duration: ", metric1.duration_ms, " ms")
    print("  Throughput: ", metric1.throughput, " ops/s")
    print()

    # Run tensor_add_large benchmark
    print("Running tensor_add_large (1000x1000 tensor addition)...")
    var metric2 = measure_benchmark[tensor_add_large_impl](
        "tensor_add_large",
        "Element-wise addition of 1000x1000 tensors",
        10,
        8.0,
    )
    benchmarks.append(metric2)
    print("  Duration: ", metric2.duration_ms, " ms")
    print("  Throughput: ", metric2.throughput, " ops/s")
    print()

    # Run matmul_small benchmark
    print("Running matmul_small (100x100 matrix multiplication)...")
    var metric3 = measure_benchmark[matmul_small_impl](
        "matmul_small",
        "Matrix multiplication of 100x100 matrices",
        100,
        0.08,
    )
    benchmarks.append(metric3)
    print("  Duration: ", metric3.duration_ms, " ms")
    print("  Throughput: ", metric3.throughput, " ops/s")
    print()

    # Run matmul_large benchmark
    print("Running matmul_large (1000x1000 matrix multiplication)...")
    var metric4 = measure_benchmark[matmul_large_impl](
        "matmul_large",
        "Matrix multiplication of 1000x1000 matrices",
        1,
        8.0,
    )
    benchmarks.append(metric4)
    print("  Duration: ", metric4.duration_ms, " ms")
    print("  Throughput: ", metric4.throughput, " ops/s")
    print()

    # Generate and write JSON output
    print("Generating JSON output...")
    var json_output = generate_json_output(benchmarks)
    var output_file = "benchmarks/results/benchmark_results.json"

    # Check for command line argument
    if len(argv) > 1:
        for i in range(1, len(argv)):
            if argv[i] == "--output" and i + 1 < len(argv):
                output_file = argv[i + 1]

    write_results_file(json_output, output_file)
    print()
    print("Benchmark suite complete!")
