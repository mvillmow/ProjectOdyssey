# Benchmarking Tools

Performance measurement and tracking tools for ML implementations.

## Available Tools

### 1. Benchmark Framework (`benchmark.mojo`)

Core benchmarking framework for measuring model performance.

**Language**: Mojo (required for accurate ML performance measurement)

**Usage**:

```mojo
from tools.benchmarking.benchmark import ModelBenchmark, BenchmarkResult
from tools.test_utils.fixtures import create_test_model
from tools.test_utils.data_generators import TensorGenerator

fn benchmark_example():
    # Create model and data
    let model = create_test_model("cnn")
    let generator = TensorGenerator()
    let input = generator.generate_batch(32, 1, 28, 28)

    # Create benchmark
    let bench = ModelBenchmark(
        name="MyModel",
        num_warmup=10,
        num_iterations=100
    )

    # Measure inference
    let result = bench.measure_inference(model, input, batch_size=32)
    result.print_summary()
```

**Features**:

- Inference latency measurement
- Training step benchmarking
- Warmup iterations for stable results
- Throughput calculation (samples/second)
- Memory tracking (TODO)

### 2. Benchmark Runner (`runner.mojo`)

CLI tool for running benchmark suites.

**Language**: Mojo (required for accurate performance measurement)

**Usage**:

```bash
# Run benchmark suite
mojo tools/benchmarking/runner.mojo
```

**Output**:

```text
============================================================
ML Odyssey Benchmark Suite
============================================================

Model: SimpleCNN (test model)

Running inference benchmark...
Benchmark: SimpleCNN Inference
  Latency: 0.123 ms
  Throughput: 260416.67 samples/sec
  Memory: 0.0 MB

Running quick benchmark...
  Quick latency: 0.118 ms

============================================================
Benchmark complete!
============================================================
```

## Benchmark Results

Results include:

- **Latency**: Average time per iteration (ms)
- **Throughput**: Samples processed per second
- **Memory**: Peak memory usage (MB) - TODO

## Design Principles

- **Accurate**: No Python overhead, pure Mojo measurement
- **Stable**: Warmup iterations for consistent results
- **Comprehensive**: Measure multiple aspects (latency, throughput, memory)
- **Comparable**: Consistent methodology across models

## Language Justification

Per [ADR-001](../../notes/review/adr/ADR-001-language-selection-tooling.md):

- **Why Mojo**: Required for accurate ML performance measurement
- **Benefits**: Zero Python overhead, precise timing, SIMD optimization
- **Critical**: Performance measurement must not introduce overhead
- **Required**: ML/AI implementation (not automation)

## Future Enhancements

- Memory usage tracking
- Multi-batch size sweeps
- Comparison reports
- JSON output for CI/CD integration
- Visualization tools (Python for plotting)

## References

- [Issue #67](https://github.com/mvillmow/ml-odyssey/issues/67): Tools planning
- [Issue #69](https://github.com/mvillmow/ml-odyssey/issues/69): Tools implementation
- [ADR-001](../../notes/review/adr/ADR-001-language-selection-tooling.md): Language strategy
- [Mojo Best Practices](../../.claude/agents/mojo-language-review-specialist.md)
