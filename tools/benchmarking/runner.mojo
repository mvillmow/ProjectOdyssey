"""
Benchmarking Tools - Benchmark Runner

Purpose: CLI tool for running benchmarks
Language: Mojo (required for accurate performance measurement)
"""

from benchmark import ModelBenchmark, BenchmarkResult, run_benchmark
from test_utils.fixtures import SimpleCNN, create_test_model
from test_utils.data_generators import TensorGenerator
from tensor import TensorShape


fn benchmark_model_suite():
    """Run a suite of benchmarks on test models."""
    print("=" * 60)
    print("ML Odyssey Benchmark Suite")
    print("=" * 60)
    print()

    # Create test model
    let model = create_test_model("cnn")
    print("Model: SimpleCNN (test model)")
    print()

    # Create test data
    let generator = TensorGenerator()
    let input = generator.generate_batch(
        32,  # batch_size
        1,   # channels
        28,  # height
        28   # width
    )

    # Create benchmark
    let bench = ModelBenchmark(
        name="SimpleCNN",
        num_warmup=10,
        num_iterations=100
    )

    # Run inference benchmark
    print("Running inference benchmark...")
    let inference_result = bench.measure_inference(model, input, batch_size=32)
    inference_result.print_summary()
    print()

    # Quick benchmark
    print("Running quick benchmark...")
    let latency = run_benchmark(model, input, num_iterations=50)
    print("  Quick latency:", latency, "ms")
    print()

    print("=" * 60)
    print("Benchmark complete!")
    print("=" * 60)


fn main():
    """Main entry point."""
    benchmark_model_suite()
