# Performance Optimization

Techniques and best practices for optimizing ML Odyssey implementations.

## Table of Contents

- [SIMD Optimization](#simd-optimization)
- [Memory Layout](#memory-layout)
- [Benchmarking](#benchmarking)
- [Common Bottlenecks](#common-bottlenecks)
- [Profiling Tools](#profiling-tools)

## SIMD Optimization

SIMD (Single Instruction Multiple Data) vectorization is the primary performance optimization
strategy in ML Odyssey. This section covers practical techniques for leveraging SIMD on modern CPUs.

### SIMD Fundamentals

Modern CPUs execute multiple data elements with a single instruction:

- **AVX-512**: 16 float32 elements or 8 float64 elements per instruction
- **AVX2**: 8 float32 elements or 4 float64 elements per instruction
- **SSE**: 4 float32 elements or 2 float64 elements per instruction

Mojo automatically selects the best SIMD width at compile time based on CPU capabilities.

### Vectorizable Operations

The following operations benefit most from SIMD:

**Element-wise arithmetic**:

```mojo
# Example: Element-wise addition with SIMD
var result = add_simd(tensor_a, tensor_b)  # 4x speedup for float32
```

**Activation functions**:

```mojo
# Example: ReLU with SIMD
fn relu_simd(mut tensor: ExTensor):
    alias width = simdwidthof[DType.float32]()

    @parameter
    fn vectorized[simd_width: Int](idx: Int):
        var values = tensor._data.simd_load[simd_width](idx)
        tensor._data.simd_store[simd_width](idx, max(values, 0.0))

    vectorize[width, vectorized](tensor.numel())
```

**Batch normalization**:

```mojo
# Example: Normalize with SIMD
@parameter
fn vectorized_norm[width: Int](idx: Int):
    var x = input.load[width](idx)
    var normalized = (x - mean) / sqrt(variance + 1e-5)
    output.store[width](idx, normalized)

vectorize[simdwidthof[DType.float32](), vectorized_norm](total_elements)
```

**Reductions (dot products, sums)**:

```mojo
# Example: Dot product with SIMD reduction
alias width = simdwidthof[DType.float32]()
var accumulator = SIMD[DType.float32, width](0.0)

@parameter
fn dot[simd_width: Int](idx: Int):
    var a_vec = a._data.simd_load[simd_width](idx)
    var b_vec = b._data.simd_load[simd_width](idx)
    accumulator += a_vec * b_vec

vectorize[width, dot](vector_size)
result = accumulator.reduce_add()
```

### Performance Characteristics

Expected speedups on modern CPUs:

| Operation | Float32 | Float64 | Notes |
|-----------|---------|---------|-------|
| Element-wise add | 3-5x | 2-3x | Simple arithmetic |
| Element-wise multiply | 3-5x | 2-3x | Pipelines well |
| ReLU activation | 3-5x | 2-3x | Trivial branching |
| Dot product | 3-5x | 2-3x | Reduction overhead |
| Matrix multiply (1024x1024) | 4-6x | 2-3x | Memory bound, good cache reuse |
| Batch norm | 2-4x | 1.5-2x | Multiple operations combined |

**Key Insight**: Float32 consistently outperforms Float64 in SIMD because:

- 2x wider SIMD vectors (8 elements vs 4 elements)
- Better cache efficiency (smaller memory footprint)
- Typical neural networks use float32 by default

### Non-Vectorizable Operations

Some operations don't benefit from SIMD:

- **Complex indexing**: Irregular memory access patterns
- **Sparse tensors**: Most values are zero, can't vectorize
- **Dynamic shapes**: Shape must be known at compile time
- **Conditional logic**: Deep branching breaks vectorization
- **Serial dependencies**: Each iteration depends on previous result

### Tuning SIMD for Your Hardware

Check SIMD capabilities at compile time:

```mojo
from sys.info import simd_width_of

fn get_simd_info():
    """Print SIMD capabilities."""
    alias float32_width = simd_width_of[DType.float32]()
    alias float64_width = simd_width_of[DType.float64]()
    print("Float32 SIMD width:", float32_width)
    print("Float64 SIMD width:", float64_width)
```

The Mojo compiler automatically selects optimal SIMD width for your CPU at compile time.
No manual configuration needed.

## Memory Layout

Efficient memory access is crucial for SIMD performance.

### Row-Major vs Column-Major

ML Odyssey uses row-major (C-style) memory layout:

```mojo
# Row-major: rows are stored contiguously
var tensor = ones([1024, 1024], DType.float32)

# Iterating row-by-row is cache-efficient
for i in range(1024):
    for j in range(1024):
        _ = tensor[i, j]  # Sequential memory access
```

### Cache Line Alignment

Modern CPUs fetch 64-byte cache lines. Align frequently accessed data:

```mojo
# Good: Access patterns aligned with cache lines
for i in range(1024):
    var row_slice = tensor[i, :]  # Contiguous access

# Avoid: Column-wise access on row-major tensors
for j in range(1024):
    var col_slice = tensor[:, j]  # Non-contiguous access
```

### Memory Prefetching

Mojo's vectorize function automatically prefetches memory for vectorized operations.
No explicit prefetching needed.

## Benchmarking

### Using the SIMD Benchmark Suite

Run the built-in benchmarks to measure SIMD performance:

```bash
# Run all SIMD benchmarks
pixi run mojo run benchmarks/bench_simd.mojo

# Output includes:
# - Operation type (add, multiply, etc.)
# - Tensor sizes (64x64 through 1024x1024)
# - Scalar vs SIMD timing
# - Speedup factor
```

### Interpreting Results

Example output:

```text
Operation | DType   | Size      | Scalar Time | SIMD Time | Speedup
-----------|---------|-----------|-------------|-----------|--------
add        | float32 | 64x64     | 0.15ms      | 0.04ms    | 3.75x
add        | float32 | 256x256   | 2.4ms       | 0.6ms     | 4.0x
add        | float32 | 1024x1024 | 38ms        | 9.5ms     | 4.0x
```

**Interpretation**:

- Smaller tensors: Lower speedup (vectorization overhead dominates)
- Larger tensors: Higher speedup (vectorization amortizes overhead)
- ~4x speedup for float32 is expected on AVX-512 CPUs

### Writing Custom Benchmarks

```mojo
from time import now

fn benchmark_custom(iterations: Int = 100) raises:
    """Benchmark custom operation."""
    var tensor_a = ones([1024, 1024], DType.float32)
    var tensor_b = ones([1024, 1024], DType.float32)

    # Warm up
    _ = add_simd(tensor_a, tensor_b)

    # Benchmark
    var start = now()
    for _ in range(iterations):
        _ = add_simd(tensor_a, tensor_b)
    var end = now()

    var elapsed_ns = end - start
    var elapsed_ms = Float64(elapsed_ns) / 1e6
    var per_iteration = elapsed_ms / Float64(iterations)

    print("Time per iteration:", per_iteration, "ms")
    print("Operations per second:", 1e3 / per_iteration)
```

## Common Bottlenecks

### Bottleneck 1: Non-Contiguous Memory Access

**Problem**: Inefficient cache utilization

```mojo
# WRONG: Column-wise access on row-major tensor
for j in range(cols):
    for i in range(rows):
        process(tensor[i, j])  # Cache misses!

# RIGHT: Row-wise access
for i in range(rows):
    for j in range(cols):
        process(tensor[i, j])  # Sequential access
```

**Solution**: Reshape operations or transpose before processing.

### Bottleneck 2: Type Conversions

**Problem**: Unnecessary casting between types

```mojo
# WRONG: Repeated casting
for idx in range(size):
    var value = Float64(tensor._data[idx])
    process(value)

# RIGHT: Cast once, then operate
var result_ptr = tensor._data.bitcast[Float64]()
for idx in range(size):
    process(result_ptr[idx])
```

### Bottleneck 3: Small Tensor Operations

**Problem**: Vectorization overhead exceeds benefit

```mojo
# WRONG: Overhead dominates for small tensors
var result = add_simd(ones([2, 2], DType.float32),
                      ones([2, 2], DType.float32))

# RIGHT: Scalar operations for small tensors
var result = add_scalar(ones([2, 2], DType.float32),
                        ones([2, 2], DType.float32))
```

**Rule of thumb**: SIMD beneficial for tensors > 256 elements.

### Bottleneck 4: Unaligned Data

**Problem**: Data starts at non-aligned memory address

```mojo
# WRONG: Unaligned allocation
var ptr = DTypePointer[DType.float32].alloc(size)

# RIGHT: Aligned allocation (automatic in ExTensor)
var tensor = zeros([m, n], DType.float32)
```

ML Odyssey tensors are automatically aligned. Avoid raw pointer operations.

## Profiling Tools

### Built-in Profiling

Use simple timing to identify bottlenecks:

```mojo
from time import now

fn profile_operation():
    """Simple profiling."""
    var start = now()
    operation_to_profile()
    var elapsed = now() - start
    print("Time:", Float64(elapsed) / 1e6, "ms")
```

### Performance Measurement

For detailed performance analysis:

```bash
# Run with performance monitoring
time pixi run mojo run examples/performance/simd_optimization.mojo

# Output includes:
# - Real time (wall clock)
# - User time (CPU time)
# - System time (I/O operations)
```

### Optimization Workflow

1. **Baseline**: Measure current performance
2. **Profile**: Identify slow functions
3. **Optimize**: Apply SIMD or algorithm improvements
4. **Benchmark**: Verify improvement
5. **Regression test**: Ensure no performance degradation

### Example Workflow

```mojo
fn main() raises:
    # 1. Baseline measurement
    var tensors = setup_test_data()

    # 2. Profile scalar version
    print("=== Scalar Version ===")
    profile_scalar(tensors)

    # 3. Profile SIMD version
    print("\n=== SIMD Version ===")
    profile_simd(tensors)

    # 4. Compare results
    print("\nVerifying correctness...")
    verify_results(tensors)
```

## References

- [Mojo Manual: SIMD](https://docs.modular.com/mojo/manual/intrinsics/simd/)
- [Mojo Manual: Vectorization](https://docs.modular.com/mojo/manual/algorithm/vectorize/)
- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)
- [Performance Optimization Guide](../core/mojo-patterns.md)

See Also:

- [SIMD Patterns Documentation](../core/mojo-patterns.md#simd-patterns)
- [Benchmarking SIMD Operations](#benchmarking)
- Example: `examples/performance/simd_optimization.mojo`
