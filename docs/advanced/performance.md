# Performance Optimization Guide

Practical strategies for optimizing ML workloads in Mojo, from profiling to deployment.

## Table of Contents

- [Performance Philosophy](#performance-philosophy)
- [SIMD Optimization](#simd-optimization)
- [Memory Management](#memory-management)
- [Profiling and Benchmarking](#profiling-and-benchmarking)
- [Common Optimizations](#common-optimizations)
- [Anti-Patterns](#anti-patterns)
- [When to Optimize](#when-to-optimize)

## Performance Philosophy

**Measure first, optimize second.** Premature optimization is the root of all evil - focus on clarity and correctness,
then profile to identify real bottlenecks.

### Three Levels of Optimization

1. **Algorithm-Level** - Choose better algorithms (e.g., efficient attention mechanisms, sparse operations)
2. **Implementation-Level** - Use language features effectively (Mojo's `fn`, ownership patterns, SIMD)
3. **Hardware-Level** - Exploit CPU/GPU capabilities (cache-friendly layouts, memory bandwidth utilization)

**Priority**: Algorithm > Implementation > Hardware

## SIMD Optimization

SIMD (Single Instruction, Multiple Data) parallelizes element-wise operations across multiple values.

### Basic SIMD Pattern

```mojo
```mojo

fn relu_simd(inout tensor: Tensor):
    """ReLU using SIMD vectorization."""
    alias simd_width = simdwidthof[DType.float32]()

    @parameter
    fn vectorized[width: Int](idx: Int):
        var val = tensor.data.simd_load[width](idx)
        tensor.data.simd_store[width](idx, max(val, 0.0))

    vectorize[simd_width, vectorized](tensor.size())

```text

### When SIMD Helps

- ✅ Element-wise operations (activations, dropout, batch norm)
- ✅ Reductions (sum, mean, variance)
- ✅ Matrix operations (dot products, outer products)
- ❌ Operations with control flow (many branches reduce SIMD effectiveness)
- ❌ Scattered memory access (poor cache utilization)

### SIMD Performance Tips

1. **Align data** - 64-byte alignment for SIMD loads/stores
2. **Process in order** - Sequential access is cache-friendly
3. **Use `alias` for constants** - Compile-time evaluation of `simdwidthof`
4. **Fuse operations** - Combine multiple operations to reduce memory traffic

See [Mojo Patterns](../core/mojo-patterns.md#simd-optimization) for detailed SIMD examples.

## Memory Management

Memory bandwidth is often the bottleneck in ML workloads. Every optimization should reduce memory accesses.

### Memory Hierarchy

```text
```text

L1 Cache (32KB)   - 4 cycles, 64 GB/s
L2 Cache (256KB)  - 10 cycles, 32 GB/s
L3 Cache (8MB)    - 40 cycles, 16 GB/s
RAM (32GB)        - 200+ cycles, 8-16 GB/s

```text

### Optimization Strategies

#### 1. Minimize Allocations

```mojo
```mojo

# BAD: Creates temporaries
fn bad_layer(x: Tensor, w: Tensor) -> Tensor:
    var scaled = x * 2.0      # Allocation
    return scaled @ w          # Allocation

# GOOD: Reuse buffers
fn good_layer(borrowed x: Tensor, borrowed w: Tensor, inout buffer: Tensor) -> Tensor:
    buffer = x @ w
    return buffer * 2.0

```text

#### 2. In-Place Operations

```mojo
```mojo

# Modify data without creating copies
fn inplace_update(inout weights: Tensor, borrowed grad: Tensor, lr: Float64):
    weights -= lr * grad  # Single in-place operation

```text

#### 3. Cache-Friendly Access

```mojo
```mojo

# Row-major iteration for row-major matrices
for i in range(rows):
    for j in range(cols):
        process(matrix[i][j])  # Sequential access

```text

#### 4. Batch Processing

```mojo
```mojo

# Process multiple examples together
fn batch_forward(batch: Tensor, weights: Tensor) -> Tensor:
    # Amortizes memory access overhead across batch
    return batch @ weights.T

```text

## Profiling and Benchmarking

### Timing Critical Sections

```mojo
```mojo

from time import now

fn benchmark_function():
    var start = now()
    # Code to benchmark
    var elapsed = now() - start
    print(f"Time: {elapsed}ms")

```text

### Profiling Tools

**CPU Profiling** (Linux/macOS):

```bash
```bash

# Compile with debug symbols
mojo build -d example.mojo

# Profile with perf
perf record ./example
perf report

```text

**Memory Profiling**:

```bash
```bash

# Use Mojo's memory tracking
mojo --memory-check example.mojo

```text

### Benchmarking Best Practices

1. **Warm up** - Run function twice to stabilize caches
2. **Multiple runs** - Take median of 10+ runs
3. **Control environment** - Disable CPU frequency scaling
4. **Isolate work** - Benchmark single operation in isolation
5. **Compare generations** - Track performance across versions

```mojo
```mojo

fn benchmark_with_warmup(f: fn() -> None, iterations: Int = 10):
    # Warmup
    f()
    f()

    # Benchmark
    var times: List[Int] = List[Int]()
    for _ in range(iterations):
        var start = now()
        f()
        times.append(now() - start)

    # Report median
    times.sort()
    print(f"Median time: {times[iterations // 2]}ms")

```text

## Common Optimizations

### 1. Loop Unrolling

```mojo
```mojo

# Let compiler unroll loop by using @parameter
@parameter
fn process_batch[batch_size: Int](data: Tensor):
    for i in range(batch_size):
        process(data[i])

```text

### 2. Constant Folding

```mojo
```mojo

# Compute constants at compile-time
alias sqrt_2 = 1.4142135623730951

fn he_init(size: Int) -> Float32:
    return sqrt_2 / sqrt(Float32(size))  # Computed at compile-time

```text

### 3. Dead Code Elimination

```mojo
```mojo

# Unused variables and functions are eliminated by compiler
fn process_with_debug(debug: Bool, data: Tensor):
    if debug:
        # This branch is eliminated if debug is compile-time constant
        print_debug_info(data)

```text

### 4. Fusion

```mojo
```mojo

# BAD: Three separate kernels
var h1 = relu(x @ w1)
var h2 = relu(h1 @ w2)
var out = h2 @ w3

# GOOD: Fused kernel
fn fused_forward(borrowed x: Tensor, borrowed w1: Tensor,
                 borrowed w2: Tensor, borrowed w3: Tensor) -> Tensor:
    var h1 = relu(x @ w1)
    var h2 = relu(h1 @ w2)
    return h2 @ w3

```text

### 5. Quantization

```mojo
```mojo

# Use lower precision when possible
fn forward_int8(borrowed x: Tensor[DType.int8]) -> Tensor[DType.int8]:
    # int8 operations are faster and use less memory than float32
    return x @ weights_int8

```text

## Anti-Patterns

### 1. Premature Optimization

```mojo
```mojo

# BAD: Optimizing unproven bottleneck
fn unclear_optimization(x: Tensor):
    # Complex SIMD code that may not be needed
    # Profile first!

```text

### 2. Unclear Memory Ownership

```mojo
```mojo

# BAD: Ambiguous ownership
fn process(x: Tensor) -> Tensor:
    # Does this move, borrow, or copy?

# GOOD: Explicit ownership
fn process(borrowed x: Tensor) -> Tensor:
    # Clearly borrows (read-only)

```text

### 3. Allocating in Loops

```mojo
```mojo

# BAD: Allocates each iteration
for i in range(1000):
    var temp = Tensor.zeros(1000, 1000)  # 1000 allocations!
    process(temp)

# GOOD: Allocate once, reuse
var temp = Tensor.zeros(1000, 1000)
for i in range(1000):
    process(temp)

```text

### 4. Ignoring Data Layout

```mojo
```mojo

# BAD: Transposed access
for j in range(cols):
    for i in range(rows):
        process(matrix[i][j])  # Cache misses!

# GOOD: Sequential access
for i in range(rows):
    for j in range(cols):
        process(matrix[i][j])  # Cache hits!

```text

## When to Optimize

### Don't Optimize

- **Startup code** - Executed once, not in hot path
- **Initialization** - Not repeated during training
- **Logging/debugging** - Overhead is acceptable
- **Error handling** - Exceptional cases are rare

### Do Optimize

- **Forward/backward passes** - Executed thousands to millions of times
- **Inner loops** - Multiplied by iteration count
- **Memory allocations** - Compounded across batches
- **Vectorizable operations** - SIMD gains are significant

### Optimization Checklist

1. [ ] Measure current performance with profiling
2. [ ] Identify actual bottleneck (not assumed)
3. [ ] Choose optimization strategy (algorithm > implementation > hardware)
4. [ ] Implement and benchmark
5. [ ] Verify correctness (unit tests before and after)
6. [ ] Document non-obvious optimizations

## Performance Patterns Reference

See [Mojo Patterns](../core/mojo-patterns.md) for:

- `fn` vs `def` performance characteristics
- `struct` vs `class` allocation strategies
- Ownership patterns for memory safety
- SIMD implementation patterns
- Buffer reuse and allocation strategies
- Type safety for compile-time optimization

## Quick Start

1. **Profile your code** - Find actual bottlenecks
2. **Check algorithm** - Optimize at this level first
3. **Use Mojo patterns** - Apply language features effectively
4. **Vectorize** - Add SIMD to hottest loops
5. **Benchmark** - Verify improvements with measurements
6. **Iterate** - Optimize next bottleneck

## Further Reading

- [Mojo Patterns Guide](../core/mojo-patterns.md) - Language optimization techniques
- SIMD optimization examples in `examples/performance/simd_optimization.mojo`
- Memory efficiency examples in `examples/performance/memory_optimization.mojo`
- [Intel VTune Profiler Guide](https://www.intel.com/content/www/us/en/develop/documentation/vtune-profiler-user-guide/)
