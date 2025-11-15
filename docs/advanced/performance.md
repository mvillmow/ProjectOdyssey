# Performance Optimization Guide

SIMD optimization, profiling, and performance tuning techniques for ML Odyssey.

## Overview

Mojo is designed for high-performance ML workloads. This guide covers optimization techniques to maximize speed
and efficiency in your neural network implementations.

## Performance Hierarchy

Optimize in this order:

1. **Algorithm** - Choose efficient algorithms first
2. **Memory Layout** - Optimize data structures
3. **SIMD** - Vectorize operations
4. **Parallelization** - Multi-threading
5. **GPU** - Offload to accelerators (future)

## SIMD Optimization

### Understanding SIMD

SIMD (Single Instruction, Multiple Data) processes multiple values simultaneously:

```mojo
from algorithm import vectorize
from sys.info import simdwidthof

fn simd_example():
    """Demonstrate SIMD performance."""
    var size = 1000000
    var data = Tensor.randn(size)

    # Scalar version (slow)
    @parameter
    fn scalar_multiply():
        for i in range(size):
            data[i] = data[i] * 2.0

    # SIMD version (fast)
    @parameter
    fn simd_multiply():
        alias width = simdwidthof[DType.float32]()

        @parameter
        fn vectorized[w: Int](idx: Int):
            var vec = data.load[w](idx)
            data.store[w](idx, vec * 2.0)

        vectorize[width, vectorized](size)

    # SIMD is typically 4-8x faster
```

### Vectorizing Loops

Convert scalar loops to SIMD for massive performance gains.

See `examples/performance/simd_optimization.mojo`](

Key pattern:

```mojo
fn relu_simd(inout tensor: Tensor):
    """ReLU activation with SIMD."""
    alias simd_width = simdwidthof[DType.float32]()

    @parameter
    fn vectorized_relu[width: Int](idx: Int):
        var val = tensor.data.simd_load[width](idx)
        tensor.data.simd_store[width](idx, max(val, 0.0))

    vectorize[simd_width, vectorized_relu](tensor.size())
```

Full example: `examples/performance/simd_optimization.mojo`

### Matrix Multiplication

Optimized matmul with SIMD:

```mojo
fn matmul_simd(borrowed a: Tensor, borrowed b: Tensor) -> Tensor:
    """Optimized matrix multiplication."""
    var m = a.shape[0]
    var n = b.shape[1]
    var k = a.shape[1]

    var result = Tensor.zeros(m, n)

    alias simd_width = simdwidthof[DType.float32]()

    for i in range(m):
        for j in range(n):
            var sum = SIMD[DType.float32, simd_width](0.0)

            @parameter
            fn dot_product[width: Int](idx: Int):
                var a_vec = a.load[width](i * k + idx)
                var b_vec = b.load[width](idx * n + j)
                sum += a_vec * b_vec

            vectorize[simd_width, dot_product](k)

            result[i, j] = sum.reduce_add()

    return result
```

## Parallelization

### Multi-Threading

Use `parallelize` for data-parallel operations:

```mojo
from algorithm import parallelize

fn batch_forward_parallel(borrowed inputs: Tensor, borrowed weights: Tensor) -> Tensor:
    """Process batch samples in parallel."""
    var batch_size = inputs.shape[0]
    var outputs = Tensor.zeros(batch_size, weights.shape[1])

    @parameter
    fn process_sample(i: Int):
        var input = inputs[i]
        outputs[i] = input @ weights

    parallelize[process_sample](batch_size, num_workers=8)

    return outputs

fn data_augmentation_parallel(inout images: Tensor):
    """Apply augmentation to batch in parallel."""

    @parameter
    fn augment_image(i: Int):
        images[i] = random_crop(images[i])
        images[i] = random_flip(images[i])

    parallelize[augment_image](images.shape[0])
```

### Gradient Computation

Parallelize backward pass:

```mojo
struct ParallelLinear:
    """Linear layer with parallel gradient computation."""
    var weight: Tensor
    var bias: Tensor

    fn backward(
        inout self,
        borrowed grad_output: Tensor,
        borrowed input: Tensor
    ) -> Tensor:
        """Compute gradients in parallel."""

        # Parallelize weight gradient computation
        @parameter
        fn compute_weight_grad(i: Int):
            for j in range(self.weight.shape[1]):
                var grad = 0.0
                for b in range(grad_output.shape[0]):
                    grad += grad_output[b, i] * input[b, j]
                self.weight.grad[i, j] = grad

        parallelize[compute_weight_grad](self.weight.shape[0])

        # Compute input gradient
        return grad_output @ self.weight
```

## Memory Optimization

### In-Place Operations

Avoid unnecessary allocations for better memory efficiency.

See `examples/performance/memory_optimization.mojo`](

Key pattern:

```mojo
# Bad: Creates temporary tensors (2 allocations)
fn bad_update(weights: Tensor, grad: Tensor, lr: Float64) -> Tensor:
    var scaled_grad = grad * lr
    return weights - scaled_grad

# Good: In-place update (no allocation)
fn good_update(inout weights: Tensor, borrowed grad: Tensor, lr: Float64):
    weights -= lr * grad

# Best: Fused SIMD in-place operation (no allocation, vectorized)
fn best_update(inout weights: Tensor, borrowed grad: Tensor, lr: Float64):
    # ... (see full example for SIMD implementation)
```

Full example: `examples/performance/memory_optimization.mojo`

### Memory Layout

Optimize data layout for cache efficiency:

```mojo
# Bad: Column-major access (cache-unfriendly)
fn bad_sum_rows(borrowed matrix: Tensor) -> Tensor:
    var result = Tensor.zeros(matrix.shape[0])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            result[i] += matrix[i, j]  # Jumps through memory
    return result

# Good: Row-major access (cache-friendly)
fn good_sum_rows(borrowed matrix: Tensor) -> Tensor:
    var result = Tensor.zeros(matrix.shape[0])
    for i in range(matrix.shape[0]):
        var sum = 0.0
        for j in range(matrix.shape[1]):
            sum += matrix[i, j]  # Sequential access
        result[i] = sum
    return result
```

### Buffer Reuse

Preallocate and reuse buffers:

```mojo
struct EfficientConv2D:
    """Conv2D with preallocated buffers."""
    var weight: Tensor
    var im2col_buffer: Tensor  # Reused buffer

    fn __init__(inout self, in_channels: Int, out_channels: Int, kernel_size: Int):
        self.weight = Tensor.randn(out_channels, in_channels, kernel_size, kernel_size)
        # Preallocate maximum buffer size
        self.im2col_buffer = Tensor.zeros(1024, in_channels * kernel_size * kernel_size)

    fn forward(inout self, borrowed input: Tensor) -> Tensor:
        # Reuse buffer instead of allocating
        im2col(input, self.im2col_buffer)
        return self.im2col_buffer @ self.weight.reshape(-1)
```

## Profiling

### Using Profiler

Built-in profiling utilities:

```mojo
from shared.utils import Profiler

fn profile_training():
    """Profile training loop."""
    var profiler = Profiler()

    for epoch in range(num_epochs):
        for batch in train_loader:
            with profiler.section("data_loading"):
                var inputs, targets = batch

            with profiler.section("forward"):
                var outputs = model.forward(inputs)

            with profiler.section("loss"):
                var loss = loss_fn(outputs, targets)

            with profiler.section("backward"):
                loss.backward()

            with profiler.section("optimizer"):
                optimizer.step(model.parameters())

    # Print results
    profiler.print_summary()
```

Output:

```text
Profiling Results:
==================
Section          | Total Time | Avg Time | Calls | %
---------------------------------------------------------
data_loading     |   2.45s    |  4.9ms   | 500   | 12%
forward          |   8.32s    | 16.6ms   | 500   | 40%
loss             |   1.23s    |  2.5ms   | 500   |  6%
backward         |   7.11s    | 14.2ms   | 500   | 34%
optimizer        |   1.65s    |  3.3ms   | 500   |  8%
---------------------------------------------------------
Total            |  20.76s    |          |       | 100%
```

### Benchmarking

Measure operation performance:

```mojo
from shared.utils import benchmark

@benchmark(iterations=100, warmup=10)
fn bench_conv2d():
    """Benchmark Conv2D forward pass."""
    var conv = Conv2D(64, 64, kernel_size=3)
    var input = Tensor.randn(32, 64, 56, 56)

    @timer
    var output = conv.forward(input)

    return output

fn main():
    var results = bench_conv2d()
    print("Avg time:", results.mean_time, "ms")
    print("Std dev:", results.std_time, "ms")
    print("Min time:", results.min_time, "ms")
    print("Max time:", results.max_time, "ms")
```

### Memory Profiling

Track memory usage:

```mojo
from shared.utils import MemoryProfiler

fn profile_memory():
    """Profile memory allocations."""
    var mem_profiler = MemoryProfiler()

    with mem_profiler.track("model_creation"):
        var model = LeNet5()

    with mem_profiler.track("data_loading"):
        var data = load_mnist()

    with mem_profiler.track("training"):
        train(model, data)

    mem_profiler.print_summary()
```

## Optimization Patterns

### Pattern 1: Fused Operations

Combine operations to reduce memory traffic:

```mojo
# Bad: Separate operations (3 memory passes)
fn bad_activation(input: Tensor) -> Tensor:
    var x = input * 2.0      # Pass 1: read input, write temp1
    var y = x + 1.0          # Pass 2: read temp1, write temp2
    return max(y, 0.0)       # Pass 3: read temp2, write output

# Good: Fused operation (1 memory pass)
fn good_activation(borrowed input: Tensor) -> Tensor:
    var output = Tensor.zeros_like(input)

    @parameter
    fn fused[width: Int](idx: Int):
        var x = input.load[width](idx)
        var result = max(x * 2.0 + 1.0, 0.0)
        output.store[width](idx, result)

    vectorize[simdwidthof[DType.float32](), fused](input.size())
    return output
```

### Pattern 2: Tiled Matrix Multiplication

Improve cache utilization:

```mojo
fn matmul_tiled(borrowed a: Tensor, borrowed b: Tensor, tile_size: Int = 32) -> Tensor:
    """Tiled matrix multiplication for better cache performance."""
    var m = a.shape[0]
    var n = b.shape[1]
    var k = a.shape[1]

    var result = Tensor.zeros(m, n)

    # Tile the computation
    for i in range(0, m, tile_size):
        for j in range(0, n, tile_size):
            for kk in range(0, k, tile_size):
                # Process tile
                var i_end = min(i + tile_size, m)
                var j_end = min(j + tile_size, n)
                var k_end = min(kk + tile_size, k)

                for ii in range(i, i_end):
                    for jj in range(j, j_end):
                        var sum = 0.0
                        for kkk in range(kk, k_end):
                            sum += a[ii, kkk] * b[kkk, jj]
                        result[ii, jj] += sum

    return result
```

### Pattern 3: Lazy Evaluation

Defer computation until necessary:

```mojo
struct LazyTensor:
    """Tensor with lazy evaluation."""
    var data: Tensor
    var pending_ops: List[Operation]
    var materialized: Bool

    fn __init__(inout self, data: Tensor):
        self.data = data
        self.pending_ops = List[Operation]()
        self.materialized = True

    fn add(inout self, borrowed other: Tensor) -> Self:
        """Queue addition for later."""
        self.pending_ops.append(AddOp(other))
        self.materialized = False
        return self

    fn multiply(inout self, scalar: Float64) -> Self:
        """Queue multiplication for later."""
        self.pending_ops.append(MulOp(scalar))
        self.materialized = False
        return self

    fn materialize(inout self):
        """Execute all pending operations at once."""
        if self.materialized:
            return

        # Fuse all operations into single kernel
        @parameter
        fn fused_ops[width: Int](idx: Int):
            var val = self.data.load[width](idx)
            for op in self.pending_ops:
                val = op.apply(val)
            self.data.store[width](idx, val)

        vectorize[simdwidthof[DType.float32](), fused_ops](self.data.size())

        self.pending_ops.clear()
        self.materialized = True
```

## Compile-Time Optimization

### Parameter Specialization

Use `@parameter` for compile-time optimization:

```mojo
fn create_conv[@parameter activation: String](
    in_channels: Int,
    out_channels: Int
) -> Conv2D:
    """Create conv layer with compile-time activation."""
    var conv = Conv2D(in_channels, out_channels, kernel_size=3)

    @parameter
    if activation == "relu":
        conv.set_activation(ReLU())
    elif activation == "tanh":
        conv.set_activation(Tanh())
    else:
        compile_error("Unknown activation: " + activation)

    return conv

# Generates optimized code for each activation
var conv1 = create_conv["relu"](64, 128)
var conv2 = create_conv["tanh"](128, 256)
```

### Static Shape Inference

Leverage compile-time shape knowledge:

```mojo
fn matmul_static[@parameter M: Int, @parameter N: Int, @parameter K: Int](
    borrowed a: Tensor,
    borrowed b: Tensor
) -> Tensor:
    """Matrix multiplication with static shapes."""
    # Shapes known at compile time enable better optimization
    var result = Tensor.zeros(M, N)

    @parameter
    for i in range(M):
        @parameter
        for j in range(N):
            var sum = 0.0
            @parameter
            for k in range(K):
                sum += a[i, k] * b[k, j]
            result[i, j] = sum

    return result

# Compiler generates optimal code for specific sizes
var output = matmul_static[128, 64, 256](a, b)
```

## Performance Best Practices

### DO

- ✅ Profile before optimizing
- ✅ Use SIMD for hot loops
- ✅ Reuse buffers when possible
- ✅ Prefer in-place operations
- ✅ Align data for SIMD
- ✅ Use compile-time parameters
- ✅ Batch operations
- ✅ Fuse operations

### DON'T

- ❌ Optimize without profiling
- ❌ Allocate in hot loops
- ❌ Use scalar operations for large data
- ❌ Ignore cache locality
- ❌ Over-parallelize small workloads
- ❌ Sacrifice readability without measurement

## Performance Checklist

Before declaring code optimized:

- [ ] Profiled and identified bottlenecks
- [ ] Vectorized hot loops with SIMD
- [ ] Minimized memory allocations
- [ ] Used in-place operations where possible
- [ ] Optimized memory layout
- [ ] Added parallelization if beneficial
- [ ] Benchmarked against baseline
- [ ] Documented performance characteristics

## Next Steps

- **[Custom Layers](custom-layers.md)** - Apply optimizations to custom layers
- **[Debugging](debugging.md)** - Debug performance issues
- **[Mojo Patterns](../core/mojo-patterns.md)** - Performance-oriented patterns
- **[Visualization](visualization.md)** - Visualize performance metrics

## Related Documentation

- [Shared Library](../core/shared-library.md) - Optimized shared components
- [Testing Strategy](../core/testing-strategy.md) - Performance testing
- [Project Structure](../core/project-structure.md) - Benchmarks directory
