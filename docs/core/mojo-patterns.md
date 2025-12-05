# Mojo Patterns

Core programming patterns and idioms for Mojo development in ML Odyssey.

## Table of Contents

- [SIMD Patterns](#simd-patterns)
- [Vectorization](#vectorization)
- [Type Safety](#type-safety)
- [Memory Management](#memory-management)
- [Performance Optimization](#performance-optimization)

## SIMD Patterns

SIMD (Single Instruction Multiple Data) is fundamental to high-performance tensor operations in Mojo.
This section covers practical patterns for vectorized computation.

### Understanding SIMD Width

SIMD width varies by data type and CPU architecture. Use `simdwidthof` to get the compile-time
SIMD width for a specific dtype:

```mojo
from sys.info import simd_width_of

# Get SIMD width at compile time
alias float32_width = simd_width_of[DType.float32]()  # Usually 8-16
alias float64_width = simd_width_of[DType.float64]()  # Usually 4-8
```

**Key Point**: Float32 can be vectorized wider than Float64 (2x more elements per instruction).
This is why Float32 operations typically see 4x speedup while Float64 sees 2x speedup.

### Pattern 1: Basic Vectorization with `vectorize`

The `vectorize` function applies a parameterized function to a range with automatic SIMD batching:

```mojo
from algorithm import vectorize
from sys.info import simd_width_of

fn relu_simd(mut tensor: ExTensor):
    """ReLU activation using SIMD vectorization."""
    alias width = simd_width_of[DType.float32]()

    @parameter
    fn vectorized_relu[simd_width: Int](idx: Int):
        # Load a vector of elements
        var values = tensor._data.simd_load[simd_width](idx)
        # Apply element-wise max with 0
        var activated = max(values, 0.0)
        # Store result back
        tensor._data.simd_store[simd_width](idx, activated)

    # Vectorize over all elements, processing `width` per iteration
    vectorize[width, vectorized_relu](tensor.numel())
```

**How it works**:

1. `simd_load[width]` loads `width` elements into a SIMD vector
2. Operations on SIMD vectors are executed in parallel
3. `simd_store[width]` writes the results back
4. `vectorize` automatically batches iterations with the specified SIMD width

**Performance**: ~4x speedup for Float32 ReLU on modern CPUs (AVX2/AVX-512).

### Pattern 2: SIMD with Reduction

Reductions (like dot products) combine multiple values into a single result:

```mojo
fn dot_product_simd(a: ExTensor, b: ExTensor) -> Float32:
    """Compute dot product using SIMD."""
    alias width = simd_width_of[DType.float32]()
    var sum = SIMD[DType.float32, width](0.0)  # Vector accumulator

    @parameter
    fn vectorized_dot[simd_width: Int](idx: Int):
        var a_vec = a._data.simd_load[simd_width](idx)
        var b_vec = b._data.simd_load[simd_width](idx)
        # Accumulate products
        sum += a_vec * b_vec

    vectorize[width, vectorized_dot](a.numel())

    # Reduce vector to scalar
    return sum.reduce_add()
```

**Key Pattern**:

1. Accumulate into a SIMD vector (not a scalar)
2. Each iteration processes `width` elements in parallel
3. Call `.reduce_add()` at the end to combine all elements

**Performance**: ~4x speedup for float32 dot products.

### Pattern 3: Nested SIMD for Multi-Dimensional Operations

For operations involving multiple loop nesting levels, apply SIMD to the innermost loop:

```mojo
fn matmul_simd(a: ExTensor, b: ExTensor) -> ExTensor:
    """Matrix multiplication with SIMD on innermost loop."""
    var m = a.shape()[0]
    var n = b.shape()[1]
    var k = a.shape()[1]
    var result = zeros([m, n], DType.float32)

    alias width = simd_width_of[DType.float32]()

    for i in range(m):
        for j in range(n):
            var sum = SIMD[DType.float32, width](0.0)

            @parameter
            fn dot_product[simd_width: Int](idx: Int):
                var a_vec = a._data.simd_load[simd_width](i * k + idx)
                var b_vec = b._data.simd_load[simd_width](idx * n + j)
                sum += a_vec * b_vec

            vectorize[width, dot_product](k)
            result[i, j] = sum.reduce_add()

    return result
```

**Rationale**: Outer loops (i, j) iterate over problem structure. Inner loop (k) vectorizes
element operations. This maximizes parallelism for each output element.

### Pattern 4: Conditional SIMD with Masking

For operations with conditionals (e.g., comparisons), use SIMD element-wise:

```mojo
fn leaky_relu_simd(mut tensor: ExTensor, alpha: Float32):
    """Leaky ReLU with SIMD."""
    alias width = simd_width_of[DType.float32]()

    @parameter
    fn vectorized_leaky_relu[simd_width: Int](idx: Int):
        var values = tensor._data.simd_load[simd_width](idx)
        # Element-wise conditional: select(condition, if_true, if_false)
        var result = select(values > 0.0, values, values * alpha)
        tensor._data.simd_store[simd_width](idx, result)

    vectorize[width, vectorized_leaky_relu](tensor.numel())
```

**Key Point**: SIMD operations support element-wise conditionals without branching,
maintaining vectorization efficiency.

### Pattern 5: Type-Specific SIMD Implementation

Different data types require different SIMD parameters. Common approach:

```mojo
fn add_simd(a: ExTensor, b: ExTensor) raises -> ExTensor:
    """Add with dispatch to type-specific SIMD."""
    if a.dtype() == DType.float32:
        return _add_simd_float32(a, b)
    elif a.dtype() == DType.float64:
        return _add_simd_float64(a, b)
    else:
        raise Error("Unsupported dtype")

fn _add_simd_float32(a: ExTensor, b: ExTensor) raises -> ExTensor:
    """SIMD addition optimized for float32."""
    alias width = simd_width_of[DType.float32]()
    var result = zeros(a.shape(), DType.float32)

    @parameter
    fn vectorized_add[simd_width: Int](idx: Int):
        var a_vec = a._data.simd_load[simd_width](idx)
        var b_vec = b._data.simd_load[simd_width](idx)
        result._data.simd_store[simd_width](idx, a_vec + b_vec)

    vectorize[width, vectorized_add](a.numel())
    return result^
```

**Why**: Float32 and Float64 have different SIMD widths, requiring separate implementations.
This maximizes performance for each type.

### Pattern 6: Fallback for Non-Contiguous Data

Real-world tensors may not be contiguous in memory. Handle this gracefully:

```mojo
fn operation_simd(a: ExTensor, b: ExTensor) -> ExTensor:
    """Operation with automatic fallback."""
    # Check if tensors are compatible for SIMD
    if a.numel() % simd_width_of[DType.float32]() != 0:
        # Fall back to scalar implementation
        return operation_scalar(a, b)

    # SIMD implementation
    return operation_simd_impl(a, b)
```

**Best Practice**: Always provide a scalar fallback for edge cases.

## Vectorization

### `@parameter` Functions

Functions decorated with `@parameter` are parameterized by the Mojo compiler and can take
compile-time parameters. This enables conditional compilation:

```mojo
@parameter
fn vectorized_op[width: Int](idx: Int):
    # This function is instantiated with specific width values
    # The compiler knows `width` at compile time
    var values = tensor._data.simd_load[width](idx)
```

### `@always_inline`

For performance-critical SIMD functions, force inlining to eliminate function call overhead:

```mojo
@always_inline
fn _add_simd_float32(a: ExTensor, b: ExTensor, mut result: ExTensor):
    """Always inlined for performance."""
    # Function body...
```

## Type Safety

Mojo's type system prevents many common bugs. Key patterns:

### Immutable References (Default)

```mojo
fn process(tensor: ExTensor):  # `read` is implicit
    # Cannot modify tensor
    print(tensor.shape())
```

### Mutable References

```mojo
fn modify(mut tensor: ExTensor):
    # Can modify tensor in place
    tensor._fill_zero()
```

### Owned Values

```mojo
fn consume(var tensor: ExTensor):
    # Function takes ownership
    # Caller cannot use tensor after this call
    tensor += 1
```

## Memory Management

### Ownership Transfer with `^`

```mojo
fn get_data(self) -> List[Int]:
    # Transfer ownership to caller
    return self.data^
```

### Constructor Patterns

- **`__init__` (constructor)**: Always use `out self` to create new instance
- **`__copyinit__`**: Use `out self, existing` to create a copy
- **`__moveinit__`**: Use `out self, deinit existing` to move ownership

```mojo
struct Point(Copyable, Movable):
    var x: Float32
    var y: Float32

    fn __init__(out self, x: Float32, y: Float32):
        self.x = x
        self.y = y
```

## Performance Optimization

### Benchmarking SIMD Operations

Expected performance improvements:

- **Float32 SIMD**: 3-5x speedup for vectorizable operations
- **Float64 SIMD**: 2-3x speedup (half the SIMD width)
- **Larger tensors**: Better speedup (better amortization of overhead)

### When to Use SIMD

✅ **Use SIMD for**:

- Element-wise arithmetic (add, multiply, divide)
- Activation functions (ReLU, sigmoid, tanh)
- Reductions (sum, dot product, mean)
- Batch normalization
- Layer operations with independent computations per element

❌ **Don't use SIMD for**:

- Complex branching logic (breaks vectorization)
- Sparse operations (inefficient memory layout)
- Single-element operations (overhead not worth it)

### SIMD vs Broadcasting

```mojo
# SIMD: Same shape, significant speedup
var a = ones([1024, 1024], DType.float32)
var b = ones([1024, 1024], DType.float32)
var c = add_simd(a, b)  # SIMD, 4x speedup

# Broadcasting: Different shapes, falls back to scalar
var x = ones([1, 1024], DType.float32)
var y = ones([1024, 1024], DType.float32)
var z = add_simd(x, y)  # Falls back to scalar
```

### Examples

See the following files for complete working examples:

- **Basic SIMD patterns**: `/examples/mojo-patterns/simd_example.mojo`
- **Performance examples**: `/examples/performance/simd_optimization.mojo`
- **SIMD arithmetic**: `/shared/core/arithmetic_simd.mojo`
- **Benchmarks**: `/benchmarks/bench_simd.mojo`

## References

- [Mojo Manual: SIMD](https://docs.modular.com/mojo/manual/intrinsics/simd/)
- [Mojo Manual: Vectorization](https://docs.modular.com/mojo/manual/algorithm/vectorize/)
- [sys.info: simd_width_of](https://docs.modular.com/mojo/stdlib/sys/info/)
