---
name: mojo-simd-optimize
description: Apply SIMD (Single Instruction Multiple Data) optimizations to Mojo code for parallel computation performance. Use when optimizing performance-critical tensor and array operations.
---

# SIMD Optimization Skill

Apply SIMD optimizations to Mojo code for parallel performance.

## When to Use

- Optimizing tensor operations
- Array/vector computations
- Performance-critical loops
- Benchmark results show optimization potential

## SIMD Basics

SIMD processes multiple data elements in parallel:

```mojo
from sys.info import simdwidthof

# Get optimal SIMD width for dtype
alias simd_width = simdwidthof[DType.float32]()  # Usually 8 or 16

# SIMD vector add
fn simd_add(a: Tensor[DType.float32], b: Tensor[DType.float32]):
    for i in range(0, a.size, simd_width):
        let vec_a = a.load[simd_width](i)
        let vec_b = b.load[simd_width](i)
        let result = vec_a + vec_b
        a.store(i, result)
```

## Optimization Patterns

### 1. Vectorize Loops

**Before:**
```mojo
fn add_scalar(a: Tensor, b: Tensor):
    for i in range(a.size):
        a[i] = a[i] + b[i]
```

**After:**
```mojo
fn add_simd(a: Tensor, b: Tensor):
    alias width = simdwidthof[DType.float32]()
    for i in range(0, a.size, width):
        a.store(i, a.load[width](i) + b.load[width](i))
```

### 2. Handle Remainder

```mojo
fn process_with_remainder(data: Tensor):
    alias width = simdwidthof[dtype]()
    let vector_end = (data.size // width) * width

    # SIMD loop
    for i in range(0, vector_end, width):
        # Vectorized processing
        pass

    # Handle remainder
    for i in range(vector_end, data.size):
        # Scalar processing
        pass
```

### 3. Alignment

```mojo
# Ensure data is properly aligned for SIMD
@parameter
fn aligned_load[width: Int](ptr: DTypePointer, offset: Int):
    # Use aligned load for better performance
    pass
```

## Performance Guidelines

- **Use SIMD for** operations on > 1000 elements
- **Typical speedup**: 4x-8x for float32
- **Test performance**: Always benchmark
- **Handle remainder**: Process leftover elements

## Examples

**Vector addition:**
```mojo
fn add[dtype: DType](a: Tensor[dtype], b: Tensor[dtype]) -> Tensor[dtype]:
    alias width = simdwidthof[dtype]()
    for i in range(0, a.size, width):
        a.store(i, a.load[width](i) + b.load[width](i))
    return a
```

**Matrix multiplication (tiled):**
```mojo
fn matmul_simd[dtype: DType](A: Matrix[dtype], B: Matrix[dtype]):
    # Tile for cache + SIMD
    alias tile = 32
    alias width = simdwidthof[dtype]()
    # Implementation with tiling and SIMD
```

See Mojo documentation for complete SIMD API.
