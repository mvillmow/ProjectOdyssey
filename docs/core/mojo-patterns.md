# Mojo Patterns for ML Development

<!-- markdownlint-disable MD051 -->

A comprehensive guide to Mojo language patterns optimized for machine learning workloads.

## Table of Contents

- [Why Mojo for ML](#why-mojo-for-ml)
- [Core Language Patterns](#core-language-patterns)
- [ML-Specific Patterns](#ml-specific-patterns)
- [Common Patterns](#common-patterns)
- [Anti-Patterns](#anti-patterns)
- [Examples](#examples)

## Why Mojo for ML

Mojo is designed from the ground up for AI/ML workloads, offering significant advantages over traditional approaches:

### Performance Benefits

**Faster Execution**: Mojo compiles to optimized machine code, delivering performance competitive with
C/C++ while maintaining Python-like syntax. For ML workloads, this translates to:

- 10-1000x faster tensor operations compared to Python
- Zero-cost abstractions - high-level code without runtime overhead
- Native SIMD support for parallel computation
- Efficient memory layout for cache-friendly access

### Type Safety

**Compile-time Error Detection**: Mojo's strong type system catches errors before runtime:

```mojo

fn matrix_multiply[dtype: DType](a: Tensor[dtype], b: Tensor[dtype]) -> Tensor[dtype]:
    # Compiler enforces shape compatibility and type consistency
    if a.shape[1] != b.shape[0]:
        raise Error("Incompatible shapes")
    return result

```

Benefits:

- Prevents shape mismatches in neural network layers
- Ensures numeric type consistency (float32 vs float64)
- Catches gradient computation errors at compile time
- Eliminates entire classes of runtime bugs

### Memory Safety

**Built-in Ownership System**: Inspired by Rust, Mojo provides memory safety without garbage collection:

- No null pointer dereferences
- No use-after-free bugs
- No data races in concurrent code
- Predictable memory usage for large models

### SIMD Optimization

**First-Class Vectorization**: Mojo treats SIMD as a core language feature, not an afterthought:

```mojo

fn vectorized_relu(inout tensor: Tensor):
    alias simd_width = simdwidthof[DType.float32]()

    @parameter
    fn relu_kernel[width: Int](idx: Int):
        var val = tensor.data.simd_load[width](idx)
        tensor.data.simd_store[width](idx, max(val, 0.0))

    vectorize[simd_width, relu_kernel](tensor.size())

```

Benefits:

- Automatic vectorization of element-wise operations
- Hardware-optimal SIMD width selection
- Explicit control when needed
- Portable across CPU architectures

### Future-Proof Design

**Built for Modern AI**: Mojo is actively developed with AI/ML as the primary use case:

- Native tensor operations
- Hardware accelerator support (GPU, TPU)
- Python interoperability for ecosystem access
- Growing ML-focused standard library

## Core Language Patterns

### Function Definitions: `fn` vs `def`

Mojo provides two function declaration keywords with different trade-offs.

#### Use `fn` for Performance-Critical Code

### When to use `fn`

- Performance-critical functions requiring compile-time optimization
- Functions with explicit type annotations
- SIMD/vectorized operations
- Functions that don't need Python-style dynamic behavior

### Characteristics

- Requires explicit type annotations
- Arguments are immutable by default
- Enables aggressive compiler optimizations
- Prevents accidental mutations

```mojo

fn forward_pass[dtype: DType](
    borrowed input: Tensor[dtype],
    borrowed weights: Tensor[dtype],
    borrowed bias: Tensor[dtype]
) -> Tensor[dtype]:
    """Optimized forward pass with compile-time type checking."""
    var output = input @ weights  # Matrix multiplication
    output += bias
    return output

```

### Benefits

- Compile-time optimization - functions are inlined and optimized
- Type safety - errors caught at compile time
- No runtime overhead - zero-cost abstractions
- Clear intent - explicitly marked as performance-sensitive

#### Use `def` for Python Compatibility

### When to use `def`

- Python-compatible functions
- Dynamic typing needed for flexibility
- Quick prototypes and experiments
- Functions requiring Python interop

### Characteristics

- Optional type annotations
- Arguments are mutable by default (Python-like)
- More flexible but less optimized
- Compatible with Python calling conventions

```mojo

def load_dataset(path: String) -> PythonObject:
    """Load dataset using Python library."""
    from python import Python

    var np = Python.import_module("numpy")
    var data = np.load(path)
    return data

```

### Benefits

- Rapid prototyping - faster to write
- Python interoperability - seamless integration
- Dynamic behavior - flexibility when needed
- Familiar syntax - easier for Python developers

#### Decision Guide

```text
Need performance? ──────────────────────> Use `fn`
    │
    └─> Need Python interop? ──────────> Use `def`
         │
         └─> Prototype? ────────────────> Use `def` (convert to `fn` later)

```

### Type Definitions: `struct` vs `class`

Mojo provides both `struct` and `class` with distinct semantics.

#### Use `struct` for Value Types

### When to use `struct`

- Value types with stack allocation
- Performance-critical data structures
- Immutable or copy-by-value semantics
- SIMD-compatible types
- Most ML components (layers, tensors, optimizers)

### Characteristics

- Stack-allocated by default (fast)
- Copy-by-value semantics
- No inheritance (composition over inheritance)
- Optimized for performance

```mojo

struct Layer:
    """Neural network layer as a value type."""
    var weights: Tensor[DType.float32]
    var bias: Tensor[DType.float32]
    var activation: String

    fn __init__(inout self, input_size: Int, output_size: Int):
        """Initialize layer with random weights."""
        self.weights = Tensor[DType.float32].randn(output_size, input_size)
        self.bias = Tensor[DType.float32].zeros(output_size)
        self.activation = "relu"

    fn forward(self, borrowed input: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Forward pass."""
        var output = input @ self.weights.T + self.bias
        if self.activation == "relu":
            return relu(output)
        return output

    fn __copyinit__(inout self, other: Self):
        """Explicit copy constructor."""
        self.weights = other.weights
        self.bias = other.bias
        self.activation = other.activation

```

### Benefits

- Fast allocation - stack allocation is faster than heap
- Cache-friendly - better memory locality
- Predictable - no hidden allocations or indirection
- Safe - no shared mutable state

#### Use `class` for Reference Types

### When to use `class`

- Reference types with heap allocation
- Object-oriented inheritance hierarchies
- Shared mutable state across references
- Python interoperability

### Characteristics

- Heap-allocated (managed memory)
- Reference semantics - multiple references to same object
- Supports inheritance
- Similar to Python classes

```mojo

class Model:
    """Neural network model as a reference type."""
    var layers: List[Layer]
    var training: Bool

    def __init__(inout self):
        """Initialize empty model."""
        self.layers = List[Layer]()
        self.training = True

    def add_layer(inout self, layer: Layer):
        """Add layer to model."""
        self.layers.append(layer)

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass through all layers."""
        var output = input
        for i in range(len(self.layers)):
            output = self.layers[i].forward(output)
        return output

```

### Benefits

- Shared state - multiple references to same object
- Inheritance - polymorphic behavior
- Python-like - familiar to Python developers
- Flexibility - dynamic behavior when needed

#### Decision Guide

```text
Need value semantics? ───────────────────> Use `struct`
    │
    ├─> Need high performance? ──────────> Use `struct`
    │
    ├─> Need inheritance? ───────────────> Use `class`
    │
    └─> Need shared state? ──────────────> Use `class`

```

### Ownership Patterns

Mojo's ownership system provides memory safety without garbage collection.

#### `borrowed`: Read-Only Access

### Use `borrowed` when

- Function needs to read data without modifying it
- No ownership transfer required
- Multiple concurrent readers needed

```mojo

fn compute_loss(borrowed predictions: Tensor, borrowed targets: Tensor) -> Float64:
    """Compute loss without taking ownership."""
    var diff = predictions - targets
    return (diff * diff).mean()

```

### Characteristics

- No ownership transfer
- Read-only access
- Multiple borrows allowed simultaneously
- Zero runtime cost

#### `owned`: Transfer Ownership

### Use `owned` when

- Function consumes/transforms data
- Ownership transfer is intentional
- Resource cleanup needed

```mojo

fn consume_tensor(owned tensor: Tensor) -> Float64:
    """Take ownership and consume tensor."""
    var result = tensor.sum()
    # tensor is destroyed here automatically
    return result

```

### Characteristics

- Transfers ownership to function
- Original variable becomes invalid
- Move semantics (no copy)
- Ensures single owner

#### `inout`: Mutable Reference

### Use `inout` when

- Function needs to modify data in place
- No ownership transfer desired
- Efficient mutations without copying

```mojo

fn update_weights(
    inout weights: Tensor,
    borrowed gradients: Tensor,
    lr: Float64
):
    """Update weights in place."""
    weights -= lr * gradients  # Modifies original

```

### Characteristics

- Mutable reference
- No ownership transfer
- In-place modifications
- Only one mutable reference at a time

#### Ownership Decision Guide

```text
Need to read? ──────────────────────────> Use `borrowed`
    │
    └─> Need to modify in place? ───────> Use `inout`
         │
         └─> Need to consume? ───────────> Use `owned`

```

## ML-Specific Patterns

### Tensor Operations with SIMD

SIMD (Single Instruction, Multiple Data) enables parallel processing of tensor elements.

#### Element-wise Operations

```mojo

fn relu_simd(inout tensor: Tensor):
    """ReLU activation using SIMD."""
    alias simd_width = simdwidthof[DType.float32]()

    @parameter
    fn vectorized_relu[width: Int](idx: Int):
        var val = tensor.data.simd_load[width](idx)
        tensor.data.simd_store[width](idx, max(val, 0.0))

    vectorize[simd_width, vectorized_relu](tensor.size())

```

### Key points

- `alias simd_width` - compile-time constant for optimal width
- `@parameter` - compile-time function generation
- `simd_load/simd_store` - load/store SIMD vectors
- `vectorize` - applies function across all elements

#### Reduction Operations

```mojo

fn sum_simd(borrowed tensor: Tensor) -> Float32:
    """Sum all elements using SIMD."""
    alias simd_width = simdwidthof\[DType.float32\]()
    var accumulator = SIMD\[DType.float32, simd_width\](0.0)

    @parameter
    fn vectorized_sum\[width: Int\](idx: Int):
        accumulator += tensor.data.simd_load\[width\](idx)

    vectorize\[simd_width, vectorized_sum\](tensor.size())
    return accumulator.reduce_add()

```

### Key points

- SIMD accumulator for parallel reduction
- `reduce_add()` - horizontal sum of SIMD vector
- Significant speedup for large tensors

#### Matrix Multiplication

```mojo

fn matmul_simd(borrowed a: Tensor, borrowed b: Tensor) -> Tensor:
    """Optimized matrix multiplication using SIMD."""
    var m = a.shape[0]
    var n = b.shape[1]
    var k = a.shape[1]

    var result = Tensor.zeros(m, n)
    alias simd_width = simdwidthof\[DType.float32\]()

    for i in range(m):
        for j in range(n):
            var sum = SIMD\[DType.float32, simd_width\](0.0)

            @parameter
            fn dot_product\[width: Int\](idx: Int):
                var a_vec = a.load\[width\](i * k + idx)
                var b_vec = b.load\[width\](idx * n + j)
                sum += a_vec * b_vec

            vectorize\[simd_width, dot_product\](k)
            result\[i, j\] = sum.reduce_add()

    return result

```

### Key points

- Vectorized inner product computation
- Cache-friendly access patterns
- Significant performance improvement over scalar code

### Memory-Efficient Implementations

#### In-Place Operations

```mojo

# Anti-pattern: Allocates temporary tensors
fn bad_update(weights: Tensor, grad: Tensor, lr: Float64) -> Tensor:
    """Inefficient update with multiple allocations."""
    var scaled_grad = grad * lr      # Allocation 1
    return weights - scaled_grad      # Allocation 2

# Good pattern: In-place update
fn good_update(inout weights: Tensor, borrowed grad: Tensor, lr: Float64):
    """Efficient in-place update without allocations."""
    weights -= lr * grad  # No temporary tensors

# Best pattern: Fused SIMD in-place update
fn best_update(inout weights: Tensor, borrowed grad: Tensor, lr: Float64):
    """Fused in-place update with SIMD."""
    alias width = simdwidthof[DType.float32]()

    @parameter
    fn fused_update[w: Int](idx: Int):
        var w_val = weights.load[w](idx)
        var g_val = grad.load[w](idx)
        weights.store[w](idx, w_val - lr * g_val)

    vectorize[width, fused_update](weights.size())

```

### Benefits

- No temporary allocations
- Reduced memory bandwidth
- Better cache utilization
- Faster execution

#### Buffer Reuse

```mojo

struct EfficientConv2D:
    """Conv2D with preallocated buffers."""
    var weight: Tensor
    var im2col_buffer: Tensor  # Reused across forward passes

    fn __init__(inout self, in_channels: Int, out_channels: Int, kernel_size: Int):
        self.weight = Tensor.randn(out_channels, in_channels, kernel_size, kernel_size)
        # Preallocate maximum buffer size
        var max_spatial = 1024  # Maximum spatial dimensions
        self.im2col_buffer = Tensor.zeros(
            max_spatial,
            in_channels * kernel_size * kernel_size
        )

    fn forward(inout self, borrowed input: Tensor) -> Tensor:
        """Forward pass reusing buffer."""
        # Reuse im2col_buffer instead of allocating new memory
        im2col_inplace(input, self.im2col_buffer)
        return self.im2col_buffer @ self.weight.reshape(-1)

```

### Benefits

- Eliminates per-forward-pass allocations
- Predictable memory usage
- Reduced memory fragmentation
- Faster execution in training loops

### Type Safety for ML Code

#### Compile-Time Shape Validation

```mojo

struct FixedShape[rows: Int, cols: Int]:
    """Tensor with compile-time known shape."""
    var data: DTypePointer[DType.float32]

    fn __init__(inout self):
        self.data = DTypePointer[DType.float32].alloc(rows * cols)

    @staticmethod
    fn matmul[K: Int](
        a: FixedShape[rows, K],
        b: FixedShape[K, cols]
    ) -> FixedShape[rows, cols]:
        """Matrix multiply with compile-time shape checking."""
        var result = FixedShape[rows, cols]()
        # Shape compatibility enforced by type system
        # ...matrix multiplication logic...
        return result

```

### Benefits

- Shape errors caught at compile time
- No runtime overhead for shape checks
- Self-documenting code
- Prevents entire class of bugs

#### Type-Safe Gradient Computation

```mojo

struct Variable[dtype: DType]:
    """Tensor with attached gradient."""
    var value: Tensor[dtype]
    var grad: Optional[Tensor[dtype]]
    var requires_grad: Bool

    fn backward(inout self):
        """Compute gradients."""
        if not self.requires_grad:
            return
        # Type system ensures gradient has same dtype as value
        self.grad = Some(Tensor[dtype].ones_like(self.value))

```

### Benefits

- Gradient dtype matches value dtype
- No runtime type conversions
- Clear gradient flow tracking
- Prevents numeric precision bugs

## Common Patterns

### Pattern 1: Layer Implementation

```mojo

struct Linear:
    """Fully connected linear layer."""
    var weight: Tensor[DType.float32]
    var bias: Tensor[DType.float32]
    var requires_grad: Bool

    fn __init__(inout self, input_size: Int, output_size: Int):
        """Initialize with He initialization."""
        self.weight = Tensor[DType.float32].randn(output_size, input_size)
        self.weight *= sqrt(2.0 / input_size)  # He initialization
        self.bias = Tensor[DType.float32].zeros(output_size)
        self.requires_grad = True

    fn forward(self, borrowed input: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Forward pass: output = input @ weight^T + bias."""
        return input @ self.weight.T + self.bias

    fn parameters(self) -> List[Tensor[DType.float32]]:
        """Return trainable parameters."""
        return [self.weight, self.bias]

```

### Pattern 2: Optimizer Implementation

```mojo

struct SGD:
    """Stochastic Gradient Descent optimizer."""
    var lr: Float64
    var momentum: Float64
    var velocities: List[Tensor[DType.float32]]

    fn __init__(inout self, lr: Float64 = 0.01, momentum: Float64 = 0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocities = List[Tensor[DType.float32]]()

    fn step(inout self, inout parameters: List[Tensor[DType.float32]]):
        """Update parameters using gradients."""
        # Initialize velocities on first call
        if len(self.velocities) == 0:
            for i in range(len(parameters)):
                self.velocities.append(Tensor[DType.float32].zeros_like(parameters[i]))

        # Update each parameter
        for i in range(len(parameters)):
            # Momentum update
            self.velocities[i] = (
                self.momentum * self.velocities[i] +
                parameters[i].grad
            )
            # Parameter update
            parameters[i] -= self.lr * self.velocities[i]

    fn zero_grad(self, inout parameters: List[Tensor[DType.float32]]):
        """Zero all gradients."""
        for i in range(len(parameters)):
            parameters[i].grad.zero_()

```

### Pattern 3: Training Loop

```mojo

fn train_epoch(
    inout model: Model,
    inout optimizer: SGD,
    borrowed train_data: DataLoader,
    loss_fn: fn(Tensor, Tensor) -> Tensor
) raises -> Float64:
    """Train for one epoch."""
    var total_loss: Float64 = 0.0
    var num_batches: Int = 0

    # Iterate over batches
    for batch in train_data:
        # Forward pass
        var predictions = model.forward(batch.data)
        var loss = loss_fn(predictions, batch.targets)

        # Backward pass
        loss.backward()

        # Optimizer step
        var params = model.parameters()
        optimizer.step(params)
        optimizer.zero_grad(params)

        # Track loss
        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches

```

### Pattern 4: Trait-Based Abstraction

```mojo

trait Module:
    """Base trait for neural network modules."""

    fn forward(inout self, borrowed input: Tensor) -> Tensor:
        """Forward pass."""
        ...

    fn parameters(inout self) -> List[Tensor]:
        """Get trainable parameters."""
        ...

trait Optimizer:
    """Base trait for optimizers."""

    fn step(self, inout parameters: List[Tensor]):
        """Update parameters."""
        ...

    fn zero_grad(self, inout parameters: List[Tensor]):
        """Zero gradients."""
        for i in range(len(parameters)):
            parameters[i].grad.zero_()

# Implementations
struct Linear(Module):
    """Linear layer implementing Module trait."""
    # ... implementation ...

struct Adam(Optimizer):
    """Adam optimizer implementing Optimizer trait."""
    # ... implementation ...

```

### Benefits

- Polymorphic behavior
- Consistent interfaces
- Composable abstractions
- Clear contracts

## Anti-Patterns

### Anti-Pattern 1: Unnecessary Allocations

```mojo

# BAD: Creates many temporary tensors
fn bad_forward(x: Tensor, w1: Tensor, w2: Tensor, w3: Tensor) -> Tensor:
    var h1 = relu(x @ w1)           # Allocation for x @ w1
    var h2 = relu(h1 @ w2)          # Allocation for h1 @ w2
    var output = sigmoid(h2 @ w3)   # Allocation for h2 @ w3
    return output

# GOOD: Reuse buffers when possible
fn good_forward(
    borrowed x: Tensor,
    borrowed w1: Tensor,
    borrowed w2: Tensor,
    borrowed w3: Tensor,
    inout buffer1: Tensor,
    inout buffer2: Tensor
) -> Tensor:
    matmul_into(x, w1, buffer1)     # Write into buffer1
    relu_inplace(buffer1)           # In-place activation
    matmul_into(buffer1, w2, buffer2)
    relu_inplace(buffer2)
    matmul_into(buffer2, w3, buffer1)
    sigmoid_inplace(buffer1)
    return buffer1

```

### Anti-Pattern 2: Missing Ownership Annotations

```mojo

# BAD: Unclear ownership semantics
fn unclear_function(x: Tensor) -> Tensor:
    # Does this modify x? Take ownership?
    return process(x)

# GOOD: Explicit ownership
fn clear_function(borrowed x: Tensor) -> Tensor:
    # Clearly borrows x (read-only)
    return process(x)

fn clear_inplace(inout x: Tensor):
    # Clearly modifies x in place
    process_inplace(x)

fn clear_consume(owned x: Tensor) -> Tensor:
    # Clearly takes ownership and consumes x
    return transform(x)

```

### Anti-Pattern 3: Ignoring SIMD Opportunities

```mojo

# BAD: Scalar operations
fn bad_relu(inout tensor: Tensor):
    for i in range(tensor.size()):
        if tensor[i] < 0:
            tensor[i] = 0

# GOOD: SIMD vectorization
fn good_relu(inout tensor: Tensor):
    alias width = simdwidthof[DType.float32]()

    @parameter
    fn vectorized[w: Int](idx: Int):
        var val = tensor.load[w](idx)
        tensor.store[w](idx, max(val, 0.0))

    vectorize[width, vectorized](tensor.size())

```

### Anti-Pattern 4: Using `class` When `struct` Suffices

```mojo

# BAD: Unnecessary heap allocation and indirection
class BadLayer:
    var weight: Tensor
    var bias: Tensor

    # Reference semantics add overhead

# GOOD: Value type with stack allocation
struct GoodLayer:
    var weight: Tensor
    var bias: Tensor

    # Value semantics are more efficient

```

### Anti-Pattern 5: Missing Type Annotations with `fn`

```mojo

# BAD: fn requires explicit types
fn bad_function(x, y):  # Compile error!
    return x + y

# GOOD: Explicit type annotations
fn good_function(x: Float32, y: Float32) -> Float32:
    return x + y

# ALSO GOOD: Generic with parameter
fn generic_function[dtype: DType](x: SIMD[dtype, 1], y: SIMD[dtype, 1]) -> SIMD[dtype, 1]:
    return x + y

```

### Anti-Pattern 6: Not Leveraging Compile-Time Features

```mojo

# BAD: Runtime computation
fn bad_alloc(size: Int) -> Tensor:
    var tensor = Tensor(size, size)  # Runtime size
    # Cannot optimize well
    return tensor

# GOOD: Compile-time size when possible
fn good_alloc[size: Int]() -> FixedTensor[size, size]:
    var tensor = FixedTensor[size, size]()
    # Compiler can optimize aggressively
    return tensor

```

## Examples

Complete examples demonstrating these patterns are available in the `examples/` directory:

- `examples/mojo-patterns/ownership_example.mojo` - Ownership and borrowing patterns
- `examples/mojo-patterns/simd_example.mojo` - SIMD vectorization
- `examples/mojo-patterns/trait_example.mojo` - Trait-based design
- `examples/performance/simd_optimization.mojo` - Advanced SIMD techniques
- `examples/performance/memory_optimization.mojo` - Memory-efficient patterns
- `examples/getting-started/first_model_model.mojo` - Complete neural network example

Run any example with:

```bash

pixi run mojo run examples/path/to/example.mojo

```

## Best Practices Summary

1. **Prefer `fn` over `def`** - Use `fn` for ML implementations, `def` only when Python interop is needed
1. **Use `struct` for ML components** - Layers, optimizers, and most ML objects should be `struct`
1. **Be explicit with ownership** - Always use `borrowed`, `inout`, or `owned` annotations
1. **Leverage SIMD** - Vectorize element-wise operations and reductions
1. **Minimize allocations** - Use in-place operations and buffer reuse
1. **Use compile-time features** - Parametric types and aliases for optimization
1. **Type everything** - Explicit types enable better optimization and error detection
1. **Compose with traits** - Define interfaces with traits for reusable abstractions

## Additional Resources

- [Mojo Language Documentation](https://docs.modular.com/mojo/)
- [ML Odyssey Architecture](../dev/architecture.md)
- [Performance Optimization Guide](../advanced/performance.md)

## Contributing

When adding new patterns:

1. Include concrete code examples
1. Explain the "why" behind the pattern
1. Show anti-patterns to avoid
1. Provide performance comparisons when relevant
1. Link to example implementations in the codebase
