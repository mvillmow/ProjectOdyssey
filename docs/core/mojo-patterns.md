# Mojo Patterns for Machine Learning

Mojo-specific patterns, idioms, and best practices for implementing neural networks and ML algorithms.

## Overview

Mojo is designed for high-performance AI/ML workloads, combining Python-like syntax with systems programming
capabilities. This guide covers Mojo-specific patterns that make ML implementations fast, safe, and maintainable.

## Language Fundamentals

### Functions: `fn` vs `def`

Use `fn` for performance-critical code, `def` for flexibility:

```mojo
# Use fn for neural network operations (type-safe, optimized)
fn forward(borrowed input: Tensor, borrowed weights: Tensor) -> Tensor:
    """Forward pass through layer (strict, fast)."""
    return input @ weights

# Use def for utilities and scripts (flexible)
def load_config(filename: String):
    """Load configuration (flexible types)."""
    var config = read_file(filename)
    return parse_toml(config)
```

**When to use `fn`:**

- Neural network forward/backward passes
- Tensor operations
- Performance-critical loops
- Type-safe APIs

**When to use `def`:**

- Scripting and automation
- Configuration loading
- Quick prototyping
- Python interop

### Structs vs Classes

Use structs for ML components:

```mojo
# Struct for neural network layers (value semantics, optimized)
struct Linear:
    """Fully connected layer with compile-time optimization."""
    var weight: Tensor
    var bias: Tensor

    fn __init__(inout self, input_size: Int, output_size: Int):
        self.weight = Tensor.randn(output_size, input_size)
        self.bias = Tensor.zeros(output_size)

    fn forward(borrowed self, borrowed input: Tensor) -> Tensor:
        return input @ self.weight.T + self.bias
```

**Prefer structs for:**

- Neural network layers
- Tensor types
- Optimizers
- Data structures
- Performance-critical code

**Use classes only for:**

- Dynamic behavior requiring inheritance
- Runtime polymorphism
- Python compatibility

## Memory Management

### Ownership and Borrowing

Mojo's ownership system prevents memory errors.

See `examples/mojo-patterns/ownership_example.mojo`](

Key patterns:

```mojo
# Borrowed: read-only access (no ownership transfer)
fn compute_loss(borrowed predictions: Tensor, borrowed targets: Tensor) -> Float64:
    var diff = predictions - targets
    return (diff * diff).mean()

# Inout: mutable reference (modify in place)
fn update_weights(inout weights: Tensor, borrowed gradients: Tensor, lr: Float64):
    weights -= lr * gradients  # Modifies original
```

**Best practices:**

- Use `borrowed` for read-only access (most common)
- Use `inout` for in-place updates
- Use `owned` only when consuming resources
- Avoid unnecessary copies

Full example: `examples/mojo-patterns/ownership_example.mojo`

### In-Place Operations

Reduce memory allocations with in-place operations:

```mojo
struct SGD:
    """Stochastic gradient descent optimizer."""
    var lr: Float64

    fn step(self, inout parameters: List[Tensor]):
        """Update parameters in place (memory efficient)."""
        for i in range(len(parameters)):
            # In-place update (no allocation)
            parameters[i] -= self.lr * parameters[i].grad

# Bad: Creates unnecessary copies
fn update_bad(weights: Tensor, grad: Tensor, lr: Float64) -> Tensor:
    return weights - lr * grad  # Allocates new tensor

# Good: In-place update
fn update_good(inout weights: Tensor, borrowed grad: Tensor, lr: Float64):
    weights -= lr * grad  # Modifies in place
```

## SIMD Optimization

### Vectorized Operations

Leverage SIMD for parallel computation.

See `examples/mojo-patterns/simd_example.mojo` for a complete working example.

Key pattern:

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

Full example: `examples/mojo-patterns/simd_example.mojo`

### Parallel Loops

Use `parallelize` for data-parallel operations:

```mojo
from algorithm import parallelize

fn batch_forward(borrowed inputs: Tensor, borrowed weights: Tensor) -> Tensor:
    """Process batch in parallel."""
    var outputs = Tensor.zeros(inputs.shape[0], weights.shape[1])

    @parameter
    fn process_sample(i: Int):
        # Each batch sample processed in parallel
        var input = inputs[i]
        outputs[i] = input @ weights

    parallelize[process_sample](inputs.shape[0])

    return outputs
```

## Trait-Based Design

### Defining Traits

Create reusable interfaces with traits.

See `examples/mojo-patterns/trait_example.mojo`](

Key pattern:

```mojo
trait Module:
    """Base trait for neural network modules."""
    fn forward(inout self, borrowed input: Tensor) -> Tensor: ...
    fn parameters(inout self) -> List[Tensor]: ...

struct Linear(Module):
    """Linear layer implementing Module trait."""
    fn forward(inout self, borrowed input: Tensor) -> Tensor:
        return input @ self.weight.T + self.bias
```

Full example: `examples/mojo-patterns/trait_example.mojo`

### Generic Functions

Write generic code that works with any trait implementation:

```mojo
fn train_step[M: Module, O: Optimizer](
    inout model: M,
    inout optimizer: O,
    borrowed inputs: Tensor,
    borrowed targets: Tensor,
) -> Float64:
    """Generic training step for any model and optimizer."""

    # Forward pass
    var outputs = model.forward(inputs)

    # Compute loss
    var loss = mse_loss(outputs, targets)

    # Backward pass
    loss.backward()

    # Update parameters
    var params = model.parameters()
    optimizer.step(params)
    optimizer.zero_grad(params)

    return loss.item()
```

## Type System

### Static Typing

Leverage compile-time type checking:

```mojo
from tensor import Tensor, DType

# Specify types for safety
fn linear_forward(
    borrowed input: Tensor[DType.float32],
    borrowed weight: Tensor[DType.float32],
    borrowed bias: Tensor[DType.float32]
) -> Tensor[DType.float32]:
    """Type-safe linear transformation."""
    return input @ weight.T + bias

# Compile-time error if types don't match
var input = Tensor[DType.float32].randn(10, 784)
var weight = Tensor[DType.float64].randn(128, 784)  # Wrong dtype!
var output = linear_forward(input, weight, bias)  # Compile error!
```

### Type Aliases

Create readable type aliases:

```mojo
alias Float32Tensor = Tensor[DType.float32]
alias IntTensor = Tensor[DType.int64]

fn cross_entropy_loss(
    borrowed predictions: Float32Tensor,
    borrowed targets: IntTensor
) -> Float64:
    """Cross entropy loss with clear types."""
    # Implementation...
    return loss
```

## Parameter Types

### Compile-Time Parameters

Use `@parameter` for compile-time constants:

```mojo
fn create_layer[@parameter activation: String](
    input_size: Int,
    output_size: Int
) -> Sequential:
    """Create layer with compile-time activation choice."""

    @parameter
    if activation == "relu":
        return Sequential([
            Linear(input_size, output_size),
            ReLU()
        ])
    elif activation == "tanh":
        return Sequential([
            Linear(input_size, output_size),
            Tanh()
        ])
    else:
        compile_error("Unknown activation")

# Optimized at compile time
var layer1 = create_layer["relu"](784, 128)
var layer2 = create_layer["tanh"](128, 10)
```

### Variadic Parameters

Accept variable number of arguments:

```mojo
fn sequential(*layers: Module) -> Sequential:
    """Create sequential model from variable number of layers."""
    var model = Sequential()
    for layer in layers:
        model.add(layer)
    return model

# Use with any number of layers
var model = sequential(
    Linear(784, 256),
    ReLU(),
    Linear(256, 128),
    ReLU(),
    Linear(128, 10)
)
```

## Error Handling

### Raising Exceptions

```mojo
fn validate_shape(borrowed tensor: Tensor, expected: List[Int]) raises:
    """Validate tensor shape, raise if mismatch."""
    if tensor.ndim() != len(expected):
        raise Error("Shape mismatch: expected " + str(len(expected)) +
                    " dimensions, got " + str(tensor.ndim()))

    for i in range(len(expected)):
        if tensor.shape[i] != expected[i]:
            raise Error("Shape mismatch at dimension " + str(i))

fn forward(inout self, borrowed input: Tensor) raises -> Tensor:
    """Forward pass with shape validation."""
    validate_shape(input, [self.batch_size, self.input_size])
    return input @ self.weight.T + self.bias
```

### Result Types

Use Result for recoverable errors:

```mojo
from utils import Result, Ok, Err

fn load_checkpoint(path: String) -> Result[Tensor, String]:
    """Load model checkpoint, return Result."""
    if not file_exists(path):
        return Err("File not found: " + path)

    var data = read_file(path)
    if not validate_checkpoint(data):
        return Err("Invalid checkpoint format")

    return Ok(parse_tensor(data))

# Handle result
var result = load_checkpoint("model.mojo")
if result.is_ok():
    var tensor = result.unwrap()
    print("Loaded successfully")
else:
    print("Error:", result.unwrap_err())
```

## Common ML Patterns

### Pattern 1: Layer with Parameters

```mojo
struct ConvLayer(Module):
    """Convolutional layer with learnable parameters."""
    var weight: Tensor
    var bias: Tensor
    var stride: Int
    var padding: Int

    fn __init__(
        inout self,
        in_channels: Int,
        out_channels: Int,
        kernel_size: Int,
        stride: Int = 1,
        padding: Int = 0
    ):
        # Xavier initialization
        var n = in_channels * kernel_size * kernel_size
        var std = (2.0 / n) ** 0.5

        self.weight = Tensor.randn(
            out_channels, in_channels, kernel_size, kernel_size
        ) * std
        self.bias = Tensor.zeros(out_channels)
        self.stride = stride
        self.padding = padding

    fn forward(inout self, borrowed input: Tensor) -> Tensor:
        return conv2d(input, self.weight, self.bias, self.stride, self.padding)

    fn parameters(inout self) -> List[Tensor]:
        return [self.weight, self.bias]
```

### Pattern 2: Custom Autograd Function

```mojo
struct ReLUBackward:
    """Custom backward pass for ReLU."""
    var input: Tensor

    fn backward(self, borrowed grad_output: Tensor) -> Tensor:
        # ReLU gradient: pass through if input > 0, else 0
        var mask = self.input > 0.0
        return grad_output * mask

fn relu_forward(borrowed input: Tensor) -> (Tensor, ReLUBackward):
    """ReLU forward with custom backward."""
    var output = max(input, 0.0)
    var backward_fn = ReLUBackward(input)
    return (output, backward_fn)
```

### Pattern 3: Model Checkpoint

```mojo
struct ModelCheckpoint:
    """Save/load model state."""

    @staticmethod
    fn save[M: Module](model: M, path: String) raises:
        """Save model parameters to file."""
        var params = model.parameters()
        var state_dict = Dict[String, Tensor]()

        for i in range(len(params)):
            state_dict["param_" + str(i)] = params[i]

        write_checkpoint(path, state_dict)

    @staticmethod
    fn load[M: Module](inout model: M, path: String) raises:
        """Load model parameters from file."""
        var state_dict = read_checkpoint(path)
        var params = model.parameters()

        for i in range(len(params)):
            params[i] = state_dict["param_" + str(i)]
```

### Pattern 4: Training Loop

```mojo
fn train_epoch[M: Module, O: Optimizer](
    inout model: M,
    inout optimizer: O,
    borrowed train_loader: BatchLoader,
    loss_fn: LossFunction
) -> Float64:
    """Train for one epoch (generic over model and optimizer)."""

    var total_loss: Float64 = 0.0
    var num_batches: Int = 0

    for batch in train_loader:
        var inputs, targets = batch

        # Forward
        var outputs = model.forward(inputs)
        var loss = loss_fn(outputs, targets)

        # Backward
        optimizer.zero_grad(model.parameters())
        loss.backward()
        optimizer.step(model.parameters())

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches
```

## Performance Tips

### 1. Avoid Unnecessary Allocations

```mojo
# Bad: Creates temporary tensor
fn bad_add(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
    var temp = a + b  # Allocation
    return temp + c   # Another allocation

# Good: Fused operation
fn good_add(borrowed a: Tensor, borrowed b: Tensor, borrowed c: Tensor) -> Tensor:
    return a + b + c  # Single allocation
```

### 2. Use SIMD Width

```mojo
from sys.info import simdwidthof

fn process_tensor(inout data: Tensor):
    """Process with optimal SIMD width."""
    alias width = simdwidthof[DType.float32]()

    @parameter
    fn vectorized[w: Int](i: Int):
        var vec = data.load[w](i)
        vec = vec * 2.0  # SIMD multiplication
        data.store[w](i, vec)

    vectorize[width, vectorized](data.size())
```

### 3. Preallocate Buffers

```mojo
struct BatchProcessor:
    """Preallocate buffers for batch processing."""
    var batch_size: Int
    var buffer: Tensor  # Reused across batches

    fn __init__(inout self, batch_size: Int, feature_size: Int):
        self.batch_size = batch_size
        self.buffer = Tensor.zeros(batch_size, feature_size)

    fn process(inout self, borrowed batch: Tensor) -> Tensor:
        # Reuse buffer (no allocation)
        self.buffer.copy_from(batch)
        self.buffer *= 2.0
        return self.buffer
```

### 4. Compile-Time Specialization

```mojo
fn create_model[@parameter hidden_size: Int]() -> Sequential:
    """Create model with compile-time hidden size."""
    return Sequential([
        Linear(784, hidden_size),  # Size known at compile time
        ReLU(),
        Linear(hidden_size, 10)
    ])

# Generates optimized code for each size
var small_model = create_model[128]()
var large_model = create_model[512]()
```

## Next Steps

- **[Performance Guide](../advanced/performance.md)** - Deep dive into optimization
- **[Custom Layers](../advanced/custom-layers.md)** - Build custom components
- **[Shared Library](shared-library.md)** - Use pre-built Mojo components
- **[Paper Implementation](paper-implementation.md)** - Apply patterns to real papers

## Related Documentation

- [Mojo Language Documentation](https://docs.modular.com/mojo/) - Official Mojo docs
- [Testing Strategy](testing-strategy.md) - Testing Mojo code
- [Project Structure](project-structure.md) - Mojo code organization
