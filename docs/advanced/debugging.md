# Debugging Guide

Debugging Mojo code, neural networks, and training issues in ML Odyssey.

## Overview

Debugging machine learning code involves challenges beyond traditional software debugging: numerical stability,
gradient issues, and training dynamics. This guide covers tools and techniques for debugging ML Odyssey implementations.

## Common Issues

### Training Issues

**Symptoms**:

- Loss not decreasing
- Loss exploding (NaN)
- Training too slow
- Overfitting quickly

**Solutions**: See [Training Debugging](#training-debugging)

### Gradient Issues

**Symptoms**:

- Gradients are zero
- Gradients exploding
- Gradients vanishing

**Solutions**: See [Gradient Debugging](#gradient-debugging)

### Shape Errors

**Symptoms**:

- Tensor shape mismatch
- Dimension errors
- Broadcasting errors

**Solutions**: See [Shape Debugging](#shape-debugging)

## Print Debugging

### Basic Logging

Add strategic print statements:

```mojo
fn forward(inout self, borrowed input: Tensor) -> Tensor:
    """Forward pass with debug logging."""

    print("Input shape:", input.shape)
    print("Input range: [", input.min(), ",", input.max(), "]")

    var x = self.conv1.forward(input)
    print("After conv1:", x.shape, "range: [", x.min(), ",", x.max(), "]")

    x = relu(x)
    print("After ReLU:", x.shape, "range: [", x.min(), ",", x.max(), "]")

    return x
```

### Conditional Logging

Log only when needed:

```mojo
fn forward(inout self, borrowed input: Tensor, debug: Bool = False) -> Tensor:
    """Forward pass with optional debugging."""

    if debug:
        print("=== Forward Pass Debug ===")
        print("Input:", input.shape)

    var x = self.layer1.forward(input)

    if debug:
        print("After layer1:", x.shape)
        check_for_nans(x, "layer1 output")

    return x

fn check_for_nans(borrowed tensor: Tensor, name: String):
    """Check tensor for NaN values."""
    if tensor.isnan().any():
        print("WARNING: NaN detected in", name)
        print("  Shape:", tensor.shape)
        print("  First NaN at index:", tensor.isnan().argmax())
```

## Training Debugging

### Loss Not Decreasing

#### Check 1: Verify gradients are flowing

```mojo
fn debug_gradients(model: Model):
    """Check if gradients are being computed."""

    for name, param in model.named_parameters():
        if param.grad is None:
            print("ERROR: No gradient for", name)
        elif param.grad.abs().sum() == 0:
            print("WARNING: Zero gradient for", name)
        else:
            var grad_norm = param.grad.norm()
            print(name, "gradient norm:", grad_norm)
```

#### Check 2: Learning rate too low or high

```mojo
fn debug_learning_rate(optimizer: Optimizer):
    """Try different learning rates."""

    var lrs = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

    for lr in lrs:
        print("\nTesting LR =", lr)
        optimizer.set_lr(lr)

        # Train for a few batches
        var losses = []
        for i, batch in enumerate(train_loader):
            if i >= 10:
                break

            var loss = train_step(model, batch, optimizer)
            losses.append(loss)

        print("  Initial loss:", losses[0])
        print("  Final loss:", losses[-1])
        print("  Trend:", "decreasing" if losses[-1] < losses[0] else "increasing")
```

#### Check 3: Data issues

```mojo
fn debug_data(data_loader: BatchLoader):
    """Verify data is correct."""

    var batch = data_loader.next()
    var inputs, targets = batch

    print("Batch size:", inputs.shape[0])
    print("Input shape:", inputs.shape)
    print("Input range: [", inputs.min(), ",", inputs.max(), "]")
    print("Target shape:", targets.shape)
    print("Target values:", targets.unique())

    # Check for NaN
    if inputs.isnan().any():
        print("ERROR: NaN in inputs!")

    # Check normalization
    print("Input mean:", inputs.mean())
    print("Input std:", inputs.std())
```

### Loss Exploding (NaN)

#### Check 1: Gradient clipping

```mojo
fn train_with_grad_clipping(
    model: Model,
    optimizer: Optimizer,
    max_norm: Float64 = 1.0
):
    """Train with gradient clipping."""

    for batch in train_loader:
        var loss = forward_and_loss(model, batch)
        loss.backward()

        # Clip gradients
        var total_norm = clip_grad_norm(model.parameters(), max_norm)

        if total_norm > max_norm:
            print("WARNING: Gradients clipped (norm was", total_norm, ")")

        optimizer.step()
        optimizer.zero_grad()
```

#### Check 2: Numerical stability

```mojo
fn add_numerical_stability(inout tensor: Tensor, epsilon: Float64 = 1e-7):
    """Add small value to prevent division by zero."""

    # Bad: Can cause NaN
    var bad_result = 1.0 / tensor

    # Good: Numerically stable
    var good_result = 1.0 / (tensor + epsilon)

    return good_result
```

#### Check 3: Check for inf/nan after each layer

```mojo
fn forward_with_checks(inout self, borrowed input: Tensor) -> Tensor:
    """Forward pass with NaN/Inf checking."""

    var x = input

    for i, layer in enumerate(self.layers):
        x = layer.forward(x)

        # Check for NaN or Inf
        if x.isnan().any():
            raise Error(f"NaN detected after layer {i}")

        if x.isinf().any():
            raise Error(f"Inf detected after layer {i}")

    return x
```

## Gradient Debugging

### Vanishing Gradients

**Symptom**: Gradients become very small in early layers

#### Check 1: Gradient magnitudes

```mojo
fn check_gradient_magnitudes(model: Model):
    """Check gradient magnitudes across layers."""

    print("Gradient Magnitudes:")
    print("-" * 50)

    for name, param in model.named_parameters():
        if param.grad is not None:
            var grad_norm = param.grad.norm()
            var grad_mean = param.grad.abs().mean()
            var grad_max = param.grad.abs().max()

            print(f"{name:20s} | norm: {grad_norm:.6f} | mean: {grad_mean:.6f} | max: {grad_max:.6f}")

            if grad_norm < 1e-7:
                print(f"  WARNING: Vanishing gradient!")
```

**Solution**: Use different activation or initialization

```mojo
# Bad: tanh can cause vanishing gradients
var bad_model = Sequential([
    Linear(784, 256),
    Tanh(),  # Gradients < 1, compound through layers
    Linear(256, 128),
    Tanh(),
    Linear(128, 10),
])

# Good: ReLU preserves gradients better
var good_model = Sequential([
    Linear(784, 256),
    ReLU(),  # Gradients 0 or 1
    Linear(256, 128),
    ReLU(),
    Linear(128, 10),
])
```

### Exploding Gradients

**Symptom**: Gradients become very large

#### Solution 1: Gradient clipping

```mojo
fn clip_grad_norm(parameters: List[Tensor], max_norm: Float64) -> Float64:
    """Clip gradients by norm."""

    # Calculate total norm
    var total_norm: Float64 = 0.0
    for param in parameters:
        if param.grad is not None:
            total_norm += param.grad.norm() ** 2

    total_norm = sqrt(total_norm)

    # Clip if needed
    if total_norm > max_norm:
        var clip_coef = max_norm / (total_norm + 1e-6)
        for param in parameters:
            if param.grad is not None:
                param.grad *= clip_coef

    return total_norm
```

#### Solution 2: Better weight initialization

```mojo
fn kaiming_init(inout tensor: Tensor, mode: String = "fan_in"):
    """Kaiming initialization (good for ReLU)."""

    var fan_in = tensor.shape[1]
    var fan_out = tensor.shape[0]

    var fan = fan_in if mode == "fan_in" else fan_out
    var std = sqrt(2.0 / fan)

    tensor = Tensor.randn(tensor.shape) * std
```

## Shape Debugging

### Tensor Shape Mismatches

#### Add shape assertions

```mojo
fn assert_shape(borrowed tensor: Tensor, expected: List[Int], name: String = "tensor"):
    """Assert tensor has expected shape."""

    if tensor.ndim() != len(expected):
        raise Error(f"{name} has {tensor.ndim()} dimensions, expected {len(expected)}")

    for i in range(len(expected)):
        if expected[i] != -1 and tensor.shape[i] != expected[i]:
            raise Error(
                f"{name} dimension {i} is {tensor.shape[i]}, expected {expected[i]}\n" +
                f"Full shape: {tensor.shape} vs expected {expected}"
            )

fn forward(inout self, borrowed input: Tensor) -> Tensor:
    """Forward with shape validation."""

    assert_shape(input, [self.batch_size, 1, 28, 28], "input")

    var x = self.conv1.forward(input)
    assert_shape(x, [self.batch_size, 6, 28, 28], "conv1 output")

    return x
```

## Performance Debugging

### Profiling

Find performance bottlenecks:

```mojo
from shared.utils import Profiler

fn profile_model(model: Model, input: Tensor):
    """Profile model forward pass."""

    var profiler = Profiler()

    # Profile each layer
    var x = input

    for i, layer in enumerate(model.layers):
        with profiler.section(f"layer_{i}"):
            x = layer.forward(x)

    profiler.print_summary()
```

### Memory Debugging

Track memory allocations:

```mojo
from shared.utils import MemoryTracker

fn debug_memory(model: Model):
    """Track memory usage."""

    var tracker = MemoryTracker()

    tracker.start()

    # Create model
    print("After model creation:", tracker.current_usage(), "MB")

    # Load data
    var data = load_dataset()
    print("After data loading:", tracker.current_usage(), "MB")

    # Training step
    var loss = train_step(model, data)
    print("After training step:", tracker.current_usage(), "MB")

    # Check for leaks
    del data
    del loss
    print("After cleanup:", tracker.current_usage(), "MB")
```

## Unit Test Debugging

### Isolate Failures

Create minimal test:

```mojo
fn test_minimal_linear():
    """Minimal test to isolate issue."""

    # Simplest possible case
    var layer = Linear(2, 2)
    var input = Tensor([[1.0, 2.0]])

    print("Weight:", layer.weight)
    print("Bias:", layer.bias)

    var output = layer.forward(input)

    print("Input:", input)
    print("Output:", output)
    print("Expected:", input @ layer.weight.T + layer.bias)

    # Should match manual calculation
```

### Compare with Reference

Test against known-good implementation:

```mojo
fn test_against_pytorch():
    """Compare with PyTorch implementation."""

    var mojo_layer = Linear(10, 5)
    var input = Tensor.randn(8, 10)

    # Copy weights to PyTorch
    var torch_layer = torch.nn.Linear(10, 5)
    torch_layer.weight.data = to_torch(mojo_layer.weight)
    torch_layer.bias.data = to_torch(mojo_layer.bias)

    # Forward pass
    var mojo_output = mojo_layer.forward(input)
    var torch_output = torch_layer(to_torch(input))

    # Compare
    var diff = (mojo_output - from_torch(torch_output)).abs().max()
    print("Max difference:", diff)

    assert diff < 1e-5, f"Outputs differ by {diff}"
```

## Debugging Tools

### Breakpoints

Use debugger:

```mojo
fn debug_forward(inout self, borrowed input: Tensor) -> Tensor:
    """Forward with breakpoint."""

    var x = self.conv1.forward(input)

    # Set breakpoint
    breakpoint()  # Execution pauses here

    x = self.conv2.forward(x)

    return x
```

### Assertions

Validate assumptions:

```mojo
fn forward(inout self, borrowed input: Tensor) -> Tensor:
    """Forward with assertions."""

    # Validate input
    assert input.shape[0] > 0, "Batch size must be > 0"
    assert not input.isnan().any(), "Input contains NaN"

    var x = self.layer1.forward(input)

    # Validate intermediate
    assert x.shape[1] == 64, "Unexpected feature dimension"
    assert (x >= 0).all(), "ReLU should produce non-negative values"

    return x
```

## Common Debugging Workflows

### Workflow 1: New Model Not Learning

1. **Check data**: Visualize inputs and labels
2. **Check gradients**: Ensure they flow to all layers
3. **Try overfitting**: Can model overfit small dataset?
4. **Reduce complexity**: Simplify model to isolate issue
5. **Check learning rate**: Try range of values

### Workflow 2: Model Works But Slow

1. **Profile**: Identify bottlenecks
2. **Check SIMD**: Are operations vectorized?
3. **Check memory**: Are there unnecessary allocations?
4. **Check I/O**: Is data loading the bottleneck?

### Workflow 3: Test Failing

1. **Minimal example**: Reduce to simplest failing case
2. **Manual calculation**: Compute expected result by hand
3. **Print intermediates**: Log all intermediate values
4. **Compare reference**: Test against known implementation

## Best Practices

### DO

- ✅ Add debug mode to models
- ✅ Validate shapes at boundaries
- ✅ Check for NaN/Inf regularly
- ✅ Log gradient statistics
- ✅ Use assertions liberally
- ✅ Profile before optimizing

### DON'T

- ❌ Debug without understanding the math
- ❌ Skip gradient checking
- ❌ Ignore warnings
- ❌ Debug in isolation (compare with references)
- ❌ Optimize before it works

## Next Steps

- **[Testing Strategy](../core/testing-strategy.md)** - Comprehensive testing
- **[Performance Guide](performance.md)** - Performance debugging
- **[Visualization](visualization.md)** - Visual debugging
- **[Custom Layers](custom-layers.md)** - Debug custom components

## Related Documentation

- [Mojo Patterns](../core/mojo-patterns.md) - Avoid common pitfalls
- [First Model Tutorial](../getting-started/first_model.md) - Troubleshooting section
- [Shared Library](../core/shared-library.md) - Debug utilities
