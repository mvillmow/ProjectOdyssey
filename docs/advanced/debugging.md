# Debugging ML Code

Comprehensive guide to debugging machine learning implementations in Mojo.

## Overview

Common ML bugs fall into three categories:

1. **Implementation Errors** - Incorrect logic, shape mismatches, type errors
2. **Numerical Issues** - NaN, infinity, gradient explosions/vanishing
3. **Performance Problems** - Slow training, memory leaks, inefficient code

## Debugging Strategies

### Systematic Approach

**Step 1: Isolate the Problem**

```mojo

# Test components individually
fn test_layer_isolation():
    var layer = Linear(10, 5)
    var input = Tensor.randn(2, 10)
    var output = layer.forward(input)
    print("Output shape:", output.shape)  # Should be (2, 5)
    print("Output range:", output.min(), "to", output.max())

```text

**Step 2: Verify Shapes**

```mojo

fn debug_shapes(model: Model, input: Tensor):
    """Print shapes at each layer."""
    print("Input shape:", input.shape)
    for i in range(len(model.layers)):
        var output = model.layers[i].forward(input)
        print(f"Layer {i} output shape:", output.shape)
        input = output

```text

**Step 3: Check for NaN/Inf**

```mojo

fn check_numerical_health(tensor: Tensor, name: String):
    """Verify tensor has valid values."""
    if has_nan(tensor):
        print(f"ERROR: {name} contains NaN!")
    if has_inf(tensor):
        print(f"ERROR: {name} contains Infinity!")
    print(f"{name} range: [{tensor.min()}, {tensor.max()}]")

```text

## Mojo Debugging Tools

### Print Debugging

```mojo

fn forward(self, x: Tensor) -> Tensor:
    print("Forward input shape:", x.shape)
    var h1 = self.layer1.forward(x)
    print("After layer1:", h1.shape, "mean:", h1.mean())
    var h2 = self.layer2.forward(h1)
    print("After layer2:", h2.shape, "mean:", h2.mean())
    return h2

```text

### Assertions

```mojo

fn forward(self, x: Tensor) -> Tensor:
    debug_assert(x.shape[1] == self.input_size, "Input size mismatch")
    var output = self.compute(x)
    debug_assert(not has_nan(output), "Output contains NaN")
    return output

```text

### Conditional Compilation

```mojo

@parameter
if DEBUG:
    print("Debug info:", x.shape, x.mean(), x.std())

```text

## Common Issues

### Issue 1: NaN in Loss

**Symptoms**: Loss becomes NaN after a few iterations

**Causes**:

- Learning rate too high
- Numerical instability in loss function
- Exploding gradients

**Solutions**:

```mojo

# 1. Clip gradients
fn clip_gradients(inout params: List[Tensor], max_norm: Float64 = 1.0):
    var total_norm: Float64 = 0.0
    for i in range(len(params)):
        total_norm += (params[i].grad * params[i].grad).sum()
    total_norm = sqrt(total_norm)

    if total_norm > max_norm:
        var scale = max_norm / total_norm
        for i in range(len(params)):
            params[i].grad *= scale

# 2. Use stable loss functions
fn stable_cross_entropy(predictions: Tensor, targets: Tensor) -> Float64:
    """Numerically stable cross-entropy."""
    var logits_max = predictions.max(dim=-1, keepdim=True)
    var shifted = predictions - logits_max
    var log_sum_exp = log((exp(shifted)).sum(dim=-1))
    return -(targets * shifted).sum(dim=-1).mean() + log_sum_exp.mean()

# 3. Reduce learning rate
var optimizer = SGD(lr=0.001)  # Start smaller

```text

### Issue 2: Gradient Explosion/Vanishing

**Symptoms**: Gradients become very large (>1e10) or very small (<1e-10)

**Diagnosis**:

```mojo

fn diagnose_gradients(model: Model):
    """Check gradient magnitudes."""
    for i in range(len(model.parameters())):
        var param = model.parameters()[i]
        var grad_norm = sqrt((param.grad * param.grad).sum())
        print(f"Parameter {i} gradient norm: {grad_norm}")

        if grad_norm > 100:
            print("WARNING: Gradient explosion!")
        elif grad_norm < 1e-6:
            print("WARNING: Gradient vanishing!")

```text

**Solutions**:

```mojo

# 1. Gradient clipping
clip_gradients(model.parameters(), max_norm=1.0)

# 2. Batch normalization
struct ModelWithBatchNorm:
    var conv1: Conv2D
    var bn1: BatchNorm
    var conv2: Conv2D
    var bn2: BatchNorm

# 3. Skip connections (ResNet-style)
fn forward_with_skip(self, x: Tensor) -> Tensor:
    var residual = x
    var out = self.layer1.forward(x)
    out = self.layer2.forward(out)
    return out + residual  # Skip connection

```text

### Issue 3: Shape Mismatches

**Symptoms**: Runtime errors about incompatible shapes

**Diagnosis**:

```mojo

fn trace_shapes(model: Model, input: Tensor):
    """Print all intermediate shapes."""
    var x = input
    print("Input:", x.shape)

    for i in range(len(model.layers)):
        try:
            x = model.layers[i].forward(x)
            print(f"After layer {i}:", x.shape)
        except e:
            print(f"ERROR at layer {i}: {e}")
            print(f"Expected input shape: {model.layers[i].expected_input_shape()}")
            raise e

```text

**Prevention**:

```mojo

struct Layer:
    fn forward(self, borrowed x: Tensor) -> Tensor:
        # Validate input shape
        if x.shape[1] != self.input_size:
            raise Error(f"Expected input size {self.input_size}, got {x.shape[1]}")

        var output = self.compute(x)

        # Validate output shape
        debug_assert(output.shape[1] == self.output_size)
        return output

```text

### Issue 4: Memory Leaks

**Symptoms**: Memory usage grows over time

**Diagnosis**:

```mojo

fn monitor_memory():
    """Track memory usage during training."""
    for epoch in range(num_epochs):
        var mem_before = get_memory_usage()

        train_epoch(model, train_data)

        var mem_after = get_memory_usage()
        print(f"Epoch {epoch}: Memory delta = {mem_after - mem_before} MB")

```text

**Solutions**:

```mojo

# 1. Reuse buffers
struct EfficientModel:
    var buffer1: Tensor
    var buffer2: Tensor

    fn forward(inout self, x: Tensor) -> Tensor:
        # Reuse pre-allocated buffers
        compute_into(x, self.buffer1)
        activate_inplace(self.buffer1)
        return self.buffer1

# 2. Clear gradients properly
optimizer.zero_grad()

# 3. Use in-place operations
fn update_inplace(inout params: Tensor, grad: Tensor, lr: Float64):
    params -= lr * grad  # In-place update

```text

## Testing Techniques

### Gradient Checking

Compare analytical gradients (backprop) with numerical gradients:

```mojo

fn numerical_gradient(model: Model, input: Tensor, epsilon: Float64 = 1e-5) -> List[Tensor]:
    """Compute gradients numerically."""
    var numerical_grads = List[Tensor]()

    for i in range(len(model.parameters())):
        var param = model.parameters()[i]
        var grad = Tensor.zeros_like(param)

        for j in range(param.size()):
            # f(x + epsilon)
            param[j] += epsilon
            var loss_plus = compute_loss(model, input)

            # f(x - epsilon)
            param[j] -= 2 * epsilon
            var loss_minus = compute_loss(model, input)

            # Numerical gradient
            grad[j] = (loss_plus - loss_minus) / (2 * epsilon)

            # Restore original value
            param[j] += epsilon

        numerical_grads.append(grad)

    return numerical_grads

fn test_gradients():
    """Verify backprop implementation."""
    var model = MyModel()
    var input = Tensor.randn(batch=2, features=10)

    # Analytical gradients
    var loss = model.forward(input)
    loss.backward()
    var analytical = [p.grad for p in model.parameters()]

    # Numerical gradients
    var numerical = numerical_gradient(model, input)

    # Compare
    for i in range(len(analytical)):
        var diff = abs(analytical[i] - numerical[i]).max()
        assert diff < 1e-4, f"Gradient mismatch at parameter {i}: {diff}"

```text

### Overfitting Single Batch

Verify model can overfit a tiny dataset:

```mojo

fn test_overfit_single_batch():
    """Model should achieve near-zero loss on single batch."""
    var model = MyModel()
    var optimizer = SGD(lr=0.01)

    # Single batch
    var x = Tensor.randn(batch=4, features=10)
    var y = Tensor.randint(0, 10, shape=(4,))

    # Train until overfitting
    for i in range(1000):
        var predictions = model.forward(x)
        var loss = cross_entropy(predictions, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 100 == 0:
            print(f"Iteration {i}: Loss = {loss.item()}")

    # Should achieve very low loss
    var final_predictions = model.forward(x)
    var final_loss = cross_entropy(final_predictions, y)
    assert final_loss < 0.01, "Model failed to overfit single batch!"

```text

## Performance Debugging

### Profiling

```mojo

fn profile_training():
    """Identify performance bottlenecks."""
    var total_forward = 0.0
    var total_backward = 0.0
    var total_optimizer = 0.0

    for epoch in range(10):
        for batch in data_loader:
            # Forward pass
            var t0 = time.now()
            var output = model.forward(batch.data)
            var loss = loss_fn(output, batch.targets)
            total_forward += time.now() - t0

            # Backward pass
            t0 = time.now()
            loss.backward()
            total_backward += time.now() - t0

            # Optimizer step
            t0 = time.now()
            optimizer.step()
            optimizer.zero_grad()
            total_optimizer += time.now() - t0

    print(f"Forward:   {total_forward:.2f}s ({total_forward/total_time*100:.1f}%)")
    print(f"Backward:  {total_backward:.2f}s ({total_backward/total_time*100:.1f}%)")
    print(f"Optimizer: {total_optimizer:.2f}s ({total_optimizer/total_time*100:.1f}%)")

```text

### Memory Profiling

```mojo

fn profile_memory():
    """Track memory allocations."""
    var baseline = get_memory_usage()

    var x = Tensor.randn(1000, 1000)
    print(f"Allocated tensor: {get_memory_usage() - baseline} MB")

    var y = x @ x
    print(f"After matmul: {get_memory_usage() - baseline} MB")

    _ = x  # Release x
    print(f"After release: {get_memory_usage() - baseline} MB")

```text

## Troubleshooting Guide

| Symptom | Likely Cause | Solution |
| ------- | ------------ | -------- |
| Loss is NaN | Learning rate too high, numerical instability | Reduce LR, clip gradients, use stable loss |
| Loss not decreasing | Learning rate too low, wrong optimizer | Increase LR, try different optimizer |
| Training slow | Inefficient implementation, large model | Profile, optimize hotspots, use SIMD |
| Memory growing | Memory leak, not releasing tensors | Use in-place ops, reuse buffers |
| Gradients exploding | Deep network, high LR | Gradient clipping, batch norm, reduce LR |
| Gradients vanishing | Deep network, poor initialization | Better init (He/Xavier), skip connections |
| Shape errors | Dimension mismatch | Add shape validation, print intermediate shapes |
| Accuracy plateaus | Underfitting, need more capacity | Add layers, increase width, train longer |

## Best Practices

1. **Start Simple** - Begin with small model and dataset to isolate issues
2. **Validate Incrementally** - Test each component before integrating
3. **Use Assertions** - Add shape and value checks liberally
4. **Monitor Everything** - Track loss, gradients, activations, memory
5. **Compare Baselines** - Verify against known implementations (PyTorch)
6. **Test Edge Cases** - Empty batches, single examples, large batches
7. **Profile Before Optimizing** - Measure to find real bottlenecks

## Related Documentation

- [Testing Strategy](../core/testing-strategy.md) - Comprehensive testing approach
- [Performance Optimization](performance.md) - Optimizing ML code
- [Mojo Patterns](../core/mojo-patterns.md) - Language-specific patterns
- [Custom Layers](custom-layers.md) - Creating and testing custom components

## Summary

**Debugging workflow**:

1. Isolate the problem (which layer/component)
2. Verify shapes and data flow
3. Check for numerical issues (NaN, inf)
4. Validate gradients with numerical checking
5. Profile to find performance bottlenecks
6. Test edge cases systematically

**Key tools**:

- Print debugging with shape/value checks
- Assertions for invariants
- Numerical gradient checking
- Single-batch overfitting test
- Profiling for performance
- Monitoring for memory leaks

**Remember**: Most ML bugs are in implementation details (shapes, gradients) rather than model architecture. Systematic debugging catches these early.
