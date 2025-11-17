# Custom Layers

Guide to creating custom neural network layers in ML Odyssey.

## Overview

Create custom layers when:

- Built-in layers don't meet your needs
- Implementing novel architectures from research papers
- Optimizing specific operations for your use case

All custom layers should integrate seamlessly with the shared library.

## Layer Structure

Every layer must implement:

```mojo

struct CustomLayer:
    """Custom neural network layer."""
    var parameters: List[Tensor]  # Trainable parameters

    fn __init__(inout self, ...):
        """Initialize layer with parameters."""
        pass

    fn forward(self, borrowed input: Tensor) -> Tensor:
        """Forward pass."""
        pass

    fn backward(self, borrowed grad_output: Tensor) -> Tensor:
        """Backward pass - compute gradients."""
        pass

    fn parameters(self) -> List[Tensor]:
        """Return trainable parameters."""
        return self.parameters

```text

## Implementation Guide

### Step 1: Define Structure

```mojo

struct MyCustomLayer:
    """Custom layer with learnable parameters."""
    var weight: Tensor
    var bias: Tensor
    var activation: String

```text

### Step 2: Initialize Parameters

```mojo

fn __init__(inout self, input_size: Int, output_size: Int):
    """Initialize with He initialization."""
    self.weight = Tensor.randn(output_size, input_size)
    self.weight *= sqrt(2.0 / input_size)
    self.bias = Tensor.zeros(output_size)
    self.activation = "relu"

```text

### Step 3: Implement Forward Pass

```mojo

fn forward(self, borrowed input: Tensor) -> Tensor:
    """Compute layer output."""
    var output = input @ self.weight.T + self.bias
    if self.activation == "relu":
        return relu(output)
    return output

```text

### Step 4: Implement Backward Pass

```mojo

fn backward(self, borrowed grad_output: Tensor, borrowed input: Tensor) -> Tensor:
    """Compute parameter gradients and input gradient."""
    # Gradient w.r.t input
    var grad_input = grad_output @ self.weight

    # Gradient w.r.t parameters
    self.weight.grad = grad_output.T @ input
    self.bias.grad = grad_output.sum(dim=0)

    return grad_input

```text

## Examples

### Example 1: Custom Attention Layer

```mojo

struct SelfAttention:
    """Self-attention mechanism."""
    var query: Tensor
    var key: Tensor
    var value: Tensor
    var scale: Float64

    fn __init__(inout self, d_model: Int):
        self.query = Tensor.randn(d_model, d_model) * 0.02
        self.key = Tensor.randn(d_model, d_model) * 0.02
        self.value = Tensor.randn(d_model, d_model) * 0.02
        self.scale = 1.0 / sqrt(Float64(d_model))

    fn forward(self, borrowed x: Tensor) -> Tensor:
        """Compute self-attention: softmax(QK^T/sqrt(d))V."""
        var Q = x @ self.query
        var K = x @ self.key
        var V = x @ self.value

        var scores = (Q @ K.T) * self.scale
        var attention_weights = softmax(scores, dim=-1)
        return attention_weights @ V

```text

### Example 2: Custom Activation

```mojo

struct Swish:
    """Swish activation: x * sigmoid(x)."""
    var beta: Float64

    fn __init__(inout self, beta: Float64 = 1.0):
        self.beta = beta

    fn forward(self, borrowed x: Tensor) -> Tensor:
        """Apply swish activation."""
        return x * sigmoid(self.beta * x)

    fn backward(self, borrowed grad_output: Tensor, borrowed x: Tensor) -> Tensor:
        """Gradient: swish'(x) = swish(x) + sigmoid(x)(1 - swish(x))."""
        var sig = sigmoid(self.beta * x)
        var swish_val = x * sig
        return grad_output * (swish_val + sig * (1.0 - swish_val))

```text

### Example 3: Custom Conv2D with Padding

```mojo

struct CustomConv2D:
    """2D Convolution with custom padding strategy."""
    var weight: Tensor  # (out_channels, in_channels, kernel_h, kernel_w)
    var bias: Tensor
    var stride: Int
    var padding: Int

    fn __init__(inout self, in_channels: Int, out_channels: Int,
                kernel_size: Int, stride: Int = 1, padding: Int = 0):
        self.weight = Tensor.randn(out_channels, in_channels, kernel_size, kernel_size)
        self.weight *= sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.bias = Tensor.zeros(out_channels)
        self.stride = stride
        self.padding = padding

    fn forward(self, borrowed input: Tensor) -> Tensor:
        """Apply 2D convolution."""
        # Apply padding if needed
        var padded = input
        if self.padding > 0:
            padded = pad_tensor(input, self.padding)

        # Convolution operation (simplified)
        return conv2d_operation(padded, self.weight, self.bias, self.stride)

```text

## Testing Custom Layers

### Unit Tests

```mojo

fn test_custom_layer_forward():
    """Test forward pass produces correct output shape."""
    var layer = MyCustomLayer(input_size=10, output_size=5)
    var input = Tensor.randn(batch_size=2, features=10)
    var output = layer.forward(input)
    assert_equal(output.shape, (2, 5))

fn test_custom_layer_gradients():
    """Test gradients are computed correctly."""
    var layer = MyCustomLayer(10, 5)
    var input = Tensor.randn(2, 10)

    # Numerical gradient
    var numerical_grad = compute_numerical_gradient(layer, input)

    # Analytical gradient
    var output = layer.forward(input)
    var grad_output = Tensor.ones_like(output)
    layer.backward(grad_output, input)

    # Compare
    assert_almost_equal(layer.weight.grad, numerical_grad, tolerance=1e-5)

```text

### Integration Tests

```mojo

fn test_custom_layer_in_network():
    """Test custom layer works in complete network."""
    var model = Sequential([
        Linear(784, 128),
        ReLU(),
        MyCustomLayer(128, 64),  # Your custom layer
        ReLU(),
        Linear(64, 10)
    ])

    var input = Tensor.randn(batch=32, features=784)
    var output = model.forward(input)
    assert_equal(output.shape, (32, 10))

```text

## Integration with Shared Library

### Adding to Shared Library

If your layer is reusable, contribute it to the shared library:

1. **Implement in** `shared/core/layers/`
2. **Add tests in** `tests/shared/core/`
3. **Document in** layer docstring
4. **Export from** `shared/core/__init__.mojo`
5. **Update** `docs/core/shared-library.md`

### Using Shared Components

Build on existing shared library components:

```mojo

from shared.core.layers import Linear
from shared.core.ops import matmul, softmax

struct CustomBlock:
    """Custom block using shared components."""
    var fc1: Linear
    var fc2: Linear
    var custom_op: MyCustomOp

    fn __init__(inout self, d_model: Int):
        self.fc1 = Linear(d_model, d_model * 4)
        self.fc2 = Linear(d_model * 4, d_model)
        self.custom_op = MyCustomOp()

    fn forward(self, borrowed x: Tensor) -> Tensor:
        var h = self.fc1.forward(x)
        h = self.custom_op.forward(h)
        return self.fc2.forward(h)

```text

## Best Practices

**1. Start Simple**

Begin with a basic implementation, verify correctness, then optimize:

```mojo

# First: Get it working
fn forward_simple(self, x: Tensor) -> Tensor:
    var result = Tensor.zeros(output_shape)
    for i in range(batch_size):
        for j in range(output_size):
            result[i, j] = compute_element(x, i, j)
    return result

# Then: Optimize with SIMD
fn forward_optimized(self, x: Tensor) -> Tensor:
    alias simd_width = simdwidthof[DType.float32]()
    # Vectorized implementation
    ...

```text

**2. Always Validate Gradients**

Use numerical gradient checking during development:

```mojo

fn validate_gradients(layer: YourLayer, input: Tensor):
    """Compare analytical vs numerical gradients."""
    var epsilon = 1e-5
    var numerical = compute_numerical_gradient(layer, input, epsilon)
    var analytical = compute_analytical_gradient(layer, input)
    assert_almost_equal(numerical, analytical, tolerance=1e-4)

```text

**3. Test Edge Cases**

```mojo

fn test_edge_cases():
    """Test boundary conditions."""
    var layer = MyCustomLayer(10, 5)

    # Empty batch
    var empty_input = Tensor.zeros(0, 10)
    var output = layer.forward(empty_input)
    assert_equal(output.shape[0], 0)

    # Single example
    var single = Tensor.randn(1, 10)
    var result = layer.forward(single)
    assert_equal(result.shape, (1, 5))

    # Large batch
    var large = Tensor.randn(1000, 10)
    var out = layer.forward(large)
    assert_equal(out.shape, (1000, 5))

```text

**4. Document Thoroughly**

```mojo

struct MyLayer:
    """One-line summary.

    Detailed description of what this layer does, its purpose,
    and when to use it.

    Args:
        input_size: Dimension of input features.
        output_size: Dimension of output features.

    Example:
        ```mojo
        var layer = MyLayer(input_size=128, output_size=64)
        var output = layer.forward(input_batch)
        ```

    References:

        - Paper: "Title" (Author et al., Year)
        - <https://arxiv.org/abs/XXXX.XXXXX>
        - <https://arxiv.org/abs/XXXX.XXXXX>

    """
    var weight: Tensor
    var bias: Tensor

```text

**5. Profile Performance**

```mojo

fn benchmark_layer():
    """Measure layer performance."""
    var layer = MyCustomLayer(1024, 1024)
    var input = Tensor.randn(128, 1024)

    # Warmup
    for _ in range(10):
        _ = layer.forward(input)

    # Benchmark
    var start = time.now()
    for _ in range(100):
        _ = layer.forward(input)
    var elapsed = (time.now() - start) / 100

    print("Average forward pass:", elapsed, "ms")

```text

**6. Follow Mojo Patterns**

- Use `fn` for performance-critical code
- Use `struct` for value semantics
- Annotate ownership (`borrowed`, `inout`, `owned`)
- Leverage SIMD for element-wise operations
- Minimize allocations in forward/backward passes

**7. Provide Usage Examples**

Include working examples in your layer's documentation showing common use cases.

## Related Documentation

- [Mojo Patterns](../core/mojo-patterns.md) - Language-specific ML patterns
- [Shared Library](../core/shared-library.md) - Built-in layers and components
- [Performance Optimization](performance.md) - Optimizing custom layers
- [Testing Strategy](../core/testing-strategy.md) - Comprehensive testing approach

## Summary

**Key takeaways**:

1. Implement required interface: `__init__`, `forward`, `backward`, `parameters`
2. Test thoroughly: unit tests, gradient checks, integration tests
3. Start simple, optimize incrementally with profiling guidance
4. Document clearly with examples and references
5. Follow Mojo patterns for performance and safety
6. Contribute reusable layers back to shared library

**Next steps**:

- Review examples in `papers/` implementations
- Study existing layers in `shared/core/layers/`
- Try implementing a simple custom activation function
- Validate gradients with numerical checking
