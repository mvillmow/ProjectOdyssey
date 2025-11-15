# Creating Custom Layers

Guide to implementing custom neural network layers with forward and backward passes.

## Overview

ML Odyssey's modular architecture makes it easy to create custom layers. This guide covers implementing layers
from scratch with proper forward/backward passes, gradient computation, and integration with the training system.

## Layer Interface

All layers implement the `Module` trait:

```mojo
trait Module:
    """Base interface for neural network modules."""

    fn forward(inout self, borrowed input: Tensor) -> Tensor:
        """Forward pass."""
        ...

    fn parameters(inout self) -> List[Tensor]:
        """Get trainable parameters."""
        ...
```

## Simple Custom Layer

### Example: Scaled Layer

A layer that scales input by a learnable parameter:

```mojo
from shared.core import Module, Tensor

struct ScaledLayer(Module):
    """Layer that scales input by learnable parameter."""
    var scale: Tensor

    fn __init__(inout self):
        # Initialize scale to 1.0
        self.scale = Tensor([1.0])

    fn forward(inout self, borrowed input: Tensor) -> Tensor:
        """Multiply input by scale parameter."""
        return input * self.scale[0]

    fn parameters(inout self) -> List[Tensor]:
        """Return scale parameter for optimization."""
        return [self.scale]
```

Usage:

```mojo
var layer = ScaledLayer()
var input = Tensor.randn(10, 20)
var output = layer.forward(input)  # Scaled by learnable parameter

# During training, optimizer will update layer.scale
```

## Intermediate Custom Layer

### Example: Custom Activation

Parametric ReLU with learnable slope.

See `examples/custom-layers/prelu_activation.mojo`](

Key implementation:

```mojo
struct PReLU(Module):
    """Parametric ReLU: PReLU(x) = max(0, x) + α * min(0, x)"""
    var alpha: Tensor  # Learnable slope for negative values

    fn forward(inout self, borrowed input: Tensor) -> Tensor:
        var positive = max(input, 0.0)
        var negative = min(input, 0.0)
        return positive + self.alpha[0] * negative
```

Full example: `examples/custom-layers/prelu_activation.mojo`

## Advanced Custom Layer

### Example: Custom Convolutional Layer

Depthwise separable convolution:

```mojo
struct DepthwiseSeparableConv(Module):
    """Depthwise separable convolution.

    Splits convolution into depthwise and pointwise operations.
    More efficient than standard convolution.
    """
    var depthwise: Conv2D  # Depthwise convolution
    var pointwise: Conv2D  # Pointwise (1x1) convolution
    var bn1: BatchNorm2D   # Batch norm after depthwise
    var bn2: BatchNorm2D   # Batch norm after pointwise

    fn __init__(
        inout self,
        in_channels: Int,
        out_channels: Int,
        kernel_size: Int,
        stride: Int = 1,
        padding: Int = 0
    ):
        """Initialize depthwise separable conv.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of convolutional kernel.
            stride: Stride of convolution.
            padding: Padding applied to input.
        """
        # Depthwise: Each input channel convolved separately
        self.depthwise = Conv2D(
            in_channels,
            in_channels,  # Same number of output channels
            kernel_size,
            stride,
            padding,
            groups=in_channels  # Key: group convolution
        )

        # Pointwise: 1x1 conv to combine channels
        self.pointwise = Conv2D(
            in_channels,
            out_channels,
            kernel_size=1
        )

        # Batch normalization
        self.bn1 = BatchNorm2D(in_channels)
        self.bn2 = BatchNorm2D(out_channels)

    fn forward(inout self, borrowed input: Tensor) -> Tensor:
        """Forward pass through depthwise separable conv."""
        # Depthwise convolution
        var x = self.depthwise.forward(input)
        x = self.bn1.forward(x)
        x = relu(x)

        # Pointwise convolution
        x = self.pointwise.forward(x)
        x = self.bn2.forward(x)
        x = relu(x)

        return x

    fn parameters(inout self) -> List[Tensor]:
        """Get all parameters."""
        var params = List[Tensor]()
        params.extend(self.depthwise.parameters())
        params.extend(self.bn1.parameters())
        params.extend(self.pointwise.parameters())
        params.extend(self.bn2.parameters())
        return params
```

## Custom Backward Pass

### Manual Gradient Computation

For layers with custom backward behavior:

```mojo
struct CustomLinear(Module):
    """Linear layer with custom backward pass."""
    var weight: Tensor
    var bias: Tensor
    var input_cache: Tensor  # Cache for backward

    fn __init__(inout self, input_size: Int, output_size: Int):
        # Xavier initialization
        var std = sqrt(2.0 / (input_size + output_size))
        self.weight = Tensor.randn(output_size, input_size) * std
        self.bias = Tensor.zeros(output_size)

    fn forward(inout self, borrowed input: Tensor) -> Tensor:
        """Forward pass, cache input for backward."""
        # Cache input for backward pass
        self.input_cache = input.clone()

        # Compute output: Y = XW^T + b
        return input @ self.weight.T + self.bias

    fn backward(inout self, borrowed grad_output: Tensor) -> Tensor:
        """Custom backward pass.

        Args:
            grad_output: Gradient from next layer.

        Returns:
            Gradient with respect to input.
        """
        var batch_size = grad_output.shape[0]

        # Gradient w.r.t. bias: sum over batch
        self.bias.grad = grad_output.sum(axis=0)

        # Gradient w.r.t. weight: grad_output^T @ input
        self.weight.grad = grad_output.T @ self.input_cache

        # Gradient w.r.t. input: grad_output @ weight
        var grad_input = grad_output @ self.weight

        return grad_input

    fn parameters(inout self) -> List[Tensor]:
        return [self.weight, self.bias]
```

## Custom Loss Function

### Example: Focal Loss

Custom loss for imbalanced datasets.

See `examples/custom-layers/focal_loss.mojo`](

Key implementation:

```mojo
struct FocalLoss:
    """Focal loss: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    """
    var alpha: Float64  # Weighting factor
    var gamma: Float64  # Focusing parameter

    fn __call__(self, borrowed predictions: Tensor, borrowed targets: Tensor) -> Tensor:
        # Get predicted probabilities for true class
        var p_t = get_class_probabilities(predictions, targets)
        # Focal loss formula
        var focal_weight = (1.0 - p_t) ** self.gamma
        return (self.alpha * focal_weight * -log(p_t + 1e-7)).mean()
```

Full example: `examples/custom-layers/focal_loss.mojo`

## Attention Mechanism

### Example: Multi-Head Attention

Multi-head self-attention mechanism used in Transformers.

See `examples/custom-layers/attention_layer.mojo`](

Key implementation:

```mojo
struct MultiHeadAttention(Module):
    """Multi-head self-attention mechanism."""
    var num_heads: Int
    var head_dim: Int
    var q_proj: Linear  # Query projection
    var k_proj: Linear  # Key projection
    var v_proj: Linear  # Value projection
    var out_proj: Linear  # Output projection

    fn forward(inout self, borrowed x: Tensor) -> Tensor:
        # Project to Q, K, V
        var q = self.q_proj.forward(x)
        var k = self.k_proj.forward(x)
        var v = self.v_proj.forward(x)
        # Reshape, compute attention, project output
        # ... (see full example)
```

Full example: `examples/custom-layers/attention_layer.mojo`

## Testing Custom Layers

### Unit Tests

```mojo
from testing import assert_equal, assert_true

fn test_prelu_forward():
    """Test PReLU forward pass."""
    var layer = PReLU(num_features=1, init_value=0.25)
    var input = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    var output = layer.forward(input)

    # PReLU(x) = max(0, x) + α * min(0, x)
    var expected = Tensor([-0.5, -0.25, 0.0, 1.0, 2.0])  # α = 0.25

    assert_tensors_close(output, expected, atol=1e-6)

fn test_prelu_gradient():
    """Test PReLU gradient computation."""
    var layer = PReLU()
    var input = Tensor.randn(10, 20)

    # Forward pass
    var output = layer.forward(input)

    # Backward pass
    var grad_output = Tensor.ones_like(output)
    output.backward(grad_output)

    # Check gradients exist
    assert_true(layer.alpha.grad is not None)

fn test_multihead_attention_shape():
    """Test multi-head attention output shape."""
    var layer = MultiHeadAttention(embed_dim=512, num_heads=8)
    var input = Tensor.randn(32, 100, 512)  # [batch, seq_len, embed_dim]

    var output = layer.forward(input)

    # Output should have same shape as input
    assert_equal(output.shape, input.shape)
```

### Integration Tests

```mojo
fn test_custom_layer_in_model():
    """Test custom layer integrated in model."""
    var model = Sequential([
        Linear(784, 256),
        PReLU(num_features=256),
        Linear(256, 128),
        PReLU(num_features=128),
        Linear(128, 10),
    ])

    var input = Tensor.randn(32, 784)
    var output = model.forward(input)

    assert_equal(output.shape, [32, 10])

fn test_custom_layer_training():
    """Test custom layer can be trained."""
    var layer = PReLU()
    var optimizer = SGD(lr=0.01)

    # Small dataset
    var X = Tensor.randn(100, 10)
    var y = Tensor.randn(100, 10)

    var initial_alpha = layer.alpha[0]

    # Train for a few steps
    for _ in range(10):
        var output = layer.forward(X)
        var loss = mse_loss(output, y)
        loss.backward()
        optimizer.step(layer.parameters())

    # Alpha should have changed
    assert_true(layer.alpha[0] != initial_alpha)
```

## Performance Optimization

### SIMD Optimization

Optimize custom layer with SIMD:

```mojo
from algorithm import vectorize

struct OptimizedPReLU(Module):
    """SIMD-optimized PReLU activation."""
    var alpha: Tensor

    fn __init__(inout self, num_features: Int = 1, init_value: Float64 = 0.25):
        self.alpha = Tensor.ones(num_features) * init_value

    fn forward(inout self, borrowed input: Tensor) -> Tensor:
        """SIMD-optimized forward pass."""
        var output = Tensor.zeros_like(input)
        alias simd_width = simdwidthof[DType.float32]()

        @parameter
        fn vectorized_prelu[width: Int](idx: Int):
            var x = input.load[width](idx)
            var positive = max(x, 0.0)
            var negative = min(x, 0.0) * self.alpha[0]
            output.store[width](idx, positive + negative)

        vectorize[simd_width, vectorized_prelu](input.size())
        return output

    fn parameters(inout self) -> List[Tensor]:
        return [self.alpha]
```

## Best Practices

### DO

- ✅ Implement `Module` trait
- ✅ Return all trainable parameters
- ✅ Cache inputs needed for backward pass
- ✅ Test forward and backward passes
- ✅ Document mathematical operations
- ✅ Validate input shapes
- ✅ Use SIMD for performance-critical operations

### DON'T

- ❌ Modify input tensors
- ❌ Forget to cache for backward pass
- ❌ Skip gradient checking
- ❌ Ignore numerical stability
- ❌ Hardcode shapes
- ❌ Skip documentation

## Common Patterns

### Pattern 1: Layer with Optional Bias

```mojo
struct LinearNoBias(Module):
    """Linear layer without bias term."""
    var weight: Tensor

    fn __init__(inout self, input_size: Int, output_size: Int):
        self.weight = Tensor.randn(output_size, input_size)

    fn forward(inout self, borrowed input: Tensor) -> Tensor:
        return input @ self.weight.T

    fn parameters(inout self) -> List[Tensor]:
        return [self.weight]  # No bias
```

### Pattern 2: Residual Connection

```mojo
struct ResidualBlock(Module):
    """Residual block with skip connection."""
    var conv1: Conv2D
    var conv2: Conv2D
    var bn1: BatchNorm2D
    var bn2: BatchNorm2D

    fn forward(inout self, borrowed input: Tensor) -> Tensor:
        var identity = input  # Save for skip connection

        # First conv block
        var out = self.conv1.forward(input)
        out = self.bn1.forward(out)
        out = relu(out)

        # Second conv block
        out = self.conv2.forward(out)
        out = self.bn2.forward(out)

        # Add skip connection
        out = out + identity

        return relu(out)
```

## Next Steps

- **[Performance Guide](performance.md)** - Optimize custom layers
- **[Testing Strategy](../core/testing-strategy.md)** - Test custom implementations
- **[Mojo Patterns](../core/mojo-patterns.md)** - Mojo-specific patterns
- **[Paper Implementation](../core/paper-implementation.md)** - Use custom layers in papers

## Related Documentation

- [Shared Library](../core/shared-library.md) - Built-in layers
- [Debugging](debugging.md) - Debug custom layers
- [Visualization](visualization.md) - Visualize layer activations
