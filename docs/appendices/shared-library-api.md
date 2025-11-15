# Shared Library API Reference

Complete API documentation for ML Odyssey's shared library including all modules, functions, and advanced usage
patterns.

> **Quick Reference**: For a concise guide, see [Shared Library Guide](../core/shared-library.md).

## Table of Contents

- [Core Module API](#core-module-api)
- [Training Module API](#training-module-api)
- [Data Module API](#data-module-api)
- [Utils Module API](#utils-module-api)
- [Advanced Usage Patterns](#advanced-usage-patterns)
- [Performance Optimization](#performance-optimization)
- [Migration Guide](#migration-guide)

## Core Module API

### Complete Layer API

```mojo
# shared/core/layers.mojo

trait Layer:
    """Base trait for all neural network layers."""

    fn forward(inout self, borrowed input: Tensor) -> Tensor:
        """
        Forward pass through the layer.

        Args:
            input: Input tensor

        Returns:
            Output tensor after applying layer transformation
        """
        ...

    fn parameters(inout self) -> List[Tensor]:
        """
        Get all trainable parameters.

        Returns:
            List of parameter tensors (weights, biases, etc.)
        """
        ...

    fn __str__(borrowed self) -> String:
        """String representation of the layer."""
        ...


struct Linear(Layer):
    """
    Fully connected linear layer: y = xW^T + b

    Attributes:
        weight: Weight matrix of shape [output_size, input_size]
        bias: Bias vector of shape [output_size]
        input_size: Number of input features
        output_size: Number of output features
    """
    var weight: Tensor
    var bias: Tensor
    var input_size: Int
    var output_size: Int

    fn __init__(inout self, input_size: Int, output_size: Int,
                bias: Bool = True):
        """
        Initialize linear layer.

        Args:
            input_size: Size of input features
            output_size: Size of output features
            bias: Whether to include bias term

        Example:
            var layer = Linear(784, 128)
            var input = Tensor.randn(32, 784)
            var output = layer.forward(input)  # Shape: [32, 128]
        """
        self.input_size = input_size
        self.output_size = output_size

        # Kaiming initialization
        var k = sqrt(1.0 / input_size)
        self.weight = Tensor.uniform(output_size, input_size, -k, k)

        if bias:
            self.bias = Tensor.uniform(output_size, -k, k)
        else:
            self.bias = Tensor.zeros(output_size)

    fn forward(inout self, borrowed input: Tensor) raises -> Tensor:
        """
        Forward pass: output = input @ weight.T + bias

        Args:
            input: Input tensor of shape [batch_size, input_size]

        Returns:
            Output tensor of shape [batch_size, output_size]

        Raises:
            ValueError: If input shape doesn't match layer input_size
        """
        if input.shape[-1] != self.input_size:
            raise ValueError("Expected input size {}, got {}".format(
                self.input_size, input.shape[-1]))

        var output = matmul(input, self.weight.T)
        output = output + self.bias.broadcast_to(output.shape)
        return output

    fn parameters(inout self) -> List[Tensor]:
        """Get [weight, bias] parameters."""
        return [self.weight, self.bias]

    fn __str__(borrowed self) -> String:
        """String representation."""
        return "Linear(input_size={}, output_size={})".format(
            self.input_size, self.output_size)


struct Conv2D(Layer):
    """
    2D convolutional layer.

    Attributes:
        weight: Kernel tensor of shape [out_channels, in_channels, kernel_h, kernel_w]
        bias: Bias tensor of shape [out_channels]
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of convolution kernel
        stride: Stride of convolution
        padding: Padding added to input
    """
    var weight: Tensor
    var bias: Tensor
    var in_channels: Int
    var out_channels: Int
    var kernel_size: Int
    var stride: Int
    var padding: Int

    fn __init__(inout self, in_channels: Int, out_channels: Int,
                kernel_size: Int, stride: Int = 1, padding: Int = 0,
                bias: Bool = True):
        """
        Initialize 2D convolutional layer.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of square kernel
            stride: Stride of convolution
            padding: Zero padding added to input
            bias: Whether to use bias

        Example:
            var conv = Conv2D(in_channels=3, out_channels=64, kernel_size=3)
            var input = Tensor.randn(32, 3, 224, 224)
            var output = conv.forward(input)  # Shape: [32, 64, 222, 222]
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # He initialization for ReLU
        var k = sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.weight = Tensor.randn(
            out_channels, in_channels, kernel_size, kernel_size) * k

        if bias:
            self.bias = Tensor.zeros(out_channels)
        else:
            self.bias = Tensor.zeros(out_channels)

    fn forward(inout self, borrowed input: Tensor) raises -> Tensor:
        """
        Forward pass: 2D convolution.

        Args:
            input: Input tensor of shape [batch, in_channels, height, width]

        Returns:
            Output tensor of shape [batch, out_channels, out_height, out_width]

        Raises:
            ValueError: If input channels don't match
        """
        if input.shape[1] != self.in_channels:
            raise ValueError("Expected {} input channels, got {}".format(
                self.in_channels, input.shape[1]))

        var output = conv2d(input, self.weight,
                           stride=self.stride, padding=self.padding)
        output = output + self.bias.reshape(1, -1, 1, 1)
        return output

    fn parameters(inout self) -> List[Tensor]:
        """Get [weight, bias] parameters."""
        return [self.weight, self.bias]


struct BatchNorm2D(Layer):
    """
    Batch normalization for 2D inputs.

    Normalizes inputs across batch and spatial dimensions.
    """
    var gamma: Tensor  # Scale parameter
    var beta: Tensor   # Shift parameter
    var running_mean: Tensor
    var running_var: Tensor
    var num_features: Int
    var eps: Float64
    var momentum: Float64
    var training: Bool

    fn __init__(inout self, num_features: Int, eps: Float64 = 1e-5,
                momentum: Float64 = 0.1):
        """
        Initialize batch normalization layer.

        Args:
            num_features: Number of features (channels)
            eps: Small constant for numerical stability
            momentum: Momentum for running statistics

        Example:
            var bn = BatchNorm2D(64)
            var input = Tensor.randn(32, 64, 28, 28)
            var output = bn.forward(input)  # Normalized output
        """
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.training = True

        self.gamma = Tensor.ones(num_features)
        self.beta = Tensor.zeros(num_features)
        self.running_mean = Tensor.zeros(num_features)
        self.running_var = Tensor.ones(num_features)

    fn forward(inout self, borrowed input: Tensor) raises -> Tensor:
        """
        Apply batch normalization.

        Args:
            input: Input of shape [batch, channels, height, width]

        Returns:
            Normalized output
        """
        if self.training:
            # Compute batch statistics
            var mean = input.mean(dim=[0, 2, 3])
            var var = input.var(dim=[0, 2, 3])

            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + \
                               self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + \
                              self.momentum * var

            # Normalize
            var normalized = (input - mean.reshape(1, -1, 1, 1)) / \
                           sqrt(var.reshape(1, -1, 1, 1) + self.eps)
        else:
            # Use running statistics
            var normalized = (input - self.running_mean.reshape(1, -1, 1, 1)) / \
                           sqrt(self.running_var.reshape(1, -1, 1, 1) + self.eps)

        # Apply scale and shift
        return self.gamma.reshape(1, -1, 1, 1) * normalized + \
               self.beta.reshape(1, -1, 1, 1)

    fn parameters(inout self) -> List[Tensor]:
        """Get [gamma, beta] parameters."""
        return [self.gamma, self.beta]

    fn train(inout self):
        """Set to training mode."""
        self.training = True

    fn eval(inout self):
        """Set to evaluation mode."""
        self.training = False
```

### Complete Activation API

```mojo
# shared/core/activations.mojo

struct ReLU:
    """Rectified Linear Unit: f(x) = max(0, x)"""

    fn forward(borrowed self, borrowed input: Tensor) -> Tensor:
        """Apply ReLU activation."""
        return maximum(input, 0.0)

    fn backward(borrowed self, borrowed grad_output: Tensor,
                borrowed input: Tensor) -> Tensor:
        """Gradient: 1 if x > 0, else 0"""
        return grad_output * (input > 0.0)


struct LeakyReLU:
    """Leaky ReLU: f(x) = max(alpha*x, x)"""
    var negative_slope: Float64

    fn __init__(inout self, negative_slope: Float64 = 0.01):
        self.negative_slope = negative_slope

    fn forward(borrowed self, borrowed input: Tensor) -> Tensor:
        """Apply Leaky ReLU."""
        return where(input > 0, input, input * self.negative_slope)


struct GELU:
    """
    Gaussian Error Linear Unit.

    GELU(x) = x * Φ(x) where Φ is the Gaussian CDF.
    Approximation: GELU(x) ≈ 0.5*x*(1 + tanh(√(2/π)*(x + 0.044715*x³)))
    """

    fn forward(borrowed self, borrowed input: Tensor) -> Tensor:
        """Apply GELU activation."""
        return 0.5 * input * (1.0 + tanh(
            sqrt(2.0 / 3.14159265359) *
            (input + 0.044715 * input * input * input)
        ))


struct Softmax:
    """Softmax activation: exp(x_i) / sum(exp(x))"""
    var dim: Int

    fn __init__(inout self, dim: Int = -1):
        self.dim = dim

    fn forward(borrowed self, borrowed input: Tensor) -> Tensor:
        """
        Apply softmax along specified dimension.

        Args:
            input: Input tensor

        Returns:
            Softmax probabilities (sum to 1 along dim)
        """
        # Subtract max for numerical stability
        var x = input - input.max(dim=self.dim, keepdim=True)
        var exp_x = exp(x)
        return exp_x / exp_x.sum(dim=self.dim, keepdim=True)
```

## Training Module API

### Complete Optimizer API

```mojo
# shared/training/optimizers.mojo

struct SGD:
    """
    Stochastic Gradient Descent optimizer.

    Implements: θ = θ - lr * (∇L + wd * θ) + momentum * v
    """
    var learning_rate: Float64
    var momentum: Float64
    var weight_decay: Float64
    var nesterov: Bool
    var velocity: Dict[Int, Tensor]

    fn __init__(inout self, learning_rate: Float64 = 0.01,
                momentum: Float64 = 0.0, weight_decay: Float64 = 0.0,
                nesterov: Bool = False):
        """
        Initialize SGD optimizer.

        Args:
            learning_rate: Learning rate
            momentum: Momentum factor
            weight_decay: L2 regularization coefficient
            nesterov: Use Nesterov momentum

        Example:
            var optimizer = SGD(learning_rate=0.1, momentum=0.9)
            for param in model.parameters():
                optimizer.step(param, param.grad)
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.velocity = Dict[Int, Tensor]()

    fn step(inout self, inout param: Tensor, borrowed grad: Tensor):
        """
        Perform single optimization step.

        Args:
            param: Parameter to update (modified in-place)
            grad: Gradient of loss w.r.t. parameter
        """
        var param_id = id(param)

        # Add weight decay
        var g = grad
        if self.weight_decay != 0:
            g = g + self.weight_decay * param

        # Apply momentum
        if self.momentum != 0:
            if param_id not in self.velocity:
                self.velocity[param_id] = Tensor.zeros_like(param)

            var v = self.velocity[param_id]
            v = self.momentum * v + g

            if self.nesterov:
                g = g + self.momentum * v
            else:
                g = v

            self.velocity[param_id] = v

        # Update parameters
        param = param - self.learning_rate * g

    fn zero_grad(inout self, inout params: List[Tensor]):
        """Zero out gradients."""
        for param in params:
            param.grad.zero_()


struct Adam:
    """
    Adam optimizer (Adaptive Moment Estimation).

    Implements: Kingma & Ba, "Adam: A Method for Stochastic Optimization"
    """
    var learning_rate: Float64
    var beta1: Float64
    var beta2: Float64
    var epsilon: Float64
    var weight_decay: Float64
    var amsgrad: Bool
    var t: Int
    var m: Dict[Int, Tensor]  # First moment
    var v: Dict[Int, Tensor]  # Second moment
    var v_max: Dict[Int, Tensor]  # For AMSGrad

    fn __init__(inout self, learning_rate: Float64 = 0.001,
                beta1: Float64 = 0.9, beta2: Float64 = 0.999,
                epsilon: Float64 = 1e-8, weight_decay: Float64 = 0.0,
                amsgrad: Bool = False):
        """
        Initialize Adam optimizer.

        Args:
            learning_rate: Learning rate
            beta1: First moment decay rate
            beta2: Second moment decay rate
            epsilon: Small constant for numerical stability
            weight_decay: Weight decay (L2 penalty)
            amsgrad: Use AMSGrad variant
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.t = 0
        self.m = Dict[Int, Tensor]()
        self.v = Dict[Int, Tensor]()
        self.v_max = Dict[Int, Tensor]()

    fn step(inout self, inout param: Tensor, borrowed grad: Tensor):
        """Perform Adam optimization step."""
        self.t += 1
        var param_id = id(param)

        # Initialize moments
        if param_id not in self.m:
            self.m[param_id] = Tensor.zeros_like(param)
            self.v[param_id] = Tensor.zeros_like(param)
            if self.amsgrad:
                self.v_max[param_id] = Tensor.zeros_like(param)

        var m = self.m[param_id]
        var v = self.v[param_id]

        # Add weight decay
        var g = grad
        if self.weight_decay != 0:
            g = g + self.weight_decay * param

        # Update biased first moment
        m = self.beta1 * m + (1 - self.beta1) * g

        # Update biased second moment
        v = self.beta2 * v + (1 - self.beta2) * g * g

        # Bias correction
        var m_hat = m / (1 - pow(self.beta1, self.t))
        var v_hat = v / (1 - pow(self.beta2, self.t))

        # AMSGrad
        if self.amsgrad:
            var v_max = self.v_max[param_id]
            v_max = maximum(v_max, v_hat)
            v_hat = v_max
            self.v_max[param_id] = v_max

        # Update parameters
        param = param - self.learning_rate * m_hat / (sqrt(v_hat) + self.epsilon)

        # Store updated moments
        self.m[param_id] = m
        self.v[param_id] = v
```

## Advanced Usage Patterns

### Custom Layer Implementation

```mojo
struct ResidualBlock(Layer):
    """
    Residual block with skip connection.

    Architecture: input -> conv1 -> bn1 -> relu -> conv2 -> bn2 -> (+input) -> relu
    """
    var conv1: Conv2D
    var bn1: BatchNorm2D
    var conv2: Conv2D
    var bn2: BatchNorm2D
    var downsample: Optional[Sequential]
    var stride: Int

    fn __init__(inout self, in_channels: Int, out_channels: Int,
                stride: Int = 1):
        """Initialize residual block."""
        self.stride = stride

        self.conv1 = Conv2D(in_channels, out_channels, kernel_size=3,
                           stride=stride, padding=1, bias=False)
        self.bn1 = BatchNorm2D(out_channels)
        self.conv2 = Conv2D(out_channels, out_channels, kernel_size=3,
                           stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm2D(out_channels)

        # Downsample if dimensions change
        if stride != 1 or in_channels != out_channels:
            self.downsample = Sequential([
                Conv2D(in_channels, out_channels, kernel_size=1,
                      stride=stride, bias=False),
                BatchNorm2D(out_channels)
            ])
        else:
            self.downsample = None

    fn forward(inout self, borrowed input: Tensor) raises -> Tensor:
        """Forward pass with residual connection."""
        var identity = input

        # Main path
        var out = self.conv1.forward(input)
        out = self.bn1.forward(out)
        out = relu(out)

        out = self.conv2.forward(out)
        out = self.bn2.forward(out)

        # Skip connection
        if self.downsample:
            identity = self.downsample.forward(input)

        out = out + identity
        out = relu(out)

        return out

    fn parameters(inout self) -> List[Tensor]:
        """Get all parameters."""
        var params = List[Tensor]()
        params.extend(self.conv1.parameters())
        params.extend(self.bn1.parameters())
        params.extend(self.conv2.parameters())
        params.extend(self.bn2.parameters())
        if self.downsample:
            params.extend(self.downsample.parameters())
        return params
```

### Multi-Head Attention

```mojo
struct MultiHeadAttention(Layer):
    """
    Multi-head self-attention mechanism.

    Used in Transformers: "Attention Is All You Need" (Vaswani et al., 2017)
    """
    var num_heads: Int
    var head_dim: Int
    var embed_dim: Int
    var q_proj: Linear
    var k_proj: Linear
    var v_proj: Linear
    var out_proj: Linear
    var dropout: Float64

    fn __init__(inout self, embed_dim: Int, num_heads: Int,
                dropout: Float64 = 0.0):
        """
        Initialize multi-head attention.

        Args:
            embed_dim: Total embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        assert embed_dim % num_heads == 0, \
            "embed_dim must be divisible by num_heads"

        self.q_proj = Linear(embed_dim, embed_dim)
        self.k_proj = Linear(embed_dim, embed_dim)
        self.v_proj = Linear(embed_dim, embed_dim)
        self.out_proj = Linear(embed_dim, embed_dim)

    fn forward(inout self, borrowed query: Tensor, borrowed key: Tensor,
               borrowed value: Tensor, borrowed mask: Optional[Tensor] = None) raises -> Tensor:
        """
        Compute multi-head attention.

        Args:
            query: Query tensor [batch, seq_len, embed_dim]
            key: Key tensor [batch, seq_len, embed_dim]
            value: Value tensor [batch, seq_len, embed_dim]
            mask: Attention mask [batch, seq_len, seq_len]

        Returns:
            Attention output [batch, seq_len, embed_dim]
        """
        var batch_size = query.shape[0]
        var seq_len = query.shape[1]

        # Project and reshape to [batch, num_heads, seq_len, head_dim]
        var Q = self.q_proj.forward(query).reshape(
            batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        var K = self.k_proj.forward(key).reshape(
            batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        var V = self.v_proj.forward(value).reshape(
            batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        var scores = matmul(Q, K.transpose(-2, -1)) / sqrt(self.head_dim)

        if mask:
            scores = scores.masked_fill(mask == 0, -1e9)

        var attn_weights = softmax(scores, dim=-1)
        if self.dropout > 0:
            attn_weights = dropout(attn_weights, p=self.dropout)

        var attn_output = matmul(attn_weights, V)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, seq_len, self.embed_dim)

        return self.out_proj.forward(attn_output)

    fn parameters(inout self) -> List[Tensor]:
        """Get all parameters."""
        var params = List[Tensor]()
        params.extend(self.q_proj.parameters())
        params.extend(self.k_proj.parameters())
        params.extend(self.v_proj.parameters())
        params.extend(self.out_proj.parameters())
        return params
```

## Performance Optimization

(See advanced documentation for performance tuning strategies)

## Migration Guide

(See development documentation for API migration guides)

This appendix provides complete API reference. For quick usage,
see [Shared Library Guide](../core/shared-library.md).
