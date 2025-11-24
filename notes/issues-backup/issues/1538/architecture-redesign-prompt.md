# Architecture Redesign Prompt: Two-Level Functional + Class-Based API

## Objective

Redesign the ML Odyssey shared library to implement a clean two-level hierarchy with a functional core and class-based wrapper layer, while migrating all tensor functionality from `src/extensor/` into the proper locations within `shared/`.

## Current State Analysis

### What Exists

1. **ExTensor Implementation** (`src/extensor/`)
   - Comprehensive tensor with 150+ operations
   - SIMD-optimized, Array API Standard compliant
   - Dynamic shapes, 13 data types
   - Files: `extensor.mojo`, `arithmetic.mojo`, `matrix.mojo`, `reduction.mojo`, `activations.mojo`, `losses.mojo`, `initializers.mojo`, `elementwise_math.mojo`, `comparison.mojo`, `broadcasting.mojo`, `shape.mojo`

1. **Shared Library Structure** (`shared/`)
   - `core/`: Stub directories (layers/, ops/, types/, utils/)
   - `data/`: Implemented (datasets, loaders, transforms) - imports non-existent `tensor.Tensor`
   - `training/`: Partially implemented (loops, metrics, optimizers/sgd as function)
   - `utils/`: Implemented (config, logging, io, profiling, random, visualization)

1. **Architecture Gaps**
   - Tests expect class-based APIs: `Linear()`, `SGD()`, `ReLU()`, `Tensor()`
   - Implementation provides functional APIs: `sgd_step()`, `relu()`, ExTensor struct
   - ExTensor is in wrong location (`src/` instead of `shared/core/types/`)
   - No layer classes implemented
   - No optimizer classes (only `sgd_step()` function)

## Target Architecture

### Two-Level Hierarchy

### Level 1: Functional Layer (Pure Functions)

- Location: `shared/core/ops/` and `shared/core/types/`
- All functions take only ExTensors (scalars represented as 0-D ExTensors)
- Pure, stateless, composable functions
- Type-generic (work with any ExTensor dtype that makes sense)
- Examples: `matmul(a: ExTensor, b: ExTensor) -> ExTensor`, `relu(x: ExTensor) -> ExTensor`, `sgd_step(...) -> ExTensor`

### Level 2: Class-Based Layer (Stateful Wrappers)

- Location: `shared/core/layers/` and `shared/training/optimizers/`
- Classes with state (weights, biases, optimizer momentum, etc.)
- Compose functional operations
- Provide high-level API matching test expectations
- Examples: `Linear`, `Conv2D`, `ReLU`, `SGD`, `Adam`

### Directory Structure (After Migration)

```text
shared/
├── core/
│   ├── types/
│   │   ├── __init__.mojo          # Export ExTensor, creation functions
│   │   ├── extensor.mojo          # ExTensor struct (from src/extensor/extensor.mojo)
│   │   └── shape.mojo             # Shape utilities (from src/extensor/shape.mojo)
│   ├── ops/
│   │   ├── __init__.mojo          # Export all operations
│   │   ├── arithmetic.mojo        # add, subtract, multiply, etc. (from src/extensor/)
│   │   ├── matrix.mojo            # matmul, transpose, dot (from src/extensor/)
│   │   ├── reduction.mojo         # sum, mean, max, min (from src/extensor/)
│   │   ├── elementwise.mojo       # exp, log, sqrt, sin, cos (from src/extensor/)
│   │   ├── comparison.mojo        # equal, less, greater (from src/extensor/)
│   │   ├── broadcasting.mojo      # broadcast utilities (from src/extensor/)
│   │   ├── activations.mojo       # relu, sigmoid, tanh, etc. (from src/extensor/)
│   │   ├── losses.mojo            # loss functions (from src/extensor/)
│   │   └── initializers.mojo      # xavier, he, etc. (from src/extensor/)
│   ├── layers/
│   │   ├── __init__.mojo          # Export all layer classes
│   │   ├── linear.mojo            # Linear layer class (NEW)
│   │   ├── conv.mojo              # Conv2D layer class (NEW)
│   │   ├── activation.mojo        # ReLU, Sigmoid, Tanh layer classes (NEW)
│   │   ├── pooling.mojo           # MaxPool2D, AvgPool2D classes (NEW)
│   │   └── normalization.mojo     # BatchNorm, LayerNorm classes (NEW)
│   └── utils/
│       └── __init__.mojo          # Utility functions
├── training/
│   └── optimizers/
│       ├── __init__.mojo          # Export all optimizer classes
│       ├── sgd.mojo               # SGD class wrapping sgd_step function
│       ├── adam.mojo              # Adam class (NEW)
│       ├── adamw.mojo             # AdamW class (NEW)
│       └── rmsprop.mojo           # RMSprop class (NEW)
├── data/                          # Already implemented (needs ExTensor import fix)
├── training/loops/                # Already implemented
├── training/metrics/              # Already implemented
└── utils/                         # Already implemented
```text

## Design Principles

### 1. ExTensor as the Universal Type

- **No Tensor alias** - everything uses ExTensor directly
- Scalars represented as 0-D ExTensors when needed
- All functional operations signature: `fn op(ExTensor, ...) -> ExTensor`
- Type-generic through ExTensor's built-in dtype support

### 2. Functional Layer Composability

- Pure functions with no side effects (except optimizer state updates)
- Functions work with any ExTensor dtype that makes sense for the operation
- Broadcasting follows Array API Standard
- All operations in `shared/core/ops/` are functional building blocks

### 3. Class Layer Responsibilities

- **State management**: Hold weights, biases, optimizer state
- **Initialization**: Initialize parameters using functional initializers
- **Forward pass**: Compose functional operations
- **Parameter management**: Track trainable parameters
- **Serialization**: Save/load state (future)

### 4. Clear Separation of Concerns

```mojo
# Level 1: Functional (in shared/core/ops/)
fn relu(x: ExTensor) -> ExTensor:
    """Pure function applying ReLU activation."""
    return max_reduce(x, 0.0)  # Functional composition

# Level 2: Class-based (in shared/core/layers/)
struct ReLU:
    """Stateless layer wrapper for ReLU activation."""

    fn forward(self, x: ExTensor) -> ExTensor:
        """Forward pass delegates to functional relu."""
        return relu(x)  # Delegates to functional layer
```text

## Implementation Tasks

### Phase 1: Migrate ExTensor to shared/core/types/

**Files to move from `src/extensor/` to `shared/core/types/`:**

1. `extensor.mojo` → `shared/core/types/extensor.mojo`
1. `shape.mojo` → `shared/core/types/shape.mojo`

### Update `shared/core/types/__init__.mojo`:

```mojo
"""Core types module - ExTensor and shape utilities."""

# Export ExTensor type
from .extensor import ExTensor

# Export creation functions
from .extensor import zeros, ones, full, empty, arange, eye, linspace
from .extensor import zeros_like, ones_like, full_like

# Export shape utilities
from .shape import reshape, squeeze, unsqueeze, expand_dims, flatten, ravel
from .shape import concatenate, stack

__all__ = [
    "ExTensor",
    "zeros", "ones", "full", "empty", "arange", "eye", "linspace",
    "zeros_like", "ones_like", "full_like",
    "reshape", "squeeze", "unsqueeze", "expand_dims", "flatten", "ravel",
    "concatenate", "stack"
]
```text

### Phase 2: Migrate Operations to shared/core/ops/

**Files to move from `src/extensor/` to `shared/core/ops/`:**

1. `arithmetic.mojo` → `shared/core/ops/arithmetic.mojo`
1. `matrix.mojo` → `shared/core/ops/matrix.mojo`
1. `reduction.mojo` → `shared/core/ops/reduction.mojo`
1. `elementwise_math.mojo` → `shared/core/ops/elementwise.mojo`
1. `comparison.mojo` → `shared/core/ops/comparison.mojo`
1. `broadcasting.mojo` → `shared/core/ops/broadcasting.mojo`
1. `activations.mojo` → `shared/core/ops/activations.mojo`
1. `losses.mojo` → `shared/core/ops/losses.mojo`
1. `initializers.mojo` → `shared/core/ops/initializers.mojo`

### Update `shared/core/ops/__init__.mojo`:

```mojo
"""Core operations module - Functional tensor operations."""

from .arithmetic import add, subtract, multiply, divide, floor_divide, modulo, power
from .arithmetic import add_backward, subtract_backward, multiply_backward, divide_backward

from .matrix import matmul, transpose, dot, outer
from .matrix import matmul_backward, transpose_backward

from .reduction import sum, mean, max_reduce, min_reduce
from .reduction import sum_backward, mean_backward, max_reduce_backward, min_reduce_backward

from .elementwise import abs, sign, exp, log, sqrt, sin, cos, tanh, clip
from .elementwise import ceil, floor, round, trunc
from .elementwise import logical_and, logical_or, logical_not, logical_xor
from .elementwise import exp_backward, log_backward, sqrt_backward, clip_backward

from .comparison import equal, not_equal, less, less_equal, greater, greater_equal

from .broadcasting import broadcast_shapes, are_shapes_broadcastable

from .activations import relu, leaky_relu, prelu, sigmoid, softmax, gelu, selu, elu
from .activations import relu_backward, leaky_relu_backward, sigmoid_backward

from .losses import mse_loss, cross_entropy_loss, binary_cross_entropy_loss
from .losses import mse_loss_backward, cross_entropy_backward

from .initializers import xavier_uniform, xavier_normal, he_uniform, he_normal
from .initializers import uniform, normal

__all__ = [
    # Arithmetic
    "add", "subtract", "multiply", "divide", "floor_divide", "modulo", "power",
    # Matrix
    "matmul", "transpose", "dot", "outer",
    # Reduction
    "sum", "mean", "max_reduce", "min_reduce",
    # Element-wise
    "abs", "sign", "exp", "log", "sqrt", "sin", "cos", "tanh", "clip",
    # Comparison
    "equal", "not_equal", "less", "less_equal", "greater", "greater_equal",
    # Broadcasting
    "broadcast_shapes", "are_shapes_broadcastable",
    # Activations
    "relu", "leaky_relu", "prelu", "sigmoid", "softmax", "gelu", "selu", "elu",
    # Losses
    "mse_loss", "cross_entropy_loss", "binary_cross_entropy_loss",
    # Initializers
    "xavier_uniform", "xavier_normal", "he_uniform", "he_normal", "uniform", "normal"
]
```text

### Phase 3: Implement Layer Classes (shared/core/layers/)

### Create `shared/core/layers/linear.mojo`:

```mojo
"""Linear (fully connected) layer implementation."""

from shared.core.types import ExTensor, zeros
from shared.core.ops import matmul, add, xavier_uniform

struct Linear:
    """Fully connected (dense) layer.

    Computes: output = input @ weights.T + bias

    Attributes:
        in_features: Number of input features
        out_features: Number of output features
        weights: Weight matrix (out_features, in_features)
        bias: Bias vector (out_features) - optional
    """

    var in_features: Int
    var out_features: Int
    var weights: ExTensor
    var bias: Optional[ExTensor]

    fn __init__(inout self, in_features: Int, out_features: Int, use_bias: Bool = True):
        """Initialize Linear layer with Xavier initialization.

        Args:
            in_features: Number of input features
            out_features: Number of output features
            use_bias: Whether to include bias term
        """
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weights using Xavier uniform
        self.weights = xavier_uniform(out_features, in_features, DType.float32)

        # Initialize bias if requested
        if use_bias:
            self.bias = zeros(out_features, DType.float32)
        else:
            self.bias = None

    fn forward(self, x: ExTensor) raises -> ExTensor:
        """Forward pass: output = x @ W.T + b

        Args:
            x: Input tensor of shape (batch_size, in_features)

        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        # Matrix multiplication: x @ W.T
        var output = matmul(x, transpose(self.weights))

        # Add bias if present
        if self.bias:
            output = add(output, self.bias.value())

        return output

    fn parameters(self) -> List[ExTensor]:
        """Get list of trainable parameters.

        Returns:
            List containing [weights] or [weights, bias]
        """
        var params = List[ExTensor]()
        params.append(self.weights)
        if self.bias:
            params.append(self.bias.value())
        return params
```text

### Create `shared/core/layers/activation.mojo`:

```mojo
"""Activation layer implementations."""

from shared.core.types import ExTensor
from shared.core.ops import relu, sigmoid, tanh as tanh_op

struct ReLU:
    """ReLU activation layer: f(x) = max(0, x)"""

    fn forward(self, x: ExTensor) -> ExTensor:
        """Apply ReLU activation."""
        return relu(x)

struct Sigmoid:
    """Sigmoid activation layer: f(x) = 1 / (1 + exp(-x))"""

    fn forward(self, x: ExTensor) -> ExTensor:
        """Apply sigmoid activation."""
        return sigmoid(x)

struct Tanh:
    """Tanh activation layer: f(x) = tanh(x)"""

    fn forward(self, x: ExTensor) -> ExTensor:
        """Apply tanh activation."""
        return tanh_op(x)
```text

### Create `shared/core/layers/conv.mojo`:

```mojo
"""Convolutional layer implementations."""

from shared.core.types import ExTensor, zeros
from shared.core.ops import he_uniform

struct Conv2D:
    """2D Convolutional layer.

    Attributes:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of convolution kernel
        stride: Stride of convolution
        padding: Padding applied to input
        weights: Convolution kernels (out_channels, in_channels, kernel_size, kernel_size)
        bias: Bias per output channel (out_channels) - optional
    """

    var in_channels: Int
    var out_channels: Int
    var kernel_size: Int
    var stride: Int
    var padding: Int
    var weights: ExTensor
    var bias: Optional[ExTensor]

    fn __init__(
        inout self,
        in_channels: Int,
        out_channels: Int,
        kernel_size: Int,
        stride: Int = 1,
        padding: Int = 0,
        use_bias: Bool = True
    ):
        """Initialize Conv2D layer.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of square convolution kernel
            stride: Stride for convolution
            padding: Zero-padding added to input
            use_bias: Whether to include bias term
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Initialize weights using He initialization
        var shape = DynamicVector[Int](4)
        shape[0] = out_channels
        shape[1] = in_channels
        shape[2] = kernel_size
        shape[3] = kernel_size
        self.weights = he_uniform(shape, DType.float32)

        if use_bias:
            self.bias = zeros(out_channels, DType.float32)
        else:
            self.bias = None

    fn forward(self, x: ExTensor) raises -> ExTensor:
        """Forward pass: 2D convolution.

        Args:
            x: Input tensor of shape (batch, in_channels, height, width)

        Returns:
            Output tensor of shape (batch, out_channels, out_height, out_width)

        Note:
            This is a placeholder. Full convolution implementation requires
            im2col or direct convolution loop with SIMD optimization.
        """
        # TODO: Implement actual convolution using functional ops
        raise Error("Conv2D forward pass not yet implemented")

    fn parameters(self) -> List[ExTensor]:
        """Get list of trainable parameters."""
        var params = List[ExTensor]()
        params.append(self.weights)
        if self.bias:
            params.append(self.bias.value())
        return params
```text

### Update `shared/core/layers/__init__.mojo`:

```mojo
"""Neural network layers module."""

from .linear import Linear
from .conv import Conv2D
from .activation import ReLU, Sigmoid, Tanh

__all__ = [
    "Linear",
    "Conv2D",
    "ReLU",
    "Sigmoid",
    "Tanh"
]
```text

### Phase 4: Implement Optimizer Classes (shared/training/optimizers/)

### Update `shared/training/optimizers/sgd.mojo`:

```mojo
"""SGD optimizer - both functional and class-based APIs."""

from shared.core.types import ExTensor, zeros_like, full_like
from shared.core.ops import subtract, multiply, add

# ============================================================================
# Functional API (Level 1)
# ============================================================================

fn sgd_step(
    params: ExTensor,
    gradients: ExTensor,
    learning_rate: Float64,
    momentum: Float64 = 0.0,
    weight_decay: Float64 = 0.0,
    velocity: ExTensor = ExTensor()
) raises -> ExTensor:
    """Functional SGD update (existing implementation).

    Pure function that computes parameter update without state.
    """
    # ... existing implementation ...

# ============================================================================
# Class-Based API (Level 2)
# ============================================================================

struct SGD:
    """SGD optimizer with momentum and weight decay.

    Wraps functional sgd_step with state management for velocity buffers.

    Attributes:
        learning_rate: Learning rate (step size)
        momentum: Momentum factor (0 = no momentum)
        dampening: Dampening for momentum
        weight_decay: L2 regularization factor
        nesterov: Whether to use Nesterov momentum
        velocity_buffers: Momentum buffers for each parameter
    """

    var learning_rate: Float64
    var momentum: Float64
    var dampening: Float64
    var weight_decay: Float64
    var nesterov: Bool
    var velocity_buffers: Dict[Int, ExTensor]  # param_id -> velocity

    fn __init__(
        inout self,
        learning_rate: Float64 = 0.01,
        momentum: Float64 = 0.0,
        dampening: Float64 = 0.0,
        weight_decay: Float64 = 0.0,
        nesterov: Bool = False
    ):
        """Initialize SGD optimizer.

        Args:
            learning_rate: Learning rate
            momentum: Momentum factor
            dampening: Dampening for momentum
            weight_decay: Weight decay (L2 penalty)
            nesterov: Enable Nesterov momentum
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.velocity_buffers = Dict[Int, ExTensor]()

    fn step(inout self, inout params: ExTensor, grads: ExTensor, param_id: Int = 0) raises:
        """Update parameters using SGD.

        Args:
            params: Parameters to update (modified in-place)
            grads: Gradients
            param_id: Identifier for this parameter (for velocity tracking)
        """
        # Get or create velocity buffer
        var velocity: ExTensor
        if param_id in self.velocity_buffers:
            velocity = self.velocity_buffers[param_id]
        else:
            velocity = zeros_like(params)

        # Call functional SGD step
        var updated_params = sgd_step(
            params,
            grads,
            self.learning_rate,
            self.momentum,
            self.weight_decay,
            velocity
        )

        # Update parameters in-place
        params = updated_params

        # Store updated velocity
        if self.momentum > 0.0:
            self.velocity_buffers[param_id] = velocity

    fn zero_grad(inout self):
        """Clear velocity buffers (if needed)."""
        # Velocity persists across steps, so this is typically not needed
        pass
```text

### Create `shared/training/optimizers/adam.mojo`:

```mojo
"""Adam optimizer implementation."""

from shared.core.types import ExTensor, zeros_like
from shared.core.ops import add, multiply, divide, subtract, sqrt, power

struct Adam:
    """Adam optimizer (Adaptive Moment Estimation).

    Combines momentum (first moment) and RMSprop (second moment)
    with bias correction for early training stability.

    Attributes:
        learning_rate: Learning rate
        beta1: Exponential decay rate for first moment
        beta2: Exponential decay rate for second moment
        epsilon: Small constant for numerical stability
        m_buffers: First moment estimates (momentum)
        v_buffers: Second moment estimates (RMSprop)
        t: Time step counter
    """

    var learning_rate: Float64
    var beta1: Float64
    var beta2: Float64
    var epsilon: Float64
    var m_buffers: Dict[Int, ExTensor]
    var v_buffers: Dict[Int, ExTensor]
    var t: Int

    fn __init__(
        inout self,
        learning_rate: Float64 = 0.001,
        beta1: Float64 = 0.9,
        beta2: Float64 = 0.999,
        epsilon: Float64 = 1e-8
    ):
        """Initialize Adam optimizer."""
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_buffers = Dict[Int, ExTensor]()
        self.v_buffers = Dict[Int, ExTensor]()
        self.t = 0

    fn step(inout self, inout params: ExTensor, grads: ExTensor, param_id: Int = 0) raises:
        """Update parameters using Adam.

        Args:
            params: Parameters to update
            grads: Gradients
            param_id: Parameter identifier
        """
        self.t += 1

        # Get or create moment buffers
        var m = self.m_buffers.get(param_id, zeros_like(params))
        var v = self.v_buffers.get(param_id, zeros_like(params))

        # Update biased first moment: m = beta1 * m + (1 - beta1) * grad
        var beta1_tensor = full_like(m, self.beta1)
        var one_minus_beta1 = full_like(grads, 1.0 - self.beta1)
        m = add(multiply(beta1_tensor, m), multiply(one_minus_beta1, grads))

        # Update biased second moment: v = beta2 * v + (1 - beta2) * grad^2
        var beta2_tensor = full_like(v, self.beta2)
        var one_minus_beta2 = full_like(grads, 1.0 - self.beta2)
        var grad_squared = multiply(grads, grads)
        v = add(multiply(beta2_tensor, v), multiply(one_minus_beta2, grad_squared))

        # Bias correction
        var bias_correction1 = 1.0 - pow(self.beta1, Float64(self.t))
        var bias_correction2 = 1.0 - pow(self.beta2, Float64(self.t))

        # Corrected moments
        var m_hat = divide(m, full_like(m, bias_correction1))
        var v_hat = divide(v, full_like(v, bias_correction2))

        # Update: params = params - lr * m_hat / (sqrt(v_hat) + epsilon)
        var denominator = add(sqrt(v_hat), full_like(v_hat, self.epsilon))
        var update = multiply(full_like(params, self.learning_rate), divide(m_hat, denominator))
        params = subtract(params, update)

        # Store updated moments
        self.m_buffers[param_id] = m
        self.v_buffers[param_id] = v
```text

**Update `shared/training/optimizers/__init__.mojo`:**

```mojo
"""Optimizers module - both functional and class-based APIs."""

# Functional API
from .sgd import sgd_step, sgd_step_simple

# Class-based API
from .sgd import SGD
from .adam import Adam

__all__ = [
    # Functional
    "sgd_step", "sgd_step_simple",
    # Class-based
    "SGD", "Adam"
]
```text

### Phase 5: Fix Import Paths

**Files to update (replace `from tensor import Tensor` with `from shared.core.types import ExTensor`):**

1. `shared/data/datasets.mojo`
1. `shared/data/transforms.mojo`
1. `shared/data/loaders.mojo`
1. `shared/data/generic_transforms.mojo`

### Example fix for `shared/data/datasets.mojo`:

```mojo
# OLD
from tensor import Tensor

# NEW
from shared.core.types import ExTensor
```text

### Phase 6: Update Core Module Exports

### Update `shared/core/__init__.mojo`:

```mojo
"""Core library module - Types, operations, and layers."""

# Export types
from .types import ExTensor
from .types import zeros, ones, full, empty, arange, eye, linspace
from .types import zeros_like, ones_like, full_like

# Export operations
from .ops import matmul, transpose, add, subtract, multiply, divide
from .ops import relu, sigmoid, tanh, softmax
from .ops import mse_loss, cross_entropy_loss
from .ops import xavier_uniform, he_uniform

# Export layers
from .layers import Linear, Conv2D, ReLU, Sigmoid, Tanh

__all__ = [
    # Types
    "ExTensor",
    # Creation functions
    "zeros", "ones", "full", "empty", "arange", "eye", "linspace",
    "zeros_like", "ones_like", "full_like",
    # Operations
    "matmul", "transpose", "add", "subtract", "multiply", "divide",
    "relu", "sigmoid", "tanh", "softmax",
    "mse_loss", "cross_entropy_loss",
    "xavier_uniform", "he_uniform",
    # Layers
    "Linear", "Conv2D", "ReLU", "Sigmoid", "Tanh"
]
```text

## Testing Strategy

### Update Test Files

### Update test imports:

```mojo
# OLD (what tests currently expect)
from tests.shared.conftest import Tensor, Shape

# NEW (after migration)
from shared.core.types import ExTensor
from shared.core.layers import Linear, Conv2D, ReLU, Sigmoid, Tanh
from shared.training.optimizers import SGD, Adam, AdamW, RMSprop
```text

### Example test update:

```mojo
# tests/shared/core/test_layers.mojo

fn test_linear_initialization() raises:
    """Test Linear layer initialization."""
    var layer = Linear(in_features=10, out_features=5, use_bias=True)
    assert_equal(layer.in_features, 10)
    assert_equal(layer.out_features, 5)
    # Check weight shape
    var weight_shape = layer.weights.shape()
    assert_equal(weight_shape[0], 5)   # out_features
    assert_equal(weight_shape[1], 10)  # in_features

fn test_linear_forward() raises:
    """Test Linear layer forward pass."""
    var layer = Linear(10, 5)

    # Create input: batch_size=2, in_features=10
    var input = ones(DynamicVector[Int](2, 10), DType.float32)

    # Forward pass
    var output = layer.forward(input)

    # Check output shape: (2, 5)
    var output_shape = output.shape()
    assert_equal(output_shape[0], 2)
    assert_equal(output_shape[1], 5)
```text

## Migration Checklist

- [ ] Phase 1: Migrate ExTensor to `shared/core/types/`
  - [ ] Move `src/extensor/extensor.mojo` → `shared/core/types/extensor.mojo`
  - [ ] Move `src/extensor/shape.mojo` → `shared/core/types/shape.mojo`
  - [ ] Update `shared/core/types/__init__.mojo`
  - [ ] Test ExTensor imports work

- [ ] Phase 2: Migrate operations to `shared/core/ops/`
  - [ ] Move arithmetic, matrix, reduction, elementwise, etc.
  - [ ] Update `shared/core/ops/__init__.mojo`
  - [ ] Test operation imports work

- [ ] Phase 3: Implement layer classes
  - [ ] Create `shared/core/layers/linear.mojo`
  - [ ] Create `shared/core/layers/activation.mojo`
  - [ ] Create `shared/core/layers/conv.mojo` (placeholder)
  - [ ] Update `shared/core/layers/__init__.mojo`
  - [ ] Test layer instantiation

- [ ] Phase 4: Implement optimizer classes
  - [ ] Update `shared/training/optimizers/sgd.mojo` (add class)
  - [ ] Create `shared/training/optimizers/adam.mojo`
  - [ ] Create `shared/training/optimizers/adamw.mojo`
  - [ ] Create `shared/training/optimizers/rmsprop.mojo`
  - [ ] Update `shared/training/optimizers/__init__.mojo`

- [ ] Phase 5: Fix import paths
  - [ ] Update `shared/data/datasets.mojo`
  - [ ] Update `shared/data/transforms.mojo`
  - [ ] Update `shared/data/loaders.mojo`
  - [ ] Update `shared/data/generic_transforms.mojo`

- [ ] Phase 6: Update test files
  - [ ] Uncomment test code in `test_layers.mojo`
  - [ ] Uncomment test code in `test_optimizers.mojo`
  - [ ] Update test imports to use ExTensor
  - [ ] Run tests and fix remaining issues

- [ ] Phase 7: Clean up
  - [ ] Remove `src/extensor/` directory
  - [ ] Update all documentation
  - [ ] Verify all imports work
  - [ ] Run full test suite

## Success Criteria

1. ✅ All ExTensor code moved from `src/extensor/` to `shared/core/`
1. ✅ Clear separation: functional ops in `ops/`, classes in `layers/` and `optimizers/`
1. ✅ No `Tensor` alias - everything uses `ExTensor`
1. ✅ Layer classes (Linear, Conv2D, ReLU, etc.) implemented
1. ✅ Optimizer classes (SGD, Adam, AdamW, RMSprop) implemented
1. ✅ All test files compile and can be uncommented
1. ✅ Clean two-level architecture: functional core + class wrapper
1. ✅ All imports work correctly
1. ✅ No code duplication between functional and class layers

## Expected Outcome

After completing this architecture redesign:

- **Functional Layer**: Pure, composable operations in `shared/core/ops/`
- **Class Layer**: Stateful wrappers in `shared/core/layers/` and `shared/training/optimizers/`
- **Type System**: ExTensor in `shared/core/types/` (no aliases)
- **Clean Hierarchy**: Functions compose into classes, classes compose into models
- **Test Compatibility**: All test stubs can be uncommented and will work
- **No src/extensor**: Everything consolidated in `shared/`

This architecture provides:

- **Flexibility**: Use functional APIs for custom operations
- **Convenience**: Use class APIs for standard layers
- **Composability**: Build complex models from simple building blocks
- **Type Safety**: ExTensor ensures consistent typing throughout
- **Maintainability**: Clear separation of concerns
