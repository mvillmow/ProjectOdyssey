# Prompt: Fix ML Odyssey Architecture - Pure Functional Design

## Context

The ML Odyssey repository has a design mismatch:

- Implementation provides functional APIs (`relu()`, `sgd_step()`) - this is CORRECT
- Tensor implementation (`ExTensor`) is in wrong location (`src/extensor/` instead of `shared/core/`)
- Tests expect class-based APIs, but we'll fix tests later - keep everything functional

## Objective

Redesign the `shared/` directory to implement a **pure functional architecture**:

**Everything is a pure function** - No classes, no internal state, caller manages all state

## Requirements

### 1. No Tensor Alias

- Everything uses `ExTensor` directly (no `Tensor` → `ExTensor` aliases)
- ExTensor is the universal type throughout the codebase
- Scalars represented as 0-D ExTensors when needed

### 2. Directory Structure

```text
shared/
├── core/
│   ├── extensor.mojo       # ExTensor type + creation (from src/extensor/)
│   ├── arithmetic.mojo     # add, subtract, multiply, divide
│   ├── matrix.mojo         # matmul, transpose, dot
│   ├── reduction.mojo      # sum, mean, max, min
│   ├── activation.mojo     # relu, sigmoid, tanh, softmax
│   ├── linear.mojo         # linear(x, w, b) function
│   ├── conv.mojo           # conv2d(x, kernel, ...) function
│   ├── pooling.mojo        # maxpool2d, avgpool2d functions
│   ├── loss.mojo           # mse_loss, cross_entropy
│   └── initializers.mojo   # xavier_uniform, he_uniform
├── training/
│   └── optimizers.mojo     # sgd_step, adam_step (return new state)
├── data/               # Fix imports: tensor.Tensor → ExTensor
└── ...
```text

### 3. Pure Functional Design

### Everything is a function

- Pure functions: `fn relu(x: ExTensor) -> ExTensor`
- Stateless, composable, type-generic
- Caller manages ALL state (weights, momentum buffers, etc.)
- Functions return new state, never mutate

### 4. Migration Tasks

### Move from `src/extensor/` to `shared/core/`:

| Source File | Destination | Notes |
|------------|-------------|-------|
| `extensor.mojo` | `shared/core/extensor.mojo` | Core type + creation functions |
| `shape.mojo` | Merge into `extensor.mojo` | Shape utilities |
| `arithmetic.mojo` | `shared/core/arithmetic.mojo` | Pure functions |
| `matrix.mojo` | `shared/core/matrix.mojo` | Pure functions |
| `reduction.mojo` | `shared/core/reduction.mojo` | Pure functions |
| `elementwise_math.mojo` | `shared/core/elementwise.mojo` | Pure functions |
| `activations.mojo` | `shared/core/activation.mojo` | Pure functions |
| `losses.mojo` | `shared/core/loss.mojo` | Pure functions |
| `initializers.mojo` | `shared/core/initializers.mojo` | Pure functions |
| `comparison.mojo` | `shared/core/comparison.mojo` | Pure functions |
| `broadcasting.mojo` | `shared/core/broadcasting.mojo` | Pure functions |

### Create new functional operations:

| File | Purpose | Signature |
|------|---------|-----------|
| `shared/core/linear.mojo` | Linear transform | `fn linear(x, w, b) -> ExTensor` |
| `shared/core/conv.mojo` | 2D convolution | `fn conv2d(x, kernel, ...) -> ExTensor` |
| `shared/core/pooling.mojo` | Pooling ops | `fn maxpool2d(x, ...) -> ExTensor` |
| `shared/training/optimizers.mojo` | Update SGD | `fn sgd_step(...) -> (params, velocity)` |

### Fix imports in:

- `shared/data/datasets.mojo` - Change `from tensor import Tensor` → `from shared.core.types import ExTensor`
- `shared/data/transforms.mojo` - Same
- `shared/data/loaders.mojo` - Same
- `shared/data/generic_transforms.mojo` - Same

## Implementation Examples

### Linear Function (Pure Functional)

```mojo
# shared/core/linear.mojo
from shared.core.extensor import ExTensor
from shared.core.matrix import matmul, transpose
from shared.core.arithmetic import add

fn linear(x: ExTensor, weights: ExTensor, bias: ExTensor) raises -> ExTensor:
    """Functional linear transformation: x @ W.T + b

    Caller manages weights and bias - function is stateless.

    Args:
        x: Input (batch_size, in_features)
        weights: Weight matrix (out_features, in_features)
        bias: Bias vector (out_features)

    Returns:
        Output (batch_size, out_features)
    """
    var out = matmul(x, transpose(weights))
    return add(out, bias)
```text

### SGD Function (Pure Functional)

```mojo
# shared/training/optimizers.mojo
from shared.core.extensor import ExTensor, full_like
from shared.core.arithmetic import add, subtract, multiply

fn sgd_step(
    params: ExTensor,
    grads: ExTensor,
    velocity: ExTensor,
    lr: Float64,
    momentum: Float64
) raises -> Tuple[ExTensor, ExTensor]:
    """Functional SGD update - returns new state.

    Caller manages velocity buffer - function is stateless.

    Args:
        params: Current parameters
        grads: Gradients
        velocity: Momentum buffer
        lr: Learning rate
        momentum: Momentum coefficient

    Returns:
        (new_params, new_velocity)
    """
    # Update velocity: v = momentum * v + grad
    var new_velocity = add(
        multiply(full_like(velocity, momentum), velocity),
        grads
    )

    # Update params: p = p - lr * v
    var new_params = subtract(
        params,
        multiply(full_like(new_velocity, lr), new_velocity)
    )

    return (new_params, new_velocity)
```text

### Usage Example (Caller Manages State)

```mojo
from shared.core import ExTensor, zeros, xavier_uniform, linear, relu
from shared.training.optimizers import sgd_step

# Caller initializes and manages ALL state
var w1 = xavier_uniform(128, 784, DType.float32)
var b1 = zeros(128, DType.float32)
var w2 = xavier_uniform(10, 128, DType.float32)
var b2 = zeros(10, DType.float32)

# Optimizer state (caller manages)
var w1_vel = zeros_like(w1)
var b1_vel = zeros_like(b1)
var w2_vel = zeros_like(w2)
var b2_vel = zeros_like(b2)

# Training loop
for epoch in range(100):
    # Forward pass - pure functions
    var h = linear(x_train, w1, b1)
    h = relu(h)
    var y_pred = linear(h, w2, b2)

    # Backward pass (compute gradients)
    var grad_w1, grad_b1, grad_w2, grad_b2 = compute_grads(...)

    # Optimizer step - returns new state
    (w1, w1_vel) = sgd_step(w1, grad_w1, w1_vel, lr=0.01, momentum=0.9)
    (b1, b1_vel) = sgd_step(b1, grad_b1, b1_vel, lr=0.01, momentum=0.9)
    (w2, w2_vel) = sgd_step(w2, grad_w2, w2_vel, lr=0.01, momentum=0.9)
    (b2, b2_vel) = sgd_step(b2, grad_b2, b2_vel, lr=0.01, momentum=0.9)
```text

## Execution Steps

1. **Migrate Core Type** (Phase 1):
   - Move `src/extensor/extensor.mojo` → `shared/core/extensor.mojo`
   - Merge `src/extensor/shape.mojo` into `extensor.mojo` (or keep separate)

1. **Migrate Operations** (Phase 2):
   - Move `src/extensor/arithmetic.mojo` → `shared/core/arithmetic.mojo`
   - Move `src/extensor/matrix.mojo` → `shared/core/matrix.mojo`
   - Move `src/extensor/reduction.mojo` → `shared/core/reduction.mojo`
   - Move `src/extensor/elementwise_math.mojo` → `shared/core/elementwise.mojo`
   - Move `src/extensor/activations.mojo` → `shared/core/activation.mojo`
   - Move `src/extensor/losses.mojo` → `shared/core/loss.mojo`
   - Move `src/extensor/initializers.mojo` → `shared/core/initializers.mojo`
   - Move `src/extensor/comparison.mojo` → `shared/core/comparison.mojo`
   - Move `src/extensor/broadcasting.mojo` → `shared/core/broadcasting.mojo`

1. **Create New Functional Operations** (Phase 3):
   - Create `shared/core/linear.mojo` with `fn linear(x, w, b) -> ExTensor`
   - Create `shared/core/conv.mojo` with `fn conv2d(x, kernel, ...) -> ExTensor` (stub for now)
   - Create `shared/core/pooling.mojo` with `fn maxpool2d(x, ...)` and `fn avgpool2d(x, ...)`

1. **Update Optimizers to Functional** (Phase 4):
   - Update `shared/training/optimizers/sgd.mojo` signature to return `(params, velocity)`
   - Create `shared/training/optimizers/__init__.mojo` exporting `sgd_step`, `adam_step`, etc.

1. **Fix Imports** (Phase 5):
   - Replace `from tensor import Tensor` → `from shared.core.extensor import ExTensor`:
     - `shared/data/datasets.mojo`
     - `shared/data/transforms.mojo`
     - `shared/data/loaders.mojo`
     - `shared/data/generic_transforms.mojo`
   - Update `from extensor import ExTensor` → `from shared.core.extensor import ExTensor`:
     - `shared/training/loops/training_loop.mojo`
     - `shared/training/loops/validation_loop.mojo`

1. **Update Module Exports** (Phase 6):
   - Create `shared/core/__init__.mojo` exporting all functions
   - Create `shared/training/__init__.mojo` exporting optimizer functions
   - Verify all imports work

1. **Clean Up** (Phase 7):
   - Delete `src/extensor/` directory entirely
   - Update documentation to reflect pure functional architecture
   - Tests will be updated later - keep them commented for now

## Success Criteria

After completion:

- [ ] `src/extensor/` deleted (all code moved to `shared/core/`)
- [ ] No `Tensor` aliases anywhere (everything uses `ExTensor`)
- [ ] All operations in `shared/core/` as pure functions
- [ ] NO classes - everything is functional
- [ ] All imports fixed (no broken `from tensor import Tensor` or `from extensor import`)
- [ ] Functional operations created: `linear()`, `conv2d()`, `maxpool2d()`, `avgpool2d()`
- [ ] Optimizers return new state: `fn sgd_step(...) -> (params, velocity)`
- [ ] All code compiles
- [ ] Pure functional architecture throughout

## Design Principles to Follow

1. **Pure Functions**: All operations are stateless, return new values
1. **Composability**: Functions work with any ExTensor dtype
1. **Caller Manages State**: Functions don't hold weights, momentum, etc.
1. **Explicit Data Flow**: All state passed in, all changes returned
1. **No Side Effects**: Functions don't mutate inputs (except where explicitly documented)

## Focus

**Only work in `shared/` directory** - migrate from `src/extensor/` and implement missing functional operations. Do NOT modify test files until the shared library is complete - tests will be updated later to use functional API.
