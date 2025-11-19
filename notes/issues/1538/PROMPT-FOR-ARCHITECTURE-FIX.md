# Prompt: Fix ML Odyssey Architecture - Two-Level Functional + Class API

## Context

The ML Odyssey repository has a design mismatch:
- Tests expect class-based APIs (`Linear()`, `SGD()`, `Tensor()`)
- Implementation provides functional APIs (`relu()`, `sgd_step()`)
- Tensor implementation (`ExTensor`) is in wrong location (`src/extensor/` instead of `shared/core/`)

## Objective

Redesign the `shared/` directory to implement a clean two-level architecture:

**Level 1: Functional Layer** - Pure functions taking only ExTensors
**Level 2: Class Layer** - Stateful wrappers composing functional operations

## Requirements

### 1. No Tensor Alias
- Everything uses `ExTensor` directly (no `Tensor` → `ExTensor` aliases)
- ExTensor is the universal type throughout the codebase
- Scalars represented as 0-D ExTensors when needed

### 2. Directory Structure

```
shared/
├── core/
│   ├── types/          # ExTensor, shape utilities (from src/extensor/)
│   ├── ops/            # Pure functions (from src/extensor/)
│   └── layers/         # Layer classes (NEW - compose ops)
├── training/
│   └── optimizers/     # Optimizer classes (NEW - wrap functional APIs)
├── data/               # Fix imports: tensor.Tensor → ExTensor
└── ...
```

### 3. Two-Level Hierarchy

**Level 1 (Functional)** - `shared/core/ops/`:
- Pure functions: `fn relu(x: ExTensor) -> ExTensor`
- Stateless, composable, type-generic
- Examples: `matmul()`, `relu()`, `sigmoid()`, `sgd_step()`

**Level 2 (Classes)** - `shared/core/layers/` and `shared/training/optimizers/`:
- Stateful wrappers with initialization
- Examples: `Linear`, `Conv2D`, `ReLU`, `SGD`, `Adam`
- Compose functional operations

### 4. Migration Tasks

**Move from `src/extensor/` to `shared/core/`:**

| Source File | Destination | Category |
|------------|-------------|----------|
| `extensor.mojo` | `shared/core/types/extensor.mojo` | Core type |
| `shape.mojo` | `shared/core/types/shape.mojo` | Type utils |
| `arithmetic.mojo` | `shared/core/ops/arithmetic.mojo` | Operations |
| `matrix.mojo` | `shared/core/ops/matrix.mojo` | Operations |
| `reduction.mojo` | `shared/core/ops/reduction.mojo` | Operations |
| `elementwise_math.mojo` | `shared/core/ops/elementwise.mojo` | Operations |
| `activations.mojo` | `shared/core/ops/activations.mojo` | Operations |
| `losses.mojo` | `shared/core/ops/losses.mojo` | Operations |
| `initializers.mojo` | `shared/core/ops/initializers.mojo` | Operations |
| `comparison.mojo` | `shared/core/ops/comparison.mojo` | Operations |
| `broadcasting.mojo` | `shared/core/ops/broadcasting.mojo` | Operations |

**Create new class-based wrappers:**

| File | Purpose | Wraps |
|------|---------|-------|
| `shared/core/layers/linear.mojo` | Linear layer | `matmul()`, `add()` |
| `shared/core/layers/conv.mojo` | Conv2D layer | Convolution ops |
| `shared/core/layers/activation.mojo` | ReLU/Sigmoid/Tanh | `relu()`, `sigmoid()`, `tanh()` |
| `shared/training/optimizers/sgd.mojo` | SGD class | `sgd_step()` function |
| `shared/training/optimizers/adam.mojo` | Adam class | Adam functional ops |

**Fix imports in:**
- `shared/data/datasets.mojo` - Change `from tensor import Tensor` → `from shared.core.types import ExTensor`
- `shared/data/transforms.mojo` - Same
- `shared/data/loaders.mojo` - Same
- `shared/data/generic_transforms.mojo` - Same

## Implementation Example

### Layer Class Template

```mojo
# shared/core/layers/linear.mojo
from shared.core.types import ExTensor, zeros
from shared.core.ops import matmul, add, transpose, xavier_uniform

struct Linear:
    var in_features: Int
    var out_features: Int
    var weights: ExTensor
    var bias: Optional[ExTensor]

    fn __init__(inout self, in_features: Int, out_features: Int, use_bias: Bool = True):
        self.in_features = in_features
        self.out_features = out_features
        self.weights = xavier_uniform(out_features, in_features, DType.float32)
        self.bias = zeros(out_features, DType.float32) if use_bias else None

    fn forward(self, x: ExTensor) raises -> ExTensor:
        var output = matmul(x, transpose(self.weights))  # Functional composition
        if self.bias:
            output = add(output, self.bias.value())
        return output
```

### Optimizer Class Template

```mojo
# shared/training/optimizers/sgd.mojo
from shared.core.types import ExTensor, zeros_like
from shared.training.optimizers.functional import sgd_step  # Functional API

struct SGD:
    var learning_rate: Float64
    var momentum: Float64
    var velocity_buffers: Dict[Int, ExTensor]

    fn __init__(inout self, learning_rate: Float64 = 0.01, momentum: Float64 = 0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity_buffers = Dict[Int, ExTensor]()

    fn step(inout self, inout params: ExTensor, grads: ExTensor, param_id: Int = 0) raises:
        var velocity = self.velocity_buffers.get(param_id, zeros_like(params))
        params = sgd_step(params, grads, self.learning_rate, self.momentum, 0.0, velocity)
        if self.momentum > 0.0:
            self.velocity_buffers[param_id] = velocity
```

## Execution Steps

1. **Migrate ExTensor** (Phase 1):
   - Move `src/extensor/extensor.mojo` → `shared/core/types/`
   - Move `src/extensor/shape.mojo` → `shared/core/types/`
   - Update `shared/core/types/__init__.mojo` exports

2. **Migrate Operations** (Phase 2):
   - Move all `src/extensor/*.mojo` → `shared/core/ops/`
   - Update `shared/core/ops/__init__.mojo` exports

3. **Create Layer Classes** (Phase 3):
   - Implement `Linear`, `Conv2D`, `ReLU`, `Sigmoid`, `Tanh` in `shared/core/layers/`
   - Update `shared/core/layers/__init__.mojo`

4. **Create Optimizer Classes** (Phase 4):
   - Keep functional `sgd_step()` in `sgd.mojo`
   - Add `SGD` class to same file
   - Implement `Adam`, `AdamW`, `RMSprop` classes
   - Update `shared/training/optimizers/__init__.mojo`

5. **Fix Imports** (Phase 5):
   - Replace `from tensor import Tensor` with `from shared.core.types import ExTensor` in:
     - `shared/data/datasets.mojo`
     - `shared/data/transforms.mojo`
     - `shared/data/loaders.mojo`
     - `shared/data/generic_transforms.mojo`

6. **Update Tests** (Phase 6):
   - Uncomment test code in `tests/shared/core/test_layers.mojo`
   - Uncomment test code in `tests/shared/training/test_optimizers.mojo`
   - Update test imports to use `ExTensor`, layer classes, optimizer classes

7. **Clean Up** (Phase 7):
   - Delete `src/extensor/` directory
   - Update `shared/core/__init__.mojo` to export everything
   - Verify all imports work

## Success Criteria

After completion:
- [ ] `src/extensor/` deleted (all code moved to `shared/core/`)
- [ ] No `Tensor` aliases anywhere (everything uses `ExTensor`)
- [ ] Functional layer in `shared/core/ops/` (pure functions)
- [ ] Class layer in `shared/core/layers/` and `shared/training/optimizers/`
- [ ] All imports fixed (no broken `from tensor import Tensor`)
- [ ] Layer classes implemented: `Linear`, `Conv2D`, `ReLU`, `Sigmoid`, `Tanh`
- [ ] Optimizer classes implemented: `SGD`, `Adam`, `AdamW`, `RMSprop`
- [ ] Test stubs can be uncommented and compile
- [ ] Clear two-level hierarchy: functions → classes

## Design Principles to Follow

1. **Composability**: Functions work with any ExTensor dtype
2. **Separation**: Functional (ops/) separate from classes (layers/, optimizers/)
3. **No Duplication**: Classes delegate to functions, don't reimplement
4. **State Management**: Only classes hold state (weights, momentum buffers)
5. **Pure Functions**: Operations in ops/ are stateless and side-effect free

## Reference

See `notes/issues/1538/architecture-redesign-prompt.md` for detailed examples, code templates, and complete migration plan.

## Focus

**Only work in `shared/` directory** - migrate from `src/extensor/` and implement missing classes. Do NOT modify test files until the shared library is complete.
