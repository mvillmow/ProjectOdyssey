# System Architecture

ML Odyssey's system design, architectural decisions, and design patterns.

## Overview

ML Odyssey is architected as a modular, hierarchical system optimized for research reproducibility and performance.
This document covers the system architecture, design decisions, and key patterns.

## High-Level Architecture

```text
┌─────────────────────────────────────────────────────────┐
│                    Papers Layer                          │
│  (LeNet-5, AlexNet, VGG, ResNet, ...)                   │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────┐
│                 Shared Library                           │
│  ┌──────────┬──────────┬──────────┬──────────┐         │
│  │  Core    │ Training │   Data   │  Utils   │         │
│  └──────────┴──────────┴──────────┴──────────┘         │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────┐
│             Mojo/MAX Runtime                             │
│  (SIMD, Parallelization, Memory Management)             │
└─────────────────────────────────────────────────────────┘
```

## Design Principles

### 1. Modularity

Each component has a single, well-defined responsibility:

```text
shared/
├── core/         # Neural network building blocks
├── training/     # Training infrastructure
├── data/         # Data processing
└── utils/        # Cross-cutting utilities
```

**Benefits**:

- Easy to understand and maintain
- Components can be developed independently
- Clear dependency graph
- Reusable across papers

### 2. Hierarchical Organization

Four-level hierarchy from sections to subcomponents:

```text
Section (e.g., shared-library)
  ├── Subsection (e.g., core)
  │   ├── Component (e.g., layers)
  │   │   └── Subcomponent (e.g., linear-layer)
```

**Benefits**:

- Clear structure
- Easy navigation
- Scalable to large projects

### 3. Trait-Based Design

Use traits for polymorphism without inheritance overhead:

```mojo
trait Module:
    """Base interface for all neural network modules."""
    fn forward(inout self, borrowed input: Tensor) -> Tensor: ...
    fn parameters(inout self) -> List[Tensor]: ...

# All layers implement Module
struct Linear(Module): ...
struct Conv2D(Module): ...
struct Sequential(Module): ...
```

**Benefits**:

- No virtual function overhead
- Compile-time polymorphism
- Type-safe interfaces

### 4. Zero-Cost Abstractions

High-level APIs with low-level performance:

```mojo
# High-level API
var model = Sequential([Linear(784, 128), ReLU(), Linear(128, 10)])

# Compiles to efficient code (no runtime overhead)
```

**Implementation**:

- Compile-time specialization with `@parameter`
- SIMD vectorization
- Inlining
- Memory layout optimization

## Core Components

### Tensor System

Foundation of all operations:

```mojo
struct Tensor:
    """Multi-dimensional array with automatic differentiation."""
    var data: DTypePointer[DType.float32]
    var shape: List[Int]
    var stride: List[Int]
    var grad: Optional[Tensor]

    fn __init__(inout self, shape: List[Int]):
        """Allocate tensor with given shape."""
        self.data = allocate(product(shape))
        self.shape = shape
        self.stride = compute_stride(shape)
        self.grad = None
```

**Key features**:

- Automatic gradient computation
- SIMD-optimized operations
- Memory-efficient storage
- Broadcasting support

### Module System

Building blocks for neural networks:

```text
Module (trait)
  ├── Layer modules (Linear, Conv2D, etc.)
  ├── Activation modules (ReLU, Tanh, etc.)
  └── Container modules (Sequential, ModuleList)
```

**Design**:

- All modules implement `Module` trait
- Parameters returned for optimization
- Forward pass defined per module
- Composable into complex models

### Training System

Orchestrates the training loop:

```mojo
struct Trainer:
    """High-level training orchestration."""
    var model: Module
    var optimizer: Optimizer
    var loss_fn: LossFunction
    var callbacks: List[Callback]

    fn train(
        inout self,
        train_loader: BatchLoader,
        val_loader: BatchLoader,
        epochs: Int
    ):
        """Main training loop."""
        for epoch in range(epochs):
            self.train_epoch(train_loader)
            self.validate(val_loader)
            self.run_callbacks()
```

**Components**:

- Optimizers (SGD, Adam, etc.)
- Loss functions
- Callbacks (early stopping, checkpointing)
- Learning rate schedulers

### Data Pipeline

Efficient data loading and processing:

```text
Dataset (trait)
  └── TensorDataset, ImageDataset, etc.
      ↓
BatchLoader
  └── Batching, shuffling, parallel loading
      ↓
Transforms
  └── Augmentation, normalization
```

## Architectural Decisions

### ADR-001: Language Selection

**Decision**: Use Mojo for ML/AI, Python for automation

**Rationale**:

- Mojo provides performance for ML workloads
- Python ecosystem for tooling
- Clear separation of concerns

**Details**: See [ADR-001](../../notes/review/adr/ADR-001-language-selection-tooling.md)

### ADR-002: Memory Management

**Decision**: Use ownership and borrowing for safety

**Rationale**:

- Prevents memory leaks
- No garbage collection overhead
- Compile-time checking

**Pattern**:

```mojo
# Borrowed: read-only, no ownership transfer
fn compute(borrowed input: Tensor) -> Tensor: ...

# Owned: takes ownership, can consume
fn consume(owned tensor: Tensor): ...

# Inout: mutable reference
fn update(inout tensor: Tensor): ...
```

### ADR-003: Autograd Design

**Decision**: Graph-free automatic differentiation

**Rationale**:

- Simpler implementation
- Lower memory overhead
- Sufficient for most use cases

**Implementation**:

```mojo
struct Tensor:
    var grad: Optional[Tensor]  # Gradient storage

    fn backward(inout self):
        """Compute gradients via reverse-mode AD."""
        # Traverse computation graph implicitly
        # Accumulate gradients in param.grad
```

### ADR-004: SIMD as Default

**Decision**: SIMD-optimize all hot loops

**Rationale**:

- 4-8x speedup on CPU
- Mojo makes SIMD easy
- Critical for performance

**Pattern**:

```mojo
alias simd_width = simdwidthof[DType.float32]()

@parameter
fn vectorized[width: Int](idx: Int):
    var vec = data.load[width](idx)
    vec = vec * 2.0  # SIMD operation
    data.store[width](idx, vec)

vectorize[simd_width, vectorized](size)
```

## Design Patterns

### Pattern 1: Builder Pattern

Construct complex objects:

```mojo
struct ModelBuilder:
    """Builder for complex models."""
    var layers: List[Module]

    fn add_conv_block(inout self, in_channels: Int, out_channels: Int):
        self.layers.append(Conv2D(in_channels, out_channels, 3))
        self.layers.append(BatchNorm2D(out_channels))
        self.layers.append(ReLU())
        return self

    fn build(owned self) -> Sequential:
        return Sequential(self.layers)

# Usage
var model = ModelBuilder()
    .add_conv_block(3, 64)
    .add_conv_block(64, 128)
    .build()
```

### Pattern 2: Strategy Pattern

Interchangeable algorithms:

```mojo
trait Optimizer:
    fn step(self, inout parameters: List[Tensor]): ...

struct SGD(Optimizer): ...
struct Adam(Optimizer): ...

# Use any optimizer
fn train[O: Optimizer](model: Model, optimizer: O):
    optimizer.step(model.parameters())
```

### Pattern 3: Observer Pattern

Callbacks for extensibility:

```mojo
trait Callback:
    fn on_epoch_end(inout self, epoch: Int, metrics: Dict): ...

struct EarlyStopping(Callback): ...
struct ModelCheckpoint(Callback): ...

# Trainer notifies callbacks
for callback in self.callbacks:
    callback.on_epoch_end(epoch, metrics)
```

### Pattern 4: Composite Pattern

Nested structures:

```mojo
struct Sequential(Module):
    """Container for sequential layers."""
    var layers: List[Module]

    fn forward(inout self, borrowed input: Tensor) -> Tensor:
        var x = input
        for layer in self.layers:
            x = layer.forward(x)  # Each layer is a Module
        return x
```

## Scalability Considerations

### Horizontal Scalability

Designed to scale across papers:

```text
papers/
├── lenet5/      # Paper 1
├── alexnet/     # Paper 2
├── vgg/         # Paper 3
└── resnet/      # Paper N
```

Each paper is independent, shares common library.

### Vertical Scalability

Support for large models (future):

- Model parallelism
- Pipeline parallelism
- Gradient checkpointing
- Mixed precision training

### Performance Scalability

Optimizations at multiple levels:

1. **Algorithm level**: Efficient algorithms
2. **Kernel level**: SIMD vectorization
3. **System level**: Parallelization
4. **Hardware level**: GPU support (future)

## Dependency Management

### Dependency Graph

```text
papers/lenet5
  └── shared/
      ├── core/
      ├── training/
      │   └── shared/core/
      ├── data/
      │   └── shared/core/
      └── utils/
```

**Rules**:

- Papers depend on shared library
- No circular dependencies
- Core has minimal dependencies
- Utils may depend on all modules

### Versioning Strategy

```text
v0.1.0 - Initial release
v0.2.0 - Breaking changes to API
v0.2.1 - Bug fixes, no API changes
```

**Semantic versioning**:

- MAJOR: Breaking changes
- MINOR: New features, backward compatible
- PATCH: Bug fixes

## Testing Architecture

### Test Pyramid

```text
    /\
   /  \  Unit Tests (fast, many)
  /____\
  \    /  Integration Tests (medium speed, some)
   \  /
    \/   System Tests (slow, few)
```

**Coverage targets**:

- Unit tests: 90%+
- Integration tests: Key workflows
- System tests: End-to-end validation

### Test Organization

Mirrors source structure:

```text
tests/
├── shared/
│   ├── core/
│   ├── training/
│   ├── data/
│   └── utils/
└── papers/
    └── lenet5/
```

## Future Architecture

### Planned Improvements

1. **Distributed Training** - Multi-GPU, multi-machine
2. **Model Compression** - Quantization, pruning
3. **AutoML** - Hyperparameter search, NAS
4. **Deployment** - Model serving, inference optimization

### Extension Points

Designed for easy extension:

- New layers: Implement `Module` trait
- New optimizers: Implement `Optimizer` trait
- New data loaders: Implement `Dataset` trait
- New callbacks: Implement `Callback` trait

## Best Practices

### For Contributors

1. **Follow patterns**: Use existing patterns
2. **Minimal dependencies**: Keep modules independent
3. **Document decisions**: Update ADRs
4. **Test thoroughly**: Match test pyramid
5. **Profile first**: Optimize based on data

### For Users

1. **Use shared library**: Don't reinvent
2. **Follow structure**: Papers in `papers/`
3. **Leverage traits**: Generic code
4. **SIMD optimize**: For performance

## Next Steps

- **[API Reference](api-reference.md)** - Detailed API documentation
- **[CI/CD](ci-cd.md)** - Build and deployment architecture
- **[Release Process](release-process.md)** - Version management

## Related Documentation

- [Project Structure](../core/project-structure.md) - Repository organization
- [Mojo Patterns](../core/mojo-patterns.md) - Language-specific patterns
- [ADRs](../../notes/review/adr/) - Architecture decision records
