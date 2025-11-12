# Training Library(WIP)

The `training` library contains training infrastructure and utilities used across all paper implementations in ML
Odyssey. All components are implemented in Mojo for maximum performance.

## Purpose

This library provides:

- Reusable optimizer implementations (SGD, Adam, AdamW, RMSprop, etc.)
- Learning rate schedulers (step decay, cosine annealing, warmup, etc.)
- Training and evaluation metrics (accuracy, loss tracking, confusion matrix, etc.)
- Training callbacks (early stopping, checkpointing, logging, etc.)
- Generic training loop implementations (basic loop, distributed training, etc.)

**Key Principle**: Only REUSABLE training infrastructure that is shared across MULTIPLE paper implementations belongs
here. Paper-specific training configurations, hyperparameters, and custom training logic should be in the respective
paper directories.

## Directory Organization

```text
training/
├── __init__.mojo           # Package root - exports main components
├── README.md               # This file
├── optimizers/             # Optimizer implementations
│   ├── __init__.mojo       # Optimizer exports
│   ├── sgd.mojo            # Stochastic Gradient Descent (with momentum)
│   ├── adam.mojo           # Adam optimizer
│   ├── adamw.mojo          # AdamW optimizer (Adam with weight decay)
│   ├── rmsprop.mojo        # RMSprop optimizer
│   └── base.mojo           # Base optimizer interface/trait
├── schedulers/             # Learning rate schedulers
│   ├── __init__.mojo       # Scheduler exports
│   ├── step_decay.mojo     # Step decay scheduler
│   ├── cosine.mojo         # Cosine annealing scheduler
│   ├── warmup.mojo         # Warmup scheduler
│   ├── exponential.mojo    # Exponential decay scheduler
│   └── base.mojo           # Base scheduler interface/trait
├── metrics/                # Training and evaluation metrics
│   ├── __init__.mojo       # Metric exports
│   ├── accuracy.mojo       # Classification accuracy
│   ├── loss_tracker.mojo   # Loss tracking and averaging
│   ├── confusion.mojo      # Confusion matrix
│   ├── precision.mojo      # Precision metric
│   ├── recall.mojo         # Recall metric
│   └── base.mojo           # Base metric interface/trait
├── callbacks/              # Training callbacks
│   ├── __init__.mojo       # Callback exports
│   ├── early_stopping.mojo # Early stopping callback
│   ├── checkpoint.mojo     # Model checkpointing
│   ├── logger.mojo         # Training progress logging
│   ├── lr_scheduler.mojo   # LR scheduler callback
│   └── base.mojo           # Base callback interface/trait
└── loops/                  # Training loop implementations
    ├── __init__.mojo       # Training loop exports
    ├── basic.mojo          # Basic training loop
    ├── validation.mojo     # Training loop with validation
    └── distributed.mojo    # Distributed training loop
```

## What Belongs in Training?

### DO Include

**Optimizers**:

- Standard optimizers used across multiple papers (SGD, Adam, RMSprop, etc.)
- Generic optimizer utilities (gradient clipping, weight decay, etc.)
- Optimizer state management (momentum, running averages, etc.)

**Schedulers**:

- Common learning rate schedules (step decay, cosine annealing, etc.)
- Generic scheduler utilities (warmup, cooldown, etc.)
- Schedule composition and chaining

**Metrics**:

- Standard evaluation metrics (accuracy, precision, recall, F1, etc.)
- Loss tracking and aggregation
- Metrics that multiple papers need

**Callbacks**:

- Common training callbacks (early stopping, checkpointing, logging)
- Reusable callback interfaces
- Callback composition and orchestration

**Training Loops**:

- Generic training loop patterns (basic loop, validation loop, etc.)
- Training utilities (batch iteration, gradient accumulation, etc.)
- Reusable training infrastructure

### DON'T Include

**Paper-Specific Code**:

- Custom optimizers designed for one paper only
- Paper-specific hyperparameters or configurations
- Domain-specific metrics (e.g., custom metrics for one experiment)
- Experiment-specific callbacks

**Configuration**:

- Training hyperparameters (learning rate values, batch sizes, etc.)
- Model architecture definitions (these belong in shared/models or papers/)
- Dataset-specific preprocessing (belongs in paper directory)

**One-Off Utilities**:

- Helper functions used by only one paper
- Experimental training techniques not yet validated across papers
- Debugging or profiling code specific to one experiment

**Rule of Thumb**: If you're implementing training infrastructure for a specific paper and you're not sure if it
should be in training/, it probably shouldn't. Wait until 2-3 papers need it, then refactor into the training library.

## Using Training Components in Papers

### Basic Training Loop Example

```mojo
from shared.training.optimizers import SGD, Adam
from shared.training.schedulers import CosineAnnealingLR
from shared.training.metrics import Accuracy, LossTracker
from shared.training.callbacks import EarlyStopping, ModelCheckpoint
from shared.training.loops import BasicTrainingLoop

# Create optimizer
var optimizer = Adam(
    learning_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8
)

# Create learning rate scheduler
var scheduler = CosineAnnealingLR(
    optimizer=optimizer,
    T_max=100,
    eta_min=1e-6
)

# Create metrics
var train_loss = LossTracker()
var train_acc = Accuracy()
var val_loss = LossTracker()
var val_acc = Accuracy()

# Create callbacks
var early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    mode="min"
)

var checkpoint = ModelCheckpoint(
    filepath="checkpoints/best_model.mojo",
    monitor="val_acc",
    mode="max"
)

# Create and run training loop
var loop = BasicTrainingLoop(
    model=my_model,
    optimizer=optimizer,
    scheduler=scheduler,
    metrics=[train_loss, train_acc, val_loss, val_acc],
    callbacks=[early_stopping, checkpoint]
)

loop.fit(
    train_data=train_loader,
    val_data=val_loader,
    epochs=100
)
```

### Custom Training Loop Example

```mojo
from shared.training.optimizers import SGD
from shared.training.metrics import Accuracy
from shared.core.types import Tensor

fn custom_training_loop(
    model: Model,
    optimizer: SGD,
    train_data: DataLoader,
    epochs: Int
):
    """Custom training loop for paper-specific needs."""
    for epoch in range(epochs):
        var total_loss: Float32 = 0.0
        var accuracy = Accuracy()

        for batch in train_data:
            # Forward pass
            var outputs = model.forward(batch.inputs)
            var loss = compute_loss(outputs, batch.targets)

            # Backward pass
            var grads = compute_gradients(loss, model)

            # Optimizer step (from training library)
            optimizer.step(model.parameters(), grads)

            # Track metrics (from training library)
            total_loss += loss.item()
            accuracy.update(outputs, batch.targets)

        # Paper-specific logging or validation
        print("Epoch:", epoch, "Loss:", total_loss / len(train_data))
        print("Accuracy:", accuracy.compute())
```

## Mojo-Specific Guidelines

### Language Features

The training library leverages Mojo's performance features:

1. **SIMD Vectorization**: Use `SIMD` types for vectorized optimizer updates
2. **Memory Safety**: Use ownership and borrowing for parameter updates
3. **Type Safety**: Use strong typing with `struct` for optimizers and metrics
4. **Zero-Cost Abstractions**: Use `@always_inline` for hot paths in training loops

### Code Style

```mojo
# Use 'fn' for performance-critical functions (optimizer steps, metric updates)
fn sgd_step(inout params: Tensor, grads: Tensor, lr: Float32):
    """SGD parameter update with strict typing."""
    # Vectorized update: params = params - lr * grads
    # Implementation with SIMD

# Use 'def' only for high-level orchestration or when flexibility is needed
def train_epoch(model, data_loader, optimizer):
    """High-level training epoch orchestration."""
    # Less strict typing for flexibility

# Prefer struct over class for optimizers and metrics
struct SGD:
    """Stochastic Gradient Descent optimizer."""
    var learning_rate: Float32
    var momentum: Float32
    var dampening: Float32
    var weight_decay: Float32
    var nesterov: Bool

    fn __init__(
        inout self,
        learning_rate: Float32 = 0.01,
        momentum: Float32 = 0.0,
        dampening: Float32 = 0.0,
        weight_decay: Float32 = 0.0,
        nesterov: Bool = False
    ):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov

    @always_inline
    fn step(self, inout params: Tensor, grads: Tensor):
        """Perform single optimization step."""
        # Inline for performance in training loop

# Use trait for base interfaces
trait Optimizer:
    """Base interface for all optimizers."""
    fn step(self, inout params: Tensor, grads: Tensor)
    fn zero_grad(self)
    fn get_lr(self) -> Float32
    fn set_lr(inout self, lr: Float32)
```

### Performance Considerations

1. **Minimize Allocations**: Reuse buffers for gradient updates and metric computation
2. **Vectorize Updates**: Use SIMD for element-wise optimizer updates
3. **Inline Hot Paths**: Use `@always_inline` for optimizer steps and metric updates
4. **Avoid Copies**: Use borrowing for read-only access to parameters and gradients
5. **Profile First**: Measure training loop performance before optimizing

### Training-Specific Performance Patterns

**Optimizer Updates**:

```mojo
# GOOD: Vectorized, in-place update
@always_inline
fn sgd_update_vectorized(inout params: Tensor, grads: Tensor, lr: Float32):
    """Vectorized SGD update using SIMD."""
    let n = params.size()
    for i in range(0, n, simdwidthof[DType.float32]()):
        let p = params.load[simdwidth=simdwidthof[DType.float32]()][i]
        let g = grads.load[simdwidthof[DType.float32]()][i]
        params.store[simdwidthof[DType.float32]()][i, p - lr * g]

# BAD: Scalar, creates copies
fn sgd_update_scalar(params: Tensor, grads: Tensor, lr: Float32) -> Tensor:
    """Non-vectorized, creates copy."""
    var new_params = params.copy()  # Unnecessary allocation
    for i in range(params.size()):
        new_params[i] = params[i] - lr * grads[i]  # Scalar operations
    return new_params
```

**Metric Accumulation**:

```mojo
# GOOD: In-place accumulation with minimal allocations
struct Accuracy:
    var correct: Int
    var total: Int

    fn __init__(inout self):
        self.correct = 0
        self.total = 0

    @always_inline
    fn update(inout self, predictions: Tensor, targets: Tensor):
        """Update accuracy in-place."""
        # Vectorized comparison and counting
        self.correct += count_matches_simd(predictions, targets)
        self.total += predictions.size()

    fn compute(self) -> Float32:
        """Compute final accuracy."""
        return Float32(self.correct) / Float32(self.total)

# BAD: Creates intermediate tensors
def compute_accuracy(predictions: Tensor, targets: Tensor) -> Float32:
    """Non-optimal accuracy computation."""
    matches = predictions == targets  # Creates new tensor
    return matches.sum() / matches.size()  # More allocations
```

## Testing

All training components must have comprehensive tests:

- Unit tests for individual optimizers, schedulers, metrics, callbacks
- Integration tests for training loop compositions
- Performance benchmarks for optimizer updates and metric computation
- Numerical accuracy tests (compare against reference implementations)
- Edge case and error condition tests

See `tests/training/` for test organization and examples.

## Contributing

When adding new components to training:

1. **Verify Shared Need**: Ensure at least 2-3 papers will use it
2. **Write Tests First**: Follow TDD principles
3. **Document Thoroughly**: Include docstrings, usage examples, and hyperparameter descriptions
4. **Benchmark Performance**: Verify optimizer/metric performance is competitive
5. **Review Carefully**: Training changes affect all papers

### Adding a New Optimizer

1. Create `shared/training/optimizers/my_optimizer.mojo`
2. Implement the `Optimizer` trait
3. Write comprehensive tests in `tests/training/optimizers/test_my_optimizer.mojo`
4. Benchmark against reference implementation (PyTorch, TensorFlow, etc.)
5. Add usage example to this README
6. Export from `shared/training/optimizers/__init__.mojo`

### Adding a New Metric

1. Create `shared/training/metrics/my_metric.mojo`
2. Implement the `Metric` trait
3. Write tests with known ground truth values
4. Verify numerical accuracy
5. Add usage example to this README
6. Export from `shared/training/metrics/__init__.mojo`

## Performance Targets

Training components should meet these performance goals:

- Optimizer updates: Within 2x of PyTorch performance
- Metric computation: Fully vectorized with SIMD
- Training loop overhead: Less than 5% of total epoch time
- Memory allocations: Minimize allocations in hot paths

## Common Training Patterns

### Pattern 1: Basic Training with Validation

```mojo
from shared.training.loops import BasicTrainingLoop
from shared.training.callbacks import EarlyStopping, ModelCheckpoint

var loop = BasicTrainingLoop(model, optimizer)
loop.add_callback(EarlyStopping(patience=10))
loop.add_callback(ModelCheckpoint("best_model.mojo"))
loop.fit(train_data, val_data, epochs=100)
```

### Pattern 2: Custom Training Loop with Standard Components

```mojo
from shared.training.optimizers import Adam
from shared.training.schedulers import CosineAnnealingLR
from shared.training.metrics import Accuracy

# Use standard optimizer and scheduler
var optimizer = Adam(lr=0.001)
var scheduler = CosineAnnealingLR(optimizer, T_max=100)

# Custom training loop logic
for epoch in range(epochs):
    for batch in train_loader:
        # Custom forward/backward pass
        loss = custom_forward_backward(model, batch)

        # Standard optimizer step
        optimizer.step(model.parameters(), grads)
        scheduler.step()
```

### Pattern 3: Composing Callbacks

```mojo
from shared.training.callbacks import CallbackList

var callbacks = CallbackList([
    EarlyStopping(patience=10),
    ModelCheckpoint("checkpoints/"),
    CSVLogger("training.log"),
    TensorBoardLogger("logs/")
])

# Use in training loop
for epoch in range(epochs):
    callbacks.on_epoch_begin(epoch)
    # Training logic...
    callbacks.on_epoch_end(epoch, logs)
```

## Paper-Specific vs Shared Training Code

### Shared Training Library

**Located**: `shared/training/`

**Contains**:

- Generic optimizer implementations (SGD, Adam, etc.)
- Standard learning rate schedulers
- Common metrics and callbacks
- Reusable training loop patterns

**Example**: SGD optimizer used by LeNet-5, AlexNet, VGG, etc.

### Paper-Specific Training Code

**Located**: `papers/lenet5/training/` (or similar)

**Contains**:

- Paper-specific hyperparameters
- Custom training configurations
- Paper-specific augmentation or preprocessing
- Experiment-specific training scripts

**Example**: LeNet-5's specific learning rate schedule, batch size, and training duration

### Decision Flowchart

```text
Is this training code needed by multiple papers?
├── YES: Put in shared/training/
│   └── Is it a standard component (optimizer, scheduler, metric)?
│       ├── YES: Implement in appropriate subdirectory
│       └── NO: Consider if it's truly reusable
└── NO: Put in papers/<paper-name>/training/
    └── Can be refactored to shared/ later if other papers need it
```

## Related Documentation

- Core Library: `shared/core/README.md`
- Models Library: `shared/models/README.md`
- Main Shared Library: `shared/README.md`
- Mojo Language Guide: <https://docs.modular.com/mojo/>
- Project Documentation: `/home/mvillmow/ml-odyssey-manual/worktrees/issue-22-plan/CLAUDE.md`

## Future Enhancements

**Planned for Implementation Phase**:

- Distributed training support (data parallel, model parallel)
- Mixed precision training utilities (FP16, BF16)
- Gradient accumulation for large batch training
- Advanced optimizer features (lookahead, SAM, etc.)
- Custom optimizer state serialization
- Advanced learning rate schedules (one-cycle, SGDR, etc.)
- More comprehensive metrics (ROC-AUC, mAP, etc.)
- Profiling and debugging callbacks

**Long-term Goals**:

- Automatic mixed precision (AMP) support
- Distributed training across multiple machines
- Model parallelism utilities
- Training performance profiler
- Hyperparameter optimization integration

## License

See repository LICENSE file.
