# Shared Library

Reusable ML components for all ML Odyssey paper implementations.

## Overview

The shared library (`shared/`) provides battle-tested, high-performance components that paper implementations can import
and use. This eliminates code duplication, ensures consistency, and allows all papers to benefit from optimizations.

**Philosophy**: Correctness first, then performance. Every component is thoroughly tested before optimization.

## Architecture

```text

shared/
├── core/           # Core ML operations
│   ├── ops/        # Tensor operations (matmul, conv, etc.)
│   ├── layers/     # Neural network layers
│   └── utils/      # Initialization, metrics
├── training/       # Training infrastructure
│   ├── optimizers/ # SGD, Adam, etc.
│   ├── schedulers/ # Learning rate schedules
│   └── callbacks/  # Training callbacks
├── data/           # Data handling
│   ├── datasets.mojo
│   ├── loaders.mojo
│   └── transforms.mojo
└── utils/          # General utilities
    └── config.mojo

```text

## Core Components

### Tensor Operations (`shared/core/ops/`)

**Basic Operations**:

```mojo

# Element-wise
fn add(a: Tensor, b: Tensor) -> Tensor
fn multiply(a: Tensor, b: Tensor) -> Tensor
fn relu(x: Tensor) -> Tensor
fn sigmoid(x: Tensor) -> Tensor

# Matrix operations
fn matmul(a: Tensor, b: Tensor) -> Tensor
fn transpose(x: Tensor, dim0: Int, dim1: Int) -> Tensor

# Reductions
fn sum(x: Tensor, dim: Optional[Int] = None) -> Tensor
fn mean(x: Tensor, dim: Optional[Int] = None) -> Tensor

```text

### Neural Network Layers (`shared/core/layers/`)

**Linear (Dense) Layer**:

```mojo

struct Linear:
    """Fully connected layer: output = input @ weight^T + bias"""
    var weight: Tensor
    var bias: Tensor

    fn __init__(inout self, input_size: Int, output_size: Int):
        self.weight = Tensor.randn(output_size, input_size) * 0.02
        self.bias = Tensor.zeros(output_size)

    fn forward(self, borrowed input: Tensor) -> Tensor:
        return input @ self.weight.T + self.bias

```text

**Convolutional Layer**:

```mojo

struct Conv2D:
    """2D Convolutional layer"""
    var weight: Tensor  # (out_channels, in_channels, kernel_h, kernel_w)
    var bias: Tensor
    var stride: Int
    var padding: Int

    fn __init__(inout self, in_channels: Int, out_channels: Int,
                kernel_size: Int, stride: Int = 1, padding: Int = 0):
        self.weight = Tensor.randn(out_channels, in_channels, kernel_size, kernel_size)
        self.bias = Tensor.zeros(out_channels)
        self.stride = stride
        self.padding = padding

```text

**Pooling Layers**:

```mojo

struct MaxPool2D:
    fn __init__(inout self, kernel_size: Int, stride: Optional[Int] = None)
    fn forward(self, borrowed input: Tensor) -> Tensor

struct AvgPool2D:
    fn __init__(inout self, kernel_size: Int, stride: Optional[Int] = None)
    fn forward(self, borrowed input: Tensor) -> Tensor

```text

### Activation Functions

```mojo

# Activations as layers
struct ReLU:
    fn forward(self, x: Tensor) -> Tensor:
        return max(x, 0.0)

struct Sigmoid:
    fn forward(self, x: Tensor) -> Tensor:
        return 1.0 / (1.0 + exp(-x))

struct Softmax:
    var dim: Int

    fn forward(self, x: Tensor) -> Tensor:
        var exp_x = exp(x - x.max(dim=self.dim, keepdim=True))
        return exp_x / exp_x.sum(dim=self.dim, keepdim=True)

```text

### Initialization (`shared/core/utils/`)

```mojo

# Weight initialization
fn xavier_init(inout tensor: Tensor, gain: Float64 = 1.0):
    """Xavier/Glorot initialization for sigmoid/tanh."""
    var fan_in = tensor.shape[-2]
    var fan_out = tensor.shape[-1]
    var std = gain * sqrt(2.0 / (fan_in + fan_out))
    tensor.normal_(0.0, std)

fn kaiming_init(inout tensor: Tensor, mode: String = "fan_in"):
    """He initialization for ReLU."""
    var fan = tensor.shape[-2] if mode == "fan_in" else tensor.shape[-1]
    var std = sqrt(2.0 / fan)
    tensor.normal_(0.0, std)

```text

## Training Components

### Optimizers (`shared/training/optimizers/`)

**SGD with Momentum**:

```mojo

struct SGD:
    """Stochastic Gradient Descent with momentum."""
    var lr: Float64
    var momentum: Float64
    var velocities: List[Tensor]

    fn __init__(inout self, lr: Float64 = 0.01, momentum: Float64 = 0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocities = List[Tensor]()

    fn step(inout self, inout parameters: List[Tensor]):
        """Update parameters with momentum."""
        if len(self.velocities) == 0:
            for param in parameters:
                self.velocities.append(Tensor.zeros_like(param))

        for i in range(len(parameters)):
            self.velocities[i] = (
                self.momentum * self.velocities[i] +
                parameters[i].grad
            )
            parameters[i] -= self.lr * self.velocities[i]

```text

**Adam Optimizer** (Planned):

```mojo

struct Adam:
    """Adam optimizer with adaptive learning rates."""
    var lr: Float64
    var beta1: Float64
    var beta2: Float64
    var epsilon: Float64
    var m: List[Tensor]  # First moment
    var v: List[Tensor]  # Second moment
    var t: Int  # Timestep

```text

### Learning Rate Schedulers (`shared/training/schedulers/`)

```mojo

struct StepLR:
    """Decay LR by gamma every step_size epochs."""
    var base_lr: Float64
    var step_size: Int
    var gamma: Float64

    fn get_lr(self, epoch: Int) -> Float64:
        return self.base_lr * pow(self.gamma, Float64(epoch // self.step_size))

struct CosineAnnealingLR:
    """Cosine annealing schedule."""
    var base_lr: Float64
    var min_lr: Float64
    var total_epochs: Int

    fn get_lr(self, epoch: Int) -> Float64:
        return self.min_lr + (self.base_lr - self.min_lr) * (
            1 + cos(pi * epoch / self.total_epochs)
        ) / 2

```text

### Callbacks (`shared/training/callbacks/`)

**Base Callback Interface**:

```mojo

trait Callback:
    fn on_train_begin(inout self, inout state: TrainingState) -> CallbackSignal
    fn on_epoch_begin(inout self, inout state: TrainingState) -> CallbackSignal
    fn on_batch_begin(inout self, inout state: TrainingState) -> CallbackSignal
    fn on_batch_end(inout self, inout state: TrainingState) -> CallbackSignal
    fn on_epoch_end(inout self, inout state: TrainingState) -> CallbackSignal
    fn on_train_end(inout self, inout state: TrainingState) -> CallbackSignal

```text

**Built-in Callbacks**:

```mojo

struct ModelCheckpoint(Callback):
    """Save best model based on monitored metric."""
    var filepath: String
    var monitor: String
    var best_value: Float64

    fn on_epoch_end(inout self, inout state: TrainingState) -> CallbackSignal:
        var current = state.metrics.get(self.monitor)
        if current < self.best_value:
            self.best_value = current
            save_model(state.model, self.filepath)
        return CONTINUE

struct EarlyStopping(Callback):
    """Stop training when metric stops improving."""
    var patience: Int
    var monitor: String
    var wait: Int = 0
    var best_value: Float64 = 1e10

    fn on_epoch_end(inout self, inout state: TrainingState) -> CallbackSignal:
        var current = state.metrics.get(self.monitor)
        if current < self.best_value:
            self.best_value = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                return STOP
        return CONTINUE

```text

## Data Components

### Dataset Interface (`shared/data/datasets.mojo`)

```mojo

trait Dataset:
    """Base dataset interface."""
    fn __len__(self) -> Int
    fn __getitem__(self, idx: Int) -> (Tensor, Tensor)

```text

### Data Loader (`shared/data/loaders.mojo`)

```mojo

struct DataLoader:
    """Efficient batch loading with optional shuffling."""
    var dataset: Dataset
    var batch_size: Int
    var shuffle: Bool

    fn __iter__(inout self) -> DataLoaderIterator:
        var indices = range(len(self.dataset))
        if self.shuffle:
            indices = shuffle_indices(indices)
        return DataLoaderIterator(self.dataset, indices, self.batch_size)

```text

### Transforms (`shared/data/transforms.mojo`)

```mojo

struct Normalize:
    """Normalize images with mean and std."""
    var mean: Float64
    var std: Float64

    fn apply(self, image: Tensor) -> Tensor:
        return (image - self.mean) / self.std

struct RandomFlip:
    """Randomly flip images horizontally."""
    var probability: Float64

    fn apply(self, image: Tensor) -> Tensor:
        if random() < self.probability:
            return flip_horizontal(image)
        return image

struct Compose:
    """Chain multiple transforms."""
    var transforms: List[Transform]

    fn apply(self, image: Tensor) -> Tensor:
        var result = image
        for transform in self.transforms:
            result = transform.apply(result)
        return result

```text

## Usage Examples

### Example 1: Building a Simple Network

```mojo

from shared.core.layers import Linear, ReLU
from shared.core.utils import kaiming_init

struct SimpleNet:
    """2-layer neural network."""
    var fc1: Linear
    var relu: ReLU
    var fc2: Linear

    fn __init__(inout self):
        self.fc1 = Linear(784, 128)
        kaiming_init(self.fc1.weight)
        self.relu = ReLU()
        self.fc2 = Linear(128, 10)
        kaiming_init(self.fc2.weight)

    fn forward(self, x: Tensor) -> Tensor:
        var h = self.fc1.forward(x)
        h = self.relu.forward(h)
        return self.fc2.forward(h)

```text

### Example 2: Training Loop

```mojo

from shared.training import SGD, StepLR
from shared.training.callbacks import ModelCheckpoint, EarlyStopping

fn train():
    var model = SimpleNet()
    var optimizer = SGD(lr=0.1, momentum=0.9)
    var scheduler = StepLR(base_lr=0.1, step_size=30, gamma=0.1)

    var callbacks = [
        ModelCheckpoint("best_model.bin", monitor="val_loss"),
        EarlyStopping(patience=10, monitor="val_loss")
    ]

    for epoch in range(100):
        # Update learning rate
        var lr = scheduler.get_lr(epoch)
        optimizer.lr = lr

        # Train epoch
        for batch in train_loader:
            var predictions = model.forward(batch.data)
            var loss = cross_entropy(predictions, batch.targets)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Callbacks
        for callback in callbacks:
            if callback.on_epoch_end(state) == STOP:
                break

```text

### Example 3: Data Pipeline

```mojo

from shared.data import DataLoader, Compose, Normalize, RandomFlip

struct MNISTDataset(Dataset):
    var images: Tensor
    var labels: Tensor
    var transform: Optional[Compose]

    fn __len__(self) -> Int:
        return self.images.shape[0]

    fn __getitem__(self, idx: Int) -> (Tensor, Tensor):
        var img = self.images[idx]
        if self.transform:
            img = self.transform.value.apply(img)
        return (img, self.labels[idx])

fn create_dataloaders():
    # Create transforms
    var train_transform = Compose([
        RandomFlip(probability=0.5),
        Normalize(mean=0.1307, std=0.3081)
    ])

    # Create datasets
    var train_dataset = MNISTDataset(train_images, train_labels, train_transform)
    var test_dataset = MNISTDataset(test_images, test_labels, Normalize(0.1307, 0.3081))

    # Create loaders
    var train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    var test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    return (train_loader, test_loader)

```text

## Contributing Components

### Adding a New Component

1. **Plan**: Document what the component does
2. **Test**: Write tests first (TDD)
3. **Implement**: Follow Mojo patterns
4. **Document**: Add docstrings and examples
5. **Export**: Add to `__init__.mojo`

### Example: Adding a New Layer

```mojo

# 1. Implement in shared/core/layers/new_layer.mojo
struct MyNewLayer:
    """Concise description of what this layer does."""
    var weight: Tensor

    fn __init__(inout self, input_size: Int, output_size: Int):
        self.weight = Tensor.randn(output_size, input_size)

    fn forward(self, borrowed input: Tensor) -> Tensor:
        return input @ self.weight.T

# 2. Add tests in tests/shared/core/test_new_layer.mojo
fn test_new_layer_forward():
    var layer = MyNewLayer(10, 5)
    var input = Tensor.randn(2, 10)
    var output = layer.forward(input)
    assert_equal(output.shape, (2, 5))

# 3. Export from shared/core/layers/__init__.mojo
from .new_layer import MyNewLayer

# 4. Update this documentation

```text

## Best Practices

1. **Use `fn` for ML Code** - Type safety and performance
2. **Test Everything** - Unit tests + gradient checks
3. **SIMD When Beneficial** - But measure first
4. **Document Clearly** - Docstrings with examples
5. **Keep it Simple** - Correctness > cleverness
6. **Link to Papers** - Reference original implementations

## Performance Considerations

**When to Optimize**:

- Component is proven bottleneck (profile first)
- Used in hot path (training loop)
- Simple optimization gives large gain

**Optimization Techniques**:

- SIMD for element-wise ops
- In-place operations
- Buffer reuse
- Cache-friendly access patterns

See [Performance Guide](../advanced/performance.md) for details.

## Testing

All shared components must have:

- Unit tests
- Gradient checks (for layers)
- Edge case tests
- Performance benchmarks

See [Testing Strategy](testing-strategy.md) for comprehensive guide.

## Related Documentation

- [Mojo Patterns](mojo-patterns.md) - Language-specific patterns
- [Custom Layers](../advanced/custom-layers.md) - Creating new components
- [Performance](../advanced/performance.md) - Optimization guide
- [API Reference](../dev/api-reference.md) - Complete API listing

## Summary

The shared library provides:

- **Core Layers**: Linear, Conv2D, Pooling, Activations
- **Training**: Optimizers (SGD), Schedulers, Callbacks
- **Data**: Dataset interface, DataLoader, Transforms
- **Utils**: Initialization, Metrics, Configuration

**Key principles**:

1. Reusable across papers
2. Thoroughly tested
3. Well-documented
4. Performance-optimized (after correctness)
5. Following Mojo best practices

**Get started**: Browse `shared/` directory, read tests for examples, import into your paper implementation.
