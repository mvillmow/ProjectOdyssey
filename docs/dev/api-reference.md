# API Reference

Comprehensive API documentation for ML Odyssey components.

## Overview

This document provides quick reference to ML Odyssey APIs. For detailed implementation guides, see component-specific
documentation.

## Core Library APIs

### Tensor Operations

```mojo

# Creating tensors

Tensor.zeros(shape: Tuple[Int, ...]) -> Tensor
Tensor.ones(shape: Tuple[Int, ...]) -> Tensor
Tensor.randn(shape: Tuple[Int, ...], seed: Int = 0) -> Tensor
Tensor.randint(low: Int, high: Int, shape: Tuple[Int, ...]) -> Tensor

# Element-wise operations

tensor + other -> Tensor  # Addition
tensor - other -> Tensor  # Subtraction
tensor * other -> Tensor  # Multiplication
tensor / other -> Tensor  # Division

# Matrix operations

tensor @ other -> Tensor  # Matrix multiplication
tensor.T -> Tensor  # Transpose

# Reductions

tensor.sum(dim: Optional[Int] = None) -> Tensor
tensor.mean(dim: Optional[Int] = None) -> Tensor
tensor.max(dim: Optional[Int] = None) -> Tensor
tensor.min(dim: Optional[Int] = None) -> Tensor

# Shape manipulation

tensor.reshape(shape: Tuple[Int, ...]) -> Tensor
tensor.view(shape: Tuple[Int, ...]) -> Tensor
tensor.flatten() -> Tensor

```

### Neural Network Layers

```mojo

```mojo

# Linear (Dense) Layer
struct Linear:
    fn __init__(inout self, input_size: Int, output_size: Int)
    fn forward(self, borrowed input: Tensor) -> Tensor
    fn backward(self, borrowed grad_output: Tensor) -> Tensor
    fn parameters(self) -> List[Tensor]

# Convolutional Layer
struct Conv2D:
    fn __init__(inout self,
                in_channels: Int,
                out_channels: Int,
                kernel_size: Int,
                stride: Int = 1,
                padding: Int = 0)
    fn forward(self, borrowed input: Tensor) -> Tensor

# Pooling Layers
struct MaxPool2D:
    fn __init__(inout self, kernel_size: Int, stride: Optional[Int] = None)
    fn forward(self, borrowed input: Tensor) -> Tensor

struct AvgPool2D:
    fn __init__(inout self, kernel_size: Int, stride: Optional[Int] = None)
    fn forward(self, borrowed input: Tensor) -> Tensor

```

### Activation Functions

```mojo

# Activation function signatures

fn relu(x: Tensor) -> Tensor
fn sigmoid(x: Tensor) -> Tensor
fn tanh(x: Tensor) -> Tensor
fn softmax(x: Tensor, dim: Int = -1) -> Tensor
fn gelu(x: Tensor) -> Tensor

# Activation layers

struct ReLU:
    fn forward(self, x: Tensor) -> Tensor

struct Sigmoid:
    fn forward(self, x: Tensor) -> Tensor

struct Softmax:
    fn __init__(inout self, dim: Int = -1)
    fn forward(self, x: Tensor) -> Tensor

```

### Loss Functions

```mojo

```mojo

fn mse_loss(predictions: Tensor, targets: Tensor) -> Float64
fn cross_entropy_loss(predictions: Tensor, targets: Tensor) -> Float64
fn binary_cross_entropy(predictions: Tensor, targets: Tensor) -> Float64
fn l1_loss(predictions: Tensor, targets: Tensor) -> Float64

```

## Training APIs

### Optimizers

```mojo

# Stochastic Gradient Descent

struct SGD:
    fn __init__(inout self, lr: Float64 = 0.01, momentum: Float64 = 0.0)
    fn step(inout self, inout parameters: List[Tensor])
    fn zero_grad(self, inout parameters: List[Tensor])

# Adam Optimizer

struct Adam:
    fn __init__(inout self,
                lr: Float64 = 0.001,
                beta1: Float64 = 0.9,
                beta2: Float64 = 0.999,
                epsilon: Float64 = 1e-8)
    fn step(inout self, inout parameters: List[Tensor])

```

### Learning Rate Schedulers

```mojo

```mojo

# Step LR
struct StepLR:
    fn __init__(inout self, base_lr: Float64, step_size: Int, gamma: Float64 = 0.1)
    fn get_lr(self, epoch: Int) -> Float64

# Cosine Annealing
struct CosineAnnealingLR:
    fn __init__(inout self, base_lr: Float64, total_epochs: Int, min_lr: Float64 = 0.0)
    fn get_lr(self, epoch: Int) -> Float64

```

### Callbacks

```mojo

# Base callback interface

trait Callback:
    fn on_train_begin(inout self, inout state: TrainingState) -> CallbackSignal
    fn on_epoch_begin(inout self, inout state: TrainingState) -> CallbackSignal
    fn on_batch_begin(inout self, inout state: TrainingState) -> CallbackSignal
    fn on_batch_end(inout self, inout state: TrainingState) -> CallbackSignal
    fn on_epoch_end(inout self, inout state: TrainingState) -> CallbackSignal
    fn on_train_end(inout self, inout state: TrainingState) -> CallbackSignal

# Built-in callbacks

struct ModelCheckpoint(Callback):
    fn __init__(inout self, filepath: String, monitor: String = "val_loss")

struct EarlyStopping(Callback):
    fn __init__(inout self, patience: Int = 5, monitor: String = "val_loss")

struct LoggingCallback(Callback):
    fn __init__(inout self, log_file: String)

```

## Data APIs

### Datasets

```mojo

```mojo

# Base dataset interface
trait Dataset:
    fn __len__(self) -> Int
    fn __getitem__(self, idx: Int) -> (Tensor, Tensor)

# Example implementation
struct MNISTDataset(Dataset):
    fn __init__(inout self, images: Tensor, labels: Tensor)
    fn __len__(self) -> Int
    fn __getitem__(self, idx: Int) -> (Tensor, Tensor)

```

### Data Loaders

```mojo

struct DataLoader:
    fn __init__(inout self,
                dataset: Dataset,
                batch_size: Int = 32,
                shuffle: Bool = False)
    fn __iter__(inout self) -> DataLoaderIterator
    fn __len__(self) -> Int

# Usage

for batch in data_loader:
    var inputs = batch.data
    var targets = batch.targets

```

### Transforms

```mojo

```mojo

# Image transformations
struct Normalize:
    fn __init__(inout self, mean: Float64, std: Float64)
    fn apply(self, image: Tensor) -> Tensor

struct RandomFlip:
    fn __init__(inout self, probability: Float64 = 0.5)
    fn apply(self, image: Tensor) -> Tensor

struct Compose:
    fn __init__(inout self, transforms: List[Transform])
    fn apply(self, image: Tensor) -> Tensor

```

## Utility APIs

### Initialization

```mojo

fn xavier_init(tensor: Tensor, gain: Float64 = 1.0)
fn kaiming_init(tensor: Tensor, mode: String = "fan_in")
fn zeros_init(tensor: Tensor)
fn ones_init(tensor: Tensor)
fn normal_init(tensor: Tensor, mean: Float64 = 0.0, std: Float64 = 0.01)

```

### Metrics

```mojo

```mojo

fn accuracy(predictions: Tensor, targets: Tensor) -> Float64
fn precision(predictions: Tensor, targets: Tensor) -> Float64
fn recall(predictions: Tensor, targets: Tensor) -> Float64
fn f1_score(predictions: Tensor, targets: Tensor) -> Float64

```

### Configuration

```mojo
struct Config:
    fn load(filepath: String) -> Config
    fn save(self, filepath: String)
    fn get[T](self, key: String, default: T) -> T
    fn set[T](inout self, key: String, value: T)
```

## Complete API Documentation

For full API documentation with detailed examples and parameters:

- **Core Library**: `docs/core/shared-library.md`
- **Training**: `docs/core/shared-library.md#training-utilities`
- **Data**: `docs/core/shared-library.md#data-utilities`
- **Mojo Patterns**: `docs/core/mojo-patterns.md`

## Usage Examples

### Basic Training Loop

```mojo

```mojo

from shared.core import Linear, ReLU, Softmax
from shared.training import SGD, cross_entropy_loss
from shared.data import DataLoader

fn train():
    # Create model
    var model = Sequential([
        Linear(784, 128),
        ReLU(),
        Linear(128, 10),
        Softmax()
    ])

    # Create optimizer
    var optimizer = SGD(lr=0.01)

    # Create data loader
    var train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Training loop
    for epoch in range(num_epochs):
        for batch in train_loader:
            # Forward pass
            var predictions = model.forward(batch.data)
            var loss = cross_entropy_loss(predictions, batch.targets)

            # Backward pass
            loss.backward()

            # Update parameters
            optimizer.step(model.parameters())
            optimizer.zero_grad(model.parameters())

```

## Python Interop

For using Python libraries from Mojo:

```mojo

from python import Python

fn use_numpy():
    var np = Python.import_module("numpy")
    var array = np.array([1, 2, 3, 4, 5])
    print(array.mean())

```

## Related Documentation

- [Shared Library Guide](../core/shared-library.md) - Detailed component documentation
- [Mojo Patterns](../core/mojo-patterns.md) - Mojo-specific patterns and idioms
- [Custom Layers](../advanced/custom-layers.md) - Creating custom components
- [Testing Strategy](../core/testing-strategy.md) - Testing your code

## Contributing

To add new APIs:

1. Implement in `shared/` directory
2. Add comprehensive tests in `tests/shared/`
3. Document in docstrings with examples
4. Update this API reference
5. Submit pull request

## Conventions

- **Naming**: `snake_case` for functions, `PascalCase` for structs
- **Ownership**: Explicit `borrowed`, `inout`, `owned` annotations
- **Types**: Use `fn` with type annotations for performance
- **Documentation**: Docstrings with Args, Returns, Examples sections

## Getting Help

- **Search Examples**: Check `examples/` directory
- **Read Tests**: Review `tests/shared/` for usage patterns
- **Ask Questions**: Open GitHub issues with `question` label
- **Check Docs**: Browse comprehensive guides in `docs/`
