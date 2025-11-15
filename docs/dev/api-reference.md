# API Reference

Complete API documentation for ML Odyssey components.

## Overview

This page provides an overview of ML Odyssey's API organization. Detailed API documentation is auto-generated
from code docstrings.

**Status**: 🚧 Auto-generated API documentation coming soon

## API Organization

### Core Module (`shared/core/`)

Neural network building blocks.

#### Layers

**Linear** - Fully connected layer

```mojo
struct Linear(Module):
    """Fully connected (dense) layer.

    Applies linear transformation: y = xW^T + b

    Args:
        input_size: Size of input features.
        output_size: Size of output features.
        bias: Whether to include bias term (default: True).

    Shape:
        - Input: (batch_size, input_size)
        - Output: (batch_size, output_size)

    Example:
        var layer = Linear(784, 128)
        var output = layer.forward(input)
    """

    fn __init__(inout self, input_size: Int, output_size: Int, bias: Bool = True)
    fn forward(inout self, borrowed input: Tensor) -> Tensor
    fn parameters(inout self) -> List[Tensor]
```

**Conv2D** - 2D convolutional layer

```mojo
struct Conv2D(Module):
    """2D convolution layer.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of convolutional kernel.
        stride: Stride of convolution (default: 1).
        padding: Zero-padding added to input (default: 0).

    Shape:
        - Input: (batch, in_channels, height, width)
        - Output: (batch, out_channels, out_height, out_width)
    """

    fn __init__(
        inout self,
        in_channels: Int,
        out_channels: Int,
        kernel_size: Int,
        stride: Int = 1,
        padding: Int = 0
    )
    fn forward(inout self, borrowed input: Tensor) -> Tensor
```

#### Activations

**ReLU** - Rectified Linear Unit

```mojo
struct ReLU(Module):
    """Rectified Linear Unit activation.

    Formula: ReLU(x) = max(0, x)

    Shape:
        - Input: (*) - Any shape
        - Output: (*) - Same shape as input

    Example:
        var relu = ReLU()
        var output = relu.forward(input)
    """

    fn forward(inout self, borrowed input: Tensor) -> Tensor
```

**Softmax** - Softmax activation

```mojo
struct Softmax(Module):
    """Softmax activation function.

    Formula: Softmax(x_i) = exp(x_i) / sum(exp(x_j))

    Args:
        dim: Dimension along which to apply softmax (default: -1).

    Shape:
        - Input: (*) - Any shape
        - Output: (*) - Same shape as input
    """

    fn __init__(inout self, dim: Int = -1)
    fn forward(inout self, borrowed input: Tensor) -> Tensor
```

#### Containers

**Sequential** - Sequential container

```mojo
struct Sequential(Module):
    """Sequential container for layers.

    Chains multiple modules together in sequence.

    Args:
        layers: List of modules to chain.

    Example:
        var model = Sequential([
            Linear(784, 128),
            ReLU(),
            Linear(128, 10)
        ])
    """

    fn __init__(inout self, layers: List[Module])
    fn forward(inout self, borrowed input: Tensor) -> Tensor
    fn add(inout self, layer: Module)
```

### Training Module (`shared/training/`)

Training infrastructure.

#### Optimizers

**SGD** - Stochastic Gradient Descent

```mojo
struct SGD(Optimizer):
    """Stochastic Gradient Descent optimizer.

    Args:
        learning_rate: Learning rate.
        momentum: Momentum factor (default: 0.0).
        weight_decay: Weight decay (L2 penalty) (default: 0.0).

    Example:
        var optimizer = SGD(lr=0.01, momentum=0.9)
        optimizer.step(model.parameters())
    """

    fn __init__(
        inout self,
        learning_rate: Float64,
        momentum: Float64 = 0.0,
        weight_decay: Float64 = 0.0
    )
    fn step(self, inout parameters: List[Tensor])
    fn zero_grad(self, inout parameters: List[Tensor])
```

**Adam** - Adaptive Moment Estimation

```mojo
struct Adam(Optimizer):
    """Adam optimizer.

    Args:
        learning_rate: Learning rate (default: 0.001).
        beta1: Exponential decay rate for first moment (default: 0.9).
        beta2: Exponential decay rate for second moment (default: 0.999).
        epsilon: Small constant for numerical stability (default: 1e-8).

    Reference:
        Kingma & Ba, "Adam: A Method for Stochastic Optimization", ICLR 2015
    """

    fn __init__(
        inout self,
        learning_rate: Float64 = 0.001,
        beta1: Float64 = 0.9,
        beta2: Float64 = 0.999,
        epsilon: Float64 = 1e-8
    )
    fn step(self, inout parameters: List[Tensor])
```

#### Loss Functions

**CrossEntropyLoss** - Cross-entropy loss

```mojo
struct CrossEntropyLoss:
    """Cross-entropy loss for classification.

    Combines LogSoftmax and NLLLoss.

    Args:
        reduction: Specifies reduction to apply (default: "mean").

    Shape:
        - Input: (batch_size, num_classes)
        - Target: (batch_size,) with class indices

    Example:
        var loss_fn = CrossEntropyLoss()
        var loss = loss_fn(predictions, targets)
    """

    fn __call__(self, borrowed predictions: Tensor, borrowed targets: Tensor) -> Tensor
```

#### Trainer

**Trainer** - High-level training orchestration

```mojo
struct Trainer:
    """High-level trainer for neural networks.

    Args:
        model: Neural network model.
        optimizer: Optimizer for parameter updates.
        loss_fn: Loss function.

    Example:
        var trainer = Trainer(model, optimizer, loss_fn)
        trainer.train(train_loader, val_loader, epochs=10)
    """

    fn __init__(
        inout self,
        model: Module,
        optimizer: Optimizer,
        loss_fn: LossFunction
    )

    fn train(
        inout self,
        train_loader: BatchLoader,
        val_loader: BatchLoader,
        epochs: Int
    ) -> TrainingHistory

    fn add_callback(inout self, callback: Callback)
```

### Data Module (`shared/data/`)

Data loading and processing.

#### Datasets

**TensorDataset** - Dataset from tensors

```mojo
struct TensorDataset(Dataset):
    """Dataset wrapping tensors.

    Args:
        inputs: Input tensor.
        targets: Target tensor.

    Example:
        var dataset = TensorDataset(X_train, y_train)
        var sample, label = dataset[0]
    """

    fn __init__(inout self, inputs: Tensor, targets: Tensor)
    fn __len__(self) -> Int
    fn __getitem__(self, idx: Int) -> (Tensor, Tensor)
```

#### Data Loaders

**BatchLoader** - Batch data loading

```mojo
struct BatchLoader:
    """Batched data loading with shuffling.

    Args:
        dataset: Dataset to load from.
        batch_size: Samples per batch.
        shuffle: Whether to shuffle data (default: False).
        drop_last: Drop incomplete last batch (default: False).

    Example:
        var loader = BatchLoader(dataset, batch_size=32, shuffle=True)
        for batch in loader:
            var inputs, targets = batch
            ...
    """

    fn __init__(
        inout self,
        dataset: Dataset,
        batch_size: Int,
        shuffle: Bool = False,
        drop_last: Bool = False
    )

    fn __iter__(inout self) -> Self
    fn __next__(inout self) -> (Tensor, Tensor)
```

### Utils Module (`shared/utils/`)

Utilities and helpers.

#### Configuration

**Config** - Configuration management

```mojo
struct Config:
    """Configuration management from TOML files.

    Example:
        var config = Config.from_file("config.toml")
        var lr = config.get[Float64]("training.learning_rate")
    """

    @staticmethod
    fn from_file(path: String) -> Config

    fn get[T: AnyType](self, key: String) -> T
    fn set[T: AnyType](inout self, key: String, value: T)
    fn has(self, key: String) -> Bool
```

#### Logging

**Logger** - Structured logging

```mojo
struct Logger:
    """Structured logger with multiple levels.

    Args:
        name: Logger name.
        level: Logging level (default: "INFO").

    Example:
        var logger = Logger("training", level="DEBUG")
        logger.info("Starting training...")
        logger.debug("Batch size:", batch_size)
    """

    fn __init__(inout self, name: String, level: String = "INFO")

    fn debug(self, message: String, *args)
    fn info(self, message: String, *args)
    fn warning(self, message: String, *args)
    fn error(self, message: String, *args)
```

## Versioning

API follows semantic versioning:

- **Major version** (v1.x.x → v2.x.x): Breaking changes
- **Minor version** (v1.1.x → v1.2.x): New features, backward compatible
- **Patch version** (v1.1.1 → v1.1.2): Bug fixes

### Deprecation Policy

1. **Deprecation warning**: Added in minor version
2. **Grace period**: 2 minor versions minimum
3. **Removal**: In next major version

Example:

```text
v0.1.0 - Feature introduced
v0.3.0 - Feature deprecated (warning added)
v0.5.0 - Still available with warning
v1.0.0 - Feature removed
```

## Auto-Generated Documentation

**Coming Soon**: Full API documentation will be auto-generated from docstrings.

**Tools**:

- Mojo documentation generator
- Cross-referencing
- Search functionality
- Examples from tests

**Location**: `docs/api/` (future)

## API Stability

### Stable APIs

**Core Module**: Stable since v0.1.0

- Linear, Conv2D layers
- ReLU, Softmax activations
- Sequential container

**Training Module**: Stable since v0.1.0

- SGD, Adam optimizers
- CrossEntropyLoss
- Basic Trainer

### Experimental APIs

**Marked with** `@experimental` **decorator**:

- May change in minor versions
- Use with caution in production

Example:

```mojo
@experimental
struct QuantizedLinear(Module):
    """Quantized linear layer (experimental).

    Warning: API may change in future versions.
    """
    ...
```

## Best Practices

### Using the API

1. **Import from public modules**: Use `from shared.core import Linear`
2. **Check version compatibility**: Verify version matches your needs
3. **Handle deprecations**: Update code when deprecation warnings appear
4. **Read docstrings**: Detailed information in code documentation

### Contributing to API

1. **Document everything**: All public APIs need docstrings
2. **Maintain backward compatibility**: Use deprecation for changes
3. **Add examples**: Include usage examples in docstrings
4. **Update changelog**: Document API changes

## Next Steps

- **[Architecture](architecture.md)** - System design
- **[Release Process](release-process.md)** - Version management
- **[Shared Library Guide](../core/shared-library.md)** - Detailed usage examples

## Related Documentation

- [Mojo Patterns](../core/mojo-patterns.md) - API usage patterns
- [Testing Strategy](../core/testing-strategy.md) - Testing APIs
- [Project Structure](../core/project-structure.md) - Code organization
