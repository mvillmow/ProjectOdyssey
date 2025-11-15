# Shared Library Guide

Understanding and using ML Odyssey's reusable components across all paper implementations.

> **📚 Detailed Reference**: For complete API documentation, advanced patterns, and examples, see
> [Shared Library API Appendix](../appendices/shared-library-api.md).

## Overview

The shared library (`shared/`) contains battle-tested, reusable components for building neural networks, training
models, and processing data. It's designed to eliminate boilerplate and let you focus on implementing paper-specific
architectures.

## Library Structure

The shared library is organized into four main modules:

```text
shared/
├── core/         # Building blocks for neural networks
├── training/     # Training infrastructure and optimization
├── data/         # Data processing and loading
└── utils/        # Utilities and helpers
```

## Core Module (`shared/core/`)

Fundamental building blocks for neural network architectures.

### Layers

Neural network layers with automatic gradient computation:

```mojo
from shared.core import Layer, Linear, Conv2D, MaxPool2D

# Fully connected layer
var fc = Layer("linear", input_size=784, output_size=128)

# Convolutional layer
var conv = Conv2D(
    in_channels=1,
    out_channels=32,
    kernel_size=3,
    stride=1,
    padding=1
)

# Pooling layer
var pool = MaxPool2D(kernel_size=2, stride=2)
```

**Available Layers**:

- `Linear` - Fully connected layer
- `Conv2D` - 2D convolution
- `Conv3D` - 3D convolution (for video/volumetric data)
- `MaxPool2D` / `AvgPool2D` - Pooling layers
- `BatchNorm2D` - Batch normalization
- `Dropout` - Dropout regularization
- `Embedding` - Embedding layer for discrete inputs

### Activation Functions

Non-linear transformations:

```mojo
from shared.core import ReLU, Sigmoid, Tanh, Softmax, LeakyReLU

# Standard activations
var relu = ReLU()
var sigmoid = Sigmoid()
var tanh = Tanh()

# Output layer activation
var softmax = Softmax(dim=-1)

# Advanced activations
var leaky_relu = LeakyReLU(negative_slope=0.01)
```

### Sequential Container

Chain layers together:

```mojo
from shared.core import Sequential

var model = Sequential([
    Layer("linear", input_size=784, output_size=256),
    ReLU(),
    Layer("linear", input_size=256, output_size=128),
    ReLU(),
    Layer("linear", input_size=128, output_size=10),
    Softmax(),
])

# Forward pass
var output = model.forward(input_tensor)

# Get parameters
var params = model.parameters()
```

### Custom Modules

Create your own components by implementing the `Module` trait:

```mojo
from shared.core import Module, Tensor

struct MyCustomLayer(Module):
    var weight: Tensor
    var bias: Tensor

    fn __init__(inout self, input_size: Int, output_size: Int):
        self.weight = Tensor.randn(output_size, input_size)
        self.bias = Tensor.zeros(output_size)

    fn forward(inout self, borrowed input: Tensor) -> Tensor:
        return input @ self.weight.T + self.bias

    fn parameters(inout self) -> List[Tensor]:
        return [self.weight, self.bias]
```

### Tensor Operations

High-performance tensor operations with SIMD:

```mojo
from shared.core.ops import matmul, conv2d, batch_norm

# Matrix multiplication (SIMD optimized)
var result = matmul(a, b)

# Convolution
var output = conv2d(input, kernel, stride=1, padding=0)

# Batch normalization
var normalized = batch_norm(input, mean, variance, gamma, beta)
```

## Training Module (`shared/training/`)

Infrastructure for training neural networks.

### Optimizers

Gradient-based optimization algorithms:

```mojo
from shared.training import SGD, Adam, RMSprop, AdamW

# Stochastic Gradient Descent
var sgd = SGD(learning_rate=0.01, momentum=0.9, weight_decay=1e-4)

# Adam optimizer
var adam = Adam(
    learning_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8
)

# RMSprop
var rmsprop = RMSprop(learning_rate=0.001, alpha=0.99)

# Update parameters
for param in model.parameters():
    optimizer.step(param, param.grad)
```

### Loss Functions

Common loss functions for training:

```mojo
from shared.training import (
    CrossEntropyLoss,
    MSELoss,
    L1Loss,
    BCELoss,
)

# Classification
var ce_loss = CrossEntropyLoss()
var loss = ce_loss(predictions, targets)

# Regression
var mse_loss = MSELoss()
var loss = mse_loss(predictions, targets)

# Binary classification
var bce_loss = BCELoss()
var loss = bce_loss(predictions, targets)
```

### Trainer

High-level training orchestration:

```mojo
from shared.training import Trainer

# Create trainer
var trainer = Trainer(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn
)

# Train
trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=10,
    verbose=True
)

# Evaluate
var metrics = trainer.evaluate(test_loader)
```

### Callbacks

Customize training behavior:

```mojo
from shared.training.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateScheduler,
    TensorBoardLogger,
)

# Early stopping
trainer.add_callback(
    EarlyStopping(patience=5, min_delta=0.001)
)

# Save best model
trainer.add_callback(
    ModelCheckpoint(
        filepath="best_model.mojo",
        monitor="val_loss",
        save_best_only=True
    )
)

# Learning rate scheduling
from shared.training.schedulers import CosineAnnealingLR

var scheduler = CosineAnnealingLR(T_max=100, eta_min=1e-6)
trainer.add_callback(LearningRateScheduler(scheduler))
```

### Learning Rate Schedulers

Adjust learning rate during training:

```mojo
from shared.training.schedulers import (
    StepLR,
    ExponentialLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
)

# Step decay
var step_lr = StepLR(step_size=10, gamma=0.1)

# Exponential decay
var exp_lr = ExponentialLR(gamma=0.95)

# Cosine annealing
var cosine_lr = CosineAnnealingLR(T_max=100, eta_min=1e-6)

# Reduce on plateau
var plateau_lr = ReduceLROnPlateau(
    mode="min",
    factor=0.5,
    patience=3
)
```

## Data Module (`shared/data/`)

Data loading, preprocessing, and augmentation.

### Datasets

Abstract dataset interface:

```mojo
from shared.data import TensorDataset, Dataset

# Tensor dataset (in-memory)
var dataset = TensorDataset(X_train, y_train)

# Custom dataset
struct MyDataset(Dataset):
    fn __len__(self) -> Int:
        return self.num_samples

    fn __getitem__(self, idx: Int) -> (Tensor, Tensor):
        return self.load_sample(idx)
```

### Data Loaders

Efficient batching and iteration:

```mojo
from shared.data import BatchLoader

var loader = BatchLoader(
    dataset=train_data,
    batch_size=32,
    shuffle=True,
    drop_last=True,
    num_workers=4
)

# Iterate over batches
for batch in loader:
    var inputs, targets = batch
    # Process batch...
```

### Data Transforms

Preprocessing and augmentation:

```mojo
from shared.data.transforms import (
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    RandomRotation,
    Compose,
)

# Compose multiple transforms
var transform = Compose([
    RandomCrop(size=28, padding=4),
    RandomHorizontalFlip(p=0.5),
    RandomRotation(degrees=15),
    Normalize(mean=0.5, std=0.5),
])

# Apply to data
var augmented = transform(image)
```

### Samplers

Control how data is sampled:

```mojo
from shared.data.samplers import (
    RandomSampler,
    SequentialSampler,
    WeightedRandomSampler,
)

# Weighted sampling for imbalanced datasets
var weights = compute_class_weights(labels)
var sampler = WeightedRandomSampler(weights, num_samples=len(dataset))

var loader = BatchLoader(dataset, batch_size=32, sampler=sampler)
```

## Utils Module (`shared/utils/`)

Utilities for logging, profiling, visualization, and more.

### Configuration

Manage hyperparameters and settings:

```mojo
from shared.utils import Config

# Load config from file
var config = Config.from_file("configs/train.toml")

# Access settings
var lr = config.get[Float64]("optimizer.learning_rate")
var batch_size = config.get[Int]("data.batch_size")

# Override from command line
config.override_from_args()
```

### Logging

Structured logging:

```mojo
from shared.utils import Logger

var logger = Logger("training", level="INFO")

logger.info("Starting training...")
logger.debug("Batch size: ", batch_size)
logger.warning("Learning rate is high: ", lr)
logger.error("Training failed: ", error)
```

### Profiling

Performance measurement:

```mojo
from shared.utils import Profiler, profile

var profiler = Profiler()

with profiler.section("forward_pass"):
    var output = model.forward(input)

with profiler.section("backward_pass"):
    var loss = loss_fn(output, target)
    loss.backward()

# Print results
profiler.print_summary()
```

### Visualization

Plot training metrics:

```mojo
from shared.utils.visualization import (
    plot_training_curves,
    plot_confusion_matrix,
    plot_feature_maps,
)

# Training curves
plot_training_curves(
    train_losses=train_losses,
    val_losses=val_losses,
    save_path="training_curves.png"
)

# Confusion matrix
plot_confusion_matrix(
    cm=confusion_matrix,
    class_names=["cat", "dog"],
    save_path="confusion_matrix.png"
)
```

### Random Number Generation

Reproducible randomness:

```mojo
from shared.utils import set_seed, random_normal, random_uniform

# Set seed for reproducibility
set_seed(42)

# Generate random tensors
var normal = random_normal(shape=(100, 100), mean=0.0, std=1.0)
var uniform = random_uniform(shape=(100, 100), low=0.0, high=1.0)
```

## Common Patterns

### Pattern 1: Simple Classifier

```mojo
from shared.core import Sequential, Layer, ReLU, Softmax
from shared.training import Trainer, SGD, CrossEntropyLoss
from shared.data import TensorDataset, BatchLoader

# Model
var model = Sequential([
    Layer("linear", input_size=784, output_size=128),
    ReLU(),
    Layer("linear", input_size=128, output_size=10),
    Softmax(),
])

# Data
var train_loader = BatchLoader(train_data, batch_size=32)

# Training
var trainer = Trainer(model, SGD(lr=0.01), CrossEntropyLoss())
trainer.train(train_loader, epochs=10)
```

### Pattern 2: Convolutional Network

```mojo
from shared.core import Sequential, Conv2D, MaxPool2D, Linear, ReLU

var model = Sequential([
    Conv2D(in_channels=1, out_channels=32, kernel_size=3),
    ReLU(),
    MaxPool2D(kernel_size=2),
    Conv2D(in_channels=32, out_channels=64, kernel_size=3),
    ReLU(),
    MaxPool2D(kernel_size=2),
    Linear(input_size=64*5*5, output_size=10),
])
```

### Pattern 3: Custom Training Loop

```mojo
from shared.training import SGD, CrossEntropyLoss

var optimizer = SGD(lr=0.01)
var loss_fn = CrossEntropyLoss()

for epoch in range(num_epochs):
    for batch in train_loader:
        var inputs, targets = batch

        # Forward
        var outputs = model.forward(inputs)
        var loss = loss_fn(outputs, targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step(model.parameters())

        print("Loss: ", loss.item())
```

## Integration Examples

### Using with Paper Implementations

Paper implementations use the shared library as a foundation:

```mojo
# In papers/lenet5/model.mojo
from shared.core import Sequential, Conv2D, MaxPool2D, Linear, ReLU

struct LeNet5:
    var model: Sequential

    fn __init__(inout self):
        # Build LeNet-5 using shared components
        self.model = Sequential([
            Conv2D(1, 6, kernel_size=5),
            ReLU(),
            MaxPool2D(kernel_size=2),
            # ... rest of architecture
        ])
```

### Extending the Library

Add your own components to the shared library:

```mojo
# In shared/core/custom_layers.mojo
from shared.core import Module, Tensor

struct ResidualBlock(Module):
    """Residual block with skip connection."""

    var conv1: Conv2D
    var conv2: Conv2D

    fn forward(inout self, borrowed input: Tensor) -> Tensor:
        var out = self.conv1.forward(input)
        out = self.conv2.forward(out)
        return out + input  # Skip connection
```

## When to Use Each Module

### Core Module

Use for: Defining model architectures, creating layers, building custom modules

**Example use cases**:

- Implementing paper architectures
- Creating custom layers
- Building sequential models

### Training Module

Use for: Training loops, optimization, callbacks, learning rate scheduling

**Example use cases**:

- Training models end-to-end
- Implementing custom training strategies
- Adding callbacks for early stopping, checkpointing

### Data Module

Use for: Loading data, batching, preprocessing, augmentation

**Example use cases**:

- Loading datasets
- Data augmentation pipelines
- Custom data sampling strategies

### Utils Module

Use for: Configuration, logging, profiling, visualization

**Example use cases**:

- Managing hyperparameters
- Performance profiling
- Plotting training curves
- Reproducible experiments

## Performance Considerations

The shared library is optimized for performance:

- **SIMD Operations**: Core operations use SIMD for parallelism
- **Memory Efficiency**: Minimal allocations and copies
- **Type Safety**: Compile-time checking prevents runtime errors
- **Zero-Cost Abstractions**: High-level APIs with low-level performance

See [Performance Guide](../advanced/performance.md) for optimization techniques.

## See Also

- **[Shared Library API Appendix](../appendices/shared-library-api.md)** - Complete API reference
- **[Paper Implementation Guide](paper-implementation.md)** - Use shared library to implement papers
- **[Custom Layers](../advanced/custom-layers.md)** - Create custom components
- **[Mojo Patterns](mojo-patterns.md)** - Learn Mojo-specific patterns
- **[API Reference](../dev/api-reference.md)** - Complete API documentation
- [Project Structure](project-structure.md) - Repository organization
- [Testing Strategy](testing-strategy.md) - Testing shared library components
