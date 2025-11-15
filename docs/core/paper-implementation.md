# Paper Implementation Guide

Complete workflow for implementing research papers in ML Odyssey.

## Overview

This guide walks you through the process of implementing a research paper, from initial planning to final benchmarking.
Each paper implementation follows a standardized structure to ensure consistency, reproducibility, and quality.

## Directory Structure

Every paper implementation follows this structure:

```text
papers/<paper-name>/
├── README.md           # Paper overview and implementation notes
├── model.mojo          # Neural network architecture
├── train.mojo          # Training script
├── evaluate.mojo       # Evaluation and benchmarking
├── config.toml         # Hyperparameters from paper
├── tests/              # Paper-specific tests
│   ├── test_model.mojo
│   ├── test_training.mojo
│   └── test_evaluation.mojo
└── results/            # Experimental results
    ├── metrics.json
    └── plots/
```

## Implementation Workflow

### Phase 1: Planning

Before writing code, understand the paper thoroughly.

#### 1.1 Read the Paper

Focus on these sections:

- **Abstract & Introduction**: Understand the problem and contribution
- **Method**: Core algorithm and architecture details
- **Experiments**: Dataset, hyperparameters, evaluation metrics
- **Results**: Expected performance benchmarks

#### 1.2 Create Implementation Plan

Document your understanding in `papers/<paper-name>/README.md`:

```markdown
# Paper Title

## Citation

[Full paper citation]

## Overview

Brief summary of the paper's contribution.

## Architecture

Detailed description of the network architecture.

## Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning rate | 0.01 | As specified in paper |
| Batch size | 32 | |
| Epochs | 100 | |

## Dataset

- Name: MNIST
- Training samples: 60,000
- Test samples: 10,000
- Input size: 28x28 grayscale

## Expected Results

Target accuracy: 99.2% on MNIST test set

## Implementation Notes

Challenges, deviations from paper, design decisions.
```

#### 1.3 Set Up Configuration

Create `config.toml` with all hyperparameters:

```toml
[model]
name = "LeNet5"
input_size = [1, 28, 28]
num_classes = 10

[training]
epochs = 100
batch_size = 32
learning_rate = 0.01
momentum = 0.9
weight_decay = 0.0005

[optimizer]
type = "SGD"
lr_schedule = "step"
step_size = 30
gamma = 0.1

[data]
dataset = "MNIST"
train_split = 0.9
val_split = 0.1
augmentation = false

[evaluation]
metrics = ["accuracy", "loss"]
save_predictions = true
```

### Phase 2: Model Implementation

Implement the neural network architecture.

#### 2.1 Create Model Structure

In `model.mojo`:

```mojo
from shared.core import Module, Sequential, Conv2D, MaxPool2D, Linear, ReLU, Tanh
from shared.core.types import Tensor

struct LeNet5(Module):
    """
    LeNet-5 architecture from 'Gradient-Based Learning Applied to Document Recognition'
    LeCun et al., 1998

    Architecture:
        INPUT[1x28x28] -> CONV1[6x28x28] -> POOL1[6x14x14] ->
        CONV2[16x10x10] -> POOL2[16x5x5] -> FC1[120] ->
        FC2[84] -> OUTPUT[10]
    """

    var conv1: Conv2D
    var pool1: MaxPool2D
    var conv2: Conv2D
    var pool2: MaxPool2D
    var fc1: Linear
    var fc2: Linear
    var fc3: Linear

    fn __init__(inout self):
        """Initialize LeNet-5 layers."""

        # Convolutional layers
        self.conv1 = Conv2D(in_channels=1, out_channels=6, kernel_size=5)
        self.pool1 = MaxPool2D(kernel_size=2, stride=2)
        self.conv2 = Conv2D(in_channels=6, out_channels=16, kernel_size=5)
        self.pool2 = MaxPool2D(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = Linear(input_size=16*5*5, output_size=120)
        self.fc2 = Linear(input_size=120, output_size=84)
        self.fc3 = Linear(input_size=84, output_size=10)

    fn forward(inout self, borrowed input: Tensor) -> Tensor:
        """Forward pass through LeNet-5."""

        # Conv block 1
        var x = self.conv1.forward(input)
        x = tanh(x)  # Original paper uses tanh
        x = self.pool1.forward(x)

        # Conv block 2
        x = self.conv2.forward(x)
        x = tanh(x)
        x = self.pool2.forward(x)

        # Flatten
        x = x.reshape(x.shape[0], -1)

        # Fully connected layers
        x = self.fc1.forward(x)
        x = tanh(x)
        x = self.fc2.forward(x)
        x = tanh(x)
        x = self.fc3.forward(x)

        return x

    fn parameters(inout self) -> List[Tensor]:
        """Get all trainable parameters."""
        var params = List[Tensor]()
        params.extend(self.conv1.parameters())
        params.extend(self.conv2.parameters())
        params.extend(self.fc1.parameters())
        params.extend(self.fc2.parameters())
        params.extend(self.fc3.parameters())
        return params
```

#### 2.2 Verify Architecture

Create tests to verify the model structure:

```mojo
# In tests/test_model.mojo
from testing import assert_equal, assert_true
from model import LeNet5

fn test_output_shape():
    """Verify output shape is correct."""
    var model = LeNet5()
    var input = Tensor.randn(1, 1, 28, 28)
    var output = model.forward(input)

    assert_equal(output.shape[0], 1)   # Batch size
    assert_equal(output.shape[1], 10)  # Num classes

fn test_parameter_count():
    """Verify number of parameters matches paper."""
    var model = LeNet5()
    var params = model.parameters()

    var total_params = 0
    for param in params:
        total_params += param.numel()

    # LeNet-5 has ~60K parameters
    assert_true(total_params > 50000 and total_params < 70000)
```

### Phase 3: Training Implementation

Implement the training pipeline.

#### 3.1 Create Training Script

In `train.mojo`:

```mojo
from shared.training import Trainer, SGD, CrossEntropyLoss
from shared.training.callbacks import ModelCheckpoint, EarlyStopping
from shared.training.schedulers import StepLR
from shared.data import BatchLoader
from shared.utils import Config, set_seed, Logger
from model import LeNet5
from data import load_mnist

fn main() raises:
    """Train LeNet-5 on MNIST."""

    # Load configuration
    var config = Config.from_file("config.toml")
    var logger = Logger("lenet5.train")

    # Reproducibility
    set_seed(config.get[Int]("seed", default=42))

    logger.info("Loading MNIST dataset...")
    var train_data, val_data = load_mnist(
        train=True,
        val_split=config.get[Float64]("data.val_split")
    )

    var train_loader = BatchLoader(
        train_data,
        batch_size=config.get[Int]("training.batch_size"),
        shuffle=True
    )

    var val_loader = BatchLoader(
        val_data,
        batch_size=config.get[Int]("training.batch_size"),
        shuffle=False
    )

    # Create model
    logger.info("Creating LeNet-5 model...")
    var model = LeNet5()

    # Optimizer (as specified in paper)
    var optimizer = SGD(
        learning_rate=config.get[Float64]("training.learning_rate"),
        momentum=config.get[Float64]("training.momentum"),
        weight_decay=config.get[Float64]("training.weight_decay")
    )

    # Learning rate scheduler
    var scheduler = StepLR(
        step_size=config.get[Int]("optimizer.step_size"),
        gamma=config.get[Float64]("optimizer.gamma")
    )

    # Loss function
    var loss_fn = CrossEntropyLoss()

    # Create trainer
    var trainer = Trainer(model, optimizer, loss_fn)

    # Add callbacks
    trainer.add_callback(
        ModelCheckpoint(
            filepath="results/best_model.mojo",
            monitor="val_accuracy",
            mode="max",
            save_best_only=True
        )
    )

    trainer.add_callback(
        EarlyStopping(patience=10, min_delta=0.0001)
    )

    # Train
    logger.info("Starting training...")
    var history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.get[Int]("training.epochs"),
        scheduler=scheduler,
        verbose=True
    )

    # Save training history
    history.save("results/training_history.json")

    logger.info("Training complete!")
```

#### 3.2 Implement Data Loading

Create `data.mojo` for dataset-specific logic:

```mojo
from shared.data import TensorDataset, download_dataset
from shared.data.transforms import Normalize, Compose
from shared.utils import train_test_split

fn load_mnist(train: Bool = True, val_split: Float64 = 0.1) raises -> (TensorDataset, TensorDataset):
    """Load and preprocess MNIST dataset."""

    # Download dataset
    var images, labels = download_dataset("MNIST", train=train)

    # Normalize to [0, 1]
    images = images.astype[Float32]() / 255.0

    # Split into train and validation
    var train_images, val_images, train_labels, val_labels = train_test_split(
        images, labels, test_size=val_split, shuffle=True
    )

    var train_data = TensorDataset(train_images, train_labels)
    var val_data = TensorDataset(val_images, val_labels)

    return train_data, val_data
```

### Phase 4: Evaluation Implementation

Implement comprehensive evaluation.

#### 4.1 Create Evaluation Script

In `evaluate.mojo`:

```mojo
from shared.training import evaluate_model
from shared.utils import load_model, Config, Logger
from shared.utils.visualization import plot_confusion_matrix, plot_training_curves
from model import LeNet5
from data import load_mnist

fn main() raises:
    """Evaluate LeNet-5 on MNIST test set."""

    var config = Config.from_file("config.toml")
    var logger = Logger("lenet5.evaluate")

    # Load test data
    logger.info("Loading test data...")
    var _, test_data = load_mnist(train=False)

    # Load trained model
    logger.info("Loading model...")
    var model = load_model[LeNet5]("results/best_model.mojo")

    # Evaluate
    logger.info("Evaluating model...")
    var metrics = evaluate_model(model, test_data)

    # Print results
    logger.info("=" * 50)
    logger.info("Test Results:")
    logger.info("  Accuracy:  {:.2f}%".format(metrics.accuracy * 100))
    logger.info("  Precision: {:.2f}%".format(metrics.precision * 100))
    logger.info("  Recall:    {:.2f}%".format(metrics.recall * 100))
    logger.info("  F1 Score:  {:.2f}".format(metrics.f1_score))
    logger.info("=" * 50)

    # Compare with paper
    var paper_accuracy = config.get[Float64]("evaluation.target_accuracy")
    var diff = metrics.accuracy - paper_accuracy

    if abs(diff) < 0.01:  # Within 1%
        logger.info("✓ Accuracy matches paper!")
    else:
        logger.warning("⚠ Accuracy differs from paper by {:.2f}%".format(diff * 100))

    # Save metrics
    metrics.save("results/test_metrics.json")

    # Visualizations
    logger.info("Creating visualizations...")
    plot_confusion_matrix(
        metrics.confusion_matrix,
        class_names=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
        save_path="results/plots/confusion_matrix.png"
    )

    # Plot training curves
    var history = load_history("results/training_history.json")
    plot_training_curves(
        history.train_losses,
        history.val_losses,
        history.train_accuracies,
        history.val_accuracies,
        save_path="results/plots/training_curves.png"
    )

    logger.info("Evaluation complete!")
```

### Phase 5: Testing

Write comprehensive tests for your implementation.

#### 5.1 Model Tests

```mojo
# tests/test_model.mojo
fn test_forward_pass():
    """Test forward pass produces correct output shape."""
    var model = LeNet5()
    var input = Tensor.randn(8, 1, 28, 28)  # Batch of 8
    var output = model.forward(input)

    assert_equal(output.shape, [8, 10])

fn test_gradient_flow():
    """Test gradients flow through all layers."""
    var model = LeNet5()
    var input = Tensor.randn(1, 1, 28, 28)
    var target = Tensor.zeros(1, 10)
    target[0, 5] = 1.0  # One-hot encoding

    var output = model.forward(input)
    var loss = cross_entropy_loss(output, target)
    loss.backward()

    # Check all parameters have gradients
    for param in model.parameters():
        assert_true(param.grad is not None)
        assert_true(param.grad.abs().sum() > 0)
```

#### 5.2 Training Tests

```mojo
# tests/test_training.mojo
fn test_overfitting_small_dataset():
    """Test model can overfit a small dataset (sanity check)."""
    var model = LeNet5()
    var optimizer = SGD(lr=0.01)

    # Small dataset (should overfit easily)
    var X = Tensor.randn(10, 1, 28, 28)
    var y = Tensor.randint(0, 10, shape=(10,))

    # Train for many epochs
    for epoch in range(100):
        var output = model.forward(X)
        var loss = cross_entropy_loss(output, y)
        loss.backward()
        optimizer.step(model.parameters())

    # Should achieve near-perfect accuracy
    var final_output = model.forward(X)
    var predictions = final_output.argmax(dim=1)
    var accuracy = (predictions == y).mean()

    assert_true(accuracy > 0.9, "Model should overfit small dataset")
```

### Phase 6: Documentation

Update README.md with results and notes.

```markdown
## Results

### MNIST Test Set

| Metric | Our Implementation | Paper |
|--------|-------------------|-------|
| Accuracy | 99.17% | 99.20% |
| Test Loss | 0.034 | - |

### Training Time

- Hardware: CPU (Intel i7)
- Time per epoch: ~2 minutes
- Total training time: ~3 hours (100 epochs)

### Observations

1. **Convergence**: Model converges faster with Adam optimizer (not in paper)
2. **Data Augmentation**: Adding random rotations improves accuracy by 0.2%
3. **Batch Size**: Paper uses full-batch gradient descent; we use mini-batches

### Deviations from Paper

- Used ReLU instead of tanh (better performance)
- Added batch normalization (not in original paper)
- Used Adam optimizer instead of SGD

## Reproducing Results

```bash

# Train model

pixi run mojo run papers/lenet5/train.mojo

# Evaluate

pixi run mojo run papers/lenet5/evaluate.mojo

# Run tests

pixi run pytest tests/papers/lenet5/

```

## Common Implementation Patterns

### Pattern: Using Pretrained Components

```mojo

# Use shared library components

from shared.core import ResNet18Backbone

var backbone = ResNet18Backbone(pretrained=True)
var model = MyCustomModel(backbone)

```

### Pattern: Custom Loss Function

```mojo

struct CustomLoss:
    fn __call__(self, output: Tensor, target: Tensor) -> Tensor:
        # Implement paper-specific loss
        var base_loss = cross_entropy_loss(output, target)
        var regularization = self.compute_regularization()
        return base_loss + regularization

```

### Pattern: Multi-GPU Training

```mojo

from shared.training import DistributedTrainer

var trainer = DistributedTrainer(
    model=model,
    optimizer=optimizer,
    num_gpus=4
)

trainer.train(train_loader, epochs=100)

```

## Troubleshooting

### Results Don't Match Paper

**Check these common issues:**

1. **Hyperparameters**: Verify all hyperparameters match the paper
2. **Initialization**: Check weight initialization scheme
3. **Data Preprocessing**: Verify normalization, augmentation
4. **Batch Size**: Different batch sizes can affect results
5. **Random Seed**: Set seed for reproducibility

### Training Doesn't Converge

**Try these solutions:**

1. **Learning Rate**: Reduce learning rate by 10x
2. **Gradient Clipping**: Add gradient clipping to prevent exploding gradients
3. **Batch Normalization**: Add batch norm for stability
4. **Weight Initialization**: Use proper initialization (Xavier, He, etc.)

### Out of Memory

**Reduce memory usage:**

1. **Batch Size**: Reduce batch size
2. **Model Size**: Use smaller model variant
3. **Gradient Accumulation**: Accumulate gradients over multiple batches

## Next Steps

- **[Testing Strategy](testing-strategy.md)** - Write comprehensive tests
- **[Performance Guide](../advanced/performance.md)** - Optimize your implementation
- **[Custom Layers](../advanced/custom-layers.md)** - Implement paper-specific components
- **[Workflow](workflow.md)** - Understand the 5-phase development process

## Related Documentation

- [Shared Library Guide](shared-library.md) - Reusable components
- [Project Structure](project-structure.md) - Repository organization
- [Configuration](configuration.md) - Managing hyperparameters
