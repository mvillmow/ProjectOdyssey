# Configuration Guide

Managing environment, project, and model configuration in ML Odyssey.

## Overview

ML Odyssey uses a layered configuration system that separates environment setup, project settings, and
model hyperparameters. This guide covers all configuration aspects from installation to production deployment.

## Configuration Layers

```text
Environment (pixi.toml)
    ↓
Project (configs/*.toml)
    ↓
Model/Paper (papers/*/config.toml)
    ↓
Runtime (CLI arguments, environment variables)
```

## Environment Configuration

### Pixi Configuration (`pixi.toml`)

Main environment configuration at repository root:

```toml
[project]
name = "ml-odyssey"
version = "0.1.0"
description = "Mojo-based AI research platform"
channels = ["conda-forge", "https://conda.modular.com/max"]
platforms = ["linux-64", "osx-arm64", "osx-64"]

[dependencies]
max = ">=24.5.0,<25"
python = ">=3.10,<3.13"
pytest = ">=7.4.0"
pre-commit = ">=3.3.0"

[tasks]
# Test tasks
test = "pytest tests/"
test-shared = "pytest tests/shared/"
test-papers = "pytest tests/papers/"

# Format tasks
format = "mojo format shared/ papers/"
format-check = "mojo format --check shared/ papers/"

# Lint tasks
lint = "pre-commit run --all-files"

# Build tasks
build = "mojo build"
build-release = "mojo build --release"

[feature.dev.dependencies]
ipython = "*"
jupyter = "*"
matplotlib = "*"

[feature.docs.dependencies]
mkdocs = "*"
mkdocs-material = "*"

[environments]
default = ["dev"]
docs = ["docs"]
ci = []
```

### Installing Pixi

```bash
# Install Pixi
curl -fsSL https://pixi.sh/install.sh | bash

# Reload shell
source ~/.bashrc  # or ~/.zshrc

# Verify installation
pixi --version
```

### Managing Dependencies

```bash
# Install all dependencies
pixi install

# Add new dependency
pixi add numpy

# Add dev dependency
pixi add --feature dev ipython

# Update dependencies
pixi update

# Remove dependency
pixi remove numpy
```

### Custom Tasks

Define reusable commands:

```toml
[tasks]
# Paper-specific training
train-lenet5 = "mojo run papers/lenet5/train.mojo"
eval-lenet5 = "mojo run papers/lenet5/evaluate.mojo"

# Benchmarking
bench-all = "mojo run benchmarks/run_all.mojo"
bench-layers = "mojo run benchmarks/layers.mojo"

# Data preparation
download-mnist = "python scripts/download_mnist.py"
download-cifar = "python scripts/download_cifar.py"
```

Run tasks:

```bash
pixi run test
pixi run train-lenet5
pixi run bench-all
```

## Project Configuration

### Configuration Directory (`configs/`)

Project-wide settings for different environments:

```text
configs/
├── dev.toml        # Development settings
├── test.toml       # Testing settings
└── prod.toml       # Production settings
```

### Development Configuration (`configs/dev.toml`)

```toml
[project]
name = "ml-odyssey"
environment = "development"

[logging]
level = "DEBUG"
format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
file = "logs/dev.log"

[data]
root_dir = "data/"
cache_dir = ".cache/"
download_on_demand = true

[training]
default_epochs = 10
checkpoint_interval = 5
early_stopping_patience = 3

[performance]
num_workers = 4
enable_profiling = true
memory_limit_gb = 8
```

### Testing Configuration (`configs/test.toml`)

```toml
[project]
name = "ml-odyssey"
environment = "testing"

[logging]
level = "WARNING"
format = "%(levelname)s - %(message)s"
file = null  # No file logging in tests

[data]
root_dir = "tests/fixtures/data/"
cache_dir = "tests/.cache/"
download_on_demand = false  # Use fixtures

[training]
default_epochs = 2  # Fast for tests
checkpoint_interval = 1
early_stopping_patience = 1

[performance]
num_workers = 1  # Single thread for tests
enable_profiling = false
memory_limit_gb = 2
```

### Production Configuration (`configs/prod.toml`)

```toml
[project]
name = "ml-odyssey"
environment = "production"

[logging]
level = "INFO"
format = "%(asctime)s - %(levelname)s - %(message)s"
file = "/var/log/ml-odyssey/prod.log"

[data]
root_dir = "/data/ml-odyssey/"
cache_dir = "/data/ml-odyssey/.cache/"
download_on_demand = false  # Pre-downloaded

[training]
default_epochs = 100
checkpoint_interval = 10
early_stopping_patience = 10

[performance]
num_workers = 8
enable_profiling = false
memory_limit_gb = 64
```

### Loading Configuration

```mojo
from shared.utils import Config

fn main() raises:
    # Load config based on environment
    var env = os.getenv("ML_ENV", "dev")
    var config = Config.from_file("configs/" + env + ".toml")

    # Access settings
    var log_level = config.get[String]("logging.level")
    var data_dir = config.get[String]("data.root_dir")
    var epochs = config.get[Int]("training.default_epochs")

    print("Environment:", env)
    print("Log level:", log_level)
    print("Data directory:", data_dir)
```

## Model Configuration

### Paper Configuration (`papers/lenet5/config.toml`)

Hyperparameters and settings for specific paper implementations:

```toml
[model]
name = "LeNet-5"
architecture = "lenet5"
input_shape = [1, 28, 28]
num_classes = 10

[dataset]
name = "MNIST"
train_size = 60000
test_size = 10000
normalize = true
augmentation = false

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

[loss]
type = "CrossEntropy"

[evaluation]
metrics = ["accuracy", "loss", "confusion_matrix"]
save_predictions = false
target_accuracy = 0.992  # From paper

[checkpointing]
save_best_only = true
monitor = "val_accuracy"
mode = "max"
filepath = "results/best_model.mojo"

[early_stopping]
enabled = true
patience = 10
min_delta = 0.0001
monitor = "val_loss"
mode = "min"

[reproducibility]
seed = 42
deterministic = true

[logging]
log_interval = 100  # Log every 100 batches
tensorboard = true
tensorboard_dir = "runs/lenet5"
```

### Using Model Configuration

```mojo
from shared.utils import Config
from shared.training import Trainer, SGD
from shared.data import load_dataset, BatchLoader
from model import LeNet5

fn main() raises:
    # Load model config
    var config = Config.from_file("config.toml")

    # Set seed for reproducibility
    if config.get[Bool]("reproducibility.deterministic"):
        set_seed(config.get[Int]("reproducibility.seed"))

    # Load dataset
    var train_data, val_data = load_dataset(
        name=config.get[String]("dataset.name"),
        normalize=config.get[Bool]("dataset.normalize"),
    )

    # Create data loaders
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
    var model = LeNet5(
        input_shape=config.get[List[Int]]("model.input_shape"),
        num_classes=config.get[Int]("model.num_classes")
    )

    # Create optimizer
    var optimizer = SGD(
        learning_rate=config.get[Float64]("training.learning_rate"),
        momentum=config.get[Float64]("training.momentum"),
        weight_decay=config.get[Float64]("training.weight_decay")
    )

    # Create trainer
    var trainer = Trainer(model, optimizer, loss_fn)

    # Add callbacks from config
    if config.get[Bool]("early_stopping.enabled"):
        trainer.add_callback(EarlyStopping(
            patience=config.get[Int]("early_stopping.patience"),
            min_delta=config.get[Float64]("early_stopping.min_delta")
        ))

    if config.get[Bool]("checkpointing.save_best_only"):
        trainer.add_callback(ModelCheckpoint(
            filepath=config.get[String]("checkpointing.filepath"),
            monitor=config.get[String]("checkpointing.monitor"),
            mode=config.get[String]("checkpointing.mode")
        ))

    # Train
    trainer.train(
        train_loader,
        val_loader,
        epochs=config.get[Int]("training.epochs")
    )
```

## Runtime Configuration

### Command-Line Arguments

Override config values at runtime:

```mojo
from shared.utils import Config, parse_args

fn main() raises:
    var config = Config.from_file("config.toml")

    # Parse command-line arguments
    var args = parse_args()

    # Override from CLI
    if args.has("lr"):
        config.set("training.learning_rate", args.get[Float64]("lr"))

    if args.has("epochs"):
        config.set("training.epochs", args.get[Int]("epochs"))

    if args.has("batch_size"):
        config.set("training.batch_size", args.get[Int]("batch_size"))

    # Use overridden config
    train_model(config)
```

Run with overrides:

```bash
# Override learning rate and epochs
pixi run mojo run train.mojo --lr 0.001 --epochs 50

# Override batch size
pixi run mojo run train.mojo --batch_size 64

# Multiple overrides
pixi run mojo run train.mojo --lr 0.001 --epochs 50 --batch_size 64
```

### Environment Variables

Set configuration through environment variables:

```bash
# Set environment
export ML_ENV=prod

# Set data directory
export ML_ODYSSEY_DATA_DIR=/data/ml-odyssey

# Set log level
export ML_ODYSSEY_LOG_LEVEL=DEBUG

# Run with environment variables
pixi run mojo run train.mojo
```

Load in code:

```mojo
from os import getenv

fn main() raises:
    var env = getenv("ML_ENV", "dev")
    var data_dir = getenv("ML_ODYSSEY_DATA_DIR", "data/")
    var log_level = getenv("ML_ODYSSEY_LOG_LEVEL", "INFO")

    print("Environment:", env)
    print("Data directory:", data_dir)
    print("Log level:", log_level)
```

### Configuration Priority

Priority order (highest to lowest):

1. **Command-line arguments** (highest priority)
2. **Environment variables**
3. **Model config** (`papers/*/config.toml`)
4. **Project config** (`configs/*.toml`)
5. **Defaults in code** (lowest priority)

Example with all layers:

```mojo
fn get_learning_rate(config: Config) -> Float64:
    # 1. Check CLI args (highest priority)
    if has_arg("lr"):
        return get_arg[Float64]("lr")

    # 2. Check environment variable
    if has_env("ML_ODYSSEY_LR"):
        return parse_float(getenv("ML_ODYSSEY_LR"))

    # 3. Check model config
    if config.has("training.learning_rate"):
        return config.get[Float64]("training.learning_rate")

    # 4. Project config would be loaded in config object

    # 5. Default value (lowest priority)
    return 0.01
```

## Configuration Best Practices

### DO

- ✅ Use TOML for configuration files (human-readable)
- ✅ Separate environment, project, and model configs
- ✅ Document all configuration options
- ✅ Provide sensible defaults
- ✅ Validate configuration at startup
- ✅ Use type-safe config access

### DON'T

- ❌ Hardcode values in code
- ❌ Mix different concerns in one config file
- ❌ Commit secrets or credentials
- ❌ Use different formats (stick to TOML)
- ❌ Skip validation of loaded config

## Configuration Examples

### Example 1: Hyperparameter Sweep

```mojo
fn run_sweep() raises:
    var base_config = Config.from_file("config.toml")

    # Sweep over learning rates
    var lrs = [0.1, 0.01, 0.001, 0.0001]

    for lr in lrs:
        var config = base_config.copy()
        config.set("training.learning_rate", lr)

        print("Training with lr =", lr)
        train_model(config)
```

### Example 2: Multi-GPU Configuration

```toml
[distributed]
enabled = true
backend = "nccl"
world_size = 4
rank = 0

[gpu]
device_ids = [0, 1, 2, 3]
mixed_precision = true
gradient_accumulation_steps = 2
```

### Example 3: Experiment Tracking

```toml
[experiment]
name = "lenet5-mnist-baseline"
project = "ml-odyssey"
tags = ["lenet5", "mnist", "baseline"]
notes = "Initial baseline run with paper hyperparameters"

[wandb]
enabled = true
project = "ml-odyssey"
entity = "your-team"
log_interval = 100
```

## Validating Configuration

Create a validation function:

```mojo
fn validate_config(config: Config) raises:
    """Validate configuration values."""

    # Required fields
    if not config.has("training.learning_rate"):
        raise Error("Missing required field: training.learning_rate")

    # Type validation
    var lr = config.get[Float64]("training.learning_rate")
    if lr <= 0.0 or lr > 1.0:
        raise Error("Learning rate must be in (0, 1], got " + str(lr))

    # Range validation
    var batch_size = config.get[Int]("training.batch_size")
    if batch_size < 1 or batch_size > 1024:
        raise Error("Batch size must be in [1, 1024], got " + str(batch_size))

    # Consistency checks
    if config.get[Bool]("early_stopping.enabled"):
        if not config.has("early_stopping.patience"):
            raise Error("early_stopping.patience required when early_stopping.enabled")

    print("✓ Configuration validated successfully")

fn main() raises:
    var config = Config.from_file("config.toml")
    validate_config(config)
    train_model(config)
```

## Configuration Templates

### Template: New Paper Implementation

```toml
[model]
name = "YourModel"
architecture = "your_architecture"
input_shape = [3, 224, 224]
num_classes = 1000

[dataset]
name = "ImageNet"
train_size = 1281167
val_size = 50000
normalize = true
augmentation = true

[training]
epochs = 90
batch_size = 256
learning_rate = 0.1
momentum = 0.9
weight_decay = 0.0001

[optimizer]
type = "SGD"
lr_schedule = "multistep"
milestones = [30, 60, 80]
gamma = 0.1

[evaluation]
metrics = ["top1_accuracy", "top5_accuracy"]
target_top1_accuracy = 0.769  # From paper

[reproducibility]
seed = 42
deterministic = true
```

## Troubleshooting

### Config Not Found

```bash
# Verify file exists
ls -la configs/dev.toml

# Check current directory
pwd

# Use absolute path
var config = Config.from_file("/path/to/ml-odyssey/configs/dev.toml")
```

### Invalid TOML Syntax

```bash
# Validate TOML syntax
python -c "import toml; toml.load('config.toml')"

# Or use online validator
# https://www.toml-lint.com/
```

### Type Mismatch

```mojo
# Wrong type
var lr = config.get[Int]("training.learning_rate")  # Error: lr is Float64

# Correct type
var lr = config.get[Float64]("training.learning_rate")

# Or use dynamic type
var lr = config.get("training.learning_rate")  # Returns variant type
```

## Next Steps

- **[Workflow](workflow.md)** - Using configuration in development workflow
- **[Paper Implementation](paper-implementation.md)** - Model-specific configuration
- **[Testing Strategy](testing-strategy.md)** - Testing with different configs
- **[Performance Guide](../advanced/performance.md)** - Performance-related configuration

## Related Documentation

- [Project Structure](project-structure.md) - Configuration file locations
- [Shared Library](shared-library.md) - Config utilities
- [First Model Tutorial](../getting-started/first_model.md) - Configuration in practice
