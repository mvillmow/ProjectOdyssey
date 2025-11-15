# Configuration Reference

Complete reference for ML Odyssey's configuration system including all settings, environment variables, and
advanced configuration patterns.

> **Quick Reference**: For a concise guide, see [Configuration Guide](../core/configuration.md).

## Table of Contents

- [Environment Configuration](#environment-configuration)
- [Project Configuration](#project-configuration)
- [Model Configuration](#model-configuration)
- [Runtime Configuration](#runtime-configuration)
- [Configuration Validation](#configuration-validation)
- [Advanced Patterns](#advanced-patterns)
- [Environment Variables Reference](#environment-variables-reference)
- [Complete Examples](#complete-examples)
- [Troubleshooting](#troubleshooting)

## Environment Configuration

### Complete Pixi.toml Reference

```toml
[project]
name = "ml-odyssey"
version = "0.1.0"
description = "Mojo-based AI research platform"
authors = ["Your Name <your.email@example.com>"]
license = "MIT"
readme = "README.md"
homepage = "https://github.com/mvillmow/ml-odyssey"
repository = "https://github.com/mvillmow/ml-odyssey"
channels = ["conda-forge", "https://conda.modular.com/max"]
platforms = ["linux-64", "osx-arm64", "osx-64", "win-64"]

[dependencies]
# Core dependencies
max = ">=24.5.0,<25"
python = ">=3.10,<3.13"
pip = "*"

# Testing
pytest = ">=7.4.0"
pytest-cov = ">=4.1.0"
pytest-xdist = ">=3.3.0"  # Parallel test execution

# Code quality
pre-commit = ">=3.3.0"
ruff = ">=0.1.0"  # Fast Python linter
mypy = ">=1.5.0"  # Type checking

# Development tools
ipython = {version = "*", optional = true}
jupyter = {version = "*", optional = true}

[dependencies.python-packages]
# Additional Python packages via pip
numpy = ">=1.24.0"
scipy = ">=1.11.0"

[feature.dev.dependencies]
# Development-only dependencies
ipython = "*"
jupyter = "*"
jupyterlab = "*"
matplotlib = "*"
seaborn = "*"
pandas = "*"
rich = "*"  # Pretty terminal output

[feature.docs.dependencies]
# Documentation dependencies
mkdocs = ">=1.5.0"
mkdocs-material = ">=9.0.0"
mkdocstrings = {version = "*", extras = ["python"]}
mkdocs-mermaid2-plugin = "*"

[feature.gpu.dependencies]
# GPU support (optional)
cuda = {version = ">=11.8", channel = "nvidia"}
cudnn = {version = ">=8.6", channel = "nvidia"}

[tasks]
# === Test Tasks ===
test = "pytest tests/"
test-v = "pytest tests/ -v"
test-vv = "pytest tests/ -vv -s"
test-cov = "pytest tests/ --cov=shared --cov=papers --cov-report=html"
test-parallel = "pytest tests/ -n auto"  # Use all CPU cores
test-shared = "pytest tests/shared/"
test-papers = "pytest tests/papers/"
test-fast = "pytest tests/ -m 'not slow'"
test-slow = "pytest tests/ -m 'slow'"

# === Format Tasks ===
format = "mojo format shared/ papers/"
format-check = "mojo format --check shared/ papers/"
format-python = "ruff format scripts/ tests/"

# === Lint Tasks ===
lint = "pre-commit run --all-files"
lint-python = "ruff check scripts/ tests/"
lint-type = "mypy scripts/ tests/"
lint-markdown = "npx markdownlint-cli2 '**/*.md'"

# === Build Tasks ===
build = "mojo build"
build-release = "mojo build --release --optimization-level=3"
build-debug = "mojo build --debug"

# === Run Tasks ===
run-lenet5 = "mojo run papers/lenet5/train.mojo"
run-example = "mojo run examples/"

# === Benchmark Tasks ===
bench = "mojo run benchmarks/run_all.mojo"
bench-layers = "mojo run benchmarks/layers.mojo"
bench-training = "mojo run benchmarks/training.mojo"

# === Documentation Tasks ===
docs-serve = "mkdocs serve"
docs-build = "mkdocs build"
docs-deploy = "mkdocs gh-deploy"

# === Utility Tasks ===
clean = "rm -rf .pytest_cache __pycache__ .mypy_cache .ruff_cache build/ dist/"
clean-all = {cmd = "pixi run clean && rm -rf .pixi", depends_on = ["clean"]}
update = "pixi update"
lock = "pixi update --locked"

[environments]
default = {features = ["dev"], solve-group = "default"}
ci = {features = [], solve-group = "default"}
docs = {features = ["docs"], solve-group = "docs"}
gpu = {features = ["dev", "gpu"], solve-group = "gpu"}

[activation]
# Environment variables set when activating the environment
scripts = ["scripts/setup_env.sh"]

[target.linux-64.dependencies]
# Linux-specific dependencies

[target.osx-arm64.dependencies]
# macOS ARM-specific dependencies

[target.win-64.dependencies]
# Windows-specific dependencies
```

## Project Configuration

### Development Configuration Complete

```toml
# configs/dev.toml
[project]
name = "ml-odyssey"
environment = "development"
version = "0.1.0"

[logging]
level = "DEBUG"
format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
file = "logs/dev.log"
console = true
json_logs = false

# Log rotation
max_bytes = 10485760  # 10MB
backup_count = 5

[data]
root_dir = "data/"
cache_dir = ".cache/"
download_on_demand = true
preprocessing_workers = 4

# Dataset-specific settings
[data.mnist]
normalize = true
flatten = false
download = true

[data.cifar10]
normalize = true
augment = false
download = true

[training]
default_epochs = 10
checkpoint_interval = 5
early_stopping_patience = 3
gradient_clip_value = 5.0
mixed_precision = false

# Logging intervals
log_interval = 100  # Log every N batches
eval_interval = 500  # Evaluate every N batches

[validation]
enabled = true
split = 0.1  # 10% of training data
shuffle = true

[performance]
num_workers = 4
batch_prefetch = 2
pin_memory = true
enable_profiling = true
memory_limit_gb = 8

[debugging]
detect_anomaly = true  # Detect NaN/Inf in gradients
deterministic = true   # For reproducibility
benchmark = false      # Don't use cudnn benchmark mode

[paths]
models = "models/"
results = "results/"
logs = "logs/"
cache = ".cache/"
```

### Testing Configuration Complete

```toml
# configs/test.toml
[project]
name = "ml-odyssey"
environment = "testing"
version = "0.1.0"

[logging]
level = "WARNING"
format = "%(levelname)s - %(message)s"
file = null
console = true
json_logs = false

[data]
root_dir = "tests/fixtures/data/"
cache_dir = "tests/.cache/"
download_on_demand = false
preprocessing_workers = 1

[data.mnist]
normalize = true
flatten = false
download = false

[training]
default_epochs = 2
checkpoint_interval = 1
early_stopping_patience = 1
gradient_clip_value = 5.0
mixed_precision = false
log_interval = 10
eval_interval = 50

[validation]
enabled = true
split = 0.2
shuffle = false  # Deterministic for tests

[performance]
num_workers = 1
batch_prefetch = 1
pin_memory = false
enable_profiling = false
memory_limit_gb = 2

[debugging]
detect_anomaly = true
deterministic = true
benchmark = false

[paths]
models = "tests/tmp/models/"
results = "tests/tmp/results/"
logs = "tests/tmp/logs/"
cache = "tests/.cache/"

[test_timeouts]
unit = 5.0        # 5 seconds max per unit test
integration = 30.0  # 30 seconds max per integration test
e2e = 120.0       # 2 minutes max for end-to-end tests
```

### Production Configuration Complete

```toml
# configs/prod.toml
[project]
name = "ml-odyssey"
environment = "production"
version = "0.1.0"

[logging]
level = "INFO"
format = "%(asctime)s - %(levelname)s - %(message)s"
file = "/var/log/ml-odyssey/prod.log"
console = false
json_logs = true  # Structured logging for parsing

max_bytes = 104857600  # 100MB
backup_count = 10

[data]
root_dir = "/data/ml-odyssey/"
cache_dir = "/data/ml-odyssey/.cache/"
download_on_demand = false
preprocessing_workers = 8

[data.mnist]
normalize = true
flatten = false
download = false  # Pre-downloaded in production

[training]
default_epochs = 100
checkpoint_interval = 10
early_stopping_patience = 10
gradient_clip_value = 5.0
mixed_precision = true  # Enable for faster training
log_interval = 1000
eval_interval = 5000

[validation]
enabled = true
split = 0.1
shuffle = true

[performance]
num_workers = 8
batch_prefetch = 4
pin_memory = true
enable_profiling = false
memory_limit_gb = 64

[debugging]
detect_anomaly = false  # Disable in production
deterministic = false   # Allow non-deterministic for speed
benchmark = true        # Use cudnn benchmark mode

[paths]
models = "/data/ml-odyssey/models/"
results = "/data/ml-odyssey/results/"
logs = "/var/log/ml-odyssey/"
cache = "/data/ml-odyssey/.cache/"

[distributed]
enabled = false
backend = "nccl"
world_size = 1
rank = 0

[monitoring]
enabled = true
metrics_port = 9090
health_check_port = 8080
```

## Model Configuration

### Complete Paper Configuration Template

```toml
# papers/<paper-name>/config.toml

[paper]
title = "Full Paper Title"
authors = ["Author 1", "Author 2"]
year = 2024
venue = "Conference/Journal Name"
url = "https://arxiv.org/abs/..."
citation = """
@article{author2024paper,
  title={Paper Title},
  author={Author, First and Author, Second},
  journal={Journal},
  year={2024}
}
"""

[model]
name = "ModelName"
architecture = "model_architecture"
version = "1.0"
input_shape = [3, 224, 224]
input_type = "image"  # image, text, audio, etc.
num_classes = 1000
pretrained = false

# Architecture-specific parameters
[model.architecture]
num_layers = 50
hidden_dim = 512
num_heads = 8
dropout = 0.1

[dataset]
name = "ImageNet"
task = "classification"  # classification, regression, generation, etc.
train_size = 1281167
val_size = 50000
test_size = 50000
num_classes = 1000

# Data preprocessing
normalize = true
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Data augmentation
augmentation = true
random_crop = true
random_flip = true
random_rotation = 15
color_jitter = {brightness = 0.2, contrast = 0.2, saturation = 0.2, hue = 0.1}

[training]
# Basic training parameters
epochs = 90
batch_size = 256
learning_rate = 0.1
momentum = 0.9
weight_decay = 0.0001
gradient_clip = 5.0

# Advanced training settings
warmup_epochs = 5
warmup_lr = 0.01
mixed_precision = true
gradient_accumulation_steps = 1

# Validation
val_interval = 1  # Validate every N epochs
val_batch_size = 512

[optimizer]
type = "SGD"  # SGD, Adam, AdamW, RMSprop
nesterov = true

# Learning rate schedule
lr_schedule = "cosine"  # step, exponential, cosine, plateau
# Step schedule
step_size = 30
gamma = 0.1
# Cosine schedule
T_max = 90
eta_min = 0.0001
# Plateau schedule
patience = 5
factor = 0.5
threshold = 0.001

[loss]
type = "CrossEntropy"  # CrossEntropy, MSE, L1, BCE, etc.
label_smoothing = 0.1
reduction = "mean"

# Loss-specific parameters
[loss.parameters]
# Add loss-specific parameters here

[evaluation]
# Metrics to compute
metrics = ["accuracy", "top5_accuracy", "loss", "precision", "recall", "f1"]

# Evaluation settings
save_predictions = true
save_attention_maps = false
compute_confusion_matrix = true

# Target metrics from paper
target_accuracy = 0.769
target_top5_accuracy = 0.899

[checkpointing]
enabled = true
save_best_only = true
save_interval = 10  # Save every N epochs
monitor = "val_accuracy"
mode = "max"  # max or min
filepath = "results/{epoch:02d}-{val_accuracy:.4f}.mojo"
max_checkpoints = 5  # Keep only 5 best checkpoints

[early_stopping]
enabled = true
patience = 10
min_delta = 0.0001
monitor = "val_loss"
mode = "min"
restore_best_weights = true

[reproducibility]
seed = 42
deterministic = true
benchmark = false

[logging]
# Training logging
log_interval = 100  # Log every N batches
log_gradients = false
log_weights = false
log_activations = false

# Tensorboard
tensorboard = true
tensorboard_dir = "runs/{model_name}"

# Weights & Biases
wandb = false
wandb_project = "ml-odyssey"
wandb_entity = "your-team"
wandb_tags = ["paper-implementation", "model-name"]

[hardware]
# Device settings
device = "cuda"  # cuda, cpu, mps
gpu_ids = [0, 1, 2, 3]
distributed = false

# Memory settings
max_memory_mb = 8192
clear_cache_interval = 100  # Clear GPU cache every N batches

[profiling]
enabled = false
record_shapes = true
profile_memory = true
with_stack = true
output_dir = "profiling/"
```

## Runtime Configuration

### Command-Line Argument Parsing

Complete implementation:

```mojo
from sys import argv
from collections import Dict

struct Args:
    """Command-line argument parser."""
    var args: Dict[String, String]

    fn __init__(inout self):
        self.args = Dict[String, String]()
        self._parse()

    fn _parse(inout self):
        """Parse command-line arguments."""
        var i = 1
        while i < len(argv()):
            var arg = argv()[i]

            if arg.startswith("--"):
                var key = arg[2:]
                if i + 1 < len(argv()) and not argv()[i + 1].startswith("--"):
                    self.args[key] = argv()[i + 1]
                    i += 2
                else:
                    self.args[key] = "true"
                    i += 1
            else:
                i += 1

    fn has(borrowed self, key: String) -> Bool:
        """Check if argument exists."""
        return key in self.args

    fn get(borrowed self, key: String, default: String = "") -> String:
        """Get argument value with default."""
        if key in self.args:
            return self.args[key]
        return default

    fn get_int(borrowed self, key: String, default: Int = 0) -> Int:
        """Get argument as integer."""
        if key in self.args:
            return int(self.args[key])
        return default

    fn get_float(borrowed self, key: String, default: Float64 = 0.0) -> Float64:
        """Get argument as float."""
        if key in self.args:
            return float(self.args[key])
        return default

    fn get_bool(borrowed self, key: String, default: Bool = False) -> Bool:
        """Get argument as boolean."""
        if key in self.args:
            var value = self.args[key].lower()
            return value == "true" or value == "1" or value == "yes"
        return default

# Usage
fn main():
    var args = Args()

    # Get configuration file
    var config_path = args.get("config", "config.toml")
    var config = Config.from_file(config_path)

    # Override from command line
    if args.has("lr"):
        config.set("training.learning_rate", args.get_float("lr"))

    if args.has("epochs"):
        config.set("training.epochs", args.get_int("epochs"))

    if args.has("batch_size"):
        config.set("training.batch_size", args.get_int("batch_size"))

    if args.has("gpu"):
        config.set("hardware.gpu_ids", parse_gpu_list(args.get("gpu")))

    # Use config
    train_model(config)
```

## Environment Variables Reference

### Complete Environment Variable List

```bash
# === Core Settings ===
export ML_ENV=dev                    # Environment: dev, test, prod
export ML_ODYSSEY_VERSION=0.1.0      # Version override

# === Paths ===
export ML_ODYSSEY_ROOT=/path/to/ml-odyssey
export ML_ODYSSEY_DATA_DIR=/data/ml-odyssey
export ML_ODYSSEY_CACHE_DIR=/cache/ml-odyssey
export ML_ODYSSEY_LOG_DIR=/logs/ml-odyssey
export ML_ODYSSEY_MODEL_DIR=/models/ml-odyssey

# === Logging ===
export ML_ODYSSEY_LOG_LEVEL=DEBUG    # DEBUG, INFO, WARNING, ERROR
export ML_ODYSSEY_LOG_FORMAT=json    # text, json
export ML_ODYSSEY_LOG_FILE=/logs/ml-odyssey.log

# === Training ===
export ML_ODYSSEY_EPOCHS=100
export ML_ODYSSEY_BATCH_SIZE=32
export ML_ODYSSEY_LR=0.001
export ML_ODYSSEY_WEIGHT_DECAY=0.0001
export ML_ODYSSEY_SEED=42

# === Hardware ===
export ML_ODYSSEY_DEVICE=cuda        # cuda, cpu, mps
export ML_ODYSSEY_GPU_IDS=0,1,2,3    # Comma-separated GPU IDs
export ML_ODYSSEY_NUM_WORKERS=4
export ML_ODYSSEY_PIN_MEMORY=true

# === Distributed Training ===
export ML_ODYSSEY_DISTRIBUTED=true
export ML_ODYSSEY_WORLD_SIZE=4
export ML_ODYSSEY_RANK=0
export ML_ODYSSEY_MASTER_ADDR=localhost
export ML_ODYSSEY_MASTER_PORT=29500

# === Experiment Tracking ===
export ML_ODYSSEY_EXPERIMENT_NAME=my-experiment
export ML_ODYSSEY_WANDB_PROJECT=ml-odyssey
export ML_ODYSSEY_WANDB_ENTITY=your-team
export ML_ODYSSEY_TENSORBOARD_DIR=runs/

# === Debugging ===
export ML_ODYSSEY_DEBUG=true
export ML_ODYSSEY_PROFILE=true
export ML_ODYSSEY_DETECT_ANOMALY=true
```

## Configuration Validation

### Comprehensive Validation

```mojo
from shared.utils import Config

fn validate_training_config(config: Config) raises:
    """Validate training configuration."""

    # Required fields
    var required = [
        "training.learning_rate",
        "training.batch_size",
        "training.epochs",
        "model.architecture",
        "data.dataset",
    ]

    for field in required:
        if not config.has(field):
            raise Error("Missing required field: " + field)

    # Range validation
    var lr = config.get[Float64]("training.learning_rate")
    if lr <= 0.0 or lr > 1.0:
        raise Error("learning_rate must be in (0, 1], got " + str(lr))

    var batch_size = config.get[Int]("training.batch_size")
    if batch_size < 1 or batch_size > 2048:
        raise Error("batch_size must be in [1, 2048], got " + str(batch_size))

    var epochs = config.get[Int]("training.epochs")
    if epochs < 1:
        raise Error("epochs must be >= 1, got " + str(epochs))

    # Type validation
    var weight_decay = config.get[Float64]("training.weight_decay", default=0.0)
    if weight_decay < 0.0:
        raise Error("weight_decay must be >= 0, got " + str(weight_decay))

    # Consistency checks
    if config.get[Bool]("early_stopping.enabled", default=False):
        if not config.has("early_stopping.patience"):
            raise Error("early_stopping.patience required when early_stopping.enabled")

        var patience = config.get[Int]("early_stopping.patience")
        if patience < 1:
            raise Error("early_stopping.patience must be >= 1, got " + str(patience))

    # Optimizer validation
    var optimizer_type = config.get[String]("optimizer.type", default="SGD")
    var valid_optimizers = ["SGD", "Adam", "AdamW", "RMSprop"]
    if optimizer_type not in valid_optimizers:
        raise Error("Invalid optimizer type: " + optimizer_type +
                   ". Valid options: " + str(valid_optimizers))

    # LR schedule validation
    var lr_schedule = config.get[String]("optimizer.lr_schedule", default="constant")
    if lr_schedule == "step":
        if not config.has("optimizer.step_size"):
            raise Error("optimizer.step_size required for step schedule")
    elif lr_schedule == "cosine":
        if not config.has("optimizer.T_max"):
            raise Error("optimizer.T_max required for cosine schedule")

    print("✓ Configuration validated successfully")
```

## Advanced Patterns

### Hierarchical Configuration

```mojo
struct Config:
    """Hierarchical configuration with inheritance."""

    fn load_with_overrides(base_path: String, override_paths: List[String]) -> Config:
        """Load config with multiple override layers."""

        # Start with base config
        var config = Config.from_file(base_path)

        # Apply overrides in order
        for override_path in override_paths:
            var override_config = Config.from_file(override_path)
            config.merge(override_config)

        return config

# Usage
var config = Config.load_with_overrides(
    "configs/base.toml",
    ["configs/prod.toml", "configs/experiment.toml"]
)
```

### Dynamic Configuration

```mojo
fn create_sweep_configs(base_config: Config) -> List[Config]:
    """Create configs for hyperparameter sweep."""

    var configs = List[Config]()

    # Sweep over learning rates
    var lrs = [0.1, 0.01, 0.001, 0.0001]

    # Sweep over batch sizes
    var batch_sizes = [16, 32, 64, 128]

    for lr in lrs:
        for batch_size in batch_sizes:
            var config = base_config.copy()
            config.set("training.learning_rate", lr)
            config.set("training.batch_size", batch_size)
            config.set("experiment.name",
                      "lr_{}_bs_{}".format(lr, batch_size))
            configs.append(config)

    return configs

# Run sweep
var base_config = Config.from_file("config.toml")
var sweep_configs = create_sweep_configs(base_config)

for config in sweep_configs:
    print("Running experiment:", config.get[String]("experiment.name"))
    train_model(config)
```

This appendix provides complete configuration reference. For quick setup,
see [Configuration Guide](../core/configuration.md).
