# Configs Directory Architecture

## Overview

The configs/ directory provides centralized configuration management for ML Odyssey, supporting reproducible experiments, paper implementations, and shared components. This document defines the architecture, standards, and patterns for configuration management.

## Design Principles

1. **Hierarchy and Inheritance**: Three-tier configuration hierarchy (defaults → paper-specific → experiment)
2. **Type Safety**: Leverage Mojo's type system through existing config utilities
3. **Reproducibility**: Every experiment must be fully reproducible from configuration
4. **Validation**: All configurations validated against schemas
5. **DRY Principle**: Avoid duplication through inheritance and merging
6. **Environment Flexibility**: Support environment variables for deployment

## Directory Structure

```
configs/
├── README.md                    # User guide and documentation
├── defaults/                    # System-wide default configurations
│   ├── training.yaml           # Default training hyperparameters
│   ├── model.yaml              # Default model architecture settings
│   ├── data.yaml               # Default data processing parameters
│   └── paths.yaml              # Default directory paths
├── papers/                      # Paper-specific configurations
│   ├── lenet5/                 # LeNet-5 paper configurations
│   │   ├── model.yaml          # LeNet-5 architecture definition
│   │   ├── training.yaml       # LeNet-5 training parameters
│   │   └── data.yaml           # MNIST data configuration
│   └── alexnet/                # Future: AlexNet configurations
│       ├── model.yaml          # AlexNet architecture
│       └── training.yaml       # ImageNet training config
├── experiments/                 # Experiment-specific overrides
│   ├── lenet5/                 # LeNet-5 experiments
│   │   ├── baseline.yaml       # Baseline reproduction
│   │   ├── augmented.yaml      # With data augmentation
│   │   └── pruned.yaml         # Model pruning experiment
│   └── README.md               # Experiment naming conventions
├── schemas/                     # Validation schemas
│   ├── training.schema.yaml    # Training configuration schema
│   ├── model.schema.yaml       # Model architecture schema
│   ├── data.schema.yaml        # Data configuration schema
│   └── experiment.schema.yaml  # Experiment metadata schema
└── templates/                   # Templates for new configurations
    ├── paper.yaml              # Template for new paper configs
    ├── experiment.yaml         # Template for new experiments
    └── README.md               # Template usage guide
```

## Configuration Hierarchy

### 1. Default Configurations (`defaults/`)

System-wide defaults that apply to all papers and experiments unless overridden.

**training.yaml**:

```yaml
# Default Training Configuration
# These values are used unless overridden by paper or experiment configs

optimizer:
  name: "sgd"
  learning_rate: 0.001
  momentum: 0.9
  weight_decay: 0.0001

scheduler:
  name: "step"
  step_size: 30
  gamma: 0.1

training:
  epochs: 100
  batch_size: 32
  validation_split: 0.1
  early_stopping:
    enabled: false
    patience: 10
    min_delta: 0.001

logging:
  interval: 10  # Log every N batches
  save_checkpoints: true
  checkpoint_frequency: 5  # Save every N epochs
```

**model.yaml**:

```yaml
# Default Model Configuration
# Common model settings and initialization

initialization:
  weight_init: "xavier_uniform"
  bias_init: "zeros"

regularization:
  dropout: 0.0
  batch_norm: false
  layer_norm: false

architecture:
  activation: "relu"
  pooling: "max"
```

**data.yaml**:

```yaml
# Default Data Processing Configuration

preprocessing:
  normalize: true
  mean: [0.485, 0.456, 0.406]  # ImageNet defaults
  std: [0.229, 0.224, 0.225]

augmentation:
  enabled: false
  random_flip: false
  random_crop: false
  color_jitter: false

loader:
  num_workers: 4
  pin_memory: true
  shuffle_train: true
  shuffle_val: false
```

### 2. Paper Configurations (`papers/`)

Paper-specific configurations that override defaults for reproducing published results.

**papers/lenet5/model.yaml**:

```yaml
# LeNet-5 Architecture Configuration
# Based on LeCun et al., 1998

name: "LeNet-5"
input_shape: [1, 28, 28]  # MNIST grayscale
num_classes: 10

layers:
  - type: "conv2d"
    filters: 6
    kernel_size: 5
    activation: "tanh"
  - type: "avgpool2d"
    kernel_size: 2
  - type: "conv2d"
    filters: 16
    kernel_size: 5
    activation: "tanh"
  - type: "avgpool2d"
    kernel_size: 2
  - type: "flatten"
  - type: "linear"
    units: 120
    activation: "tanh"
  - type: "linear"
    units: 84
    activation: "tanh"
  - type: "linear"
    units: 10
    activation: "softmax"
```

### 3. Experiment Configurations (`experiments/`)

Experiment-specific overrides for testing variations and improvements.

**experiments/lenet5/augmented.yaml**:

```yaml
# LeNet-5 with Data Augmentation Experiment

extends:
  - defaults/training.yaml
  - papers/lenet5/model.yaml
  - papers/lenet5/data.yaml

experiment:
  name: "lenet5_augmented"
  description: "LeNet-5 with modern data augmentation"
  tags: ["augmentation", "regularization"]

# Override augmentation settings
data:
  augmentation:
    enabled: true
    random_rotation: 10  # degrees
    random_shift: 0.1    # fraction
    random_zoom: 0.1     # fraction

# Adjust training for augmented data
training:
  epochs: 50  # Fewer epochs needed with augmentation
  batch_size: 64
```

## Configuration Formats

### YAML Format (Primary)

YAML is the primary format for human-readable configurations.

**Advantages**:

- Human-readable and writable
- Support for comments and documentation
- Hierarchical structure
- Multi-line strings for descriptions

**Example Structure**:

```yaml
# Configuration header comment
version: "1.0"
description: |
  Multi-line description of this configuration.
  Can include implementation notes.

# Nested configuration
section:
  subsection:
    parameter: value
    list_param: [1, 2, 3]
    
# Environment variable substitution
paths:
  data_dir: "${DATA_DIR:-./data}"
  cache_dir: "${CACHE_DIR:-./cache}"
```

### JSON Format (Secondary)

JSON supported for programmatic generation and interoperability.

**Use Cases**:

- Auto-generated configurations
- API responses
- Cross-language compatibility

## Validation Strategy

### Schema Definition

Each configuration type has a corresponding schema in `schemas/`.

**training.schema.yaml**:

```yaml
# Training Configuration Schema
type: object
properties:
  optimizer:
    type: object
    required: [name, learning_rate]
    properties:
      name:
        type: string
        enum: ["sgd", "adam", "adamw", "rmsprop"]
      learning_rate:
        type: number
        minimum: 0
        maximum: 1
  training:
    type: object
    properties:
      epochs:
        type: integer
        minimum: 1
      batch_size:
        type: integer
        minimum: 1
        multipleOf: 1
```

### Validation Implementation

Using existing `shared/utils/config.mojo`:

```mojo
from shared.utils import Config, ConfigValidator

fn validate_training_config(config: Config) raises:
    """Validate training configuration against schema."""
    
    # Create validator
    var validator = ConfigValidator()
    
    # Define required fields
    validator.require("optimizer.name")
    validator.require("optimizer.learning_rate")
    validator.require("training.epochs")
    validator.require("training.batch_size")
    
    # Validate types
    config.validate_type("optimizer.name", "string")
    config.validate_type("optimizer.learning_rate", "float")
    config.validate_type("training.epochs", "int")
    config.validate_type("training.batch_size", "int")
    
    # Validate ranges
    config.validate_range("optimizer.learning_rate", 0.0, 1.0)
    config.validate_range("training.epochs", 1.0, Float64.MAX)
    
    # Validate enums
    var valid_optimizers = List[String]()
    valid_optimizers.append("sgd")
    valid_optimizers.append("adam")
    valid_optimizers.append("adamw")
    config.validate_enum("optimizer.name", valid_optimizers)
```

## Configuration Loading Pattern

### Standard Loading Workflow

```mojo
from shared.utils import Config, load_config, merge_configs

fn load_experiment_config(experiment_name: String) raises -> Config:
    """Load complete configuration for an experiment.
    
    Merges defaults → paper config → experiment config
    """
    # Load default configuration
    var defaults = load_config("configs/defaults/training.yaml")
    
    # Load paper-specific config
    var paper_config = load_config("configs/papers/lenet5/training.yaml")
    
    # Load experiment config
    var exp_config = load_config(
        "configs/experiments/lenet5/" + experiment_name + ".yaml"
    )
    
    # Merge configurations (later configs override earlier)
    var config = merge_configs(defaults, paper_config)
    config = merge_configs(config, exp_config)
    
    # Substitute environment variables
    config = config.substitute_env_vars()
    
    # Validate final configuration
    validate_training_config(config)
    
    return config
```

## Naming Conventions

### File Names

- **Lowercase with underscores**: `training_config.yaml`, `data_augmentation.yaml`
- **Paper names match paper directories**: `lenet5/`, `alexnet/`
- **Descriptive experiment names**: `baseline.yaml`, `pruned_90.yaml`

### Configuration Keys

- **Lowercase with underscores**: `learning_rate`, `batch_size`
- **Nested keys use dot notation in flat configs**: `"optimizer.learning_rate"`
- **Boolean flags are positive**: `use_augmentation` not `disable_augmentation`

### Special Keys

- `extends`: List of configurations to inherit from
- `version`: Configuration format version
- `experiment`: Experiment metadata (name, description, tags)
- `override`: Explicit overrides that should not be merged

## Environment Variable Support

### Substitution Syntax

```yaml
# Basic substitution
data_dir: "${DATA_DIR}"

# With default value
cache_dir: "${CACHE_DIR:-./cache}"

# Nested substitution
model_path: "${MODEL_DIR:-./models}/${EXPERIMENT_NAME}/checkpoint.pt"
```

### Common Variables

- `DATA_DIR`: Root data directory
- `CACHE_DIR`: Cache directory for processed data
- `MODEL_DIR`: Model checkpoint directory
- `LOG_DIR`: Logging output directory
- `EXPERIMENT_NAME`: Current experiment name
- `PAPER_NAME`: Current paper being reproduced

## Integration with Mojo Config Utilities

### Loading Configurations

```mojo
from shared.utils import Config

fn main() raises:
    # Load YAML configuration
    var config = Config.from_yaml("configs/experiments/lenet5/baseline.yaml")
    
    # Access configuration values
    var lr = config.get_float("optimizer.learning_rate", default=0.001)
    var batch_size = config.get_int("training.batch_size", default=32)
    var epochs = config.get_int("training.epochs", default=10)
    
    # Check for optional values
    if config.has("scheduler.step_size"):
        var step_size = config.get_int("scheduler.step_size")
```

### Creating Configurations Programmatically

```mojo
fn create_experiment_config(
    learning_rate: Float64,
    batch_size: Int,
    epochs: Int
) raises -> Config:
    """Create experiment configuration programmatically."""
    var config = Config()
    
    # Set basic parameters
    config.set("optimizer.name", "adam")
    config.set("optimizer.learning_rate", learning_rate)
    config.set("training.batch_size", batch_size)
    config.set("training.epochs", epochs)
    
    # Save to file
    config.to_yaml("configs/experiments/generated/exp001.yaml")
    
    return config
```

## Templates

### Paper Configuration Template

**templates/paper.yaml**:

```yaml
# Paper Configuration Template
# Copy this file to configs/papers/<paper_name>/ and customize

paper:
  name: "Paper Name"
  authors: "Author et al."
  year: 2024
  url: "https://arxiv.org/abs/..."

# Model architecture (override as needed)
extends:
  - ../../defaults/model.yaml

model:
  name: "model_name"
  # Add architecture-specific configuration

# Training parameters (override as needed)
training:
  # Paper-specific training parameters
  
# Data configuration (override as needed)  
data:
  dataset: "dataset_name"
  # Paper-specific data configuration
```

### Experiment Template

**templates/experiment.yaml**:

```yaml
# Experiment Configuration Template
# Copy to configs/experiments/<paper>/<experiment>.yaml

experiment:
  name: "experiment_name"
  description: "What this experiment tests"
  paper: "paper_name"
  tags: ["tag1", "tag2"]
  date: "2024-11-14"

# Inherit from paper configuration
extends:
  - ../../papers/<paper_name>/model.yaml
  - ../../papers/<paper_name>/training.yaml

# Experiment-specific overrides
training:
  # Override specific parameters for this experiment

# Results tracking
tracking:
  metrics: ["loss", "accuracy", "f1_score"]
  log_frequency: 10
  save_predictions: false
```

## Best Practices

### 1. Configuration Inheritance

- Start with defaults
- Override at paper level for reproduction
- Override at experiment level for variations
- Use `extends` to explicitly declare inheritance

### 2. Documentation

- Document all configurations with comments
- Include paper references in paper configs
- Describe experiment purpose in experiment configs
- Keep README files in each directory level

### 3. Validation

- Always validate configurations before use
- Define clear schemas for each config type
- Fail fast on invalid configurations
- Provide helpful error messages

### 4. Version Control

- Track all configurations in git
- Tag configurations used for published results
- Never modify configs used in completed experiments
- Create new configs for variations

### 5. Reproducibility

- Include all parameters needed for reproduction
- Document random seeds explicitly
- Version control data preprocessing parameters
- Include environment information

## Migration Guide

### For Existing Paper Implementations

1. Identify all hardcoded parameters
2. Create paper-specific config in `configs/papers/<paper>/`
3. Update code to load from configuration
4. Test reproduction with config-driven approach
5. Document any paper-specific quirks

### For New Papers

1. Start with paper template
2. Define architecture in `model.yaml`
3. Set training parameters in `training.yaml`
4. Configure data in `data.yaml`
5. Create baseline experiment config
6. Run and validate reproduction

## Testing Strategy

### Unit Tests (Issue #73)

```mojo
# Test configuration loading
fn test_load_paper_config():
    var config = load_config("configs/papers/lenet5/model.yaml")
    assert config.get_string("name") == "LeNet-5"
    assert config.get_int("num_classes") == 10

# Test configuration merging
fn test_merge_configs():
    var base = Config()
    base.set("lr", 0.001)
    base.set("epochs", 10)
    
    var override = Config()
    override.set("lr", 0.01)
    
    var merged = merge_configs(base, override)
    assert merged.get_float("lr") == 0.01  # Overridden
    assert merged.get_int("epochs") == 10  # From base

# Test validation
fn test_validate_training_config():
    var config = Config()
    config.set("optimizer.name", "invalid_optimizer")
    
    # Should raise validation error
    var raised = False
    try:
        validate_training_config(config)
    except:
        raised = True
    assert raised
```

### Integration Tests

- Test full configuration loading pipeline
- Verify paper reproduction with configs
- Test experiment variations
- Validate environment variable substitution

## Security Considerations

### Path Traversal Prevention

- Validate all file paths
- Restrict config loading to configs/ directory
- No execution of code from configs
- Sanitize environment variables

### Sensitive Data

- Never commit credentials to configs
- Use environment variables for secrets
- Document required environment variables
- Provide secure defaults

## Future Enhancements

### Version 2.0 Features

1. **Config Server**: REST API for configuration management
2. **Hot Reloading**: Update configs without restart
3. **A/B Testing**: Built-in experiment comparison
4. **Distributed Configs**: Support for distributed training
5. **Config UI**: Web interface for configuration editing

### Version 3.0 Vision

1. **AutoML Integration**: Automatic hyperparameter search
2. **Config Optimization**: Learn optimal configs from results
3. **Cross-Paper Transfer**: Transfer learning for configs
4. **Config Recommendation**: ML-based config suggestions

## Conclusion

This configuration architecture provides a robust, scalable foundation for managing ML experiments in ML Odyssey. The three-tier hierarchy (defaults → paper → experiment) ensures both reproducibility and flexibility, while the integration with Mojo's type system provides safety and performance.

Key benefits:

- **Reproducibility**: Every experiment fully defined by configuration
- **Reusability**: Inherit and override configurations efficiently  
- **Type Safety**: Leverage Mojo's type system for validation
- **Flexibility**: Support for multiple formats and environments
- **Scalability**: Grows with the project from single papers to hundreds

The design follows KISS and YAGNI principles while providing a clear path for future enhancements as the project grows.
