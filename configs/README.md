# Configuration System

The `configs/` directory provides a comprehensive configuration management system for the ML Odyssey project.
It enables easy configuration of models, training procedures, and data processing pipelines using YAML files
with support for inheritance, environment variables, and validation.

## Quick Start

### Running LeNet-5 Baseline

```bash
# Load and train using baseline configuration
cd /path/to/ml-odyssey
mojo run papers/lenet5/train.mojo --config configs/experiments/lenet5/baseline.yaml
```

### Running LeNet-5 with Augmentation

```bash
# Use augmented experiment configuration
mojo run papers/lenet5/train.mojo --config configs/experiments/lenet5/augmented.yaml
```

### Using Custom Configuration

```bash
# Create custom experiment configuration
cp configs/templates/experiment.yaml configs/experiments/custom/my_experiment.yaml
# Edit the file to customize
vim configs/experiments/custom/my_experiment.yaml
# Run with custom configuration
mojo run papers/lenet5/train.mojo --config configs/experiments/custom/my_experiment.yaml
```

## Directory Structure

```text
configs/
├── README.md                           # This file
├── defaults/                           # Default configurations used as base
│   ├── training.yaml                  # Default training parameters
│   ├── model.yaml                     # Default model settings
│   ├── data.yaml                      # Default data processing
│   └── paths.yaml                     # Default directory paths
├── papers/                            # Paper-specific configurations
│   └── lenet5/                        # LeNet-5 paper reproduction
│       ├── model.yaml                 # LeNet-5 architecture
│       ├── training.yaml              # LeNet-5 training config
│       └── data.yaml                  # LeNet-5 data config
├── experiments/                       # Experiment variations
│   └── lenet5/                        # LeNet-5 experiments
│       ├── baseline.yaml              # Paper reproduction
│       └── augmented.yaml             # With data augmentation
├── schemas/                           # JSON schemas for validation
│   ├── training.schema.yaml           # Training config schema
│   ├── model.schema.yaml              # Model config schema
│   └── data.schema.yaml               # Data config schema
└── templates/                         # Templates for new configs
    ├── paper.yaml                     # New paper template
    └── experiment.yaml                # New experiment template
```

## Configuration Files

### Default Configurations (`configs/defaults/`)

Default configurations serve as the base for all other configurations. They define sensible defaults
for common parameters.

#### training.yaml

Defines default training parameters:

- Optimizer settings (SGD, learning rate, momentum)
- Learning rate scheduler configuration
- Training loop parameters (epochs, batch size, seed)
- Early stopping configuration
- Gradient clipping and accumulation
- Logging and checkpointing
- Hardware settings (device, precision, workers)

**Example usage in paper config**:

```yaml
extends:
  - ../../defaults/training.yaml

optimizer:
  learning_rate: 0.01  # Override default 0.001
```

#### model.yaml

Defines default model settings:

- Weight initialization methods
- Regularization techniques (dropout, batch norm)
- Architecture defaults (activation, pooling, padding)

#### data.yaml

Defines default data processing:

- Preprocessing (normalization, resizing)
- Data augmentation settings (disabled by default)
- Data loading parameters (batch size, workers)
- Dataset split configuration

#### paths.yaml

Defines default directory paths with environment variable support:

- Data directory: `${DATA_DIR:-./data}`
- Cache directory: `${CACHE_DIR:-~/.cache/ml-odyssey}`
- Output directory: `${OUTPUT_DIR:-./output}`
- Dataset-specific paths
- Experiment tracking paths

**Environment variable syntax**: `${VAR_NAME:-default_value}`

- If `VAR_NAME` is set in the environment, its value is used
- Otherwise, `default_value` is used
- Example: `${DATA_DIR:-./data}` expands to `/home/user/data` if `DATA_DIR=/home/user/data`, else `./data`

### Paper Configurations (`configs/papers/<paper_name>/`)

Paper-specific configurations define the exact model architecture and training procedure from the paper.

#### LeNet-5 Example

**model.yaml**: Complete LeNet-5 architecture

- Input shape: 1x28x28 (MNIST)
- 2 convolutional blocks with average pooling
- 2 fully connected layers
- Output: 10 classes
- Activation: tanh (as per original paper)

**training.yaml**: Training parameters from paper

- SGD optimizer with lr=0.01 (higher than modern defaults)
- Step learning rate schedule with gamma=0.5
- 20 epochs, batch size=128
- Seed=1998 (year of paper)

**data.yaml**: MNIST dataset configuration

- Normalization with MNIST-specific mean/std
- No data augmentation
- 60K training, 10K validation, 10K test splits
- Batch size=128

### Experiment Configurations (`configs/experiments/<paper>/<experiment>/`)

Experiment configurations extend paper configurations with variations to test hypotheses.

#### LeNet-5 Baseline

**baseline.yaml**: Exact paper reproduction

- Inherits all settings from paper configuration
- Expects 99.1% test accuracy (as reported)
- Tracks: train/val loss, train/val/test accuracy

#### LeNet-5 Augmented

**augmented.yaml**: Modern techniques experiment

- Adds data augmentation (rotation, translation, scaling)
- Uses Adam optimizer instead of SGD
- Cosine annealing learning rate schedule
- Extended to 30 epochs
- Tracks learning rate in addition to accuracy

## Creating New Configurations

### Adding a New Paper

1. Create directory structure:

   ```bash
   mkdir -p configs/papers/your_paper
   cd configs/papers/your_paper
   ```

1. Copy template and customize:

   ```bash
   cp ../../templates/paper.yaml model.yaml
   cp ../../defaults/training.yaml training.yaml
   cp ../../defaults/data.yaml data.yaml
   ```

1. Edit each file to match paper specifications:

   ```yaml
   # model.yaml
   name: "YourModel"
   paper: "Author et al., 2024"
   input_shape: [3, 224, 224]
   num_classes: 1000

   layers:
     - name: "conv1"
       type: "conv2d"
       out_channels: 64
       kernel_size: 7
       stride: 2
       # ... more layers
   ```

1. Update training and data configurations as needed

### Adding a New Experiment

1. Create directory if needed:

   ```bash
   mkdir -p configs/experiments/paper_name
   ```

1. Copy and customize template:

   ```bash
   cp ../../templates/experiment.yaml my_experiment.yaml
   ```

1. Define inheritance:

   ```yaml
   extends:
     - ../../papers/paper_name/model.yaml
     - ../../papers/paper_name/training.yaml
     - ../../papers/paper_name/data.yaml
   ```

1. Add only the overrides for your experiment:

   ```yaml
   # Only change what's different from paper config
   optimizer:
     learning_rate: 0.005  # Override to test different LR

   training:
     epochs: 50  # Run longer

   augmentation:
     enabled: true
     random_rotation: 10
   ```

## Configuration Inheritance

Configurations can inherit from other configurations using the `extends` field:

```yaml
extends:
  - ../../papers/lenet5/model.yaml
  - ../../papers/lenet5/training.yaml
  - ../../papers/lenet5/data.yaml
```

**Merging Rules**:

1. Load all parent configurations in order
1. Merge dictionaries recursively
1. Later files override earlier values
1. Current file overrides all parents

**Example**:

```yaml
# Parent: training.yaml
optimizer:
  name: "sgd"
  learning_rate: 0.001

# Child: baseline.yaml
extends:
  - training.yaml

optimizer:
  learning_rate: 0.01  # Override, but keep name="sgd"
```

Result after merging:

```yaml
optimizer:
  name: "sgd"          # From parent
  learning_rate: 0.01  # From child (override)
```

## Environment Variables

All configuration files support environment variable substitution using the pattern:
`${VAR_NAME:-default_value}`

### Common Variables

```bash
# Set before running
export DATA_DIR=/path/to/data
export CACHE_DIR=/path/to/cache
export OUTPUT_DIR=/path/to/output
```

### In Configuration

```yaml
# configs/papers/lenet5/data.yaml
dataset:
  path: "${DATA_DIR:-./data}/mnist"

# configs/defaults/paths.yaml
paths:
  data_dir: "${DATA_DIR:-./data}"
  output_dir: "${OUTPUT_DIR:-./output}"
  cache_dir: "${CACHE_DIR:-~/.cache/ml-odyssey}"
```

## Integration with Mojo Code

### Loading Configurations

```mojo
# In your Mojo training script
from shared.utils.config import ConfigManager

fn main() raises:
    # Load configuration with inheritance and env var substitution
    var config = ConfigManager.load_with_merge("configs/experiments/lenet5/baseline.yaml")

    # Access nested values
    var learning_rate = config["optimizer"]["learning_rate"]
    var batch_size = config["training"]["batch_size"]
    var data_dir = config["paths"]["data_dir"]  # Env vars expanded

    # Validate against schema
    if not ConfigManager.validate(config, "configs/schemas/training.schema.yaml"):
        print("Configuration validation failed!")
        return

    # Use in training loop
    train_model(config)
```

### Configuration Structure in Code

```mojo
# Access configuration values
let learning_rate = config.get("optimizer.learning_rate", 0.001)
let epochs = config.get("training.epochs", 100)
let batch_size = config.get("training.batch_size", 32)

# Iterate over layers
for layer in config["layers"]:
    let layer_type = layer["type"]
    let layer_name = layer["name"]
    # Build layer based on config
```

## YAML Syntax Guide

### Basic Types

```yaml
# Strings
name: "LeNet-5"

# Numbers
learning_rate: 0.01
batch_size: 128

# Booleans
shuffle: true
normalize: false

# Lists
mean: [0.485, 0.456, 0.406]
sizes: [64, 128, 256]

# Nested objects
optimizer:
  name: "sgd"
  learning_rate: 0.01

# Null values
activation: null
```

### Comments

```yaml
# Full line comment
training:
  epochs: 100  # Inline comment
```

### Multiline Strings

```yaml
notes: |
  This is a multiline note.
  It can span multiple lines.
  Perfect for documentation.
```

## Configuration Validation

Schemas are provided to validate configurations:

```bash
# Validate training configuration (Python)
python3 -c "
import yaml
import json
from jsonschema import validate

config = yaml.safe_load(open('configs/experiments/lenet5/baseline.yaml'))
schema = yaml.safe_load(open('configs/schemas/training.schema.yaml'))
validate(instance=config, schema=schema)
print('Configuration is valid!')
"

# Validate YAML syntax
python3 -c "import yaml; yaml.safe_load(open('configs/experiments/lenet5/baseline.yaml'))"
```

## Best Practices

### 1. Keep Defaults Simple

Defaults should work for most cases. Paper-specific and experiment configs override as needed.

### 2. Document Changes

When creating an experiment, explain what you changed and why:

```yaml
experiment:
  name: "lenet5_augmented"
  description: "Tests impact of modern augmentation on LeNet-5"
  tags: ["ablation", "augmentation"]
```

### 3. Use Environment Variables for Paths

Never hardcode absolute paths. Use environment variables:

```yaml
# Good
data_dir: "${DATA_DIR:-./data}"

# Avoid
data_dir: "/home/user/data"
```

### 4. Test Configuration Loading

Always validate new configurations:

```bash
python3 -c "import yaml; yaml.safe_load(open('configs/my_config.yaml'))"
```

### 5. Keep Hierarchy Clean

- Defaults define base values
- Paper configs refine for specific papers
- Experiments override for variations

Avoid deep nesting of extends.

## Common Tasks

### Change Learning Rate

```yaml
# In experiment config
extends:
  - ../../papers/lenet5/training.yaml

optimizer:
  learning_rate: 0.005  # Override just this value
```

### Add Data Augmentation

```yaml
# In experiment config
augmentation:
  enabled: true
  random_rotation: 15
  random_horizontal_flip: true
```

### Use Different Optimizer

```yaml
# In experiment config
optimizer:
  name: "adam"
  learning_rate: 0.001
```

### Change Batch Size

```yaml
# In experiment config
training:
  batch_size: 64
```

## Troubleshooting

### Configuration Not Loading

1. Check YAML syntax:

   ```bash
   python3 -c "import yaml; yaml.safe_load(open('your_config.yaml'))"
   ```

1. Check extends paths are relative and correct:

   ```yaml
   extends:
     - ../../papers/lenet5/model.yaml  # Good (relative)
     - /absolute/path/model.yaml       # Bad (absolute)
   ```

1. Verify file exists:

   ```bash
   ls -la configs/experiments/lenet5/baseline.yaml
   ```

### Environment Variables Not Expanding

1. Check syntax: `${VAR_NAME:-default}`

1. Verify variable is set:

   ```bash
   echo $DATA_DIR
   ```

1. Set if needed:

   ```bash
   export DATA_DIR=/path/to/data
   ```

### Schema Validation Fails

1. Check schema file exists:

   ```bash
   ls configs/schemas/training.schema.yaml
   ```

1. Validate schema itself:

   ```bash
   python3 -c "import yaml; yaml.safe_load(open('configs/schemas/training.schema.yaml'))"
   ```

1. Check configuration matches schema requirements

## References

- YAML Syntax: [https://yaml.org/spec/1.2/spec.html](https://yaml.org/spec/1.2/spec.html)
- JSON Schema: [https://json-schema.org/](https://json-schema.org/)
- Mojo Config Utils: `shared/utils/config.mojo`
- Papers: `papers/` directory structure
