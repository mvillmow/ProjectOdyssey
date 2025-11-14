# Example Configuration Files for Issue #74

## Default Configurations

### configs/defaults/training.yaml

```yaml
# Default Training Configuration
# These values are used unless overridden by paper or experiment configs
# Last updated: 2024-11-14

# Optimizer configuration
optimizer:
  name: "sgd"              # Optimizer type: sgd, adam, adamw, rmsprop
  learning_rate: 0.001     # Initial learning rate
  momentum: 0.9            # Momentum factor (SGD)
  weight_decay: 0.0001     # L2 regularization
  betas: [0.9, 0.999]      # Adam beta parameters
  eps: 1.0e-08             # Adam epsilon

# Learning rate scheduler
scheduler:
  name: "step"             # Scheduler type: step, cosine, exponential, none
  step_size: 30            # Steps between LR decay (step scheduler)
  gamma: 0.1               # LR decay factor
  min_lr: 1.0e-06          # Minimum learning rate

# Training loop configuration
training:
  epochs: 100              # Maximum training epochs
  batch_size: 32           # Training batch size
  validation_split: 0.1    # Fraction of data for validation
  shuffle: true            # Shuffle training data
  seed: 42                 # Random seed for reproducibility
  
  # Early stopping
  early_stopping:
    enabled: false         # Enable early stopping
    patience: 10           # Epochs without improvement
    min_delta: 0.001       # Minimum change to qualify as improvement
    mode: "min"            # min for loss, max for accuracy

# Gradient configuration
gradient:
  clip_norm: 0.0           # Gradient clipping (0 = disabled)
  accumulation_steps: 1    # Gradient accumulation steps

# Logging configuration
logging:
  level: "INFO"            # Log level: DEBUG, INFO, WARNING, ERROR
  interval: 10             # Log every N batches
  save_checkpoints: true   # Save model checkpoints
  checkpoint_frequency: 5  # Save every N epochs
  best_only: true          # Save only best checkpoint
  metric: "val_loss"       # Metric for best checkpoint
  mode: "min"              # min or max for metric

# Hardware configuration  
hardware:
  device: "auto"           # Device: auto, cpu, cuda, mps
  mixed_precision: false   # Use mixed precision training
  num_workers: 4           # Dataloader workers
  pin_memory: true         # Pin memory for GPU
```

### configs/defaults/model.yaml

```yaml
# Default Model Configuration
# Common model settings and initialization
# Last updated: 2024-11-14

# Weight initialization
initialization:
  weight_init: "xavier_uniform"  # xavier_uniform, xavier_normal, kaiming_uniform, kaiming_normal, normal, uniform
  bias_init: "zeros"              # zeros, ones, normal, uniform
  gain: 1.0                       # Gain factor for Xavier/Kaiming

# Regularization techniques
regularization:
  dropout: 0.0                    # Dropout probability (0 = disabled)
  batch_norm: false               # Use batch normalization
  layer_norm: false               # Use layer normalization
  weight_decay: 0.0001            # L2 regularization strength

# Common architecture settings
architecture:
  activation: "relu"              # Default activation: relu, tanh, sigmoid, gelu, swish
  pooling: "max"                  # Pooling type: max, avg
  padding: "same"                 # Padding strategy: same, valid
  bias: true                      # Use bias terms
```

### configs/defaults/data.yaml

```yaml
# Default Data Configuration
# Data loading and preprocessing settings
# Last updated: 2024-11-14

# Data preprocessing
preprocessing:
  normalize: true                         # Apply normalization
  mean: [0.485, 0.456, 0.406]            # ImageNet normalization mean
  std: [0.229, 0.224, 0.225]             # ImageNet normalization std
  resize: null                            # Target size [height, width]
  center_crop: null                       # Center crop size

# Data augmentation
augmentation:
  enabled: false                          # Enable augmentation
  random_horizontal_flip: false           # Horizontal flip probability
  random_vertical_flip: false             # Vertical flip probability
  random_rotation: 0                      # Max rotation degrees
  random_crop: null                       # Random crop size
  color_jitter:
    enabled: false
    brightness: 0.2
    contrast: 0.2
    saturation: 0.2
    hue: 0.1

# Data loading
loader:
  batch_size: 32                         # Batch size (can be overridden by training config)
  num_workers: 4                         # Parallel data loading workers
  pin_memory: true                       # Pin memory for GPU transfer
  shuffle_train: true                    # Shuffle training data
  shuffle_val: false                    # Shuffle validation data
  shuffle_test: false                   # Shuffle test data
  drop_last: true                       # Drop incomplete batches

# Dataset split
split:
  train: 0.8                             # Training data fraction
  val: 0.1                               # Validation data fraction
  test: 0.1                              # Test data fraction
  stratified: true                       # Use stratified split
```

## Paper-Specific Configurations

### configs/papers/lenet5/model.yaml

```yaml
# LeNet-5 Model Architecture Configuration
# Based on: LeCun et al., "Gradient-Based Learning Applied to Document Recognition" (1998)
# Paper URL: http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
# Last updated: 2024-11-14

# Model metadata
name: "LeNet-5"
paper: "LeCun et al., 1998"
input_shape: [1, 28, 28]  # MNIST: 1 channel, 28x28 pixels
num_classes: 10           # MNIST: 10 digit classes

# Architecture definition (sequential layers)
layers:
  # C1: Convolutional Layer 1
  - name: "C1"
    type: "conv2d"
    in_channels: 1
    out_channels: 6
    kernel_size: 5
    stride: 1
    padding: 2           # To maintain 28x28 output
    activation: "tanh"   # Original paper used tanh
    
  # S2: Subsampling Layer 1 (Average Pooling)
  - name: "S2"
    type: "avgpool2d"
    kernel_size: 2
    stride: 2
    # Output: 6x14x14
    
  # C3: Convolutional Layer 2
  - name: "C3"
    type: "conv2d"
    in_channels: 6
    out_channels: 16
    kernel_size: 5
    stride: 1
    padding: 0
    activation: "tanh"
    # Note: Original paper had complex connectivity table
    # We use full connectivity for simplicity
    
  # S4: Subsampling Layer 2
  - name: "S4"
    type: "avgpool2d"
    kernel_size: 2
    stride: 2
    # Output: 16x5x5
    
  # C5: Convolutional Layer 3 (Fully Connected)
  - name: "C5"
    type: "conv2d"
    in_channels: 16
    out_channels: 120
    kernel_size: 5      # Input is 5x5, so this becomes FC
    stride: 1
    padding: 0
    activation: "tanh"
    # Output: 120x1x1
    
  # Flatten for fully connected layers
  - name: "flatten"
    type: "flatten"
    
  # F6: Fully Connected Layer
  - name: "F6"
    type: "linear"
    in_features: 120
    out_features: 84
    activation: "tanh"
    
  # Output Layer
  - name: "output"
    type: "linear"
    in_features: 84
    out_features: 10
    activation: null     # Raw logits for cross-entropy loss

# Loss function
loss:
  name: "cross_entropy"
  label_smoothing: 0.0
```

### configs/papers/lenet5/training.yaml

```yaml
# LeNet-5 Training Configuration
# Based on original paper training procedure
# Last updated: 2024-11-14

# Optimizer (paper used custom SGD variant)
optimizer:
  name: "sgd"
  learning_rate: 0.01    # Higher than modern defaults
  momentum: 0.9
  weight_decay: 0.0      # No weight decay in original

# Learning rate schedule (paper used manual schedule)
scheduler:
  name: "step"
  step_size: 30
  gamma: 0.5             # Halve LR periodically

# Training parameters
training:
  epochs: 20             # Original trained for fewer epochs
  batch_size: 128        # Larger batch than default
  validation_split: 0.2  # 10K validation from 60K training
  shuffle: true
  seed: 1998             # Year of paper publication

# Specific to LeNet-5
gradient:
  clip_norm: 0.0         # No gradient clipping in original

# Logging
logging:
  interval: 100          # Log every 100 batches
  save_checkpoints: true
  checkpoint_frequency: 5
```

### configs/papers/lenet5/data.yaml

```yaml
# LeNet-5 Data Configuration for MNIST
# Last updated: 2024-11-14

# Dataset information
dataset:
  name: "MNIST"
  path: "${DATA_DIR:-./data}/mnist"
  download: true

# MNIST-specific preprocessing
preprocessing:
  normalize: true
  mean: [0.1307]         # MNIST mean
  std: [0.3081]          # MNIST std
  resize: [28, 28]       # Already 28x28, but explicit
  center_crop: null

# No augmentation in original paper
augmentation:
  enabled: false

# Data loading
loader:
  batch_size: 128        # Override from training config
  num_workers: 2         # Less workers for small dataset
  pin_memory: true
  shuffle_train: true
  shuffle_val: false
  shuffle_test: false
  drop_last: false       # Keep all data

# Dataset split (MNIST standard)
split:
  train: 60000           # Absolute numbers for MNIST
  val: 10000            # From training set
  test: 10000           # Separate test set
```

## Experiment Configurations

### configs/experiments/lenet5/baseline.yaml

```yaml
# LeNet-5 Baseline Experiment
# Reproduces original paper results
# Last updated: 2024-11-14

# Experiment metadata
experiment:
  name: "lenet5_baseline"
  description: "Baseline LeNet-5 reproduction on MNIST"
  paper: "lenet5"
  tags: ["baseline", "reproduction", "mnist"]
  author: "ML Odyssey Team"
  date: "2024-11-14"

# Inherit from paper configuration
extends:
  - ../../papers/lenet5/model.yaml
  - ../../papers/lenet5/training.yaml
  - ../../papers/lenet5/data.yaml

# No overrides for baseline - use paper defaults exactly

# Results tracking
tracking:
  metrics: ["train_loss", "val_loss", "train_acc", "val_acc", "test_acc"]
  log_frequency: 100
  save_predictions: false
  save_confusion_matrix: true

# Expected results (from paper)
expected_results:
  test_accuracy: 0.991    # 99.1% accuracy reported in paper
  tolerance: 0.003        # ±0.3% acceptable variance
```

### configs/experiments/lenet5/augmented.yaml

```yaml
# LeNet-5 with Modern Augmentation
# Tests impact of data augmentation on LeNet-5
# Last updated: 2024-11-14

# Experiment metadata
experiment:
  name: "lenet5_augmented"
  description: "LeNet-5 with modern data augmentation techniques"
  paper: "lenet5"
  tags: ["augmentation", "ablation", "mnist"]
  author: "ML Odyssey Team"
  date: "2024-11-14"

# Inherit base configuration
extends:
  - ../../papers/lenet5/model.yaml
  - ../../papers/lenet5/training.yaml
  - ../../papers/lenet5/data.yaml

# Override: Add data augmentation
augmentation:
  enabled: true
  random_rotation: 15           # ±15 degrees
  random_affine:
    degrees: 0
    translate: [0.1, 0.1]      # 10% translation
    scale: [0.9, 1.1]          # 90-110% scaling
    shear: 5                   # 5 degree shear
  random_erasing:
    enabled: true
    probability: 0.1
    scale: [0.02, 0.33]
    ratio: [0.3, 3.3]

# Adjust training for augmented data
training:
  epochs: 30                   # More epochs for augmented data
  batch_size: 64              # Smaller batch for more updates

# Different optimizer
optimizer:
  name: "adam"                # Modern optimizer
  learning_rate: 0.001
  weight_decay: 0.0001        # Add regularization

# Cosine annealing schedule
scheduler:
  name: "cosine"
  T_max: 30                   # Match epochs
  eta_min: 1.0e-06

# Enhanced tracking
tracking:
  metrics: ["train_loss", "val_loss", "train_acc", "val_acc", "test_acc", "learning_rate"]
  log_frequency: 50
  save_predictions: true
  save_confusion_matrix: true
  tensorboard: true
  wandb:
    enabled: false
    project: "ml-odyssey"
    name: "lenet5_augmented"
```

## Schema Files

### configs/schemas/training.schema.yaml

```yaml
# JSON Schema for Training Configuration Validation
# Last updated: 2024-11-14

$schema: "http://json-schema.org/draft-07/schema#"
title: "Training Configuration Schema"
type: object
required: ["optimizer", "training"]

properties:
  optimizer:
    type: object
    required: ["name", "learning_rate"]
    properties:
      name:
        type: string
        enum: ["sgd", "adam", "adamw", "rmsprop", "adagrad"]
      learning_rate:
        type: number
        minimum: 0
        maximum: 1
      momentum:
        type: number
        minimum: 0
        maximum: 1
      weight_decay:
        type: number
        minimum: 0
        maximum: 1
      betas:
        type: array
        items:
          type: number
          minimum: 0
          maximum: 1
        minItems: 2
        maxItems: 2
      eps:
        type: number
        minimum: 0
        
  scheduler:
    type: object
    properties:
      name:
        type: string
        enum: ["step", "cosine", "exponential", "polynomial", "none"]
      step_size:
        type: integer
        minimum: 1
      gamma:
        type: number
        minimum: 0
        maximum: 1
        
  training:
    type: object
    required: ["epochs", "batch_size"]
    properties:
      epochs:
        type: integer
        minimum: 1
        maximum: 10000
      batch_size:
        type: integer
        minimum: 1
        multipleOf: 1
      validation_split:
        type: number
        minimum: 0
        maximum: 1
      seed:
        type: integer
        
  gradient:
    type: object
    properties:
      clip_norm:
        type: number
        minimum: 0
      accumulation_steps:
        type: integer
        minimum: 1
        
  logging:
    type: object
    properties:
      level:
        type: string
        enum: ["DEBUG", "INFO", "WARNING", "ERROR"]
      interval:
        type: integer
        minimum: 1
      save_checkpoints:
        type: boolean
      checkpoint_frequency:
        type: integer
        minimum: 1
```

## Template Files

### configs/templates/paper.yaml

```yaml
# Paper Configuration Template
# Copy to configs/papers/<paper_name>/ and customize
# Last updated: 2024-11-14

# Paper metadata (required)
paper:
  name: "PAPER_NAME"              # e.g., "ResNet"
  authors: "Author et al."        # e.g., "He et al."
  year: 2024                      # Publication year
  url: "https://arxiv.org/abs/..." # Paper URL
  conference: "CONFERENCE"        # e.g., "CVPR", "NeurIPS"

# Model architecture (customize completely)
model:
  name: "model_name"
  input_shape: [3, 224, 224]     # [channels, height, width]
  num_classes: 1000               # Number of output classes
  
  # Define architecture here
  layers:
    - type: "conv2d"
      # ... layer parameters

# Training configuration (override defaults as needed)
training:
  epochs: 100
  batch_size: 256
  # Add paper-specific training settings

# Data configuration (override defaults as needed)
data:
  dataset: "dataset_name"
  # Add dataset-specific settings

# Notes and references
notes: |
  Add any implementation notes, special considerations,
  or deviations from the paper here.
```

### configs/templates/experiment.yaml

```yaml
# Experiment Configuration Template
# Copy to configs/experiments/<paper>/<experiment_name>.yaml
# Last updated: 2024-11-14

# Experiment metadata (required)
experiment:
  name: "EXPERIMENT_NAME"
  description: "What this experiment tests or explores"
  paper: "PAPER_NAME"             # Which paper this builds on
  tags: []                        # e.g., ["ablation", "hyperparameter"]
  author: "Your Name"
  date: "YYYY-MM-DD"
  
  # Hypothesis (optional but recommended)
  hypothesis: |
    What you expect to discover or validate
    with this experiment.

# Configuration inheritance (required)
extends:
  - ../../papers/PAPER_NAME/model.yaml
  - ../../papers/PAPER_NAME/training.yaml
  - ../../papers/PAPER_NAME/data.yaml

# Experiment-specific overrides
# Only include what you're changing from the base configuration

# Example: Override learning rate
#optimizer:
#  learning_rate: 0.01

# Example: Change batch size
#training:
#  batch_size: 64

# Results tracking (optional but recommended)
tracking:
  metrics: ["loss", "accuracy"]
  log_frequency: 100
  save_predictions: false
  
# Expected results (optional)
expected_results:
  metric_name: expected_value
  tolerance: acceptable_variance
  
# Notes (optional)
notes: |
  Any special considerations, observations,
  or future work based on this experiment.
```

These example files provide concrete, implementable specifications for Issue #74.
