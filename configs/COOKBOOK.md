# Configuration Cookbook

This cookbook provides ready-to-use configuration recipes for common ML scenarios.

## Table of Contents

1. [Multi-GPU Configuration](#1-multi-gpu-configuration)
2. [Distributed Training Setup](#2-distributed-training-setup)
3. [Hyperparameter Sweep](#3-hyperparameter-sweep)
4. [A/B Testing Configuration](#4-ab-testing-configuration)
5. [Custom Architecture](#5-custom-architecture)
6. [Transfer Learning](#6-transfer-learning)
7. [Mixed Precision Training](#7-mixed-precision-training)
8. [Gradient Accumulation](#8-gradient-accumulation)
9. [Early Stopping](#9-early-stopping)
10. [Custom Logging](#10-custom-logging)
11. [Data Augmentation](#11-data-augmentation)
12. [Learning Rate Scheduling](#12-learning-rate-scheduling)

## 1. Multi-GPU Configuration

Configure training across multiple GPUs on a single machine.

```yaml
# configs/experiments/multi_gpu.yaml
training:
  device: "cuda"
  num_gpus: 4
  distributed_backend: "nccl"
  
  # Per-GPU batch size
  batch_size: 32  # Total batch = 32 * 4 = 128
  
  # Gradient synchronization
  gradient_sync_interval: 1
  
  # Mixed precision for efficiency
  mixed_precision:
    enabled: true
    opt_level: "O1"  # O0=FP32, O1=Mixed, O2=Almost FP16

parallel:
  data_parallel: true
  model_parallel: false
  pipeline_parallel: false

# Adjust learning rate for larger batch
optimizer:
  learning_rate: 0.004  # Base LR * num_gpus
```

## 2. Distributed Training Setup

Configure training across multiple machines.

```yaml
# configs/experiments/distributed.yaml
distributed:
  enabled: true
  backend: "nccl"  # nccl for GPU, gloo for CPU
  
  # Master node
  master_addr: "${MASTER_ADDR:-localhost}"
  master_port: "${MASTER_PORT:-29500}"
  
  # World size and rank set by launcher
  world_size: "${WORLD_SIZE:-1}"
  rank: "${RANK:-0}"
  
  # Communication settings
  init_method: "env://"
  timeout_seconds: 1800
  
  # Gradient compression
  compression:
    enabled: true
    algorithm: "powersgd"
    rank: 2

training:
  # Effective batch size = batch_size * world_size
  batch_size: 64
  gradient_accumulation_steps: 1
  
  # Checkpoint only from rank 0
  save_checkpoints: "${RANK:-0} == 0"
```

## 3. Hyperparameter Sweep

Configuration for hyperparameter search.

```yaml
# configs/experiments/sweep.yaml
sweep:
  method: "grid"  # grid, random, bayesian
  metric:
    name: "validation_loss"
    goal: "minimize"
  
  # Parameter ranges
  parameters:
    learning_rate:
      values: [0.001, 0.01, 0.1]
    
    batch_size:
      values: [16, 32, 64]
    
    dropout:
      min: 0.1
      max: 0.5
      distribution: "uniform"
    
    optimizer:
      values: ["sgd", "adam", "rmsprop"]
    
    weight_decay:
      values: [0.0, 0.0001, 0.001]

  # Early termination
  early_terminate:
    type: "hyperband"
    min_iter: 3
    eta: 3

# Fixed parameters
training:
  epochs: 100
  validation_split: 0.2
```

## 4. A/B Testing Configuration

Compare two model variants.

```yaml
# configs/experiments/ab_test.yaml
variants:
  control:
    model:
      architecture: "baseline"
      layers: [784, 128, 64, 10]
      activation: "relu"
    
    optimizer:
      name: "sgd"
      learning_rate: 0.01
  
  treatment:
    model:
      architecture: "improved"
      layers: [784, 256, 128, 64, 10]
      activation: "gelu"
      dropout: 0.3
    
    optimizer:
      name: "adam"
      learning_rate: 0.001

# Shared settings
training:
  epochs: 50
  batch_size: 32
  seed: 42  # Same seed for fair comparison

# Metrics to compare
evaluation:
  metrics:
    - "accuracy"
    - "loss"
    - "inference_time"
    - "memory_usage"
```

## 5. Custom Architecture

Define a custom model architecture.

```yaml
# configs/experiments/custom_arch.yaml
model:
  architecture: "custom"
  
  # Layer definitions
  layers:
    - type: "conv2d"
      filters: 32
      kernel_size: 3
      activation: "relu"
      padding: "same"
    
    - type: "maxpool2d"
      pool_size: 2
      strides: 2
    
    - type: "conv2d"
      filters: 64
      kernel_size: 3
      activation: "relu"
      padding: "same"
    
    - type: "global_avgpool2d"
    
    - type: "dense"
      units: 128
      activation: "relu"
      dropout: 0.5
    
    - type: "dense"
      units: 10
      activation: "softmax"
  
  # Weight initialization
  init:
    method: "he_normal"  # For ReLU
    seed: 42
```

## 6. Transfer Learning

Configure transfer learning from a pretrained model.

```yaml
# configs/experiments/transfer_learning.yaml
pretrained:
  model_path: "${PRETRAINED_MODEL_PATH}"
  freeze_layers: true
  freeze_until: "layer_10"  # Freeze up to this layer
  
  # Fine-tuning schedule
  fine_tuning:
    enabled: true
    unfreeze_at_epoch: 10
    unfreeze_layers: 5  # Unfreeze last N layers

model:
  # New head for transfer learning
  custom_head:
    - type: "global_avgpool2d"
    - type: "dense"
      units: 256
      activation: "relu"
      dropout: 0.5
    - type: "dense"
      units: "${NUM_CLASSES:-10}"
      activation: "softmax"

training:
  # Lower learning rate for pretrained weights
  learning_rate: 0.0001
  # Higher learning rate for new head
  discriminative_lr:
    enabled: true
    head_lr_multiplier: 10
```

## 7. Mixed Precision Training

Configure automatic mixed precision for faster training.

```yaml
# configs/experiments/mixed_precision.yaml
mixed_precision:
  enabled: true
  backend: "apex"  # or "native" for PyTorch AMP
  opt_level: "O1"
  
  # O0: FP32 (baseline)
  # O1: Mixed Precision (recommended)
  # O2: "Almost FP16" Mixed Precision
  # O3: FP16
  
  # Loss scaling
  loss_scale: "dynamic"  # or fixed value like 128
  initial_loss_scale: 65536
  loss_scale_window: 2000
  
  # Keep specific layers in FP32
  fp32_layers:
    - "batch_norm"
    - "layer_norm"

# Adjust batch size (can use larger with FP16)
training:
  batch_size: 256  # 2x normal batch size
  
  # Gradient clipping recommended
  gradient_clip_norm: 1.0
```

## 8. Gradient Accumulation

Simulate large batches on limited memory.

```yaml
# configs/experiments/gradient_accumulation.yaml
training:
  # Micro batch (fits in memory)
  batch_size: 8
  
  # Accumulation steps
  gradient_accumulation_steps: 16
  # Effective batch size = 8 * 16 = 128
  
  # Only step optimizer after accumulation
  optimizer_step_interval: 16
  
  # Adjust learning rate for effective batch size
  learning_rate: 0.004  # 4x base LR

memory_optimization:
  gradient_checkpointing: true
  clear_gradients: true
  
  # Optionally offload to CPU
  offload:
    optimizer_state: false
    gradients: false
```

## 9. Early Stopping

Configure early stopping to prevent overfitting.

```yaml
# configs/experiments/early_stopping.yaml
early_stopping:
  enabled: true
  monitor: "validation_loss"
  mode: "min"  # min for loss, max for accuracy
  
  # Patience settings
  patience: 10  # Epochs to wait
  min_delta: 0.0001  # Minimum change
  
  # Restore best weights
  restore_best_weights: true
  
  # Learning rate reduction on plateau
  reduce_lr_on_plateau:
    enabled: true
    factor: 0.5
    patience: 5
    min_lr: 0.00001

# Checkpointing best model
checkpointing:
  save_best_only: true
  monitor: "validation_loss"
  mode: "min"
```

## 10. Custom Logging

Configure detailed logging and monitoring.

```yaml
# configs/experiments/custom_logging.yaml
logging:
  level: "INFO"
  
  # Console logging
  console:
    enabled: true
    interval: 10  # Log every N batches
    
  # File logging
  file:
    enabled: true
    path: "${LOG_DIR}/training.log"
    max_size_mb: 100
    backup_count: 5
  
  # TensorBoard
  tensorboard:
    enabled: true
    log_dir: "${LOG_DIR}/tensorboard"
    histogram_freq: 10  # Weight histograms
    profile_batch: [100, 110]  # Profile batches
    
  # Weights & Biases
  wandb:
    enabled: true
    project: "${WANDB_PROJECT}"
    entity: "${WANDB_ENTITY}"
    tags: ["experiment", "baseline"]
    config_exclude: ["wandb", "logging"]
  
  # Custom metrics
  custom_metrics:
    - name: "gradient_norm"
      log_interval: 100
    - name: "weight_norm"
      log_interval: 1000
    - name: "activation_stats"
      log_interval: 1000

# What to log
metrics:
  train:
    - "loss"
    - "accuracy"
    - "learning_rate"
  
  validation:
    - "loss"
    - "accuracy"
    - "precision"
    - "recall"
    - "f1_score"
```

## 11. Data Augmentation

Configure data augmentation strategies.

```yaml
# configs/experiments/augmentation.yaml
data_augmentation:
  # Image augmentations
  image:
    random_crop:
      enabled: true
      size: [28, 28]
      padding: 4
    
    random_horizontal_flip:
      enabled: true
      probability: 0.5
    
    random_rotation:
      enabled: true
      degrees: 15
    
    color_jitter:
      enabled: true
      brightness: 0.2
      contrast: 0.2
      saturation: 0.2
      hue: 0.1
    
    random_erasing:
      enabled: true
      probability: 0.5
      scale: [0.02, 0.33]
      ratio: [0.3, 3.3]
  
  # MixUp augmentation
  mixup:
    enabled: true
    alpha: 0.2
  
  # CutMix augmentation
  cutmix:
    enabled: true
    alpha: 1.0
    probability: 0.5

# Test time augmentation
tta:
  enabled: false
  num_augmentations: 5
  aggregation: "mean"  # mean, max, or voting
```

## 12. Learning Rate Scheduling

Configure various learning rate schedules.

```yaml
# configs/experiments/lr_scheduling.yaml
lr_scheduler:
  # Step decay
  step:
    enabled: false
    step_size: 30
    gamma: 0.1
  
  # Exponential decay
  exponential:
    enabled: false
    gamma: 0.95
  
  # Cosine annealing
  cosine:
    enabled: true
    T_max: 100  # Maximum epochs
    eta_min: 0.00001
  
  # Cosine annealing with warm restarts
  cosine_warm_restarts:
    enabled: false
    T_0: 10  # Initial restart interval
    T_mult: 2  # Interval multiplier
    eta_min: 0.00001
  
  # Linear warmup
  warmup:
    enabled: true
    warmup_epochs: 5
    warmup_method: "linear"  # linear or exponential
    warmup_factor: 0.1
  
  # One cycle policy
  one_cycle:
    enabled: false
    max_lr: 0.1
    epochs: 100
    pct_start: 0.3
    anneal_strategy: "cos"
    
  # Custom schedule
  custom:
    enabled: false
    milestones: [30, 60, 90]
    values: [0.1, 0.01, 0.001, 0.0001]
```

## Usage Tips

### Combining Recipes

You can combine multiple recipes for your use case:

```yaml
# configs/experiments/combined.yaml
# Combine multi-GPU + mixed precision + gradient accumulation
training:
  num_gpus: 2
  batch_size: 16
  gradient_accumulation_steps: 4
  
mixed_precision:
  enabled: true
  opt_level: "O1"
```

### Environment Variables

Use environment variables for flexibility:

```bash
# Set environment variables
export ML_ODYSSEY_GPUS=4
export ML_ODYSSEY_BATCH_SIZE=64

# Reference in config
num_gpus: "${ML_ODYSSEY_GPUS:-1}"
batch_size: "${ML_ODYSSEY_BATCH_SIZE:-32}"
```

### Override Hierarchy

Start with defaults, override for specific needs:

1. Load `defaults/training.yaml`
2. Override with `papers/lenet5/training.yaml`
3. Override with `experiments/custom.yaml`

### Validation

Always validate your configuration:

```bash
python scripts/lint_configs.py configs/experiments/my_config.yaml
```

## Summary

These recipes provide starting points for common ML scenarios. Adapt them to your specific needs, and always validate configurations before training!
