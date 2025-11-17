# Configuration Best Practices

This guide outlines best practices for working with configurations in ML Odyssey.

## Table of Contents

- [Configuration Anti-Patterns](#configuration-anti-patterns)
- [Performance Optimization](#performance-optimization)
- [Security Guidelines](#security-guidelines)
- [Versioning Strategies](#versioning-strategies)
- [Maintenance Recommendations](#maintenance-recommendations)

## Configuration Anti-Patterns

### ❌ DON'T: Hardcode Sensitive Values

```yaml
# Bad - Never commit credentials
database:
  password: "my-secret-password"
  api_key: "sk-1234567890abcdef"
```

```yaml
# Good - Use environment variables
database:
  password: "${DB_PASSWORD}"
  api_key: "${API_KEY}"
```

### ❌ DON'T: Use Deeply Nested Structures

```yaml
# Bad - Too deeply nested
model:
  layers:
    conv:
      layer1:
        filters:
          size: 32
```

```yaml
# Good - Flatter structure
model:
  conv1_filters: 32
  conv1_kernel: 3
```

### ❌ DON'T: Mix Configuration and Code

```yaml
# Bad - Logic in config
training:
  lr: "0.001 * epoch / 10"  # Don't embed code
```

```yaml
# Good - Pure values
training:
  base_lr: 0.001
  lr_schedule: "linear"  # Let code handle logic
```

### ❌ DON'T: Use Ambiguous Names

```yaml
# Bad - Unclear units
timeout: 30  # Seconds? Minutes? Iterations?
size: 1024   # Bytes? KB? Elements?
```

```yaml
# Good - Clear units
timeout_seconds: 30
batch_size: 1024
buffer_size_mb: 256
```

## Performance Optimization

### 1. Use Lazy Loading for Large Configs

```python
# Load only what you need
config = Config.from_yaml("config.yaml")
if config.has("advanced_settings"):
    advanced = Config.from_yaml("advanced.yaml")
```

### 2. Cache Parsed Configurations

```python
# Cache configs that don't change
_config_cache = {}

def get_config(path):
    if path not in _config_cache:
        _config_cache[path] = Config.from_yaml(path)
    return _config_cache[path]
```

### 3. Minimize File I/O

```yaml
# Good - Single file with sections
training:
  epochs: 100
  batch_size: 32
model:
  layers: [64, 32, 10]
```

### 4. Use Simple Data Types

```yaml
# Prefer simple types that parse quickly
layers: [64, 32, 10]  # Simple list
activation: "relu"     # Simple string
```

## Security Guidelines

### 1. Environment Variable Substitution

```yaml
# Use environment variables for sensitive data
database:
  host: "${DB_HOST:-localhost}"  # With default
  port: "${DB_PORT:-5432}"
  password: "${DB_PASSWORD}"     # Required
```

### 2. Separate Sensitive Configs

```bash
# Keep sensitive configs separate
configs/
  public.yaml       # Safe to commit
  secrets.yaml      # In .gitignore
```

### 3. Validate Input Ranges

```yaml
# Use schemas to enforce valid ranges
training:
  learning_rate: 0.001  # Schema: min: 0.0, max: 1.0
  epochs: 100          # Schema: min: 1, max: 10000
```

### 4. Avoid Command Injection

```yaml
# Never execute config values as commands
output_dir: "./results"  # Good
cleanup_cmd: "rm -rf *"  # Bad - Never execute
```

## Versioning Strategies

### 1. Semantic Versioning for Configs

```yaml
# Include version in config
version: "1.2.0"  # major.minor.patch
# major: Breaking changes
# minor: New features, backward compatible
# patch: Bug fixes
```

### 2. Migration Paths

```yaml
# Provide migration info
version: "2.0.0"
deprecated:
  - "model.num_layers"  # Use model.layers instead
  - "optimizer.type"    # Use optimizer.name instead
```

### 3. Backward Compatibility

```python
# Support old and new keys during transition
def get_optimizer_name(config):
    # Try new key first, fall back to old
    return config.get("optimizer.name",
                      config.get("optimizer.type", "sgd"))
```

### 4. Config Change Logs

```yaml
# Document changes in CHANGELOG
# Version 2.0.0
# - Renamed optimizer.type to optimizer.name
# - Added support for learning rate schedules
# - Deprecated model.num_layers
```

## Maintenance Recommendations

### 1. Regular Validation

```bash
# Run validation as part of CI/CD
python scripts/lint_configs.py configs/
python scripts/validate_schemas.py configs/
```

### 2. Documentation Standards

```yaml
# Always document non-obvious values
model:
  dropout: 0.5  # Applied after each dense layer
  init_method: "he_normal"  # For ReLU activations
```

### 3. Consistent Formatting

```yaml
# Use consistent style
optimizer:  # 2-space indent
  name: "adam"
  learning_rate: 0.001
  betas: [0.9, 0.999]  # Inline lists for simple values
```

### 4. Modular Organization

```text
configs/
  defaults/       # Shared defaults
  papers/         # Paper-specific
  experiments/    # Experiment variations
  templates/      # Starting points
```

### 5. Testing Configurations

```python
# Test configs with different values
def test_config_loading():
    config = Config.from_yaml("test_config.yaml")
    assert config.get_float("learning_rate") == 0.001
    assert config.get_int("batch_size") == 32
```

### 6. Config Diffing

```bash
# Track config changes
diff configs/defaults/training.yaml configs/experiments/custom.yaml
```

### 7. Automated Cleanup

```python
# Remove unused parameters
python scripts/lint_configs.py --remove-unused configs/
```

## Common Patterns

### 1. Override Hierarchy

```yaml
# Base -> Paper -> Experiment
# defaults/training.yaml
epochs: 100

# papers/lenet5/training.yaml
epochs: 50  # Override default

# experiments/lenet5/quick.yaml
epochs: 10  # Override paper default
```

### 2. Environment-Specific Configs

```yaml
# configs/env/development.yaml
debug: true
log_level: "DEBUG"
checkpoint_interval: 100

# configs/env/production.yaml
debug: false
log_level: "INFO"
checkpoint_interval: 1000
```

### 3. Feature Flags

```yaml
# Enable/disable features
features:
  mixed_precision: false
  gradient_checkpointing: false
  data_parallel: true
```

## Summary

Following these best practices will help you:

- Create maintainable, secure configurations
- Optimize configuration loading performance
- Avoid common pitfalls and anti-patterns
- Enable smooth version transitions
- Facilitate team collaboration

Remember: Keep it simple, secure, and well-documented!
