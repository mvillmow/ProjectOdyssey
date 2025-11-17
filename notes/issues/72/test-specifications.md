# Test Specifications for Issue #73

## Unit Tests Required

### 1. Configuration Loading Tests

**Test File**: `tests/configs/test_loading.mojo`

```mojo
fn test_load_default_training_config():
    """Test loading default training configuration."""
    var config = load_config("configs/defaults/training.yaml")
    assert config.has("optimizer.name")
    assert config.get_string("optimizer.name") == "sgd"

fn test_load_paper_config():
    """Test loading paper-specific configuration."""
    var config = load_config("configs/papers/lenet5/model.yaml")
    assert config.get_string("name") == "LeNet-5"
    assert config.get_int("num_classes") == 10

fn test_load_experiment_config():
    """Test loading experiment configuration."""
    var config = load_config("configs/experiments/lenet5/baseline.yaml")
    assert config.has("extends")
```

### 2. Configuration Merging Tests

**Test File**: `tests/configs/test_merging.mojo`

```mojo
fn test_merge_default_and_paper():
    """Test merging default and paper configurations."""
    var defaults = load_config("configs/defaults/training.yaml")
    var paper = load_config("configs/papers/lenet5/training.yaml")
    var merged = merge_configs(defaults, paper)
    
    # Paper values override defaults
    assert merged.get_float("learning_rate") == paper.get_float("learning_rate")
    # Default values retained if not in paper
    assert merged.has("scheduler.name")

fn test_three_level_merge():
    """Test default → paper → experiment merging."""
    var defaults = load_config("configs/defaults/training.yaml")
    var paper = load_config("configs/papers/lenet5/training.yaml")
    var exp = load_config("configs/experiments/lenet5/augmented.yaml")
    
    var merged = merge_configs(defaults, paper)
    merged = merge_configs(merged, exp)
    
    # Experiment overrides all
    assert merged.get_bool("data.augmentation.enabled") == True
```

### 3. Validation Tests

**Test File**: `tests/configs/test_validation.mojo`

```mojo
fn test_validate_training_config():
    """Test training configuration validation."""
    var config = load_config("configs/defaults/training.yaml")
    validate_training_config(config)  # Should not raise

fn test_validate_invalid_optimizer():
    """Test validation catches invalid optimizer."""
    var config = Config()
    config.set("optimizer.name", "invalid_optimizer")
    
    var raised = False
    try:
        validate_training_config(config)
    except:
        raised = True
    assert raised

fn test_validate_out_of_range():
    """Test validation catches out-of-range values."""
    var config = Config()
    config.set("optimizer.learning_rate", -0.001)  # Negative LR
    
    var raised = False
    try:
        config.validate_range("optimizer.learning_rate", 0.0, 1.0)
    except:
        raised = True
    assert raised
```

### 4. Environment Variable Tests

**Test File**: `tests/configs/test_env_vars.mojo`

```mojo
fn test_env_var_substitution():
    """Test environment variable substitution in configs."""
    # Set environment variable
    # Create config with ${DATA_DIR}
    # Load and substitute
    # Verify substitution occurred
    pass

fn test_env_var_with_default():
    """Test environment variable with default value."""
    # Create config with ${MISSING_VAR:-/default/path}
    # Load and substitute
    # Verify default used
    pass
```

### 5. Schema Validation Tests

**Test File**: `tests/configs/test_schemas.py` (Python for jsonschema)

```python
import yaml
import jsonschema

def test_training_schema():
    """Test training configuration against schema."""
    with open("configs/schemas/training.schema.yaml") as f:
        schema = yaml.safe_load(f)
    
    with open("configs/defaults/training.yaml") as f:
        config = yaml.safe_load(f)
    
    # Should validate without errors
    jsonschema.validate(config, schema)

def test_model_schema():
    """Test model configuration against schema."""
    with open("configs/schemas/model.schema.yaml") as f:
        schema = yaml.safe_load(f)
    
    with open("configs/papers/lenet5/model.yaml") as f:
        config = yaml.safe_load(f)
    
    # Should validate without errors
    jsonschema.validate(config, schema)
```

## Integration Tests Required

### 1. End-to-End Configuration Loading

**Test File**: `tests/configs/test_integration.mojo`

```mojo
fn test_load_complete_experiment_config():
    """Test loading complete experiment configuration."""
    var config = load_experiment_config("lenet5", "baseline")
    
    # Should have all required sections
    assert config.has("model")
    assert config.has("training")
    assert config.has("data")
    
    # Values should be correctly merged
    assert config.get_string("model.name") == "LeNet-5"
    assert config.get_float("training.learning_rate") > 0
```

### 2. Paper Implementation Integration

```mojo
fn test_lenet5_loads_config():
    """Test LeNet-5 implementation loads its configuration."""
    # This would be in papers/lenet5/tests/
    from papers.lenet5 import create_model
    
    var model = create_model("configs/papers/lenet5/model.yaml")
    assert model is not None
```

## Property-Based Tests

### 1. Configuration Invariants

```mojo
fn test_merge_associativity():
    """Test that merge order doesn't affect final result."""
    # (A merge B) merge C == A merge (B merge C)
    pass

fn test_validation_completeness():
    """Test that all required fields are validated."""
    # Generate random configs
    # Ensure validation catches all issues
    pass
```

## Test Coverage Requirements

- **Line Coverage**: Minimum 90% for config loading code
- **Branch Coverage**: All error paths tested
- **Integration Coverage**: All config files loaded at least once

## Test Data

### Fixtures Directory

Create `tests/configs/fixtures/`:

- `valid_training.yaml` - Valid training config
- `invalid_training.yaml` - Invalid training config (for error testing)
- `minimal.yaml` - Minimal valid config
- `complex.yaml` - Complex nested config

## Test Execution

```bash
# Run all config tests
mojo test tests/configs/

# Run specific test file
mojo test tests/configs/test_loading.mojo

# Run with coverage
mojo test --coverage tests/configs/

# Python schema tests
pytest tests/configs/test_schemas.py
```

## Success Criteria for Issue #73

- [ ] All unit tests passing
- [ ] All integration tests passing
- [ ] 90% code coverage achieved
- [ ] Error cases properly tested
- [ ] Performance benchmarks established
- [ ] Test documentation complete
