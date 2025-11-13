# Issue #1556: [Test] Shared Test Infrastructure

## Objective

Create comprehensive shared test utilities and fixtures for the ML Odyssey test suite, providing mock tensors, data loaders, models, file I/O helpers, and configuration fixtures to support testing across all components.

## Deliverables

- `/tests/shared/fixtures/__init__.mojo` - Module documentation
- `/tests/shared/fixtures/mock_tensors.mojo` - Tensor test utilities
- `/tests/shared/fixtures/mock_data.mojo` - Data loader mocks
- `/tests/shared/fixtures/mock_models.mojo` - Model architecture mocks
- `/tests/shared/fixtures/test_io_helpers.mojo` - File I/O helpers
- `/tests/shared/fixtures/config_fixtures.mojo` - Configuration fixtures
- `/tests/shared/fixtures/__init__.py` - Python pytest fixtures

## Success Criteria

- [ ] All Mojo fixture files created with clear documentation
- [ ] Python pytest fixtures module created
- [ ] Each fixture includes usage examples in docstrings
- [ ] Code follows Mojo best practices (fn vs def, type hints)
- [ ] All files properly documented in this README
- [ ] No scope creep - focus only on test utilities

## References

- [Test Architecture](../48/test-architecture.md) - Overall test design
- [Shared Library Tests README](/tests/shared/README.md) - Test suite organization
- [Conftest.mojo](/tests/shared/conftest.mojo) - Existing test utilities
- [CLAUDE.md](/CLAUDE.md) - Mojo language guidelines

## Implementation Notes

### Design Decisions

1. **Mock Tensor Utilities** - Provide simple tensor creation and comparison functions:
   - Random tensor generation with deterministic seeds
   - Pattern-based tensors (zeros, ones, sequential)
   - Floating-point comparison with epsilon tolerance
   - Shape validation utilities

2. **Mock Data Loaders** - Simple dataset and loader implementations:
   - Configurable mock dataset with batch support
   - Classification and regression sample generators
   - Basic data loader with batching

3. **Mock Model Architectures** - Minimal models for testing:
   - Simple linear model (single layer)
   - Small MLP (2-3 layers)
   - Mock layer implementations for unit tests

4. **File I/O Test Helpers** - Temporary file management:
   - Temporary directory creation and cleanup
   - Mock configuration file generators
   - Path utilities for test resources

5. **Configuration Fixtures** - Sample configurations:
   - Valid YAML/JSON configuration examples
   - Invalid configurations for error testing
   - Configuration validation helpers

6. **Python Test Utilities** - pytest integration:
   - Common pytest fixtures
   - Markdown validation utilities
   - Link checking helpers
   - File structure validators

### Implementation Strategy

Following TDD and YAGNI principles:

- Start with minimal implementations that satisfy immediate testing needs
- Avoid over-engineering - keep fixtures simple and concrete
- No complex mocking frameworks - use real implementations where possible
- Add functionality only when tests require it
- Maintain clear documentation and usage examples

### Files Created

#### 1. `/tests/shared/fixtures/__init__.mojo`

Module documentation and public API exports for Mojo fixtures.

#### 2. `/tests/shared/fixtures/mock_tensors.mojo`

Tensor test utilities providing:

- `create_random_tensor(shape: List[Int], seed: Int = 42)` - Random tensor generation
- `create_zeros_tensor(shape: List[Int])` - Zero-filled tensors
- `create_ones_tensor(shape: List[Int])` - One-filled tensors
- `create_sequential_tensor(shape: List[Int], start: Float32 = 0.0)` - Sequential values
- `assert_tensors_equal(a: Tensor, b: Tensor, epsilon: Float64 = 1e-6)` - Comparison with tolerance
- `assert_shape_equal(tensor: Tensor, expected_shape: List[Int])` - Shape validation

#### 3. `/tests/shared/fixtures/mock_data.mojo`

Mock dataset and data loader implementations:

- `struct MockDataset` - Simple dataset with configurable samples
- `struct MockClassificationDataset` - Classification data generator
- `struct MockRegressionDataset` - Regression data generator
- `struct MockDataLoader` - Basic data loader with batching
- Helper functions for creating sample data

#### 4. `/tests/shared/fixtures/mock_models.mojo`

Mock model architectures for testing:

- `struct SimpleLinearModel` - Single linear layer
- `struct SimpleMLP` - 2-3 layer MLP
- `struct MockLayer` - Minimal layer implementation
- Forward pass implementations for testing

#### 5. `/tests/shared/fixtures/test_io_helpers.mojo`

File I/O utilities for tests:

- `create_temp_dir()` - Create temporary directory
- `cleanup_temp_dir(path: String)` - Remove temporary directory
- `create_mock_config(path: String, content: String)` - Write config file
- `create_mock_checkpoint(path: String)` - Create mock checkpoint
- `get_test_data_path(filename: String)` - Resolve test data paths

#### 6. `/tests/shared/fixtures/config_fixtures.mojo`

Configuration fixtures and samples:

- `valid_yaml_config()` - Sample valid YAML configuration
- `valid_json_config()` - Sample valid JSON configuration
- `invalid_config_missing_fields()` - Invalid config examples
- `invalid_config_wrong_types()` - Type error examples
- Configuration validation helpers

#### 7. `/tests/shared/fixtures/__init__.py`

Python pytest fixtures:

- Common pytest fixtures for file validation
- Markdown linting helpers
- Link checking utilities
- Repository structure validators
- Test data discovery helpers

### Usage Examples

#### Mock Tensors

```mojo
from tests.shared.fixtures.mock_tensors import create_random_tensor, assert_tensors_equal

fn test_layer_forward():
    # Create test input with deterministic seed
    var input = create_random_tensor([2, 10], seed=42)

    # Run layer forward pass
    var output = layer.forward(input)

    # Compare with expected output
    var expected = create_random_tensor([2, 5], seed=123)
    assert_tensors_equal(output, expected, epsilon=1e-6)
```

#### Mock Data

```mojo
from tests.shared.fixtures.mock_data import MockClassificationDataset, MockDataLoader

fn test_training_loop():
    # Create mock dataset
    var dataset = MockClassificationDataset(
        num_samples=100,
        input_dim=10,
        num_classes=5
    )

    # Create data loader
    var loader = MockDataLoader(dataset, batch_size=32)

    # Test batching
    for batch in loader:
        assert batch.size() <= 32
```

#### Mock Models

```mojo
from tests.shared.fixtures.mock_models import SimpleMLP

fn test_training_workflow():
    # Create simple test model
    var model = SimpleMLP(input_dim=10, hidden_dim=20, output_dim=5)

    # Test forward pass
    var input = create_random_tensor([2, 10])
    var output = model.forward(input)
    assert output.shape() == [2, 5]
```

#### Test I/O Helpers

```mojo
from tests.shared.fixtures.test_io_helpers import create_temp_dir, cleanup_temp_dir

fn test_checkpoint_saving():
    # Create temporary directory
    var temp_dir = create_temp_dir()

    try:
        # Test checkpoint operations
        var checkpoint_path = temp_dir + "/model.ckpt"
        save_checkpoint(model, checkpoint_path)
        assert file_exists(checkpoint_path)
    finally:
        # Clean up
        cleanup_temp_dir(temp_dir)
```

#### Configuration Fixtures

```mojo
from tests.shared.fixtures.config_fixtures import valid_yaml_config, invalid_config_missing_fields

fn test_config_validation():
    # Test valid config
    var valid_cfg = valid_yaml_config()
    assert validate_config(valid_cfg) == True

    # Test invalid config
    var invalid_cfg = invalid_config_missing_fields()
    assert validate_config(invalid_cfg) == False
```

### Key Design Principles Applied

1. **KISS** - Simple, straightforward implementations without unnecessary complexity
2. **YAGNI** - Only implemented what's needed for current testing requirements
3. **DRY** - Reusable fixtures prevent duplication across tests
4. **TDD** - Fixtures designed to support test-first development
5. **POLA** - Intuitive APIs that match common testing patterns

### Next Steps

After completing this issue:

1. Use fixtures in existing test files to validate functionality
2. Gather feedback on fixture usability
3. Extend fixtures as new testing needs arise
4. Document common testing patterns using these fixtures
5. Create integration tests using multiple fixtures together

### Related Issues

- Issue #48 - Test implementation (parent)
- Issue #49 - Shared library implementation (consumer)
- Future issues will use these fixtures for testing

## Files Created Summary

All files successfully created in `/tests/shared/fixtures/`:

### 1. `__init__.mojo` (Module Documentation)

- Documents the fixtures package
- Explains module organization and usage patterns
- No explicit exports needed (Mojo automatic namespace)

### 2. `mock_tensors.mojo` (318 lines)

**Tensor Creation Functions**:

- `create_random_tensor()` - Random tensors with deterministic seeds
- `create_zeros_tensor()` - Zero-filled tensors
- `create_ones_tensor()` - One-filled tensors
- `create_sequential_tensor()` - Sequential values for testing indexing
- `create_constant_tensor()` - Constant-filled tensors

**Comparison Functions**:

- `assert_tensors_equal()` - Element-wise comparison with epsilon tolerance
- `assert_shape_equal()` - Shape validation
- `calculate_tensor_size()` - Size calculation helper

**Statistics Functions**:

- `tensor_mean()` - Calculate mean for validation
- `tensor_min()` - Find minimum value
- `tensor_max()` - Find maximum value

### 3. `mock_data.mojo` (418 lines)

**Dataset Implementations**:

- `struct MockDataset` - Basic dataset with configurable dimensions
- `struct MockClassificationDataset` - Classification data with class labels
- `struct MockRegressionDataset` - Regression data with correlated outputs

**Data Loader**:

- `struct MockDataLoader` - Batching support with size calculations
- `get_batch_size()` - Determine batch sizes (including partial last batch)
- `get_batch_indices()` - Get sample indices for batch

**Helper Functions**:

- `create_mock_batch()` - Quick batch creation without full setup
- `create_mock_classification_batch()` - Classification batches

### 4. `mock_models.mojo` (453 lines)

**Layer Implementations**:

- `struct MockLayer` - Minimal layer for testing (scaled transformation)
- `forward()` - Simple forward pass with scaling

**Model Implementations**:

- `struct SimpleLinearModel` - Single linear layer model
  - Matrix-vector multiplication: y = Wx + b
  - Weight randomization support
  - Parameter counting
- `struct SimpleMLP` - Multi-layer perceptron (1-2 hidden layers)
  - Configurable architecture
  - ReLU activation between layers
  - Forward pass through all layers

### 5. `test_io_helpers.mojo` (320 lines)

**Directory Management**:

- `create_temp_dir()` - Create unique temporary directories
- `cleanup_temp_dir()` - Safe removal with /tmp validation
- `temp_file_path()` - Path construction helper

**File Creation**:

- `create_mock_config()` - Write configuration files
- `create_mock_checkpoint()` - Create checkpoint files
- `create_mock_text_file()` - Generate text files

**Path Utilities**:

- `get_test_data_path()` - Resolve test data paths
- `file_exists()` / `dir_exists()` - Existence checking
- `join_paths()` - Path joining with separators
- `get_filename()` - Extract filename from path
- `get_extension()` - Extract file extension

**Test Data Organization**:

- `get_fixtures_dir()` - Base fixtures directory
- `get_images_dir()` - Images subdirectory
- `get_tensors_dir()` - Tensors subdirectory
- `get_models_dir()` - Models subdirectory
- `get_reference_dir()` - Reference outputs subdirectory

### 6. `config_fixtures.mojo` (427 lines)

**Valid Configurations**:

- `valid_yaml_config()` - Complete valid YAML configuration
- `valid_json_config()` - Complete valid JSON configuration
- `minimal_valid_config()` - Minimal required fields

**Invalid Configurations (for error testing)**:

- `invalid_config_missing_fields()` - Missing required fields
- `invalid_config_wrong_types()` - Type errors
- `invalid_config_negative_values()` - Negative value errors
- `invalid_config_out_of_range()` - Range validation errors
- `invalid_yaml_syntax()` - YAML syntax errors
- `invalid_json_syntax()` - JSON syntax errors

**Configuration Templates**:

- `config_template_classification()` - Classification task template
- `config_template_regression()` - Regression task template

**Validation Helpers**:

- `has_required_fields()` - Check for required fields
- `is_valid_yaml_syntax()` - Basic YAML syntax check
- `is_valid_json_syntax()` - Basic JSON syntax check

### 7. `__init__.py` (Python Pytest Fixtures, 397 lines)

**Pytest Fixtures**:

- `@pytest.fixture temp_dir()` - Auto-cleanup temporary directories
- `@pytest.fixture mock_config_file()` - Factory for config files
- `@pytest.fixture mock_text_file()` - Factory for text files

**Markdown Validation**:

- `validate_markdown_links()` - Check for broken links
- `validate_markdown_code_blocks()` - Validate code block formatting
- `find_repo_root()` - Locate repository root

**File Structure Validation**:

- `validate_directory_structure()` - Check expected layout
- `find_files_by_pattern()` - Glob-based file search

**Configuration Validation**:

- `validate_yaml_file()` - YAML syntax validation
- `validate_json_file()` - JSON syntax validation

**Test Data Helpers**:

- `get_fixtures_dir()` - Python fixtures directory path
- `get_test_data_path()` - Test data file paths
- `create_sample_yaml_config()` - Sample config dict
- `create_sample_json_config()` - Sample config dict

**Assertion Helpers**:

- `assert_files_equal()` - Compare file contents
- `assert_file_contains()` - Check file contains string

## Total Implementation Stats

- **7 files created**
- **~2,350 lines of documented code**
- **Mojo files**: 1,936 lines
- **Python file**: 397 lines
- **0 existing files modified** (minimal changes principle)

## Key Features

1. **Comprehensive Coverage**: Fixtures for tensors, data, models, I/O, and configs
2. **Well Documented**: Every function has docstrings with examples
3. **Deterministic**: All random operations use fixed seeds for reproducibility
4. **Simple & Concrete**: No complex mocking - straightforward implementations
5. **Type Safe**: Uses Mojo's type system (fn functions, typed parameters)
6. **TDD Ready**: Designed to support test-driven development workflows

## Files Modified

All files created (no existing files modified to maintain minimal changes principle).

## Testing Strategy

These fixtures will be validated by:

1. Using them in existing test files
2. Verifying they produce expected test data
3. Confirming they integrate with Mojo test framework
4. Ensuring documentation examples work correctly

## Usage Patterns

### Pattern 1: Simple Tensor Testing

```mojo
from tests.shared.fixtures.mock_tensors import create_ones_tensor, assert_tensors_equal

fn test_layer_identity():
    var input = create_ones_tensor([10])
    var output = identity_layer.forward(input)
    assert_tensors_equal(input, output, epsilon=1e-6)
```

### Pattern 2: Data Loading Testing

```mojo
from tests.shared.fixtures.mock_data import MockDataLoader

fn test_batch_sizes():
    var loader = MockDataLoader(num_samples=100, batch_size=32)
    assert loader.__len__() == 4  # 3 full + 1 partial
    assert loader.get_batch_size(3) == 4  # Last batch
```

### Pattern 3: Training Loop Testing

```mojo
from tests.shared.fixtures.mock_models import SimpleMLP
from tests.shared.fixtures.mock_data import create_mock_batch

fn test_training_step():
    var model = SimpleMLP(input_dim=10, hidden_dim=20, output_dim=5)
    var (inputs, targets) = create_mock_batch(32, 10, 5)

    for i in range(len(inputs)):
        var output = model.forward(inputs[i])
        # Test forward pass completes
```

### Pattern 4: Configuration Testing

```mojo
from tests.shared.fixtures.config_fixtures import valid_yaml_config, invalid_config_missing_fields

fn test_config_validation():
    # Test valid config
    var valid = valid_yaml_config()
    assert parse_config(valid) is not None

    # Test invalid config
    var invalid = invalid_config_missing_fields()
    # Should raise error
```

### Pattern 5: Python Pytest Integration

```python
def test_markdown_validation(temp_dir):
    # Create test markdown file
    readme = temp_dir / "README.md"
    readme.write_text("# Test\n\n[Broken](missing.md)")

    # Validate links
    from tests.shared.fixtures import validate_markdown_links
    broken = validate_markdown_links(readme)
    assert len(broken) > 0  # Should find broken link
```

---

**Status**: Completed

**Last Updated**: 2025-11-13
