# Issue #446: [Package] Test Fixtures

## Objective

Package test fixtures for easy reuse, ensuring fixtures are well-organized, documented, and accessible to all tests across the project.

## Deliverables

- Organized fixture modules
- Clear import patterns for fixtures
- Fixture catalog documentation
- Usage examples for common fixtures
- Best practices for fixture use

## Success Criteria

- [ ] Fixtures are easily importable from any test
- [ ] Fixture organization is logical and clear
- [ ] Documentation covers all fixtures with examples
- [ ] Common fixture patterns are documented
- [ ] Fixtures follow consistent naming conventions

## Current State Analysis

### Existing Fixture Organization

### File Structure

```text
tests/
├── shared/
│   └── conftest.mojo           # TestFixtures struct + data generators
│       ├── TestFixtures.deterministic_seed()
│       ├── TestFixtures.set_seed()
│       ├── create_test_vector()
│       ├── create_test_matrix()
│       └── create_sequential_vector()
├── helpers/
│   └── fixtures.mojo           # ExTensor fixtures (placeholder/TODO)
```text

### Current Import Patterns

```mojo
// Seed fixtures
from tests.shared.conftest import TestFixtures

// Data generators
from tests.shared.conftest import create_test_vector, create_test_matrix

// ExTensor fixtures (when implemented)
from tests.helpers.fixtures import random_tensor, sequential_tensor
```text

### Packaging Approach

**For Test Fixtures**, packaging means:

1. **Clear Organization**: Fixtures grouped by purpose
1. **Easy Discovery**: Developers can find fixtures easily
1. **Consistent Naming**: Fixtures follow predictable patterns
1. **Documentation**: Usage examples and best practices

## Packaging Strategy

### 1. Module Organization

### Current Structure Assessment

- `conftest.mojo` has both fixtures AND generators (okay for now)
- `helpers/fixtures.mojo` for ExTensor-specific (good separation)
- Clear separation between general and domain-specific

**Recommendation**: Keep current structure unless it grows unwieldy

**Future Structure** (if needed):

```text
tests/
├── shared/
│   ├── fixtures/
│   │   ├── __init__.mojo
│   │   ├── seed.mojo          # Seed fixtures
│   │   ├── data.mojo          # Data generators
│   │   ├── models.mojo        # Model fixtures
│   │   └── datasets.mojo      # Dataset fixtures
│   └── conftest.mojo          # Imports all fixtures for convenience
├── helpers/
│   └── fixtures.mojo          # ExTensor fixtures
```text

### Only split if

- Module exceeds 500 lines
- Clear benefit to organization
- Reduces cognitive load

### 2. Import Standardization

### Recommended Patterns

**Pattern 1: Direct Import** (for specific fixtures):

```mojo
from tests.shared.conftest import TestFixtures, create_test_vector
```text

**Pattern 2: Grouped Import** (for multiple fixtures):

```mojo
from tests.shared.conftest import (
    TestFixtures,
    create_test_vector,
    create_test_matrix,
    create_sequential_vector
)
```text

**Pattern 3: Module Import** (for namespace clarity):

```mojo
from tests.shared import conftest

fn test_something() raises:
    conftest.TestFixtures.set_seed()
    var vec = conftest.create_test_vector(10)
```text

**Best Practice**: Use Pattern 2 (grouped import) for clarity

### 3. Fixture Catalog Documentation

**Create**: `docs/testing/fixtures-catalog.md`

```markdown
# Test Fixtures Catalog

## Seed Fixtures

### TestFixtures.deterministic_seed()
Returns a fixed seed value for reproducible tests.

**Returns**: `Int` - Always returns 42

**Usage**:
```mojo

var seed = TestFixtures.deterministic_seed()
random.seed(seed)

```text
### TestFixtures.set_seed()
Sets the global random seed to a deterministic value.

**Usage**:
```mojo

TestFixtures.set_seed()
var random_val = randn()  // Reproducible value

```text
**Best Practice**: Call at start of each test that uses random values

## Data Generator Fixtures

### create_test_vector(size, value=1.0)
Creates a vector filled with a uniform value.

**Parameters**:
- `size: Int` - Vector size
- `value: Float32` - Fill value (default 1.0)

**Returns**: `List[Float32]`

**Usage**:
```mojo

var ones = create_test_vector(100, 1.0)
var zeros = create_test_vector(100, 0.0)
var custom = create_test_vector(50, 3.14)

```text
### create_test_matrix(rows, cols, value=1.0)
Creates a matrix filled with a uniform value.

**Parameters**:
- `rows: Int` - Number of rows
- `cols: Int` - Number of columns
- `value: Float32` - Fill value (default 1.0)

**Returns**: `List[List[Float32]]`

**Usage**:
```mojo

var matrix = create_test_matrix(10, 5, 2.0)

```text
### create_sequential_vector(size, start=0.0)
Creates a vector with sequential values.

**Parameters**:
- `size: Int` - Vector size
- `start: Float32` - Starting value (default 0.0)

**Returns**: `List[Float32]` - Values [start, start+1, start+2, ...]

**Usage**:
```mojo

var seq = create_sequential_vector(5, 10.0)
// Result: [10.0, 11.0, 12.0, 13.0, 14.0]

```text
## ExTensor Fixtures

(To be documented when implemented)

### random_tensor(shape, dtype=DType.float32)
### sequential_tensor(shape, dtype=DType.float32)
### ones_tensor(shape, dtype=DType.float32)
### zeros_tensor(shape, dtype=DType.float32)
### nan_tensor(shape)
### inf_tensor(shape)

## Model Fixtures

(To be documented when implemented)

### TestFixtures.simple_linear_model()
### TestFixtures.small_mlp()

## Dataset Fixtures

(To be documented when implemented)

### TestFixtures.toy_classification_dataset()
### TestFixtures.toy_regression_dataset()
```text

### 4. Fixture Usage Guide

**Create**: `docs/testing/fixture-usage.md`

```markdown
# Fixture Usage Guide

## When to Use Fixtures

**Use fixtures when**:
- Multiple tests need the same test data
- Setup is complex or expensive
- You want deterministic, reproducible tests
- You need to avoid test data duplication

**Don't use fixtures when**:
- Data is trivial (e.g., single integer)
- Each test needs unique data
- Fixture adds unnecessary abstraction

## Common Patterns

### Pattern 1: Deterministic Random Data

```mojo

fn test_random_operation() raises:
    """Test with reproducible random data."""
    TestFixtures.set_seed()  // Ensures reproducibility
    var data = create_test_vector(100)
    // ... test code ...

```text
### Pattern 2: Known Test Data

```mojo

fn test_with_known_values() raises:
    """Test with predictable values."""
    var input = create_sequential_vector(5, 0.0)
    // input = [0.0, 1.0, 2.0, 3.0, 4.0]

    var result = my_function(input)
    assert_equal(result, expected)

```text
### Pattern 3: Fixture Reuse

```mojo

fn test_operation_1() raises:
    """First test using shared fixture."""
    var data = create_test_vector(100, 1.0)
    // ... test code ...

fn test_operation_2() raises:
    """Second test using same fixture."""
    var data = create_test_vector(100, 1.0)
    // Gets same data, but independent copy
    // ... test code ...

```text
### Pattern 4: Fixture Composition

```mojo

fn test_complex_scenario() raises:
    """Combine multiple fixtures."""
    TestFixtures.set_seed()  // For reproducibility

    var input = create_sequential_vector(10)
    var weights = create_test_vector(10, 0.5)
    var bias = 1.0

    var result = weighted_sum(input, weights, bias)
    // ... assertions ...

```text
## Best Practices

1. **Always use deterministic seeding for random data**
   ```mojo

   TestFixtures.set_seed()  // Do this

   ```

1. **Choose descriptive fixture parameters**

   ```mojo

   // Good
   var training_data = create_test_matrix(1000, 784, 0.0)

   // Less clear
   var data = create_test_matrix(1000, 784)

   ```

2. **Don't modify shared fixtures**

   ```mojo

   // Each test gets its own copy, modifications are safe
   var vec = create_test_vector(10, 1.0)
   vec[0] = 999.0  // Okay, doesn't affect other tests

   ```

3. **Document expected fixture properties in tests**

   ```mojo

   fn test_normalization() raises:
       """Test normalization with known input."""
       // Create input: [0, 1, 2, 3, 4]
       var input = create_sequential_vector(5, 0.0)

       var normalized = normalize(input)
       // Expected: mean=0, std=1
       assert_almost_equal(mean(normalized), 0.0, tolerance=1e-6)

   ```

```text

### 5. Naming Conventions

### Establish Standard Patterns

### Fixtures returning data

- `create_*` - Generators that create new data each call
- `*_tensor` - Tensor fixtures
- `*_model` - Model fixtures
- `*_dataset` - Dataset fixtures

### Fixtures for setup/teardown

- `set_*` - Setup functions
- `reset_*` - Reset/cleanup functions

### Examples

```mojo
// Data creation
create_test_vector()
create_test_matrix()
random_tensor()
ones_tensor()

// Models
simple_linear_model()
small_mlp()

// Datasets
toy_classification_dataset()
toy_regression_dataset()

// Setup/teardown
set_seed()
reset_global_state()
```text

## Deliverables Summary

### 1. Module Organization

- [ ] Review current organization (already good)
- [ ] Document module structure
- [ ] Create migration plan if reorganization needed (unlikely)

### 2. Import Documentation

- [ ] Document recommended import patterns
- [ ] Create examples for each pattern
- [ ] Add to testing guide

### 3. Fixture Catalog

- [ ] Create `docs/testing/fixtures-catalog.md`
- [ ] Document all current fixtures
- [ ] Add placeholders for future fixtures
- [ ] Include usage examples

### 4. Usage Guide

- [ ] Create `docs/testing/fixture-usage.md`
- [ ] Document common patterns
- [ ] Provide best practices
- [ ] Include anti-patterns to avoid

### 5. Naming Standards

- [ ] Document naming conventions
- [ ] Ensure current fixtures follow standards
- [ ] Create guidelines for new fixtures

## References

- **Source Plan**: [notes/plan/02-shared-library/04-testing/01-test-framework/03-test-fixtures/plan.md](../../../plan/02-shared-library/04-testing/01-test-framework/03-test-fixtures/plan.md)
- **Related Issues**:
  - Issue #443: [Plan] Test Fixtures
  - Issue #444: [Test] Test Fixtures
  - Issue #445: [Impl] Test Fixtures
  - Issue #447: [Cleanup] Test Fixtures
- **Existing Code**:
  - `/tests/shared/conftest.mojo`
  - `/tests/helpers/fixtures.mojo`

## Implementation Notes

### Packaging Focus

**Key Insight**: Fixture packaging is about **organization and discoverability**, not binary distribution.

### Priorities

1. **High**: Create fixture catalog documentation
1. **High**: Create usage guide with patterns
1. **Medium**: Document import patterns
1. **Low**: Reorganize modules (only if needed)

### Minimal Changes Principle

**Current State**: Fixtures are well-organized in conftest.mojo
**Action**: Document what exists, don't reorganize unnecessarily

### Focus on

- Making fixtures easy to find (catalog)
- Making fixtures easy to use (usage guide)
- Making fixtures follow consistent patterns (naming conventions)
