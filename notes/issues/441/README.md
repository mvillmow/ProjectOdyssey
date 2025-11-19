# Issue #441: [Package] Test Utilities

## Objective

Package test utilities for easy reuse across the project and potentially in other projects, ensuring assertion helpers, comparison functions, and data generators are well-organized and accessible.

## Deliverables

- Organized test utility modules
- Clear import paths for utilities
- Usage documentation and examples
- API reference for all utilities
- Best practices guide

## Success Criteria

- [ ] Test utilities are importable from any test file
- [ ] Import paths are intuitive and consistent
- [ ] Documentation covers all utilities with examples
- [ ] API reference is complete and accurate
- [ ] Common usage patterns are documented

## Current State Analysis

### Existing Organization

**File Structure**:
```text
tests/
├── shared/
│   └── conftest.mojo         # Core test utilities (337 lines)
│       ├── Assertion Functions (7 functions)
│       ├── Test Fixtures (TestFixtures struct)
│       ├── Benchmark Utilities (BenchmarkResult struct)
│       ├── Test Helpers (2 functions)
│       └── Test Data Generators (3 functions)
├── helpers/
│   ├── assertions.mojo        # ExTensor assertions (365 lines)
│   │   ├── Basic Assertions (4 functions)
│   │   └── ExTensor-Specific Assertions (8 functions)
│   ├── fixtures.mojo          # Placeholder (17 lines)
│   └── utils.mojo             # Placeholder (14 lines)
```

**Current Import Patterns**:
```mojo
// From any test file
from tests.shared.conftest import assert_true, assert_equal, TestFixtures
from tests.helpers.assertions import assert_shape, assert_all_close
```

### Packaging Considerations

**What "Packaging" Means for Test Utilities**:
1. **Organization**: Clear module structure
2. **Accessibility**: Easy imports from any test
3. **Documentation**: Usage guides and API reference
4. **Discoverability**: Developers can find what they need
5. **Consistency**: Uniform patterns across utilities

**NOT about**:
- Creating `.mojopkg` binary packages (test utils are source-distributed)
- External distribution (internal project use)
- Version management (follows project versioning)

## Packaging Strategy

### 1. Module Organization

**Current Structure is Good**:
- `conftest.mojo` for general test utilities ✓
- `helpers/assertions.mojo` for ExTensor-specific assertions ✓
- Clear separation of concerns ✓

**Potential Improvements** (if needed):
```text
tests/
├── shared/
│   ├── conftest.mojo           # Core utilities (keep as is)
│   └── generators.mojo         # Data generators (if growing large)
├── helpers/
│   ├── assertions.mojo         # Assertions (keep as is)
│   ├── fixtures.mojo           # Fixtures (implement or remove)
│   ├── utils.mojo              # Utilities (implement or remove)
│   └── mocks.mojo              # Mock objects (if implemented)
```

**Decision**: Only split if modules become unwieldy (>500 lines)

### 2. Import Path Standardization

**Recommended Pattern**:
```mojo
// Core assertions and utilities
from tests.shared.conftest import (
    assert_true, assert_false, assert_equal, assert_almost_equal,
    TestFixtures, BenchmarkResult,
    create_test_vector, create_test_matrix
)

// ExTensor-specific assertions
from tests.helpers.assertions import (
    assert_shape, assert_dtype, assert_all_close,
    assert_contiguous
)

// Fixtures (if implemented)
from tests.helpers.fixtures import (
    random_tensor, sequential_tensor
)
```

**Best Practices**:
- Import explicitly (no `import *`)
- Group imports by module
- Document commonly used imports

### 3. Documentation Structure

**Create**: `docs/testing/` directory with:

**3.1. Quick Start Guide** (`docs/testing/quick-start.md`):
```markdown
# Test Utilities Quick Start

## Writing Your First Test

```mojo
from tests.shared.conftest import assert_true, assert_equal

fn test_my_feature() raises:
    """Test description."""
    var result = my_function(42)
    assert_equal(result, expected_value)
```

## Common Assertions

- `assert_true(condition)` - Boolean assertion
- `assert_equal(a, b)` - Exact equality
- `assert_almost_equal(a, b, tolerance)` - Float near-equality

## Test Data Generation

```mojo
from tests.shared.conftest import create_test_vector

fn test_with_data() raises:
    var data = create_test_vector(100, value=1.0)
    # Use data in test
```

**3.2. API Reference** (`docs/testing/api-reference.md`):
```markdown
# Test Utilities API Reference

## Assertion Functions

### assert_true(condition, message="")
Assert that condition is true.

**Parameters**:
- `condition: Bool` - The condition to check
- `message: String` - Optional error message

**Raises**:
- `Error` if condition is false

**Example**:
```mojo
assert_true(x > 0, "x must be positive")
```

### assert_equal[T: Comparable](a, b, message="")
Assert exact equality of two values.

**Parameters**:
- `a: T` - First value
- `b: T` - Second value
- `message: String` - Optional error message

**Raises**:
- `Error` if a != b

**Example**:
```mojo
assert_equal(result, 42, "Expected 42")
```

[... document all utilities ...]
```

**3.3. Best Practices** (`docs/testing/best-practices.md`):
```markdown
# Test Utilities Best Practices

## Choosing the Right Assertion

- **Exact equality**: Use `assert_equal` for integers, strings
- **Float comparison**: Use `assert_almost_equal` or `assert_close_float`
- **Tensor comparison**: Use `assert_all_close` from helpers.assertions
- **Boolean conditions**: Use `assert_true` or `assert_false`

## Handling Floating-Point Comparisons

```mojo
// WRONG: Exact equality fails due to floating-point precision
assert_equal(0.1 + 0.2, 0.3)  // May fail!

// RIGHT: Use tolerance
assert_almost_equal(0.1 + 0.2, 0.3, tolerance=1e-10)
```

## Test Data Generation

```mojo
// Use deterministic seeding for reproducible tests
TestFixtures.set_seed()
var random_data = create_random_vector(100)
```

## Error Message Guidelines

```mojo
// GOOD: Descriptive message
assert_true(x > 0, "Temperature must be positive, got " + str(x))

// OKAY: Default message
assert_true(x > 0)  // Uses generic "Assertion failed"

// AVOID: Redundant message
assert_true(x > 0, "x > 0")  // Doesn't add information
```
```

**3.4. Examples** (`docs/testing/examples.md`):
```markdown
# Test Utilities Examples

## Example 1: Simple Unit Test

```mojo
from tests.shared.conftest import assert_equal

fn test_addition() raises:
    """Test integer addition."""
    var result = 2 + 2
    assert_equal(result, 4)
```

## Example 2: Float Comparison

```mojo
from tests.shared.conftest import assert_almost_equal

fn test_division() raises:
    """Test floating-point division."""
    var result = 1.0 / 3.0
    assert_almost_equal(result, 0.333333, tolerance=1e-6)
```

## Example 3: Tensor Validation

```mojo
from tests.helpers.assertions import assert_shape, assert_all_close

fn test_tensor_operation() raises:
    """Test tensor transformation."""
    var input = create_tensor((3, 4))
    var output = my_transform(input)

    assert_shape(output, DynamicVector[Int](3, 4))
    assert_all_close(output, expected, rtol=1e-5, atol=1e-8)
```
```

### 4. Utility Catalog

**Create**: `tests/README.md` with utility catalog:
```markdown
# Test Utilities Catalog

## Core Utilities (`tests/shared/conftest.mojo`)

### Assertions
- `assert_true(condition, message="")`
- `assert_false(condition, message="")`
- `assert_equal[T](a, b, message="")`
- `assert_not_equal[T](a, b, message="")`
- `assert_almost_equal(a, b, tolerance=1e-6, message="")`
- `assert_greater(a, b, message="")`
- `assert_less(a, b, message="")`

### Fixtures
- `TestFixtures.deterministic_seed() -> Int`
- `TestFixtures.set_seed()`

### Data Generators
- `create_test_vector(size, value=1.0) -> List[Float32]`
- `create_test_matrix(rows, cols, value=1.0) -> List[List[Float32]]`
- `create_sequential_vector(size, start=0.0) -> List[Float32]`

## ExTensor Utilities (`tests/helpers/assertions.mojo`)

### Shape & Type Assertions
- `assert_shape[T](tensor, expected, message="")`
- `assert_dtype[T](tensor, expected_dtype, message="")`
- `assert_numel[T](tensor, expected_numel, message="")`
- `assert_dim[T](tensor, expected_dim, message="")`

### Value Assertions
- `assert_value_at[T](tensor, index, expected, tolerance=1e-8, message="")`
- `assert_all_values[T](tensor, expected, tolerance=1e-8, message="")`
- `assert_all_close[T](a, b, rtol=1e-5, atol=1e-8, message="")`

### Memory Assertions
- `assert_contiguous[T](tensor, message="")`
```

## Deliverables Summary

### 1. Module Organization
- [ ] Review current structure (already good)
- [ ] Split modules if needed (unlikely)
- [ ] Document module organization

### 2. Import Documentation
- [ ] Document recommended import patterns
- [ ] Create import examples for common scenarios
- [ ] Add to quick start guide

### 3. API Documentation
- [ ] Create `docs/testing/api-reference.md`
- [ ] Document all assertions with examples
- [ ] Document all generators with examples
- [ ] Document all fixtures with examples

### 4. Usage Documentation
- [ ] Create `docs/testing/quick-start.md`
- [ ] Create `docs/testing/best-practices.md`
- [ ] Create `docs/testing/examples.md`
- [ ] Create `tests/README.md` utility catalog

### 5. Organization Documentation
- [ ] Document module structure
- [ ] Explain file organization rationale
- [ ] Guide for adding new utilities

## References

- **Source Plan**: [notes/plan/02-shared-library/04-testing/01-test-framework/02-test-utilities/plan.md](../../../plan/02-shared-library/04-testing/01-test-framework/02-test-utilities/plan.md)
- **Related Issues**:
  - Issue #438: [Plan] Test Utilities
  - Issue #439: [Test] Test Utilities
  - Issue #440: [Impl] Test Utilities
  - Issue #442: [Cleanup] Test Utilities
- **Existing Code**:
  - `/tests/shared/conftest.mojo`
  - `/tests/helpers/assertions.mojo`

## Implementation Notes

### Packaging Approach

**Key Insight**: For test utilities, "packaging" is primarily about **documentation and organization**, not binary distribution.

**Focus**:
1. Clear import paths (already good)
2. Comprehensive documentation (needs creation)
3. Usage examples (needs creation)
4. Discoverability (catalog in README)

### Minimal Changes Principle

**Current State**: Utilities are well-organized and importable
**Action Needed**: Document what exists, don't reorganize unnecessarily

**Priorities**:
1. **High**: Create documentation (API reference, quick start, examples)
2. **Medium**: Create utility catalog in tests/README.md
3. **Low**: Reorganize modules (only if they grow unwieldy)

### Next Steps

1. Create API reference documentation
2. Create quick start guide
3. Create best practices guide
4. Create example collection
5. Create utility catalog
6. Verify documentation accuracy with actual usage
