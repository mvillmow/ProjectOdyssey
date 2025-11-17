---
name: phase-test-tdd
description: Generate test files and coordinate test-driven development following ML Odyssey TDD principles. Use when creating tests before implementation or when asked to write tests first.
---

# Test-Driven Development Skill

This skill generates test files and coordinates TDD workflow for ML Odyssey components.

## When to Use

- User asks to write tests (e.g., "create tests for X")
- Starting test phase of development
- Before implementation (TDD principle)
- Need to generate test templates

## TDD Workflow

### 1. Write Tests First

```bash
# Generate test file from template
./scripts/generate_test.sh <component-name> <test-type>

# Example: Generate unit test for tensor operations
./scripts/generate_test.sh "tensor_ops" "unit"
```

### 2. Run Tests (Red)

```bash
# Run tests - should fail initially
mojo test tests/<test-file>

# Or use Python for test infrastructure
pytest tests/<test-file>
```

### 3. Implement Code (Green)

Implement minimal code to make tests pass.

### 4. Refactor (Refactor)

Improve code while keeping tests passing.

## Test Types

### Unit Tests

- Test individual functions/classes
- Fast execution
- Isolated from dependencies
- Location: `tests/unit/`

### Integration Tests

- Test component interactions
- May use real dependencies
- Location: `tests/integration/`

### Performance Tests

- Benchmark critical operations
- Verify SIMD optimizations
- Location: `tests/performance/`

## Mojo Test Structure

```mojo
from testing import assert_equal, assert_true

fn test_function_name() raises:
    """Test description."""
    # Arrange
    let input = setup_test_data()

    # Act
    let result = function_under_test(input)

    # Assert
    assert_equal(result, expected_value)

fn test_edge_case() raises:
    """Test edge case description."""
    # Test implementation
    pass
```

## Python Test Structure

```python
import pytest

class TestComponentName:
    """Test suite for ComponentName."""

    def test_basic_functionality(self):
        """Test basic functionality."""
        # Arrange
        input_data = setup_data()

        # Act
        result = component.process(input_data)

        # Assert
        assert result == expected

    def test_edge_case(self):
        """Test edge case."""
        # Implementation
        pass
```

## Test Coverage Requirements

- **Minimum coverage**: 80% for new code
- **Critical paths**: 100% coverage
- **Edge cases**: Must be tested
- **Error handling**: Must be tested

## Error Handling

- **Test fails unexpectedly**: Check test setup
- **Test timeout**: Optimize or increase timeout
- **Import errors**: Verify dependencies installed
- **Mojo test failures**: Check type safety and memory management

## Examples

**Generate unit test:**

```bash
./scripts/generate_test.sh "matrix_multiply" "unit"
```

**Generate integration test:**

```bash
./scripts/generate_test.sh "neural_network" "integration"
```

**Run specific test:**

```bash
./scripts/run_test.sh "test_matrix_multiply"
```

## Scripts Available

- `scripts/generate_test.sh` - Generate test file from template
- `scripts/run_test.sh` - Run specific test file
- `scripts/check_coverage.sh` - Check test coverage

## Templates

- `templates/unit_test_mojo.mojo` - Mojo unit test template
- `templates/unit_test_python.py` - Python unit test template
- `templates/integration_test.py` - Integration test template

## TDD Principles

1. **Write test first** - Before any implementation
2. **Red-Green-Refactor** - Fail → Pass → Improve
3. **Small steps** - One test at a time
4. **Fast feedback** - Tests should run quickly
5. **Comprehensive** - Cover edge cases and errors

See CLAUDE.md for development principles including TDD requirements.
