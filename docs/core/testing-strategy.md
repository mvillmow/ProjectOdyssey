# Testing Strategy

Comprehensive testing approach for ML Odyssey following Test-Driven Development (TDD) principles.

## Overview

ML Odyssey uses TDD as a core development practice. Tests guide implementation, catch bugs early, and enable confident
refactoring.

**Philosophy**: Write tests first, implement second. Tests are specifications, not afterthoughts.

## Testing Philosophy

### Why TDD

**Benefits**:

- **Better Design** - Writing tests first clarifies interfaces
- **Fewer Bugs** - Catch errors before they reach production
- **Refactoring Safety** - Change implementation with confidence
- **Living Documentation** - Tests show how code should be used
- **Faster Debugging** - Failures pinpoint exactly what broke

### What to Test

**MUST Test** (Critical):

- Core functionality and primary use cases
- Public API contracts and guarantees
- Data integrity (anything that can corrupt/lose data)
- Security-sensitive code
- Integration points between modules

**SHOULD Test** (Important):

- Common user workflows
- Boundary conditions and edge cases
- Error handling for expected failures
- Performance requirements (if specified)

**DON'T Test**:

- Trivial getters/setters
- Private implementation details
- Third-party library code
- 100% line coverage for its own sake

### Coverage Goals

- **Line Coverage**: ≥90%
- **Branch Coverage**: ≥85%
- **Function Coverage**: 100% of public functions
- **Critical Path**: 100% of core functionality

**Rule**: If deleting a test wouldn't reduce confidence, delete it.

## Test Structure

### Directory Organization

```text

tests/
├── unit/              # Fast unit tests
├── integration/       # Cross-module tests
├── shared/            # Shared library tests
│   ├── core/          # Core module tests
│   ├── training/      # Training tests
│   └── data/          # Data tests
├── agents/            # Agent system tests
└── fixtures/          # Shared test data

```text

### Test File Naming

**Mojo Tests**:

- Files: `test_*.mojo` (e.g., `test_layers.mojo`)
- Functions: `test_<component>_<behavior>` (e.g., `test_sgd_basic_update`)

**Python Tests**:

- Files: `test_*.py` (e.g., `test_validation.py`)
- Classes: `Test<ComponentName>` (e.g., `TestAgentDelegation`)
- Functions: `test_<behavior>` (e.g., `test_delegation_chains`)

## Test Types

### 1. Unit Tests

**Purpose**: Test individual functions/components in isolation

**Characteristics**:

- Fast (< 1 second per test)
- No external dependencies
- Test single function/class
- Deterministic results

**Example**:

```mojo

fn test_sgd_basic_update():
    """Test SGD performs parameter update correctly."""
    # Arrange
    var params = Tensor(List[Float32](1.0, 2.0, 3.0), Shape(3))
    var grads = Tensor(List[Float32](0.1, 0.2, 0.3), Shape(3))
    var optimizer = SGD(lr=0.1)

    # Act
    optimizer.step(params, grads)

    # Assert: params = params - lr * grads
    assert_almost_equal(params[0], 0.99)  # 1.0 - 0.1*0.1
    assert_almost_equal(params[1], 1.98)  # 2.0 - 0.1*0.2
    assert_almost_equal(params[2], 2.97)  # 3.0 - 0.1*0.3

```text

### 2. Integration Tests

**Purpose**: Test component interactions and workflows

**Characteristics**:

- Test multiple components together
- End-to-end workflows
- May use real dependencies
- Slower than unit tests (seconds to minutes)

**Example**:

```mojo

fn test_training_workflow():
    """Test complete training loop with validation."""
    var model = SimpleModel(in_features=10, out_features=2)
    var train_data = create_synthetic_dataset(n_samples=100)
    var val_data = create_synthetic_dataset(n_samples=20)
    var optimizer = SGD(lr=0.01)

    var losses = List[Float32]()
    for epoch in range(5):
        var epoch_loss = train_epoch(model, train_data, optimizer)
        losses.append(epoch_loss)

    # Verify loss decreases
    assert_true(losses[-1] < losses[0])

    # Verify validation accuracy
    var val_acc = evaluate(model, val_data)
    assert_true(val_acc > 0.5)  # Better than random

```text

### 3. Numerical Accuracy Tests

**Purpose**: Validate ML correctness against reference implementations

**Example**:

```mojo

fn test_sgd_matches_pytorch():
    """Test SGD produces same results as PyTorch."""
    # Mojo implementation
    var params_mojo = Tensor(List[Float32](1.0, 2.0, 3.0), Shape(3))
    var grads = Tensor(List[Float32](0.1, 0.2, 0.3), Shape(3))
    var optimizer_mojo = SGD(lr=0.1)
    optimizer_mojo.step(params_mojo, grads)

    # PyTorch reference
    var params_torch = torch.tensor([1.0, 2.0, 3.0])
    var grads_torch = torch.tensor([0.1, 0.2, 0.3])
    params_torch -= 0.1 * grads_torch

    # Compare
    for i in range(3):
        assert_almost_equal(params_mojo[i], params_torch[i].item(), tolerance=1e-6)

```text

## TDD Workflow

### Red-Green-Refactor Cycle

**1. Red - Write Failing Test**

Write the test BEFORE implementation:

```mojo

fn test_tensor_multiply():
    """Test element-wise multiplication."""
    var a = Tensor(List[Float32](2.0, 3.0), Shape(2))
    var b = Tensor(List[Float32](4.0, 5.0), Shape(2))
    var result = a * b  # Doesn't exist yet
    assert_almost_equal(result[0], 8.0)
    assert_almost_equal(result[1], 15.0)

```text

Run test - should FAIL.

**2. Green - Minimal Implementation**

Implement just enough to pass:

```mojo

struct Tensor:
    fn __mul__(self, other: Tensor) -> Tensor:
        var result = Tensor(self.shape)
        for i in range(self.size()):
            result[i] = self[i] * other[i]
        return result

```text

Run test - should PASS.

**3. Refactor - Improve Code**

Optimize while keeping tests passing:

```mojo

struct Tensor:
    fn __mul__(self, other: Tensor) -> Tensor:
        var result = Tensor(self.shape)
        alias width = simdwidthof[DType.float32]()

        @parameter
        fn multiply_simd[w: Int](idx: Int):
            result.store[w](idx, self.load[w](idx) * other.load[w](idx))

        vectorize[width, multiply_simd](self.size())
        return result

```text

Run tests - should still PASS.

## Writing Tests

### Test Structure (Arrange-Act-Assert)

```mojo

fn test_function_name() raises:
    """Clear docstring describing what is tested."""
    # Arrange: Set up test data and preconditions
    var input = create_test_data()
    var expected = compute_expected_result()

    # Act: Execute the code under test
    var result = function_under_test(input)

    # Assert: Verify expected behavior
    assert_equal(result, expected)

```text

### Mojo Test Patterns

**Use `fn` for Tests**:

```mojo

fn test_tensor_addition() raises:
    """Test tensor element-wise addition."""
    var a = Tensor(List[Float32](1.0, 2.0), Shape(2))
    var b = Tensor(List[Float32](3.0, 4.0), Shape(2))
    var result = a + b
    assert_almost_equal(result[0], 4.0)
    assert_almost_equal(result[1], 6.0)

```text

**Gradient Checking**:

```mojo

fn test_layer_gradients():
    """Test layer computes correct gradients."""
    var layer = Linear(10, 5)
    var input = Tensor.randn(2, 10)

    # Numerical gradient
    var numerical = compute_numerical_gradient(layer, input, epsilon=1e-5)

    # Analytical gradient
    var output = layer.forward(input)
    var grad_output = Tensor.ones_like(output)
    layer.backward(grad_output, input)

    # Compare
    assert_almost_equal(layer.weight.grad, numerical, tolerance=1e-3)

```text

## Running Tests

### Local Execution

```bash

# Run all Mojo tests
mojo test tests/

# Run specific test file
mojo test tests/shared/core/test_layers.mojo

# Run with verbose output
mojo test -v tests/

# Run Python tests
pytest tests/

# Run with coverage
pytest --cov=scripts --cov-report=html tests/

```text

### CI Integration

All tests run automatically in CI:

```yaml

# .github/workflows/test.yml
name: Test Suite

on: [pull_request, push]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:

      - uses: actions/checkout@v4
      - name: Setup Mojo
      - name: Setup Mojo

        uses: modularml/setup-mojo@v1

      - name: Run tests
      - name: Run tests

        run: mojo test tests/

```text

## Best Practices

### 1. Test One Thing Per Test

```mojo

# Good: Focused test
fn test_sgd_basic_update():
    """Test SGD performs basic parameter update."""
    # Test only basic update
    pass

fn test_sgd_momentum():
    """Test SGD applies momentum correctly."""
    # Test only momentum
    pass

# Avoid: Testing multiple concepts
fn test_sgd_everything():
    """Test update, momentum, weight decay, LR."""
    # Too much in one test
    pass

```text

### 2. Clear Test Names

```mojo

# Good: Descriptive names
fn test_sgd_updates_parameters_correctly()
fn test_dataloader_handles_empty_dataset()
fn test_conv2d_output_shape_with_padding()

# Avoid: Vague names
fn test_sgd()
fn test_data()
fn test_conv()

```text

### 3. Test Independence

```mojo

# Good: Each test is independent
fn test_optimizer_step_1():
    var optimizer = SGD(lr=0.1)
    # Test doesn't depend on other tests
    pass

fn test_optimizer_step_2():
    var optimizer = SGD(lr=0.1)
    # Fresh optimizer for each test
    pass

# Avoid: Tests depend on shared state
var global_optimizer = SGD(lr=0.1)  # DON'T DO THIS

fn test_step_1():
    global_optimizer.step()  # Modifies shared state

fn test_step_2():
    # This depends on test_1 running first - BAD!
    global_optimizer.step()

```text

### 4. Validate Gradients

Always check gradients for ML components:

```mojo

fn validate_gradients(layer: Layer, input: Tensor):
    """Compare analytical vs numerical gradients."""
    var epsilon = 1e-5
    var numerical = compute_numerical_gradient(layer, input, epsilon)
    var analytical = compute_analytical_gradient(layer, input)
    assert_almost_equal(numerical, analytical, tolerance=1e-4)

```text

### 5. Test Edge Cases

```mojo

fn test_edge_cases():
    """Test boundary conditions."""
    var layer = Linear(10, 5)

    # Empty batch
    var empty = Tensor.zeros(0, 10)
    var out = layer.forward(empty)
    assert_equal(out.shape[0], 0)

    # Single example
    var single = Tensor.randn(1, 10)
    var result = layer.forward(single)
    assert_equal(result.shape, (1, 5))

    # Large batch
    var large = Tensor.randn(1000, 10)
    var output = layer.forward(large)
    assert_equal(output.shape, (1000, 5))

```text

## Test Coverage

### Measuring Coverage

```bash

# Mojo coverage (when available)
mojo test --coverage tests/

# Python coverage
pytest --cov=scripts --cov-report=html tests/
open htmlcov/index.html

```text

### Coverage Reporting in CI

```yaml

- name: Check Coverage
- name: Check Coverage

  run: |
    pytest --cov=shared --cov-report=xml tests/
    # Upload to coverage service

```text

## Troubleshooting

### Test Failures

```bash

# Run specific test for debugging
mojo test tests/test_file.mojo::test_specific_function

# Add debug output
fn test_my_feature():
    var result = function_under_test(data)
    print("Debug result:", result)  # Temporary
    assert_equal(result, expected)

```text

### Flaky Tests

If a test passes sometimes and fails other times:

1. Check for uninitialized variables
2. Verify no dependency on test execution order
3. Check for timing issues (use fixed seeds)
4. Ensure proper cleanup between tests

## Related Documentation

- [Shared Library](shared-library.md) - Components to test
- [Mojo Patterns](mojo-patterns.md) - Testing patterns
- [Custom Layers](../advanced/custom-layers.md) - Testing custom components
- [Debugging](../advanced/debugging.md) - Debugging tests

## Summary

**Testing Philosophy**:

- TDD is mandatory - write tests first
- Focus on critical paths, not coverage numbers
- Test behavior, not implementation
- Keep tests fast and independent

**TDD Workflow**:

1. Write failing test (Red)
2. Implement minimal code (Green)
3. Refactor for quality (Refactor)
4. Repeat

**Test Types**:

- Unit tests (fast, isolated)
- Integration tests (workflows)
- Numerical accuracy tests (validate ML correctness)

**Key Practices**:

1. One concept per test
2. Clear, descriptive names
3. Test independence
4. Validate gradients
5. Test edge cases

**Remember**: Tests are specifications that guide implementation and enable confident refactoring. Write tests you trust.
