# Testing Strategy

ML Odyssey's comprehensive testing philosophy, practices, and guidelines.

## Overview

ML Odyssey follows a rigorous testing strategy based on Test-Driven Development (TDD) principles. Every component,
from low-level tensor operations to high-level training loops, is thoroughly tested to ensure correctness,
performance, and reliability.

## Testing Philosophy

### Core Principles

1. **Test-Driven Development**: Write tests before implementation
2. **Comprehensive Coverage**: All public APIs must be tested
3. **Fast Execution**: Unit tests should run in < 1 second each
4. **Isolated Tests**: Each test is independent and can run in any order
5. **Readable Tests**: Tests serve as documentation

### Test Types

ML Odyssey uses a hierarchical testing approach:

```text
Unit Tests → Integration Tests → System Tests → Benchmarks
```

## Test Organization

Tests mirror the source structure:

```text
tests/
├── shared/              # Tests for shared library
│   ├── core/            # Core component tests
│   │   ├── test_layers.mojo
│   │   ├── test_activations.mojo
│   │   └── test_ops.mojo
│   ├── training/        # Training system tests
│   │   ├── test_optimizers.mojo
│   │   ├── test_trainer.mojo
│   │   └── test_callbacks.mojo
│   ├── data/            # Data processing tests
│   │   ├── test_datasets.mojo
│   │   ├── test_loaders.mojo
│   │   └── test_transforms.mojo
│   └── utils/           # Utility tests
│       ├── test_config.mojo
│       ├── test_logging.mojo
│       └── test_profiling.mojo
├── papers/              # Tests for paper implementations
│   ├── lenet5/
│   │   ├── test_model.mojo
│   │   ├── test_training.mojo
│   │   └── test_evaluation.mojo
├── foundation/          # Infrastructure tests
│   └── docs/
│       ├── test_documentation.py
│       └── test_links.py
└── conftest.py          # Pytest configuration
```

## Unit Testing

Unit tests verify individual components in isolation.

### Writing Unit Tests

```mojo
from testing import assert_equal, assert_true, assert_raises
from shared.core import Linear

fn test_linear_forward():
    """Test Linear layer forward pass."""
    var layer = Linear(input_size=10, output_size=5)
    var input = Tensor.randn(8, 10)  # Batch of 8

    var output = layer.forward(input)

    # Verify output shape
    assert_equal(output.shape[0], 8)
    assert_equal(output.shape[1], 5)

fn test_linear_backward():
    """Test Linear layer backward pass."""
    var layer = Linear(input_size=10, output_size=5)
    var input = Tensor.randn(8, 10)
    var grad_output = Tensor.randn(8, 5)

    var output = layer.forward(input)
    var grad_input = layer.backward(grad_output)

    # Verify gradient shape
    assert_equal(grad_input.shape, input.shape)

    # Verify weight gradients exist
    assert_true(layer.weight.grad is not None)
    assert_true(layer.bias.grad is not None)

fn test_linear_parameter_update():
    """Test parameters can be updated."""
    var layer = Linear(input_size=10, output_size=5)
    var old_weight = layer.weight.clone()

    # Simulate gradient descent
    layer.weight = layer.weight - 0.1 * layer.weight.grad

    # Verify weights changed
    assert_true((layer.weight != old_weight).any())
```

### Test Naming Conventions

- **File names**: `test_<module>.mojo`
- **Function names**: `test_<component>_<behavior>()`
- **Descriptive**: Name clearly describes what is tested

Examples:

```mojo
fn test_relu_forward()              # Good: Clear what's tested
fn test_output_shape()              # Bad: Too vague
fn test_sgd_updates_parameters()    # Good: Specific behavior
fn test_training()                  # Bad: Too broad
```

### Testing Patterns

#### Pattern 1: Test Shape Correctness

```mojo
fn test_conv2d_output_shape():
    """Verify Conv2D produces correct output dimensions."""
    var conv = Conv2D(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1
    )

    var input = Tensor.randn(8, 3, 32, 32)
    var output = conv.forward(input)

    assert_equal(output.shape, [8, 64, 32, 32])
```

#### Pattern 2: Test Gradient Flow

```mojo
fn test_model_gradient_flow():
    """Verify gradients flow through entire model."""
    var model = Sequential([
        Linear(10, 20),
        ReLU(),
        Linear(20, 5),
    ])

    var input = Tensor.randn(1, 10)
    var target = Tensor.zeros(1, 5)
    target[0, 2] = 1.0

    var output = model.forward(input)
    var loss = cross_entropy_loss(output, target)
    loss.backward()

    # All parameters should have gradients
    for param in model.parameters():
        assert_true(param.grad is not None)
        assert_true(param.grad.abs().sum() > 0)
```

#### Pattern 3: Test Numerical Stability

```mojo
fn test_softmax_numerical_stability():
    """Verify Softmax handles large values without overflow."""
    var softmax = Softmax()

    # Large values that could cause overflow
    var input = Tensor([1000.0, 1001.0, 999.0])
    var output = softmax.forward(input)

    # Output should still be valid probabilities
    assert_true(output.sum() ≈ 1.0)
    assert_true((output >= 0.0).all())
    assert_true((output <= 1.0).all())
```

#### Pattern 4: Test Edge Cases

```mojo
fn test_batchnorm_single_sample():
    """Test BatchNorm with batch size of 1."""
    var bn = BatchNorm2D(num_features=64)

    var input = Tensor.randn(1, 64, 32, 32)  # Batch size = 1
    var output = bn.forward(input)

    assert_equal(output.shape, input.shape)

fn test_empty_batch():
    """Test handling of empty batch."""
    var model = Linear(10, 5)
    var input = Tensor.empty(0, 10)

    with assert_raises[ValueError]():
        _ = model.forward(input)
```

## Integration Testing

Integration tests verify components work together correctly.

### End-to-End Training

```mojo
fn test_training_end_to_end():
    """Test complete training pipeline."""
    # Create simple dataset
    var X = Tensor.randn(100, 10)
    var y = Tensor.randint(0, 5, shape=(100,))
    var dataset = TensorDataset(X, y)
    var loader = BatchLoader(dataset, batch_size=16)

    # Create model
    var model = Sequential([
        Linear(10, 20),
        ReLU(),
        Linear(20, 5),
    ])

    # Train for a few epochs
    var optimizer = SGD(lr=0.01)
    var loss_fn = CrossEntropyLoss()

    var initial_loss: Float64
    var final_loss: Float64

    for epoch in range(10):
        for batch in loader:
            var inputs, targets = batch
            var output = model.forward(inputs)
            var loss = loss_fn(output, targets)

            if epoch == 0:
                initial_loss = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step(model.parameters())

            final_loss = loss.item()

    # Loss should decrease
    assert_true(final_loss < initial_loss)
```

### Data Pipeline Testing

```mojo
fn test_data_augmentation_pipeline():
    """Test data augmentation transformations."""
    var transform = Compose([
        RandomCrop(size=28, padding=4),
        RandomHorizontalFlip(p=0.5),
        Normalize(mean=0.5, std=0.5),
    ])

    var image = Tensor.randn(1, 32, 32)
    var augmented = transform(image)

    # Output should have correct shape
    assert_equal(augmented.shape, [1, 28, 28])

    # Values should be normalized
    assert_true(augmented.mean() ≈ 0.0, atol=0.5)
```

## System Testing

System tests verify entire workflows and paper implementations.

### Paper Implementation Testing

```mojo
fn test_lenet5_mnist_overfitting():
    """Test LeNet-5 can overfit small MNIST subset."""
    # Small subset for quick overfitting test
    var X_train, y_train = load_mnist_subset(num_samples=50)

    var model = LeNet5()
    var optimizer = SGD(lr=0.01)
    var trainer = Trainer(model, optimizer, CrossEntropyLoss())

    var train_loader = BatchLoader(
        TensorDataset(X_train, y_train),
        batch_size=10
    )

    # Train until overfitting
    trainer.train(train_loader, epochs=50)

    # Should achieve high accuracy on training set
    var accuracy = evaluate_accuracy(model, train_loader)
    assert_true(accuracy > 0.95, "Should overfit small dataset")

fn test_lenet5_achieves_target_accuracy():
    """Test LeNet-5 reaches expected accuracy on MNIST."""
    # This is a longer-running test (may take minutes)
    var train_data, test_data = load_mnist()
    var model = LeNet5()

    # Train with paper hyperparameters
    var trainer = Trainer(
        model,
        SGD(lr=0.01, momentum=0.9),
        CrossEntropyLoss()
    )

    trainer.train(
        BatchLoader(train_data, batch_size=32),
        epochs=100
    )

    # Evaluate on test set
    var test_accuracy = evaluate_accuracy(
        model,
        BatchLoader(test_data, batch_size=32)
    )

    # Should match paper results (±1%)
    var target_accuracy = 0.992
    assert_true(
        abs(test_accuracy - target_accuracy) < 0.01,
        f"Accuracy {test_accuracy} should be close to {target_accuracy}"
    )
```

## Test Utilities

### Common Test Fixtures

```mojo
# In conftest.py or conftest.mojo

@fixture
fn random_tensor() -> Tensor:
    """Generate random tensor for testing."""
    return Tensor.randn(8, 10)

@fixture
fn simple_model() -> Sequential:
    """Create simple model for testing."""
    return Sequential([
        Linear(10, 20),
        ReLU(),
        Linear(20, 5),
    ])

@fixture
fn sample_dataset() -> TensorDataset:
    """Create sample dataset for testing."""
    var X = Tensor.randn(100, 10)
    var y = Tensor.randint(0, 5, shape=(100,))
    return TensorDataset(X, y)
```

### Assertion Helpers

```mojo
fn assert_tensors_close(a: Tensor, b: Tensor, rtol: Float64 = 1e-5, atol: Float64 = 1e-8):
    """Assert two tensors are approximately equal."""
    var diff = (a - b).abs()
    var tolerance = atol + rtol * b.abs()
    assert_true((diff <= tolerance).all())

fn assert_shape(tensor: Tensor, expected_shape: List[Int]):
    """Assert tensor has expected shape."""
    assert_equal(tensor.shape.size(), expected_shape.size())
    for i in range(expected_shape.size()):
        assert_equal(tensor.shape[i], expected_shape[i])

fn assert_no_nans(tensor: Tensor):
    """Assert tensor contains no NaN values."""
    assert_true(not tensor.isnan().any())
```

## Running Tests

### Command Line

```bash
# Run all tests
pixi run pytest tests/

# Run specific test file
pixi run pytest tests/shared/core/test_layers.mojo

# Run tests matching pattern
pixi run pytest tests/ -k "test_linear"

# Run with verbose output
pixi run pytest tests/ -v

# Run with coverage
pixi run pytest tests/ --cov=shared --cov-report=html

# Run only fast tests (< 1 second)
pixi run pytest tests/ -m "not slow"
```

### CI Integration

Tests run automatically on every pull request:

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install Pixi
        run: curl -fsSL https://pixi.sh/install.sh | bash
      - name: Run tests
        run: pixi run pytest tests/
```

## Test Markers

Categorize tests with markers:

```mojo
@pytest.mark.fast
fn test_relu_forward():
    """Fast unit test."""
    pass

@pytest.mark.slow
fn test_full_training():
    """Slow integration test."""
    pass

@pytest.mark.gpu
fn test_cuda_operations():
    """Test requiring GPU."""
    pass

@pytest.mark.skip(reason="Not implemented yet")
fn test_future_feature():
    """Placeholder for future test."""
    pass
```

Run specific markers:

```bash
# Run only fast tests
pixi run pytest tests/ -m fast

# Skip slow tests
pixi run pytest tests/ -m "not slow"

# Run GPU tests only if GPU available
pixi run pytest tests/ -m gpu
```

## Test Coverage

### Measuring Coverage

```bash
# Generate coverage report
pixi run pytest tests/ --cov=shared --cov=papers

# HTML report
pixi run pytest tests/ --cov=shared --cov-report=html
open htmlcov/index.html
```

### Coverage Requirements

- **Shared Library**: Minimum 90% coverage
- **Paper Implementations**: Minimum 80% coverage
- **Critical Paths**: 100% coverage (training loops, gradient computation)

### Improving Coverage

```bash
# Find uncovered lines
pixi run pytest tests/ --cov=shared --cov-report=term-missing

# Generate coverage report
pixi run coverage html
```

## Benchmarking Tests

Performance benchmarks verify speed and memory usage:

```mojo
from shared.utils import Profiler, benchmark

@benchmark(iterations=100)
fn bench_linear_forward():
    """Benchmark Linear layer forward pass."""
    var layer = Linear(1000, 1000)
    var input = Tensor.randn(128, 1000)

    @timer
    var output = layer.forward(input)

    return output

fn test_conv2d_performance():
    """Verify Conv2D performance meets requirements."""
    var conv = Conv2D(64, 64, kernel_size=3)
    var input = Tensor.randn(32, 64, 224, 224)

    var profiler = Profiler()
    with profiler.section("forward"):
        var output = conv.forward(input)

    # Should complete in < 100ms on CPU
    assert_true(profiler.get_time("forward") < 0.1)
```

## Common Testing Patterns

### Testing Error Handling

```mojo
fn test_invalid_input_raises():
    """Test that invalid input raises appropriate error."""
    var layer = Linear(10, 5)
    var invalid_input = Tensor.randn(8, 15)  # Wrong size

    with assert_raises[ValueError]():
        _ = layer.forward(invalid_input)
```

### Testing Randomness

```mojo
fn test_dropout_reproducibility():
    """Test dropout gives same results with same seed."""
    var dropout = Dropout(p=0.5)
    var input = Tensor.randn(10, 10)

    set_seed(42)
    var output1 = dropout.forward(input)

    set_seed(42)
    var output2 = dropout.forward(input)

    assert_tensors_close(output1, output2)
```

### Testing Model Saving/Loading

```mojo
fn test_model_save_load():
    """Test model can be saved and loaded."""
    var model = LeNet5()
    var input = Tensor.randn(1, 1, 28, 28)

    # Get output before saving
    var output_before = model.forward(input)

    # Save and load
    save_model(model, "test_model.mojo")
    var loaded_model = load_model[LeNet5]("test_model.mojo")

    # Output should be identical
    var output_after = loaded_model.forward(input)
    assert_tensors_close(output_before, output_after)
```

## Debugging Failed Tests

### Using Print Debugging

```mojo
fn test_model_output():
    var model = MyModel()
    var input = Tensor.randn(1, 10)
    var output = model.forward(input)

    print("Input shape:", input.shape)
    print("Output shape:", output.shape)
    print("Output values:", output)

    assert_equal(output.shape, [1, 5])
```

### Using Breakpoints

```mojo
fn test_training_step():
    var model = MyModel()
    var optimizer = SGD(lr=0.01)

    # Set breakpoint
    breakpoint()

    var output = model.forward(input)
```

### Isolating Failures

```bash
# Run single test
pixi run pytest tests/test_model.mojo::test_forward_pass -v

# Run with print output
pixi run pytest tests/test_model.mojo -s

# Run with debugger on failure
pixi run pytest tests/test_model.mojo --pdb
```

## Best Practices

### DO

- ✅ Write tests before implementation (TDD)
- ✅ Test one thing per test function
- ✅ Use descriptive test names
- ✅ Test edge cases and error conditions
- ✅ Keep tests fast and isolated
- ✅ Use fixtures for common setup
- ✅ Assert specific values, not just "no error"

### DON'T

- ❌ Write tests that depend on each other
- ❌ Test implementation details (test behavior)
- ❌ Skip testing error paths
- ❌ Use random data without setting seed
- ❌ Write tests that take > 1 second (use markers)
- ❌ Ignore failing tests

## Next Steps

- **[Paper Implementation Guide](paper-implementation.md)** - Apply testing to papers
- **[Workflow](workflow.md)** - TDD in the development workflow
- **[Performance Guide](../advanced/performance.md)** - Performance testing
- **[CI/CD](../dev/ci-cd.md)** - Continuous integration setup

## Related Documentation

- [Project Structure](project-structure.md) - Test organization
- [Shared Library](shared-library.md) - Components to test
- [Mojo Patterns](mojo-patterns.md) - Mojo-specific testing patterns
