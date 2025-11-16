# Testing Utilities

Reusable testing components and utilities for ML implementations.

## Available Tools

### 1. Data Generators (`data_generators.mojo`)

Generate synthetic test data for ML model testing.

**Language**: Mojo (required for performance-critical data generation, SIMD optimization)

**Usage**:

```mojo
from tools.test_utils.data_generators import TensorGenerator

fn test_example():
    let generator = TensorGenerator()

    # Generate random tensor
    let random_data = generator.generate_random(
        TensorShape(32, 3, 28, 28),
        min_val=0.0,
        max_val=1.0
    )

    # Generate zeros
    let zeros = generator.generate_zeros(TensorShape(10, 10))

    # Generate ones
    let ones = generator.generate_ones(TensorShape(5, 5))

    # Generate batch
    let batch = generator.generate_batch(
        32,  # batch_size
        3,   # channels
        28,  # height
        28   # width
    )
```

**Features**:

- Random tensors with uniform distribution
- Zero and one-filled tensors
- Batch generation with configurable dimensions
- Type-safe with compile-time checks

### 2. Test Fixtures (`fixtures.mojo`)

Common test models for infrastructure testing.

**Language**: Mojo (required for type safety, model compatibility)

**Usage**:

```mojo
from tools.test_utils.fixtures import SimpleCNN, LinearModel, create_test_model

fn test_model():
    # Create simple CNN
    let cnn = create_test_model("cnn")

    # Create linear model
    let linear = LinearModel(in_features=784, out_features=10)

    # Use in tests
    let input = generator.generate_batch(32, 1, 28, 28)
    let output = cnn.forward(input)
```

**Available Fixtures**:

- `SimpleCNN`: Minimal 2-layer CNN for testing
- `LinearModel`: Simple fully-connected layer
- Factory functions for easy creation

## Design Principles

- **Lightweight**: Minimal models for fast testing
- **Type-safe**: Compile-time checks via Mojo
- **Realistic**: Representative of actual use cases
- **Composable**: Work with other testing tools

## Language Justification

Per [ADR-001](../../notes/review/adr/ADR-001-language-selection-tooling.md):

- **Why Mojo**: Performance-critical data generation, type safety, memory efficiency
- **Benefits**: SIMD optimization, zero Python overhead, compile-time validation
- **Required**: ML/AI implementation (not automation)

## Future Enhancements

- More sophisticated data distributions (Gaussian, etc.)
- Coverage analysis integration
- Performance profiling utilities
- Dataset fixtures (MNIST, CIFAR)

## References

- [Issue #67](https://github.com/mvillmow/ml-odyssey/issues/67): Tools planning
- [Issue #69](https://github.com/mvillmow/ml-odyssey/issues/69): Tools implementation
- [ADR-001](../../notes/review/adr/ADR-001-language-selection-tooling.md): Language strategy
- [Mojo Best Practices](../../.claude/agents/mojo-language-review-specialist.md)
