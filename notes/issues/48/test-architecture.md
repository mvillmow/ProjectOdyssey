# Test Architecture for Shared Library

## Overview

This document describes the comprehensive test architecture for the ML Odyssey shared library. The test suite
follows Test-Driven Development (TDD) principles and establishes tests BEFORE implementation to guide development
in Issue #49.

## Design Philosophy

### Quality Over Quantity

- Focus on critical path coverage (≥90% target)
- Test behavior, not implementation details
- Each test must add value
- Skip trivial tests (simple getters, obvious constructors)

### TDD Approach

Tests are written FIRST to:

1. Define clear API contracts
2. Validate expected behavior
3. Guide implementation decisions
4. Catch regressions early

### Real Implementations Over Mocks

- Use real implementations whenever possible
- Create simple, concrete test data
- Minimal mocking (only for complex external dependencies)
- Avoid elaborate mock frameworks

## Directory Structure

```text
tests/
└── shared/                          # All shared library tests
    ├── __init__.mojo                # Test package root
    ├── README.md                    # Test suite documentation
    ├── conftest.mojo                # Shared test fixtures and utilities
    ├── core/                        # Core module tests
    │   ├── __init__.mojo
    │   ├── test_layers.mojo         # Layer implementations
    │   ├── test_tensors.mojo        # Tensor operations
    │   ├── test_activations.mojo    # Activation functions
    │   ├── test_initializers.mojo   # Parameter initialization
    │   └── test_module.mojo         # Module base class
    ├── training/                    # Training module tests
    │   ├── __init__.mojo
    │   ├── test_optimizers.mojo     # Optimizer implementations
    │   ├── test_schedulers.mojo     # LR schedulers
    │   ├── test_metrics.mojo        # Training metrics
    │   ├── test_callbacks.mojo      # Training callbacks
    │   └── test_loops.mojo          # Training loops
    ├── data/                        # Data module tests
    │   ├── __init__.mojo
    │   ├── test_datasets.mojo       # Dataset implementations
    │   ├── test_loaders.mojo        # Data loaders
    │   └── test_transforms.mojo     # Data transforms
    ├── utils/                       # Utils module tests
    │   ├── __init__.mojo
    │   ├── test_logging.mojo        # Logging utilities
    │   ├── test_visualization.mojo  # Visualization tools
    │   └── test_config.mojo         # Configuration management
    ├── integration/                 # Integration tests
    │   ├── __init__.mojo
    │   ├── test_training_workflow.mojo
    │   ├── test_data_pipeline.mojo
    │   └── test_end_to_end.mojo
    └── benchmarks/                  # Performance benchmarks
        ├── __init__.mojo
        ├── bench_optimizers.mojo
        ├── bench_layers.mojo
        └── bench_data_loading.mojo
```

## Test Categories

### 1. Unit Tests (Critical - MUST Have)

**Purpose**: Test individual functions and components in isolation

**Coverage**:

- Core functionality (main features and use cases)
- Security-sensitive code (validation, boundaries)
- Public API contracts
- Error handling for critical paths
- Integration points between modules

**Skip**:

- Trivial getters/setters
- Simple constructors with no logic
- Private implementation details

**Example**: `test_sgd_basic_update()` - validates SGD parameter update formula

### 2. Integration Tests (Important - SHOULD Have)

**Purpose**: Test cross-module workflows and component interactions

**Coverage**:

- End-to-end training workflows
- Data loading pipelines
- Model construction and forward passes
- Gradient computation flows

**Example**: `test_training_workflow()` - complete training loop with validation

### 3. Performance Benchmarks (Important - SHOULD Have)

**Purpose**: Establish baseline performance metrics and prevent regressions

**Coverage**:

- Optimizer update throughput
- Layer forward/backward pass speed
- Data loading performance
- Memory allocation patterns

**Targets**:

- Within 2x of PyTorch for optimizers
- SIMD-vectorized operations
- < 5 minutes total CI test time

**Example**: `bench_sgd_update_speed()` - measure parameter updates per second

### 4. Property-Based Tests (Nice to Have)

**Purpose**: Validate mathematical properties and invariants

**Coverage**:

- Optimizer convergence properties
- Layer gradient correctness
- Tensor operation commutativity
- Data transformation invertibility

**Example**: `test_optimizer_property_decreasing_loss()` - loss should decrease on convex functions

### 5. Numerical Accuracy Tests (Critical for ML - MUST Have)

**Purpose**: Validate correctness against reference implementations

**Coverage**:

- Compare optimizer updates to PyTorch
- Validate layer outputs against known values
- Check gradient computations
- Verify metric calculations

**Example**: `test_sgd_matches_pytorch()` - parameter updates match PyTorch exactly

## Test Naming Conventions

### Test Functions

```mojo
# Pattern: test_<component>_<behavior>
fn test_sgd_basic_update():
    """Test SGD performs basic parameter update."""
    pass

fn test_sgd_momentum_accumulation():
    """Test SGD accumulates momentum correctly."""
    pass

fn test_conv2d_output_shape():
    """Test Conv2D computes correct output shape."""
    pass
```

### Property Tests

```mojo
# Pattern: test_<component>_property_<property>
fn test_optimizer_property_decreasing_loss():
    """Property: Optimizer should decrease loss on convex function."""
    pass

fn test_layer_property_gradient_shape():
    """Property: Layer gradient shape matches parameter shape."""
    pass
```

### Benchmark Functions

```mojo
# Pattern: bench_<component>_<metric>
fn bench_sgd_update_speed():
    """Benchmark SGD parameter update throughput."""
    pass

fn bench_conv2d_forward_throughput():
    """Benchmark Conv2D forward pass FLOPS."""
    pass
```

## Test Data Strategy

### Fixtures Location

```text
tests/fixtures/
├── images/              # Sample images for data tests
│   ├── mnist_sample/    # 10 MNIST images
│   └── synthetic/       # Generated test images
├── tensors/             # Pre-computed tensor data
│   ├── reference_outputs.json
│   └── gradient_tests.json
├── models/              # Small model weights
│   └── simple_linear.mojo
└── reference/           # Reference implementation outputs
    ├── pytorch_sgd.json
    └── pytorch_conv2d.json
```

### Test Data Principles

1. **Small and Fast**: Tests should run in < 5 minutes total
2. **Deterministic**: Use fixed random seeds
3. **Realistic**: Representative of actual use cases
4. **Known Ground Truth**: Use analytically solvable problems

### Example Fixtures

```mojo
struct TestFixtures:
    """Shared test fixtures and utilities."""

    @staticmethod
    fn small_tensor() -> Tensor:
        """Create small tensor for unit tests (3x3)."""
        var data = List[Float32](
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0
        )
        return Tensor(data, Shape(3, 3))

    @staticmethod
    fn simple_linear_model() -> Linear:
        """Create simple Linear layer with known weights."""
        var layer = Linear(in_features=10, out_features=5)
        # Initialize with known values for testing
        layer.weights.fill(0.1)
        layer.bias.fill(0.0)
        return layer

    @staticmethod
    fn synthetic_dataset(n_samples: Int = 100) -> TensorDataset:
        """Create synthetic dataset for testing."""
        var x = Tensor.randn(n_samples, 10, seed=42)
        var y = Tensor.randint(0, 10, n_samples, seed=42)
        return TensorDataset(x, y)
```

## Coverage Requirements

### Targets

- **Line Coverage**: ≥90% of all lines executed
- **Branch Coverage**: ≥85% of all branches tested
- **Function Coverage**: 100% of public functions tested
- **Critical Path Coverage**: 100% of core functionality

### Exemptions

- Debug-only code
- Unreachable error paths
- Platform-specific code not available on test machine
- Deprecated code marked for removal

### Coverage Tools

```bash
# Generate coverage report
mojo test --coverage tests/shared/

# Check coverage threshold
python scripts/check_coverage.py --threshold 90 --path shared/

# Generate HTML report
mojo test --coverage --html tests/shared/
```

## CI Integration

### GitHub Actions Workflow

**File**: `.github/workflows/test-shared.yml`

```yaml
name: Test Shared Library

on:
  pull_request:
    paths:
      - 'shared/**'
      - 'tests/shared/**'
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 10

    steps:
      - uses: actions/checkout@v4

      - name: Setup Mojo
        uses: modular/setup-mojo@v1

      - name: Run unit tests
        run: |
          mojo test tests/shared/core/
          mojo test tests/shared/training/
          mojo test tests/shared/data/
          mojo test tests/shared/utils/

      - name: Run integration tests
        run: mojo test tests/shared/integration/

      - name: Generate coverage
        run: mojo test --coverage tests/shared/

      - name: Check coverage threshold
        run: python scripts/check_coverage.py --threshold 90

      - name: Upload coverage report
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml
          flags: shared-library

      - name: Run benchmarks (on main only)
        if: github.ref == 'refs/heads/main'
        run: mojo test tests/shared/benchmarks/
```

### Test Execution Requirements

- ✅ Tests must run automatically on PR creation
- ✅ Tests must pass before merge is allowed
- ✅ Tests must complete in < 10 minutes (timeout)
- ✅ Tests must be deterministic (no flaky tests)
- ✅ Coverage must meet ≥90% threshold
- ❌ Do NOT add tests that require manual setup
- ❌ Do NOT add tests that can't run in CI

## Mojo Testing Best Practices

### Use `fn` for Test Functions

```mojo
# GOOD: Use fn for type safety and performance
fn test_optimizer_convergence():
    """Test optimizer converges on simple function."""
    var optimizer = SGD(lr=0.01)
    var loss = run_optimizer(optimizer)
    assert_true(loss < initial_loss)

# AVOID: Using def for tests (less type safety)
def test_something():
    pass
```

### Use Structs for Test Fixtures

```mojo
struct OptimizerTestFixture:
    """Reusable optimizer test setup."""
    var model: SimpleModel
    var optimizer: SGD
    var data: Tensor
    var targets: Tensor

    fn __init__(inout self):
        self.model = SimpleModel()
        self.optimizer = SGD(lr=0.01)
        self.data = Tensor.randn(32, 10, seed=42)
        self.targets = Tensor.randint(0, 10, 32, seed=42)

    fn reset(inout self):
        """Reset fixture to initial state."""
        self.model = SimpleModel()
        self.optimizer.reset_state()
```

### Use SIMD for Performance Tests

```mojo
fn bench_vectorized_operation():
    """Benchmark SIMD-optimized operation."""
    alias simd_width = simdwidthof[DType.float32]()
    let n = 1_000_000

    var data = Tensor.randn(n, seed=42)
    var result = Tensor(n)

    var start = time.now()
    for i in range(0, n, simd_width):
        let vec = data.load[simd_width](i)
        result.store[simd_width](i, vec * 2.0)
    var elapsed = time.now() - start

    # Target: > 1 GB/s throughput
    let throughput = (n * sizeof[Float32]()) / elapsed
    assert_true(throughput > 1_000_000_000)
```

### Memory Safety Testing

```mojo
fn test_tensor_ownership():
    """Test tensor ownership transfer."""
    var t1 = Tensor.randn(10, 10)
    let original_ptr = t1.data

    # Transfer ownership
    var t2 = t1^  # Move semantics
    assert_equal(t2.data, original_ptr)
    # t1 is now invalid (moved)

fn test_tensor_borrowing():
    """Test tensor borrowing."""
    var t1 = Tensor.randn(10, 10)
    let sum1 = compute_sum(t1)  # Borrowed
    let sum2 = compute_sum(t1)  # Still valid
    assert_equal(sum1, sum2)
```

## Test Modules Overview

### Core Module Tests

**File**: `tests/shared/core/test_layers.mojo`

**Critical Tests** (MUST have):

- `test_linear_forward()` - Linear layer forward pass
- `test_linear_backward()` - Linear layer gradient computation
- `test_conv2d_output_shape()` - Conv2D shape computation
- `test_conv2d_forward()` - Conv2D forward pass
- `test_relu_activation()` - ReLU zeros negative values
- `test_maxpool_forward()` - MaxPool downsampling

**Important Tests** (SHOULD have):

- `test_linear_bias()` - Bias addition
- `test_conv2d_padding()` - Padding modes
- `test_conv2d_stride()` - Strided convolution

**File**: `tests/shared/core/test_tensors.mojo`

**Critical Tests**:

- `test_tensor_creation()` - Tensor initialization
- `test_tensor_indexing()` - Element access
- `test_tensor_broadcasting()` - Broadcasting rules
- `test_tensor_reshape()` - Reshaping operations
- `test_tensor_slicing()` - Slice operations

**File**: `tests/shared/core/test_activations.mojo`

**Critical Tests**:

- `test_relu()` - ReLU correctness
- `test_sigmoid()` - Sigmoid range [0, 1]
- `test_tanh()` - Tanh range [-1, 1]
- `test_softmax()` - Softmax sums to 1

### Training Module Tests

**File**: `tests/shared/training/test_optimizers.mojo`

**Critical Tests** (MUST have):

- `test_sgd_parameter_update()` - Basic SGD update
- `test_sgd_momentum()` - Momentum accumulation
- `test_adam_bias_correction()` - Adam bias correction
- `test_adam_parameter_update()` - Adam update formula
- `test_optimizer_zero_grad()` - Gradient clearing

**Numerical Accuracy Tests**:

- `test_sgd_matches_pytorch()` - Compare to PyTorch SGD
- `test_adam_matches_pytorch()` - Compare to PyTorch Adam

**Property Tests**:

- `test_optimizer_property_decreasing_loss()` - Loss decreases on convex function

**File**: `tests/shared/training/test_schedulers.mojo`

**Critical Tests**:

- `test_step_lr_schedule()` - StepLR decreases at steps
- `test_cosine_annealing()` - Cosine curve shape
- `test_exponential_decay()` - Exponential formula

**File**: `tests/shared/training/test_metrics.mojo`

**Critical Tests**:

- `test_accuracy_computation()` - Accuracy calculation
- `test_loss_tracker_averaging()` - Loss averaging
- `test_confusion_matrix()` - Confusion matrix construction

### Integration Tests

**File**: `tests/shared/integration/test_training_workflow.mojo`

**Critical Tests**:

- `test_basic_training_loop()` - Complete training cycle
- `test_training_with_validation()` - Train + validation
- `test_training_with_callbacks()` - Callback invocation

**File**: `tests/shared/integration/test_data_pipeline.mojo`

**Critical Tests**:

- `test_dataloader_batching()` - Batch creation
- `test_dataloader_shuffling()` - Data shuffling
- `test_transform_composition()` - Transform pipeline

## Performance Benchmarks

### Benchmark Structure

```mojo
struct BenchmarkResult:
    var name: String
    var duration_ms: Float64
    var throughput: Float64
    var memory_mb: Float64

fn bench_sgd_update_speed() -> BenchmarkResult:
    """Benchmark SGD parameter update throughput."""
    let param_counts = List[Int](1_000_000, 10_000_000, 100_000_000)
    var results = List[BenchmarkResult]()

    for n in param_counts:
        var params = Tensor.randn(n)
        var grads = Tensor.randn(n)
        var optimizer = SGD(lr=0.01)

        let start = time.now()
        let n_iters = 100
        for _ in range(n_iters):
            optimizer.step(params, grads)
        let elapsed = (time.now() - start) / n_iters

        let throughput = n / elapsed  # params/second
        results.append(BenchmarkResult(
            name="SGD-" + str(n),
            duration_ms=elapsed * 1000,
            throughput=throughput,
            memory_mb=n * sizeof[Float32]() / 1_000_000
        ))

    return results
```

### Performance Targets

**Optimizers**:

- SGD update: Within 2x of PyTorch
- Adam update: Within 2x of PyTorch
- Memory overhead: < 10% beyond theoretical minimum

**Layers**:

- Conv2D forward: ≥ 1 TFLOPS on test hardware
- Linear forward: Fully vectorized (SIMD)
- Activation functions: < 1ns per element

**Data Loading**:

- Batch creation: < 10ms for 32-batch
- Transform application: ≥ 100 images/second
- Shuffling: < 100ms for 10K samples

## Coordination with Implementation

### TDD Workflow

1. **Test Specialist** (Issue #48 - current):
   - Design test architecture
   - Write test specifications
   - Create test files with assertions
   - Define expected API contracts

2. **Implementation Specialist** (Issue #49):
   - Read test specifications
   - Implement to make tests pass
   - Follow API contracts from tests
   - Add implementation details

3. **Test Specialist** (Issue #48 - validation):
   - Run tests against implementation
   - Verify coverage meets ≥90%
   - Add missing edge case tests
   - Refine benchmarks

### API Contract Definition

Tests define the expected API:

```mojo
# Test defines the contract
fn test_sgd_basic_update():
    """Test SGD performs basic parameter update.

    Contract:
    - SGD(lr=Float32) constructor
    - step(inout params: Tensor, grads: Tensor) method
    - Parameters updated in-place
    - Formula: params = params - lr * grads
    """
    var params = Tensor(List[Float32](1.0, 2.0, 3.0), Shape(3))
    var grads = Tensor(List[Float32](0.1, 0.2, 0.3), Shape(3))
    var optimizer = SGD(lr=0.1)

    optimizer.step(params, grads)

    # Expected: [1.0 - 0.1*0.1, 2.0 - 0.1*0.2, 3.0 - 0.1*0.3]
    assert_almost_equal(params[0], 0.99, tolerance=1e-6)
    assert_almost_equal(params[1], 1.98, tolerance=1e-6)
    assert_almost_equal(params[2], 2.97, tolerance=1e-6)
```

## Test Utilities

### Assertion Functions

```mojo
fn assert_true(condition: Bool, message: String = "Assertion failed"):
    """Assert condition is true."""
    if not condition:
        raise AssertionError(message)

fn assert_equal[T: Comparable](a: T, b: T, message: String = ""):
    """Assert exact equality."""
    if a != b:
        raise AssertionError(message or str(a) + " != " + str(b))

fn assert_almost_equal(a: Float32, b: Float32, tolerance: Float32 = 1e-6):
    """Assert floating point near-equality."""
    if abs(a - b) > tolerance:
        raise AssertionError(str(a) + " !≈ " + str(b))

fn assert_tensor_equal(a: Tensor, b: Tensor, tolerance: Float32 = 1e-6):
    """Assert tensor element-wise near-equality."""
    assert_equal(a.shape, b.shape)
    for i in range(a.size()):
        assert_almost_equal(a[i], b[i], tolerance)

fn assert_shape_equal(tensor: Tensor, expected_shape: Shape):
    """Assert tensor has expected shape."""
    assert_equal(tensor.shape, expected_shape)
```

## Next Steps

1. Create test directory structure
2. Implement test files with comprehensive test cases
3. Add test fixtures and utilities
4. Configure CI integration
5. Document test execution procedures
6. Create coverage reporting scripts

## Success Criteria

- [ ] Test architecture documented (this file)
- [ ] ≥90% code coverage achievable
- [ ] All critical tests specified
- [ ] Integration tests defined
- [ ] Performance benchmarks designed
- [ ] CI integration planned
- [ ] Test utilities specified
- [ ] TDD workflow with Issue #49 coordinated

---

**Document Status**: ✅ Complete - Test architecture fully specified

**Next Action**: Create test files in `tests/shared/`
