# Issue #445: [Impl] Test Fixtures

## Objective

Implement missing test fixtures and enhance existing ones based on findings from the Test phase (#444), focusing on fixtures that provide consistent test environments and reduce test code duplication.

## Deliverables

- Enhanced or new tensor fixtures (if Tensor available)
- Enhanced or new model fixtures (if models available)
- Enhanced or new dataset fixtures (if needed)
- Configuration fixtures (if needed)
- Improved fixture scoping (if feasible in Mojo)

## Success Criteria

- [ ] All gaps identified in Test phase are addressed
- [ ] Fixtures provide consistent test data
- [ ] Setup and teardown work correctly
- [ ] Fixtures reduce test code duplication measurably
- [ ] All fixtures are well-documented
- [ ] Fixtures handle edge cases correctly

## Current State Analysis

### Existing Fixtures

**Implemented** (conftest.mojo):
- `TestFixtures.deterministic_seed()` - Returns 42
- `TestFixtures.set_seed()` - Sets deterministic random seed

**Placeholders/TODOs** (conftest.mojo, lines 160-181):
```mojo
# TODO(#1538): Add tensor fixture methods when Tensor type is implemented
# @staticmethod
# fn small_tensor() -> Tensor:
#     """Create small 3x3 tensor for unit tests."""
#     pass
#
# @staticmethod
# fn random_tensor(rows: Int, cols: Int) -> Tensor:
#     """Create random tensor with deterministic seed."""
#     pass
```

**Placeholders** (helpers/fixtures.mojo):
```mojo
# TODO: Implement fixture functions once ExTensor is fully functional
# These will include:
# - random_tensor(shape, dtype) -> ExTensor
# - sequential_tensor(shape, dtype) -> ExTensor
# - nan_tensor(shape) -> ExTensor
# - inf_tensor(shape) -> ExTensor
# - ones_like(tensor) -> ExTensor
# - zeros_like(tensor) -> ExTensor
```

**Data Generators** (conftest.mojo, implemented):
- `create_test_vector(size, value)` - Uniform value vector
- `create_test_matrix(rows, cols, value)` - Uniform value matrix
- `create_sequential_vector(size, start)` - Sequential values

### Implementation Decisions

Based on **YAGNI** and **Minimal Changes** principles:

**Decision Matrix**:

| Fixture Type | Needed Now? | Dependencies Ready? | Action |
|-------------|-------------|---------------------|--------|
| Seed fixtures | Yes | Yes | Already implemented ✓ |
| Data generators | Yes | Yes | Already implemented ✓ |
| Tensor fixtures | Maybe | Check ExTensor | Implement if ready |
| Model fixtures | Maybe | Check models | Implement if ready |
| Dataset fixtures | Maybe | Check datasets | Implement if ready |
| Config fixtures | Maybe | Check configs | Implement if needed |

## Implementation Strategy

### Phase 1: Assess Actual Needs (from Test Phase #444)

Review Test phase findings to determine:
1. Are seed fixtures sufficient?
2. Are data generators adequate?
3. Are tensor fixtures actually needed now?
4. Are there gaps in current fixtures?

### Phase 2: Implement High-Priority Fixtures

#### Option A: Tensor Fixtures (if ExTensor ready)

**Check Availability**:
```mojo
// Try importing ExTensor
from extensor import ExTensor  // If this works, implement fixtures
```

**If Available**, implement in `helpers/fixtures.mojo`:
```mojo
"""Test fixtures for ExTensor testing."""

from extensor import ExTensor
from tests.shared.conftest import TestFixtures

fn random_tensor(shape: DynamicVector[Int], dtype: DType = DType.float32) -> ExTensor:
    """Create random tensor with deterministic seed.

    Args:
        shape: Tensor shape.
        dtype: Data type.

    Returns:
        Random ExTensor with deterministic values.
    """
    TestFixtures.set_seed()
    return ExTensor.randn(shape, dtype=dtype)

fn sequential_tensor(shape: DynamicVector[Int], dtype: DType = DType.float32) -> ExTensor:
    """Create tensor with sequential values [0, 1, 2, ...].

    Args:
        shape: Tensor shape.
        dtype: Data type.

    Returns:
        ExTensor with sequential values.
    """
    var numel = 1
    for i in range(len(shape)):
        numel *= shape[i]

    var data = List[Float32](capacity=numel)
    for i in range(numel):
        data.append(Float32(i))

    return ExTensor(data, shape, dtype=dtype)

fn ones_tensor(shape: DynamicVector[Int], dtype: DType = DType.float32) -> ExTensor:
    """Create tensor filled with ones.

    Args:
        shape: Tensor shape.
        dtype: Data type.

    Returns:
        ExTensor filled with 1.0.
    """
    return ExTensor.ones(shape, dtype=dtype)

fn zeros_tensor(shape: DynamicVector[Int], dtype: DType = DType.float32) -> ExTensor:
    """Create tensor filled with zeros.

    Args:
        shape: Tensor shape.
        dtype: Data type.

    Returns:
        ExTensor filled with 0.0.
    """
    return ExTensor.zeros(shape, dtype=dtype)

fn nan_tensor(shape: DynamicVector[Int]) -> ExTensor:
    """Create tensor filled with NaN values.

    Args:
        shape: Tensor shape.

    Returns:
        ExTensor filled with NaN.
    """
    var numel = 1
    for i in range(len(shape)):
        numel *= shape[i]

    var nan_val = Float64(0.0) / Float64(0.0)
    var data = List[Float64](capacity=numel)
    for _ in range(numel):
        data.append(nan_val)

    return ExTensor(data, shape, dtype=DType.float64)

fn inf_tensor(shape: DynamicVector[Int]) -> ExTensor:
    """Create tensor filled with infinity values.

    Args:
        shape: Tensor shape.

    Returns:
        ExTensor filled with +inf.
    """
    var numel = 1
    for i in range(len(shape)):
        numel *= shape[i]

    var inf_val = Float64(1.0) / Float64(0.0)
    var data = List[Float64](capacity=numel)
    for _ in range(numel):
        data.append(inf_val)

    return ExTensor(data, shape, dtype=DType.float64)
```

#### Option B: Model Fixtures (if models ready)

**If model classes are available**, add to conftest.mojo:
```mojo
@staticmethod
fn simple_linear_model() -> Linear:
    """Create simple Linear layer with known weights.

    Returns:
        Linear layer with deterministic weights.
    """
    Self.set_seed()
    return Linear(input_dim=10, output_dim=5)

@staticmethod
fn small_mlp() -> Sequential:
    """Create small multi-layer perceptron.

    Returns:
        2-layer MLP with deterministic weights.
    """
    Self.set_seed()
    return Sequential(
        Linear(10, 20),
        ReLU(),
        Linear(20, 5)
    )
```

#### Option C: Dataset Fixtures (if needed)

**If dataset classes are available**:
```mojo
@staticmethod
fn toy_classification_dataset() -> Dataset:
    """Create toy classification dataset (XOR problem).

    Returns:
        Dataset with 100 samples, 2 features, 2 classes.
    """
    Self.set_seed()
    # Implement XOR dataset generation
    return XORDataset(n_samples=100)

@staticmethod
fn toy_regression_dataset() -> Dataset:
    """Create toy regression dataset (linear relationship).

    Returns:
        Dataset with 100 samples, known linear relationship.
    """
    Self.set_seed()
    # Implement linear regression dataset
    return LinearDataset(n_samples=100, slope=2.0, intercept=1.0)
```

#### Option D: Configuration Fixtures (if needed)

**If configuration management is in place**:
```mojo
@staticmethod
fn default_training_config() -> TrainingConfig:
    """Create default training configuration.

    Returns:
        TrainingConfig with standard hyperparameters.
    """
    return TrainingConfig(
        learning_rate=0.001,
        batch_size=32,
        epochs=10,
        optimizer="adam"
    )

@staticmethod
fn test_optimizer_config() -> OptimizerConfig:
    """Create test optimizer configuration.

    Returns:
        OptimizerConfig for testing.
    """
    return OptimizerConfig(
        learning_rate=0.01,
        momentum=0.9,
        weight_decay=1e-4
    )
```

### Phase 3: Enhance Existing Fixtures (if gaps found)

**Potential Enhancements**:

**1. Additional Data Generators**:
```mojo
fn create_random_vector(size: Int, min_val: Float32 = 0.0, max_val: Float32 = 1.0) -> List[Float32]:
    """Create random vector in range [min_val, max_val].

    Args:
        size: Vector size.
        min_val: Minimum value.
        max_val: Maximum value.

    Returns:
        List of random Float32 values.
    """
    TestFixtures.set_seed()
    var vec = List[Float32](capacity=size)
    var range_val = max_val - min_val

    for _ in range(size):
        vec.append(min_val + randn() * range_val)

    return vec

fn create_sparse_vector(size: Int, sparsity: Float32 = 0.9) -> List[Float32]:
    """Create sparse vector (mostly zeros).

    Args:
        size: Vector size.
        sparsity: Fraction of zeros (0.0-1.0).

    Returns:
        List with specified sparsity.
    """
    TestFixtures.set_seed()
    var vec = List[Float32](capacity=size)

    for _ in range(size):
        if randn() < sparsity:
            vec.append(0.0)
        else:
            vec.append(randn())

    return vec
```

**2. Fixture Cleanup Helpers**:
```mojo
fn reset_global_state():
    """Reset global test state between tests.

    Call at start of each test to ensure clean state.
    """
    TestFixtures.set_seed()
    # Clear any global caches
    # Reset any singletons
    # etc.
```

### Phase 4: Documentation

Update all new fixtures with:
- Clear docstrings
- Usage examples
- Parameter descriptions
- Return value descriptions
- Notes on determinism and seeding

## Decision Process

**Step 1**: Review Test phase (#444) findings
- What fixtures are actually used in tests?
- What gaps were identified?
- What pain points exist?

**Step 2**: Check dependencies
- Is ExTensor available? Check imports
- Are model classes available? Check shared/core/
- Are dataset classes available? Check shared/data/
- Are config classes available? Check shared/utils/

**Step 3**: Prioritize implementations
- **Must have**: Fixtures actively used in tests
- **Nice to have**: Fixtures that would reduce duplication
- **Skip**: Fixtures with unavailable dependencies

**Step 4**: Implement based on priorities
- Start with most impactful
- Follow minimal changes principle
- Document as you go

## References

- **Source Plan**: [notes/plan/02-shared-library/04-testing/01-test-framework/03-test-fixtures/plan.md](../../../plan/02-shared-library/04-testing/01-test-framework/03-test-fixtures/plan.md)
- **Related Issues**:
  - Issue #443: [Plan] Test Fixtures
  - Issue #444: [Test] Test Fixtures (source of requirements)
  - Issue #446: [Package] Test Fixtures
  - Issue #447: [Cleanup] Test Fixtures
- **Existing Code**:
  - `/tests/shared/conftest.mojo` (TestFixtures struct)
  - `/tests/helpers/fixtures.mojo` (placeholder)

## Implementation Notes

### Findings from Test Phase

(To be filled based on Issue #444 results)

**Fixtures Actually Needed**:
- TBD

**Dependencies Available**:
- ExTensor: TBD
- Models: TBD
- Datasets: TBD
- Configs: TBD

**Gaps Identified**:
- TBD

### Implementation Decisions

**Decision Log**:

| Date | Fixture Type | Decision | Rationale |
|------|-------------|----------|-----------|
| TBD | Tensor fixtures | Implement/Skip | TBD |
| TBD | Model fixtures | Implement/Skip | TBD |
| TBD | Dataset fixtures | Implement/Skip | TBD |
| TBD | Config fixtures | Implement/Skip | TBD |
| TBD | Enhanced generators | Implement/Skip | TBD |

### Code Changes

**Files Modified**:
- `/tests/shared/conftest.mojo` - Add new fixtures if needed
- `/tests/helpers/fixtures.mojo` - Implement or remove placeholders

**Files Created**:
- TBD based on actual needs

### Testing Validation

After implementation:
1. Run fixture tests from Issue #444
2. Verify all new fixtures pass their tests
3. Run full test suite to ensure fixtures work in real tests
4. Verify fixtures reduce code duplication as intended
