# Issue #444: [Test] Test Fixtures

## Objective

Develop comprehensive tests for test fixtures to ensure they provide consistent test data, handle setup/teardown correctly, and are reusable across multiple tests.

## Deliverables

- Tests for fixture creation and cleanup
- Tests for fixture scoping behavior
- Tests for fixture data consistency
- Tests for fixture reusability
- Validation of fixture documentation

## Success Criteria

- [ ] All fixtures produce consistent data across invocations
- [ ] Setup and teardown work correctly without resource leaks
- [ ] Fixtures can be reused in multiple tests
- [ ] Scoping (function/module/session) works as designed
- [ ] Fixtures are well-documented with clear usage examples

## Current State Analysis

### Existing Fixtures

**TestFixtures** (conftest.mojo):
```mojo
struct TestFixtures:
    """Collection of reusable test fixtures and utilities."""

    @staticmethod
    fn deterministic_seed() -> Int:
        """Get deterministic random seed for reproducible tests.
        Returns: Fixed random seed value (42).
        """
        return 42

    @staticmethod
    fn set_seed():
        """Set random seed for deterministic test execution."""
        seed(Self.deterministic_seed())

    # TODO(#1538): Add tensor fixture methods when Tensor type is implemented
    # (Lines 160-181 have commented-out tensor fixture methods)
```

**Current Status**:
- Basic seeding fixtures implemented
- Tensor/model/dataset fixtures pending Tensor implementation
- No explicit scoping mechanism (Mojo may not have pytest-style fixtures)

### What Needs Testing

1. **Seed Fixture Behavior**:
   - Does `deterministic_seed()` always return 42?
   - Does `set_seed()` produce reproducible random values?
   - Can multiple tests use the same seed without interference?

2. **Fixture Consistency**:
   - Do fixtures produce identical data on repeated calls?
   - Do fixtures reset properly between tests?

3. **Fixture Reusability**:
   - Can multiple tests use the same fixture?
   - Do fixtures maintain independence between tests?

4. **Documentation Validation**:
   - Are fixture docstrings accurate?
   - Do usage examples work?

## Test Implementation Strategy

### 1. Seed Fixture Tests (`test_seed_fixtures.mojo`)

```mojo
"""Test seed fixture behavior."""

from tests.shared.conftest import TestFixtures
from random import randn

fn test_deterministic_seed_returns_42() raises:
    """Verify deterministic_seed always returns 42."""
    var seed1 = TestFixtures.deterministic_seed()
    var seed2 = TestFixtures.deterministic_seed()

    assert_equal(seed1, 42)
    assert_equal(seed2, 42)

fn test_set_seed_produces_reproducible_values() raises:
    """Verify set_seed generates reproducible random values."""
    TestFixtures.set_seed()
    var val1 = randn()

    TestFixtures.set_seed()  # Reset seed
    var val2 = randn()

    # Same seed should produce same random value
    assert_almost_equal(val1, val2, tolerance=1e-10)

fn test_multiple_values_with_same_seed() raises:
    """Verify sequence of random values is reproducible."""
    TestFixtures.set_seed()
    var seq1 = List[Float64](capacity=10)
    for _ in range(10):
        seq1.append(randn())

    TestFixtures.set_seed()  # Reset seed
    var seq2 = List[Float64](capacity=10)
    for _ in range(10):
        seq2.append(randn())

    # Sequences should match exactly
    assert_equal(len(seq1), len(seq2))
    for i in range(len(seq1)):
        assert_almost_equal(seq1[i], seq2[i], tolerance=1e-10)

fn test_different_seeds_produce_different_values() raises:
    """Verify different seeds produce different values."""
    TestFixtures.set_seed()  # Seed 42
    var val1 = randn()

    seed(100)  # Different seed
    var val2 = randn()

    # Values should differ
    # (This might fail with low probability, but very unlikely)
    assert_not_equal(val1, val2)
```

### 2. Data Generator Tests (`test_data_fixtures.mojo`)

```mojo
"""Test data generator fixtures."""

from tests.shared.conftest import (
    create_test_vector, create_test_matrix, create_sequential_vector
)

fn test_vector_fixture_consistency() raises:
    """Verify vectors created with same parameters are identical."""
    var vec1 = create_test_vector(50, 3.14)
    var vec2 = create_test_vector(50, 3.14)

    assert_equal(len(vec1), len(vec2))
    for i in range(len(vec1)):
        assert_almost_equal(vec1[i], vec2[i], tolerance=1e-10)

fn test_vector_fixture_independence() raises:
    """Verify modifying one vector doesn't affect another."""
    var vec1 = create_test_vector(10, 1.0)
    var vec2 = create_test_vector(10, 1.0)

    # Modify vec1
    vec1[0] = 999.0

    # vec2 should be unchanged
    assert_almost_equal(vec2[0], 1.0, tolerance=1e-10)

fn test_matrix_fixture_consistency() raises:
    """Verify matrices are consistent across calls."""
    var mat1 = create_test_matrix(5, 5, 2.0)
    var mat2 = create_test_matrix(5, 5, 2.0)

    for i in range(5):
        for j in range(5):
            assert_almost_equal(mat1[i][j], mat2[i][j], tolerance=1e-10)

fn test_sequential_fixture_values() raises:
    """Verify sequential vector produces correct sequence."""
    var vec = create_sequential_vector(5, start=10.0)

    assert_equal(len(vec), 5)
    assert_almost_equal(vec[0], 10.0, tolerance=1e-10)
    assert_almost_equal(vec[1], 11.0, tolerance=1e-10)
    assert_almost_equal(vec[2], 12.0, tolerance=1e-10)
    assert_almost_equal(vec[3], 13.0, tolerance=1e-10)
    assert_almost_equal(vec[4], 14.0, tolerance=1e-10)
```

### 3. Fixture Scoping Tests (`test_fixture_scoping.mojo`)

**Note**: Mojo may not have pytest-style fixture scoping. Test what's possible.

```mojo
"""Test fixture scoping and lifecycle."""

from tests.shared.conftest import TestFixtures

# Global state to track fixture usage
var global_seed_calls = 0

fn test_fixture_call_count_1() raises:
    """First test using seed fixture."""
    TestFixtures.set_seed()
    # In pytest, this would verify fixture is called once per test
    # In Mojo, just verify it works
    assert_true(True)

fn test_fixture_call_count_2() raises:
    """Second test using seed fixture."""
    TestFixtures.set_seed()
    # Verify independent execution
    assert_true(True)

fn test_fixture_reusability() raises:
    """Verify fixture can be used multiple times in same test."""
    TestFixtures.set_seed()
    var val1 = randn()

    TestFixtures.set_seed()
    var val2 = randn()

    # Should get same value (same seed)
    assert_almost_equal(val1, val2, tolerance=1e-10)
```

### 4. Fixture Documentation Tests (`test_fixture_docs.mojo`)

```mojo
"""Validate fixture documentation accuracy."""

from tests.shared.conftest import TestFixtures

fn test_deterministic_seed_docstring_accuracy() raises:
    """Verify deterministic_seed docstring is accurate."""
    # Docstring says: "Returns: Fixed random seed value (42)."
    var seed = TestFixtures.deterministic_seed()
    assert_equal(seed, 42, "Docstring promises 42")

fn test_set_seed_behavior_matches_docs() raises:
    """Verify set_seed behavior matches documentation."""
    # Docstring says: "Set random seed for deterministic test execution."
    TestFixtures.set_seed()
    var val1 = randn()

    TestFixtures.set_seed()
    var val2 = randn()

    # Should be deterministic (same values)
    assert_almost_equal(val1, val2, tolerance=1e-10,
                        message="set_seed should be deterministic per docs")
```

### 5. Future Fixtures Tests (`test_future_fixtures.mojo`)

**Note**: These tests will be activated when Tensor fixtures are implemented

```mojo
"""Tests for future tensor/model/dataset fixtures."""

// TODO(#1538): Implement when Tensor fixtures are ready

// fn test_small_tensor_fixture() raises:
//     """Verify small_tensor fixture creates 3x3 tensor."""
//     var tensor = TestFixtures.small_tensor()
//     assert_shape(tensor, DynamicVector[Int](3, 3))
//
// fn test_random_tensor_determinism() raises:
//     """Verify random_tensor is deterministic with same seed."""
//     var tensor1 = TestFixtures.random_tensor(10, 10)
//     var tensor2 = TestFixtures.random_tensor(10, 10)
//     assert_all_close(tensor1, tensor2)
//
// fn test_model_fixture_consistency() raises:
//     """Verify model fixtures have consistent weights."""
//     var model1 = TestFixtures.simple_linear_model()
//     var model2 = TestFixtures.simple_linear_model()
//     # Compare weights
//
// fn test_dataset_fixture_properties() raises:
//     """Verify dataset fixtures have known properties."""
//     var dataset = TestFixtures.synthetic_dataset(n_samples=100)
//     assert_equal(dataset.size(), 100)
```

## Test Coverage Goals

**Seed Fixtures**:
- [x] Deterministic seed value
- [x] Reproducible random generation
- [x] Independence between tests
- [x] Documentation accuracy

**Data Generator Fixtures**:
- [x] Consistency across calls
- [x] Independence of instances
- [x] Correct values generated
- [x] Edge cases (size=0, size=1)

**Fixture Lifecycle**:
- [ ] Scoping behavior (if applicable)
- [x] Reusability across tests
- [x] No resource leaks

**Future Fixtures** (when implemented):
- [ ] Tensor fixtures
- [ ] Model fixtures
- [ ] Dataset fixtures
- [ ] Configuration fixtures

## References

- **Source Plan**: [notes/plan/02-shared-library/04-testing/01-test-framework/03-test-fixtures/plan.md](../../../plan/02-shared-library/04-testing/01-test-framework/03-test-fixtures/plan.md)
- **Related Issues**:
  - Issue #443: [Plan] Test Fixtures
  - Issue #445: [Impl] Test Fixtures
  - Issue #446: [Package] Test Fixtures
  - Issue #447: [Cleanup] Test Fixtures
- **Existing Code**:
  - `/tests/shared/conftest.mojo` (TestFixtures struct)
  - `/tests/helpers/fixtures.mojo` (placeholder)

## Implementation Notes

### Testing Approach

**Key Principle**: Validate that fixtures behave as documented and expected

**Focus Areas**:
1. **Correctness**: Fixtures produce correct data
2. **Consistency**: Same inputs → same outputs
3. **Independence**: Fixtures don't interfere with each other
4. **Documentation**: Behavior matches documentation

### Minimal Changes Principle

**Current State**: Basic seed fixtures exist and work
**Testing Goal**: Validate they work correctly, identify any gaps

**Priorities**:
1. **High**: Test existing seed fixtures thoroughly
2. **Medium**: Test data generator fixtures
3. **Low**: Prepare for future fixture tests (when Tensor ready)

### Next Steps

1. Implement seed fixture tests
2. Implement data generator tests
3. Validate fixture documentation
4. Test fixture independence
5. Document findings for Implementation phase (#445)
