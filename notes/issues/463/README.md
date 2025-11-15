# Issue #463: [Plan] Test Data - Design and Documentation

## Objective

Design comprehensive unit tests for data utilities including base dataset, data loader, and augmentations. These tests will verify data access patterns, batching logic, shuffling behavior, and augmentation correctness using small, controlled test datasets rather than large real datasets.

## Deliverables

- Tests for dataset interface compliance (len, getitem)
- Tests for data loader batching and shuffling behavior
- Tests for all augmentation types
- Tests for edge cases (empty datasets, single-item datasets)
- Integration tests for complete data pipeline workflows

## Success Criteria

- [ ] Dataset tests verify interface compliance
- [ ] Loader tests verify batching and shuffling
- [ ] Augmentation tests verify property preservation
- [ ] Edge cases are handled correctly

## Design Decisions

### Testing Strategy

**Small In-Memory Datasets**: Use toy datasets that fit entirely in memory for fast test execution. This approach prioritizes speed over realism while still validating core functionality.

**Deterministic Testing**: Use deterministic, known-output data for batching tests to enable exact verification of batch contents. This ensures tests are reproducible and failures are easy to diagnose.

**Seed-Controlled Shuffling**: Test shuffling behavior using controlled random seeds, allowing verification that shuffling produces different orders while remaining deterministic for testing purposes.

**Property-Based Augmentation Testing**: For augmentations, verify that transformations preserve key properties (e.g., a flipped image has the same dimensions, rotations preserve data type) rather than comparing pixel values exactly.

### Test Coverage Areas

1. **Dataset Interface Compliance**
   - Verify `__len__` returns correct count
   - Verify `__getitem__` returns expected data structure
   - Test boundary conditions (index 0, last index)
   - Test invalid indices raise appropriate errors

2. **Data Loader Operations**
   - Verify batching creates correct batch sizes
   - Verify last batch handling (drop_last vs. partial batch)
   - Verify shuffling produces different orders across epochs
   - Test worker processes (if applicable)
   - Verify collation functions work correctly

3. **Augmentation Testing**
   - Test each augmentation type independently
   - Verify augmentations preserve tensor shapes
   - Verify augmentations preserve data types
   - Test composition of multiple augmentations
   - Verify random augmentations with controlled seeds

4. **Edge Cases**
   - Empty dataset (length 0)
   - Single-item dataset (length 1)
   - Batch size larger than dataset size
   - Invalid augmentation parameters

5. **Integration Tests**
   - Complete pipeline: dataset → loader → augmentation → batch
   - Verify data flows correctly through entire pipeline
   - Test realistic usage patterns

### Test Data Design

**Synthetic Data**: Create simple synthetic datasets with known properties (e.g., sequential integers, simple patterns) that make it easy to verify correctness.

**Fixtures**: Use pytest fixtures to provide reusable test datasets, loaders, and augmentation pipelines across multiple test functions.

**Parametrized Tests**: Use pytest parametrization to test multiple configurations (batch sizes, shuffle settings, augmentation types) efficiently.

## References

### Source Plan

[notes/plan/02-shared-library/04-testing/02-unit-tests/03-test-data/plan.md](../../../plan/02-shared-library/04-testing/02-unit-tests/03-test-data/plan.md)

### Related Issues

- Issue #464: [Test] Test Data - Implementation
- Issue #465: [Impl] Test Data - Test Implementation
- Issue #466: [Package] Test Data - Integration and Packaging
- Issue #467: [Cleanup] Test Data - Cleanup and Finalization

### Parent Plan

[notes/plan/02-shared-library/04-testing/02-unit-tests/plan.md](../../../plan/02-shared-library/04-testing/02-unit-tests/plan.md)

### Shared Documentation

- [Agent Hierarchy](../../../agents/hierarchy.md) - Agent roles and delegation patterns
- [5-Phase Development Workflow](../../review/README.md) - Complete workflow documentation
- [TDD Guidelines](../../../CLAUDE.md#key-development-principles) - Test-driven development approach

## Implementation Notes

This section will be populated during the Test, Implementation, Packaging, and Cleanup phases with:

- Specific test framework choices (pytest, unittest, etc.)
- Mock object strategies
- Performance benchmarks for test suite
- Issues discovered during implementation
- Lessons learned and best practices

## Test Implementation Strategy

### Phase 1: Dataset Interface Tests

Create tests that verify the basic dataset contract:

```python
def test_dataset_length():
    """Verify __len__ returns correct count"""
    dataset = create_toy_dataset(size=100)
    assert len(dataset) == 100

def test_dataset_getitem():
    """Verify __getitem__ returns expected structure"""
    dataset = create_toy_dataset(size=10)
    item = dataset[0]
    assert isinstance(item, tuple)  # (data, label)
    assert item[0].shape == expected_shape
```

### Phase 2: Data Loader Tests

Create tests that verify batching and shuffling:

```python
def test_loader_batching():
    """Verify batches have correct size"""
    loader = create_toy_loader(batch_size=4, dataset_size=10)
    batches = list(loader)
    assert len(batches) == 3  # 4 + 4 + 2
    assert batches[0][0].shape[0] == 4
    assert batches[2][0].shape[0] == 2

def test_loader_shuffling():
    """Verify shuffling produces different orders"""
    loader = create_toy_loader(shuffle=True, seed=42)
    epoch1_indices = extract_indices(loader)
    epoch2_indices = extract_indices(loader)
    assert epoch1_indices != epoch2_indices
```

### Phase 3: Augmentation Tests

Create tests that verify augmentation properties:

```python
def test_flip_augmentation():
    """Verify flip preserves shape and type"""
    image = create_test_image(height=28, width=28)
    augmented = flip_horizontal(image)
    assert augmented.shape == image.shape
    assert augmented.dtype == image.dtype
    # Verify flip actually occurred
    assert not torch.equal(augmented, image)
```

### Phase 4: Edge Case Tests

Create tests for boundary conditions:

```python
def test_empty_dataset():
    """Verify empty dataset handled gracefully"""
    dataset = create_toy_dataset(size=0)
    assert len(dataset) == 0
    loader = create_toy_loader(dataset=dataset, batch_size=4)
    batches = list(loader)
    assert len(batches) == 0

def test_batch_size_larger_than_dataset():
    """Verify large batch size doesn't break loader"""
    loader = create_toy_loader(batch_size=100, dataset_size=10)
    batches = list(loader)
    assert len(batches) == 1
    assert batches[0][0].shape[0] == 10
```

### Phase 5: Integration Tests

Create tests for complete pipelines:

```python
def test_complete_data_pipeline():
    """Verify data flows through entire pipeline"""
    dataset = create_toy_dataset(size=100)
    augmentation = compose_augmentations([flip, rotate])
    loader = create_toy_loader(dataset=dataset, batch_size=4)

    for batch_idx, (data, labels) in enumerate(loader):
        augmented_data = augmentation(data)
        # Verify batch properties
        assert data.shape[0] == min(4, 100 - batch_idx * 4)
        assert augmented_data.shape == data.shape
        assert labels.shape[0] == data.shape[0]
```

## Key Considerations

### Performance

- Tests should run quickly (< 1 second for unit tests)
- Use small datasets (10-100 items) for speed
- Mock expensive operations (file I/O, GPU operations)

### Maintainability

- Use descriptive test names that explain what is being tested
- Keep tests focused on single behaviors
- Use fixtures to reduce code duplication
- Document non-obvious test logic

### Reliability

- Tests must be deterministic (no flaky tests)
- Use seeds for any random operations
- Avoid timing-dependent assertions
- Test cleanup (resources released properly)

### Coverage

- Aim for high code coverage (> 90%)
- Include both positive and negative test cases
- Test error conditions and exceptions
- Verify error messages are helpful
