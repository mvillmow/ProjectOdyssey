# Issue #429: [Test] Data Utils - Test Suite Implementation

## Objective

Implement comprehensive test suite for data utilities including base dataset interface, data loader with batching/shuffling, and file loading functionality.

## Deliverables

### Test Coverage

- **Base Dataset Tests**:
  - Dataset interface compliance (\_\_len\_\_, \_\_getitem\_\_)
  - TensorDataset implementation
  - FileDataset with lazy loading
  - Error handling for invalid indices

- **Data Loader Tests**:
  - Batching logic (including partial final batch)
  - Shuffling with deterministic seeds
  - Iterator interface and reset
  - Edge cases (empty dataset, batch size > dataset size)

- **File Loading Tests**:
  - Image file loading (.jpg, .png, .bmp)
  - NumPy file loading (.npy, .npz)
  - CSV file loading (.csv)
  - Error handling for unsupported formats

- **Integration Tests**:
  - Dataset + DataLoader workflows
  - File loading + augmentation pipelines
  - Multi-epoch training simulation

### Test Files

Current test files:

- `/home/user/ml-odyssey/tests/shared/data/datasets/test_base_dataset.mojo`
- `/home/user/ml-odyssey/tests/shared/data/datasets/test_tensor_dataset.mojo`
- `/home/user/ml-odyssey/tests/shared/data/datasets/test_file_dataset.mojo`
- `/home/user/ml-odyssey/tests/shared/data/test_loaders.mojo` (expected)

## Success Criteria

- [ ] All base dataset tests pass
- [ ] All data loader tests pass
- [ ] File loading tests cover major formats
- [ ] Edge cases comprehensively tested
- [ ] Integration tests validate workflows
- [ ] Test coverage ≥ 90% for data utils
- [ ] Reproducibility tests verify deterministic behavior

## Test Organization

```text
tests/shared/data/
├── datasets/
│   ├── test_base_dataset.mojo      # Base interface tests
│   ├── test_tensor_dataset.mojo    # In-memory dataset tests
│   └── test_file_dataset.mojo      # File-based dataset tests
├── loaders/
│   └── test_data_loader.mojo       # Batching and shuffling tests
└── test_data_integration.mojo      # End-to-end integration tests
```text

## Implementation Status

### Pending Test Files

All test files need to be reviewed and extended to cover:

1. **File Loading Tests**:
   - Test placeholder behavior (current state)
   - Test proper file loading (when implemented)
   - Test error handling for missing/corrupt files
   - Test format detection based on extension

1. **Data Loader Tests**:
   - Batching correctness
   - Shuffling reproducibility
   - Iterator protocol compliance
   - Edge case handling

1. **Integration Tests**:
   - Full data pipeline workflows
   - Memory efficiency under load
   - Multi-epoch scenarios

## Test Scenarios

### 1. Base Dataset Tests

```mojo
fn test_dataset_length():
    """Test __len__ returns correct count."""
    var dataset = TensorDataset(data, labels)
    assert_equal(len(dataset), expected_length)

fn test_dataset_getitem():
    """Test __getitem__ retrieves correct samples."""
    var dataset = TensorDataset(data, labels)
    var sample = dataset[5]
    assert_equal(sample.data, expected_data)
    assert_equal(sample.label, expected_label)

fn test_dataset_invalid_index():
    """Test __getitem__ raises error for invalid index."""
    var dataset = TensorDataset(data, labels)
    # Should raise error
    try:
        _ = dataset[-1]
        assert_true(False, "Should have raised error")
    except:
        pass  # Expected

fn test_dataset_empty():
    """Test empty dataset handles correctly."""
    var dataset = TensorDataset(empty_data, empty_labels)
    assert_equal(len(dataset), 0)
```text

### 2. Data Loader Tests

```mojo
fn test_loader_batching():
    """Test batching creates correct sized batches."""
    var dataset = TensorDataset(data, labels)
    var loader = DataLoader(dataset, batch_size=32)

    for batch in loader:
        # All batches except last should be size 32
        assert_true(
            batch.size == 32 or batch.is_last_batch
        )

fn test_loader_shuffling():
    """Test shuffling randomizes order."""
    var dataset = TensorDataset(data, labels)

    var loader1 = DataLoader(dataset, batch_size=32, shuffle=True, seed=42)
    var loader2 = DataLoader(dataset, batch_size=32, shuffle=True, seed=42)

    # Same seed should produce same order
    for batch1, batch2 in zip(loader1, loader2):
        assert_equal(batch1.data, batch2.data)

fn test_loader_no_shuffle():
    """Test non-shuffled loader maintains order."""
    var dataset = TensorDataset(sequential_data, labels)
    var loader = DataLoader(dataset, batch_size=32, shuffle=False)

    var expected_first = sequential_data[0]
    var actual_first = next(iter(loader)).data[0]
    assert_equal(actual_first, expected_first)

fn test_loader_partial_batch():
    """Test final batch handles partial correctly."""
    # 100 samples, batch_size=32 → [32, 32, 32, 4]
    var dataset = TensorDataset(data_100, labels_100)
    var loader = DataLoader(dataset, batch_size=32)

    var batches = list(loader)
    assert_equal(len(batches), 4)
    assert_equal(batches[3].size, 4)

fn test_loader_empty_dataset():
    """Test loader handles empty dataset."""
    var dataset = TensorDataset(empty_data, empty_labels)
    var loader = DataLoader(dataset, batch_size=32)

    var batches = list(loader)
    assert_equal(len(batches), 0)

fn test_loader_batch_size_larger_than_dataset():
    """Test batch size larger than dataset."""
    var dataset = TensorDataset(data_10, labels_10)
    var loader = DataLoader(dataset, batch_size=100)

    var batches = list(loader)
    assert_equal(len(batches), 1)
    assert_equal(batches[0].size, 10)
```text

### 3. File Loading Tests

```mojo
fn test_file_dataset_loading():
    """Test FileDataset loads from file paths."""
    var file_paths = [
        "data/image1.jpg",
        "data/image2.jpg",
        "data/image3.jpg",
    ]
    var dataset = FileDataset(file_paths)

    assert_equal(len(dataset), 3)
    var sample = dataset[0]
    # Currently returns placeholder, will return actual data when implemented
    assert_true(sample.num_elements() > 0)

fn test_file_loading_image_formats():
    """Test loading various image formats."""
    # When implemented, should load actual images
    var formats = [".jpg", ".png", ".bmp"]
    for fmt in formats:
        var path = "test_image" + fmt
        var dataset = FileDataset([path])
        var data = dataset[0]
        # Verify image loaded (when implemented)

fn test_file_loading_numpy():
    """Test loading NumPy files."""
    # When implemented
    var dataset = FileDataset(["data.npy"])
    var data = dataset[0]
    # Verify numpy data loaded

fn test_file_loading_csv():
    """Test loading CSV files."""
    # When implemented
    var dataset = FileDataset(["data.csv"])
    var data = dataset[0]
    # Verify CSV data parsed

fn test_file_loading_unsupported_format():
    """Test error for unsupported file format."""
    var dataset = FileDataset(["data.unknown"])
    try:
        _ = dataset[0]
        assert_true(False, "Should raise error")
    except:
        pass  # Expected

fn test_file_loading_missing_file():
    """Test error for missing file."""
    var dataset = FileDataset(["nonexistent.jpg"])
    try:
        _ = dataset[0]
        assert_true(False, "Should raise error")
    except:
        pass  # Expected
```text

### 4. Integration Tests

```mojo
fn test_data_pipeline_end_to_end():
    """Test complete data loading pipeline."""
    # Create dataset
    var dataset = TensorDataset(data, labels)

    # Create data loader
    var loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Simulate training loop
    for epoch in range(3):
        for batch in loader:
            # Verify batch structure
            assert_true(batch.data.num_elements() > 0)
            assert_true(batch.labels.num_elements() > 0)
            assert_equal(len(batch.data), len(batch.labels))

fn test_data_pipeline_with_augmentation():
    """Test data pipeline with augmentations."""
    var dataset = TensorDataset(images, labels)
    var loader = DataLoader(dataset, batch_size=32)

    # Create augmentation pipeline
    var transforms = List[Transform]()
    transforms.append(RandomHorizontalFlip(0.5))
    transforms.append(Normalize(0.5, 0.5))
    var augmentations = Pipeline(transforms^)

    # Apply augmentations in training loop
    for batch in loader:
        var augmented = augmentations(batch.data)
        # Verify augmentation doesn't change shape
        assert_equal(augmented.shape, batch.data.shape)
```text

## References

### Source Plan

- [Data Utils Plan](../../../../../../../home/user/ml-odyssey/notes/plan/02-shared-library/03-data-utils/plan.md)

### Related Issues

- Issue #428: [Plan] Data Utils
- Issue #429: [Test] Data Utils (this issue)
- Issue #430: [Impl] Data Utils
- Issue #431: [Package] Data Utils
- Issue #432: [Cleanup] Data Utils

### Implementation Files

- `/home/user/ml-odyssey/shared/data/datasets.mojo`
- `/home/user/ml-odyssey/shared/data/loaders.mojo`
- `/home/user/ml-odyssey/shared/data/samplers.mojo`

## Implementation Notes

### Current Test State

Existing test files need to be reviewed for:

- Coverage of edge cases
- File loading placeholder behavior
- Integration with augmentation pipelines

### Testing Strategy

1. **Unit Tests First**: Test each component in isolation
1. **Integration Tests**: Test component interactions
1. **Property-Based Tests**: Verify invariants (e.g., shuffle preserves count)
1. **Performance Tests**: Ensure acceptable loading/batching speed

### Key Testing Patterns

1. **Deterministic Randomness**: Use seeds for reproducible tests
1. **Edge Case Coverage**: Empty, single element, oversized batches
1. **Error Validation**: Verify appropriate errors for invalid inputs
1. **Round-Trip Testing**: Load → process → verify integrity

### File Loading Testing Notes

**Current State**: FileDataset has TODO for proper file loading

### Test Strategy

1. Test current placeholder behavior
1. Add tests for expected behavior (when implemented)
1. Use test fixtures for file formats
1. Mock file I/O for unit tests

### Implementation Priority

- High: Test placeholder doesn't crash
- Medium: Test expected interface
- Low: Test all file formats comprehensively

---

**Status**: Test planning complete, implementation pending

**Last Updated**: 2025-11-19

**Prepared By**: Implementation Specialist (Level 3)
