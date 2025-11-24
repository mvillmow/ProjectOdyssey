# Issue #373: [Plan] Dataset Length - Design and Documentation

## Objective

Design and document the `__len__` method for datasets to return the total number of samples, enabling size queries,
progress tracking, and proper batch calculation in data loaders. This is a fundamental property for dataset
manipulation that must work consistently with Python's built-in `len()` function.

## Deliverables

- `__len__` method specification returning integer count
- Architecture design for correct counting across all dataset types
- Support for both static and dynamic sizing approaches
- Clear behavior specification for infinite/streaming datasets
- API contract documentation ensuring consistency with `__getitem__` indexing

## Success Criteria

- [ ] `__len__` returns correct sample count for all dataset types
- [ ] Works seamlessly with Python's `len()` builtin function
- [ ] Consistent with actual indexable range (no off-by-one errors)
- [ ] Clear error handling for unsized/infinite datasets
- [ ] Design document covers all edge cases and implementation strategies

## Design Decisions

### 1. Dataset Type Categories

The implementation must handle three distinct dataset categories:

**Static Datasets** (Fixed Size):

- In-memory datasets with known size at initialization
- File-based datasets with countable records
- Implementation: Direct return of stored length value
- Example: `return len(self._data)` for array-backed datasets

**Dynamic Datasets** (Computable Size):

- File-based datasets requiring scanning to determine size
- Datasets with lazy loading but finite content
- Implementation: Cache computed size after first access
- Example: Count lines in CSV file, cache result for subsequent calls

**Streaming/Infinite Datasets** (Undefined Size):

- Datasets from infinite generators or streams
- Real-time data feeds without predetermined bounds
- Implementation: Raise `TypeError` (matches Python convention for unsized iterables)
- Example: `raise TypeError("Dataset has infinite length")`

### 2. Consistency with Indexing

Critical design constraint: `__len__` must guarantee that all indices `0 <= i < len(dataset)` are valid for `__getitem__`:

```python
# This contract must always hold
for i in range(len(dataset)):
    sample = dataset[i]  # Must never raise IndexError
```text

Implementation strategies:

- For static datasets: Store and return fixed length
- For dynamic datasets: Cache length after first computation to ensure consistency
- For datasets with mutations: Update cached length on append/remove operations

### 3. Performance Considerations

### Lazy Computation for File-Based Datasets

- First call to `__len__` may scan entire file
- Result must be cached to avoid repeated expensive operations
- Trade-off: Initial latency vs. memory overhead of caching

### Memory-Efficient Counting

- For large files, count records without loading data into memory
- Use iterative scanning rather than loading entire dataset
- Example: Count newlines in CSV without parsing data

### 4. Error Handling Strategy

### Unsized Datasets

- Raise `TypeError` with descriptive message
- Follows Python convention (e.g., `len(generator)` raises TypeError)
- Message format: `"Dataset type 'X' does not support len()"`

### Corrupted/Inaccessible Data

- If size computation fails (file unreadable, network error), propagate exception
- Don't silently return 0 or None
- Allow caller to handle data access errors appropriately

### 5. Integration with Data Loaders

The `__len__` implementation enables:

### Progress Tracking

- Data loaders can display progress: "Batch 45/100"
- Training loops can estimate time remaining

### Batch Calculation

- `num_batches = len(dataset) // batch_size`
- Enables proper epoch iteration

### Validation

- Sanity check that dataset is not empty
- Verify expected dataset size matches actual

## API Specification

```python
def __len__(self) -> int:
    """
    Return the total number of samples in the dataset.

    Returns:
        int: Total number of samples available for indexing.
             All indices 0 <= i < len(self) must be valid for __getitem__.

    Raises:
        TypeError: If dataset has infinite/undefined length (e.g., streaming datasets)

    Notes:
        - For file-based datasets, first call may perform one-time scanning
        - Result is cached for subsequent calls to maintain consistency
        - Must remain constant for the lifetime of the dataset (unless explicitly mutated)
    """
```text

## Implementation Strategies

### Strategy 1: Static Length (In-Memory Datasets)

```python
class InMemoryDataset:
    def __init__(self, data):
        self._data = data
        self._length = len(data)  # Cache at initialization

    def __len__(self):
        return self._length  # O(1) lookup
```text

**Pros**: Instant, no computation needed
**Cons**: Only works for pre-loaded data

### Strategy 2: Lazy Cached Length (File-Based Datasets)

```python
class FileDataset:
    def __init__(self, filepath):
        self._filepath = filepath
        self._cached_length = None  # Computed on first access

    def __len__(self):
        if self._cached_length is None:
            self._cached_length = self._count_records()
        return self._cached_length

    def _count_records(self):
        # Count without loading into memory
        count = 0
        with open(self._filepath, 'r') as f:
            for _ in f:
                count += 1
        return count
```text

**Pros**: Memory-efficient, only computes when needed
**Cons**: First call has latency

### Strategy 3: Undefined Length (Streaming Datasets)

```python
class StreamingDataset:
    def __init__(self, stream):
        self._stream = stream

    def __len__(self):
        raise TypeError(
            f"{type(self).__name__} is a streaming dataset with undefined length. "
            "Use iteration instead of len()."
        )
```text

**Pros**: Clear error message guides user to correct API
**Cons**: Cannot use with APIs requiring `len()`

## Edge Cases

### Empty Datasets

```python
# Must return 0, not raise error
empty_dataset = InMemoryDataset([])
assert len(empty_dataset) == 0
```text

### Single-Sample Datasets

```python
# Must return 1
single_dataset = InMemoryDataset([sample])
assert len(single_dataset) == 1
```text

### Corrupted Files

```python
# If file is corrupted, propagate exception during counting
# Don't silently return wrong length
try:
    length = len(corrupted_dataset)
except IOError as e:
    # Caller handles appropriately
    pass
```text

### Consistency After Caching

```python
# Length must not change after first call (unless dataset mutated)
dataset = FileDataset("data.csv")
len1 = len(dataset)
len2 = len(dataset)
assert len1 == len2  # Must be identical
```text

## References

- **Source Plan**: [notes/plan/02-shared-library/03-data-utils/01-base-dataset/02-dataset-length/plan.md](../../../../plan/02-shared-library/03-data-utils/01-base-dataset/02-dataset-length/plan.md)
- **Parent Component**: Base Dataset (Issue #368)
- **Related Issues**:
  - #374 - [Test] Dataset Length - Test Implementation
  - #375 - [Impl] Dataset Length - Implementation
  - #376 - [Package] Dataset Length - Integration and Packaging
  - #377 - [Cleanup] Dataset Length - Cleanup and Finalization
- **Python Documentation**: [Data Model - `__len__`](https://docs.python.org/3/reference/datamodel.html#object.__len__)

## Implementation Notes

(This section will be filled during the implementation phase with discoveries, challenges, and solutions encountered)
