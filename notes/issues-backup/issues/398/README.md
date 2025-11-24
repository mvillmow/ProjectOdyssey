# Issue #398: [Plan] Iteration - Design and Documentation

## Objective

Design and document the iterator interface for data loaders to enable sequential batch access. This component
implements the Python iterator protocol to provide a clean mechanism for training loops to traverse batches,
supporting both finite epoch-based iteration and infinite streaming patterns.

## Deliverables

This planning phase will produce:

- Detailed specification for `__iter__` method returning iterator instance
- Design for `__next__` method yielding batches sequentially
- State management strategy for iterator position and epoch tracking
- Exception handling strategy for `StopIteration` at epoch boundaries
- Infinite iteration mode design for continuous training scenarios
- API contracts and interface documentation
- Integration patterns with batching (#395) and shuffling (#396) components

## Success Criteria

- [ ] Complete specification for iterator protocol implementation
- [ ] Documented behavior for finite vs infinite iteration modes
- [ ] State management design for multi-epoch support
- [ ] Integration design with batching and shuffling components
- [ ] API documentation for `__iter__` and `__next__` methods
- [ ] Error handling and edge case documentation
- [ ] Design review completed and approved

## Design Decisions

### 1. Python Iterator Protocol Compliance

**Decision**: Implement standard Python iterator protocol with `__iter__` and `__next__` methods.

### Rationale

- Enables use with native Python `for` loops and iteration constructs
- Familiar pattern for Python developers
- Simplifies integration with existing Python ML frameworks
- Allows use of built-in functions like `next()`, `iter()`, and iteration utilities

### Implementation

- `__iter__()` returns `self` to make the data loader itself iterable
- `__next__()` yields the next batch and raises `StopIteration` at epoch end
- State tracking internal to the iterator instance

### 2. Iteration Modes: Finite vs Infinite

**Decision**: Support both finite epoch-based iteration and infinite streaming.

### Rationale

- Finite mode: Standard training with defined epochs (most common use case)
- Infinite mode: Online learning, continuous training, or reinforcement learning scenarios
- Different applications have different iteration requirements

### Design

- Finite mode (default): Raises `StopIteration` after all batches yielded once
- Infinite mode: Automatically resets to beginning after epoch end, never raises `StopIteration`
- Mode controlled by configuration parameter in data loader initialization
- Mode cannot change during iteration (prevents inconsistent state)

### 3. State Management for Multi-Epoch Support

**Decision**: Track iterator position internally with reset capability for new epochs.

### Rationale

- Training loops typically iterate multiple epochs over the same dataset
- State must persist across `__next__` calls within an epoch
- State must reset cleanly between epochs
- Support for resumption if training is interrupted

### State Variables

- `_current_batch_index`: Position within current epoch (0 to num_batches-1)
- `_epoch_complete`: Boolean flag indicating if current epoch finished
- `_num_batches`: Total number of batches per epoch (computed from dataset size and batch size)

### Reset Behavior

- Calling `__iter__()` resets state for new epoch
- Resets `_current_batch_index` to 0
- Clears `_epoch_complete` flag
- Triggers reshuffling if shuffle mode enabled (delegates to shuffling component #396)

### 4. Integration with Batching and Shuffling

**Decision**: Iterator delegates to batching (#395) and shuffling (#396) components.

### Rationale

- Separation of concerns: iterator handles traversal, not data organization
- Batching component owns batch creation logic
- Shuffling component owns randomization logic
- Iterator simply accesses pre-organized batches sequentially

### Integration Pattern

```python
def __next__(self):
    if self._current_batch_index >= self._num_batches:
        if self._infinite_mode:
            self._reset_for_new_epoch()
        else:
            raise StopIteration

    # Delegate to batching component for batch retrieval
    batch = self._get_batch_at_index(self._current_batch_index)
    self._current_batch_index += 1
    return batch
```text

### 5. Error Handling and Edge Cases

**Decision**: Explicit handling of edge cases with clear error messages.

### Edge Cases

1. **Empty dataset**: Immediately raise `StopIteration` on first `__next__` call
1. **Batch size larger than dataset**: Return single batch containing all samples
1. **Dataset size not divisible by batch size**: Final batch contains remaining samples
1. **Concurrent iteration**: Not supported - data loader is not thread-safe
1. **Mid-iteration reset**: Calling `__iter__()` during iteration resets state (restarts epoch)

### Error Messages

- Clear messages indicating which edge case was encountered
- Guidance on how to resolve (e.g., "Dataset is empty. Provide non-empty dataset.")

### 6. Memory and Performance Considerations

**Decision**: Lazy batch generation with minimal memory overhead.

### Rationale

- Large datasets should not require loading entire dataset into memory
- Batches generated on-demand during iteration
- State tracking uses minimal memory (few integer counters)

### Performance

- O(1) time complexity for `__next__` operation (batch already prepared by batching component)
- O(1) space complexity for iterator state
- No data copying - batches reference underlying dataset storage

## API Specification

### DataLoader Iterator Interface

```python
class DataLoader:
    """Iterator interface for sequential batch access."""

    def __iter__(self) -> 'DataLoader':
        """
        Return iterator instance (self).

        Resets iteration state for new epoch:
        - Resets batch index to 0
        - Triggers reshuffling if enabled
        - Clears epoch completion flag

        Returns:
            Self reference for iterator protocol compliance
        """
        pass

    def __next__(self) -> Batch:
        """
        Yield next batch in sequence.

        Returns:
            Next batch from the dataset

        Raises:
            StopIteration: When epoch complete (finite mode only)

        Behavior:
            - Finite mode: Raises StopIteration after all batches yielded
            - Infinite mode: Automatically resets and continues indefinitely
        """
        pass
```text

### Usage Examples

#### Finite Iteration (Standard Training)

```python
# Create data loader with finite iteration (default)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Standard training loop - multiple epochs
for epoch in range(num_epochs):
    for batch in loader:  # __iter__() called, then __next__() repeatedly
        # Training step with batch
        train_step(batch)
    # StopIteration raised automatically at epoch end
    # Next epoch: __iter__() resets state
```text

#### Infinite Iteration (Continuous Training)

```python
# Create data loader with infinite iteration
loader = DataLoader(dataset, batch_size=32, shuffle=True, infinite=True)

# Infinite training loop
for batch in loader:  # Never raises StopIteration
    # Training step with batch
    train_step(batch)

    # Manual stopping condition required
    if should_stop():
        break
```text

#### Manual Iteration Control

```python
loader = DataLoader(dataset, batch_size=32)

# Get iterator
iterator = iter(loader)  # Calls __iter__()

# Manual batch retrieval
batch1 = next(iterator)  # Calls __next__()
batch2 = next(iterator)
# ... continue until StopIteration
```text

## References

### Source Plan

- [Iteration Plan](notes/plan/02-shared-library/03-data-utils/02-data-loader/03-iteration/plan.md)
  - Component specification
- [Data Loader Plan](notes/plan/02-shared-library/03-data-utils/02-data-loader/plan.md)
  - Parent component context

### Related Issues

- Issue #399: [Test] Iteration - Test implementation for iterator interface
- Issue #400: [Impl] Iteration - Implementation of iterator methods
- Issue #401: [Package] Iteration - Integration and packaging
- Issue #402: [Cleanup] Iteration - Finalization and refactoring

### Dependencies

- Issue #395: [Impl] Batching - Provides batch creation functionality
- Issue #396: [Impl] Shuffling - Provides randomization functionality

### Architecture Documentation

- Python Iterator Protocol: [PEP 234](https://www.python.org/dev/peps/pep-0234/)
- Data Utils Architecture: `notes/plan/02-shared-library/03-data-utils/plan.md`

## Implementation Notes

This section will be populated during the Test (#399), Implementation (#400), and Packaging (#401) phases with:

- Implementation discoveries and decisions
- Test coverage insights
- Integration challenges and solutions
- Performance optimization notes
- Edge cases encountered during development

---

**Status**: Planning phase complete - ready for parallel Test/Implementation/Packaging phases

**Last Updated**: 2025-11-15

**Phase**: Plan (Issue #398)
