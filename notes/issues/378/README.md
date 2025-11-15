# Issue #378: [Plan] Dataset Getitem - Design and Documentation

## Objective

Design and document the `__getitem__` method for datasets to retrieve individual samples by index. This is the core data access method that enables both random access and iteration, essential for all dataset operations.

## Deliverables

- `__getitem__` method specification returning sample(s)
- Support for integer indexing and slicing
- Consistent return format (tuple of data, label)
- Clear error handling for out-of-bounds access
- Comprehensive API documentation
- Usage examples and patterns

## Success Criteria

- [ ] Returns correct sample for valid indices
- [ ] Supports both positive and negative indexing
- [ ] Slice notation works correctly
- [ ] Clear errors for invalid indices
- [ ] Documentation covers all usage patterns
- [ ] Design aligns with Python sequence protocol

## Design Decisions

### 1. Return Format

**Decision**: Use consistent `(data, label)` tuple format for all retrievals

**Rationale**:

- Follows PyTorch dataset conventions
- Predictable interface for users
- Easy to destructure: `data, label = dataset[i]`
- Extensible to multi-label or multi-modal datasets

**Alternatives Considered**:

- Dictionary format `{"data": ..., "label": ...}` - more verbose, less Pythonic
- Named tuple - adds complexity without clear benefit
- Raw data only - insufficient for supervised learning

### 2. Indexing Support

**Decision**: Support both positive and negative indexing, following Python list semantics

**Rationale**:

- Users expect Python-like behavior
- Negative indexing is idiomatic (`dataset[-1]` for last sample)
- Minimal implementation complexity
- Enables reverse iteration patterns

**Implementation Notes**:

- Convert negative indices to positive: `index = index % len(self)`
- Validate range: `0 <= index < len(self)`
- Raise `IndexError` for out-of-bounds access

### 3. Slice Support

**Decision**: Implement slice support for range access

**Rationale**:

- Enables batch operations: `dataset[0:10]`
- Supports sampling patterns: `dataset[::2]` (every other sample)
- Consistent with Python sequence protocol
- Required for efficient data loading

**Implementation Notes**:

- Return list of samples for slices
- Apply same `(data, label)` format to each sample
- Handle empty slices gracefully
- Step parameter must be supported

### 4. Transform Application

**Decision**: Apply transforms AFTER data retrieval, not during storage

**Rationale**:

- Separation of concerns (storage vs. preprocessing)
- Enables dynamic transforms (e.g., random augmentation)
- Reduces memory footprint (store raw data once)
- Allows transform pipeline composition

**Implementation Pattern**:

```python
def __getitem__(self, index):
    # 1. Retrieve raw data
    data, label = self._load_raw_data(index)

    # 2. Apply transforms if configured
    if self.transform:
        data = self.transform(data)

    # 3. Return processed result
    return data, label
```

### 5. Error Handling

**Decision**: Use Python standard exceptions with clear messages

**Exceptions**:

- `IndexError` - Out-of-bounds access
- `TypeError` - Invalid index type (e.g., float, string)
- `ValueError` - Invalid slice parameters

**Error Messages**:

- Include index value and dataset size
- Example: `"Index 150 out of range for dataset of size 100"`
- Help users debug quickly

### 6. Thread Safety

**Decision**: Design for thread-safe access, but don't enforce locking at this level

**Rationale**:

- Data loaders may use multiple workers (multiprocessing)
- File I/O should be stateless (open, read, close per access)
- Caching should use thread-safe structures (if implemented)
- Let higher-level components (DataLoader) manage concurrency

**Implementation Notes**:

- Avoid shared mutable state
- Use local variables for all operations
- Document thread-safety guarantees
- Defer to parent class for caching logic

### 7. Performance Considerations

**Decision**: Optimize for single-sample access, not batch operations

**Rationale**:

- `__getitem__` is called per sample by data loaders
- Batch operations happen at DataLoader level
- Keep implementation simple and fast
- Avoid premature optimization

**Best Practices**:

- Lazy loading (load on access, not construction)
- Minimize allocations in hot path
- Use efficient file I/O (mmap for large files)
- Profile before optimizing

## References

### Source Plan

- [Dataset Getitem Plan](/home/mvillmow/ml-odyssey-manual/notes/plan/02-shared-library/03-data-utils/01-base-dataset/03-dataset-getitem/plan.md)
- [Base Dataset Plan](/home/mvillmow/ml-odyssey-manual/notes/plan/02-shared-library/03-data-utils/01-base-dataset/plan.md)

### Related Issues

- Issue #374 - [Plan] Dataset Interface (sibling component)
- Issue #376 - [Plan] Dataset Length (sibling component)
- Issue #379 - [Test] Dataset Getitem (testing phase)
- Issue #380 - [Impl] Dataset Getitem (implementation phase)
- Issue #381 - [Package] Dataset Getitem (packaging phase)
- Issue #382 - [Cleanup] Dataset Getitem (cleanup phase)

### Documentation

- [5-Phase Development Workflow](/home/mvillmow/ml-odyssey-manual/notes/review/README.md)
- [Agent Hierarchy](/home/mvillmow/ml-odyssey-manual/agents/hierarchy.md)
- [Documentation Specialist Role](/home/mvillmow/ml-odyssey-manual/.claude/agents/documentation-specialist.md)

## Implementation Notes

### API Signature

```python
def __getitem__(self, index: int | slice) -> tuple[Any, Any] | list[tuple[Any, Any]]:
    """
    Retrieve sample(s) by index.

    Parameters
    ----------
    index : int | slice
        Integer index for single sample or slice for multiple samples.
        Supports negative indexing following Python list semantics.

    Returns
    -------
    tuple[Any, Any] | list[tuple[Any, Any]]
        For integer index: (data, label) tuple
        For slice: list of (data, label) tuples

    Raises
    ------
    IndexError
        If index is out of bounds
    TypeError
        If index is not int or slice
    """
    pass
```

### Edge Cases to Handle

1. **Empty Dataset**: `len(dataset) == 0`
   - Any access should raise `IndexError`

2. **Single Sample Dataset**: `len(dataset) == 1`
   - Valid indices: `0`, `-1`
   - Invalid: any other value

3. **Negative Indexing**: `dataset[-5]`
   - Convert to positive: `index = len(dataset) + index`
   - Validate range

4. **Empty Slice**: `dataset[5:5]`
   - Return empty list `[]`

5. **Reverse Slice**: `dataset[10:0:-1]`
   - Return samples in reverse order

6. **Step Slice**: `dataset[0:10:2]`
   - Return every 2nd sample

### Testing Strategy

See Issue #379 for detailed test plan. Key test categories:

1. **Basic Access**: Single index retrieval
2. **Negative Indexing**: `dataset[-1]`, `dataset[-5]`
3. **Slice Access**: Range, step, reverse
4. **Error Cases**: Out of bounds, invalid types
5. **Transform Application**: Verify transforms applied
6. **Edge Cases**: Empty dataset, single sample

### Performance Requirements

- Single sample access: `< 1ms` (excluding I/O)
- Slice access: Linear in number of samples
- No memory leaks (verify with profiling)

### Documentation Requirements

1. **Docstring**: Complete with examples
2. **Usage Guide**: Common patterns
3. **Error Reference**: All exceptions documented
4. **Performance Notes**: Expected characteristics

## Completion Checklist

- [ ] Design decisions documented (this file)
- [ ] API signature defined
- [ ] Error handling strategy documented
- [ ] Edge cases identified
- [ ] Testing strategy outlined
- [ ] Performance requirements specified
- [ ] Related issues updated (#379-382)
- [ ] Planning phase marked complete

## Notes

*This section will be updated during implementation with findings and decisions.*
