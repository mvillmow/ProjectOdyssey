# Issue #383: [Plan] Base Dataset - Design and Documentation

## Objective

Define the foundational dataset interface for consistent data access across all implementations. This includes the dataset interface specification, length method for dataset size, and getitem method for indexed access to enable interchangeable dataset implementations.

## Deliverables

- Dataset interface defining required methods (`__len__`, `__getitem__`)
- Length implementation for dataset size queries
- Getitem implementation for indexed data access
- Comprehensive design documentation and usage examples

## Success Criteria

- [ ] Dataset interface is clear and minimal
- [ ] Length method returns correct dataset size
- [ ] Getitem method provides indexed access to samples
- [ ] Interface follows standard Python sequence protocols
- [ ] Documentation explains usage patterns and API contracts
- [ ] All child plans are completed successfully

## Design Decisions

### 1. Interface Design Philosophy

**Decision**: Implement a minimal, Pythonic interface following Python's sequence protocol.

### Rationale

- Adheres to Python's "duck typing" philosophy and existing conventions
- Makes datasets interchangeable with standard Python sequences
- Reduces learning curve for developers familiar with Python
- Enables use with standard Python tools (len(), indexing, slicing)

### Key Methods

- `__len__()` - Returns total number of samples (enables `len(dataset)`)
- `__getitem__(index)` - Returns sample at index (enables `dataset[i]`)

### 2. Abstract Base Class vs Trait

**Decision**: Use abstract base class (ABC) approach for interface definition.

### Rationale

- Provides clear contract enforcement through abstract methods
- Enables isinstance() checks for type validation
- Standard Python pattern for defining interfaces
- Allows for optional default implementations of utility methods
- Compatible with Mojo's trait system for future migration

### 3. Data Return Format

**Decision**: Standardize on tuple format `(data, label)` for consistency.

### Rationale

- Consistent return format simplifies downstream processing
- Aligns with PyTorch dataset conventions (reduces friction for familiar developers)
- Tuple unpacking provides clean syntax: `data, label = dataset[i]`
- Extensible to multi-output scenarios if needed

### 4. Index Support Strategy

**Decision**: Support both integer indexing and slice notation.

### Rationale

- Integer indexing: Essential for single sample retrieval
- Slice notation: Enables batch retrieval without explicit loops
- Negative indexing: Pythonic way to access from end of dataset
- Consistent with Python sequence behavior

### Implementation Notes

- Validate indices are within bounds `[0, len(dataset))`
- Support negative indices via modulo arithmetic
- Handle slice objects for range access
- Raise clear IndexError for out-of-bounds access

### 5. Transform Pipeline Integration

**Decision**: Apply transforms/preprocessing within `__getitem__` method.

### Rationale

- Lazy evaluation: Only process data when accessed
- Memory efficiency: Avoid storing transformed data
- Flexibility: Different transforms can be applied to same dataset
- Composability: Transforms can be chained/swapped easily

### Considerations

- Transforms should be optional (default: no transformation)
- Transform function signature: `transform(data) -> data`
- Thread-safety for concurrent data loading

### 6. Dataset Size Handling

**Decision**: Distinguish between sized and unsized datasets.

### Rationale

- Static datasets (files, arrays): Known size, return count directly
- Dynamic datasets (generators, streams): May need lazy computation or caching
- Infinite datasets (continuous streams): Raise NotImplementedError or return special value

### Implementation

- For file-based: Scan directory or use cached metadata
- For in-memory: Return array/list length directly
- For infinite: Raise TypeError with clear message

### 7. Thread Safety Requirements

**Decision**: Design interface to be thread-safe for concurrent access.

### Rationale

- Data loaders often use multiple workers for performance
- Concurrent access is common in training pipelines
- Prevents race conditions and data corruption

### Implementation Strategy

- Avoid mutable shared state in dataset instances
- Use thread-local storage for worker-specific resources
- Document thread-safety guarantees clearly

### 8. Error Handling Standards

**Decision**: Use standard Python exceptions with descriptive messages.

### Error Types

- `IndexError`: Out-of-bounds index access
- `TypeError`: Invalid index type (not int or slice)
- `NotImplementedError`: Unsupported operations (e.g., len() on infinite dataset)
- `ValueError`: Invalid configuration or parameters

### Error Messages

- Include actual index and valid range
- Suggest corrective action when possible
- Maintain consistency across all dataset implementations

### 9. Documentation Requirements

**Decision**: Provide comprehensive documentation for interface and implementations.

### Required Documentation

- Abstract base class docstrings with method contracts
- Usage examples for common patterns
- Type hints for all public methods
- Notes on thread-safety and performance characteristics

### 10. Compatibility Considerations

**Decision**: Design for cross-compatibility with PyTorch and future Mojo implementations.

### Strategy

- Mirror PyTorch Dataset API where appropriate
- Use standard Python types (no PyTorch-specific tensors in interface)
- Design for future Mojo migration with traits
- Maintain clean separation between interface and implementation

## References

### Source Plan

[/notes/plan/02-shared-library/03-data-utils/01-base-dataset/plan.md](notes/plan/02-shared-library/03-data-utils/01-base-dataset/plan.md)

### Child Plans

- [Dataset Interface](notes/plan/02-shared-library/03-data-utils/01-base-dataset/01-dataset-interface/plan.md)
- [Dataset Length](notes/plan/02-shared-library/03-data-utils/01-base-dataset/02-dataset-length/plan.md)
- [Dataset Getitem](notes/plan/02-shared-library/03-data-utils/01-base-dataset/03-dataset-getitem/plan.md)

### Related Issues

- Issue #383: [Plan] Base Dataset (this issue)
- Issue #384: [Test] Base Dataset
- Issue #385: [Implementation] Base Dataset
- Issue #386: [Package] Base Dataset
- Issue #387: [Cleanup] Base Dataset

### Shared Documentation

- [Agent Hierarchy](agents/hierarchy.md)
- [5-Phase Development Workflow](notes/review/README.md)
- [Delegation Rules](agents/delegation-rules.md)

## Implementation Notes

This section will be populated during the implementation phase with:

- Discoveries and insights from development
- Technical challenges encountered
- Solutions to unexpected issues
- Performance observations
- Integration notes with other components
