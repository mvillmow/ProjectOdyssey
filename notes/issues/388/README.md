# Issue #388: [Plan] Batching - Design and Documentation

## Objective

Design and document the batching functionality to group individual dataset samples into mini-batches for efficient training. This component is critical for reducing training time through parallelization and providing better gradient estimates than single-sample updates.

## Deliverables

- **Batching Architecture Specification**: Design document covering the batching mechanism, including API contracts and interfaces
- **API Design**: Interface specifications for:
  - Basic batching with fixed size
  - Final partial batch handling (drop or pad)
  - Custom collate functions for batch assembly
  - Variable-length sequence batching support
- **Design Documentation**: Comprehensive documentation of:
  - Batch dimension stacking strategy
  - Default collate function behavior
  - Custom collate function interface
  - Edge case handling (partial batches, variable-length data)

## Success Criteria

- [ ] Batches have correct size (except possibly last batch when configured to keep partial)
- [ ] All data included or properly dropped based on configuration
- [ ] Custom collate functions work correctly for complex data structures
- [ ] Variable-length data batches properly with appropriate padding/masking strategy
- [ ] Design document is complete and approved
- [ ] API contracts are clearly defined
- [ ] Edge cases are identified and solutions documented

## Design Decisions

### 1. Batching Strategy

**Decision**: Implement tensor stacking along a new batch dimension (dimension 0).

**Rationale**:
- Standard practice in ML frameworks (PyTorch, TensorFlow)
- Enables efficient SIMD operations on batched data
- Compatible with Mojo's tensor operations

**Example**:
- Input: N samples of shape `(C, H, W)` (e.g., images with channels, height, width)
- Output: Batch of shape `(N, C, H, W)` where N is batch size

### 2. Final Batch Handling

**Decision**: Support two modes via configuration:
1. **Drop mode**: Discard final batch if smaller than batch_size
2. **Keep mode**: Include final batch even if partial

**Rationale**:
- Drop mode: Ensures all batches have consistent size (simplifies training code)
- Keep mode: Ensures all data is used (important for validation/testing)

**Trade-offs**:
- Drop mode may waste data (up to batch_size - 1 samples)
- Keep mode requires handling variable batch sizes in training loop

### 3. Collate Function Design

**Decision**: Provide a default collate function with support for custom implementations.

**Default Collate Behavior**:
- Stack tensors along batch dimension
- Handle tuples/lists by recursively applying collate
- Preserve non-tensor types (labels, metadata)

**Custom Collate Interface**:

```mojo
fn collate_fn(samples: List[Sample]) -> Batch:
    """Custom collate function signature.

    Args:
        samples: List of individual samples from dataset

    Returns:
        Batched data structure
    """
    pass
```

**Use Cases**:
- Variable-length sequences (apply padding)
- Complex nested data structures
- Special preprocessing per batch
- Custom memory layouts

### 4. Variable-Length Sequence Support

**Decision**: Require custom collate function for variable-length sequences.

**Rationale**:
- Different applications need different padding strategies:
  - Zero padding
  - Mask-based padding
  - Pack/unpack strategies
- Avoid imposing specific padding choice in core batching logic
- Keeps batching code simple and flexible

**Recommended Pattern**:

```mojo
fn pad_collate(samples: List[Sequence]) -> Batch:
    """Collate variable-length sequences with padding.

    - Find max length in batch
    - Pad all sequences to max length
    - Create attention mask for valid positions
    """
    var max_len = find_max_length(samples)
    var padded_data = pad_sequences(samples, max_len)
    var mask = create_attention_mask(samples, max_len)
    return Batch(data=padded_data, mask=mask)
```

### 5. Memory Efficiency Considerations

**Decision**: Pre-allocate batch tensor when size is known.

**Rationale**:
- Avoid repeated memory allocations
- Reduce memory fragmentation
- Improve performance for large batches

**Implementation Strategy**:
- Calculate total batch size upfront
- Allocate single contiguous tensor
- Copy samples into pre-allocated space

### 6. Error Handling

**Decision**: Fail fast on incompatible sample shapes.

**Error Conditions**:
- Samples have different shapes (without custom collate)
- Samples have different types
- Batch size is invalid (≤ 0)
- Dataset is empty

**Rationale**:
- Early detection prevents silent failures
- Clear error messages improve debugging
- Prevents invalid data from reaching training loop

## References

### Source Plan

- [Batching Plan](../../../plan/02-shared-library/03-data-utils/02-data-loader/01-batching/plan.md)
- [Data Loader Plan (Parent)](../../../plan/02-shared-library/03-data-utils/02-data-loader/plan.md)

### Related Issues

- **Issue #389**: [Test] Batching - Test Implementation
- **Issue #390**: [Impl] Batching - Implementation
- **Issue #391**: [Package] Batching - Integration and Packaging
- **Issue #392**: [Cleanup] Batching - Cleanup and Finalization

### Related Components

- Dataset Interface (provides individual samples)
- Data Loader (parent component)
- Shuffling (sibling component, issue #393-396)
- Iteration (sibling component, issue #397-400)

### Architecture Documentation

- [Agent Hierarchy](../../../agents/hierarchy.md)
- [5-Phase Development Workflow](../../../notes/review/README.md)
- [Data Utils Architecture](../../../plan/02-shared-library/03-data-utils/plan.md)

## Implementation Notes

*This section will be populated during implementation phases with findings, decisions, and observations.*

### Key Findings

(To be added during Test, Implementation, and Packaging phases)

### Design Adjustments

(To be added if design needs refinement based on implementation experience)

### Performance Considerations

(To be added after benchmarking and optimization)

### Lessons Learned

(To be added during Cleanup phase)
