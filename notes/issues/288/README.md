# Issue #288: [Plan] Confusion Matrix - Design and Documentation

## Objective

Design and document the confusion matrix component for detailed classification analysis. The confusion matrix will show the distribution of predictions versus ground truth labels, revealing which classes are confused with each other to enable identification of systematic errors and class-specific performance issues.

## Deliverables

- Confusion matrix specification (2D array of prediction counts)
- Row and column labels with class names
- Normalization options (by row, column, or total)
- Derived metrics specifications (per-class precision, recall)
- API design for matrix accumulation and incremental updates
- Design documentation for implementation phases

## Success Criteria

- [ ] Matrix correctly counts all predictions
- [ ] Normalization produces interpretable percentages
- [ ] Incremental updates work correctly
- [ ] Derived metrics match manual calculations
- [ ] API contracts are clearly defined
- [ ] Design supports both accumulation and single-batch computation

## Design Decisions

### Architecture Design

**Confusion Matrix Structure:**

- **Matrix Shape**: NxN for N classes (square matrix)
- **Row Convention**: Rows represent ground truth labels
- **Column Convention**: Columns represent predicted labels
- **Cell Values**: Count of samples where ground_truth=row and prediction=column
- **Data Type**: Integer counters (accumulate over batches)

**Key Design Choices:**

1. **Accumulation Strategy**:
   - Support incremental updates for large datasets
   - Allow both batch-wise accumulation and single-batch computation
   - Enable reset functionality for starting new evaluation runs

2. **Normalization Options**:
   - Row normalization: Show percentage of each true class predicted as each class
   - Column normalization: Show percentage of each predicted class from each true class
   - Total normalization: Show percentage of all samples
   - No normalization: Raw counts

3. **Derived Metrics**:
   - Per-class precision: diagonal / column_sum
   - Per-class recall: diagonal / row_sum
   - Per-class F1-score: 2 * (precision * recall) / (precision + recall)
   - Support for extracting these metrics without manual calculation

### API Design

**Core Operations:**

```mojo
# Initialize confusion matrix
struct ConfusionMatrix:
    var matrix: Tensor[DType.int32]  # NxN matrix
    var num_classes: Int
    var class_names: Optional[List[String]]

    fn __init__(inout self, num_classes: Int, class_names: Optional[List[String]] = None)

    fn update(inout self, predictions: Tensor, labels: Tensor)
    fn reset(inout self)
    fn normalize(self, mode: String = "none") -> Tensor[DType.float32]
    fn get_precision(self) -> Tensor[DType.float32]
    fn get_recall(self) -> Tensor[DType.float32]
    fn get_f1_score(self) -> Tensor[DType.float32]
```

**Input Handling:**

- Accept class indices (integers 0 to N-1)
- Support logits (automatically convert to class indices via argmax)
- Validate inputs are within valid class range
- Handle edge cases (empty batches, single-class predictions)

**Output Formats:**

- Raw matrix: Integer counts
- Normalized matrix: Float percentages
- Labeled output: Include class names if provided
- Pretty-print formatting for visualization

### Implementation Strategy

**Phase 1 - Test (Issue #289):**

- Test matrix accumulation correctness
- Test normalization modes (row, column, total)
- Test incremental updates across batches
- Test derived metrics calculations
- Test edge cases (empty batches, single class, all correct/incorrect)

**Phase 2 - Implementation (Issue #290):**

1. Implement ConfusionMatrix struct with initialization
2. Implement update() method for accumulating predictions
3. Implement normalize() method with multiple modes
4. Implement derived metric extractors (precision, recall, F1)
5. Add reset() for starting new evaluation runs

**Phase 3 - Packaging (Issue #291):**

- Integrate with metrics module
- Export public API
- Add example usage in documentation
- Create visualization helpers (optional)

**Phase 4 - Cleanup (Issue #292):**

- Refactor for clarity and performance
- Optimize memory usage for large class counts
- Add comprehensive docstrings
- Final validation against test suite

### Technical Considerations

**Memory Management:**

- Matrix size scales as O(N²) for N classes
- Use owned tensor for internal storage
- Consider sparse representation for large N (future optimization)

**Performance:**

- Accumulation is O(batch_size) per update
- Normalization is O(N²) (computed on-demand, not cached)
- Derived metrics are O(N) each

**Edge Cases:**

- Empty batch: No-op, don't update matrix
- Single class in batch: Update only relevant row/column
- Classes outside [0, N-1]: Raise error or clip (design choice needed)
- All correct predictions: Diagonal only populated
- All incorrect predictions: Diagonal zeros, off-diagonal populated

**Integration Points:**

- Works with Accuracy metrics (share predictions/labels)
- Complements Loss tracking (different aspects of performance)
- Used by evaluation loops in training framework

## References

### Source Plan

[notes/plan/02-shared-library/01-core-operations/04-metrics/03-confusion-matrix/plan.md](../../../plan/02-shared-library/01-core-operations/04-metrics/03-confusion-matrix/plan.md)

### Parent Context

[notes/plan/02-shared-library/01-core-operations/04-metrics/plan.md](../../../plan/02-shared-library/01-core-operations/04-metrics/plan.md)

### Related Issues

- Issue #289: [Test] Confusion Matrix - Test suite implementation
- Issue #290: [Implementation] Confusion Matrix - Core functionality
- Issue #291: [Package] Confusion Matrix - Integration and packaging
- Issue #292: [Cleanup] Confusion Matrix - Refactoring and finalization

### Related Components

- Issue #283: [Plan] Accuracy metrics
- Issue #286: [Plan] Loss Tracking metrics

## Implementation Notes

*This section will be populated during subsequent phases (Test, Implementation, Packaging, Cleanup) with:*

- Design clarifications discovered during implementation
- Performance optimization findings
- Edge case handling decisions
- API refinements based on usage patterns
- Integration challenges and solutions
