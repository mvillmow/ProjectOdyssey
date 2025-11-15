# Issue #278: [Plan] Accuracy - Design and Documentation

## Objective

Implement accuracy metrics for evaluating classification model performance, including top-1 accuracy for single predictions, top-k accuracy for k-best predictions, and per-class accuracy for detailed analysis.

## Deliverables

- Top-1 accuracy: percentage of correct predictions
- Top-k accuracy: percentage where correct label is in top k predictions
- Per-class accuracy: accuracy broken down by class
- Support for batched evaluation
- Comprehensive API documentation
- Design specifications for accuracy metric implementations

## Success Criteria

- [ ] Top-1 accuracy correctly counts exact matches
- [ ] Top-k accuracy properly evaluates k-best predictions
- [ ] Per-class accuracy provides class-wise breakdown
- [ ] Metrics handle edge cases (empty batches, single class)
- [ ] Design documentation clearly specifies API contracts
- [ ] All interface specifications are complete and unambiguous

## Design Decisions

### 1. Metric Input Flexibility

**Decision**: Support both logits and probabilities as inputs

**Rationale**: Different parts of the training pipeline may produce either raw logits (pre-softmax) or normalized probabilities (post-softmax). Supporting both input types makes the metrics more flexible and easier to integrate.

**Implementation approach**:

- Auto-detect input type or allow explicit specification
- Convert logits to probabilities internally when needed
- Document expected input formats clearly

### 2. Incremental vs. Batch Computation

**Decision**: Support both incremental updates and single-batch computation

**Rationale**: Large datasets may not fit in memory, requiring incremental metric updates across mini-batches. However, single-batch computation is simpler and sufficient for smaller evaluations.

**Implementation approach**:

- Provide stateful accumulator for incremental updates
- Maintain running counts and totals
- Allow reset/finalize operations
- Support simple one-shot computation for convenience

### 3. Top-K Accuracy Evaluation

**Decision**: Implement top-k accuracy as a relaxed accuracy metric

**Rationale**: For classification with many classes, top-1 accuracy can be overly strict. Top-k accuracy (checking if correct label is in top k predictions) provides a more nuanced evaluation metric.

**Implementation approach**:

- Use efficient k-largest selection (not full sort)
- Support configurable k value
- Default to k=5 for convenience

### 4. Per-Class Accuracy Representation

**Decision**: Return per-class accuracy as a class-indexed structure

**Rationale**: Per-class accuracy helps identify which classes the model struggles with. A structured output (e.g., dictionary or tensor indexed by class) makes this data easy to analyze and visualize.

**Implementation approach**:

- Track correct predictions and total samples per class
- Return accuracy values indexed by class ID
- Support optional class names for human-readable output
- Handle classes with zero samples gracefully

### 5. Edge Case Handling

**Decision**: Define clear behavior for edge cases

**Edge cases to handle**:

- Empty batches (no samples): Return None or NaN
- Single-class prediction: Return 1.0 or 0.0 based on correctness
- Missing classes in batch: Per-class metrics skip missing classes
- K larger than number of classes: Treat as guaranteed correct

**Rationale**: Explicit edge case handling prevents silent failures and makes debugging easier.

### 6. Memory Management

**Decision**: Use Mojo's ownership system for efficient memory handling

**Rationale**: Accuracy metrics may process large tensors. Using Mojo's `owned` and `borrowed` parameters ensures efficient memory usage without unnecessary copies.

**Implementation approach**:

- Accept predictions and labels as `borrowed` (read-only)
- Return metric values as owned primitives (Float64)
- Avoid unnecessary tensor copies

### 7. API Design

**Proposed interfaces**:

```mojo
# Top-1 accuracy - single batch
fn top1_accuracy(
    predictions: borrowed Tensor,
    labels: borrowed Tensor
) -> Float64:
    """Calculate top-1 accuracy for classification predictions."""
    pass

# Top-k accuracy - single batch
fn topk_accuracy(
    predictions: borrowed Tensor,
    labels: borrowed Tensor,
    k: Int = 5
) -> Float64:
    """Calculate top-k accuracy for classification predictions."""
    pass

# Per-class accuracy - single batch
fn per_class_accuracy(
    predictions: borrowed Tensor,
    labels: borrowed Tensor,
    num_classes: Int
) -> Tensor:  # Shape: [num_classes]
    """Calculate per-class accuracy breakdown."""
    pass

# Incremental accuracy accumulator
struct AccuracyMetric:
    var correct_count: Int
    var total_count: Int
    var class_correct: Tensor  # Per-class correct counts
    var class_total: Tensor     # Per-class total counts

    fn __init__(inout self, num_classes: Int):
        """Initialize accumulator for incremental updates."""
        pass

    fn update(inout self, predictions: borrowed Tensor, labels: borrowed Tensor):
        """Update metrics with new batch."""
        pass

    fn compute(self) -> Float64:
        """Compute final accuracy value."""
        pass

    fn compute_per_class(self) -> Tensor:
        """Compute per-class accuracy values."""
        pass

    fn reset(inout self):
        """Reset accumulated values."""
        pass
```

### 8. Testing Strategy

**Unit tests needed**:

- Perfect predictions (100% accuracy)
- Random predictions (approximately 1/num_classes accuracy)
- Edge cases (empty batch, single class, k > num_classes)
- Incremental updates match batch computation
- Per-class accuracy sums correctly

**Integration tests needed**:

- Use with actual model outputs
- Verify compatibility with tensor operations
- Test with different batch sizes

## References

### Source Plan

[/notes/plan/02-shared-library/01-core-operations/04-metrics/01-accuracy/plan.md](/home/mvillmow/ml-odyssey-manual/notes/plan/02-shared-library/01-core-operations/04-metrics/01-accuracy/plan.md)

### Parent Context

[/notes/plan/02-shared-library/01-core-operations/04-metrics/plan.md](/home/mvillmow/ml-odyssey-manual/notes/plan/02-shared-library/01-core-operations/04-metrics/plan.md) - Metrics subsection

### Related Issues

- Issue #279: [Test] Accuracy - Write Tests
- Issue #280: [Impl] Accuracy - Implementation
- Issue #281: [Package] Accuracy - Integration and Packaging
- Issue #282: [Cleanup] Accuracy - Refactor and Finalize

### Comprehensive Documentation

- [Agent Hierarchy](/home/mvillmow/ml-odyssey-manual/agents/agent-hierarchy.md) - Full agent specifications
- [5-Phase Workflow](/home/mvillmow/ml-odyssey-manual/notes/review/README.md) - Phase dependencies and workflow

## Implementation Notes

This section will be populated during the implementation phases (Test, Implementation, Packaging, Cleanup) with:

- Discovered issues or challenges
- API refinements based on implementation experience
- Performance observations
- Integration challenges
- Lessons learned

Initially empty - to be filled as work progresses through subsequent phases.
