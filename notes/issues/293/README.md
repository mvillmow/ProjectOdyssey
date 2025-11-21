# Issue #293: [Plan] Metrics - Design and Documentation

## Objective

Design and document evaluation metrics for assessing model performance, including accuracy metrics for classification tasks, loss tracking for monitoring training progress, and confusion matrix for detailed classification analysis.

## Deliverables

- Detailed specifications for three metric components:
  - **Accuracy metrics**: Top-1, top-k, and per-class accuracy
  - **Loss tracking**: Cumulative tracking, moving averages, and statistical summaries
  - **Confusion matrix**: 2D prediction vs ground truth matrix with normalization options
- Architecture design for metric computation and storage
- API contracts and interfaces for metric integration
- Comprehensive design documentation covering:
  - Input/output specifications
  - Edge case handling strategies
  - Memory efficiency considerations
  - Integration with training pipeline

## Success Criteria

- [ ] Accuracy metrics correctly compare predictions to labels
  - Top-1 accuracy counts exact matches
  - Top-k accuracy evaluates k-best predictions
  - Per-class accuracy provides class-wise breakdown
  - Handles both logits and probabilities
- [ ] Loss tracking maintains accurate statistics
  - Values accumulate correctly across batches
  - Moving averages smooth loss curves appropriately
  - Statistics (mean, std, min, max) accurately summarize behavior
  - Multiple loss components track independently
- [ ] Confusion matrix properly categorizes predictions
  - Matrix correctly counts all predictions
  - Normalization produces interpretable percentages
  - Incremental updates work correctly
  - Derived metrics (precision, recall, F1) match manual calculations
- [ ] All child plans are completed successfully
- [ ] Edge cases handled (empty batches, single-class predictions)
- [ ] Outputs are easy to interpret and log

## Design Decisions

### 1. Component Architecture

**Decision**: Implement metrics as three independent, composable components (Accuracy, Loss Tracking, Confusion Matrix).

### Rationale

- Each metric serves a distinct purpose in model evaluation
- Independent components allow flexible usage (use only what's needed)
- Easier to test and maintain individual components
- Supports incremental development (can implement one at a time)

### Trade-offs

- Requires consistent API patterns across components
- May need a higher-level wrapper for common use cases
- Could lead to code duplication if not carefully designed

### 2. Incremental vs Batch Computation

**Decision**: Support both incremental updates and single-batch computation for all metrics.

### Rationale

- Training typically processes mini-batches, requiring incremental updates
- Validation often evaluates full datasets, benefiting from batched computation
- Large datasets may not fit in memory, requiring incremental processing
- Flexibility enables different evaluation workflows

### Implementation Strategy

- Metrics maintain internal state for accumulation
- Provide `update()` method for incremental additions
- Provide `compute()` method to calculate final result
- Support `reset()` to clear state between epochs

### 3. Input Format Handling

**Decision**: Accept both logits and probabilities for accuracy metrics.

### Rationale

- Different model implementations output different formats
- Softmax application is computationally expensive
- Top-k selection works on both logits and probabilities
- Reduces burden on calling code

### Implementation

- Internally convert to consistent format (likely argmax for predictions)
- Document which operations are performed automatically
- Avoid redundant computations (don't softmax if already probabilities)

### 4. Memory Efficiency

**Decision**: Prioritize memory efficiency for loss tracking and confusion matrix.

### Rationale

- Loss tracking over long runs can accumulate many values
- Confusion matrix for large class counts (e.g., ImageNet: 1000 classes) requires significant memory
- Training is memory-constrained (GPU memory limits)

### Implementation Strategy

- Loss tracking: Store only necessary statistics (not all individual values)
- Moving averages: Use fixed-size circular buffer
- Confusion matrix: Use sparse representation if appropriate
- Provide options to trade memory for functionality (e.g., disable detailed tracking)

### 5. Normalization Options

**Decision**: Support multiple normalization modes for confusion matrix (by row, column, total, none).

### Rationale

- Different normalizations reveal different insights:
  - Row normalization (by true label): Shows recall per class
  - Column normalization (by predicted label): Shows precision per class
  - Total normalization: Shows overall distribution
  - No normalization: Shows raw counts
- Scientific papers use different conventions
- Visualization tools expect different formats

### Trade-offs

- Increases API surface area
- Requires clear documentation of what each mode means
- May confuse users unfamiliar with confusion matrices

### 6. Edge Case Handling

**Decision**: Define clear behavior for edge cases (empty batches, single class, NaN/Inf values).

### Rationale

- Edge cases cause runtime failures if not handled
- Silent failures are worse than clear errors
- Consistent behavior aids debugging

### Defined Behaviors

- Empty batches: Return 0.0 for accuracy, skip for loss tracking
- Single class: Confusion matrix becomes 1x1, accuracy is 100% if correct
- NaN/Inf in loss: Raise error (indicates training instability)
- Mismatched prediction/label counts: Raise error

### 7. Derived Metrics

**Decision**: Provide helpers to extract precision, recall, and F1 from confusion matrix.

### Rationale

- These metrics are commonly needed for classification evaluation
- Calculations are straightforward but error-prone to implement manually
- Encourages best practices in model evaluation

### Scope Limitation

- Only include most common metrics (precision, recall, F1)
- Avoid expanding to specialized metrics (MCC, Cohen's kappa) initially
- Users can compute custom metrics from confusion matrix directly

### 8. API Design Principles

**Decision**: Follow consistent API patterns across all metric components.

### Rationale

- Reduces cognitive load when using multiple metrics
- Enables generic metric handling code
- Facilitates future extension with new metrics

### Common API Pattern

```mojo
struct MetricName:
    fn __init__(inout self, ...config_params):
        """Initialize metric with configuration."""

    fn update(inout self, predictions, labels):
        """Update metric state with new batch."""

    fn compute(self) -> ResultType:
        """Compute final metric value from accumulated state."""

    fn reset(inout self):
        """Clear accumulated state."""
```text

### 9. Integration Points

**Decision**: Design metrics to integrate with training pipeline through standardized interfaces.

### Rationale

- Metrics are consumed by training loops, validation, and logging
- Standardized interfaces enable generic training code
- Reduces coupling between training logic and specific metrics

### Integration Strategy

- Metrics should be passable as collection (List of metrics)
- Training loop calls `update()` during validation
- Logging code calls `compute()` at end of epoch
- Automatic reset between epochs

### 10. Performance Considerations

**Decision**: Use SIMD operations for performance-critical paths (accuracy computation, matrix accumulation).

### Rationale

- Metrics are computed frequently during training
- SIMD can significantly accelerate element-wise operations
- Mojo's SIMD support makes this straightforward

### Implementation Notes

- Vectorize comparisons for accuracy calculation
- Use SIMD for confusion matrix accumulation
- Profile to identify bottlenecks before optimizing

## References

### Source Plans

- [Metrics Plan](notes/plan/02-shared-library/01-core-operations/04-metrics/plan.md)
- [Accuracy Plan](notes/plan/02-shared-library/01-core-operations/04-metrics/01-accuracy/plan.md)
- [Loss Tracking Plan](notes/plan/02-shared-library/01-core-operations/04-metrics/02-loss-tracking/plan.md)
- [Confusion Matrix Plan](notes/plan/02-shared-library/01-core-operations/04-metrics/03-confusion-matrix/plan.md)

### Related Issues

- **Test Phase**: Issue #294 - [Test] Metrics
- **Implementation Phase**: Issue #295 - [Impl] Metrics
- **Packaging Phase**: Issue #296 - [Package] Metrics
- **Cleanup Phase**: Issue #297 - [Cleanup] Metrics

### Parent Context

- [Core Operations Plan](notes/plan/02-shared-library/01-core-operations/plan.md)
- [Shared Library Plan](notes/plan/02-shared-library/plan.md)

## Implementation Notes

*This section will be populated during the implementation phase with findings, decisions, and notes discovered during development.*

## Next Steps

1. **Review this planning document** - Ensure all design decisions are sound
1. **Begin Test Phase** (Issue #294) - Write comprehensive tests following TDD
1. **Begin Implementation** (Issue #295) - Implement metrics following the design
1. **Begin Packaging** (Issue #296) - Integrate metrics into shared library
1. **Cleanup Phase** (Issue #297) - Refactor and finalize after parallel phases complete

---

**Planning Phase Status**: ✅ Complete

**Created**: 2025-11-15

**Last Updated**: 2025-11-15
