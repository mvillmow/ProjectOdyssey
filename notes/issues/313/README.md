# Issue #313: [Plan] Validation Loop - Design and Documentation

## Objective

Design and document the validation loop for evaluating model performance on held-out data without updating weights.
This loop performs forward passes, computes metrics, and aggregates results to assess model generalization during and
after training.

## Deliverables

- **Validation metrics computation**: Loss, accuracy, and other evaluation metrics
- **Aggregated statistics**: Per-batch and overall results across validation set
- **Callback invocations**: Event hooks for validation start, batch, and end events
- **Memory-efficient implementation**: No gradient computation or storage during validation
- **Comprehensive design documentation**: Architecture, API contracts, and implementation guidelines

## Success Criteria

- [ ] Loop runs without computing gradients (uses no-gradient context)
- [ ] Metrics aggregate correctly across batches (proper averaging/summing)
- [ ] Results match expected validation behavior (matches reference implementations)
- [ ] Callbacks fire at appropriate points (start, batch, end events)
- [ ] Memory usage is efficient (no gradient storage, minimal overhead)
- [ ] API contracts are clearly defined and documented
- [ ] Design decisions are documented with rationale
- [ ] Integration points with trainer interface are specified

## Design Decisions

### 1. Gradient Computation

**Decision**: Disable gradient computation during validation loop

**Rationale**:

- Validation is inference-only, no weight updates needed
- Saves significant memory (no backward graph storage)
- Improves performance by skipping gradient calculations
- Standard practice in deep learning frameworks

**Implementation Approach**:

- Use context manager to disable gradients (equivalent to PyTorch's `torch.no_grad()`)
- Ensure model is in evaluation mode before validation
- Document memory savings and performance benefits

### 2. Model Evaluation Mode

**Decision**: Require model to be in evaluation mode during validation

**Rationale**:

- Disables dropout layers (use all neurons for stable predictions)
- Batch normalization uses running statistics instead of batch statistics
- Ensures deterministic and reproducible validation results
- Matches expected behavior during inference

**Implementation Approach**:

- Add explicit check/assertion for evaluation mode
- Document expected model state before validation
- Provide clear error messages if model is in training mode

### 3. Metrics Aggregation

**Decision**: Support both per-batch and aggregated metrics

**Rationale**:

- Per-batch metrics useful for debugging and variance analysis
- Aggregated metrics provide overall performance summary
- Allows flexible analysis of validation results
- Matches standard ML framework patterns

**Implementation Approach**:

- Compute metrics for each batch (loss, accuracy, etc.)
- Aggregate using appropriate statistics (mean for loss, weighted average for accuracy)
- Return both per-batch and aggregated results
- Handle different batch sizes correctly (weighted averaging)

### 4. Validation Modes

**Decision**: Support both full validation and subset validation

**Rationale**:

- Full validation provides complete accuracy assessment
- Subset validation enables faster iteration during development
- Subset validation useful for large datasets where full validation is expensive
- Common pattern in production ML systems

**Implementation Approach**:

- Add configuration parameter for number of validation batches
- Default to full validation (None or -1)
- Allow specifying exact number of batches for subset validation
- Document trade-offs between speed and accuracy

### 5. Callback Integration

**Decision**: Provide callback hooks at key validation events

**Rationale**:

- Enables extensibility without modifying core loop
- Supports logging, monitoring, and custom behavior
- Matches callback pattern from training loop
- Allows users to inject custom logic (early stopping, logging, etc.)

**Implementation Approach**:

- `on_validation_start()`: Called before validation begins
- `on_validation_batch()`: Called after each batch
- `on_validation_end()`: Called after validation completes
- Pass relevant context (metrics, batch results) to callbacks

### 6. API Design

**Decision**: Validation loop as method on Trainer interface

**Rationale**:

- Keeps validation logic encapsulated with training logic
- Allows sharing state (model, device, configuration)
- Simplifies user interface (single trainer object)
- Standard pattern in ML frameworks

**Interface**:

```mojo
fn validate(
    inout self,
    validation_loader: DataLoader,
    metrics: List[Metric],
    num_batches: Optional[Int] = None
) -> ValidationResults:
    """Run validation loop on validation dataset.

    Args:
        validation_loader: DataLoader for validation data
        metrics: List of metrics to compute (e.g., accuracy, loss)
        num_batches: Optional limit on number of batches (None = full validation)

    Returns:
        ValidationResults containing per-batch and aggregated metrics
    """
    pass
```

### 7. Error Handling

**Decision**: Fail fast with clear error messages for common issues

**Rationale**:

- Helps users quickly identify configuration issues
- Prevents silent failures or incorrect results
- Improves debugging experience
- Reduces support burden

**Key Checks**:

- Model is in evaluation mode (not training mode)
- Validation data loader is not empty
- Metrics are compatible with model outputs
- Device compatibility (model and data on same device)

### 8. Memory Management

**Decision**: Minimize memory footprint during validation

**Rationale**:

- Validation often runs on large datasets
- Memory is limited, especially on GPUs
- No need to store intermediate activations or gradients
- Enables larger batch sizes for faster validation

**Implementation Approach**:

- Use no-gradient context throughout
- Avoid storing unnecessary intermediate results
- Use in-place operations where possible
- Document memory requirements and optimizations

## Architecture

### Component Structure

```text
Trainer (BaseTrainer)
├── validate() method
│   ├── Gradient context management (disable gradients)
│   ├── Model mode management (ensure evaluation mode)
│   ├── Batch iteration (with optional limit)
│   ├── Metrics computation (per-batch)
│   ├── Results aggregation (overall)
│   └── Callback invocations (start, batch, end)
└── ValidationResults
    ├── per_batch_metrics: Dict[String, List[Float64]]
    ├── aggregated_metrics: Dict[String, Float64]
    ├── num_batches: Int
    └── num_samples: Int
```

### Integration Points

1. **Trainer Interface**: Validation loop is a method on the BaseTrainer
2. **Data Loaders**: Uses DataLoader interface for batch iteration
3. **Metrics**: Uses Metric interface for computation and aggregation
4. **Callbacks**: Integrates with callback system for extensibility
5. **Model**: Requires model to implement evaluation mode toggle

### Data Flow

```text
1. validate() called with validation_loader and metrics
2. Set model to evaluation mode
3. Enter no-gradient context
4. Trigger on_validation_start() callback
5. For each batch (up to num_batches if specified):
   a. Load batch from validation_loader
   b. Forward pass through model
   c. Compute metrics for batch
   d. Store per-batch results
   e. Trigger on_validation_batch() callback
6. Aggregate metrics across all batches
7. Trigger on_validation_end() callback
8. Return ValidationResults
```

## API Contracts

### Trainer.validate()

**Signature**:

```mojo
fn validate(
    inout self,
    validation_loader: DataLoader,
    metrics: List[Metric],
    num_batches: Optional[Int] = None
) raises -> ValidationResults
```

**Preconditions**:

- Model must be initialized and loaded
- Validation data loader must not be empty
- Metrics must be compatible with model outputs
- Model and data must be on compatible devices

**Postconditions**:

- Model remains in evaluation mode
- ValidationResults contains valid metrics for all batches processed
- All callbacks have been invoked appropriately
- No gradients have been computed or stored

**Exceptions**:

- `ModelNotInEvalModeError`: Model is not in evaluation mode
- `EmptyDataLoaderError`: Validation data loader has no batches
- `MetricCompatibilityError`: Metric not compatible with model outputs
- `DeviceMismatchError`: Model and data on different devices

### ValidationResults

**Structure**:

```mojo
struct ValidationResults:
    """Results from validation loop."""

    var per_batch_metrics: Dict[String, List[Float64]]
    """Metrics for each batch (metric_name -> list of values)."""

    var aggregated_metrics: Dict[String, Float64]
    """Aggregated metrics across all batches (metric_name -> value)."""

    var num_batches: Int
    """Number of batches processed."""

    var num_samples: Int
    """Total number of samples processed."""

    fn __init__(inout self):
        """Initialize empty validation results."""
        pass

    fn add_batch_metrics(inout self, metrics: Dict[String, Float64]):
        """Add metrics from a single batch."""
        pass

    fn aggregate(inout self):
        """Compute aggregated metrics from per-batch metrics."""
        pass
```

### Callbacks

**Validation Start**:

```mojo
fn on_validation_start(inout self, trainer: Trainer):
    """Called before validation loop begins.

    Args:
        trainer: Trainer instance running validation
    """
    pass
```

**Validation Batch**:

```mojo
fn on_validation_batch(
    inout self,
    trainer: Trainer,
    batch_idx: Int,
    metrics: Dict[String, Float64]
):
    """Called after each validation batch.

    Args:
        trainer: Trainer instance running validation
        batch_idx: Index of current batch
        metrics: Metrics computed for this batch
    """
    pass
```

**Validation End**:

```mojo
fn on_validation_end(
    inout self,
    trainer: Trainer,
    results: ValidationResults
):
    """Called after validation loop completes.

    Args:
        trainer: Trainer instance that ran validation
        results: Final validation results
    """
    pass
```

## Implementation Guidelines

### Mojo Language Patterns

Following [mojo-language-review-specialist.md](../../../../.claude/agents/mojo-language-review-specialist.md):

1. **Use `fn` over `def`**: Validation loop is performance-critical, use strict `fn`
2. **Memory safety**: Use `borrowed` for read-only model access, `inout` for trainer state
3. **SIMD optimization**: Consider SIMD for metrics aggregation if processing large result arrays
4. **Ownership**: Clear ownership of ValidationResults (returned owned to caller)

### Error Handling Strategy

1. **Precondition validation**: Check all preconditions at start of validate()
2. **Clear error messages**: Include context (which check failed, expected vs actual state)
3. **Early exit**: Fail fast if preconditions not met
4. **Resource cleanup**: Ensure model mode restored even on error (use defer or try/finally equivalent)

### Testing Strategy

See related issue #314 for test implementation. Key test categories:

1. **Functional tests**: Correct metrics computation and aggregation
2. **Mode tests**: Model in evaluation mode, gradients disabled
3. **Callback tests**: Callbacks invoked at correct times with correct data
4. **Edge cases**: Empty validation set, single batch, very large datasets
5. **Integration tests**: With real model and data loader

### Performance Considerations

1. **Batch size**: Larger batches during validation (no gradient memory overhead)
2. **Device utilization**: Maximize GPU utilization with efficient batch processing
3. **Metrics computation**: Compute metrics efficiently (vectorized operations)
4. **Memory footprint**: Monitor and optimize memory usage

## References

### Source Plan

- [notes/plan/02-shared-library/02-training-utils/01-base-trainer/03-validation-loop/plan.md](../../../plan/02-shared-library/02-training-utils/01-base-trainer/03-validation-loop/plan.md)

### Parent Component

- [notes/plan/02-shared-library/02-training-utils/01-base-trainer/plan.md](../../../plan/02-shared-library/02-training-utils/01-base-trainer/plan.md)

### Related Issues

- **#314**: [Test] Validation Loop - Test implementation following TDD
- **#315**: [Impl] Validation Loop - Core implementation
- **#316**: [Package] Validation Loop - Integration and packaging
- **#317**: [Cleanup] Validation Loop - Cleanup and finalization

### Dependencies

This component depends on:

- Trainer interface (#310-312) - Base interface for trainer
- Training loop (#307-309) - Shared patterns and utilities
- Data loaders (shared library) - Batch iteration
- Metrics system (shared library) - Metrics computation

## Implementation Notes

*This section will be filled during implementation phases (issues #314-317).*

### Design Changes

*Document any changes to the design made during implementation.*

### Discovered Issues

*Document issues or challenges discovered during implementation.*

### Performance Metrics

*Document performance benchmarks and optimizations.*

### Lessons Learned

*Document lessons learned for future validation loop implementations.*
