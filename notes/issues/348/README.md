# Issue #348: [Plan] Early Stopping - Design and Documentation

## Objective

Design and document the early stopping callback component that terminates training when validation performance stops
improving. This callback prevents overfitting by monitoring a validation metric and stopping after a patience period
without improvement, saving time and computing resources.

## Deliverables

- **Architecture Design**: Detailed design of early stopping callback structure and integration points
- **API Specification**: Complete interface definition including configuration parameters and hook methods
- **Integration Design**: How early stopping integrates with the base trainer and callback system
- **Configuration Documentation**: Specification of all configuration parameters (metric, patience, delta, mode)
- **Design Documentation**: Comprehensive planning documentation at `/notes/issues/348/README.md`

## Success Criteria

- [ ] Architecture design document completed with clear callback structure
- [ ] API specification defines all required methods and parameters
- [ ] Integration design shows how early stopping hooks into training loop
- [ ] Configuration parameters documented with types, defaults, and valid ranges
- [ ] Design addresses both minimize and maximize metric modes
- [ ] Best model restoration mechanism is clearly specified
- [ ] Logging and metadata output format is defined
- [ ] Design review completed and approved

## Design Decisions

### 1. Callback Architecture

**Decision**: Implement as a stateful callback with hook methods for epoch-level events.

**Rationale**:

- Follows the established callback pattern from parent component design
- Maintains clean separation from core training logic
- Allows easy composition with other callbacks (checkpointing, logging)
- Stateful design enables tracking of best metric, patience counter, and best weights

**Key Components**:

- Metric tracker (stores best value and epoch)
- Patience counter (tracks epochs without improvement)
- Best model state cache (optional, for restoration)
- Stopping flag (signals training termination)

### 2. Monitoring Strategy

**Decision**: Monitor metrics at epoch boundaries using validation results.

**Rationale**:

- Validation metrics are computed once per epoch
- Epoch-level granularity provides stable signal (less noisy than batch-level)
- Aligns with typical early stopping usage in research papers
- Simplifies implementation and state management

**Hook Points**:

- `on_validation_end(epoch, metrics)` - Primary hook for metric monitoring
- `on_train_begin()` - Initialize state
- `should_stop()` - Query method for trainer

### 3. Improvement Detection

**Decision**: Use configurable minimum delta threshold with mode-aware comparison.

**Parameters**:

- `metric`: String name of metric to monitor (e.g., "val_loss", "val_accuracy")
- `mode`: Enum ("minimize" or "maximize") - whether lower/higher is better
- `min_delta`: Float threshold for considering improvement (default: 0.0)
- `patience`: Int number of epochs without improvement before stopping (default: 10)

**Logic**:

```text
if mode == "minimize":
    improved = (current_metric < best_metric - min_delta)
else:  # maximize
    improved = (current_metric > best_metric + min_delta)
```

**Rationale**:

- Min delta prevents stopping due to noise in validation metrics
- Mode parameter makes the callback applicable to any metric type
- Simple, well-understood logic from Keras/PyTorch implementations

### 4. Best Model Restoration

**Decision**: Optional best model restoration via configuration flag.

**Parameters**:

- `restore_best_weights`: Bool flag (default: True)

**Mechanism**:

- When improvement detected, store deep copy of model weights
- On early stopping trigger, restore weights from best epoch
- Only cache weights when restoration is enabled (memory efficiency)

**Rationale**:

- Common use case is training until validation plateaus, then using best weights
- Optional to support cases where final weights are preferred
- Memory trade-off controlled by configuration flag

### 5. Logging and Metadata

**Decision**: Emit structured logging at key events with metadata export.

**Log Events**:

- Improvement detected: "Epoch {epoch}: {metric} improved from {old_val} to {new_val}"
- Patience increment: "Epoch {epoch}: No improvement ({counter}/{patience})"
- Early stopping triggered: "Early stopping triggered at epoch {epoch}. Best {metric}: {best_val} at epoch {best_epoch}"
- Best weights restored: "Restored model weights from epoch {best_epoch}"

**Metadata Output**:

```python
{
    "stopped_epoch": int,
    "best_epoch": int,
    "best_value": float,
    "metric_name": str,
    "patience_used": int,
    "patience_limit": int
}
```

**Rationale**:

- Clear logging helps users understand stopping decisions
- Metadata enables reproducibility and analysis
- Structured format supports automated processing

### 6. Edge Cases and Error Handling

**Decisions**:

- **Missing metric**: Raise clear error if monitored metric not in validation results
- **Invalid mode**: Validate mode parameter in constructor (only "minimize" or "maximize")
- **Negative patience**: Validate patience > 0 in constructor
- **No validation**: Skip monitoring if `on_validation_end` not called (warn once)

**Rationale**:

- Fail fast with clear errors rather than silent failures
- Validation at construction time prevents runtime surprises
- Warning for missing validation helps debug integration issues

### 7. Type System and Memory Management

**Decision**: Use Mojo structs with owned/borrowed parameters following project patterns.

**Key Patterns**:

```mojo
struct EarlyStoppingCallback(Callback):
    var metric_name: String
    var mode: String  # "minimize" or "maximize"
    var patience: Int
    var min_delta: Float64
    var restore_best_weights: Bool

    var best_value: Float64
    var best_epoch: Int
    var wait_count: Int
    var stopped_epoch: Int
    var best_weights: Optional[ModelWeights]  # Cached when restore enabled

    fn __init__(inout self,
                metric: String,
                mode: String = "minimize",
                patience: Int = 10,
                min_delta: Float64 = 0.0,
                restore_best_weights: Bool = True):
        # Validation and initialization
        pass

    fn on_train_begin(inout self) -> None:
        # Reset state
        pass

    fn on_validation_end(inout self, epoch: Int, metrics: Dict[String, Float64]) -> None:
        # Monitor metric and update state
        pass

    fn should_stop(self) -> Bool:
        # Return whether training should terminate
        pass

    fn get_metadata(self) -> Dict[String, Variant]:
        # Return structured metadata
        pass
```

**Rationale**:

- Struct-based design for value semantics and performance
- Owned strings and state for clear ownership
- Optional for conditionally-cached weights
- Borrowed parameters for efficient callbacks
- Follows Mojo best practices from project standards

## References

### Source Plans

- **Component Plan**: [/notes/plan/02-shared-library/02-training-utils/03-callbacks/02-early-stopping/plan.md](/notes/plan/02-shared-library/02-training-utils/03-callbacks/02-early-stopping/plan.md)
- **Parent Plan (Callbacks)**: [/notes/plan/02-shared-library/02-training-utils/03-callbacks/plan.md](/notes/plan/02-shared-library/02-training-utils/03-callbacks/plan.md)
- **Grandparent Plan (Training Utils)**: [/notes/plan/02-shared-library/02-training-utils/plan.md](/notes/plan/02-shared-library/02-training-utils/plan.md)

### Related Issues

- **Issue #348**: [Plan] Early Stopping - Design and Documentation (this issue)
- **Issue #349**: [Test] Early Stopping - Test Implementation
- **Issue #350**: [Impl] Early Stopping - Implementation
- **Issue #351**: [Package] Early Stopping - Integration
- **Issue #352**: [Cleanup] Early Stopping - Finalization

### Sibling Components

- **Checkpointing Callback**: Issues #343-347 (model state persistence)
- **Logging Callback**: Issues #353-357 (progress tracking)
- **Base Trainer**: Issues #308-312 (integration point for callbacks)

### Documentation

- **Agent Guidelines**: [/agents/README.md](/agents/README.md)
- **Delegation Rules**: [/agents/delegation-rules.md](/agents/delegation-rules.md)
- **Mojo Language Patterns**: [/.claude/agents/mojo-language-review-specialist.md](/.claude/agents/mojo-language-review-specialist.md)

## Implementation Notes

(To be filled during Test, Implementation, and Packaging phases)

### Test Phase (Issue #349)

Notes from test implementation will be added here

### Implementation Phase (Issue #350)

Notes from implementation will be added here

### Packaging Phase (Issue #351)

Notes from packaging and integration will be added here

### Cleanup Phase (Issue #352)

Notes from cleanup and finalization will be added here

## Appendix: Research and Best Practices

### Common Patterns from ML Frameworks

**Keras EarlyStopping**:

- Monitors single metric with mode and min_delta
- Patience counter for consecutive epochs without improvement
- Optional restore_best_weights flag
- Baseline parameter for minimum acceptable value

**PyTorch Lightning EarlyStopping**:

- Similar to Keras with additional features
- Strict mode (stop on first metric increase for minimize)
- Check_on_train_epoch_end for training metric monitoring
- Divergence threshold for detecting training instabilities

**Design Takeaways**:

- Core interface (metric, mode, patience, min_delta) is well-established
- Best weights restoration is a standard feature
- Simplicity is valued - advanced features are often unused
- Clear logging is essential for debugging training issues

### Typical Configuration Values

From literature review and common practice:

- **Patience**: 5-20 epochs (smaller for small datasets, larger for large datasets)
- **Min delta**: 0.0001-0.001 (scale with metric magnitude)
- **Mode**: Almost always "minimize" for loss, "maximize" for accuracy/F1
- **Restore best**: Usually True (prevents overfitting)

### Integration with Training Loop

Expected trainer integration pattern:

```mojo
fn train(inout self, epochs: Int, callbacks: List[Callback]):
    for callback in callbacks:
        callback.on_train_begin()

    for epoch in range(epochs):
        # Training step
        train_loss = self._train_epoch()

        # Validation step
        val_metrics = self._validate()

        for callback in callbacks:
            callback.on_validation_end(epoch, val_metrics)

        # Check early stopping
        if any(callback.should_stop() for callback in callbacks):
            print("Early stopping triggered")
            break

    for callback in callbacks:
        callback.on_train_end()
```

This design ensures clean separation between callback logic and trainer implementation.
