# Issue #358: [Plan] Callbacks - Design and Documentation

## Objective

Build a callback system for extending training functionality without modifying core training logic, providing clean extension mechanisms for checkpointing, early stopping, and logging capabilities.

## Deliverables

- Checkpointing callback for model persistence
- Early stopping callback for training termination
- Logging callback for progress tracking

## Success Criteria

- [ ] Checkpointing saves and restores complete training state
- [ ] Early stopping terminates training when appropriate
- [ ] Logging callback provides clear training progress visibility
- [ ] All child plans are completed successfully

## Design Decisions

### Architecture

### Callback Hook System

The callback system will use a clean hook-based architecture that allows extending training workflows without tight coupling to the trainer implementation. Key hook points in the training loop:

- `on_train_begin()` - Initialize callback state before training starts
- `on_epoch_begin(epoch)` - Prepare for new epoch
- `on_batch_begin(batch)` - Pre-batch processing
- `on_batch_end(batch, metrics)` - Post-batch processing and metric logging
- `on_epoch_end(epoch, metrics)` - Epoch completion, checkpoint/early-stop decisions
- `on_train_end()` - Cleanup and final reporting

### Interface Design

Callbacks will implement a common base interface with optional hook methods. The trainer will iterate through registered callbacks and invoke hooks at appropriate points. This design ensures:

- Callbacks can be composed without conflicts
- New callbacks can be added without modifying trainer code
- Callback execution order is predictable and controllable

### Component Specifications

**1. Checkpointing Callback**

Key design decisions:

- **State Completeness**: Save all stateful components (model weights, optimizer state, scheduler state, RNG state)
- **Metadata Inclusion**: Include epoch, step, metrics, timestamp for each checkpoint
- **Retention Strategies**: Support configurable checkpoint retention:
  - Keep last N checkpoints
  - Keep best K by validation metric
  - Automatic cleanup of old checkpoints
- **Format Versioning**: Make checkpoint format version-aware for backward compatibility
- **Trigger Conditions**: Support multiple triggers (every N epochs, best metric improvement)

**2. Early Stopping Callback**

Key design decisions:

- **Metric Monitoring**: Track validation metric over time
- **Patience Counter**: Stop after N epochs without improvement
- **Improvement Detection**: Use minimum delta threshold to ignore noise (e.g., 0.001)
- **Mode Support**: Handle both minimize (loss) and maximize (accuracy) modes
- **Best Model Restoration**: Optionally restore best weights when stopping
- **Logging Integration**: Provide clear stopping reason and statistics

Common patience values: 5-20 epochs depending on dataset size and training dynamics.

**3. Logging Callback**

Key design decisions:

- **Multi-Destination Output**: Support console, file, and structured formats (JSON, CSV)
- **Verbosity Balance**: Console shows summary, file logs capture details
- **Configurable Frequency**: Log every N batches or every epoch
- **Metric Tracking**: Record loss, accuracy, learning rate, custom metrics
- **Summary Reports**: Provide per-epoch statistics and training summaries
- **Log Rotation**: Support file rotation for long training runs
- **Parser-Friendly Format**: Structure logs for easy visualization and analysis

### Key Principles

### Simplicity and Composability

- Callbacks should do one thing well
- Multiple callbacks can be combined without conflicts
- Configuration should be straightforward and intuitive

### Loose Coupling

- Callbacks receive state through method parameters, not direct references
- Trainer doesn't need to know callback implementation details
- Callbacks don't depend on specific trainer implementations

### Extensibility

- Users can create custom callbacks by implementing the base interface
- Common patterns (checkpointing, early stopping, logging) provided out of the box
- Hook points are well-defined and documented

## References

**Source Plan**: [notes/plan/02-shared-library/02-training-utils/03-callbacks/plan.md](notes/plan/02-shared-library/02-training-utils/03-callbacks/plan.md)

**Parent Plan**: [notes/plan/02-shared-library/02-training-utils/plan.md](notes/plan/02-shared-library/02-training-utils/plan.md)

### Child Plans

- [Checkpointing](notes/plan/02-shared-library/02-training-utils/03-callbacks/01-checkpointing/plan.md)
- [Early Stopping](notes/plan/02-shared-library/02-training-utils/03-callbacks/02-early-stopping/plan.md)
- [Logging Callback](notes/plan/02-shared-library/02-training-utils/03-callbacks/03-logging-callback/plan.md)

### Related Issues

- Issue #359: [Test] Callbacks
- Issue #360: [Impl] Callbacks
- Issue #361: [Package] Callbacks
- Issue #362: [Cleanup] Callbacks

## Implementation Notes

This section will be populated during the implementation phases (Test, Implementation, Packaging, Cleanup) with findings, decisions, and lessons learned.
