# Issue #323: [Plan] Step Scheduler - Design and Documentation

## Objective

Design and document the step decay learning rate scheduler that reduces the learning rate by a fixed factor at
specified intervals (steps or epochs). This simple but effective strategy helps models converge by reducing the
learning rate as training progresses.

## Deliverables

- Current learning rate based on step count
- Schedule update at each training step
- State for serialization and resumption
- API specification and interface design
- Implementation guidelines document

## Success Criteria

- [ ] Learning rate decreases at correct intervals
- [ ] Decay factor applies correctly
- [ ] Both step and epoch modes work
- [ ] State can be saved and restored
- [ ] API contracts clearly defined
- [ ] Architecture design documented

## Design Decisions

### Core Architecture

**Scheduler Type**: Step Decay (Discrete)

The step scheduler implements a simple but effective learning rate reduction strategy:

```text
lr(t) = lr_initial * (gamma ^ floor(t / step_size))
```text

Where:

- `lr_initial`: Initial learning rate
- `gamma`: Decay factor (typically 0.1)
- `step_size`: Number of steps/epochs between reductions
- `t`: Current step/epoch counter

**Example Schedule** (lr=0.1, step_size=30, gamma=0.1):

- Steps 0-29: lr = 0.1
- Steps 30-59: lr = 0.01
- Steps 60-89: lr = 0.001
- Steps 90+: lr = 0.0001

### Mode Support

### Dual Mode Operation

1. **Step-based mode**: Counts individual training steps (batches)
1. **Epoch-based mode**: Counts complete passes through dataset

Both modes use the same decay formula but differ in what constitutes a "step" in the calculation.

### State Management

### Serialization Requirements

The scheduler must support training resumption by saving/restoring:

- Current step/epoch counter
- Initial learning rate
- Decay configuration (step_size, gamma)
- Current computed learning rate
- Mode (step vs epoch)

This enables:

- Training checkpointing
- Distributed training synchronization
- Experiment reproducibility

### Integration Points

### Optimizer Integration

The scheduler must work with the optimizer's learning rate parameter:

1. Scheduler computes new learning rate based on step count
1. Optimizer applies the learning rate to parameter updates
1. Updates happen before/after optimizer step (configurable)

### Common Integration Pattern

```mojo
optimizer.step()  # Update parameters
scheduler.step()  # Update learning rate for next iteration
```text

### Configuration Defaults

### Recommended Default Values

- `step_size`: 30 (reduces every 30 steps/epochs)
- `gamma`: 0.1 (10x reduction per interval)
- `mode`: "step" (step-based by default)

These values are based on common practice in literature (e.g., ResNet training).

### API Design Principles

### Interface Requirements

1. **Simple Construction**: Initialize with minimal required parameters
1. **Explicit Updates**: Clear `step()` method for scheduler updates
1. **State Access**: `get_lr()` method to query current learning rate
1. **State Persistence**: `state_dict()` and `load_state_dict()` for serialization
1. **Type Safety**: Leverage Mojo's type system for compile-time safety

### Example API Usage

```mojo
# Initialize scheduler
var scheduler = StepScheduler(
    initial_lr=0.1,
    step_size=30,
    gamma=0.1,
    mode="step"
)

# Training loop
for epoch in range(num_epochs):
    for batch in data_loader:
        loss = model.forward(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()  # Update learning rate

        current_lr = scheduler.get_lr()
```text

### Implementation Strategy

**Phased Implementation** (aligned with 5-phase workflow):

1. **Plan Phase** (Issue #323): This document - architecture and API design
1. **Test Phase** (Issue #324): TDD implementation with comprehensive test coverage
1. **Implementation Phase** (Issue #325): Core scheduler logic in Mojo
1. **Packaging Phase** (Issue #326): Integration with training utilities
1. **Cleanup Phase** (Issue #327): Optimization and finalization

### Key Considerations

### Numerical Stability

- Use integer step counter to avoid floating-point accumulation errors
- Compute decay exponent as `floor(step / step_size)` for exact interval boundaries
- Store initial_lr separately from current_lr to prevent drift

### Edge Cases

- `step_size = 0`: Invalid configuration (raise error)
- `gamma <= 0` or `gamma >= 1`: Invalid decay factor (raise error)
- Negative step counter: Invalid state (raise error)
- First step (step=0): Should use initial_lr without decay

### Performance

- Scheduler update is lightweight (simple multiplication)
- No SIMD optimization needed (single scalar operation)
- State serialization should be efficient (small state size)

## References

### Source Plan

- [Step Scheduler Plan](notes/plan/02-shared-library/02-training-utils/02-lr-schedulers/01-step-scheduler/plan.md)
- [Parent LR Schedulers Plan](notes/plan/02-shared-library/02-training-utils/02-lr-schedulers/plan.md)

### Related Issues

- Issue #324: [Test] Step Scheduler - Test Implementation
- Issue #325: [Impl] Step Scheduler - Core Implementation
- Issue #326: [Package] Step Scheduler - Integration
- Issue #327: [Cleanup] Step Scheduler - Finalization

### Related Components

- Optimizer integration (to be defined in training utils)
- Training loop integration (to be defined in training orchestration)
- State serialization utilities (to be defined in shared library)

## Implementation Notes

_This section will be populated during subsequent phases (Test, Implementation, Packaging, Cleanup) with:_

- Discoveries made during implementation
- API adjustments or refinements
- Integration challenges and solutions
- Performance observations
- Edge cases encountered in testing

_Initially empty - to be filled as work progresses._
