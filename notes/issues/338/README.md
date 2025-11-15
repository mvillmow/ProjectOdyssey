# Issue #338: [Plan] LR Schedulers - Design and Documentation

## Objective

Design and document learning rate scheduling strategies for dynamically adjusting learning rates during training, including step decay, cosine annealing, and warmup schedulers to improve convergence and final model performance.

## Deliverables

- Step scheduler for discrete learning rate drops at specified intervals
- Cosine scheduler for smooth annealing following a cosine curve
- Warmup scheduler for gradual training start with linear/exponential increase
- Comprehensive API specifications and design documentation
- Architecture design for scheduler composition and integration

## Success Criteria

- [ ] Schedulers correctly adjust learning rates over time
- [ ] Step scheduler reduces rate at specified intervals
- [ ] Cosine scheduler follows proper annealing curve
- [ ] Warmup scheduler gradually increases to target rate
- [ ] All child components (step, cosine, warmup) are documented
- [ ] Scheduler composition patterns are defined
- [ ] Integration with training loop is specified
- [ ] State management for serialization/resumption is designed
- [ ] API contracts and interfaces are documented

## Design Decisions

### 1. Scheduler Architecture

**Common Interface**: All schedulers will implement a common interface with:

- `get_lr(current_step: Int) -> Float64` - Calculate learning rate for current step
- `state_dict() -> Dict` - Serialize scheduler state
- `load_state_dict(state: Dict)` - Restore scheduler state
- Support for both step-based and epoch-based scheduling

**Rationale**: Common interface enables scheduler composition, testing, and integration with training loops.

### 2. Step Scheduler Design

**Formula**: `lr = initial_lr * (gamma ** (current_step // step_size))`

**Configuration**:

- `initial_lr: Float64` - Starting learning rate
- `step_size: Int` - Interval between rate reductions
- `gamma: Float64` - Multiplicative decay factor (typically 0.1)

**Typical Usage**: `step_size=30`, `gamma=0.1` for discrete learning rate drops every 30 epochs.

**Rationale**: Simple, effective strategy widely used in deep learning. Easy to understand and configure.

### 3. Cosine Scheduler Design

**Formula**: `lr = min_lr + (max_lr - min_lr) * (1 + cos(pi * current_step / total_steps)) / 2`

**Configuration**:

- `initial_lr: Float64` - Maximum learning rate at start
- `total_steps: Int` - Total training steps/epochs
- `min_lr: Float64` - Minimum learning rate (optional, default 0)

**Rationale**: Smooth, continuous decay without sudden drops. Often produces better final performance than step decay. Natural progression from fast to slow learning.

### 4. Warmup Scheduler Design

**Linear Warmup Formula**: `lr = target_lr * (current_step / warmup_steps)`

**Exponential Warmup Formula**: `lr = target_lr * ((current_step / warmup_steps) ** 2)`

**Configuration**:

- `target_lr: Float64` - Final learning rate after warmup
- `warmup_steps: Int` - Duration of warmup phase (typically 1000-10000 steps)
- `strategy: String` - "linear" or "exponential"

**Rationale**: Stabilizes training at the start, especially critical for large learning rates or batch sizes. Prevents early instability and gradient explosion.

### 5. Scheduler Composition

**Pattern**: Warmup + Decay (e.g., warmup then cosine annealing)

**Implementation Approach**:

- Support chaining schedulers sequentially
- Warmup completes first, then transitions to decay scheduler
- Example: Linear warmup for 1000 steps, then cosine decay for remaining training

**Rationale**: Combining warmup with decay schedulers (especially cosine) is a best practice in modern deep learning. Provides stable start and smooth convergence.

### 6. State Management

**Requirements**:

- Save current step/epoch counter
- Save configuration parameters
- Support training resumption from checkpoints

**Format**: Dictionary-based state for easy serialization

**Rationale**: Training interruptions are common. State management enables seamless resumption without loss of schedule progress.

### 7. Integration with Optimizer

**Approach**: Schedulers update optimizer's learning rate parameter

**Interface**: `update_optimizer(optimizer: Optimizer, lr: Float64)`

**Rationale**: Clean separation of concerns. Schedulers calculate rates, optimizers apply them. Enables testing schedulers independently.

### 8. Mojo Implementation Considerations

**Type Safety**:

- Use `Float64` for learning rates to avoid precision loss
- Use `Int` for step/epoch counters
- Strong typing for configuration structs

**Memory Safety**:

- Schedulers are lightweight (config + counter state)
- Use `borrowed` for optimizer updates
- Use `owned` for state serialization

**Performance**:

- Learning rate calculation is not performance-critical (once per step)
- Prioritize clarity over micro-optimization
- Simple formulas enable inlining

**Rationale**: Mojo's type system provides compile-time guarantees. Schedulers are simple enough that performance is not a concern.

## References

### Source Plan

- [LR Schedulers Plan](/home/mvillmow/ml-odyssey-manual/notes/plan/02-shared-library/02-training-utils/02-lr-schedulers/plan.md)
- [Step Scheduler Plan](/home/mvillmow/ml-odyssey-manual/notes/plan/02-shared-library/02-training-utils/02-lr-schedulers/01-step-scheduler/plan.md)
- [Cosine Scheduler Plan](/home/mvillmow/ml-odyssey-manual/notes/plan/02-shared-library/02-training-utils/02-lr-schedulers/02-cosine-scheduler/plan.md)
- [Warmup Scheduler Plan](/home/mvillmow/ml-odyssey-manual/notes/plan/02-shared-library/02-training-utils/02-lr-schedulers/03-warmup-scheduler/plan.md)

### Related Issues

- Issue #338 (this issue) - Planning phase
- Issue #339 - Test phase (write tests for schedulers)
- Issue #340 - Implementation phase (implement schedulers in Mojo)
- Issue #341 - Packaging phase (integrate schedulers)
- Issue #342 - Cleanup phase (refactor and finalize)

### Additional Documentation

- [Training Utils Architecture](/home/mvillmow/ml-odyssey-manual/notes/plan/02-shared-library/02-training-utils/plan.md)
- [Mojo Language Guidelines](/home/mvillmow/ml-odyssey-manual/.claude/agents/mojo-language-review-specialist.md)
- [5-Phase Development Workflow](/home/mvillmow/ml-odyssey-manual/notes/review/README.md)

## Implementation Notes

*This section will be filled during subsequent phases (Test, Implementation, Packaging, Cleanup) with findings and decisions made during development.*

### Phase-Specific Notes

**Test Phase (Issue #339)**:

- TBD

**Implementation Phase (Issue #340)**:

- TBD

**Packaging Phase (Issue #341)**:

- TBD

**Cleanup Phase (Issue #342)**:

- TBD
