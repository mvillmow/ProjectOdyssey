# Issue #328: [Plan] Cosine Scheduler - Design and Documentation

## Objective

Implement cosine annealing learning rate scheduler that smoothly decreases the learning rate following a cosine curve, providing gradual, continuous decay without sudden drops for better final performance than step decay.

## Deliverables

- Cosine annealing scheduler specification
- API design for configurable minimum learning rate
- Support for both step-based and epoch-based scheduling
- State management design for serialization and resumption
- Mathematical specification of cosine decay curve
- Integration design with training loop

## Success Criteria

- [ ] Learning rate follows cosine curve correctly
- [ ] Decay is smooth and continuous
- [ ] Minimum learning rate is respected
- [ ] State can be saved and restored
- [ ] API design supports both step-based and epoch-based scheduling
- [ ] Composability with warmup scheduler documented
- [ ] Integration with training loop specified

## Design Decisions

### Mathematical Foundation

**Cosine Annealing Formula**:

```text
lr = min_lr + (max_lr - min_lr) * (1 + cos(pi * current_step / total_steps)) / 2
```

**Key Properties**:

- Smooth, continuous decay from `max_lr` to `min_lr`
- Follows cosine curve: starts fast, slows toward minimum
- No sudden drops (unlike step decay)
- Predictable behavior over training schedule
- Often leads to better final performance

### API Design Considerations

**Required Parameters**:

- `initial_lr`: Maximum learning rate (start of schedule)
- `total_steps`: Total training steps or epochs
- `current_step`: Current training step or epoch

**Optional Parameters**:

- `min_lr`: Minimum learning rate (default: 0.0)
- `mode`: Step-based or epoch-based scheduling (default: step-based)

**Output**:

- Current learning rate value following cosine curve

### Scheduler Modes

**Step-Based Scheduling**:

- Updates every training step (batch)
- Finer-grained control over learning rate
- Common for large datasets or long training runs
- Example: `total_steps = num_epochs * steps_per_epoch`

**Epoch-Based Scheduling**:

- Updates every epoch
- Coarser-grained control
- Simpler to reason about
- Common for smaller datasets or shorter training runs

### State Management

**Serializable State**:

- `initial_lr`: Starting learning rate
- `min_lr`: Minimum learning rate
- `total_steps`: Total schedule duration
- `current_step`: Current position in schedule
- `mode`: Step-based or epoch-based

**Use Cases**:

- Training resumption after interruption
- Checkpointing and recovery
- Distributed training synchronization

### Composability with Warmup

**Common Pattern**: Warmup + Cosine Annealing

- Phase 1: Linear warmup (0 to initial_lr over warmup_steps)
- Phase 2: Cosine annealing (initial_lr to min_lr over remaining steps)

**Benefits**:

- Stable training start (warmup prevents early divergence)
- Smooth final convergence (cosine annealing)
- Best of both strategies

**Implementation Note**: Design scheduler to compose cleanly with warmup scheduler (see issue #340 for warmup implementation).

### Integration with Training Loop

**Update Frequency**:

- Step-based: Call `scheduler.step()` after each batch
- Epoch-based: Call `scheduler.step()` after each epoch

**Learning Rate Application**:

- Scheduler calculates new learning rate
- Training loop applies to optimizer
- Common pattern: `optimizer.lr = scheduler.get_lr()`

**State Persistence**:

- Save scheduler state with model checkpoints
- Restore state when resuming training
- Ensures learning rate continuity

### Mojo Implementation Considerations

**Type Safety**:

- Use `Float64` for learning rate calculations (precision)
- Strong typing for step counts and parameters
- Avoid floating point precision issues

**SIMD Optimization**:

- Cosine calculation can leverage SIMD (if batched)
- Most likely not needed for single value calculation
- Focus on correctness first, optimize if needed

**Memory Management**:

- Lightweight state (few scalar values)
- No dynamic allocations in hot path
- Suitable for inline storage in training loop

**Interface Design**:

- Follow Mojo idioms: `fn` over `def`, `owned`/`borrowed` parameters
- Make scheduler a struct with methods
- Support serialization via trait implementation

## References

- [Source Plan](/home/mvillmow/ml-odyssey-manual/notes/plan/02-shared-library/02-training-utils/02-lr-schedulers/02-cosine-scheduler/plan.md)
- [Parent Plan: LR Schedulers](/home/mvillmow/ml-odyssey-manual/notes/plan/02-shared-library/02-training-utils/02-lr-schedulers/plan.md)
- Related Issues:
  - #329: [Test] Cosine Scheduler - Test Implementation
  - #330: [Impl] Cosine Scheduler - Implementation
  - #331: [Package] Cosine Scheduler - Integration and Packaging
  - #332: [Cleanup] Cosine Scheduler - Cleanup and Finalization
  - #324: [Impl] Step Scheduler (preceding scheduler implementation)
  - #340: [Impl] Warmup Scheduler (composability with warmup)

## Implementation Notes

**Status**: Planning in progress

This section will be updated as implementation progresses with:

- Design decisions made during implementation
- API changes or refinements
- Performance characteristics discovered
- Integration challenges and solutions
- Lessons learned for other scheduler implementations

**Ready for**: Issue #329 (Test Phase) after planning approval
