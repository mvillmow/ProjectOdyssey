# Issue #333: [Plan] Warmup Scheduler - Design and Documentation

## Objective

Design and document a learning rate warmup scheduler that gradually increases the learning rate from a small value to the target value over a specified period, helping stabilize training at the start especially for large learning rates or batch sizes.

## Deliverables

- Learning rate warmup scheduler specification with API design
- Architecture documentation for linear and exponential warmup strategies
- Interface contracts for scheduler state management and serialization
- Integration design for chaining with decay schedulers
- Comprehensive design documentation at `/notes/issues/333/README.md`

## Success Criteria

- [ ] Learning rate increases smoothly during warmup phase
- [ ] Target learning rate is reached exactly at warmup end
- [ ] Both linear and exponential warmup strategies are specified
- [ ] Design supports chaining with decay schedulers (warmup then decay)
- [ ] API design supports both step-based and epoch-based warmup
- [ ] State serialization and resumption design is documented
- [ ] Design documentation is complete and approved

## Design Decisions

### Architecture Overview

The warmup scheduler will be implemented as a standalone component that can be composed with other learning rate schedulers. It will support two primary warmup strategies:

1. **Linear Warmup**: `lr = target_lr * (current_step / warmup_steps)`
2. **Exponential Warmup**: `lr = target_lr * exp(log(init_lr/target_lr) * (1 - current_step/warmup_steps))`

### API Design

#### Core Interface

```mojo
struct WarmupScheduler:
    """Learning rate warmup scheduler with linear or exponential strategies."""

    var target_lr: Float64
    var warmup_steps: Int
    var strategy: WarmupStrategy  # LINEAR or EXPONENTIAL
    var current_step: Int
    var initial_lr: Float64

    fn __init__(inout self, target_lr: Float64, warmup_steps: Int,
                strategy: WarmupStrategy = WarmupStrategy.LINEAR,
                initial_lr: Float64 = 0.0):
        """Initialize warmup scheduler.

        Args:
            target_lr: Target learning rate to reach at end of warmup
            warmup_steps: Number of steps for warmup period
            strategy: Warmup strategy (LINEAR or EXPONENTIAL)
            initial_lr: Initial learning rate (default: 0.0 for linear, target_lr/1000 for exponential)
        """
        pass

    fn get_lr(self) -> Float64:
        """Get current learning rate based on warmup progress.

        Returns:
            Current learning rate during warmup, or target_lr if warmup complete
        """
        pass

    fn step(inout self):
        """Advance warmup scheduler by one step."""
        pass

    fn is_warmup_complete(self) -> Bool:
        """Check if warmup period is complete."""
        pass

    fn get_state(self) -> Dict[String, Variant]:
        """Get scheduler state for serialization."""
        pass

    fn load_state(inout self, state: Dict[String, Variant]):
        """Load scheduler state from serialized data."""
        pass
```

#### Warmup Strategy Enumeration

```mojo
enum WarmupStrategy:
    LINEAR
    EXPONENTIAL
```

### Scheduler Chaining Design

The warmup scheduler will support composition with decay schedulers through a simple interface:

```mojo
struct ChainedScheduler:
    """Compose multiple schedulers sequentially."""

    var warmup: WarmupScheduler
    var main_scheduler: LRScheduler  # Step, Cosine, etc.
    var transition_step: Int

    fn get_lr(self) -> Float64:
        """Get learning rate from appropriate scheduler."""
        if self.warmup.is_warmup_complete():
            return self.main_scheduler.get_lr()
        else:
            return self.warmup.get_lr()
```

### Implementation Strategy

#### Linear Warmup

- Formula: `lr = target_lr * (current_step / warmup_steps)`
- Simple linear interpolation from 0 (or initial_lr) to target_lr
- Most common and easiest to reason about
- Recommended for most use cases

#### Exponential Warmup

- Formula: `lr = initial_lr * (target_lr/initial_lr)^(current_step/warmup_steps)`
- Smoother acceleration at the start
- Useful when very small initial learning rates are needed
- Default initial_lr = target_lr / 1000

### Step vs Epoch-Based Warmup

The scheduler will be step-based by default, but support epoch-based warmup through parameter conversion:

```mojo
fn from_epochs(target_lr: Float64, warmup_epochs: Int,
               steps_per_epoch: Int, strategy: WarmupStrategy) -> WarmupScheduler:
    """Create warmup scheduler from epoch-based parameters.

    Args:
        target_lr: Target learning rate
        warmup_epochs: Number of epochs for warmup
        steps_per_epoch: Training steps per epoch
        strategy: Warmup strategy

    Returns:
        WarmupScheduler configured for epoch-based warmup
    """
    warmup_steps = warmup_epochs * steps_per_epoch
    return WarmupScheduler(target_lr, warmup_steps, strategy)
```

### State Management

The scheduler state includes:

- `target_lr`: Target learning rate
- `warmup_steps`: Total warmup steps
- `current_step`: Current step in warmup
- `strategy`: Warmup strategy type
- `initial_lr`: Initial learning rate

This enables:

- Checkpoint saving and loading
- Training interruption and resumption
- Distributed training synchronization

### Common Warmup Periods

Based on research and common practice:

- **Small models/datasets**: 1000-2000 steps
- **Medium models**: 5000-10000 steps
- **Large models**: 10000-40000 steps
- **Very large models (GPT-scale)**: Up to 375M tokens (varies by batch size)

### Integration with Training Loop

```mojo
# Example usage in training loop
var warmup = WarmupScheduler(target_lr=0.1, warmup_steps=1000, strategy=WarmupStrategy.LINEAR)
var cosine = CosineScheduler(initial_lr=0.1, total_steps=100000)
var scheduler = ChainedScheduler(warmup, cosine, transition_step=1000)

for step in range(total_steps):
    lr = scheduler.get_lr()
    optimizer.set_lr(lr)

    # Training step
    loss = train_step(batch)

    scheduler.step()
```

### Error Handling

The scheduler should validate:

- `warmup_steps > 0`
- `target_lr > 0`
- `initial_lr >= 0` and `initial_lr < target_lr` (for exponential warmup)
- State consistency during serialization/deserialization

### Testing Strategy

1. **Unit tests**:
   - Linear warmup reaches target_lr at warmup_steps
   - Exponential warmup reaches target_lr at warmup_steps
   - Learning rate is monotonically increasing
   - State save/load preserves scheduler behavior

2. **Integration tests**:
   - Chaining with step scheduler
   - Chaining with cosine scheduler
   - Epoch-based vs step-based equivalence

3. **Edge cases**:
   - warmup_steps = 1 (immediate transition)
   - current_step > warmup_steps (stays at target_lr)
   - Resumption from checkpoint mid-warmup

## References

### Source Plan

- [Warmup Scheduler Plan](/home/mvillmow/ml-odyssey-manual/notes/plan/02-shared-library/02-training-utils/02-lr-schedulers/03-warmup-scheduler/plan.md)
- [Parent: LR Schedulers Plan](/home/mvillmow/ml-odyssey-manual/notes/plan/02-shared-library/02-training-utils/02-lr-schedulers/plan.md)

### Related Issues

- Issue #333: [Plan] Warmup Scheduler - Design and Documentation (this issue)
- Issue #334: [Test] Warmup Scheduler - Test Implementation
- Issue #335: [Impl] Warmup Scheduler - Core Implementation
- Issue #336: [Package] Warmup Scheduler - Integration and Packaging
- Issue #337: [Cleanup] Warmup Scheduler - Refactor and Finalize

### Related Components

- Step Scheduler (issues #323-327)
- Cosine Scheduler (issues #328-332)
- Training Loop Integration (shared-library/training-utils)

### Research References

- [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677) - Introduces linear warmup for large batch training
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) - Uses warmup with polynomial decay
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer paper with warmup schedule

## Implementation Notes

(This section will be populated during the Test, Implementation, and Packaging phases)

### Discoveries

- TBD

### Challenges

- TBD

### Decisions Made During Implementation

- TBD
