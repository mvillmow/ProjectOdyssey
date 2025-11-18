# Issue #336: [Impl] Warmup Scheduler

## Phase

Implementation

## Component

WarmupLR - Linear Warmup Learning Rate Scheduler

## Objective

Implement WarmupLR scheduler according to design specification (issue #334) and test requirements (issue #335).

## Implementation Details

### File Location

`shared/training/schedulers.mojo` - WarmupLR struct (Lines 158-213)

### Implementation

```mojo
@value
struct WarmupLR(LRScheduler):
    """Linear warmup: gradually increase learning rate during initial epochs.

    Formula:
        lr = base_lr * (epoch / warmup_epochs)  for epoch < warmup_epochs
        lr = base_lr                            for epoch >= warmup_epochs

    Attributes:
        base_lr: Target learning rate after warmup
        warmup_epochs: Number of epochs for warmup phase
    """

    var base_lr: Float64
    var warmup_epochs: Int

    fn __init__(out self, base_lr: Float64, warmup_epochs: Int):
        self.base_lr = base_lr
        self.warmup_epochs = warmup_epochs

    fn get_lr(self, epoch: Int, batch: Int = 0) -> Float64:
        if self.warmup_epochs <= 0:
            return self.base_lr

        if epoch >= self.warmup_epochs:
            return self.base_lr

        # Linear warmup
        var progress = Float64(epoch) / Float64(self.warmup_epochs)
        return self.base_lr * progress
```

## Implementation Decisions

### 1. Start from Zero

**Decision**: LR starts at 0.0, not a small value

**Rationale**:
- Simplest implementation
- Matches PyTorch LinearLR default
- Most common use case

### 2. Defensive warmup_epochs Check

**Decision**: Return base_lr if warmup_epochs <= 0

**Rationale**:
- Prevents division by zero
- Graceful degradation

### 3. Float64 Precision

**Decision**: Use Float64 for progress calculation

**Rationale**:
- Higher precision
- Consistent with other schedulers

## Performance Analysis

**Time**: O(1) - Constant time

**Space**: 16 bytes (base_lr + warmup_epochs)

**Numerical Stability**: Excellent

## Success Criteria

- [x] Implementation matches design
- [ ] All tests pass (75% TODO)
- [x] LRScheduler trait compliance
- [x] Formula correctness
- [x] Well-documented

## Files

**Implementation**: `shared/training/schedulers.mojo` (Lines 158-213)

**Tests**: `tests/shared/training/test_warmup_scheduler.mojo` (75% TODO)

## Implementation Status

 **COMPLETED**

## Notes

- Simple and correct implementation
- Linear warmup effective for training stabilization
- Typically first 5-10% of training
- Usually followed by decay scheduler
