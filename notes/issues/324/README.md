# Issue #324: [Plan] Step Scheduler

## Phase

Planning & Design

## Component

StepLR - Step Decay Learning Rate Scheduler

## Objective

Design a step decay learning rate scheduler that reduces the learning rate by a fixed multiplicative factor at specified epoch intervals. This scheduler implements the classic "step decay" strategy commonly used in deep learning training.

## Design Specification

### Algorithm

The step decay scheduler implements the following formula:

```text
lr(epoch) = base_lr × gamma^⌊epoch / step_size⌋
```text

Where:

- `base_lr`: Initial learning rate (e.g., 0.1)
- `gamma`: Multiplicative decay factor (e.g., 0.1 means reduce to 10%)
- `step_size`: Number of epochs between each decay (e.g., 30)
- `⌊x⌋`: Floor function (integer division)

### Example Behavior

With `base_lr=0.1`, `step_size=10`, `gamma=0.1`:

- Epochs 0-9: lr = 0.1
- Epochs 10-19: lr = 0.01 (0.1 × 0.1¹)
- Epochs 20-29: lr = 0.001 (0.1 × 0.1²)
- Epochs 30-39: lr = 0.0001 (0.1 × 0.1³)

### API Design

#### Struct Definition

```mojo
@value
struct StepLR(LRScheduler):
    """Step decay learning rate scheduler.

    Reduces learning rate by gamma every step_size epochs.

    Attributes:
        base_lr: Initial learning rate
        step_size: Number of epochs between LR reductions
        gamma: Multiplicative decay factor
    """
    var base_lr: Float64
    var step_size: Int
    var gamma: Float64
```text

#### Constructor

```mojo
fn __init__(
    out self,
    base_lr: Float64,
    step_size: Int,
    gamma: Float64
):
    """Initialize StepLR scheduler.

    Args:
        base_lr: Initial learning rate (must be > 0)
        step_size: Epochs between decay (must be > 0)
        gamma: Decay factor (typically 0 < gamma < 1)

    Raises:
        Error if base_lr <= 0
        Error if step_size <= 0
        Error if gamma <= 0
    """
```text

#### Core Method

```mojo
fn get_lr(self, epoch: Int, batch: Int = 0) -> Float64:
    """Compute learning rate for given epoch.

    Args:
        epoch: Current epoch number (0-indexed)
        batch: Current batch (unused, for interface compatibility)

    Returns:
        Learning rate for this epoch

    Formula:
        lr = base_lr × gamma^⌊epoch / step_size⌋
    """
```text

### Interface Compliance

StepLR implements the `LRScheduler` trait from `shared/training/base.mojo`:

```mojo
trait LRScheduler:
    """Base interface for learning rate schedulers."""

    fn get_lr(self, epoch: Int, batch: Int = 0) -> Float64:
        """Return learning rate for current epoch/batch."""
        ...
```text

### Configuration Parameters

| Parameter | Type | Default | Valid Range | Description |
|-----------|------|---------|-------------|-------------|
| `base_lr` | Float64 | Required | (0, ∞) | Initial learning rate |
| `step_size` | Int | Required | [1, ∞) | Epochs between each decay |
| `gamma` | Float64 | Required | (0, 1] | Multiplicative decay factor |

### Common Values

- `step_size`: 30, 50, 100 (depends on total epochs)
- `gamma`: 0.1, 0.5, 0.9 (0.1 is most common)

### Edge Cases

1. **step_size = 0**: Should raise error (undefined behavior)
1. **gamma = 1.0**: Valid but no decay (LR stays constant)
1. **gamma = 0.0**: Edge case - LR becomes 0 after first step
1. **Very large epochs**: LR may underflow to 0 (acceptable)
1. **negative epoch**: Undefined behavior (should validate)

### Integration Points

#### With Optimizer

```mojo
# Optimizer must expose current learning rate
var optimizer = SGD(learning_rate=0.1)

# Scheduler computes new LR each epoch
var scheduler = StepLR(base_lr=0.1, step_size=30, gamma=0.1)

# Training loop updates optimizer LR
for epoch in range(100):
    var new_lr = scheduler.get_lr(epoch)
    optimizer.set_lr(new_lr)  # Optimizer must implement this
```text

#### With Training Loop

```mojo
# Training loop calls scheduler each epoch
fn train(model, data, scheduler):
    for epoch in range(num_epochs):
        # Update learning rate
        var lr = scheduler.get_lr(epoch)

        # Train one epoch with current LR
        for batch in data:
            # ... training logic ...
            pass
```text

### State Management

For checkpointing and resumption:

```mojo
fn state_dict(self) -> Dict[String, Variant]:
    """Return scheduler state for checkpointing.

    Returns:
        Dictionary containing:
        - "base_lr": Initial learning rate
        - "step_size": Step size parameter
        - "gamma": Decay factor
    """

fn load_state_dict(inout self, state: Dict[String, Variant]):
    """Load scheduler state from checkpoint.

    Args:
        state: Dictionary from state_dict()
    """
```text

## Design Rationale

### Why Step Decay

1. **Simplicity**: Easy to understand and implement
1. **Proven effectiveness**: Used in many classic papers (AlexNet, VGG, ResNet)
1. **Predictable**: Discrete jumps make debugging easier
1. **Hyperparameter efficiency**: Only 2 parameters (step_size, gamma)

### Design Decisions

1. **Stateless design**: Scheduler doesn't track current epoch
   - Rationale: Simpler implementation, easier to reason about
   - Training loop provides epoch number to `get_lr()`

1. **Separate from optimizer**: Scheduler computes LR, doesn't modify optimizer
   - Rationale: Separation of concerns, reusable across optimizers
   - Training loop applies LR to optimizer

1. **Floor function**: Use integer division for step calculation
   - Rationale: Standard implementation, no floating point issues
   - Ensures discrete steps at exact epochs

1. **No minimum LR**: Allow LR to decay to very small values
   - Rationale: Some training benefits from extremely small LR
   - User can control via gamma and step_size

## Performance Considerations

- **Computation**: O(1) - simple arithmetic, no loops
- **Memory**: O(1) - only stores 3 parameters
- **Overhead**: Negligible compared to training time

## Validation Strategy

1. **Mathematical correctness**: Verify formula implementation
1. **Edge cases**: Test boundary conditions (gamma=1, step_size=1, etc.)
1. **Integration**: Test with actual optimizer and training loop
1. **Reproducibility**: Same parameters → same LR schedule

## Success Criteria

- [ ] API matches LRScheduler trait interface
- [ ] Formula correctly implements step decay
- [ ] Parameters validated on construction
- [ ] Edge cases handled properly
- [ ] Integration with optimizer works
- [ ] State can be saved and restored
- [ ] Documentation is complete and clear

## Files

### Implementation

- `shared/training/schedulers.mojo` - StepLR struct implementation

### Tests

- `tests/shared/training/test_step_scheduler.mojo` - Comprehensive test suite

### Documentation

- `shared/training/README.md` - Usage examples and API docs

## References

- [PyTorch StepLR](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html)
- [Keras Step Decay](https://keras.io/api/optimizers/learning_rate_schedules/exponential_decay/)
- Classic papers using step decay:
  - AlexNet (Krizhevsky et al., 2012)
  - VGGNet (Simonyan & Zisserman, 2014)
  - ResNet (He et al., 2015)

## Implementation Status

✅ **COMPLETED** - Implementation exists in `shared/training/schedulers.mojo`

This design document was created retrospectively to document the existing implementation.

## Notes

- Step decay is simple but effective for many problems
- More sophisticated schedulers (cosine annealing) often perform better
- Consider combining with warmup for large learning rates
- Monitor training loss to tune step_size and gamma parameters
