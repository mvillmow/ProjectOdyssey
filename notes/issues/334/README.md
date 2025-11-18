# Issue #334: [Plan] Warmup Scheduler

## Phase

Planning & Design

## Component

WarmupLR - Linear Warmup Learning Rate Scheduler

## Objective

Design a linear warmup learning rate scheduler that gradually increases the learning rate from zero to the target value over a specified number of epochs. This helps stabilize training at the start, especially when using large learning rates or large batch sizes.

## Design Specification

### Algorithm

The warmup scheduler implements the following piecewise linear formula:

```
lr(epoch) = base_lr × (epoch / warmup_epochs)    if epoch < warmup_epochs
lr(epoch) = base_lr                              if epoch >= warmup_epochs
```

Where:
- `base_lr`: Target learning rate after warmup completes (e.g., 0.1)
- `warmup_epochs`: Number of epochs for the warmup phase (e.g., 10)
- `epoch`: Current training epoch (0-indexed)

### Example Behavior

With `base_lr=0.1`, `warmup_epochs=10`:
- Epoch 0: lr = 0.0 (0%)
- Epoch 5: lr = 0.05 (50%)
- Epoch 10: lr = 0.1 (100%, warmup complete)
- Epoch 15: lr = 0.1 (stays at base_lr)

### API Design

#### Struct Definition

```mojo
@value
struct WarmupLR(LRScheduler):
    """Linear warmup learning rate scheduler.

    Gradually increases learning rate from 0 to base_lr over warmup_epochs,
    then maintains base_lr for subsequent epochs.

    Attributes:
        base_lr: Target learning rate after warmup
        warmup_epochs: Number of epochs for warmup phase
    """
    var base_lr: Float64
    var warmup_epochs: Int
```

#### Constructor

```mojo
fn __init__(
    out self,
    base_lr: Float64,
    warmup_epochs: Int
):
    """Initialize Warmup scheduler.

    Args:
        base_lr: Target learning rate (must be > 0)
        warmup_epochs: Epochs for warmup (must be > 0)

    Raises:
        Error if base_lr <= 0
        Error if warmup_epochs <= 0
    """
```

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
        lr = base_lr × (epoch / warmup_epochs)  if epoch < warmup_epochs
        lr = base_lr                            if epoch >= warmup_epochs
    """
```

### Interface Compliance

WarmupLR implements the `LRScheduler` trait:

```mojo
trait LRScheduler:
    """Base interface for learning rate schedulers."""

    fn get_lr(self, epoch: Int, batch: Int = 0) -> Float64:
        """Return learning rate for current epoch/batch."""
        ...
```

### Configuration Parameters

| Parameter | Type | Default | Valid Range | Description |
|-----------|------|---------|-------------|-------------|
| `base_lr` | Float64 | Required | (0, ) | Target learning rate |
| `warmup_epochs` | Int | Required | [1, ) | Epochs for warmup |

**Common Values**:
- `warmup_epochs`: 5, 10, 20 (5-10% of total epochs)
- `base_lr`: 0.001, 0.01, 0.1 (depends on optimizer)

### Edge Cases

1. **warmup_epochs = 0**: Should raise error (no warmup)
2. **warmup_epochs = 1**: Valid but minimal (instant ramp)
3. **epoch = 0**: Returns 0.0 (0% progress)
4. **epoch >= warmup_epochs**: Returns base_lr (100% progress)
5. **negative epoch**: Undefined behavior (should validate)

### Integration Points

#### With Optimizer

```mojo
var optimizer = SGD(learning_rate=0.0)  # Start at zero
var scheduler = WarmupLR(base_lr=0.1, warmup_epochs=10)

# Training loop updates optimizer LR
for epoch in range(100):
    var new_lr = scheduler.get_lr(epoch)
    optimizer.set_lr(new_lr)
```

#### With Training Loop

```mojo
fn train(model, data, scheduler):
    for epoch in range(num_epochs):
        var lr = scheduler.get_lr(epoch)

        for batch in data:
            # Training logic with current LR
            pass
```

#### With Decay Scheduler (Composition Pattern)

```mojo
var warmup = WarmupLR(base_lr=0.1, warmup_epochs=10)
var cosine = CosineAnnealingLR(base_lr=0.1, T_max=90, eta_min=0.0)

# Combined schedule
for epoch in range(100):
    var lr: Float64
    if epoch < 10:
        lr = warmup.get_lr(epoch)
    else:
        lr = cosine.get_lr(epoch - 10)
    optimizer.set_lr(lr)
```

## Design Rationale

### Why Warmup?

1. **Large learning rates**: Prevents instability at start with high LR
2. **Large batch sizes**: Stabilizes gradient estimates with large batches
3. **Deep networks**: Helps deep models settle before aggressive optimization
4. **Adam/AdamW**: Reduces initial variance in adaptive moments

### Research Evidence

- **BERT** (Devlin et al., 2018): Uses warmup for stable Transformer training
- **ImageNet in 1 Hour** (Goyal et al., 2017): Critical for large-batch training
- **Attention Is All You Need** (Vaswani et al., 2017): Warmup with inverse sqrt decay

### Design Decisions

1. **Linear vs Exponential**: Linear warmup chosen for simplicity
   - Rationale: Most common in literature, easy to understand
   - Exponential warmup adds complexity without clear benefit

2. **Start from Zero**: LR begins at 0.0 instead of small value
   - Rationale: Simplest approach, matches PyTorch default
   - Alternative: Could start at base_lr / 100

3. **Constant after Warmup**: LR stays at base_lr after warmup completes
   - Rationale: Warmup is typically combined with decay scheduler
   - Pure warmup is rare; usually followed by decay

4. **Stateless Design**: Scheduler doesn't track current epoch
   - Rationale: Consistent with StepLR and CosineAnnealingLR
   - Training loop provides epoch number to `get_lr()`

5. **No Minimum LR**: Starts from exactly zero
   - Rationale: Initial training steps use very small gradients
   - Prevents instability from large initial LR

## Performance Considerations

- **Computation**: O(1) - simple arithmetic, no loops
- **Memory**: O(1) - only stores 2 parameters (16 bytes)
- **Overhead**: Negligible compared to training time

## Validation Strategy

1. **Mathematical correctness**: Verify linear interpolation formula
2. **Edge cases**: Test boundary conditions (epoch=0, epoch=warmup_epochs)
3. **Integration**: Test with actual optimizer and training loop
4. **Reproducibility**: Same parameters ’ same LR schedule
5. **Composition**: Test chaining with decay schedulers

## Success Criteria

- [ ] API matches LRScheduler trait interface
- [ ] Formula correctly implements linear warmup
- [ ] Parameters validated on construction
- [ ] Edge cases handled properly
- [ ] Integration with optimizer works
- [ ] Composition with decay schedulers works
- [ ] Documentation is complete and clear

## Files

**Implementation**:
- `shared/training/schedulers.mojo` - WarmupLR struct implementation (lines 158-213)

**Tests**:
- `tests/shared/training/test_warmup_scheduler.mojo` - Comprehensive test suite

**Documentation**:
- `shared/training/README.md` - Usage examples and API docs

## Comparison with Other Schedulers

| Aspect | WarmupLR | StepLR | CosineAnnealingLR |
|--------|----------|--------|-------------------|
| Purpose | Stabilize start | Periodic decay | Smooth decay |
| LR Curve | Linear increase | Step decrease | Cosine decrease |
| When to use | Initial epochs | Entire training | Entire training |
| Typical duration | 5-10% of epochs | 100% of epochs | 100% of epochs |
| Combination | Usually with decay | Standalone | Usually standalone |

## Warmup Durations

Based on research and practice:

**Small models/datasets**:
- Warmup epochs: 5-10
- Total epochs: 100-200
- Warmup ratio: 5-10%

**Medium models**:
- Warmup epochs: 10-20
- Total epochs: 200-500
- Warmup ratio: 5-10%

**Large models (ResNet, Transformers)**:
- Warmup epochs: 5-20
- Total epochs: 100-300
- Warmup ratio: 5-10%

**Very large models (BERT, GPT)**:
- Warmup steps: 10,000-40,000
- Total steps: 100,000-1,000,000
- Warmup ratio: 5-10%

## Common Warmup + Decay Patterns

### Pattern 1: Warmup + Cosine Annealing

```mojo
var warmup = WarmupLR(base_lr=0.1, warmup_epochs=10)
var cosine = CosineAnnealingLR(base_lr=0.1, T_max=90, eta_min=0.0)

# 10 epochs warmup, 90 epochs cosine decay
```

**Use case**: Modern deep learning (ResNets, Transformers)

### Pattern 2: Warmup + Step Decay

```mojo
var warmup = WarmupLR(base_lr=0.1, warmup_epochs=10)
var step = StepLR(base_lr=0.1, step_size=30, gamma=0.1)

# 10 epochs warmup, then step decay every 30 epochs
```

**Use case**: Traditional CNNs (AlexNet, VGG)

### Pattern 3: Warmup Only

```mojo
var warmup = WarmupLR(base_lr=0.1, warmup_epochs=10)

# 10 epochs warmup, then constant LR
```

**Use case**: Fine-tuning pre-trained models

## References

- [PyTorch LinearLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LinearLR.html)
- **BERT**: *BERT: Pre-training of Deep Bidirectional Transformers* (Devlin et al., 2018)
- **ImageNet in 1 Hour**: *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour* (Goyal et al., 2017)
- **Transformer**: *Attention Is All You Need* (Vaswani et al., 2017)

## Implementation Status

 **COMPLETED** - Implementation exists in `shared/training/schedulers.mojo`

This design document was created retrospectively to document the existing implementation.

## Notes

- Linear warmup is simple and effective
- Almost always combined with a decay scheduler
- Critical for training with large learning rates or large batches
- Typical warmup duration: 5-10% of total training
- After warmup, switch to decay scheduler (cosine, step, etc.)
- The formula `lr = base_lr × (epoch / warmup_epochs)` produces smooth linear increase
- At epoch 0, LR = 0; at warmup_epochs, LR = base_lr
