# Issue #329: [Plan] Cosine Scheduler

## Phase

Planning & Design

## Component

CosineAnnealingLR - Cosine Annealing Learning Rate Scheduler

## Objective

Design a cosine annealing learning rate scheduler that smoothly decreases the learning rate following a cosine curve from an initial value to a minimum value. This scheduler provides continuous, gradual decay that often outperforms step decay strategies.

## Design Specification

### Algorithm

The cosine annealing scheduler implements the following formula:

```text
lr(epoch) = eta_min + (base_lr - eta_min) × (1 + cos(π × epoch / T_max)) / 2
```text

Where:

- `base_lr`: Initial learning rate (e.g., 0.1)
- `eta_min`: Minimum learning rate (e.g., 0.0)
- `T_max`: Total number of epochs (period of cosine)
- `π`: Pi constant (3.14159...)
- `cos()`: Cosine function

### Example Behavior

With `base_lr=0.1`, `T_max=100`, `eta_min=0.0`:

- Epoch 0: lr = 0.1 (maximum, cos(0) = 1)
- Epoch 25: lr ≈ 0.0854 (cos(π/4) ≈ 0.707)
- Epoch 50: lr = 0.05 (halfway, cos(π/2) = 0)
- Epoch 75: lr ≈ 0.0146 (cos(3π/4) ≈ -0.707)
- Epoch 100: lr = 0.0 (minimum, cos(π) = -1)

**Smooth Decay Curve**: Unlike step decay, the learning rate changes continuously at every epoch, following a smooth cosine curve.

### API Design

#### Struct Definition

```mojo
@value
struct CosineAnnealingLR(LRScheduler):
    """Cosine annealing learning rate scheduler.

    Smoothly decreases learning rate following cosine curve.

    Attributes:
        base_lr: Initial learning rate
        T_max: Maximum number of epochs (period)
        eta_min: Minimum learning rate
    """
    var base_lr: Float64
    var T_max: Int
    var eta_min: Float64
```text

#### Constructor

```mojo
fn __init__(
    out self,
    base_lr: Float64,
    T_max: Int,
    eta_min: Float64 = 0.0
):
    """Initialize CosineAnnealingLR scheduler.

    Args:
        base_lr: Initial learning rate (must be > eta_min)
        T_max: Maximum epochs for one cycle (must be > 0)
        eta_min: Minimum learning rate (default 0.0, must be >= 0)

    Raises:
        Error if base_lr <= eta_min
        Error if T_max <= 0
        Error if eta_min < 0
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
        lr = eta_min + (base_lr - eta_min) × (1 + cos(π × epoch / T_max)) / 2

    Notes:
        - Epoch is clamped to [0, T_max] range
        - Beyond T_max, LR stays at eta_min
    """
```text

### Interface Compliance

CosineAnnealingLR implements the `LRScheduler` trait from `shared/training/base.mojo`:

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
| `base_lr` | Float64 | Required | (eta_min, ∞) | Initial learning rate |
| `T_max` | Int | Required | [1, ∞) | Total epochs (period) |
| `eta_min` | Float64 | 0.0 | [0, base_lr) | Minimum learning rate |

### Common Values

- `T_max`: Usually set to total training epochs (e.g., 100, 200, 500)
- `eta_min`: Often 0.0, sometimes 1e-6 or 1e-4 to avoid very small LR

### Edge Cases

1. **T_max = 0**: Should raise error (undefined period)
1. **T_max = 1**: Valid but degenerate (immediate decay to eta_min)
1. **epoch > T_max**: LR stays at eta_min (cosine beyond period)
1. **eta_min = base_lr**: Valid but constant LR (no annealing)
1. **eta_min > base_lr**: Invalid, should raise error
1. **Negative epoch**: Undefined behavior (should validate)

### Mathematical Properties

1. **Smoothness**: First derivative continuous everywhere
1. **Monotonic decrease**: LR strictly decreases from 0 to T_max
1. **Boundary values**:
   - `lr(0) = base_lr`
   - `lr(T_max) = eta_min`
1. **Symmetry**: Decay is symmetric around T_max/2

### Integration Points

#### With Optimizer

```mojo
# Optimizer must expose current learning rate
var optimizer = SGD(learning_rate=0.1)

# Scheduler computes new LR each epoch
var scheduler = CosineAnnealingLR(base_lr=0.1, T_max=100, eta_min=0.0)

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

#### With Warmup

Cosine annealing is often combined with warmup:

```mojo
# Warmup for first 10 epochs, then cosine decay
var warmup = WarmupLR(base_lr=0.1, warmup_epochs=10)
var cosine = CosineAnnealingLR(base_lr=0.1, T_max=90, eta_min=0.0)

for epoch in range(100):
    if epoch < 10:
        var lr = warmup.get_lr(epoch)
    else:
        var lr = cosine.get_lr(epoch - 10)
    optimizer.set_lr(lr)
```text

### State Management

For checkpointing and resumption:

```mojo
fn state_dict(self) -> Dict[String, Variant]:
    """Return scheduler state for checkpointing.

    Returns:
        Dictionary containing:
        - "base_lr": Initial learning rate
        - "T_max": Maximum epochs
        - "eta_min": Minimum learning rate
    """

fn load_state_dict(inout self, state: Dict[String, Variant]):
    """Load scheduler state from checkpoint.

    Args:
        state: Dictionary from state_dict()
    """
```text

## Design Rationale

### Why Cosine Annealing

1. **Smooth decay**: Continuous changes avoid sudden jumps in loss landscape
1. **Better final performance**: Often achieves lower final loss than step decay
1. **Natural decay curve**: Cosine follows natural optimization dynamics
1. **Simple and effective**: One parameter (T_max), proven in many papers
1. **SOTA results**: Used in many state-of-the-art papers (ResNet, Transformer)

### Design Decisions

1. **Stateless design**: Scheduler doesn't track current epoch
   - Rationale: Same as StepLR - simpler, more flexible
   - Training loop provides epoch number to `get_lr()`

1. **Clamping beyond T_max**: LR stays at eta_min after T_max
   - Rationale: Graceful continuation if training runs longer
   - Alternative: Could restart cycle (cosine annealing with restarts)

1. **Default eta_min = 0.0**: Decay all the way to zero
   - Rationale: Common choice in literature
   - User can set eta_min > 0 to avoid very small LR

1. **Period = T_max, not 2*T_max**: One complete cosine cycle
   - Rationale: Matches PyTorch convention
   - Full cycle goes from +1 to -1 on cosine curve

1. **No restart support**: Single cycle only
   - Rationale: KISS principle - keep simple
   - Advanced users can implement restarts in training loop

### Comparison with Step Decay

| Aspect | Cosine Annealing | Step Decay |
|--------|------------------|------------|
| **Decay pattern** | Smooth, continuous | Discrete jumps |
| **Parameters** | 1 (T_max) | 2 (step_size, gamma) |
| **Tuning difficulty** | Easier (just set T_max = epochs) | Harder (need to tune both params) |
| **Final performance** | Often better | Good but may plateau |
| **Interpretability** | Less intuitive | Very intuitive |
| **Popularity** | Modern papers | Classic papers |

## Performance Considerations

- **Computation**: O(1) - single cosine computation
- **Memory**: O(1) - only stores 3 parameters
- **Overhead**: Negligible (<< 0.01% of training time)

### Numerical Stability

### Cosine computation

- Use standard library `cos()` function
- Input range: [0, π] (well-behaved)
- Output range: [-1, 1] (bounded)
- No risk of overflow or underflow

## Validation Strategy

1. **Mathematical correctness**: Verify formula implementation
1. **Boundary conditions**: Test epoch=0, epoch=T_max, epoch>T_max
1. **Smoothness**: Verify LR changes continuously
1. **Integration**: Test with actual optimizer and training loop
1. **Reproducibility**: Same parameters → same LR schedule

## Success Criteria

- [ ] API matches LRScheduler trait interface
- [ ] Formula correctly implements cosine annealing
- [ ] Parameters validated on construction
- [ ] Edge cases handled properly
- [ ] Integration with optimizer works
- [ ] State can be saved and restored
- [ ] Documentation is complete and clear
- [ ] Decay curve is smooth and continuous

## Files

### Implementation

- `shared/training/schedulers.mojo` - CosineAnnealingLR struct implementation

### Tests

- `tests/shared/training/test_cosine_scheduler.mojo` - Comprehensive test suite

### Documentation

- `shared/training/README.md` - Usage examples and API docs

## References

- [SGDR: Stochastic Gradient Descent with Warm Restarts (Loshchilov & Hutter, 2016)](https://arxiv.org/abs/1608.03983) - Introduced cosine annealing
- [PyTorch CosineAnnealingLR](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html)
- [Keras CosineDecay](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/CosineDecay)
- Papers using cosine annealing:
  - ResNet variants (He et al.)
  - Transformer models (Vaswani et al.)
  - Modern vision models (EfficientNet, ViT)

## Implementation Status

✅ **COMPLETED** - Implementation exists in `shared/training/schedulers.mojo`

This design document was created retrospectively to document the existing implementation.

## Notes

- Cosine annealing is the de facto standard in modern deep learning
- Often combined with warmup for best results
- Restarts can be added but increase complexity
- T_max should match total training epochs for best results
- Consider eta_min > 0 to avoid numerical issues with very small LR
