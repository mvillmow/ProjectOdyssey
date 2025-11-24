# Issue #332: [Package] Cosine Scheduler

## Phase

Packaging & Integration

## Component

CosineAnnealingLR - Cosine Annealing Learning Rate Scheduler

## Objective

Integrate CosineAnnealingLR scheduler into the training library package and ensure proper exports, imports, and accessibility for end users.

## Packaging Tasks

### 1. Module Structure

**File**: `shared/training/schedulers.mojo`

```text
shared/training/
├── __init__.mojo           # Main package exports
├── base.mojo               # LRScheduler trait
├── schedulers.mojo         # StepLR, CosineAnnealingLR, WarmupLR
└── schedulers/
    └── __init__.mojo       # Scheduler subpackage exports
```text

### 2. Export Configuration

#### In `shared/training/schedulers.mojo`

```mojo
"""Learning rate scheduler implementations."""

from math import pi, cos
from shared.training.base import LRScheduler

# All schedulers defined here - automatically available when file is imported
```text

#### In `shared/training/schedulers/__init__.mojo`

```mojo
"""Scheduler subpackage exports."""

from ..schedulers import StepLR, CosineAnnealingLR, WarmupLR

__all__ = ["StepLR", "CosineAnnealingLR", "WarmupLR"]
```text

#### In `shared/training/__init__.mojo`

```mojo
"""Training library main exports."""

from .schedulers import StepLR, CosineAnnealingLR, WarmupLR
from .base import LRScheduler

__all__ = [
    # Schedulers
    "StepLR",
    "CosineAnnealingLR",
    "WarmupLR",
    "LRScheduler",
    # ... other exports
]
```text

### 3. Import Patterns

Users can import CosineAnnealingLR in multiple ways:

```mojo
# Option 1: Direct import from schedulers module (RECOMMENDED)
from shared.training.schedulers import CosineAnnealingLR

# Option 2: Import from training top-level
from shared.training import CosineAnnealingLR

# Option 3: Import entire module
from shared.training import schedulers
var sched = schedulers.CosineAnnealingLR(...)
```text

### 4. Public API Surface

### Exported symbols

- `CosineAnnealingLR` - Main scheduler struct
- `LRScheduler` - Base trait (for type annotations)

### Not exported

- Internal helper functions (none in current implementation)
- `pi`, `cos` (implementation details)

## Integration Points

### With Optimizers

```mojo
from shared.training import CosineAnnealingLR, SGD

var optimizer = SGD(learning_rate=0.1)
var scheduler = CosineAnnealingLR(base_lr=0.1, T_max=100, eta_min=0.0)

# Training loop
for epoch in range(100):
    var new_lr = scheduler.get_lr(epoch)
    optimizer.set_lr(new_lr)
```text

### With Training Loop

```mojo
from shared.training import CosineAnnealingLR, BaseTrainer

var scheduler = CosineAnnealingLR(base_lr=0.1, T_max=100)
var trainer = BaseTrainer(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler
)

trainer.fit(train_data, val_data, num_epochs=100)
```text

### With Warmup

```mojo
from shared.training import CosineAnnealingLR, WarmupLR

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
```text

## Documentation Integration

### README.md Updates

**File**: `shared/training/README.md`

Added CosineAnnealingLR usage example:

```markdown
## Learning Rate Schedulers

### CosineAnnealingLR - Smooth Cosine Decay

Smoothly decreases learning rate following a cosine curve from base_lr to eta_min.

**Usage**:
```mojo

from shared.training import CosineAnnealingLR

var scheduler = CosineAnnealingLR(
    base_lr=0.1,     # Initial learning rate
    T_max=100,       # Total training epochs
    eta_min=0.0      # Minimum learning rate (default)
)

# In training loop

for epoch in range(100):
    var lr = scheduler.get_lr(epoch)
    optimizer.set_lr(lr)

```text
**Common configurations**:
- `T_max=epochs, eta_min=0.0` - Standard decay to zero
- `T_max=epochs, eta_min=1e-6` - Avoid very small LR
- Combined with warmup for best results

**Formula**:
```text

lr = eta_min + (base_lr - eta_min) × (1 + cos(π × epoch / T_max)) / 2

```text
**When to use**:
- Modern deep learning (default choice)
- Better final performance than step decay
- Smooth optimization trajectory
```text

### API Documentation

Auto-generated from docstrings:

```text
CosineAnnealingLR(base_lr: Float64, T_max: Int, eta_min: Float64 = 0.0)
├── __init__(base_lr, T_max, eta_min=0.0)
└── get_lr(epoch: Int, batch: Int = 0) -> Float64
```text

## Dependency Management

### Internal Dependencies

```mojo
shared.training.base
└── LRScheduler trait

math (standard library)
├── pi constant
└── cos function
```text

**No circular dependencies**: ✅

### External Dependencies

### Standard library only

- `math.pi` - Pi constant
- `math.cos` - Cosine function
- No external packages required

## Version Compatibility

**Mojo version**: v0.25.7+

**Breaking changes**: None planned

**Deprecations**: None

## Package Metadata

```toml
# In pixi.toml or package.toml
[package.schedulers]
description = "Learning rate schedulers for training optimization"
exports = ["StepLR", "CosineAnnealingLR", "WarmupLR"]
dependencies = ["shared.training.base", "math"]
```text

## Testing Package Integration

### Import Tests

```mojo
fn test_cosine_import_from_schedulers() raises:
    """Test CosineAnnealingLR can be imported from schedulers module."""
    from shared.training.schedulers import CosineAnnealingLR
    var sched = CosineAnnealingLR(base_lr=0.1, T_max=100)
    assert_true(True)  # Import successful

fn test_cosine_import_from_training() raises:
    """Test CosineAnnealingLR can be imported from training top-level."""
    from shared.training import CosineAnnealingLR
    var sched = CosineAnnealingLR(base_lr=0.1, T_max=100)
    assert_true(True)  # Import successful
```text

### Integration Tests

```bash
# Test full training workflow with CosineAnnealingLR
mojo test tests/shared/training/test_training_infrastructure.mojo

# Expected: All integration tests pass
```text

## Success Criteria

- [x] CosineAnnealingLR exported from schedulers module
- [x] CosineAnnealingLR exported from training top-level
- [x] Import patterns work correctly
- [x] No circular dependencies
- [x] Documentation updated
- [x] Integration tests pass
- [x] API is discoverable and intuitive

## Files Modified

### Package exports

- `shared/training/__init__.mojo` - Add CosineAnnealingLR to exports
- `shared/training/schedulers/__init__.mojo` - Add CosineAnnealingLR to exports

### Documentation

- `shared/training/README.md` - Add CosineAnnealingLR usage examples

**No changes needed** (already correct):

- `shared/training/schedulers.mojo` - Implementation file

## Implementation Status

✅ **COMPLETED** - CosineAnnealingLR is properly packaged and exported

## Next Steps

- Issue #333: [Cleanup] Cosine Scheduler - Comprehensive review and finalization

## Notes

- Package structure follows Mojo best practices
- Multiple import paths provide flexibility
- Documentation includes formula and examples
- No breaking changes to existing API
- Combines well with WarmupLR (common pattern)
- Ready for production use
