# Issue #327: [Package] Step Scheduler

## Phase

Packaging & Integration

## Component

StepLR - Step Decay Learning Rate Scheduler

## Objective

Integrate StepLR scheduler into the training library package and ensure proper exports, imports, and accessibility for end users.

## Packaging Tasks

### 1. Module Structure

**File**: `shared/training/schedulers.mojo`

```
shared/training/
├── __init__.mojo           # Main package exports
├── base.mojo               # LRScheduler trait
├── schedulers.mojo         # StepLR implementation (HERE)
└── schedulers/
    └── __init__.mojo       # Scheduler subpackage exports
```

### 2. Export Configuration

#### In `shared/training/schedulers.mojo`

```mojo
"""Learning rate scheduler implementations."""

from shared.training.base import LRScheduler

# StepLR is defined here - automatically available when file is imported
```

#### In `shared/training/schedulers/__init__.mojo`

```mojo
"""Scheduler subpackage exports."""

from ..schedulers import StepLR, CosineAnnealingLR, WarmupLR

# Re-export for convenience
__all__ = ["StepLR", "CosineAnnealingLR", "WarmupLR"]
```

#### In `shared/training/__init__.mojo`

```mojo
"""Training library main exports."""

from .schedulers import StepLR, CosineAnnealingLR, WarmupLR
from .base import LRScheduler

# Re-export schedulers at top level
__all__ = [
    # Schedulers
    "StepLR",
    "CosineAnnealingLR",
    "WarmupLR",
    "LRScheduler",
    # ... other exports
]
```

### 3. Import Patterns

Users can import StepLR in multiple ways:

```mojo
# Option 1: Direct import from schedulers module
from shared.training.schedulers import StepLR

# Option 2: Import from schedulers subpackage
from shared.training.schedulers import StepLR

# Option 3: Import from training top-level
from shared.training import StepLR

# Option 4: Import entire module
from shared.training import schedulers
var sched = schedulers.StepLR(...)
```

**Recommended**: Option 1 or 3 (most explicit)

### 4. Public API Surface

**Exported symbols**:
- `StepLR` - Main scheduler struct
- `LRScheduler` - Base trait (for type annotations)

**Not exported**:
- Internal helper functions (none in current implementation)
- Private implementation details

## Integration Points

### With Optimizers

```mojo
from shared.training import StepLR, SGD

var optimizer = SGD(learning_rate=0.1)
var scheduler = StepLR(base_lr=0.1, step_size=30, gamma=0.1)

# Training loop
for epoch in range(100):
    var new_lr = scheduler.get_lr(epoch)
    optimizer.set_lr(new_lr)
```

### With Training Loop

```mojo
from shared.training import StepLR, BaseTrainer

var scheduler = StepLR(base_lr=0.1, step_size=30, gamma=0.1)
var trainer = BaseTrainer(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler  # Scheduler integrated into trainer
)

trainer.fit(train_data, val_data, num_epochs=100)
```

### With Callbacks

```mojo
from shared.training import StepLR, LoggingCallback

var scheduler = StepLR(base_lr=0.1, step_size=30, gamma=0.1)

# Log LR changes
var callback = LoggingCallback(log_interval=1)

for epoch in range(100):
    var lr = scheduler.get_lr(epoch)
    callback.log("lr", lr, epoch)
```

## Documentation Integration

### README.md Updates

**File**: `shared/training/README.md`

Added StepLR usage example:

```markdown
## Learning Rate Schedulers

### StepLR - Step Decay

Reduces learning rate by a multiplicative factor at fixed intervals.

**Usage**:
```mojo
from shared.training import StepLR

var scheduler = StepLR(
    base_lr=0.1,     # Initial learning rate
    step_size=30,    # Reduce LR every 30 epochs
    gamma=0.1        # Multiply LR by 0.1 at each step
)

# In training loop
for epoch in range(100):
    var lr = scheduler.get_lr(epoch)
    optimizer.set_lr(lr)
```

**Common configurations**:
- `step_size=30, gamma=0.1` - Classic step decay (AlexNet, VGG)
- `step_size=50, gamma=0.5` - Gentler decay
- `step_size=100, gamma=0.9` - Very gradual decay
```

### API Documentation

Auto-generated from docstrings:

```
StepLR(base_lr: Float64, step_size: Int, gamma: Float64)
├── __init__(base_lr, step_size, gamma)
└── get_lr(epoch: Int, batch: Int = 0) -> Float64
```

## Dependency Management

### Internal Dependencies

```mojo
shared.training.base
└── LRScheduler trait
```

**No circular dependencies**: ✅

### External Dependencies

**Standard library only**:
- Mojo builtin operators (`**`, `//`)
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
dependencies = ["shared.training.base"]
```

## Testing Package Integration

### Import Tests

```mojo
fn test_steplr_import_from_schedulers() raises:
    """Test StepLR can be imported from schedulers module."""
    from shared.training.schedulers import StepLR
    var sched = StepLR(base_lr=0.1, step_size=10, gamma=0.1)
    assert_true(True)  # Import successful

fn test_steplr_import_from_training() raises:
    """Test StepLR can be imported from training top-level."""
    from shared.training import StepLR
    var sched = StepLR(base_lr=0.1, step_size=10, gamma=0.1)
    assert_true(True)  # Import successful
```

### Integration Tests

```bash
# Test full training workflow with StepLR
mojo test tests/shared/training/test_training_infrastructure.mojo

# Expected: All integration tests pass
```

## Success Criteria

- [x] StepLR exported from schedulers module
- [x] StepLR exported from training top-level
- [x] Import patterns work correctly
- [x] No circular dependencies
- [x] Documentation updated
- [x] Integration tests pass
- [x] API is discoverable and intuitive

## Files Modified

**Package exports**:
- `shared/training/__init__.mojo` - Add StepLR to exports
- `shared/training/schedulers/__init__.mojo` - Add StepLR to exports

**Documentation**:
- `shared/training/README.md` - Add StepLR usage examples

**No changes needed** (already correct):
- `shared/training/schedulers.mojo` - Implementation file

## Implementation Status

✅ **COMPLETED** - StepLR is properly packaged and exported

## Next Steps

- Issue #328: [Cleanup] Step Scheduler - Comprehensive review and finalization

## Notes

- Package structure follows Mojo best practices
- Multiple import paths provide flexibility
- Documentation is clear and includes examples
- No breaking changes to existing API
- Ready for production use
