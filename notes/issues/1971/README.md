# Issue #1971: Export scheduler classes from schedulers package

## Objective

Export the three scheduler classes (StepLR, CosineAnnealingLR, WarmupLR) from the
shared/training/schedulers package __init__.mojo file so they can be imported by test files and
other modules.

## Deliverables

- Updated `/shared/training/schedulers/__init__.mojo` with scheduler class exports
- All three scheduler classes now properly exported:
  - StepLR: Step decay scheduler
  - CosineAnnealingLR: Cosine annealing scheduler
  - WarmupLR: Linear warmup scheduler

## Success Criteria

- [x] Scheduler classes exported from __init__.mojo
- [x] Import statement correctly references schedulers module
- [x] All three classes (StepLR, CosineAnnealingLR, WarmupLR) are exported
- [x] Code passes pre-commit checks
- [x] Commit created with proper message format

## References

- Scheduler implementations: `/shared/training/schedulers.mojo`
- Package init file: `/shared/training/schedulers/__init__.mojo`

## Implementation Notes

### Changes Made

**File**: `/shared/training/schedulers/__init__.mojo`

**Before**:
```mojo
# Export scheduler implementations
from .step_decay import step_lr, multistep_lr, exponential_lr, constant_lr

# TODO: Implement remaining schedulers
# from .cosine import cosine_annealing_lr
# from .warmup import warmup_lr
```

**After**:
```mojo
# Export scheduler implementations
from .step_decay import step_lr, multistep_lr, exponential_lr, constant_lr

# Export scheduler classes
from ..schedulers import StepLR, CosineAnnealingLR, WarmupLR
```

### Problem Solved

The three scheduler classes were defined in `/shared/training/schedulers.mojo` but were not
exported from the package __init__.mojo file. This caused import failures in test files that
tried to import these classes from the schedulers package. By adding the export statement, these
classes are now properly exposed to the package's public API.

### Key Details

- The classes are defined in the parent module `schedulers.mojo` (sibling to the schedulers package)
- Therefore the import uses `from ..schedulers import` to go up one level and import from that module
- All three classes implement the LRScheduler trait and are fully functional
- No modifications to the scheduler implementations were needed
