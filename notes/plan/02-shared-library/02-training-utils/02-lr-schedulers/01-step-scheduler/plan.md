# Step Scheduler

## Overview

Implement step decay learning rate scheduler that reduces the learning rate by a fixed factor at specified intervals (steps or epochs). This simple but effective strategy helps models converge by reducing the learning rate as training progresses.

## Parent Plan

[../plan.md](../plan.md)

## Child Plans

None (leaf node)

## Inputs

- Initial learning rate
- Step size (interval for rate reduction)
- Decay factor (gamma)
- Current training step or epoch

## Outputs

- Current learning rate based on step count
- Schedule update at each training step
- State for serialization and resumption

## Steps

1. Implement step counting and interval detection
2. Apply decay factor at specified intervals
3. Support both step-based and epoch-based scheduling
4. Add state management for training resumption

## Success Criteria

- [ ] Learning rate decreases at correct intervals
- [ ] Decay factor applies correctly
- [ ] Both step and epoch modes work
- [ ] State can be saved and restored

## Notes

Step scheduler is simple: multiply learning rate by gamma every step_size steps. Common values: step_size=30, gamma=0.1. Ensure scheduler works with optimizer's learning rate.
