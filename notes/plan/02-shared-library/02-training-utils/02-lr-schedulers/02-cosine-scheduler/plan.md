# Cosine Scheduler

## Overview

Implement cosine annealing learning rate scheduler that smoothly decreases the learning rate following a cosine curve. This scheduler provides gradual, continuous decay without sudden drops, often leading to better final performance than step decay.

## Parent Plan

[../plan.md](../plan.md)

## Child Plans

None (leaf node)

## Inputs

- Initial learning rate
- Total training steps or epochs
- Minimum learning rate (optional)
- Current training step or epoch

## Outputs

- Current learning rate following cosine curve
- Smooth decay from initial to minimum rate
- State for serialization and resumption

## Steps

1. Implement cosine annealing formula
2. Support configurable minimum learning rate
3. Handle both step-based and epoch-based scheduling
4. Add state management for training resumption

## Success Criteria

- [ ] Learning rate follows cosine curve correctly
- [ ] Decay is smooth and continuous
- [ ] Minimum learning rate is respected
- [ ] State can be saved and restored

## Notes

Cosine formula: lr = min_lr + (max_lr - min_lr) *(1 + cos(pi* current_step / total_steps)) / 2. Provides smooth decay. Often used with warmup for best results.
