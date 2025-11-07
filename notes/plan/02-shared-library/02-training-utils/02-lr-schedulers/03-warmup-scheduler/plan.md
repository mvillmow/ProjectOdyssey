# Warmup Scheduler

## Overview
Implement learning rate warmup scheduler that gradually increases the learning rate from a small value to the target value over a specified period. Warmup helps stabilize training at the start, especially for large learning rates or batch sizes.

## Parent Plan
[../plan.md](../plan.md)

## Child Plans
None (leaf node)

## Inputs
- Target learning rate
- Warmup steps or epochs
- Warmup strategy (linear or exponential)
- Current training step or epoch

## Outputs
- Current learning rate during warmup phase
- Smooth transition to target learning rate
- State for serialization and resumption

## Steps
1. Implement linear warmup (gradual increase)
2. Add exponential warmup option
3. Support both step-based and epoch-based warmup
4. Enable chaining with other schedulers (warmup then decay)

## Success Criteria
- [ ] Learning rate increases smoothly during warmup
- [ ] Target learning rate reached at warmup end
- [ ] Both linear and exponential strategies work
- [ ] Chains correctly with decay schedulers

## Notes
Linear warmup: lr = target_lr * (current_step / warmup_steps). Common warmup period: 1000-10000 steps. Often combined with cosine or step decay after warmup completes.
