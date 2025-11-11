# Early Stopping

## Overview
Implement early stopping callback that terminates training when validation performance stops improving. This prevents overfitting by monitoring a validation metric and stopping after a patience period without improvement, saving time and computing resources.

## Parent Plan
[../plan.md](../plan.md)

## Child Plans
None (leaf node)

## Inputs
- Validation metric to monitor
- Patience (epochs without improvement before stopping)
- Minimum delta (threshold for considering improvement)
- Mode (minimize or maximize metric)

## Outputs
- Training termination signal when patience exhausted
- Best metric value and epoch tracking
- Restoration of best model weights (optional)
- Early stopping metadata in training logs

## Steps
1. Implement metric monitoring and comparison
2. Add patience counter for improvement tracking
3. Support both minimize and maximize modes
4. Enable best model restoration on stopping
5. Provide clear stopping reason in logs

## Success Criteria
- [ ] Stops training after patience period without improvement
- [ ] Correctly identifies metric improvement
- [ ] Both minimize and maximize modes work
- [ ] Best model restoration works when enabled

## Notes
Common patience values: 5-20 epochs. Min delta helps ignore noise (e.g., 0.001). Always save best model during monitoring. Make stopping criteria clear in logs.
