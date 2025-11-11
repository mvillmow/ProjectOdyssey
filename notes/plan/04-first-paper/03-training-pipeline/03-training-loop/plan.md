# Training Loop

## Overview

Implement the main training loop that iterates over epochs and batches, performs forward and backward passes, updates parameters, and logs progress.

## Parent Plan
[Parent](../plan.md)

## Child Plans
- [01-implement-epoch-loop](./01-implement-epoch-loop/plan.md)
- [02-implement-batch-loop](./02-implement-batch-loop/plan.md)
- [03-add-logging](./03-add-logging/plan.md)

## Inputs
- Implement epoch iteration loop
- Implement batch processing loop
- Add training progress logging
- Track loss over time

## Outputs
- Completed training loop
- Implement epoch iteration loop (completed)

## Steps
1. Implement Epoch Loop
2. Implement Batch Loop
3. Add Logging

## Success Criteria
- [ ] Training loop iterates over all epochs
- [ ] Batch loop processes all data
- [ ] Loss is logged each epoch
- [ ] Progress is displayed clearly
- [ ] Training completes successfully

## Notes
- Start simple, add features incrementally
- Log loss every epoch
- Show progress bar for batches
- Handle end of epoch correctly
- Save state periodically