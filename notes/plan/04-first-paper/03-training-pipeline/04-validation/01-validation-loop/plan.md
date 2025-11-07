# Validation Loop

## Overview

Implement validation loop to evaluate model on validation set without updating parameters.

## Parent Plan
[Parent](../plan.md)

## Child Plans
None (leaf node)

## Inputs
- Iterate over validation data
- Compute validation loss
- No parameter updates

## Outputs
- Completed validation loop

## Steps
1. Iterate over validation data
2. Compute validation loss
3. No parameter updates

## Success Criteria
- [ ] Validation loop implemented
- [ ] Loss computed correctly
- [ ] No gradient updates
- [ ] Tests pass

## Notes
- Similar to training loop but no backward/update
- Set model to evaluation mode
- Iterate over validation dataloader
- Average loss over validation set