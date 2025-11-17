# Validation Loop

## Overview

Implement the validation loop for evaluating model performance on held-out data without updating weights. This loop performs forward passes, computes metrics, and aggregates results to assess model generalization during and after training.

## Parent Plan

[../plan.md](../plan.md)

## Child Plans

None (leaf node)

## Inputs

- Trained model (in evaluation mode)
- Validation data loader
- Evaluation metrics
- Validation configuration

## Outputs

- Validation metrics (loss, accuracy, etc.)
- Aggregated statistics across validation set
- Per-batch and overall results
- Callback invocations for validation events

## Steps

1. Implement batch iteration over validation data
2. Create forward pass without gradient computation
3. Compute and aggregate validation metrics
4. Add callback hooks for validation events
5. Support both during-training and post-training validation

## Success Criteria

- [ ] Loop runs without computing gradients
- [ ] Metrics aggregate correctly across batches
- [ ] Results match expected validation behavior
- [ ] Callbacks fire at appropriate points
- [ ] Memory usage is efficient (no gradient storage)

## Notes

Ensure model is in evaluation mode (disable dropout, batch norm uses running stats). Don't compute gradients to save memory. Support both full validation and subset validation for speed.
