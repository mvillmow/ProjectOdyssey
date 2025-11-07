# Accuracy

## Overview
Implement accuracy metrics for evaluating classification model performance. This includes top-1 accuracy for single predictions, top-k accuracy for k-best predictions, and per-class accuracy for detailed analysis. Accuracy metrics are essential for understanding model behavior.

## Parent Plan
[../plan.md](../plan.md)

## Child Plans
None (leaf node)

## Inputs
- Model predictions (logits or probabilities)
- Ground truth labels
- k value for top-k accuracy
- Class information for per-class metrics

## Outputs
- Top-1 accuracy: percentage of correct predictions
- Top-k accuracy: percentage where correct label in top k predictions
- Per-class accuracy: accuracy broken down by class
- Support for batched evaluation

## Steps
1. Implement top-1 accuracy for standard classification
2. Create top-k accuracy for relaxed evaluation
3. Build per-class accuracy for detailed analysis
4. Support both batched and incremental computation

## Success Criteria
- [ ] Top-1 accuracy correctly counts exact matches
- [ ] Top-k accuracy properly evaluates k-best predictions
- [ ] Per-class accuracy provides class-wise breakdown
- [ ] Metrics handle edge cases (empty batches, single class)

## Notes
Ensure accuracy calculations handle both logits and probabilities. Support incremental updates for large datasets. Make per-class accuracy easy to interpret and visualize.
