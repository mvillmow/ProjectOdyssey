# Confusion Matrix

## Overview
Implement confusion matrix for detailed classification analysis. The confusion matrix shows the distribution of predictions versus ground truth labels, revealing which classes are confused with each other. This enables identification of systematic errors and class-specific performance issues.

## Parent Plan
[../plan.md](../plan.md)

## Child Plans
None (leaf node)

## Inputs
- Model predictions (class indices or logits)
- Ground truth labels (class indices)
- Number of classes
- Class names for labeling

## Outputs
- Confusion matrix (2D array of prediction counts)
- Row and column labels with class names
- Normalization options (by row, column, or total)
- Derived metrics (per-class precision, recall)

## Steps
1. Implement confusion matrix accumulation
2. Add normalization options for easier interpretation
3. Support incremental updates for large datasets
4. Provide helpers to extract precision, recall, F1 per class

## Success Criteria
- [ ] Matrix correctly counts all predictions
- [ ] Normalization produces interpretable percentages
- [ ] Incremental updates work correctly
- [ ] Derived metrics match manual calculations

## Notes
Confusion matrix is NxN for N classes. Rows typically represent ground truth, columns represent predictions. Support both accumulation and single-batch computation.
