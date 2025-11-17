# Dataset Loader

## Overview

Implement Dataset and DataLoader classes to provide a clean interface for accessing training and test data, enabling efficient iteration during training.

## Parent Plan

[Parent](../plan.md)

## Child Plans

- [01-implement-dataset](./01-implement-dataset/plan.md)
- [02-implement-dataloader](./02-implement-dataloader/plan.md)
- [03-test-data-loading](./03-test-data-loading/plan.md)

## Inputs

- Implement Dataset class
- Implement DataLoader class
- Test data loading functionality

## Outputs

- Completed dataset loader
- Implement Dataset class (completed)

## Steps

1. Implement Dataset
2. Implement DataLoader
3. Test Data Loading

## Success Criteria

- [ ] Dataset class provides standard interface
- [ ] DataLoader iterates over batches
- [ ] Data loading is efficient
- [ ] Supports shuffling
- [ ] All data loading tests pass

## Notes

- Follow PyTorch-like interface
- Dataset: __len__ and __getitem__
- DataLoader: batching and shuffling
- Keep implementation simple
- Test with different batch sizes
