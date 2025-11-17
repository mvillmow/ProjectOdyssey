# Test Data

## Overview

Write unit tests for data utilities including base dataset, data loader, and augmentations. These tests verify data access patterns, batching logic, shuffling behavior, and augmentation correctness without requiring large real datasets.

## Parent Plan

[../plan.md](../plan.md)

## Child Plans

None (leaf node)

## Inputs

- Implemented data utilities
- Toy datasets for testing
- Known batching and shuffling patterns
- Sample data for augmentation testing

## Outputs

- Tests for dataset interface compliance
- Tests for data loader batching and shuffling
- Tests for all augmentation types
- Tests for edge cases (empty, single item)
- Integration tests for data pipeline

## Steps

1. Write tests for dataset interface (len, getitem)
2. Create tests for data loader operations
3. Build tests for augmentations (verify properties preserved)
4. Add edge case tests (empty datasets, size 1)
5. Verify complete data pipeline workflows

## Success Criteria

- [ ] Dataset tests verify interface compliance
- [ ] Loader tests verify batching and shuffling
- [ ] Augmentation tests verify property preservation
- [ ] Edge cases are handled correctly

## Notes

Use small in-memory datasets for speed. Verify batching with deterministic data. Test shuffling with seed control. For augmentations, verify properties (e.g., flipped image has same size).
