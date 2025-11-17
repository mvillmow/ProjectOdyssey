# Integration Tests

## Overview

Implement integration tests that verify complete workflows work correctly, including end-to-end training, checkpointing, and reproducibility with fixed random seeds.

## Parent Plan

[Parent](../plan.md)

## Child Plans

- [01-test-end-to-end](./01-test-end-to-end/plan.md)
- [02-test-checkpointing](./02-test-checkpointing/plan.md)
- [03-test-reproducibility](./03-test-reproducibility/plan.md)

## Inputs

- Test end-to-end training workflow
- Test model checkpointing and loading
- Verify training reproducibility

## Outputs

- Completed integration tests
- Test end-to-end training workflow (completed)

## Steps

1. Test End-to-End
2. Test Checkpointing
3. Test Reproducibility

## Success Criteria

- [ ] End-to-end training completes successfully
- [ ] Checkpointing works correctly
- [ ] Training is reproducible with seeds
- [ ] All integration tests pass
- [ ] Tests cover critical workflows

## Notes

- Use small subset of data for speed
- Test with few epochs (2-3)
- Verify loss decreases
- Test checkpoint save/load cycle
- Verify same seed gives same results
