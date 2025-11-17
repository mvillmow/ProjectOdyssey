# Data Loader

## Overview

Create a data loader for efficient batch processing of datasets. This includes batching for grouping samples, shuffling for randomized training order, and iteration support for accessing batches sequentially. The data loader bridges datasets and training loops.

## Parent Plan

[../plan.md](../plan.md)

## Child Plans

- [01-batching/plan.md](01-batching/plan.md)
- [02-shuffling/plan.md](02-shuffling/plan.md)
- [03-iteration/plan.md](03-iteration/plan.md)

## Inputs

- Dataset implementing base dataset interface
- Batch size and shuffling configuration
- Sampling strategy preferences

## Outputs

- Batching mechanism for grouping samples
- Shuffling support for randomized access
- Iterator interface for sequential batch access

## Steps

1. Implement batching to group dataset samples
2. Add shuffling for randomized sample order
3. Create iterator for traversing batches

## Success Criteria

- [ ] Batching correctly groups samples by batch size
- [ ] Shuffling randomizes sample order when enabled
- [ ] Iterator provides clean batch access
- [ ] All child plans are completed successfully

## Notes

Start with simple sequential batching. Add shuffling with proper random seed handling. Ensure the loader handles partial batches at the end of datasets correctly.
