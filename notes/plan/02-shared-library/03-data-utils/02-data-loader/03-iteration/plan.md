# Iteration

## Overview
Implement iterator interface for data loaders to enable sequential batch access. The iterator provides the mechanism for training loops to traverse batches, supporting both finite epoch-based iteration and infinite streaming patterns.

## Parent Plan
[../plan.md](../plan.md)

## Child Plans
None (leaf node)

## Inputs
- Batched and optionally shuffled data
- Iteration mode (finite or infinite)
- Iterator state management
- Multi-epoch handling

## Outputs
- __iter__ method returning iterator
- __next__ method yielding batches
- StopIteration at epoch end (finite mode)
- State tracking for resumption

## Steps
1. Implement __iter__ to return iterator instance
2. Create __next__ to yield batches sequentially
3. Handle epoch boundaries with StopIteration
4. Support infinite iteration mode for continuous training

## Success Criteria
- [ ] Iterator yields all batches in order
- [ ] Works with Python for-loop syntax
- [ ] Proper StopIteration at epoch end
- [ ] Can reset for multiple epochs

## Notes
Standard Python iterator protocol: __iter__ returns self, __next__ returns items. Raise StopIteration when epoch complete. Support resetting for new epochs. Make iteration state-aware for resumption.
