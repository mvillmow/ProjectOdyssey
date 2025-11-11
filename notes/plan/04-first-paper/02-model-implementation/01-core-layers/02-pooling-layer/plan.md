# Pooling Layer

## Overview

Implement an average pooling layer that downsamples input by averaging values in pooling windows, used to reduce spatial dimensions in the network.

## Parent Plan
[Parent](../plan.md)

## Child Plans
None (leaf node)

## Inputs
- Implement average pooling operation
- Support configurable pool size
- Test pooling correctness

## Outputs
- Completed pooling layer

## Steps
1. Implement average pooling operation
2. Support configurable pool size
3. Test pooling correctness

## Success Criteria
- [ ] Pooling layer correctly averages regions
- [ ] Supports different pool sizes
- [ ] Output shapes are correct
- [ ] Handles batched inputs
- [ ] Tests pass

## Notes
- Average pooling: mean of pool window
- Pool size typically 2x2
- Reduces height and width by factor of pool_size
- No learnable parameters