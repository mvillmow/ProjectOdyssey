# Test Shapes

## Overview

Write tests to validate tensor shapes at each stage of the network, ensuring dimensional correctness throughout the forward pass.

## Parent Plan
[Parent](../plan.md)

## Child Plans
None (leaf node)

## Inputs
- Test shapes after each layer
- Verify shape transformations
- Test with different input sizes

## Outputs
- Completed test shapes

## Steps
1. Test shapes after each layer
2. Verify shape transformations
3. Test with different input sizes

## Success Criteria
- [ ] Shape tests pass
- [ ] All layer outputs have correct shapes
- [ ] Shape transformations verified
- [ ] Tests cover all layers

## Notes
- Input: (batch, 1, 28, 28)
- After Conv1: (batch, 6, 24, 24)
- After Pool1: (batch, 6, 12, 12)
- After Conv2: (batch, 16, 8, 8)
- After Pool2: (batch, 16, 4, 4)
- After Flatten: (batch, 256)
- Output: (batch, 10)