# FC Layer

## Overview

Implement a fully connected (dense) layer that performs matrix multiplication between inputs and weights, plus bias addition.

## Parent Plan
[Parent](../plan.md)

## Child Plans
None (leaf node)

## Inputs
- Implement fully connected layer
- Support matrix multiplication
- Add bias term

## Outputs
- Completed fc layer

## Steps
1. Implement fully connected layer
2. Support matrix multiplication
3. Add bias term

## Success Criteria
- [ ] FC layer performs correct matmul
- [ ] Bias is added correctly
- [ ] Handles batched inputs
- [ ] Output shapes are correct
- [ ] Tests pass

## Notes
- Implements: output = input @ weights + bias
- Input shape: (batch, in_features)
- Output shape: (batch, out_features)
- Weights shape: (in_features, out_features)
- Bias shape: (out_features,)