# Xavier Glorot

## Overview

Implement Xavier (also called Glorot) initialization for neural network weights. This initialization method scales weights based on the number of input and output units to maintain variance across layers, preventing vanishing or exploding gradients in networks with sigmoid/tanh activations.

## Parent Plan

[../plan.md](../plan.md)

## Child Plans

None (leaf node)

## Inputs

- Layer dimensions (number of input and output units)
- Distribution type (uniform or normal)
- Random seed for reproducibility

## Outputs

- Xavier uniform initialization: U(-sqrt(6/(fan_in + fan_out)), sqrt(6/(fan_in + fan_out)))
- Xavier normal initialization: N(0, sqrt(2/(fan_in + fan_out)))
- Properly scaled weight tensors

## Steps

1. Implement Xavier uniform variant with correct scaling
2. Create Xavier normal variant with proper variance
3. Ensure correct calculation of fan_in and fan_out
4. Support random seed for reproducible initialization

## Success Criteria

- [ ] Weights have correct variance based on layer dimensions
- [ ] Uniform variant uses correct bounds
- [ ] Normal variant uses correct standard deviation
- [ ] Random seed produces reproducible results

## Notes

Xavier initialization is designed for sigmoid and tanh activations. The key is variance scaling: Var(W) = 2/(fan_in + fan_out). Test that initialized weights have approximately the right variance.
