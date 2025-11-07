# Kaiming He

## Overview
Implement Kaiming (also called He) initialization for neural network weights. This initialization method is specifically designed for ReLU activations and scales weights based on the number of input units, accounting for the fact that ReLU zeros out half the activations.

## Parent Plan
[../plan.md](../plan.md)

## Child Plans
None (leaf node)

## Inputs
- Layer dimensions (number of input and output units)
- Mode (fan_in or fan_out for variance calculation)
- Distribution type (uniform or normal)
- Random seed for reproducibility

## Outputs
- Kaiming uniform initialization: U(-sqrt(6/fan), sqrt(6/fan))
- Kaiming normal initialization: N(0, sqrt(2/fan))
- Properly scaled weight tensors for ReLU networks

## Steps
1. Implement Kaiming uniform variant with correct scaling
2. Create Kaiming normal variant with proper variance
3. Support both fan_in and fan_out modes
4. Ensure random seed produces reproducible results

## Success Criteria
- [ ] Weights have correct variance for ReLU activations
- [ ] Both fan_in and fan_out modes work correctly
- [ ] Uniform and normal variants use proper scaling
- [ ] Random seed produces reproducible initialization

## Notes
Kaiming initialization uses Var(W) = 2/fan (where fan is fan_in or fan_out) to account for ReLU zeroing half the activations. This is critical for deep ReLU networks to avoid gradient problems.
