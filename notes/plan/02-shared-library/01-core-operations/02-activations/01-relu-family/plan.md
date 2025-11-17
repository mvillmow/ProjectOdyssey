# ReLU Family

## Overview

Implement ReLU (Rectified Linear Unit) activation functions and their variants. This includes standard ReLU for sparse activation, Leaky ReLU with small negative slope, and PReLU with learnable negative slope parameter. ReLU variants are widely used in modern deep learning.

## Parent Plan

[../plan.md](../plan.md)

## Child Plans

None (leaf node)

## Inputs

- Input tensors from previous layers
- Activation parameters (leak value for Leaky ReLU, alpha for PReLU)
- Gradient information for backpropagation

## Outputs

- ReLU activation: max(0, x)
- Leaky ReLU activation: max(alpha*x, x) with small alpha
- PReLU activation: learnable parameter version of Leaky ReLU

## Steps

1. Implement standard ReLU activation
2. Create Leaky ReLU with configurable leak parameter
3. Build PReLU with learnable slope parameter
4. Ensure proper gradient computation for backpropagation

## Success Criteria

- [ ] ReLU correctly zeros negative values
- [ ] Leaky ReLU preserves small negative gradients
- [ ] PReLU parameter updates work correctly
- [ ] All variants handle edge cases (zeros, large values)

## Notes

ReLU is the simplest: just max(0, x). Leaky ReLU uses a small constant (typically 0.01) for negative values. PReLU makes this learnable but requires parameter storage.
