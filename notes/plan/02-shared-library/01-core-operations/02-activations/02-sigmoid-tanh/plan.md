# Sigmoid Tanh

## Overview

Implement sigmoid and tanh activation functions that produce bounded outputs. Sigmoid maps inputs to (0,1) range and is used for binary classification and gates. Tanh maps to (-1,1) range and is used in RNNs and other architectures. Both require careful numerical handling.

## Parent Plan

[../plan.md](../plan.md)

## Child Plans

None (leaf node)

## Inputs

- Input tensors from previous layers
- Numerical stability requirements
- Gradient information for backpropagation

## Outputs

- Sigmoid activation: 1 / (1 + exp(-x))
- Tanh activation: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
- Numerically stable implementations

## Steps

1. Implement sigmoid with numerical stability (clip large values)
2. Create tanh with stable computation
3. Ensure proper gradient computation
4. Handle edge cases (very large/small inputs)

## Success Criteria

- [ ] Sigmoid outputs are in (0, 1) range
- [ ] Tanh outputs are in (-1, 1) range
- [ ] Functions are numerically stable for extreme inputs
- [ ] Gradients compute correctly for backpropagation

## Notes

Use numerically stable implementations: for sigmoid, use log-sum-exp trick or clip inputs. For tanh, can use stable exponential formulations or relationship to sigmoid. Test with extreme values.
