# Uniform Normal

## Overview

Implement basic uniform and normal distribution initializers for neural network weights. These simple initializers provide random weight initialization without specific variance scaling, useful for biases, embeddings, or when using custom initialization schemes.

## Parent Plan

[../plan.md](../plan.md)

## Child Plans

None (leaf node)

## Inputs

- Tensor shape to initialize
- Distribution parameters (range for uniform, mean/std for normal)
- Random seed for reproducibility

## Outputs

- Uniform distribution: U(low, high) with configurable bounds
- Normal distribution: N(mean, std) with configurable parameters
- Zero initialization for convenience
- Constant initialization with specified value

## Steps

1. Implement uniform distribution with configurable range
2. Create normal distribution with mean and standard deviation
3. Add zero initialization helper
4. Build constant initialization for specific values

## Success Criteria

- [ ] Uniform distribution samples within specified range
- [ ] Normal distribution has correct mean and standard deviation
- [ ] Zero and constant initializers work correctly
- [ ] Random seed produces reproducible results

## Notes

These are the building blocks for more sophisticated initializers. Ensure proper random seed handling. Provide sensible defaults: uniform in [-0.1, 0.1], normal with mean=0 and std=0.01.
