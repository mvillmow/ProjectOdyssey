# Softmax GELU

## Overview
Implement modern activation functions: softmax for converting logits to probability distributions and GELU (Gaussian Error Linear Unit) for smooth non-linear activation. Softmax is essential for multi-class classification, while GELU is used in transformers and modern architectures.

## Parent Plan
[../plan.md](../plan.md)

## Child Plans
None (leaf node)

## Inputs
- Input tensors (logits for softmax, features for GELU)
- Axis specification for softmax reduction
- Numerical stability requirements

## Outputs
- Softmax activation: exp(x) / sum(exp(x)) with numerical stability
- GELU activation: x * Phi(x) where Phi is Gaussian CDF
- Approximations where exact computation is expensive

## Steps
1. Implement numerically stable softmax with axis support
2. Create GELU using exact or approximation formula
3. Ensure proper gradient computation
4. Handle edge cases and numerical stability

## Success Criteria
- [ ] Softmax outputs sum to 1.0 along specified axis
- [ ] Softmax is numerically stable for large logits
- [ ] GELU produces smooth activation curve
- [ ] Both functions work correctly in forward and backward passes

## Notes
For softmax, subtract max value before exp for numerical stability. GELU can use exact formula with erf function or approximation with tanh. Test softmax normalization carefully.
