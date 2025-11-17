# Basic Arithmetic

## Overview

Implement element-wise arithmetic operations on tensors including addition, subtraction, multiplication, and division. These fundamental operations support broadcasting and form the basis for more complex tensor computations in neural networks.

## Parent Plan

[../plan.md](../plan.md)

## Child Plans

None (leaf node)

## Inputs

- Tensor operands with potentially different shapes
- Broadcasting rules for shape compatibility
- Numerical handling for edge cases

## Outputs

- Addition operation with broadcasting
- Subtraction operation with broadcasting
- Multiplication operation with broadcasting
- Division operation with broadcasting and zero handling

## Steps

1. Implement addition with broadcasting support
2. Create subtraction following broadcasting rules
3. Build multiplication with element-wise semantics
4. Implement division with safe zero handling

## Success Criteria

- [ ] All operations produce correct results
- [ ] Broadcasting works according to standard rules
- [ ] Edge cases (zeros, large values) handled properly
- [ ] Operations work with various tensor shapes and dtypes

## Notes

Start with simple cases without broadcasting, then add broadcasting incrementally. Ensure division handles zeros gracefully (infinity or error as appropriate). Test with various tensor shapes.
