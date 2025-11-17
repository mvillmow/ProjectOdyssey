# Tensor Ops

## Overview

Implement fundamental tensor operations including basic arithmetic (add, subtract, multiply, divide), matrix operations (matmul, transpose), and reduction operations (sum, mean, max, min). These operations form the computational foundation for all neural network operations.

## Parent Plan

[../plan.md](../plan.md)

## Child Plans

- [01-basic-arithmetic/plan.md](01-basic-arithmetic/plan.md)
- [02-matrix-ops/plan.md](02-matrix-ops/plan.md)
- [03-reduction-ops/plan.md](03-reduction-ops/plan.md)

## Inputs

- Mojo tensor types and SIMD operations
- Mathematical specifications for each operation
- Test cases with expected outputs

## Outputs

- Element-wise arithmetic operations
- Matrix multiplication and transpose
- Reduction operations along tensor dimensions

## Steps

1. Implement basic arithmetic operations with broadcasting support
2. Create matrix operations for linear algebra
3. Build reduction operations for aggregating tensor values

## Success Criteria

- [ ] Arithmetic operations handle broadcasting correctly
- [ ] Matrix operations produce mathematically correct results
- [ ] Reductions work across specified dimensions
- [ ] All child plans are completed successfully

## Notes

Start with simple implementations without optimization. Ensure operations handle edge cases like dimension mismatches and empty tensors. Add broadcasting support incrementally.
