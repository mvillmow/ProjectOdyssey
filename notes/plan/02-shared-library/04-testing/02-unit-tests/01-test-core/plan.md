# Test Core

## Overview

Write comprehensive unit tests for all core operations including tensor ops, activations, initializers, and metrics. These tests verify mathematical correctness, handle edge cases, and ensure numerical stability of fundamental building blocks.

## Parent Plan

[../plan.md](../plan.md)

## Child Plans

None (leaf node)

## Inputs

- Implemented core operations
- Mathematical specifications for verification
- Edge case scenarios
- Performance baselines

## Outputs

- Tests for all tensor operations
- Tests for all activation functions
- Tests for all initializers
- Tests for all metrics
- Edge case coverage
- Numerical stability verification

## Steps

1. Write tests for tensor operations (arithmetic, matrix, reductions)
2. Create tests for activation functions
3. Build tests for initializers (verify distributions)
4. Implement tests for metrics (accuracy, loss, confusion matrix)
5. Add edge case tests (zeros, infinities, NaN, empty)

## Success Criteria

- [ ] All core operations have unit tests
- [ ] Tests verify mathematical correctness
- [ ] Edge cases are covered
- [ ] Numerical stability is tested

## Notes

Use known mathematical results for verification. Test with various tensor shapes and dtypes. Verify distributions for initializers statistically. Check metric calculations manually for small examples.
