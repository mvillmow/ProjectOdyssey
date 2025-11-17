# Core Operations

## Overview

Implement fundamental mathematical operations needed for machine learning. This includes tensor operations (arithmetic, matrix operations, reductions), activation functions (ReLU family, sigmoid/tanh, softmax/GELU), weight initializers (Xavier/Glorot, Kaiming/He, uniform/normal), and evaluation metrics (accuracy, loss tracking, confusion matrix).

## Parent Plan

[../plan.md](../plan.md)

## Child Plans

- [01-tensor-ops/plan.md](01-tensor-ops/plan.md)
- [02-activations/plan.md](02-activations/plan.md)
- [03-initializers/plan.md](03-initializers/plan.md)
- [04-metrics/plan.md](04-metrics/plan.md)

## Inputs

- Mojo tensor types and basic SIMD operations
- Mathematical specifications for each operation
- Understanding of numerical stability requirements

## Outputs

- Comprehensive tensor operation library
- Complete set of activation functions
- Standard weight initialization methods
- Evaluation metrics for model assessment

## Steps

1. Implement tensor operations for element-wise arithmetic, matrix operations, and reductions
2. Build activation functions with proper gradient handling
3. Create weight initializers following standard distributions
4. Implement metrics for tracking model performance

## Success Criteria

- [ ] All tensor operations produce correct results
- [ ] Activation functions are numerically stable
- [ ] Initializers follow proper statistical distributions
- [ ] Metrics accurately measure model performance
- [ ] All child plans are completed successfully

## Notes

Prioritize correctness and numerical stability over performance. Use simple, readable implementations. Ensure all operations handle edge cases properly (zeros, infinities, NaN values).
