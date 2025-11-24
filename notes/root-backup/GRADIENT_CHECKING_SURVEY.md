# Gradient Checking Retrofit Survey

## Overview

This document surveys all backward passes in the codebase to identify which ones have numerical gradient checking
coverage and which ones need it.

## Summary

### Backward Passes Implemented

| Module | Backward Function | Test File | Has Gradient Checking | Status |
|--------|-------------------|-----------|----------------------|--------|
| **activation.mojo** | relu_backward | test_activations.mojo | ✓ YES | Complete |
| **activation.mojo** | leaky_relu_backward | test_activations.mojo | ✓ YES | Complete |
| **activation.mojo** | prelu_backward | test_activations.mojo | ✓ YES | Complete |
| **activation.mojo** | sigmoid_backward | test_activations.mojo | ✓ YES | Complete |
| **activation.mojo** | tanh_backward | test_activations.mojo | ✓ YES | Complete |
| **activation.mojo** | softmax_backward | test_activations.mojo | ✓ YES | Complete |
| **activation.mojo** | gelu_backward | test_activations.mojo | ✗ NO | **NEEDS WORK** |
| **activation.mojo** | swish_backward | test_activations.mojo | ✗ NO | **NEEDS WORK** |
| **activation.mojo** | mish_backward | test_activations.mojo | ✗ NO | **NEEDS WORK** |
| **activation.mojo** | elu_backward | test_activations.mojo | ✓ YES | Complete |
| **arithmetic.mojo** | add_backward | test_arithmetic_backward.mojo | ✗ NO | **NEEDS WORK** |
| **arithmetic.mojo** | subtract_backward | test_arithmetic_backward.mojo | ✗ NO | **NEEDS WORK** |
| **arithmetic.mojo** | multiply_backward | test_arithmetic_backward.mojo | ✗ NO | **NEEDS WORK** |
| **arithmetic.mojo** | divide_backward | test_arithmetic_backward.mojo | ✗ NO | **NEEDS WORK** |
| **conv.mojo** | conv2d_backward | test_backward.mojo | ✓ YES | Complete |
| **dropout.mojo** | dropout_backward | test_dropout.mojo | ✗ NO | **NEEDS WORK** |
| **dropout.mojo** | dropout2d_backward | test_dropout.mojo | ✗ NO | **NEEDS WORK** |
| **elementwise.mojo** | exp_backward | test_elementwise.mojo | ✗ NO | **NEEDS WORK** |
| **elementwise.mojo** | log_backward | test_elementwise.mojo | ✗ NO | **NEEDS WORK** |
| **elementwise.mojo** | sqrt_backward | test_elementwise.mojo | ✗ NO | **NEEDS WORK** |
| **elementwise.mojo** | abs_backward | test_elementwise.mojo | ✗ NO | **NEEDS WORK** |
| **elementwise.mojo** | clip_backward | test_elementwise.mojo | ✗ NO | **NEEDS WORK** |
| **elementwise.mojo** | log10_backward | test_elementwise.mojo | ✗ NO | **NEEDS WORK** |
| **elementwise.mojo** | log2_backward | test_elementwise.mojo | ✗ NO | **NEEDS WORK** |
| **linear.mojo** | linear_backward | test_backward.mojo | ✓ YES | Complete |
| **loss.mojo** | cross_entropy_backward | test_backward.mojo | ✓ YES | Complete |
| **matrix.mojo** | matmul_backward | test_matrix.mojo | ✗ PARTIAL | **NEEDS WORK** |
| **matrix.mojo** | transpose_backward | test_matrix.mojo | ✗ PARTIAL | **NEEDS WORK** |
| **pooling.mojo** | maxpool2d_backward | test_backward.mojo | ✓ YES | Complete |
| **pooling.mojo** | avgpool2d_backward | test_backward.mojo | ✓ YES | Complete |
| **reduction.mojo** | sum_backward | None | ✗ NO | **NEEDS WORK** |
| **reduction.mojo** | mean_backward | None | ✗ NO | **NEEDS WORK** |
| **reduction.mojo** | max_reduce_backward | None | ✗ NO | **NEEDS WORK** |
| **reduction.mojo** | min_reduce_backward | None | ✗ NO | **NEEDS WORK** |

## Status Breakdown

### Already Complete (8 backward passes)

1. relu_backward - test_activations.mojo
2. leaky_relu_backward - test_activations.mojo
3. prelu_backward - test_activations.mojo
4. sigmoid_backward - test_activations.mojo
5. tanh_backward - test_activations.mojo
6. softmax_backward - test_activations.mojo
7. elu_backward - test_activations.mojo
8. linear_backward - test_backward.mojo
9. conv2d_backward - test_backward.mojo
10. cross_entropy_backward - test_backward.mojo
11. maxpool2d_backward - test_backward.mojo
12. avgpool2d_backward - test_backward.mojo

### Need Gradient Checking (25 backward passes)

#### Activations (3)

- gelu_backward - test_activations.mojo
- swish_backward - test_activations.mojo
- mish_backward - test_activations.mojo

#### Arithmetic (4)

- add_backward - test_arithmetic_backward.mojo
- subtract_backward - test_arithmetic_backward.mojo
- multiply_backward - test_arithmetic_backward.mojo
- divide_backward - test_arithmetic_backward.mojo

#### Elementwise (7)

- exp_backward - test_elementwise.mojo
- log_backward - test_elementwise.mojo
- sqrt_backward - test_elementwise.mojo
- abs_backward - test_elementwise.mojo
- clip_backward - test_elementwise.mojo
- log10_backward - test_elementwise.mojo
- log2_backward - test_elementwise.mojo

#### Dropout (2)

- dropout_backward - test_dropout.mojo
- dropout2d_backward - test_dropout.mojo

#### Matrix (2)

- matmul_backward - test_matrix.mojo
- transpose_backward - test_matrix.mojo

#### Reduction (4) - NO TEST FILE YET

- sum_backward - NEEDS NEW FILE
- mean_backward - NEEDS NEW FILE
- max_reduce_backward - NEEDS NEW FILE
- min_reduce_backward - NEEDS NEW FILE

## Notes

- **Arithmetic**: test_arithmetic_backward.mojo exists but backward tests don't use check_gradient
- **Elementwise**: test_elementwise.mojo exists but backward tests only verify hard-coded values
- **Reduction**: No dedicated test file exists yet
- **Matrix**: Tests exist but are minimal
- **Dropout**: Tests exist but don't include gradient checking

## Work Plan

1. Add gradient checking tests to test_activations.mojo for GELU, Swish, Mish
2. Add gradient checking tests to test_arithmetic_backward.mojo for all 4 arithmetic ops
3. Add gradient checking tests to test_elementwise.mojo for all 7 elementwise ops
4. Add gradient checking tests to test_dropout.mojo for dropout and dropout2d
5. Add gradient checking tests to test_matrix.mojo for matmul and transpose
6. Create new test_reduction.mojo with gradient checking for all 4 reduction ops

All modifications should follow the established pattern:

- Use non-uniform test data
- Use rtol=1e-3, atol=1e-6 tolerances (adjusted as needed)
- Use the check_gradient helper function
- Keep tests focused and simple
