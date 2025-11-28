# Issue #2073: Fix BCE Backward Gradient Computation

## Objective

Fix the incorrect backward gradient computation in the Binary Cross-Entropy (BCE) loss function that was causing massive gradient mismatches during numerical gradient checking.

## Problem

The `binary_cross_entropy_backward` function in `/home/mvillmow/ml-odyssey/shared/core/loss.mojo` was computing the gradient using the MSE formula `(predictions - targets)` instead of the correct BCE formula.

**Test Failure**:
- Test: `test_binary_cross_entropy_backward_gradient` in `tests/shared/core/test_losses.mojo`
- Analytical gradient: -0.30000001192092896
- Numerical gradient: -1.4287233352661133
- Difference: 1.1287233233451843 (4.76x mismatch!)

## Root Cause

The implementation used a "simplified" gradient formula that doesn't apply to BCE. The gradient was computed as:
```
∂BCE/∂p = (p - y)  # WRONG - This is MSE gradient!
```

Instead of the correct BCE gradient:
```
∂BCE/∂p = -y/p + (1-y)/(1-p) = (p - y) / (p(1-p))
```

## Solution

Updated `binary_cross_entropy_backward` to compute the correct gradient formula:
```mojo
∂BCE/∂p = (p - y) / (p(1-p) + epsilon)
```

Where:
- **Numerator**: `(predictions - targets)` - difference between predicted and target
- **Denominator**: `predictions * (1 - predictions) + epsilon` - variance-like term that scales the gradient
- **Epsilon**: 1e-7 for numerical stability to prevent division by near-zero values

## Changes Made

**File**: `/home/mvillmow/ml-odyssey/shared/core/loss.mojo`

1. Updated docstring to explain the correct formula derivation
2. Implemented full gradient computation with:
   - Numerator: `(predictions - targets)`
   - Denominator: `predictions * (1 - predictions) + epsilon`
   - Division operation and upstream gradient multiplication

## Mathematical Correctness

The BCE loss is defined as:
```
L = -[y*log(p) + (1-y)*log(1-p)]
```

Taking the derivative with respect to p:
```
∂L/∂p = -[y/p - (1-y)/(1-p)]
      = -[y(1-p) - (1-y)p] / (p(1-p))
      = -[y - yp - p + yp] / (p(1-p))
      = -[y - p] / (p(1-p))
      = (p - y) / (p(1-p))
```

## Verification

The fix ensures:
- Analytical gradients match numerical gradients (finite difference check)
- Gradient values are mathematically correct
- Numerical stability via epsilon in denominator
- Gradient shape matches predictions shape

## References

- Binary Cross-Entropy: https://en.wikipedia.org/wiki/Cross_entropy#Binary_cross_entropy
- Gradient derivation verified by numerical gradient checking
- Test file: `tests/shared/core/test_losses.mojo::test_binary_cross_entropy_backward_gradient`
