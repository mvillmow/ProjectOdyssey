# Issue #2067: Fix Batch Normalization Backward Gradient Bug

## Objective

Fix the batch normalization backward gradient computation that is failing numerical gradient validation. The analytical gradient diverges significantly from the numerical gradient, indicating a mathematical error in the backward pass implementation.

## Problem Statement

The test `test_batch_norm2d_backward_gradient_input()` fails with:

```
Batch norm gradient w.r.t. input: gradient mismatch at index 0
  Analytical: -4.76837158203125e-07
  Numerical:  -0.0035762786865234375
  Difference: 0.0035758018493652344
```

This indicates the analytical gradient (computed by the backward function) is off by a factor of ~7500x.

## Root Cause Analysis

The batch normalization backward pass implements the chain rule through four key quantities:

1. **x_norm** = (x - mean) / std
2. **y** = gamma * x_norm + beta
3. **mean** = E[x]
4. **var** = E[(x - mean)^2]

The backward pass must propagate gradients through all four intermediate values. The bug is in how `grad_mean` is computed.

### Current Implementation (Buggy)

In `/home/mvillmow/ml-odyssey/shared/core/normalization.mojo` lines 549-550 (float32 training) and 637-638 (float64 training):

```mojo
# Compute grad_var and grad_mean
var grad_var = Float32(0.0)
var grad_mean = Float32(0.0)

for b in range(batch):
    for h in range(height):
        for w in range(width):
            var idx = b * (channels * height * width) + c * (height * width) + h * width + w
            var grad_out = grad_output_ptr.bitcast[Float32]()[idx]
            var x_val = x_ptr.bitcast[Float32]()[idx]

            var grad_x_norm = grad_out * gamma_val
            grad_var += grad_x_norm * (x_val - mean_val) * Float32(-0.5) * pow_scalar_f32(var_val + Float32(epsilon), Float32(-1.5))
            grad_mean += grad_x_norm * Float32(-1.0) / std

# BUG: This line contributes zero!
grad_mean += grad_var * (Float32(-2.0) * Float32(0.0))  # mean(x - mean) = 0
```

The issue is on the last line: `grad_var * (Float32(-2.0) * Float32(0.0))` is always zero because we're multiplying by `0.0` instead of computing `mean(x - mean)`.

### Correct Mathematical Formulation

The backward pass needs to account for how `mean` and `var` interact:

1. **grad_x_norm** = grad_output * gamma
2. **grad_var** = sum_i(grad_x_norm[i] * (x[i] - mean) * -0.5 * (var + eps)^(-3/2))
3. **grad_mean_from_norm** = sum_i(grad_x_norm[i] * -1/sqrt(var + eps))
4. **grad_mean_from_var** = grad_var * sum_i(-2 * (x[i] - mean)) / N
5. **grad_mean** = grad_mean_from_norm + grad_mean_from_var
6. **grad_input[i]** = grad_x_norm[i] / sqrt(var + eps) +
                        grad_var * 2 * (x[i] - mean) / N +
                        grad_mean / N

The bug is that we're computing `grad_mean_from_var` incorrectly.

## Solution

The gradient of mean with respect to variance contribution needs to properly account for the sum of deviations:

1. Compute `grad_mean_from_var` properly as `grad_var * (-2.0 / N) * sum_i(x[i] - mean)`
2. But since `sum_i(x[i] - mean) = 0` mathematically, this doesn't contribute
3. However, we still need to propagate `grad_mean` correctly in the final gradient computation

The key insight is that `grad_mean` (as a scalar) needs to be added per-element when computing `grad_input`.

## Implementation Fix

The fix reorganizes the backward gradient computation to:

1. Compute `grad_var` as before (correct)
2. Compute `grad_mean` from normalization term only (not variance term, since mean of deviations is zero)
3. In the grad_input computation, properly account for both the direct contribution and the scaled contributions from variance

## Files to Modify

- `/home/mvillmow/ml-odyssey/shared/core/normalization.mojo` - Fix batch_norm2d_backward function (lines 534-565 for float32, 616-653 for float64)

## Test Coverage

The failing test:
- `/home/mvillmow/ml-odyssey/tests/shared/core/test_normalization.mojo:test_batch_norm2d_backward_gradient_input()`

Uses numerical gradient validation via finite differences with epsilon=1e-4, which provides a gold-standard check.

## Success Criteria

- Test `test_batch_norm2d_backward_gradient_input()` passes
- Analytical gradient matches numerical gradient to within tolerance (rtol=1e-2, atol=1e-5)
- All other batch norm tests continue to pass
- Both float32 and float64 implementations fixed

## Technical Details

### Batch Norm Forward Pass (Reminder)

```
mean = E[x] over (batch, height, width) per channel
var = E[(x - mean)^2]
x_norm = (x - mean) / sqrt(var + eps)
output = gamma * x_norm + beta
```

### Batch Norm Backward Pass (Corrected)

```
# Step 1: Gradients w.r.t. scale and shift (correct)
grad_beta = sum(grad_output)
grad_gamma = sum(grad_output * x_norm)

# Step 2: Gradients through normalization (correct)
grad_x_norm = grad_output * gamma

# Step 3: Gradients through variance computation
grad_var = sum(grad_x_norm * (x - mean) * -0.5 * (var + eps)^(-3/2))

# Step 4: Gradients through mean computation
grad_mean = sum(grad_x_norm * -1/sqrt(var + eps))
# Note: Contribution from grad_var is zero since mean(x - mean) = 0

# Step 5: Final gradient w.r.t. input
grad_input = grad_x_norm / sqrt(var + eps) +
             grad_var * 2 * (x - mean) / N +
             grad_mean / N
```

where N = batch * height * width (number of elements per channel).

## Implementation Status

Reorganized batch norm backward for clarity and correctness:

1. **Code Refactoring** (Completed):
   - Separated gradient computation into two clear passes
   - First pass: Accumulate `grad_var` and `grad_mean` as scalar aggregates across spatial elements
   - Second pass: Compute `grad_input` for each element using three terms
   - Removed the problematic zero-contribution line
   - Both float32 and float64 implementations updated identically

2. **Mathematical Review** (In Progress):
   - The backward formula implements the chain rule through mean and variance
   - Test failure suggests numerical issue in gradient computation (7500x magnitude error)
   - Further investigation needed to identify root cause:
     - Could be in formula derivation
     - Could be in scaling/normalization factor
     - Could be in test setup expectations

## Code Changes Made

- Removed: `grad_mean += grad_var * (Float32(-2.0) * Float32(0.0))`  (line 550, 638)
- Reorganized gradient computation into explicit two-pass structure
- Clarified comments explaining each gradient component
- Both float32 and float64 paths updated consistently

## Next Steps for Debugging

1. Review batch norm backward mathematics from authoritative sources
2. Verify numerical gradient checker correctness
3. Consider simpler test case to isolate the issue
4. Trace through example computation step-by-step
5. Compare with PyTorch/TensorFlow implementations

## References

- Original batch norm paper: Ioffe & Szegedy (2015) "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"
- Gradient flow explanation: https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
