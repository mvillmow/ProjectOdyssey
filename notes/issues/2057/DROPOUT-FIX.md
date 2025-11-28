# Fix #3: Dropout Backward Gradient Computation

## Problem Statement

**Test**: `test_dropout_backward_gradient()` in `tests/shared/core/test_dropout.mojo`

**Error**:
```
Unhandled exception - Gradient check failed for float32: gradient mismatch at index 0
```

**Files Affected**:
- `tests/shared/core/test_dropout.mojo` (lines 196-224)
- `shared/core/dropout.mojo` (lines 224-257)

## Technical Analysis

### Dropout Mathematics

**Forward Pass** (`dropout()` lines 16-96):
```
output = x * mask / (1 - p)
```

Where:
- `mask[i] = 1.0` with probability `(1-p)` (element kept)
- `mask[i] = 0.0` with probability `p` (element dropped)
- Division by `(1-p)` ensures expected value unchanged

**Backward Pass** (`dropout_backward()` lines 224-257):
```
grad_input = grad_output * mask / (1 - p)
```

This should be correct because:
- Gradient flows only through non-dropped elements (where mask = 1)
- Scaling by `1/(1-p)` matches forward pass scaling
- Chain rule: `∂L/∂x = ∂L/∂output * ∂output/∂x = ∂L/∂output * mask/(1-p)`

### Test Structure Analysis

**Test Code** (lines 196-224):

```mojo
fn test_dropout_backward_gradient() raises:
    var x = zeros(shape, DType.float32)
    # Set test values...

    # Forward pass - creates mask with seed=42
    var (output, mask) = dropout(x, p=0.3, training=True, seed=42)
    var grad_out = ones_like(output)

    # Forward wrapper for gradient checking
    fn forward(x: ExTensor) raises escaping -> ExTensor:
        var (out, _) = dropout(x, p=0.3, training=True, seed=42)
        return out

    # Backward wrapper - uses captured mask
    fn backward(grad: ExTensor, x: ExTensor) raises escaping -> ExTensor:
        return dropout_backward(grad, mask, p=0.3)

    # Numerical gradient checking
    check_gradient(forward, backward, x, grad_out, rtol=1e-3, atol=1e-6)
```

### Gradient Checking Process

`check_gradient()` (from `tests/helpers/gradient_checking.mojo`):

1. Computes analytical gradient: `analytical = backward(grad_out, x)`
2. For each input element i:
   - Perturb: `x_plus = x with x[i] += epsilon`
   - Perturb: `x_minus = x with x[i] -= epsilon`
   - Forward: `out_plus = forward(x_plus)`
   - Forward: `out_minus = forward(x_minus)`
   - Loss: `loss_plus = sum(out_plus * grad_out)`
   - Loss: `loss_minus = sum(out_minus * grad_out)`
   - Numerical gradient: `grad[i] = (loss_plus - loss_minus) / (2 * epsilon)`
3. Compare analytical vs numerical with tolerances

### Critical Insight: Fixed Seed Behavior

**The Problem**:

With `seed=42` fixed in the forward wrapper:
- Every call to `forward(x_plus)` generates SAME mask (mask_A)
- Every call to `forward(x_minus)` generates SAME mask (mask_A)
- The captured `mask` from line 210 is also generated with seed=42 (mask_A)

This means ALL masks are identical!

**Expected Behavior**:

Since masks are identical:
```
out_plus[j] = x_plus[j] * mask[j] / (1-p)
out_minus[j] = x_minus[j] * mask[j] / (1-p)

loss_plus = Σ(x_plus[j] * mask[j] / (1-p) * grad_out[j])
loss_minus = Σ(x_minus[j] * mask[j] / (1-p) * grad_out[j])

For element i:
loss_plus includes: (x[i] + ε) * mask[i] / (1-p) * grad_out[i]
loss_minus includes: (x[i] - ε) * mask[i] / (1-p) * grad_out[i]

numerical_grad[i] = [(x[i] + ε) * mask[i] / (1-p) * grad_out[i] -
                     (x[i] - ε) * mask[i] / (1-p) * grad_out[i]] / (2ε)
                  = [2ε * mask[i] / (1-p) * grad_out[i]] / (2ε)
                  = mask[i] / (1-p) * grad_out[i]

analytical_grad[i] = dropout_backward(...)[i]
                   = grad_out[i] * mask[i] / (1-p)
```

**These should match exactly!**

## Hypotheses for Failure

### Hypothesis 1: RNG State Issue
Despite same seed, the RNG state might not reset correctly, causing different masks.

**Test**: Print masks from forward calls and compare.

### Hypothesis 2: Mask Capture Bug
The captured `mask` variable might not be correctly shared with the backward wrapper due to escaping closure issues.

**Test**: Print mask values in backward function.

### Hypothesis 3: Implementation Bug
The `dropout_backward()` implementation might not correctly apply the formula.

**Current Implementation** (lines 253-257):
```mojo
var masked_grad = multiply(grad_output, mask)
var scale = 1.0 / (1.0 - p)
var scale_tensor = full_like(grad_output, scale)
return multiply(masked_grad, scale_tensor)
```

This looks correct: `grad_output * mask * (1/(1-p))`

**Test**: Manually verify multiply operations.

### Hypothesis 4: Numerical Precision
Float32 precision might cause issues with the gradient checking tolerances.

**Test**: Use Float64 or looser tolerances.

### Hypothesis 5: Test Bug
The test itself might be structured incorrectly.

**Issue**: The backward wrapper captures `mask` from line 210, but gradient checking might expect the backward to recompute or receive a different mask.

**But**: Since seed is fixed, all masks should be identical.

## Investigation Plan

### Step 1: Add Diagnostic Logging

Create `scripts/diagnose_dropout.mojo` to:
1. Print input values
2. Print mask values from multiple forward calls with same seed
3. Print output values
4. Print gradient values
5. Manually compute numerical gradient
6. Compare analytical vs numerical for each element

### Step 2: Verify Mask Consistency

Check that:
```mojo
mask_1 = dropout(x, p=0.3, seed=42)
mask_2 = dropout(x, p=0.3, seed=42)
assert all(mask_1 == mask_2)
```

### Step 3: Verify Implementation

Manually verify:
```mojo
grad = dropout_backward(grad_out, mask, p=0.3)
assert grad[i] == grad_out[i] * mask[i] / (1 - p)  for all i
```

### Step 4: Check Gradient Checking Logic

Verify `check_gradient()` is implemented correctly for this case.

## Potential Fixes

### Fix 1: Test Structure
If the issue is test structure, modify test to not use fixed seed in forward wrapper.

### Fix 2: Implementation
If implementation has a bug, fix the formula application.

### Fix 3: Seed Handling
If RNG state isn't reset properly, fix seed application in `dropout()`.

## Success Criteria

- [ ] `test_dropout_backward_gradient()` passes
- [ ] All other dropout tests continue passing
- [ ] Numerical gradient check passes with `rtol=1e-3, atol=1e-6`
- [ ] Clear documentation of fix reasoning

## Files to Modify

Likely:
- `shared/core/dropout.mojo` - Fix backward implementation if bug found
- `tests/shared/core/test_dropout.mojo` - Fix test structure if needed

## Next Steps

1. Run diagnostic script
2. Identify root cause
3. Implement fix
4. Verify all tests pass
5. Document findings
