# Dropout Backward Gradient Issue Analysis

## Problem Statement

Test `test_dropout_backward_gradient()` fails with:
```
Gradient check failed for float32: gradient mismatch at index 0
```

## Code Analysis

### Test Structure (lines 196-224)

```mojo
fn test_dropout_backward_gradient() raises:
    var x = zeros(shape, DType.float32)
    # Set non-uniform values...

    # Line 210: Create mask once
    var (output, mask) = dropout(x, p=0.3, training=True, seed=42)
    var grad_out = ones_like(output)

    # Line 214: Forward wrapper - REGENERATES mask with same seed
    fn forward(x: ExTensor) raises escaping -> ExTensor:
        var (out, _) = dropout(x, p=0.3, training=True, seed=42)
        return out

    # Line 219: Backward wrapper - uses CAPTURED mask
    fn backward(grad: ExTensor, x: ExTensor) raises escaping -> ExTensor:
        return dropout_backward(grad, mask, p=0.3)

    # Line 224: Gradient check
    check_gradient(forward, backward, x, grad_out, rtol=1e-3, atol=1e-6)
```

### Gradient Checking Process

`check_gradient` does:

1. For each input element i:
   - Create `x_plus = x with x[i] += epsilon`
   - Create `x_minus = x with x[i] -= epsilon`
   - Call `forward(x_plus)` → generates mask with seed=42
   - Call `forward(x_minus)` → generates mask with seed=42
   - Compute numerical gradient

2. Call `backward(grad_out, x)` → uses mask from line 210

3. Compare numerical vs analytical

### Key Insight: Seed Behavior

With fixed seed=42:
- Every call to `dropout(., p=0.3, training=True, seed=42)` generates SAME mask
- Mask is independent of input values
- All forward passes during gradient checking use identical masks

This means:
- `output_plus = (x + ε) * mask / (1-p)` with mask from seed 42
- `output_minus = (x - ε) * mask / (1-p)` with mask from seed 42
- `numerical_grad[i] = (loss_plus - loss_minus) / (2ε)`

Where:
- `loss_plus = sum(output_plus * grad_out)`
- `loss_minus = sum(output_minus * grad_out)`

Since mask is same:
- `loss_plus = sum((x + ε) * mask / (1-p) * grad_out)`
- `loss_minus = sum((x - ε) * mask / (1-p) * grad_out)`

For element i:
- `loss_plus = ... + (x[i] + ε) * mask[i] / (1-p) * grad_out[i] + ...`
- `loss_minus = ... + (x[i] - ε) * mask[i] / (1-p) * grad_out[i] + ...`
- `numerical_grad[i] = (2ε * mask[i] / (1-p) * grad_out[i]) / (2ε)`
- `numerical_grad[i] = mask[i] / (1-p) * grad_out[i]`

And analytical gradient:
- `grad_input = dropout_backward(grad_out, mask, p=0.3)`
- `grad_input = grad_out * mask / (1-p)`
- `grad_input[i] = grad_out[i] * mask[i] / (1-p)`

**These should match!**

## Hypothesis

The issue might be:

1. **Mask mismatch**: Despite same seed, masks might differ due to RNG state
2. **Scaling issue**: Implementation might not correctly apply `1/(1-p)` scaling
3. **Numerical precision**: Float32 precision issues in gradient checking
4. **Test bug**: Test itself might be incorrect

## Next Steps

1. Add diagnostic logging to see actual values
2. Verify mask consistency across calls
3. Check scaling factor application
4. Validate gradient computation step-by-step
