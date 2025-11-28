# Issue #2057 Analysis: Arithmetic Backward Gradient Computation

## Problem Summary

The backward pass implementations for arithmetic operations in `shared/core/arithmetic.mojo` have incorrect gradient computations, causing 11 test failures in `tests/shared/core/test_arithmetic_backward.mojo`.

## Root Causes Identified

### 1. reduce_broadcast_dims Logic Error

The `_reduce_broadcast_dims` helper function has a subtle bug in how it handles broadcast dimension reduction after prepended dimensions are summed.

**Issue Location**: Lines 393-396

```mojo
# CURRENT (WRONG)
for i in range(min(orig_ndim, grad_ndim)):
    var dim_idx = i if orig_ndim < grad_ndim else i + (grad_ndim - orig_ndim)
    if i < orig_ndim and original_shape[i] == 1 and i < len(result.shape()) and result.shape()[i] > 1:
        result = sum(result, axis=i, keepdims=True)
```

**Problem**: After summing prepended dimensions in the first loop, `result.shape()` changes, but the loop still uses the original `i` index. For example:

- Input: grad shape [2, 3, 4], original shape [3, 4]
- After first loop: result shape [3, 4]
- Second loop tries to check if original_shape[0] (index 3) == 1, but now result has only 2 dims!

**Fix Required**: After handling prepended dimensions, need to iterate with adjusted indices.

### 2. Type Precision Loss in Negation

The `subtract_backward` function uses `_get_float64` and `_set_float64` to negate gradients, but this can lose precision when the tensor is float32.

**Issue Location**: Lines 463-464

```mojo
# CURRENT - Uses float64 conversion
neg_grad._set_float64(i, -grad_output._get_float64(i))
```

**Problem**: Converting float32 → float64 → float32 loses the original precision and can cause gradient mismatches in numerical gradient checking.

**Fix Required**: Use type-aware negation or create tensor directly.

### 3. Similar Issue in divide_backward

The `divide_backward` function has the same float64 conversion issues on lines 545-546 and 554-555.

## Test Expectations

### Simple Cases (Same Shape)

For tensors with same shape:
- **add_backward**: grad_a = grad_output, grad_b = grad_output
- **subtract_backward**: grad_a = grad_output, grad_b = -grad_output
- **multiply_backward**: grad_a = grad_output * b, grad_b = grad_output * a
- **divide_backward**: grad_a = grad_output / b, grad_b = -grad_output * a / b²

### Broadcast Cases

When b broadcasts from [3] to [2, 3]:
- The gradient for b must be reduced back to shape [3]
- Example: add_backward with [2,3] + [3] should produce grad_b of shape [3] with summed values

## Implementation Plan

1. **Fix reduce_broadcast_dims**:
   - Properly handle the case where prepended dimensions are removed
   - Ensure loop indices are correct after shape changes
   - Consider using a more robust approach

2. **Fix type conversions**:
   - Use type-safe operations instead of float64 conversion
   - Preserve original dtypes throughout computation

3. **Verify with gradient checking**:
   - Run numerical gradient checks for all operations
   - Ensure analytical gradients match numerical gradients

## Files to Modify

- `/home/mvillmow/ml-odyssey/shared/core/arithmetic.mojo`
  - `_reduce_broadcast_dims()` - Fix broadcast reduction logic
  - `subtract_backward()` - Fix type-safe negation
  - `divide_backward()` - Fix type-safe operations

## Testing Strategy

1. Run simple same-shape tests
2. Run scalar broadcast tests
3. Run general broadcast tests
4. Run numerical gradient checking tests
5. Verify all 23 tests pass
