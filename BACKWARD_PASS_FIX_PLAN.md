# ExTensor Backward Pass Fix Plan

**Date**: 2025-11-18
**Status**: Review Complete - Ready for Implementation
**Completion**: ~40% → Target: 95%

---

## Executive Summary

The comprehensive review identified **3 critical blockers** and **20+ missing backward passes**. Current implementation will fail for:
- ❌ Any network with bias terms (broadcasting)
- ❌ Networks with exp/log operations
- ❌ Max pooling layers
- ❌ Stable training (division by zero)

**Estimated Fix Time**: 2-3 weeks focused development

---

## Critical Issues Found

### 🔴 BLOCKER #1: Broadcasting Not Implemented
**File**: `src/extensor/arithmetic.mojo`
**Lines**: 498-650
**Functions**: `add_backward`, `subtract_backward`, `multiply_backward`, `divide_backward`

**Problem**:
```python
# Forward with broadcasting
a = ones([3, 1, 5])  # Bias term
b = ones([3, 4, 5])  # Activations
c = add(a, b)        # Shape (3, 4, 5) ✓

# Backward - BROKEN!
grad_c = ones([3, 4, 5])
grad_a, grad_b = add_backward(grad_c, a.shape(), b.shape())
# grad_a is (3, 4, 5) but should be (3, 1, 5) ❌
```

**Impact**: Blocks ANY network with bias terms, batch norm, broadcasting

### 🔴 BLOCKER #2: Division by Zero
**File**: `src/extensor/arithmetic.mojo`
**Lines**: 633-648

**Problem**: `divide_backward` doesn't check for division by zero
```mojo
var b_squared = multiply(b, b)  # b² could be 0
var grad_b = divide(temp, b_squared)  # NaN/Inf!
```

**Impact**: Training instability, gradient explosion

### 🔴 BLOCKER #3: No Shape Validation
**All backward passes**

**Problem**: No validation that inputs have compatible shapes
**Impact**: Silent bugs, memory corruption

---

## Missing Backward Passes (20+)

### Critical for Training (P2 HIGH)

**Elementwise Math** - 15 operations in `elementwise_math.mojo`:
1. ✅ `abs` - `grad * sign(x)`
2. ✅ `exp` - `grad * exp(x)` or `grad * output`
3. ✅ `log` - `grad / x`
4. ✅ `sqrt` - `grad / (2 * sqrt(x))` or `grad / (2 * output)`
5. ⚠️ `sin` - `grad * cos(x)`
6. ⚠️ `cos` - `grad * (-sin(x))`
7. ✅ `clip` - `grad * (min <= x <= max)`
8. ✅ `log10`, `log2` - `grad / (x * ln(base))`
9. ❌ `sign`, `ceil`, `floor`, `round`, `trunc` - Non-differentiable (grad = 0)

**Reductions** - 2 operations in `reduction.mojo`:
1. ✅ `max_reduce` - Gradient flows only to max element(s)
2. ✅ `min_reduce` - Gradient flows only to min element(s)

**Matrix** - 2 operations in `matrix.mojo`:
1. ⚠️ `dot` - Inner product gradient
2. ⚠️ `outer` - Outer product gradient

**Legend**: ✅ Critical | ⚠️ Nice to have | ❌ Non-differentiable

---

## Test Coverage Gaps

| Category | Forward | Backward | Tests | Coverage |
|----------|---------|----------|-------|----------|
| Activations | 7 | 7 | 7 | 100% ✅ |
| Matrix | 4 | 2 | 0 | 0% ❌ |
| Arithmetic | 7 | 4 | 0 | 0% ❌ |
| Reduction | 4 | 2 | 0 | 0% ❌ |
| Elementwise | 15 | 0 | 0 | 0% ❌ |

**Total Test Coverage**: 17.5% (7 / 40 operations)

---

## Prioritized Fix Plan

### 🔴 Phase 1: CRITICAL FIXES (Week 1)
**Goal**: Make training actually work

1. **Fix Broadcasting Reduction** (2 days)
   - Implement `reduce_broadcast_dims()` utility
   - Update `add_backward`, `subtract_backward`, `multiply_backward`, `divide_backward`
   - Algorithm: Sum gradient over dimensions that were broadcast

2. **Add Numerical Stability** (0.5 days)
   - Add epsilon to `divide_backward` (prevent /0)
   - Clamp values in `sqrt_backward` (prevent negative sqrt)
   - Add stability checks to reductions

3. **Add Shape Validation** (0.5 days)
   - Validate gradient shape matches output shape
   - Validate input shapes are compatible
   - Add descriptive error messages

**Deliverable**: Basic training works for simple networks

### 🟡 Phase 2: HIGH PRIORITY (Week 2)
**Goal**: Support common architectures

4. **Implement Elementwise Math Backward** (2 days)
   - `exp_backward`, `log_backward`, `sqrt_backward` - Day 1
   - `abs_backward`, `clip_backward`, `log10_backward`, `log2_backward` - Day 2

5. **Implement Max/Min Reduction Backward** (1 day)
   - `max_reduce_backward` - For max pooling
   - `min_reduce_backward` - For min pooling
   - Handle ties (multiple max/min elements)

6. **Add Comprehensive Tests** (2 days)
   - Matrix backward tests (matmul, transpose)
   - Arithmetic backward tests (all 4 ops, with broadcasting!)
   - Reduction backward tests (sum, mean, max, min)
   - Elementwise backward tests (exp, log, sqrt, abs)
   - Integration test (multi-layer network)

**Deliverable**: Can train CNNs, ResNets, common architectures

### 🟢 Phase 3: MEDIUM PRIORITY (Week 3)
**Goal**: Complete the framework

7. **Implement Remaining Math Backward** (1 day)
   - `sin_backward`, `cos_backward` - Trigonometric
   - `power_backward` - General power operation

8. **Implement Matrix Ops Backward** (0.5 days)
   - `dot_backward` - 1D inner product
   - `outer_backward` - 1D outer product

9. **Add Numerical Gradient Checking** (1 day)
   - Implement `check_gradient(forward_fn, backward_fn, x)`
   - Add to all backward pass tests
   - Verify mathematical correctness

**Deliverable**: Production-ready framework with validation

---

## Implementation Guide

### Broadcasting Reduction Algorithm

```mojo
fn reduce_broadcast_dims(grad: ExTensor, original_shape: DynamicVector[Int], broadcast_shape: DynamicVector[Int]) raises -> ExTensor:
    """Reduce gradient from broadcast shape back to original shape.

    Example:
        original: [3, 1, 5]
        broadcast: [3, 4, 5]
        grad: [3, 4, 5]
        result: [3, 1, 5] by summing over dimension 1
    """
    var result = grad

    # For each dimension
    for i in range(len(original_shape)):
        # If dimension was broadcast (size 1 → size N)
        if original_shape[i] == 1 and broadcast_shape[i] > 1:
            # Sum over this dimension, keep dims
            result = sum(result, axis=i, keepdims=True)

    # Handle prepended dimensions (when original had fewer dims)
    if len(original_shape) < len(broadcast_shape):
        let dims_to_sum = len(broadcast_shape) - len(original_shape)
        for i in range(dims_to_sum):
            result = sum(result, axis=0, keepdims=False)

    return result
```

### Max/Min Reduction Backward

```mojo
fn max_reduce_backward(grad_output: ExTensor, x: ExTensor, axis: Int) raises -> ExTensor:
    """Gradient for max reduction.

    Only the maximum element(s) receive gradient.
    If multiple elements are max, gradient is split equally.
    """
    var result = zeros(x.shape(), x.dtype())

    # Find max value along axis
    var max_val = max_reduce(x, axis)

    # Create mask: 1.0 where x == max_val, 0.0 elsewhere
    var mask = equal(x, broadcast(max_val, x.shape()))

    # Count how many elements are max (for splitting gradient)
    var count = sum(mask, axis=axis, keepdims=True)

    # Broadcast gradient and divide by count
    var grad_broadcast = broadcast(grad_output, x.shape())
    result = divide(multiply(grad_broadcast, mask), count)

    return result
```

### Numerical Gradient Checking

```mojo
fn check_gradient(
    forward_fn: fn(ExTensor) -> ExTensor,
    backward_fn: fn(ExTensor, ExTensor) -> ExTensor,
    x: ExTensor,
    epsilon: Float64 = 1e-5
) raises -> Bool:
    """Verify backward pass using numerical differentiation."""

    # Analytical gradient
    var y = forward_fn(x)
    var grad_out = ones_like(y)
    var grad_analytical = backward_fn(grad_out, x)

    # Numerical gradient
    var grad_numerical = zeros_like(x)
    for i in range(x.numel()):
        # f(x + eps)
        x[i] += epsilon
        var y_plus = forward_fn(x)

        # f(x - eps)
        x[i] -= 2 * epsilon
        var y_minus = forward_fn(x)

        # Restore x
        x[i] += epsilon

        # Gradient: (f(x+eps) - f(x-eps)) / (2*eps)
        grad_numerical[i] = (y_plus - y_minus) / (2 * epsilon)

    # Compare
    return allclose(grad_analytical, grad_numerical, rtol=1e-3, atol=1e-5)
```

---

## Testing Strategy

### Test Categories

1. **Unit Tests** - Each backward function
   - Shape correctness
   - Value correctness (known examples)
   - Edge cases (zeros, ones, large values)

2. **Numerical Tests** - Verify against finite differences
   - All backward functions
   - Random inputs
   - Multiple data types

3. **Broadcasting Tests** - CRITICAL
   - All broadcast patterns: (1,) + (N,), (M, 1) + (M, N), etc.
   - Verify gradient shape matches input shape

4. **Integration Tests** - Multi-operation chains
   - Simple network: x → matmul → relu → matmul → softmax
   - Verify gradient flows correctly through all layers

### Test Files to Create

- `tests/extensor/test_matrix_backward.mojo` - Matrix op gradients
- `tests/extensor/test_arithmetic_backward.mojo` - Arithmetic op gradients + broadcasting!
- `tests/extensor/test_reduction_backward.mojo` - Reduction gradients
- `tests/extensor/test_elementwise_backward.mojo` - Math op gradients
- `tests/extensor/test_gradient_checking.mojo` - Numerical verification framework

---

## Success Criteria

### Phase 1 Complete
- ✅ Broadcasting works correctly (verified with tests)
- ✅ No division by zero errors
- ✅ Shape validation catches bugs
- ✅ Simple MLP trains without errors

### Phase 2 Complete
- ✅ All common operations have backward passes
- ✅ 100+ backward pass tests
- ✅ Numerical gradient checking passes
- ✅ Can train ResNet-18 architecture

### Phase 3 Complete
- ✅ All operations have backward passes
- ✅ Complete test coverage
- ✅ Production-ready stability
- ✅ Documentation for all backward passes

---

## Risk Assessment

**Current Risk**: 🔴 HIGH - Will fail in production

**After Phase 1**: 🟡 MEDIUM - Basic training works, advanced features missing

**After Phase 2**: 🟢 LOW - Production ready for common architectures

**After Phase 3**: ✅ MINIMAL - Complete, tested, stable framework

---

## Resources Needed

- **Time**: 2-3 weeks focused development
- **Testing**: Access to numerical gradient checking framework
- **Validation**: Small neural network for integration testing

---

**Next Steps**:
1. ✅ Review complete
2. ⏳ Begin Phase 1: Fix broadcasting reduction
3. ⏳ Add numerical stability
4. ⏳ Add shape validation
5. ⏳ Phase 2 & 3 implementation

**Report References**:
- Full review: `/home/user/ml-odyssey/backward_pass_review.md`
- This plan: `/home/user/ml-odyssey/BACKWARD_PASS_FIX_PLAN.md`
