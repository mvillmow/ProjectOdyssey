# Dtype Refactoring, Test Coverage, and Numerical Safety - Implementation Plan

## Overview

This document outlines the implementation plan for three major improvements to ML Odyssey:

1. **Dtype Refactoring** - Eliminate dtype branching with generics
2. **Test Coverage** - Implement empty test files and add edge cases
3. **Numerical Safety Mode** - Add optional NaN/Inf checking and gradient monitoring

## 1. Dtype Refactoring

### Current Problem

Every operation has dtype-specific branches:
```mojo
if tensor.dtype() == DType.float32:
    for i in range(size):
        var val = tensor._data.bitcast[Float32]()[i]
        result._data.bitcast[Float32]()[i] = process(val)
elif tensor.dtype() == DType.float64:
    for i in range(size):
        var val = tensor._data.bitcast[Float64]()[i]
        result._data.bitcast[Float64]()[i] = process(val)
# ... repeat for 10+ dtypes
```

This results in ~40 lines per operation, duplicated across all modules.

### Solution: Generic Dtype Dispatch

Create `shared/core/dtype_dispatch.mojo` with helpers:

```mojo
fn elementwise_unary[
    op: fn[dtype: DType](Scalar[dtype]) -> Scalar[dtype]
](tensor: ExTensor) -> ExTensor:
    """Apply unary operation with automatic dtype dispatch."""

    @parameter
    fn dispatch[dtype: DType]():
        var ptr = tensor._data.bitcast[Scalar[dtype]]()
        var result_ptr = result._data.bitcast[Scalar[dtype]]()
        for i in range(tensor.numel()):
            result_ptr[i] = op[dtype](ptr[i])

    # Runtime dispatch to compile-time specialized version
    if tensor.dtype() == DType.float32:
        dispatch[DType.float32]()
    elif tensor.dtype() == DType.float64:
        dispatch[DType.float64]()
    # ... etc
```

### Files to Refactor

Priority order:
1. `shared/core/activation.mojo` - 10 activation functions
2. `shared/core/elementwise.mojo` - 26 element-wise ops
3. `shared/core/arithmetic.mojo` - 12 arithmetic ops
4. `shared/core/normalization.mojo` - 2 normalization ops
5. `shared/core/dropout.mojo` - 2 dropout ops

### Benefits

- Reduce code from ~500 lines to ~100 lines (~80% reduction)
- Single source of truth for operations
- Easier to add new dtypes
- Compile-time specialization (no runtime overhead)

## 2. Numerical Safety Mode

### Design

Create `shared/core/numerical_safety.mojo` with:

```mojo
@parameter
fn check_tensor_safety[enable: Bool = False](
    tensor: ExTensor,
    name: String = "tensor"
) raises:
    """Check tensor for NaN/Inf values (compile-time optional)."""

    @parameter
    if enable:
        if has_nan(tensor):
            raise Error(name + " contains NaN values")
        if has_inf(tensor):
            raise Error(name + " contains Inf values")
    # If enable=False, this function compiles to nothing (zero overhead)

fn has_nan(tensor: ExTensor) -> Bool:
    """Check if tensor contains any NaN values."""
    # Implementation for all dtypes

fn has_inf(tensor: ExTensor) -> Bool:
    """Check if tensor contains any Inf values."""
    # Implementation for all dtypes

fn check_gradient_norm(
    gradients: List[ExTensor],
    max_norm: Float64 = 1000.0
) raises:
    """Check if gradient norm exceeds threshold (gradient explosion detection)."""
    var total_norm = compute_total_norm(gradients)
    if total_norm > max_norm:
        raise Error("Gradient explosion detected: norm = " + str(total_norm))

fn compute_total_norm(gradients: List[ExTensor]) -> Float64:
    """Compute L2 norm of all gradients."""
    # sqrt(sum(grad^2 for all grads))
```

### Usage Pattern

```mojo
# Training code
var output = linear(x, weights, bias)

# Add safety checks (compile-time enabled/disabled)
check_tensor_safety[enable=True](output, "linear_output")

var loss = cross_entropy(output, targets)
var grad = cross_entropy_backward(grad_loss, output, targets)

# Check for gradient explosion
check_gradient_norm([grad], max_norm=100.0)
```

### Configuration

Add to model config:
```mojo
struct SafetyConfig:
    var check_nan_inf: Bool
    var check_gradients: Bool
    var max_gradient_norm: Float64
    var verbose: Bool  # Print warnings instead of raising
```

## 3. Test Coverage Improvements

### Empty Test Files to Implement

#### A. `test_activations.mojo` (Priority: HIGH)

Test all 10 activation functions:

```mojo
fn test_relu_basic() raises:
    """Test ReLU with known values."""
    var x = ExTensor([[-2, -1, 0, 1, 2]])
    var y = relu(x)
    assert_almost_equal(y[0], 0.0)
    assert_almost_equal(y[4], 2.0)

fn test_relu_backward() raises:
    """Test ReLU gradient."""
    var x = ExTensor([[-1, 1]])
    var grad_out = ExTensor([[1, 1]])
    var grad_in = relu_backward(grad_out, x)
    assert_almost_equal(grad_in[0], 0.0)  # x < 0
    assert_almost_equal(grad_in[1], 1.0)  # x > 0

# Repeat for: leaky_relu, prelu, sigmoid, tanh, softmax, gelu, swish, mish, elu
```

Tests per function:
- Basic correctness (known values)
- Backward pass (gradient checking)
- Edge cases (0, very large, very small)
- Dtype support (float32, float64)

**Total:** ~30 tests

#### B. `test_initializers.mojo` (Priority: MEDIUM)

Test weight initialization functions:

```mojo
fn test_xavier_uniform_range() raises:
    """Test Xavier uniform initializer produces correct range."""
    var W = xavier_uniform(1000, 100, DType.float32)
    var limit = sqrt(6.0 / (1000 + 100))

    # All values should be in [-limit, limit]
    for i in range(W.numel()):
        var val = W._get_float64(i)
        assert_true(val >= -limit and val <= limit)

fn test_xavier_uniform_mean_std() raises:
    """Test Xavier uniform has approximately correct statistics."""
    var W = xavier_uniform(1000, 1000, DType.float32)
    var mean = compute_mean(W)
    var std = compute_std(W)

    assert_almost_equal(mean, 0.0, tol=0.01)
    # Uniform distribution: std = range / sqrt(12)
    var expected_std = sqrt(6.0 / 2000) / sqrt(3.0)
    assert_almost_equal(std, expected_std, tol=0.01)

# Repeat for: xavier_normal, kaiming_uniform, kaiming_normal, uniform, normal, constant
```

**Total:** ~20 tests

#### C. `test_tensors.mojo` (Priority: MEDIUM)

Test basic tensor operations:

```mojo
fn test_tensor_creation() raises:
    """Test tensor creation with different shapes."""
    var t1 = zeros([3, 4], DType.float32)
    assert_equal(t1.shape()[0], 3)
    assert_equal(t1.shape()[1], 4)
    assert_equal(t1.numel(), 12)

fn test_tensor_properties() raises:
    """Test shape, dtype, numel, dim properties."""
    var t = ones([2, 3, 4], DType.float64)
    assert_equal(t.dim(), 3)
    assert_equal(t.dtype(), DType.float64)

fn test_is_contiguous() raises:
    """Test contiguous memory layout detection."""
    var t = zeros([3, 4], DType.float32)
    assert_true(t.is_contiguous())

    # TODO: Test after implementing transpose/view operations
```

**Total:** ~15 tests

### Edge Case Tests to Add

Add to existing test files:

**`test_conv.mojo`:**
- Stride > 1
- Padding > 0
- Non-square kernels (3x5, 5x3)
- Single channel vs multi-channel

**`test_pooling.mojo`:**
- Stride != kernel_size
- Padding > 0
- Non-square windows

**`test_normalization.mojo`:**
- Training vs inference mode
- Running statistics correctness
- Batch size = 1 edge case

## Implementation Order

### Week 1: Dtype Dispatch Framework
- [ ] Day 1-2: Create `dtype_dispatch.mojo` helpers
- [ ] Day 3-4: Refactor `activation.mojo`
- [ ] Day 5: Refactor `elementwise.mojo`

### Week 2: Numerical Safety + Test Coverage
- [ ] Day 1-2: Create `numerical_safety.mojo`
- [ ] Day 3: Implement `test_activations.mojo`
- [ ] Day 4: Implement `test_initializers.mojo`
- [ ] Day 5: Implement `test_tensors.mojo`

### Week 3: Remaining Refactoring + Edge Cases
- [ ] Day 1-2: Refactor remaining modules
- [ ] Day 3-4: Add edge case tests
- [ ] Day 5: Integration testing and documentation

## Success Criteria

- [ ] Code reduction: 500+ lines → 100 lines in refactored modules
- [ ] Test coverage: 80+ new tests added
- [ ] All tests pass: `mojo test tests/shared/`
- [ ] Zero performance regression (numerical safety disabled by default)
- [ ] Documentation updated with new APIs

## Risks & Mitigations

**Risk:** Breaking existing code
**Mitigation:** Maintain backward compatibility, comprehensive testing

**Risk:** Performance regression
**Mitigation:** Use @parameter for compile-time specialization, benchmark hot paths

**Risk:** Complex generic syntax
**Mitigation:** Provide clear examples, good documentation

## Next Steps

1. Review this plan
2. Create feature branch: `feature/dtype-refactor-safety-tests`
3. Start with dtype dispatch helpers
4. Implement incrementally with tests
5. Create PR when complete
