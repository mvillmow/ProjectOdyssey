# ML Odyssey - Implementation Plan for Missing Features

**Issue**: #1538
**Date**: 2025-11-20
**Status**: Test coverage analysis complete, implementation plan in progress

## Executive Summary

This document outlines missing features and test coverage gaps in the ML Odyssey pure functional
architecture. All implementations must follow the pure functional design principles:

- No classes or internal state
- Caller manages all state
- Functions return new values, never mutate inputs

## Current Test Coverage Analysis

### ✅ Fully Tested Modules

### Implemented with comprehensive tests

1. **Convolution** (`shared/core/conv.mojo`)
   - ✅ 8 tests in `test_conv.mojo` (397 lines)
   - Tests: initialization, shapes, padding, stride, numerical correctness, multi-channel, batching

1. **Pooling** (`shared/core/pooling.mojo`)
   - ✅ 14 tests in `test_pooling.mojo` (467 lines)
   - Tests: maxpool2d, avgpool2d, global_avgpool2d with all parameter combinations

1. **Linear Layers** (`shared/core/linear.mojo`)
   - ✅ Tests in `test_layers.mojo` (709 lines total)
   - Tests: initialization, forward pass, no-bias variant, PyTorch validation

1. **Activations** (`shared/core/activation.mojo`)
   - ✅ Tests in `test_layers.mojo`
   - Tests: relu, sigmoid, tanh range validation, PyTorch validation
   - ✅ All activation functions fully implemented with backward passes

1. **Optimizers** (`shared/training/optimizers/`)
   - ✅ SGD: 6 tests (basic, momentum, weight decay, nesterov, PyTorch validation)
   - ✅ Adam: 4 tests (update, bias correction, PyTorch validation)

### ⚠️ Implemented But Not Tested

### Modules exist but have empty/stub test files

1. **Tensors** (`shared/core/extensor.mojo`)
   - ❌ `test_tensors.mojo` is empty (0 lines)
   - Core tensor operations need comprehensive testing
   - Need tests for: creation, indexing, slicing, reshaping, shape operations

1. **Initializers** (`shared/core/initializers.mojo`)
   - ❌ `test_initializers.mojo` is empty (0 lines)
   - Need tests for: xavier_uniform, xavier_normal, kaiming_uniform, kaiming_normal, uniform, normal, constant

1. **Arithmetic** (`shared/core/arithmetic.mojo`)
   - ❌ No test file exists
   - ✅ Has backward functions implemented
   - Need tests for: add, subtract, multiply, divide, power, and their backward passes

1. **Matrix Operations** (`shared/core/matrix.mojo`)
   - ❌ No test file exists
   - ✅ Has backward functions implemented
   - Need tests for: matmul, transpose, dot, outer, and their backward passes

1. **Element-wise Operations** (`shared/core/elementwise.mojo`)
   - ❌ No test file exists
   - ✅ Has backward functions implemented
   - Need tests for: exp, log, sqrt, abs, sin, cos, clip, logical ops, and backward passes

1. **Comparison Operations** (`shared/core/comparison.mojo`)
   - ❌ No test file exists
   - Need tests for: equal, not_equal, less, less_equal, greater, greater_equal

1. **Broadcasting** (`shared/core/broadcasting.mojo`)
   - ❌ No test file exists
   - Need tests for: broadcast_shapes, are_shapes_broadcastable, compute_broadcast_strides

1. **Reduction Operations** (`shared/core/reduction.mojo`)
   - ❌ No test file exists
   - ✅ Has backward functions implemented
   - Need tests for: sum, mean, max, min, variance, std, and backward passes

1. **Loss Functions** (`shared/core/loss.mojo`)
   - ❌ No test file exists
   - ✅ Has backward functions for binary_cross_entropy and mean_squared_error
   - ⚠️ cross_entropy and cross_entropy_backward NOT implemented (placeholder only)
   - Need tests for all loss functions and backward passes

### ❌ Missing Critical Features

### Features referenced in tests but not implemented

1. **Backward Passes for Neural Network Operations**
   - ❌ `linear_backward` - Not implemented
   - ❌ `conv2d_backward` - Not implemented
   - ❌ `maxpool2d_backward` - Not implemented
   - ❌ `avgpool2d_backward` - Not implemented
   - These are CRITICAL for training neural networks

1. **Missing Optimizers**
   - ❌ RMSprop - Referenced in `test_optimizers.mojo` but not implemented
   - ❌ AdamW - Referenced in tests but may just be Adam with weight decay

1. **Cross-Entropy Loss**
   - ❌ `cross_entropy` - Placeholder only, raises error
   - ❌ `cross_entropy_backward` - Placeholder only, raises error
   - This is CRITICAL for classification tasks

1. **Normalization Layers**
   - ❌ Batch Normalization - Not implemented
   - ❌ Layer Normalization - Not implemented
   - ❌ Group Normalization - Not implemented

1. **Dropout**
   - ❌ Dropout - Not implemented
   - ❌ Dropout2d - Not implemented

1. **Advanced Activations**
   - ✅ ReLU, LeakyReLU, PReLU, Sigmoid, Tanh, Softmax, GELU - All implemented
   - ❌ Swish/SiLU - Not implemented
   - ❌ Mish - Not implemented
   - ❌ ELU - Not implemented

## Implementation Plan

### Phase 1: Critical Backward Passes (Priority: CRITICAL)

**Why**: Cannot train neural networks without backward passes

### Tasks

1. Implement `linear_backward(grad_output, input, weights) -> (grad_input, grad_weights, grad_bias)`
1. Implement `conv2d_backward(grad_output, input, kernel, stride, padding) -> (grad_input, grad_kernel, grad_bias)`
1. Implement `maxpool2d_backward(grad_output, input, kernel_size, stride, padding) -> grad_input`
1. Implement `avgpool2d_backward(grad_output, input, kernel_size, stride, padding) -> grad_input`

### Testing Requirements

- Numerical gradient checking using finite differences
- Compare against PyTorch backward passes
- Test with various tensor shapes and parameters

**Estimated Complexity**: High (requires careful gradient computation)

### Phase 2: Cross-Entropy Loss (Priority: CRITICAL)

**Why**: Essential for classification tasks

### Tasks

1. Implement `cross_entropy(predictions, targets, reduction='mean') -> loss`
   - Support both class indices and one-hot encoded targets
   - Support reduction modes: 'mean', 'sum', 'none'
   - Numerically stable implementation (log-sum-exp trick)

1. Implement `cross_entropy_backward(predictions, targets, reduction='mean') -> grad_predictions`
   - Match PyTorch's cross_entropy gradient behavior

### Testing Requirements

- Test with class indices
- Test with one-hot encoded targets
- Test reduction modes
- PyTorch validation
- Numerical stability tests (large logits)

**Estimated Complexity**: Medium (requires numerical stability considerations)

### Phase 3: Core Module Tests (Priority: HIGH)

**Why**: Existing implementations need verification

### Tasks

1. **Tensor Tests** (`test_tensors.mojo`)
   - Creation functions: zeros, ones, full, empty, arange, eye, linspace
   - Shape operations: reshape, transpose, permute
   - Indexing and slicing
   - dtype handling

1. **Arithmetic Tests** (`test_arithmetic.mojo`)
   - Forward: add, subtract, multiply, divide, floor_divide, modulo, power
   - Backward: all backward passes
   - Broadcasting behavior
   - Edge cases (division by zero, overflow)

1. **Matrix Tests** (`test_matrix.mojo`)
   - matmul with various shapes
   - transpose and transpose_backward
   - dot and outer products
   - matmul_backward gradient checking

1. **Element-wise Tests** (`test_elementwise.mojo`)
   - All element-wise functions (exp, log, sqrt, abs, sin, cos, etc.)
   - All backward passes
   - Numerical stability (log of small numbers, sqrt of zero)
   - Edge cases

1. **Reduction Tests** (`test_reduction.mojo`)
   - sum, mean, max, min, variance, std
   - Axis reduction
   - Backward passes
   - keepdims parameter

1. **Loss Tests** (`test_loss.mojo`)
   - binary_cross_entropy
   - mean_squared_error
   - cross_entropy (after Phase 2 implementation)
   - All backward passes
   - Reduction modes

1. **Initializer Tests** (`test_initializers.mojo`)
   - xavier_uniform, xavier_normal
   - kaiming_uniform, kaiming_normal
   - uniform, normal, constant
   - Statistical properties (mean, std)

1. **Comparison Tests** (`test_comparison.mojo`)
   - equal, not_equal
   - less, less_equal, greater, greater_equal
   - Broadcasting behavior

1. **Broadcasting Tests** (`test_broadcasting.mojo`)
   - broadcast_shapes
   - are_shapes_broadcastable
   - compute_broadcast_strides
   - Edge cases

### Testing Requirements

- Each test file should have 10-20 focused tests
- Cover happy path, edge cases, and error conditions
- Include PyTorch validation where applicable
- Use property-based testing for numerical operations

**Estimated Complexity**: Medium (mostly straightforward testing)

### Phase 4: Missing Optimizers (Priority: MEDIUM)

**Why**: RMSprop is referenced in tests but not implemented

### Tasks

1. Implement RMSprop optimizer
   - `rmsprop_step(params, gradients, square_avg, t, lr, alpha, eps, weight_decay, momentum) -> (new_params, new_square_avg)`
   - Pure functional design
   - Support momentum
   - Support weight decay

### Testing Requirements

- Initialization tests
- Parameter update tests
- Momentum tests
- PyTorch validation
- Numerical stability

**Estimated Complexity**: Low (similar to Adam)

### Phase 5: Normalization Layers (Priority: MEDIUM)

**Why**: Important for training deep networks

### Tasks

1. **Batch Normalization**
   - `batch_norm2d(x, gamma, beta, running_mean, running_var, training, momentum, eps) -> (output, new_running_mean, new_running_var)`
   - `batch_norm2d_backward(grad_output, x, gamma, mean, var, eps) -> (grad_input, grad_gamma, grad_beta)`

1. **Layer Normalization**
   - `layer_norm(x, gamma, beta, eps) -> output`
   - `layer_norm_backward(grad_output, x, gamma, mean, var, eps) -> (grad_input, grad_gamma, grad_beta)`

### Testing Requirements

- Forward pass correctness
- Backward pass gradient checking
- Training vs inference mode (for batch norm)
- PyTorch validation
- Numerical stability

**Estimated Complexity**: High (complex gradient computation, running statistics)

### Phase 6: Regularization (Priority: MEDIUM)

**Why**: Essential for preventing overfitting

### Tasks

1. **Dropout**
   - `dropout(x, p, training, seed) -> output` (with mask for backward)
   - `dropout_backward(grad_output, mask, p) -> grad_input`

1. **Dropout2d** (spatial dropout for CNNs)
   - `dropout2d(x, p, training, seed) -> output`
   - `dropout2d_backward(grad_output, mask, p) -> grad_input`

### Design Considerations

- Pure functional design requires returning mask
- Mask must be passed to backward function
- Training vs inference mode

### Testing Requirements

- Dropout rate verification (approximately p% dropped)
- Training vs inference behavior
- Gradient flow through backward
- Reproducibility with seed

**Estimated Complexity**: Low (simple operation, tricky API design)

### Phase 7: Advanced Activations (Priority: LOW)

**Why**: Nice to have for modern architectures

### Tasks

1. Swish/SiLU: `swish(x) = x * sigmoid(x)`
1. Mish: `mish(x) = x * tanh(softplus(x))`
1. ELU: `elu(x, alpha) = x if x > 0 else alpha * (exp(x) - 1)`
1. All backward passes

### Testing Requirements

- Forward pass correctness
- Backward pass gradient checking
- PyTorch validation

**Estimated Complexity**: Low (simple mathematical operations)

## Testing Strategy

### 1. Unit Tests

- Each function gets dedicated tests
- Test happy path, edge cases, error conditions
- Use assert_almost_equal for numerical comparisons

### 2. Property-Based Tests

- Batch independence: f(concat([x1, x2])) == concat([f(x1), f(x2)])
- Deterministic: f(x) always produces same output
- Shape preservation where expected
- Gradient magnitudes within reasonable bounds

### 3. PyTorch Validation

- Compare outputs against PyTorch for same inputs
- Validate forward and backward passes
- Ensure numerical accuracy within tolerance (1e-5 for float32)

### 4. Gradient Checking

- Use finite differences to verify backward passes
- Formula: (f(x + eps) - f(x - eps)) / (2 * eps)
- Should match analytical gradient within tolerance

### 5. Numerical Stability Tests

- Test with extreme values (very large, very small, zero)
- Test edge cases (log(0), sqrt(-1), division by zero)
- Ensure graceful error handling or stable computation

## Priority Matrix

| Phase | Priority | Effort | Impact | Order |
|-------|----------|--------|--------|-------|
| Phase 1: Backward Passes | CRITICAL | High | Critical | 1st |
| Phase 2: Cross-Entropy | CRITICAL | Medium | Critical | 2nd |
| Phase 3: Core Tests | HIGH | Medium | High | 3rd |
| Phase 4: RMSprop | MEDIUM | Low | Medium | 4th |
| Phase 5: Normalization | MEDIUM | High | High | 5th |
| Phase 6: Dropout | MEDIUM | Low | Medium | 6th |
| Phase 7: Advanced Activations | LOW | Low | Low | 7th |

## Implementation Workflow

For each phase:

1. **Design** - Write function signatures and docstrings
1. **Test First** - Write comprehensive tests (TDD)
1. **Implement** - Write the implementation
1. **Validate** - Run tests, PyTorch validation, gradient checking
1. **Document** - Update documentation and examples
1. **Commit** - Commit with clear message following conventional commits

## Next Steps

### Immediate Actions

1. ✅ Review and approve this plan
1. **Start Phase 1**: Implement backward passes for linear, conv2d, and pooling
1. Create test files for backward pass validation
1. Run gradient checking against finite differences
1. Validate against PyTorch backward passes

### Success Criteria

- All backward passes produce correct gradients (gradient checking passes)
- All backward passes match PyTorch within 1e-5 tolerance
- Can train a simple CNN end-to-end (LeNet-5 style)
- All tests pass without errors

## Notes

- All implementations must be pure functional (no classes, no internal state)
- Caller manages all state (weights, running statistics, optimizer state)
- Functions return new values, never mutate inputs
- Use ExTensor exclusively (no Tensor alias)
- Follow existing code style and conventions
- Document all functions with clear docstrings
- Include usage examples in docstrings
