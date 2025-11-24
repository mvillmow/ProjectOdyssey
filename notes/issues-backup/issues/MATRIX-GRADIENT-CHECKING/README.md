# Matrix Operations Gradient Checking Tests

## Objective

Add numerical gradient checking tests to `tests/shared/core/test_matrix.mojo` for matrix operations backward passes (matmul and transpose) to validate correctness using finite differences.

## Deliverables

- `tests/shared/core/test_matrix.mojo` - Updated with 3 new numerical gradient tests:
  - `test_matmul_backward_gradient_a()` - Validates matmul gradient w.r.t. input A
  - `test_matmul_backward_gradient_b()` - Validates matmul gradient w.r.t. input B
  - `test_transpose_backward_gradient()` - Validates transpose backward gradient

## Test Summary

### Tests Added

1. **test_matmul_backward_gradient_a()**
   - Tests gradient computation w.r.t. matrix A in matmul(A, B)
   - Setup: A (6x4), B (4x2) with non-uniform values
   - Validates: analytical gradient matches finite differences to rtol=1e-3, atol=1e-6
   - Pattern: Forward wraps `matmul(inp, B)`, backward extracts grad_a from matmul_backward

2. **test_matmul_backward_gradient_b()**
   - Tests gradient computation w.r.t. matrix B in matmul(A, B)
   - Setup: A (3x4), B (4x2) with non-uniform values
   - Validates: analytical gradient matches finite differences to rtol=1e-3, atol=1e-6
   - Pattern: Forward wraps `matmul(A, inp)`, backward extracts grad_b from matmul_backward

3. **test_transpose_backward_gradient()**
   - Tests gradient computation through transpose operation
   - Setup: x (3x4) with non-uniform values
   - Validates: analytical gradient matches finite differences to rtol=1e-3, atol=1e-6
   - Pattern: Since transpose is its own inverse, gradient is simply transposed

## Imports Added

```mojo
from shared.core.extensor import ExTensor, zeros, ones, zeros_like, ones_like
from tests.helpers.gradient_checking import check_gradient, compute_numerical_gradient, assert_gradients_close
```

## Key Implementation Details

### Pattern Following

All tests follow the pattern established in `tests/shared/core/test_backward.mojo`:

1. **Non-uniform initialization**: Input values initialized with `Float32(i) * scale - offset` to avoid all-ones/all-zeros
2. **Forward/backward wrappers**: Wrapped functions match the check_gradient API
3. **Gradient output**: ones_like(output) for upstream gradient
4. **Tolerances**: rtol=1e-3, atol=1e-6 (Float32 appropriate)

### Test Data

- **matmul_backward_gradient_a**: A[6,4], B[4,2] - Values in range [-1.4, 1.4]
- **matmul_backward_gradient_b**: A[3,4], B[4,2] - Values in range [-0.7, 0.9]
- **transpose_backward_gradient**: x[3,4] - Values in range [-2.0, 0.8]

## Files Modified

- `/home/mvillmow/ml-odyssey/tests/shared/core/test_matrix.mojo`
  - Added imports for `zeros_like`, `ones_like`, gradient checking helpers
  - Added 3 numerical gradient test functions
  - Updated `main()` to call new tests

## Integration Notes

### Gradient Checking Helpers

Uses existing infrastructure from `tests/helpers/gradient_checking.mojo`:

- `check_gradient()` - Main validation function (args: forward_fn, backward_fn, x, grad_output, rtol, atol)
- `compute_numerical_gradient()` - Finite differences implementation
- `assert_gradients_close()` - Element-wise tolerance comparison

### Tolerances

Float32 tolerances (per gradient_checking.mojo documentation):

- rtol (relative tolerance): 1e-3
- atol (absolute tolerance): 1e-6

These are the standard for Float32 numerical gradient checking with central differences.

## Success Criteria

- All 3 new tests use numerical gradient checking via `check_gradient()`
- Tests use non-uniform initialization (no all-ones/all-zeros)
- Tests pass finite differences validation with Float32 tolerances
- Existing tests remain unmodified and passing
- main() function includes calls to all new test functions

## Notes

### Compilation Status

Current codebase has compilation issues unrelated to these test additions:

- Core modules use deprecated `let` keyword (should be `var`)
- Core modules have infrastructure errors (tuple returns, type annotations)
- Gradient checking helpers have non-movable ExTensor issues in nested functions

These pre-existing issues block test execution but do not affect test design.

### Test Design Rationale

- **Two tests for matmul**: matmul_backward returns gradients w.r.t. both A and B; both paths need validation
- **Transpose gradient**: Validates that gradient flow through simple operations works correctly
- **Non-uniform data**: Catches issues that would be hidden by uniform initializations
- **Finite differences**: Gold standard for gradient validation (O(ε²) error with central differences)

## References

- Gradient checking pattern: `tests/shared/core/test_backward.mojo` (linear, conv2d, cross-entropy tests)
- Gradient checking infrastructure: `tests/helpers/gradient_checking.mojo`
- Numerical gradient formula: Central differences (f(x+ε) - f(x-ε)) / 2ε
