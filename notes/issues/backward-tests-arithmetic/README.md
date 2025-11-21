# Arithmetic Backward Tests with Numerical Gradient Checking

## Objective

Add numerical gradient checking tests to validate analytical gradients against numerical gradients computed via finite differences for arithmetic operations (add, subtract, multiply, divide).

## Deliverables

- Extended `tests/shared/core/test_arithmetic_backward.mojo` with 11 new gradient checking tests
- Total test count: 23 tests (12 existing analytical + 11 new numerical gradient checking)
- Tests use `check_gradient()` from `tests.helpers.gradient_checking`
- Coverage: All 4 arithmetic operations with numerical validation

## Test Coverage Summary

### Original Tests (Analytical Validation) - 12 tests

1. **test_add_backward** - Element-wise addition, same shape
2. **test_add_scalar_backward** - Scalar addition broadcast [2,3] + [1]
3. **test_subtract_backward** - Element-wise subtraction, same shape
4. **test_subtract_scalar_backward** - Scalar subtraction broadcast
5. **test_multiply_backward** - Element-wise multiplication, same shape
6. **test_multiply_scalar_backward** - Scalar multiplication broadcast
7. **test_divide_backward** - Element-wise division, same shape
8. **test_divide_scalar_backward** - Scalar division broadcast
9. **test_add_broadcast** - Addition with [2,3] + [3] broadcasting
10. **test_subtract_broadcast** - Subtraction with broadcasting
11. **test_multiply_broadcast** - Multiplication with broadcasting
12. **test_divide_broadcast** - Division with broadcasting

### New Tests (Numerical Gradient Checking) - 11 tests

#### Tests 13-16: Addition, Subtraction, Multiplication, Division (A operand)

13. **test_add_backward_gradient** - Validates add backward for first operand with numerical gradient
14. **test_subtract_backward_gradient** - Validates subtract backward for first operand
15. **test_multiply_backward_gradient** - Validates multiply backward for first operand (product rule)
16. **test_divide_backward_gradient** - Validates divide backward for first operand (quotient rule)

#### Tests 17-20: Operations (B operand)

17. **test_add_backward_b_gradient** - Gradient w.r.t. second operand of addition
18. **test_subtract_backward_b_gradient** - Gradient w.r.t. second operand (negated)
19. **test_multiply_backward_b_gradient** - Gradient w.r.t. second operand (product rule)
20. **test_divide_backward_b_gradient** - Gradient w.r.t. denominator (quotient rule)

#### Tests 21-23: Broadcasting with Numerical Validation

21. **test_add_backward_broadcast_gradient** - Addition with broadcast [3] -> [2,3]
22. **test_multiply_backward_broadcast_gradient** - Multiplication with broadcast [3] -> [2,3]
23. **test_divide_backward_broadcast_gradient** - Division with broadcast [3] -> [2,3]

## Implementation Details

### Numerical Gradient Checking Pattern

Each new test follows this structure:

```mojo
fn test_op_backward_gradient() raises:
    """Test op_backward with numerical gradient checking."""
    # 1. Create test tensors with non-uniform values
    var shape = create_shape_vec(3, 4)
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)

    # Initialize with diverse values
    for i in range(12):
        a._data.bitcast[Float32]()[i] = Float32(i) * 0.1 - 1.2
        b._data.bitcast[Float32]()[i] = Float32(i) * 0.15 - 0.8

    # 2. Define forward function
    fn forward(inp: ExTensor) raises -> ExTensor:
        return op(inp, b)

    # 3. Define backward function
    fn backward(grad_out: ExTensor, inp: ExTensor) raises -> ExTensor:
        var grads = op_backward(grad_out, inp, b)
        return grads.grad_a

    # 4. Run numerical gradient check
    var output = forward(a)
    var grad_output = ones_like(output)
    check_gradient(forward, backward, a, grad_output, rtol=1e-3, atol=1e-6)
```

### Gradient Checking Parameters

- **epsilon**: 1e-5 (central difference perturbation)
- **rtol** (relative tolerance): 1e-3 (suitable for Float32)
- **atol** (absolute tolerance): 1e-6
- **Formula**: Analytical gradient must match `|analytical - numerical| <= atol + rtol * |numerical|`

### Test Data Initialization

All tests use **non-uniform values** to ensure comprehensive validation:

- Addition: values from `-1.2 + 0.1*i` to `0.15*i - 0.8`
- Subtraction: values from `0.5 + 0.1*i` to `-1.5 + 0.2*i`
- Multiplication: values from `0.1 + 0.1*i` to `0.2 + 0.15*i` (no zeros)
- Division: values from `0.5 + 0.2*i` to `1.0 + 0.1*i` (denominators > 0)

This diverse test data exercises the gradient computation across a wider range of inputs.

## File Changes

### File: `/home/mvillmow/ml-odyssey/tests/shared/core/test_arithmetic_backward.mojo`

**Changes Made**:
1. Added import: `from tests.helpers.gradient_checking import check_gradient, compute_numerical_gradient`
2. Added 11 new test functions (tests 13-23)
3. Updated module docstring to mention numerical gradient checking

**Line Count**: 865 lines (up from 497)

## Operations Covered

### Addition (add)

- Gradient rule: `∂(A+B)/∂A = 1`, `∂(A+B)/∂B = 1`
- Tests validate that gradient passes through unchanged
- Broadcasting test ensures gradient reduction works correctly

### Subtraction (subtract)

- Gradient rule: `∂(A-B)/∂A = 1`, `∂(A-B)/∂B = -1`
- Tests validate negation of gradient for second operand
- Broadcasting reduction verified

### Multiplication (multiply)

- Gradient rule: `∂(A*B)/∂A = B`, `∂(A*B)/∂B = A`
- Product rule validated numerically
- Non-zero values ensure proper multiplication behavior
- Broadcasting dimension reduction verified

### Division (divide)

- Gradient rule: `∂(A/B)/∂A = 1/B`, `∂(A/B)/∂B = -A/B²`
- Quotient rule with numerical validation
- Denominator always > 0 to avoid division by zero
- Numerical stability with epsilon handling verified

## Integration with CI/CD

The test file is located at the standard path and uses the standard import patterns:

- Imports from `tests.shared.conftest` for assertion helpers
- Imports from `tests.helpers.gradient_checking` for numerical validation
- Follows existing test patterns in the codebase

## Quality Assurance

### Gradient Checking Validation

Each test uses the gold-standard numerical gradient checking:

1. **Finite Differences**: Central difference formula with O(ε²) accuracy
2. **Analytical Gradients**: Computed via backward passes
3. **Comparison**: Element-wise comparison within tolerance

### Error Messages

If a test fails, the `check_gradient()` function provides clear error messages indicating:
- Which element failed
- The mismatch magnitude
- Expected tolerance

### Edge Cases Covered

- **Zero values**: Subtraction and addition handle zero without issues
- **Small denominators**: Division test data avoids values near zero
- **Broadcasting**: Multiple dimensional reduction patterns tested
- **Non-uniform data**: Tests don't rely on symmetric/simple values

## Running the Tests

The test file can be run as part of the standard test suite:

```bash
# If Mojo testing is configured
mojo test tests/shared/core/test_arithmetic_backward.mojo

# Or as part of full test suite
python3 -m pytest tests/
```

## References

- **Numerical Gradient Helper**: `/home/mvillmow/ml-odyssey/tests/helpers/gradient_checking.mojo`
- **Backward Operations**: `/home/mvillmow/ml-odyssey/shared/core/arithmetic.mojo` (lines 549-716)
- **GradientPair Type**: `shared.core.gradient_types.GradientPair`
- **ExTensor Type**: `shared.core.extensor.ExTensor`

## Implementation Notes

### Why Numerical Gradient Checking?

Numerical gradient checking is the gold standard for validating backward passes because:

1. **Independent Verification**: Uses only the forward function, not backward implementation
2. **Finite Differences**: Math-based approach not dependent on implementation details
3. **Comprehensive**: Catches subtle bugs in gradient computation
4. **Standards-Based**: Central difference method has well-understood error properties

### Tolerances

The chosen tolerances (rtol=1e-3, atol=1e-6) are standard for Float32:

- **rtol=1e-3**: Accounts for floating-point rounding in product/division
- **atol=1e-6**: Catches absolute errors in edge cases
- **Combination**: Handles both small (near zero) and large gradients

### Test Organization

Tests are organized in a logical sequence:

1. Original tests (1-12): Validate analytical correctness
2. New tests (13-20): Numerical validation of single operations
3. New tests (21-23): Numerical validation with broadcasting

This organization makes it easy to:
- Run subset of tests for debugging
- Understand which operations are covered
- Identify patterns in test structure

## Success Criteria Met

- ✓ All 4 arithmetic operations covered (add, subtract, multiply, divide)
- ✓ Both operands (A and B) validated for each operation
- ✓ Broadcasting cases tested with numerical gradient checking
- ✓ Non-uniform test data uses diverse value ranges
- ✓ Proper tolerances for Float32 (rtol=1e-3, atol=1e-6)
- ✓ Imports organized correctly
- ✓ Tests follow existing patterns in the codebase
- ✓ Comprehensive docstrings for each test
- ✓ Edge cases handled (no zero denominators, non-zero factors for multiplication)

## Summary

11 new numerical gradient checking tests added to arithmetic backward operations suite. These tests validate that analytical gradients computed via backward passes match the gold-standard numerical gradients computed via finite differences. All tests use real implementations with simple, non-uniform test data, following the Test Engineer guidelines for realistic gradient validation.
