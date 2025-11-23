# Pooling Backward Gradient Checking Tests

## Objective

Add numerical gradient checking tests for pooling backward passes (maxpool2d and avgpool2d) to validate gradient computation accuracy using finite difference approximations.

## Deliverables

- `tests/shared/core/test_backward.mojo` - Added two new test functions:
  - `test_maxpool2d_backward_gradient()` - Lines 472-498
  - `test_avgpool2d_backward_gradient()` - Lines 501-527
- Updated `main()` function to invoke new tests (lines 647-648, 657-658)

## Tests Added

### test_maxpool2d_backward_gradient()

- Creates 1x2x4x4 input tensor with non-uniform values (range: -1.6 to 2.4)
- Validates maxpool2d backward pass using numerical gradient checking
- Uses `check_gradient()` with rtol=1e-3, atol=1e-6 (Float32 tolerances)
- Follows established pattern from `test_linear_backward_gradient()` and `test_conv2d_backward_gradient()`

### test_avgpool2d_backward_gradient()

- Creates 1x2x4x4 input tensor with non-uniform values (range: -1.6 to 2.4)
- Validates avgpool2d backward pass using numerical gradient checking
- Uses `check_gradient()` with rtol=1e-3, atol=1e-6 (Float32 tolerances)
- Follows established pattern from other gradient checking tests

## Implementation Details

### Gradient Checking Pattern

```mojo
fn test_operation_backward_gradient() raises:
    # Create input with non-uniform values
    var input_shape = DynamicVector[Int](4)
    input_shape[0] = 1
    input_shape[1] = 2
    input_shape[2] = 4
    input_shape[3] = 4
    var x = zeros(input_shape, DType.float32)

    # Initialize with non-uniform values
    for i in range(1 * 2 * 4 * 4):
        x._data.bitcast[Float32]()[i] = Float32(i) * 0.1 - 1.6

    # Forward function wrapper
    fn forward(inp: ExTensor) raises -> ExTensor:
        return operation(inp, kernel_size=2, stride=2, padding=0)

    # Backward function wrapper (only return grad_input)
    fn backward(grad_out: ExTensor, inp: ExTensor) raises -> ExTensor:
        return operation_backward(grad_out, inp, kernel_size=2, stride=2, padding=0)

    var output = forward(x)
    var grad_output = ones_like(output)

    # Numerical gradient checking
    check_gradient(forward, backward, x, grad_output, rtol=1e-3, atol=1e-6)
```

## Test Data

- **Input shape**: [1, 2, 4, 4] (batch=1, channels=2, height=4, width=4)
- **Values**: Non-uniform, computed as `Float32(i) * 0.1 - 1.6`
- **Range**: -1.6 to 2.4
- **Kernel**: 2x2, stride=2, padding=0
- **Output shape**: [1, 2, 2, 2] (after max/avg pooling)

## Tolerances

- **rtol**: 1e-3 (relative tolerance for Float32)
- **atol**: 1e-6 (absolute tolerance for Float32)
- These match tolerances used in other backward pass gradient checking tests

## Numerical Gradient Checking Method

The `check_gradient()` function validates backpropagation using finite differences:

```
numerical_grad[i] ≈ (f(x + ε) - f(x - ε)) / (2ε)
```

This is the gold standard for validating gradient computation and catches:

- Incorrect derivatives
- Off-by-one errors in indexing
- Shape mismatches in gradient computation
- Gradient accumulation bugs

## Integration

Both tests are registered in `main()` function:

- Call site 1: After `test_maxpool2d_backward_gradient_routing()` (line 647)
- Call site 2: After `test_avgpool2d_backward_gradient_distribution()` (line 657)
- Print statements confirm test execution

## References

- Pattern established by: `test_linear_backward_gradient()` (lines 144-193)
- Pattern established by: `test_conv2d_backward_gradient()` (lines 298-347)
- Pattern established by: `test_cross_entropy_backward_gradient()` (lines 508-547)

## Commit

- Commit hash: `13b7c84c3b1d10c6240c87363790f679f3e6b29a`
- Message: `test(backward): Add numerical gradient checking for pooling operations`
- File modified: `tests/shared/core/test_backward.mojo` (+64 lines)

## Success Criteria

- [x] Tests implement numerical gradient checking via `check_gradient()`
- [x] Tests use non-uniform input values (not all ones/zeros)
- [x] Tests use appropriate Float32 tolerances (rtol=1e-3, atol=1e-6)
- [x] Tests integrated into `main()` function
- [x] Follows established test patterns from codebase
- [x] Commit uses conventional commit format
- [x] Code passes pre-commit hooks (formatting, whitespace, etc.)
