# Arithmetic Operation Gradients - Quick Reference

All arithmetic operations support **broadcasting**. Gradients must be **summed over broadcast dimensions** to match
input shapes.

## Addition

**Forward**:

```text
z = x + y
```

**Gradients**:

```text
∂z/∂x = 1
∂z/∂y = 1
```

**Code**:

```mojo
fn add_backward(
    grad_output: ExTensor, a: ExTensor, b: ExTensor
) raises -> (ExTensor, ExTensor):
    var grad_a = grad_output
    var grad_b = grad_output

    // Sum over broadcast dimensions
    grad_a = sum_to_shape(grad_a, a.shape())
    grad_b = sum_to_shape(grad_b, b.shape())

    return (grad_a, grad_b)
```

**Key Property**: Gradient is passthrough (scaled by 1)

**Broadcasting Example**:

```text
Input shapes: a(3,4), b(4)
Forward: a + b → (3,4)  [b broadcast to (3,4)]
Backward: grad_a = grad_output → (3,4)
          grad_b = sum(grad_output, axis=0) → (4)
```

---

## Subtraction

**Forward**:

```text
z = x - y
```

**Gradients**:

```text
∂z/∂x = +1
∂z/∂y = -1  ← Note the sign flip!
```

**Code**:

```mojo
fn subtract_backward(
    grad_output: ExTensor, a: ExTensor, b: ExTensor
) raises -> (ExTensor, ExTensor):
    var grad_a = grad_output
    var grad_b = negate(grad_output)  // Flip sign

    // Sum over broadcast dimensions
    grad_a = sum_to_shape(grad_a, a.shape())
    grad_b = sum_to_shape(grad_b, b.shape())

    return (grad_a, grad_b)
```

**Key Property**: Second operand gets negative gradient

---

## Multiplication (Element-wise)

**Forward**:

```text
z = x * y  (element-wise)
```

**Gradients**:

```text
∂z/∂x = y
∂z/∂y = x
```

**Code**:

```mojo
fn multiply_backward(
    grad_output: ExTensor, a: ExTensor, b: ExTensor
) raises -> (ExTensor, ExTensor):
    var grad_a = multiply(grad_output, b)  // ∂z/∂a = b
    var grad_b = multiply(grad_output, a)  // ∂z/∂b = a

    // Sum over broadcast dimensions
    grad_a = sum_to_shape(grad_a, a.shape())
    grad_b = sum_to_shape(grad_b, b.shape())

    return (grad_a, grad_b)
```

**Key Property**: Gradient w.r.t. one input is the other input (times upstream gradient)

**Example**:

```text
a = [2, 3], b = [4, 5], grad_output = [1, 1]
grad_a = grad_output * b = [1*4, 1*5] = [4, 5]
grad_b = grad_output * a = [1*2, 1*3] = [2, 3]
```

---

## Division

**Forward**:

```text
z = x / y
```

**Gradients**:

```text
∂z/∂x = 1/y
∂z/∂y = -x/y²
```

**Code**:

```mojo
fn divide_backward(
    grad_output: ExTensor, numerator: ExTensor, denominator: ExTensor
) raises -> (ExTensor, ExTensor):
    // ∂z/∂x = 1/y (with epsilon for stability)
    var safe_denom = clip(denominator, 1e-8, Float64.max)
    var grad_numerator = divide(grad_output, safe_denom)

    // ∂z/∂y = -x/y²
    var denom_squared = multiply(safe_denom, safe_denom)
    var neg_num_over_denom2 = negate(divide(numerator, denom_squared))
    var grad_denominator = multiply(grad_output, neg_num_over_denom2)

    // Sum over broadcast dimensions
    grad_numerator = sum_to_shape(grad_numerator, numerator.shape())
    grad_denominator = sum_to_shape(grad_denominator, denominator.shape())

    return (grad_numerator, grad_denominator)
```

**Key Property**: Gradient w.r.t. denominator has quadratic dependence (y²)

**Numerical Stability**: Clamp denominator to prevent division by zero

---

## Floor Division

**Forward**:

```text
z = floor(x / y)
```

**Gradients**:

```text
∂z/∂x = 0  (not differentiable - floor is piecewise constant)
∂z/∂y = 0
```

**Code**:

```mojo
fn floor_divide_backward(
    grad_output: ExTensor, a: ExTensor, b: ExTensor
) raises -> (ExTensor, ExTensor):
    // Floor operation breaks differentiability
    var grad_a = zeros_like(a)
    var grad_b = zeros_like(b)
    return (grad_a, grad_b)
```

**Key Property**: No gradient (discontinuous function)

**Use Case**: Rarely used in backpropagation (indices, discrete operations)

---

## Modulo

**Forward**:

```text
z = x mod y
```

**Gradients**:

```text
∂z/∂x = 1  (locally)
∂z/∂y = complex (involves floor(x/y))
```

**Code**:

```mojo
fn modulo_backward(
    grad_output: ExTensor, a: ExTensor, b: ExTensor
) raises -> (ExTensor, ExTensor):
    // Simplified: treat as passthrough for a, zero for b
    var grad_a = grad_output
    var grad_b = zeros_like(b)

    grad_a = sum_to_shape(grad_a, a.shape())
    grad_b = sum_to_shape(grad_b, b.shape())

    return (grad_a, grad_b)
```

**Key Property**: Rarely differentiable in practice (discontinuities at multiples of y)

**Use Case**: Avoid in differentiable code paths

---

## Power

**Forward**:

```text
z = x^n  (scalar exponent n)
```

**Gradient w.r.t. base**:

```text
∂z/∂x = n * x^(n-1)
```

**Gradient w.r.t. exponent** (if variable):

```text
∂z/∂n = x^n * log(x)
```

**Code** (fixed exponent):

```mojo
fn power_backward(
    grad_output: ExTensor, base: ExTensor, exponent: Float64
) raises -> ExTensor:
    // ∂(x^n)/∂x = n * x^(n-1)
    var x_power_n_minus_1 = power(base, exponent - 1.0)
    var local_grad = multiply_scalar(x_power_n_minus_1, exponent)
    return multiply(grad_output, local_grad)
```

**Code** (variable exponent):

```mojo
fn power_backward_full(
    grad_output: ExTensor, base: ExTensor, exponent: ExTensor
) raises -> (ExTensor, ExTensor):
    // ∂(x^n)/∂x = n * x^(n-1)
    var x_power_n_minus_1 = power(base, subtract_scalar(exponent, 1.0))
    var grad_base = multiply(multiply(grad_output, exponent), x_power_n_minus_1)

    // ∂(x^n)/∂n = x^n * log(x)
    var x_power_n = power(base, exponent)
    var log_base = log(clip(base, 1e-8, Float64.max))  // Stability
    var grad_exponent = multiply(multiply(grad_output, x_power_n), log_base)

    return (grad_base, grad_exponent)
```

**Special Cases**:

- n=2 (square): `∂z/∂x = 2x`
- n=0.5 (sqrt): `∂z/∂x = 1/(2√x)` (unstable at x=0)
- n=-1 (reciprocal): `∂z/∂x = -1/x²`

---

## Negation

**Forward**:

```text
z = -x
```

**Gradient**:

```text
∂z/∂x = -1
```

**Code**:

```mojo
fn negate_backward(grad_output: ExTensor, input: ExTensor) raises -> ExTensor:
    return negate(grad_output)
```

**Key Property**: Flips sign of gradient

---

## Absolute Value

**Forward**:

```text
z = |x|
```

**Gradient**:

```text
∂z/∂x = sign(x) = {
    +1  if x > 0
    -1  if x < 0
     0  if x = 0  (subgradient)
}
```

**Code**:

```mojo
fn abs_backward(grad_output: ExTensor, input: ExTensor) raises -> ExTensor:
    var grad = zeros_like(input)

    for i in range(input._numel):
        var x = input._get_float64(i)
        var sign: Float64
        if x > 0:
            sign = 1.0
        elif x < 0:
            sign = -1.0
        else:
            sign = 0.0  // Subgradient at zero

        grad._set_float64(i, grad_output._get_float64(i) * sign)

    return grad
```

**Key Property**: Not differentiable at x=0 (use subgradient=0)

---

## Sign Function

**Forward**:

```text
z = sign(x) = {
    +1  if x > 0
    -1  if x < 0
     0  if x = 0
}
```

**Gradient**:

```text
∂z/∂x = 0  (discontinuous - zero everywhere except at x=0)
```

**Code**:

```mojo
fn sign_backward(grad_output: ExTensor, input: ExTensor) raises -> ExTensor:
    // Sign function is not differentiable (piecewise constant)
    return zeros_like(input)
```

**Key Property**: No gradient (discontinuous)

**Straight-Through Estimator** (STE): Sometimes used in quantization:

```mojo
fn sign_backward_ste(grad_output: ExTensor, input: ExTensor) raises -> ExTensor:
    // Pass gradient through as if sign was identity
    return grad_output  // Biased but enables learning
```

---

## Clamp (Clip)

**Forward**:

```text
z = clip(x, min, max) = {
    min  if x < min
    x    if min ≤ x ≤ max
    max  if x > max
}
```

**Gradient**:

```text
∂z/∂x = {
    0  if x < min or x > max
    1  if min ≤ x ≤ max
}
```

**Code**:

```mojo
fn clip_backward(
    grad_output: ExTensor, input: ExTensor, min_val: Float64, max_val: Float64
) raises -> ExTensor:
    var grad = zeros_like(input)

    for i in range(input._numel):
        var x = input._get_float64(i)
        if x >= min_val and x <= max_val:
            grad._set_float64(i, grad_output._get_float64(i))
        // Else: gradient is zero (saturated)

    return grad
```

**Key Property**: Gradient is zero outside [min, max] (saturation)

---

## Broadcasting Rules Summary

When inputs have different shapes, NumPy/PyTorch broadcasting rules apply:

**Rule 1**: Prepend 1s to smaller rank tensor
**Rule 2**: Dimensions of size 1 are broadcast (replicated)
**Rule 3**: All other dimensions must match

**Example**:

```text
a: (3, 1, 5)
b: (   4, 5)
→ b becomes (1, 4, 5)
→ Result: (3, 4, 5)
```

**Backward Broadcasting**:

Sum gradients over dimensions that were broadcast in forward:

```mojo
fn sum_to_shape(grad: ExTensor, target_shape: DynamicVector[Int]) raises -> ExTensor:
    var result = grad

    // Handle rank mismatch
    while result.dim() > target_shape.size:
        result = sum(result, axis=0)  // Remove leading dimensions

    // Handle size-1 dimensions
    for dim in range(result.dim()):
        if target_shape[dim] == 1 and result.shape()[dim] > 1:
            result = sum(result, axis=dim, keepdims=True)

    return result
```

---

## Summary Table

| Operation | Forward | ∂z/∂x | ∂z/∂y | Broadcast? | Notes |
|-----------|---------|-------|-------|------------|-------|
| Add | x + y | 1 | 1 | Yes | Passthrough |
| Subtract | x - y | 1 | -1 | Yes | Sign flip for y |
| Multiply | x * y | y | x | Yes | Swap operands |
| Divide | x / y | 1/y | -x/y² | Yes | Needs epsilon |
| Power | x^n | n·x^(n-1) | x^n·log(x) | Partial | Base or exp |
| Negate | -x | -1 | - | No | Sign flip |
| Abs | \|x\| | sign(x) | - | No | Zero at x=0 |
| Sign | sign(x) | 0 | - | No | Discontinuous |
| Clip | clip(x,a,b) | mask | - | No | Zero outside |

**Common Pitfalls**:

- Forgetting to negate gradient for subtraction
- Missing broadcast dimension summing
- Division by zero (use epsilon!)
- Power with negative base and fractional exponent (complex)

---

**Test Files**: `tests/shared/core/test_arithmetic.mojo`

**Implementation**: `shared/core/arithmetic.mojo`
