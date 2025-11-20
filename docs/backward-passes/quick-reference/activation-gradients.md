# Activation Function Gradients - Quick Reference

All activation functions are **element-wise**: `y[i] = f(x[i])`.

Gradient formula: `∂y/∂x[i] = f'(x[i]) * grad_output[i]`

## ReLU (Rectified Linear Unit)

**Forward**:

```text
relu(x) = max(0, x)
```

**Gradient**:

```text
∂relu/∂x = {
    1  if x > 0
    0  if x ≤ 0
}
```

**Code**:

```mojo
fn relu_backward(grad_output: ExTensor, input: ExTensor) raises -> ExTensor:
    var grad = zeros_like(input)
    for i in range(input._numel):
        if input._get_float64(i) > 0:
            grad._set_float64(i, grad_output._get_float64(i))
    return grad
```

**Key Property**: Gradient is 0 or 1 (binary gate)

---

## Leaky ReLU

**Forward**:

```text
leaky_relu(x, α) = {
    x   if x > 0
    αx  if x ≤ 0
}
```

**Gradient**:

```text
∂leaky_relu/∂x = {
    1  if x > 0
    α  if x ≤ 0
}
```

**Code**:

```mojo
fn leaky_relu_backward(
    grad_output: ExTensor, input: ExTensor, alpha: Float64 = 0.01
) raises -> ExTensor:
    var grad = zeros_like(input)
    for i in range(input._numel):
        var multiplier = 1.0 if input._get_float64(i) > 0 else alpha
        grad._set_float64(i, grad_output._get_float64(i) * multiplier)
    return grad
```

**Key Property**: Non-zero gradient for negative values (prevents dying neurons)

---

## PReLU (Parametric ReLU)

**Forward**:

```text
prelu(x, α) = max(α*x, x)
```

**Gradient w.r.t. input**:

```text
∂prelu/∂x = {
    1  if x > 0
    α  if x ≤ 0
}
```

**Gradient w.r.t. alpha**:

```text
∂prelu/∂α = {
    0  if x > 0
    x  if x ≤ 0
}
```

**Code**:

```mojo
fn prelu_backward(
    grad_output: ExTensor, input: ExTensor, alpha: ExTensor
) raises -> (ExTensor, ExTensor):  // Returns (grad_input, grad_alpha)
    var grad_input = zeros_like(input)
    var grad_alpha = zeros_like(alpha)

    for i in range(input._numel):
        var x = input._get_float64(i)
        var a = alpha._get_float64(i if alpha._numel > 1 else 0)

        if x > 0:
            grad_input._set_float64(i, grad_output._get_float64(i))
        else:
            grad_input._set_float64(i, grad_output._get_float64(i) * a)
            grad_alpha._set_float64(i, grad_output._get_float64(i) * x)

    return (grad_input, grad_alpha)
```

---

## Sigmoid

**Forward**:

```text
σ(x) = 1 / (1 + e^(-x))
```

**Gradient**:

```text
∂σ/∂x = σ(x) * (1 - σ(x))
```

**Code**:

```mojo
fn sigmoid_backward(grad_output: ExTensor, input: ExTensor) raises -> ExTensor:
    var sigmoid_out = sigmoid(input)  // Can cache from forward
    var one_minus_sigmoid = subtract(ones_like(sigmoid_out), sigmoid_out)
    var local_grad = multiply(sigmoid_out, one_minus_sigmoid)
    return multiply(grad_output, local_grad)
```

**Key Property**: Maximum gradient at x=0 (σ(0)=0.5, gradient=0.25), vanishes for large |x|

**Range**: σ(x) ∈ (0, 1), gradient ∈ (0, 0.25]

---

## Tanh (Hyperbolic Tangent)

**Forward**:

```text
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
```

**Gradient**:

```text
∂tanh/∂x = 1 - tanh²(x)
```

**Code**:

```mojo
fn tanh_backward(grad_output: ExTensor, input: ExTensor) raises -> ExTensor:
    var tanh_out = tanh(input)  // Can cache from forward
    var tanh_squared = multiply(tanh_out, tanh_out)
    var local_grad = subtract(ones_like(tanh_out), tanh_squared)
    return multiply(grad_output, local_grad)
```

**Key Property**: Zero-centered (unlike sigmoid), maximum gradient at x=0 (gradient=1)

**Range**: tanh(x) ∈ (-1, 1), gradient ∈ (0, 1]

---

## Softmax

**Forward**:

```text
softmax(x)[i] = e^(xi) / Σⱼ e^(xⱼ)
```

**Jacobian** (NOT element-wise!):

```text
∂softmax(x)[i]/∂xⱼ = softmax(x)[i] * (δᵢⱼ - softmax(x)[j])
where δᵢⱼ = Kronecker delta (1 if i=j, else 0)
```

**Gradient** (Jacobian-vector product):

```text
grad_input[i] = softmax(x)[i] * (grad_output[i] - Σⱼ grad_output[j] * softmax(x)[j])
```

**Code**:

```mojo
fn softmax_backward(grad_output: ExTensor, input: ExTensor) raises -> ExTensor:
    var softmax_out = softmax(input)
    var grad_input = zeros_like(input)

    var num_classes = input.shape()[-1]
    var batch_size = input._numel / num_classes

    for b in range(batch_size):
        var offset = b * num_classes

        // Compute dot product: sum(grad_output * softmax_out)
        var dot_prod: Float64 = 0.0
        for i in range(num_classes):
            dot_prod += grad_output._get_float64(offset + i) *
                        softmax_out._get_float64(offset + i)

        // Jacobian-vector product
        for i in range(num_classes):
            var s = softmax_out._get_float64(offset + i)
            var g = grad_output._get_float64(offset + i)
            grad_input._set_float64(offset + i, s * (g - dot_prod))

    return grad_input
```

**Key Property**: Outputs sum to 1 (probability distribution), gradients are coupled (not independent)

---

## GELU (Gaussian Error Linear Unit)

**Forward** (exact):

```text
gelu(x) = x * Φ(x) = x * 0.5 * (1 + erf(x/√2))
where Φ(x) is Gaussian CDF
```

**Forward** (tanh approximation):

```text
gelu(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
```

**Gradient** (exact):

```text
∂gelu/∂x = Φ(x) + x * φ(x)
where φ(x) = (1/√(2π)) * e^(-x²/2) is Gaussian PDF
```

**Code** (exact):

```mojo
fn gelu_backward(grad_output: ExTensor, input: ExTensor) raises -> ExTensor:
    var grad = zeros_like(input)
    let sqrt_2 = sqrt(2.0)
    let inv_sqrt_2pi = 1.0 / sqrt(2.0 * 3.14159265358979)

    for i in range(input._numel):
        var x = input._get_float64(i)
        var cdf = 0.5 * (1.0 + erf(x / sqrt_2))  // Φ(x)
        var pdf = inv_sqrt_2pi * exp(-0.5 * x * x)  // φ(x)
        var local_grad = cdf + x * pdf
        grad._set_float64(i, grad_output._get_float64(i) * local_grad)

    return grad
```

**Key Property**: Smooth approximation of ReLU, better gradient flow than ReLU

---

## Swish (SiLU)

**Forward**:

```text
swish(x) = x * σ(x) = x / (1 + e^(-x))
```

**Gradient**:

```text
∂swish/∂x = σ(x) + x * σ(x) * (1 - σ(x))
          = swish(x) + σ(x) * (1 - swish(x))
```

**Code**:

```mojo
fn swish_backward(grad_output: ExTensor, input: ExTensor) raises -> ExTensor:
    var sigmoid_out = sigmoid(input)
    var swish_out = multiply(input, sigmoid_out)

    // ∂swish/∂x = swish + σ(1 - swish)
    var one_minus_swish = subtract(ones_like(swish_out), swish_out)
    var term1 = swish_out
    var term2 = multiply(sigmoid_out, one_minus_swish)
    var local_grad = add(term1, term2)

    return multiply(grad_output, local_grad)
```

**Key Property**: Self-gated (uses own value), smooth and non-monotonic

---

## Mish

**Forward**:

```text
mish(x) = x * tanh(softplus(x))
        = x * tanh(ln(1 + e^x))
```

**Gradient**:

```text
∂mish/∂x = tanh(softplus(x)) + x * sech²(softplus(x)) * σ(x)
where sech²(x) = 1 - tanh²(x)
```

**Code**:

```mojo
fn mish_backward(grad_output: ExTensor, input: ExTensor) raises -> ExTensor:
    var grad = zeros_like(input)

    for i in range(input._numel):
        var x = input._get_float64(i)
        var sp = log(1.0 + exp(x))  // softplus(x)
        var tanh_sp = math_tanh(sp)
        var sigmoid_x = 1.0 / (1.0 + exp(-x))
        var sech2_sp = 1.0 - tanh_sp * tanh_sp

        var local_grad = tanh_sp + x * sech2_sp * sigmoid_x
        grad._set_float64(i, grad_output._get_float64(i) * local_grad)

    return grad
```

**Key Property**: Smooth and unbounded above, self-regularizing

---

## ELU (Exponential Linear Unit)

**Forward**:

```text
elu(x, α) = {
    x           if x > 0
    α(e^x - 1)  if x ≤ 0
}
```

**Gradient**:

```text
∂elu/∂x = {
    1           if x > 0
    α * e^x     if x ≤ 0
}
```

**Code**:

```mojo
fn elu_backward(
    grad_output: ExTensor, input: ExTensor, alpha: Float64 = 1.0
) raises -> ExTensor:
    var grad = zeros_like(input)

    for i in range(input._numel):
        var x = input._get_float64(i)
        var multiplier: Float64
        if x > 0:
            multiplier = 1.0
        else:
            multiplier = alpha * exp(x)

        grad._set_float64(i, grad_output._get_float64(i) * multiplier)

    return grad
```

**Key Property**: Negative values push mean closer to zero, smooth everywhere

---

## Summary Table

| Activation | Range | Gradient Range | Zero-Centered | Smooth | Monotonic |
|------------|-------|----------------|---------------|--------|-----------|
| ReLU | [0, ∞) | {0, 1} | No | No (at 0) | Yes |
| Leaky ReLU | (-∞, ∞) | {α, 1} | No | No (at 0) | Yes |
| PReLU | (-∞, ∞) | {α, 1} | No | No (at 0) | Yes |
| Sigmoid | (0, 1) | (0, 0.25] | No | Yes | Yes |
| Tanh | (-1, 1) | (0, 1] | Yes | Yes | Yes |
| Softmax | (0, 1), Σ=1 | varies | No | Yes | N/A |
| GELU | (-∞, ∞) | (0, ~1.08] | No | Yes | Yes |
| Swish | (-∞, ∞) | varies | No | Yes | No |
| Mish | (-∞, ∞) | varies | No | Yes | No |
| ELU | (-α, ∞) | (0, 1] | ~Yes | Yes | Yes |

**Choosing an Activation**:

- **Default**: ReLU (fast, simple, works well)
- **Vanishing gradients**: Leaky ReLU, PReLU, ELU
- **Classification output**: Sigmoid (binary), Softmax (multi-class)
- **Hidden layers (advanced)**: GELU, Swish, Mish (smoother gradients)
- **Zero-centered needed**: Tanh, ELU

---

**Test Files**: `tests/shared/core/test_activations.mojo`, `tests/shared/core/test_advanced_activations.mojo`

**Implementation**: `shared/core/activation.mojo`
