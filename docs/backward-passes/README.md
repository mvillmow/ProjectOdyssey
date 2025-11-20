# Backward Passes: Gradient Computation in ML Odyssey

## Introduction to Backpropagation

### What is Automatic Differentiation?

Automatic differentiation (autodiff) is the process of automatically computing gradients of functions defined by
computer programs. Unlike numerical differentiation (finite differences) or symbolic differentiation (formula
manipulation), autodiff computes exact gradients efficiently by applying the chain rule mechanically.

**Key Concepts**:

- **Forward Pass**: Compute the function output from inputs
- **Backward Pass**: Compute gradients of outputs with respect to inputs
- **Chain Rule**: The mathematical foundation allowing gradient propagation through compositions

### Forward Pass vs Backward Pass

**Forward Pass** (Inference):

```mojo
// Compute y = f(x)
var x = ExTensor(...)  // Input
var h1 = relu(matmul(x, W1))  // Hidden layer
var y = matmul(h1, W2)  // Output
```

**Backward Pass** (Training):

```mojo
// Compute ∂loss/∂W1 and ∂loss/∂W2
var grad_y = ones_like(y)  // Gradient from loss
var grad_h1 = matmul_backward(grad_y, h1, W2)  // Backprop through linear
var grad_x = relu_backward(grad_h1, h1)  // Backprop through relu
```

The backward pass applies the **chain rule** in reverse order, propagating gradients from outputs back to inputs.

### Chain Rule Fundamentals

The chain rule is the mathematical foundation of backpropagation:

**Single Variable**:

```text
If y = f(g(x)), then dy/dx = (df/dg) * (dg/dx)
```

**Multiple Variables**:

```text
If z = f(x, y), x = g(t), y = h(t), then:
dz/dt = (∂f/∂x)(dx/dt) + (∂f/∂y)(dy/dt)
```

**Neural Network Example**:

```text
loss = L(f3(f2(f1(x))))

∂loss/∂x = (∂L/∂f3) * (∂f3/∂f2) * (∂f2/∂f1) * (∂f1/∂x)
```

Each backward function computes one term in this product and multiplies it by the gradient from the next layer.

### Why Numerical Validation Matters

Backward passes must be **mathematically correct**. Even small gradient errors compound through layers, causing:

- Training divergence or failure to converge
- Incorrect parameter updates
- Poor model performance despite correct architecture

**Numerical gradient checking** is the **gold standard** for validating backward implementations:

```mojo
// Analytical gradient (what we implement)
var grad_analytical = operation_backward(grad_output, input)

// Numerical gradient (ground truth via finite differences)
var grad_numerical = compute_numerical_gradient(forward_fn, input)

// They must match within tolerance
assert_gradients_close(grad_analytical, grad_numerical, rtol=1e-4)
```

**Why this works**: Finite differences directly approximate the derivative definition, providing an independent
verification method.

## Gradient Computation Theory

### Mathematical Foundations

**Scalar Derivative** (Single variable calculus):

```text
f'(x) = lim[h→0] (f(x+h) - f(x)) / h
```

**Partial Derivative** (Multivariable calculus):

```text
∂f/∂xi = lim[h→0] (f(x1,...,xi+h,...,xn) - f(x1,...,xi,...,xn)) / h
```

**Gradient** (Vector of partial derivatives):

```text
∇f(x) = [∂f/∂x1, ∂f/∂x2, ..., ∂f/∂xn]ᵀ
```

### Vector and Matrix Gradients (Jacobians)

For vector-valued functions f: ℝⁿ → ℝᵐ, the **Jacobian matrix** contains all partial derivatives:

```text
J = [∂fi/∂xj]  (m × n matrix)

J = ⎡ ∂f1/∂x1  ∂f1/∂x2  ...  ∂f1/∂xn ⎤
    ⎢ ∂f2/∂x1  ∂f2/∂x2  ...  ∂f2/∂xn ⎥
    ⎢    ⋮         ⋮      ⋱      ⋮    ⎥
    ⎣ ∂fm/∂x1  ∂fm/∂x2  ...  ∂fm/∂xn ⎦
```

**Example: Matrix Multiplication**

For Y = A @ B (matrix multiplication):

```text
∂Y/∂A = ? (Jacobian is 4D tensor!)
```

In practice, we use the **Vector-Jacobian Product (VJP)** for efficiency:

```text
Given upstream gradient dL/dY, compute:
dL/dA = dL/dY @ Bᵀ  (chain rule with matrix calculus)
dL/dB = Aᵀ @ dL/dY
```

This is what backward functions implement—not the full Jacobian, but the product of upstream gradient with the Jacobian.

### Broadcasting Gradients

When operations broadcast inputs (e.g., adding scalar to matrix), gradients must be **summed over broadcast dimensions**
to match input shapes.

**Example: Scalar + Matrix**

```mojo
var a = scalar(2.0)       // Shape: ()
var b = matrix(3, 4)      // Shape: (3, 4)
var c = add(a, b)         // Shape: (3, 4) - a is broadcast
```

**Forward**: `a` is broadcast to `(3, 4)` by replication.

**Backward**:

```mojo
var grad_c = ones(3, 4)   // Gradient from upstream
var grad_b = grad_c        // Shape: (3, 4) - direct passthrough
var grad_a = sum(grad_c)   // Shape: () - sum over broadcast dimensions!
```

**Rule**: Sum gradients over dimensions that were broadcast in the forward pass.

### Reduction Gradients

Reduction operations (sum, mean, max) **remove dimensions**, so gradients must be **broadcast back** to input shape.

**Example: Sum Reduction**

```mojo
var x = tensor(3, 4)      // Shape: (3, 4)
var y = sum(x, axis=1)    // Shape: (3,) - reduced axis 1
```

**Backward**:

```mojo
var grad_y = ones(3)      // Shape: (3,)
var grad_x = sum_backward(grad_y, x, axis=1)  // Shape: (3, 4)
```

The gradient is **broadcast** from `(3,)` to `(3, 4)` by replication along the reduced axis.

**Mean Reduction** adds an additional scaling factor:

```text
∂mean(x)/∂xi = 1/N  (where N is the number of elements summed)
```

**Max/Min Reduction** routes gradient only to the maximum/minimum position:

```text
∂max(x)/∂xi = 1 if xi == max(x), else 0
```

## Implementation in ML Odyssey

### Pure Functional Architecture

ML Odyssey uses a **pure functional API** for all operations:

- **No mutation**: All operations return new tensors
- **No side effects**: Functions depend only on inputs
- **Composability**: Operations chain naturally

```mojo
// Pure functional style
var y = relu(add(matmul(x, W), b))  // Compose operations

// NOT imperative style (no .apply(), .backward() methods)
```

**Benefits**:

- Easier to reason about gradient flow
- Safer for parallel execution
- Clearer separation of forward/backward passes

### ExTensor Gradient Flow

The `ExTensor` type is the core tensor abstraction:

```mojo
struct ExTensor:
    var _data: DTypePointer[DType.uint8]  // Raw data buffer
    var _shape: DynamicVector[Int]        // Tensor dimensions
    var _dtype: DType                     // Data type
    var _numel: Int                       // Total elements
```

**Gradient Flow Pattern**:

```text
Forward:  input → operation → output
Backward: grad_output → operation_backward → grad_input
```

Each operation has two functions:

1. **Forward** (e.g., `relu`): Computes output from input
2. **Backward** (e.g., `relu_backward`): Computes input gradient from output gradient

### Backward Function Signatures

All backward functions follow a **consistent signature pattern**:

**Unary Operations** (one input):

```mojo
fn operation_backward(
    grad_output: ExTensor,  // Gradient from upstream
    input: ExTensor         // Original input from forward pass
) raises -> ExTensor:       // Returns gradient w.r.t input
```

**Binary Operations** (two inputs):

```mojo
fn operation_backward(
    grad_output: ExTensor,  // Gradient from upstream
    input_a: ExTensor,      // Original first input
    input_b: ExTensor       // Original second input
) raises -> (ExTensor, ExTensor):  // Returns (grad_a, grad_b)
```

**Operations with Parameters** (e.g., pooling with kernel_size):

```mojo
fn operation_backward(
    grad_output: ExTensor,
    input: ExTensor,
    kernel_size: Int,      // Forward pass parameters
    stride: Int,
    padding: Int
) raises -> ExTensor:
```

**Key Principle**: Backward functions receive **all information from the forward pass** needed to compute gradients.

### Common Patterns

**Pattern 1: Element-wise Operations** (e.g., ReLU, sigmoid)

```mojo
fn relu_backward(grad_output: ExTensor, input: ExTensor) raises -> ExTensor:
    var grad_input = zeros_like(input)
    for i in range(input._numel):
        if input._get_float64(i) > 0:
            grad_input._set_float64(i, grad_output._get_float64(i))
    return grad_input
```

**Pattern**: Gradient is `grad_output * local_derivative`, computed element-wise.

**Pattern 2: Matrix Operations** (e.g., matmul)

```mojo
fn matmul_backward(
    grad_output: ExTensor, a: ExTensor, b: ExTensor
) raises -> (ExTensor, ExTensor):
    var grad_a = matmul(grad_output, transpose(b))
    var grad_b = matmul(transpose(a), grad_output)
    return (grad_a, grad_b)
```

**Pattern**: Use matrix calculus identities and chain rule.

**Pattern 3: Reduction Operations** (e.g., sum, mean)

```mojo
fn sum_backward(
    grad_output: ExTensor, input_shape: DynamicVector[Int], axis: Int
) raises -> ExTensor:
    // Broadcast grad_output back to input_shape
    return broadcast(grad_output, input_shape)
```

**Pattern**: Broadcast gradient back to input shape over reduced dimensions.

**Pattern 4: Broadcasting Operations** (e.g., scalar + tensor)

```mojo
fn add_backward(
    grad_output: ExTensor, a: ExTensor, b: ExTensor
) raises -> (ExTensor, ExTensor):
    var grad_a = sum_over_broadcast_dims(grad_output, a.shape())
    var grad_b = sum_over_broadcast_dims(grad_output, b.shape())
    return (grad_a, grad_b)
```

**Pattern**: Sum gradients over dimensions that were broadcast in forward pass.

## Operation-Specific Gradients

### 4.1 Activation Functions

All activation functions are **element-wise**: `y[i] = f(x[i])`.

Gradient formula: `∂y/∂x[i] = f'(x[i]) * grad_output[i]`

#### ReLU (Rectified Linear Unit)

**Forward**:

```text
relu(x) = max(0, x) = {
    x  if x > 0
    0  if x ≤ 0
}
```

**Derivative**:

```text
∂relu/∂x = {
    1  if x > 0
    0  if x ≤ 0
}
```

**Backward Implementation**:

```mojo
fn relu_backward(grad_output: ExTensor, input: ExTensor) raises -> ExTensor:
    var grad_input = zeros_like(input)
    for i in range(input._numel):
        if input._get_float64(i) > 0.0:
            grad_input._set_float64(i, grad_output._get_float64(i))
    return grad_input
```

**Key Point**: Gradient passes through for positive values, zero otherwise.

#### Sigmoid

**Forward**:

```text
σ(x) = 1 / (1 + e^(-x))
```

**Derivative**:

```text
∂σ/∂x = σ(x) * (1 - σ(x))
```

**Backward Implementation**:

```mojo
fn sigmoid_backward(grad_output: ExTensor, input: ExTensor) raises -> ExTensor:
    var sigmoid_out = sigmoid(input)  // Recompute (or cache from forward)
    var grad = multiply(sigmoid_out, subtract(ones_like(sigmoid_out), sigmoid_out))
    return multiply(grad_output, grad)
```

**Optimization**: Can cache `sigmoid(input)` from forward pass to avoid recomputation.

#### Tanh (Hyperbolic Tangent)

**Forward**:

```text
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
```

**Derivative**:

```text
∂tanh/∂x = 1 - tanh²(x)
```

**Backward Implementation**:

```mojo
fn tanh_backward(grad_output: ExTensor, input: ExTensor) raises -> ExTensor:
    var tanh_out = tanh(input)  // Recompute or cache
    var tanh_squared = multiply(tanh_out, tanh_out)
    var grad = subtract(ones_like(tanh_out), tanh_squared)
    return multiply(grad_output, grad)
```

#### Leaky ReLU

**Forward**:

```text
leaky_relu(x, α) = max(αx, x) = {
    x   if x > 0
    αx  if x ≤ 0
}
```

**Derivative**:

```text
∂leaky_relu/∂x = {
    1  if x > 0
    α  if x ≤ 0
}
```

**Backward Implementation**:

```mojo
fn leaky_relu_backward(
    grad_output: ExTensor, input: ExTensor, alpha: Float64 = 0.01
) raises -> ExTensor:
    var grad_input = zeros_like(input)
    for i in range(input._numel):
        var grad_multiplier = 1.0 if input._get_float64(i) > 0 else alpha
        grad_input._set_float64(i, grad_output._get_float64(i) * grad_multiplier)
    return grad_input
```

#### GELU (Gaussian Error Linear Unit)

**Forward** (exact):

```text
gelu(x) = x * Φ(x) = x * 0.5 * (1 + erf(x/√2))
```

**Derivative** (complex, involves error function):

```text
∂gelu/∂x = Φ(x) + x * φ(x)
where φ(x) = (1/√(2π)) * e^(-x²/2) (Gaussian PDF)
```

**Backward Implementation**:

```mojo
fn gelu_backward(grad_output: ExTensor, input: ExTensor) raises -> ExTensor:
    var grad_input = zeros_like(input)
    let sqrt_2 = sqrt(2.0)
    let inv_sqrt_2pi = 1.0 / sqrt(2.0 * 3.14159265358979)

    for i in range(input._numel):
        var x = input._get_float64(i)
        var cdf = 0.5 * (1.0 + erf(x / sqrt_2))
        var pdf = inv_sqrt_2pi * exp(-0.5 * x * x)
        var grad = cdf + x * pdf
        grad_input._set_float64(i, grad_output._get_float64(i) * grad)
    return grad_input
```

#### Softmax

Softmax is **not element-wise**—each output depends on **all inputs**.

**Forward**:

```text
softmax(x)[i] = e^(xi) / Σⱼ e^(xⱼ)
```

**Jacobian** (for single sample):

```text
∂softmax(x)[i]/∂xⱼ = softmax(x)[i] * (δᵢⱼ - softmax(x)[j])

where δᵢⱼ = {1 if i=j, 0 otherwise} (Kronecker delta)
```

**Backward Implementation** (using Jacobian-vector product):

```mojo
fn softmax_backward(grad_output: ExTensor, input: ExTensor) raises -> ExTensor:
    var softmax_out = softmax(input)  // Recompute or cache

    // For each sample in batch
    var grad_input = zeros_like(input)
    var num_classes = input.shape()[-1]
    var batch_size = input._numel / num_classes

    for b in range(batch_size):
        var offset = b * num_classes
        // Compute dot product: sum(grad_output * softmax_out)
        var dot_product: Float64 = 0.0
        for i in range(num_classes):
            dot_product += grad_output._get_float64(offset + i) *
                           softmax_out._get_float64(offset + i)

        // Jacobian-vector product
        for i in range(num_classes):
            var s = softmax_out._get_float64(offset + i)
            var g = grad_output._get_float64(offset + i)
            grad_input._set_float64(offset + i, s * (g - dot_product))

    return grad_input
```

**Key Insight**: The derivative involves a dot product term because outputs are coupled.

### 4.2 Arithmetic Operations

#### Addition

**Forward**:

```text
z = x + y
```

**Derivatives**:

```text
∂z/∂x = 1
∂z/∂y = 1
```

**Backward Implementation**:

```mojo
fn add_backward(
    grad_output: ExTensor, a: ExTensor, b: ExTensor
) raises -> (ExTensor, ExTensor):
    // Gradient is just passthrough (scaled by 1)
    var grad_a = grad_output
    var grad_b = grad_output

    // Handle broadcasting: sum over broadcast dimensions
    grad_a = sum_to_shape(grad_a, a.shape())
    grad_b = sum_to_shape(grad_b, b.shape())

    return (grad_a, grad_b)
```

**Broadcasting Example**:

```mojo
// Forward: (3,4) + (4,) → (3,4) (broadcast second)
var a = tensor(3, 4)
var b = tensor(4)
var c = add(a, b)  // b broadcast to (3,4)

// Backward: grad_c is (3,4)
var (grad_a, grad_b) = add_backward(grad_c, a, b)
// grad_a: (3,4) - passthrough
// grad_b: sum(grad_c, axis=0) → (4,) - sum over broadcast axis
```

#### Subtraction

**Forward**:

```text
z = x - y
```

**Derivatives**:

```text
∂z/∂x = +1
∂z/∂y = -1  (note the minus sign!)
```

**Backward Implementation**:

```mojo
fn subtract_backward(
    grad_output: ExTensor, a: ExTensor, b: ExTensor
) raises -> (ExTensor, ExTensor):
    var grad_a = grad_output
    var grad_b = negate(grad_output)  // Flip sign for second operand

    // Handle broadcasting
    grad_a = sum_to_shape(grad_a, a.shape())
    grad_b = sum_to_shape(grad_b, b.shape())

    return (grad_a, grad_b)
```

#### Multiplication (Element-wise)

**Forward**:

```text
z = x * y  (element-wise)
```

**Derivatives**:

```text
∂z/∂x = y
∂z/∂y = x
```

**Backward Implementation**:

```mojo
fn multiply_backward(
    grad_output: ExTensor, a: ExTensor, b: ExTensor
) raises -> (ExTensor, ExTensor):
    var grad_a = multiply(grad_output, b)  // ∂z/∂a = b
    var grad_b = multiply(grad_output, a)  // ∂z/∂b = a

    // Handle broadcasting
    grad_a = sum_to_shape(grad_a, a.shape())
    grad_b = sum_to_shape(grad_b, b.shape())

    return (grad_a, grad_b)
```

**Example**:

```text
If a = [2, 3], b = [4, 5], grad_output = [1, 1], then:
grad_a = grad_output * b = [1*4, 1*5] = [4, 5]
grad_b = grad_output * a = [1*2, 1*3] = [2, 3]
```

#### Division

**Forward**:

```text
z = x / y
```

**Derivatives**:

```text
∂z/∂x = 1/y
∂z/∂y = -x/y²
```

**Backward Implementation**:

```mojo
fn divide_backward(
    grad_output: ExTensor, a: ExTensor, b: ExTensor
) raises -> (ExTensor, ExTensor):
    // ∂z/∂a = 1/b
    var grad_a = divide(grad_output, b)

    // ∂z/∂b = -a/b²
    var b_squared = multiply(b, b)
    var neg_a_over_b_squared = negate(divide(a, b_squared))
    var grad_b = multiply(grad_output, neg_a_over_b_squared)

    // Handle broadcasting
    grad_a = sum_to_shape(grad_a, a.shape())
    grad_b = sum_to_shape(grad_b, b.shape())

    return (grad_a, grad_b)
```

#### Power

**Forward**:

```text
z = x^n
```

**Derivative**:

```text
∂z/∂x = n * x^(n-1)
```

**Backward Implementation**:

```mojo
fn power_backward(
    grad_output: ExTensor, base: ExTensor, exponent: Float64
) raises -> ExTensor:
    // ∂(x^n)/∂x = n * x^(n-1)
    var x_power_n_minus_1 = power(base, exponent - 1.0)
    var grad = multiply_scalar(x_power_n_minus_1, exponent)
    return multiply(grad_output, grad)
```

### 4.3 Loss Functions

#### Mean Squared Error (MSE)

**Forward**:

```text
MSE(pred, target) = (pred - target)²
```

**Derivative**:

```text
∂MSE/∂pred = 2(pred - target)
```

**Backward Implementation**:

```mojo
fn mean_squared_error_backward(
    grad_output: ExTensor, predictions: ExTensor, targets: ExTensor
) raises -> ExTensor:
    // Gradient: 2 * (predictions - targets)
    var diff = subtract(predictions, targets)
    var two = full_like(diff, 2.0)
    var grad = multiply(two, diff)
    return multiply(grad_output, grad)
```

**Note**: If using `mean(mse(...))`, must chain with `mean_backward` which scales by `1/N`.

#### Binary Cross-Entropy (BCE)

**Forward**:

```text
BCE(p, y) = -[y*log(p) + (1-y)*log(1-p)]
where p = predictions, y = targets
```

**Derivative** (simplified form):

```text
∂BCE/∂p = (p - y) / (p(1-p))  (exact)
∂BCE/∂p ≈ (p - y)             (common approximation)
```

**Backward Implementation** (simplified):

```mojo
fn binary_cross_entropy_backward(
    grad_output: ExTensor, predictions: ExTensor, targets: ExTensor
) raises -> ExTensor:
    // Simplified gradient: (predictions - targets)
    var grad = subtract(predictions, targets)
    return multiply(grad_output, grad)
```

**Why simplified form?**: When combined with sigmoid activation, the `1/(p(1-p))` term cancels with sigmoid's
derivative, leaving clean `(p - y)` gradient.

#### Cross-Entropy with Softmax

**Forward** (numerically stable):

```text
CE(logits, targets) = -sum(targets * log(softmax(logits)))
```

**Derivative** (combined softmax + cross-entropy):

```text
∂CE/∂logits = softmax(logits) - targets
```

**Backward Implementation**:

```mojo
fn cross_entropy_backward(
    grad_output: ExTensor, logits: ExTensor, targets: ExTensor
) raises -> ExTensor:
    // Remarkably simple when combined with softmax!
    var softmax_out = softmax(logits)
    var grad = subtract(softmax_out, targets)
    return multiply(grad_output, grad)
```

**Key Insight**: The Jacobian of softmax and the gradient of log-likelihood **combine beautifully** to produce this
simple form. This is why softmax + cross-entropy is always used together.

**Derivation**:

```text
Let s = softmax(x), then:
∂(-log(si))/∂xⱼ = ∂(-log(si))/∂si * ∂si/∂xⱼ
                = -1/si * si(δᵢⱼ - sⱼ)
                = -(δᵢⱼ - sⱼ)
                = sⱼ - δᵢⱼ

For one-hot target yᵢ (1 for correct class, 0 otherwise):
∂CE/∂xⱼ = sum over i of: yi * (sⱼ - δᵢⱼ)
        = yc * (sⱼ - δcⱼ)  (only correct class c has yi=1)
        = sⱼ - δcⱼ
        = sⱼ - yⱼ  (since y is one-hot)
        = softmax(x) - y
```

### 4.4 Matrix Operations

#### Matrix Multiplication (matmul)

**Forward**:

```text
C = A @ B  (matrix multiplication)
Cᵢₖ = Σⱼ Aᵢⱼ * Bⱼₖ
```

**Derivatives** (using matrix calculus):

```text
∂C/∂A = grad_C @ Bᵀ
∂C/∂B = Aᵀ @ grad_C
```

**Backward Implementation**:

```mojo
fn matmul_backward(
    grad_output: ExTensor, a: ExTensor, b: ExTensor
) raises -> (ExTensor, ExTensor):
    // ∂C/∂A = grad_C @ Bᵀ
    var grad_a = matmul(grad_output, transpose(b))

    // ∂C/∂B = Aᵀ @ grad_C
    var grad_b = matmul(transpose(a), grad_output)

    return (grad_a, grad_b)
```

**Shape Validation**:

```text
Forward:  A(m,n) @ B(n,p) → C(m,p)
Backward: grad_C(m,p) @ Bᵀ(p,n) → grad_A(m,n) ✓
          Aᵀ(n,m) @ grad_C(m,p) → grad_B(n,p) ✓
```

**Intuition**: To get gradient w.r.t. left operand, multiply with transposed right operand. Vice versa for right
operand.

#### Transpose

**Forward**:

```text
B = Aᵀ  (transpose)
Bᵢⱼ = Aⱼᵢ
```

**Derivative**:

```text
∂B/∂A = transpose(grad_B)
```

**Backward Implementation**:

```mojo
fn transpose_backward(grad_output: ExTensor, input: ExTensor) raises -> ExTensor:
    // Gradient of transpose is transpose of gradient
    return transpose(grad_output)
```

**Intuition**: If you transpose forward, transpose backward. Transposing twice returns to original orientation.

#### Dot Product

**Forward**:

```text
c = a · b = Σᵢ aᵢbᵢ  (scalar output)
```

**Derivatives**:

```text
∂c/∂a = b
∂c/∂b = a
```

**Backward Implementation**:

```mojo
fn dot_backward(
    grad_output: ExTensor, a: ExTensor, b: ExTensor
) raises -> (ExTensor, ExTensor):
    // grad_output is scalar, broadcast to vector
    var grad_a = multiply_scalar(b, grad_output._get_float64(0))
    var grad_b = multiply_scalar(a, grad_output._get_float64(0))
    return (grad_a, grad_b)
```

#### Outer Product

**Forward**:

```text
C = a ⊗ b (outer product)
Cᵢⱼ = aᵢ * bⱼ
```

**Derivatives**:

```text
∂C/∂aᵢ = bⱼ for all j (gradient is sum over j: Σⱼ grad_Cᵢⱼ * bⱼ)
∂C/∂bⱼ = aᵢ for all i (gradient is sum over i: Σᵢ grad_Cᵢⱼ * aᵢ)
```

**Backward Implementation**:

```mojo
fn outer_backward(
    grad_output: ExTensor, a: ExTensor, b: ExTensor
) raises -> (ExTensor, ExTensor):
    // grad_a = grad_output @ b (sum over columns)
    var grad_a = matmul(grad_output, b.reshape(-1, 1)).reshape(a.shape())

    // grad_b = aᵀ @ grad_output (sum over rows)
    var grad_b = matmul(a.reshape(1, -1), grad_output).reshape(b.shape())

    return (grad_a, grad_b)
```

### 4.5 Reduction Operations

#### Sum

**Forward**:

```text
y = sum(x)  (reduce to scalar)
y = Σᵢ xᵢ
```

**Derivative**:

```text
∂y/∂xᵢ = 1  for all i
```

**Backward Implementation** (scalar output):

```mojo
fn sum_backward(
    grad_output: ExTensor, input_shape: DynamicVector[Int]
) raises -> ExTensor:
    // Broadcast scalar gradient to input shape
    var grad_input = ExTensor(input_shape, grad_output.dtype())
    var scalar_grad = grad_output._get_float64(0)

    for i in range(grad_input._numel):
        grad_input._set_float64(i, scalar_grad)

    return grad_input
```

**Backward Implementation** (axis reduction):

```mojo
fn sum_backward_axis(
    grad_output: ExTensor, input_shape: DynamicVector[Int], axis: Int
) raises -> ExTensor:
    // Broadcast gradient along reduced axis
    // Example: input (3,4) → sum(axis=1) → (3,)
    //          grad_output (3,) → grad_input (3,4)

    var grad_input = ExTensor(input_shape, grad_output.dtype())

    // Broadcast by repeating along reduced axis
    // ... (implementation details in reduction.mojo)

    return grad_input
```

**Key Principle**: Sum reduction **distributes gradient equally** to all inputs that contributed to each output element.

#### Mean

**Forward**:

```text
y = mean(x) = (1/N) * sum(x)
where N = number of elements
```

**Derivative**:

```text
∂y/∂xᵢ = 1/N  for all i
```

**Backward Implementation**:

```mojo
fn mean_backward(
    grad_output: ExTensor, input_shape: DynamicVector[Int]
) raises -> ExTensor:
    // Gradient is (1/N) * broadcast(grad_output)
    var grad_input = sum_backward(grad_output, input_shape)  // Broadcast

    var count = Float64(compute_numel(input_shape))
    for i in range(grad_input._numel):
        var g = grad_input._get_float64(i)
        grad_input._set_float64(i, g / count)

    return grad_input
```

**Key Difference from Sum**: Additional `1/N` scaling factor.

#### Max/Min Reduction

**Forward**:

```text
y = max(x) = max{x₁, x₂, ..., xₙ}
```

**Derivative**:

```text
∂y/∂xᵢ = {
    1  if xᵢ == max(x)
    0  otherwise
}
```

**Backward Implementation**:

```mojo
fn max_backward(
    grad_output: ExTensor, input: ExTensor
) raises -> ExTensor:
    var grad_input = zeros_like(input)

    // Find index of maximum element
    var max_val = input._get_float64(0)
    var max_idx = 0
    for i in range(1, input._numel):
        var val = input._get_float64(i)
        if val > max_val:
            max_val = val
            max_idx = i

    // Route gradient only to maximum position
    grad_input._set_float64(max_idx, grad_output._get_float64(0))

    return grad_input
```

**Key Principle**: Gradient flows **only to the maximum element**, all others get zero gradient.

**Axis Reduction**: When reducing along an axis, each "slice" has its own maximum:

```mojo
fn max_backward_axis(
    grad_output: ExTensor, input: ExTensor, axis: Int
) raises -> ExTensor:
    var grad_input = zeros_like(input)

    // For each position in output, find which input position was maximum
    // ... (track argmax during forward pass for efficiency)

    return grad_input
```

**Tie-Breaking**: If multiple elements tie for maximum, gradient is typically routed to the **first occurrence**
(implementation-dependent).

## Testing Backward Passes

### Numerical Gradient Checking (Gold Standard)

Numerical gradient checking validates analytical gradients using **finite differences**—the **gold standard** for
gradient correctness.

**Central Difference Formula**:

```text
f'(x) ≈ (f(x + ε) - f(x - ε)) / (2ε)

Error: O(ε²)  (much better than forward/backward difference O(ε))
```

**Implementation in ML Odyssey**:

```mojo
fn compute_numerical_gradient(
    forward_fn: fn(ExTensor) raises -> ExTensor,
    x: ExTensor,
    epsilon: Float64 = 1e-5
) raises -> ExTensor:
    var grad = zeros_like(x)

    for i in range(x._numel):
        var original = x._get_float64(i)

        // Perturb +ε
        x._set_float64(i, original + epsilon)
        var f_plus = forward_fn(x)

        // Perturb -ε
        x._set_float64(i, original - epsilon)
        var f_minus = forward_fn(x)

        // Central difference
        var grad_val = (f_plus._get_float64(0) - f_minus._get_float64(0)) / (2.0 * epsilon)
        grad._set_float64(i, grad_val)

        // Restore
        x._set_float64(i, original)

    return grad
```

**Why this works**:

- Directly approximates the definition of derivative
- Independent of analytical implementation (detects errors)
- Convergence rate known: error scales with ε²

### Central Difference Method

**Why central difference?**

**Forward Difference**:

```text
f'(x) ≈ (f(x + ε) - f(x)) / ε
Error: O(ε)
```

**Backward Difference**:

```text
f'(x) ≈ (f(x) - f(x - ε)) / ε
Error: O(ε)
```

**Central Difference**:

```text
f'(x) ≈ (f(x + ε) - f(x - ε)) / (2ε)
Error: O(ε²)
```

**Taylor Series Derivation**:

```text
f(x + ε) = f(x) + ε·f'(x) + (ε²/2)·f''(x) + O(ε³)
f(x - ε) = f(x) - ε·f'(x) + (ε²/2)·f''(x) + O(ε³)

Subtracting:
f(x + ε) - f(x - ε) = 2ε·f'(x) + O(ε³)

Therefore:
f'(x) = (f(x + ε) - f(x - ε)) / (2ε) + O(ε²)
```

The first-order error terms cancel, leaving **second-order accuracy**.

### Tolerance Guidelines

Choosing tolerances depends on floating-point precision and numerical stability.

**Recommended Tolerances**:

| Data Type | Relative Tolerance (rtol) | Absolute Tolerance (atol) | Epsilon (ε) |
|-----------|---------------------------|---------------------------|-------------|
| Float16   | 1e-2                      | 1e-4                      | 1e-3        |
| Float32   | 1e-4                      | 1e-7                      | 1e-5        |
| Float64   | 1e-7                      | 1e-10                     | 1e-7        |

**Tolerance Check Formula**:

```text
|analytical - numerical| ≤ atol + rtol * |numerical|
```

This handles both:

- **Small gradients**: Absolute tolerance prevents false failures near zero
- **Large gradients**: Relative tolerance scales with magnitude

**Choosing Epsilon**:

Too large: Truncation error dominates (poor approximation)
Too small: Roundoff error dominates (floating-point precision limits)

**Optimal ε** ≈ √(machine epsilon) ≈ 1e-5 for float32

**Trade-off Visualization**:

```text
Total Error
    |     /
    |    /  Truncation error (∝ ε²)
    |   /\
    |  /  \___
    | /       \___  Roundoff error (∝ 1/ε)
    |/____________\____________
               ε_optimal
```

### Example Test Patterns

**Pattern 1: Simple Function Test**

```mojo
fn test_relu_gradient() raises:
    // Setup
    var shape = DynamicVector[Int](1)
    shape[0] = 4
    var x = zeros(shape, DType.float32)
    x._data.bitcast[Float32]()[0] = -1.0
    x._data.bitcast[Float32]()[1] = 0.0
    x._data.bitcast[Float32]()[2] = 1.0
    x._data.bitcast[Float32]()[3] = 2.0

    // Forward
    fn forward(inp: ExTensor) raises -> ExTensor:
        return relu(inp)

    var y = relu(x)
    var grad_out = ones_like(y)

    // Backward (analytical)
    var grad_analytical = relu_backward(grad_out, x)

    // Backward (numerical)
    var grad_numerical = compute_numerical_gradient(forward, x)

    // Compare
    assert_gradients_close(grad_analytical, grad_numerical, rtol=1e-4, atol=1e-7)
```

**Pattern 2: Helper Function for Convenience**

```mojo
fn check_gradient(
    forward_fn: fn(ExTensor) raises -> ExTensor,
    backward_fn: fn(ExTensor, ExTensor) raises -> ExTensor,
    x: ExTensor,
    grad_output: ExTensor,
    rtol: Float64 = 1e-4,
    atol: Float64 = 1e-7
) raises:
    var analytical = backward_fn(grad_output, x)
    var numerical = compute_numerical_gradient(forward_fn, x)
    assert_gradients_close(analytical, numerical, rtol, atol)


fn test_sigmoid_gradient() raises:
    var x = create_test_tensor(10, DType.float32)

    fn forward(inp: ExTensor) raises -> ExTensor:
        return sigmoid(inp)

    var grad_out = ones_like(sigmoid(x))
    check_gradient(forward, sigmoid_backward, x, grad_out)
```

**Pattern 3: Batched Operations**

```mojo
fn test_matmul_gradient() raises:
    var a = create_tensor(4, 3, DType.float32)
    var b = create_tensor(3, 5, DType.float32)

    // Check gradient w.r.t. first input
    fn forward_a(inp: ExTensor) raises -> ExTensor:
        return sum(matmul(inp, b))  // Scalar output for gradient checking

    var grad_output = ones(4, 5)
    var (grad_a, _) = matmul_backward(grad_output, a, b)
    var numerical_a = compute_numerical_gradient(forward_a, a)
    assert_gradients_close(grad_a, numerical_a)

    // Check gradient w.r.t. second input
    fn forward_b(inp: ExTensor) raises -> ExTensor:
        return sum(matmul(a, inp))

    var (_, grad_b) = matmul_backward(grad_output, a, b)
    var numerical_b = compute_numerical_gradient(forward_b, b)
    assert_gradients_close(grad_b, numerical_b)
```

**Pattern 4: Testing with Known Values**

```mojo
fn test_multiply_gradient_known_values() raises:
    // Specific test case with hand-computed gradients
    var a = scalar(2.0)
    var b = scalar(3.0)
    var c = multiply(a, b)  // c = 6.0

    // If upstream gradient is 1.0:
    var grad_out = scalar(1.0)
    var (grad_a, grad_b) = multiply_backward(grad_out, a, b)

    // Expected: grad_a = b = 3.0, grad_b = a = 2.0
    assert_almost_equal(grad_a._get_float64(0), 3.0, tolerance=1e-7)
    assert_almost_equal(grad_b._get_float64(0), 2.0, tolerance=1e-7)
```

**Pattern 5: Comprehensive Operation Coverage**

```mojo
fn test_all_activations() raises:
    var activations = [
        ("relu", relu, relu_backward),
        ("sigmoid", sigmoid, sigmoid_backward),
        ("tanh", tanh, tanh_backward),
        ("leaky_relu", leaky_relu, leaky_relu_backward),
    ]

    for (name, forward_fn, backward_fn) in activations:
        var x = create_random_tensor(100, DType.float32)
        check_gradient(forward_fn, backward_fn, x, ones_like(forward_fn(x)))
        print("✓ " + name + " gradient check passed")
```

## Common Pitfalls

### 1. Forgetting to Sum Over Broadcast Dimensions

**Problem**: When operations broadcast inputs, gradients must be summed to match input shapes.

**Example (Incorrect)**:

```mojo
fn add_backward_wrong(grad_output: ExTensor, a: ExTensor, b: ExTensor)
    raises -> (ExTensor, ExTensor):
    // WRONG: Doesn't handle broadcasting!
    return (grad_output, grad_output)
```

**What happens**:

```mojo
var a = tensor(3, 4)  // Shape: (3, 4)
var b = tensor(4)     // Shape: (4)
var c = add(a, b)     // Shape: (3, 4) - b broadcast to (3, 4)

var grad_c = ones(3, 4)
var (grad_a, grad_b) = add_backward_wrong(grad_c, a, b)
// grad_a: (3, 4) ✓
// grad_b: (3, 4) ✗ WRONG! Should be (4)
```

**Correct Implementation**:

```mojo
fn add_backward(grad_output: ExTensor, a: ExTensor, b: ExTensor)
    raises -> (ExTensor, ExTensor):
    var grad_a = sum_to_shape(grad_output, a.shape())  // Sum over broadcast dims
    var grad_b = sum_to_shape(grad_output, b.shape())  // Sum over broadcast dims
    return (grad_a, grad_b)
```

**Helper Function**:

```mojo
fn sum_to_shape(grad: ExTensor, target_shape: DynamicVector[Int]) raises -> ExTensor:
    // Sum over dimensions that were broadcast in forward pass
    var result = grad

    // Handle scalar case
    if target_shape.size == 0:
        return sum(result)  // Sum all dimensions

    // Sum over broadcast dimensions
    for dim in range(result.dim()):
        if dim >= target_shape.size or target_shape[dim] == 1:
            result = sum(result, axis=dim, keepdims=True)

    return result.reshape(target_shape)
```

### 2. Incorrect Shape Transformations

**Problem**: Forgetting to account for shape changes in operations like transpose, reshape, or reduction.

**Example: Transpose** (Incorrect):

```mojo
fn transpose_backward_wrong(grad_output: ExTensor, input: ExTensor) raises -> ExTensor:
    // WRONG: Just returns gradient without adjusting shape
    return grad_output
```

**Correct**:

```mojo
fn transpose_backward(grad_output: ExTensor, input: ExTensor) raises -> ExTensor:
    // Transpose the gradient to match input shape
    return transpose(grad_output)
```

**Example: Matrix Multiplication** (Incorrect):

```mojo
fn matmul_backward_wrong(grad_output: ExTensor, a: ExTensor, b: ExTensor)
    raises -> (ExTensor, ExTensor):
    // WRONG: Shape mismatch!
    return (grad_output, grad_output)
```

**Shape Analysis**:

```text
Forward:  A(m,n) @ B(n,p) → C(m,p)
Backward: Need grad_A(m,n) and grad_B(n,p) from grad_C(m,p)

grad_C(m,p) directly doesn't match either input shape!
```

**Correct**:

```mojo
fn matmul_backward(grad_output: ExTensor, a: ExTensor, b: ExTensor)
    raises -> (ExTensor, ExTensor):
    // grad_A = grad_C @ Bᵀ:  (m,p) @ (p,n) = (m,n) ✓
    var grad_a = matmul(grad_output, transpose(b))

    // grad_B = Aᵀ @ grad_C:  (n,m) @ (m,p) = (n,p) ✓
    var grad_b = matmul(transpose(a), grad_output)

    return (grad_a, grad_b)
```

### 3. Gradient Scaling Errors (Mean Backward)

**Problem**: Forgetting the `1/N` factor when backpropagating through mean.

**Example** (Incorrect):

```mojo
fn mean_backward_wrong(grad_output: ExTensor, input_shape: DynamicVector[Int])
    raises -> ExTensor:
    // WRONG: Missing 1/N scaling!
    return broadcast(grad_output, input_shape)
```

**What happens**:

```text
Forward:  x = [1, 2, 3, 4] → mean(x) = 2.5
          mean = (1/4) * sum(x)

Backward: If grad_output = 1.0, then:
          ∂mean/∂x = 1/N = 0.25 for each element

Without scaling, gradient is 4× too large!
```

**Correct**:

```mojo
fn mean_backward(grad_output: ExTensor, input_shape: DynamicVector[Int])
    raises -> ExTensor:
    var grad_input = broadcast(grad_output, input_shape)

    // Scale by 1/N
    var count = Float64(compute_numel(input_shape))
    for i in range(grad_input._numel):
        var g = grad_input._get_float64(i)
        grad_input._set_float64(i, g / count)

    return grad_input
```

**General Rule**: If forward pass has scaling factor, backward pass must have same factor.

### 4. Numerical Instability (Log, Divide by Zero)

**Problem**: Operations like `log`, `divide`, and `sqrt` are numerically unstable near singular points.

**Example: Log** (Unsafe):

```mojo
fn log_backward_unsafe(grad_output: ExTensor, input: ExTensor) raises -> ExTensor:
    // WRONG: Divide by zero if input == 0!
    return divide(grad_output, input)  // ∂log(x)/∂x = 1/x
```

**What happens**:

```text
If input = 0, then gradient = 1/0 = inf or nan
```

**Correct** (with epsilon):

```mojo
fn log_backward(grad_output: ExTensor, input: ExTensor, epsilon: Float64 = 1e-8)
    raises -> ExTensor:
    // Clamp input to avoid division by zero
    var safe_input = clip(input, epsilon, Float64.max_value())
    return divide(grad_output, safe_input)
```

**Example: Sqrt** (Unstable at zero):

```mojo
fn sqrt_backward_unsafe(grad_output: ExTensor, input: ExTensor) raises -> ExTensor:
    // WRONG: Gradient explodes as input → 0
    var sqrt_input = sqrt(input)
    return divide(grad_output, multiply_scalar(sqrt_input, 2.0))  // ∂√x/∂x = 1/(2√x)
}

fn sqrt_backward_safe(grad_output: ExTensor, input: ExTensor, epsilon: Float64 = 1e-8)
    raises -> ExTensor:
    // Clamp to prevent division by zero
    var safe_sqrt = sqrt(clip(input, epsilon, Float64.max_value()))
    return divide(grad_output, multiply_scalar(safe_sqrt, 2.0))
}
```

**Example: Division** (Zero denominator):

```mojo
fn divide_backward(grad_output: ExTensor, numerator: ExTensor, denominator: ExTensor)
    raises -> (ExTensor, ExTensor):
    // ∂(a/b)/∂a = 1/b
    var safe_denom = clip(denominator, 1e-8, Float64.max_value())
    var grad_numerator = divide(grad_output, safe_denom)

    // ∂(a/b)/∂b = -a/b²
    var denom_squared = multiply(safe_denom, safe_denom)
    var grad_denominator = divide(negate(multiply(grad_output, numerator)), denom_squared)

    return (grad_numerator, grad_denominator)
```

**Best Practices**:

- Always use `epsilon` clamping for `log`, `sqrt`, `divide`
- Check for `inf` and `nan` in tests
- Use numerically stable algorithms (e.g., log-sum-exp for softmax)

## Advanced Topics

### Second-Order Gradients (Hessians)

**What are second-order gradients?**

While first-order gradients are derivatives of loss w.r.t. parameters:

```text
First order:  ∂L/∂θ  (gradient vector)
```

Second-order gradients are derivatives of gradients (Hessian matrix):

```text
Second order: ∂²L/∂θᵢ∂θⱼ  (Hessian matrix)
```

**Use Cases**:

- **Second-order optimization**: Newton's method, L-BFGS
- **Uncertainty estimation**: Laplace approximation
- **Gradient penalty**: Regularization techniques

**Implementation Strategy**:

To compute second-order gradients, apply automatic differentiation **twice**:

```mojo
// First order: ∂L/∂x
var grad_first = backward_pass(loss, x)

// Second order: ∂(∂L/∂x)/∂x
var grad_second = backward_pass(grad_first, x)
```

**Example: Hessian-Vector Product**:

```mojo
fn hessian_vector_product(
    loss_fn: fn(ExTensor) raises -> ExTensor,
    x: ExTensor,
    v: ExTensor
) raises -> ExTensor:
    """Compute H·v where H is Hessian of loss w.r.t. x."""

    // First backward: compute gradient
    fn grad_fn(inp: ExTensor) raises -> ExTensor:
        var loss = loss_fn(inp)
        var grad = compute_gradient(loss, inp)
        return dot(grad, v)  // Directional derivative in direction v

    // Second backward: gradient of (gradient · v)
    var loss = loss_fn(x)
    var grad = compute_gradient(loss, x)
    var grad_dot_v = dot(grad, v)
    var hessian_v = compute_gradient(grad_dot_v, x)

    return hessian_v
```

**Challenges**:

- **Memory intensive**: Full Hessian is O(n²) for n parameters
- **Computationally expensive**: Requires two backward passes
- **Sparse in practice**: Most neural network Hessians are nearly diagonal

**Practical Approximations**:

- **Diagonal Hessian**: Only compute ∂²L/∂θᵢ² (ignores cross-terms)
- **Fisher Information**: Expected Hessian (used in K-FAC)
- **Low-rank approximations**: L-BFGS stores Hessian implicitly

### Gradient Checkpointing

**Problem**: Deep networks require storing all intermediate activations for backward pass, consuming massive memory.

**Solution**: **Gradient checkpointing** trades compute for memory by recomputing activations during backward pass.

**Strategy**:

```text
Forward pass:  Store only checkpoint activations (every N layers)
Backward pass: Recompute activations between checkpoints as needed
```

**Example** (conceptual):

```mojo
// Standard backward (memory intensive)
fn standard_backward():
    // Forward: store all activations
    var a1 = f1(x)
    var a2 = f2(a1)
    var a3 = f3(a2)
    var a4 = f4(a3)
    var loss = L(a4)

    // Backward: use stored activations
    var grad_a4 = loss_backward(loss, a4)
    var grad_a3 = f4_backward(grad_a4, a3)  // Uses stored a3
    var grad_a2 = f3_backward(grad_a3, a2)  // Uses stored a2
    var grad_a1 = f2_backward(grad_a2, a1)  // Uses stored a1
    var grad_x  = f1_backward(grad_a1, x)

// Gradient checkpointing (memory efficient)
fn checkpointed_backward():
    // Forward: store only checkpoints (a1, a3)
    var a1 = f1(x)
    # checkpoint(a1)
    var a2 = f2(a1)  // Don't store
    var a3 = f3(a2)
    # checkpoint(a3)
    var a4 = f4(a3)  // Don't store
    var loss = L(a4)

    // Backward: recompute between checkpoints
    var grad_a4 = loss_backward(loss, a4)
    var a3_recomputed = f3(f2(a1))  // Recompute from checkpoint
    var grad_a3 = f4_backward(grad_a4, a3_recomputed)
    var grad_a2 = f3_backward(grad_a3, f2(a1))
    # ... continue
```

**Trade-offs**:

- **Memory savings**: O(√n) memory for n layers (vs O(n) standard)
- **Compute overhead**: ~33% extra forward pass computations
- **When to use**: Very deep networks, limited GPU memory

**Implementation in ML Odyssey** (future work):

```mojo
fn checkpoint(operation: fn(ExTensor) raises -> ExTensor, input: ExTensor)
    raises -> ExTensor:
    """Mark operation for gradient checkpointing."""
    # Store only input (not output) for this operation
    # During backward, recompute operation from stored input
```

### Mixed Precision Gradients

**What is mixed precision training?**

Use different floating-point precisions for different parts of training:

- **Forward pass**: Float16 (faster, less memory)
- **Gradients**: Float16 (faster communication)
- **Parameter updates**: Float32 (numerical stability)

**Benefits**:

- **Speed**: 2-3× faster on modern GPUs (Tensor Cores)
- **Memory**: 2× reduction in activation memory
- **Throughput**: Larger batch sizes fit in memory

**Challenges**:

- **Underflow**: Small gradients become zero in Float16
- **Overflow**: Large gradients become infinity
- **Precision loss**: Accumulation errors in Float16

**Solution: Loss Scaling**:

```mojo
// Forward pass (Float16)
var loss_fp16 = compute_loss(predictions_fp16, targets_fp16)

// Scale loss before backward (prevent underflow)
var scaled_loss = multiply_scalar(loss_fp16, 1024.0)  // Loss scaling factor

// Backward pass (Float16 gradients)
var gradients_fp16 = backward_pass(scaled_loss)

// Unscale gradients before update (Float32)
var gradients_fp32 = cast(gradients_fp16, DType.float32)
gradients_fp32 = divide_scalar(gradients_fp32, 1024.0)  // Unscale

// Update parameters (Float32 master copy)
parameters_fp32 = optimizer_step(parameters_fp32, gradients_fp32)

// Cast parameters back to Float16 for next iteration
parameters_fp16 = cast(parameters_fp32, DType.float16)
```

**Dynamic Loss Scaling**:

```mojo
fn adjust_loss_scale(current_scale: Float32, overflow_detected: Bool) -> Float32:
    if overflow_detected:
        return current_scale / 2.0  // Reduce scale
    else:
        return current_scale * 1.1  // Gradually increase
```

**Implementation Considerations**:

- **Gradient accumulation**: Must stay in Float32
- **Batch normalization**: Statistics should use Float32
- **Layer normalization**: Safe in Float16

**ML Odyssey Support** (planned):

```mojo
fn backward_mixed_precision(
    grad_output: ExTensor[DType.float16],
    input: ExTensor[DType.float16],
    loss_scale: Float32
) raises -> ExTensor[DType.float32]:
    // Backward in Float16
    var grad_fp16 = operation_backward(grad_output, input)

    // Cast to Float32 and unscale
    var grad_fp32 = cast(grad_fp16, DType.float32)
    return divide_scalar(grad_fp32, loss_scale)
```

### SIMD Optimization for Gradients

Mojo's SIMD capabilities enable **vectorized gradient computation** for massive speedups.

**Scalar Implementation** (slow):

```mojo
fn relu_backward_scalar(grad_output: ExTensor, input: ExTensor) raises -> ExTensor:
    var grad_input = zeros_like(input)

    // Process one element at a time
    for i in range(input._numel):
        if input._get_float64(i) > 0.0:
            grad_input._set_float64(i, grad_output._get_float64(i))

    return grad_input
```

**SIMD Implementation** (fast):

```mojo
fn relu_backward_simd[simd_width: Int = simdwidthof[DType.float32]()](
    grad_output: ExTensor, input: ExTensor
) raises -> ExTensor:
    var grad_input = zeros_like(input)

    var input_ptr = input._data.bitcast[Float32]()
    var grad_out_ptr = grad_output._data.bitcast[Float32]()
    var grad_in_ptr = grad_input._data.bitcast[Float32]()

    // Process simd_width elements at once (e.g., 8 float32s)
    @parameter
    fn vectorized[width: Int](i: Int):
        var inp_vec = input_ptr.load[width=width](i)
        var grad_vec = grad_out_ptr.load[width=width](i)
        var zero_vec = SIMD[DType.float32, width](0.0)

        // Mask: 1 where input > 0, else 0
        var mask = inp_vec > zero_vec

        // Apply mask: gradient * mask
        var result = grad_vec.select(mask, zero_vec)
        grad_in_ptr.store[width=width](i, result)

    vectorize[vectorized, simd_width](input._numel)

    return grad_input
```

**Speedup**: 4-8× faster for float32 (depending on hardware SIMD width)

**Key SIMD Patterns for Gradients**:

**Pattern 1: Element-wise with mask** (ReLU, Leaky ReLU):

```mojo
var mask = input > threshold
var result = grad_output.select(mask, alternative_value)
```

**Pattern 2: Element-wise multiplication** (Sigmoid, Tanh):

```mojo
var activation = compute_activation_simd(input)
var local_grad = compute_local_derivative_simd(activation)
var result = grad_output * local_grad
```

**Pattern 3: Reduction with SIMD** (Sum, Mean):

```mojo
var sum_vec = SIMD[DType.float32, simd_width](0.0)
for i in range(0, numel, simd_width):
    sum_vec += data_ptr.load[width=simd_width](i)

var sum_scalar = sum_vec.reduce_add()  // Horizontal reduction
```

**Pattern 4: Broadcasting with SIMD**:

```mojo
// Broadcast scalar gradient to vector
var scalar_grad = SIMD[DType.float32, simd_width](grad_scalar)
for i in range(0, numel, simd_width):
    result_ptr.store[width=simd_width](i, scalar_grad)
```

**Best Practices**:

- Use `@parameter` for compile-time optimization
- Align memory accesses to SIMD width
- Handle remainder elements separately (tail loop)
- Prefer SIMD for operations on >1000 elements

**Performance Tips**:

```mojo
// Good: Aligned access
@parameter
fn aligned_load[width: Int](i: Int):
    var data = ptr.load[width=width](i * width)  // Aligned

// Bad: Unaligned access (slower)
fn unaligned_load[width: Int](i: Int):
    var data = ptr.load[width=width](i * width + 1)  // Misaligned
```

**ML Odyssey Convention**:

All performance-critical backward passes should have SIMD implementations with `[simd_width: Int]` parameter.

## Quick Reference

See the `quick-reference/` directory for operation-specific formula sheets:

- [Activation Gradients](quick-reference/activation-gradients.md) - All activation function derivatives
- [Arithmetic Gradients](quick-reference/arithmetic-gradients.md) - Element-wise operation derivatives
- [Loss Gradients](quick-reference/loss-gradients.md) - Loss function derivatives
- [Matrix Gradients](quick-reference/matrix-gradients.md) - Matrix operation derivatives
- [Reduction Gradients](quick-reference/reduction-gradients.md) - Reduction operation derivatives

## Code Examples

See the `examples/` directory for complete implementations:

- [Simple MLP Backward](examples/simple_mlp_backward.mojo) - Complete forward/backward for 2-layer MLP
- [Convolutional Network Backward](examples/conv_net_backward.mojo) - Conv2D, pooling, and activation gradients
- [Custom Loss Backward](examples/custom_loss_backward.mojo) - Implementing custom loss functions
- [Numerical Validation Example](examples/numerical_validation_example.mojo) - Gradient checking template

## References

### Textbooks

- **Neural Networks and Deep Learning** by Michael Nielsen (Free online)
  - Chapter 2: How the backpropagation algorithm works
  - <http://neuralnetworksanddeeplearning.com/>

- **Deep Learning** by Goodfellow, Bengio, and Courville (MIT Press)
  - Chapter 6: Deep Feedforward Networks
  - Section 6.5: Backpropagation

- **Pattern Recognition and Machine Learning** by Christopher Bishop
  - Chapter 5: Neural Networks
  - Section 5.3: Error Backpropagation

### Research Papers

- **Automatic Differentiation in Machine Learning: A Survey** (Baydin et al., 2018)
- **Backpropagation Applied to Handwritten Zip Code Recognition** (LeCun et al., 1989)

### Online Resources

- **PyTorch Autograd Documentation**: <https://pytorch.org/docs/stable/autograd.html>
- **JAX Autodiff Documentation**: <https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html>
- **Stanford CS231n**: Backpropagation lecture notes

### ML Odyssey Test Files

All 47 backward functions are validated in these test files:

- [test_activations.mojo](../../tests/shared/core/test_activations.mojo) - 10 activation backward tests
- [test_arithmetic.mojo](../../tests/shared/core/test_arithmetic.mojo) - 7 arithmetic backward tests
- [test_matrix.mojo](../../tests/shared/core/test_matrix.mojo) - 4 matrix operation backward tests
- [test_backward.mojo](../../tests/shared/core/test_backward.mojo) - Linear, conv2d, pooling, cross-entropy
- [gradient_checking.mojo](../../tests/helpers/gradient_checking.mojo) - Numerical gradient utilities

### Implementation Files

Core backward pass implementations:

- [activation.mojo](../../shared/core/activation.mojo) - Activation function backwards
- [arithmetic.mojo](../../shared/core/arithmetic.mojo) - Arithmetic operation backwards
- [matrix.mojo](../../shared/core/matrix.mojo) - Matrix operation backwards
- [loss.mojo](../../shared/core/loss.mojo) - Loss function backwards
- [reduction.mojo](../../shared/core/reduction.mojo) - Reduction operation backwards
- [linear.mojo](../../shared/core/linear.mojo) - Linear layer backward
- [conv.mojo](../../shared/core/conv.mojo) - Convolutional layer backward
- [pooling.mojo](../../shared/core/pooling.mojo) - Pooling layer backwards

---

**Contributors**: See [CONTRIBUTORS.md](../../CONTRIBUTORS.md)

**License**: MIT License - See [LICENSE](../../LICENSE)

**Last Updated**: 2025-01-20
