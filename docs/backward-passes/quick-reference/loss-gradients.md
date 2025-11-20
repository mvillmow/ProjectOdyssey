# Loss Function Gradients - Quick Reference

Loss functions compute the discrepancy between predictions and targets. All losses should be **reduced** (mean/sum) to
scalar for backpropagation.

## Mean Squared Error (MSE)

**Forward** (element-wise):

```text
MSE(pred, target) = (pred - target)²
```

**Forward** (scalar):

```text
MSE = mean((pred - target)²)
```

**Gradient**:

```text
∂MSE/∂pred = 2(pred - target)  [element-wise]
∂MSE/∂pred = (2/N)(pred - target)  [after mean reduction]
```

**Code**:

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

**Use Case**: Regression tasks

**Properties**:

- Squared error penalizes large errors more
- Sensitive to outliers
- Differentiable everywhere

---

## Mean Absolute Error (MAE / L1 Loss)

**Forward**:

```text
MAE(pred, target) = |pred - target|
```

**Gradient**:

```text
∂MAE/∂pred = sign(pred - target) = {
    +1  if pred > target
    -1  if pred < target
     0  if pred = target  (subgradient)
}
```

**Code**:

```mojo
fn mean_absolute_error_backward(
    grad_output: ExTensor, predictions: ExTensor, targets: ExTensor
) raises -> ExTensor:
    var grad = zeros_like(predictions)

    for i in range(predictions._numel):
        var diff = predictions._get_float64(i) - targets._get_float64(i)
        var sign: Float64
        if diff > 0:
            sign = 1.0
        elif diff < 0:
            sign = -1.0
        else:
            sign = 0.0  // Subgradient

        grad._set_float64(i, grad_output._get_float64(i) * sign)

    return grad
```

**Use Case**: Regression with outliers (robust to outliers)

**Properties**:

- Linear penalty (vs quadratic for MSE)
- Not differentiable at pred=target
- Less sensitive to outliers than MSE

---

## Binary Cross-Entropy (BCE)

**Forward**:

```text
BCE(p, y) = -[y·log(p) + (1-y)·log(1-p)]
where:
  p = predictions (probabilities in [0,1])
  y = targets (0 or 1)
```

**Gradient** (exact):

```text
∂BCE/∂p = -(y/p - (1-y)/(1-p))
        = (p - y) / (p(1-p))
```

**Gradient** (simplified, common):

```text
∂BCE/∂p ≈ p - y
```

**Code** (simplified):

```mojo
fn binary_cross_entropy_backward(
    grad_output: ExTensor, predictions: ExTensor, targets: ExTensor
) raises -> ExTensor:
    // Simplified gradient: predictions - targets
    var grad = subtract(predictions, targets)
    return multiply(grad_output, grad)
```

**Use Case**: Binary classification (use with sigmoid activation)

**Properties**:

- Works with probabilistic predictions
- Penalizes confident wrong predictions more
- Numerically stable with epsilon clipping

**Why simplified form?**: When combined with sigmoid, denominator cancels:

```text
If pred = σ(z), then:
∂BCE/∂z = ∂BCE/∂p · ∂p/∂z
        = [(p-y)/(p(1-p))] · [p(1-p)]
        = p - y  (clean!)
```

---

## Cross-Entropy (Multi-class)

**Forward** (with softmax):

```text
CE(logits, targets) = -sum(targets · log(softmax(logits)))
```

**Gradient** (combined softmax + cross-entropy):

```text
∂CE/∂logits = softmax(logits) - targets
```

**Code**:

```mojo
fn cross_entropy_backward(
    grad_output: ExTensor, logits: ExTensor, targets: ExTensor
) raises -> ExTensor:
    // Remarkably simple: softmax - targets
    var softmax_out = softmax(logits)
    var grad = subtract(softmax_out, targets)
    return multiply(grad_output, grad)
```

**Use Case**: Multi-class classification

**Properties**:

- Always use with softmax (never sigmoid!)
- Gradient is clean: `softmax(x) - y`
- One-hot targets: gradient is -1 for correct class, +p for others

**Derivation** (why so simple):

```text
Let s = softmax(x), CE = -log(s_correct)

For one-hot target y (y_i = 1 for correct class, 0 otherwise):

∂CE/∂x_j = ∂CE/∂s_i · ∂s_i/∂x_j
         = -1/s_i · s_i(δ_ij - s_j)
         = -(δ_ij - s_j)
         = s_j - δ_ij
         = s_j - y_j  (since y is one-hot)

Therefore: ∂CE/∂x = softmax(x) - y
```

---

## Kullback-Leibler Divergence (KL Divergence)

**Forward**:

```text
KL(p || q) = sum(p · log(p/q))
where:
  p = target distribution
  q = predicted distribution
```

**Gradient w.r.t. q**:

```text
∂KL/∂q = -p/q
```

**Code**:

```mojo
fn kl_divergence_backward(
    grad_output: ExTensor, predictions: ExTensor, targets: ExTensor, epsilon: Float64 = 1e-8
) raises -> ExTensor:
    // ∂KL/∂q = -p/q
    var safe_predictions = clip(predictions, epsilon, Float64.max)
    var grad = negate(divide(targets, safe_predictions))
    return multiply(grad_output, grad)
```

**Use Case**: Distribution matching, knowledge distillation

**Properties**:

- Asymmetric: KL(p||q) ≠ KL(q||p)
- Always non-negative
- Requires p, q to be valid probability distributions

---

## Hinge Loss (SVM Loss)

**Forward**:

```text
Hinge(pred, target) = max(0, 1 - target·pred)
where target ∈ {-1, +1}
```

**Gradient**:

```text
∂Hinge/∂pred = {
    -target  if target·pred < 1
     0       if target·pred ≥ 1
}
```

**Code**:

```mojo
fn hinge_loss_backward(
    grad_output: ExTensor, predictions: ExTensor, targets: ExTensor
) raises -> ExTensor:
    var grad = zeros_like(predictions)

    for i in range(predictions._numel):
        var pred = predictions._get_float64(i)
        var target = targets._get_float64(i)

        if target * pred < 1.0:
            grad._set_float64(i, grad_output._get_float64(i) * (-target))
        // Else: gradient is zero (margin satisfied)

    return grad
```

**Use Case**: Support Vector Machines, margin-based classification

**Properties**:

- Encourages margin of at least 1
- Non-differentiable at target·pred = 1
- Sparse gradients (zero when margin satisfied)

---

## Focal Loss

**Forward**:

```text
Focal(p, y) = -α(1-p)^γ·log(p)  if y=1
            = -α·p^γ·log(1-p)    if y=0

where:
  α = balancing factor (default 0.25)
  γ = focusing parameter (default 2.0)
```

**Gradient** (y=1 case):

```text
∂Focal/∂p = α[(1-p)^γ/p + γ(1-p)^(γ-1)·log(p)]
```

**Code**:

```mojo
fn focal_loss_backward(
    grad_output: ExTensor,
    predictions: ExTensor,
    targets: ExTensor,
    alpha: Float64 = 0.25,
    gamma: Float64 = 2.0,
    epsilon: Float64 = 1e-8
) raises -> ExTensor:
    var grad = zeros_like(predictions)

    for i in range(predictions._numel):
        var p = clip_scalar(predictions._get_float64(i), epsilon, 1.0 - epsilon)
        var y = targets._get_float64(i)

        var local_grad: Float64
        if y == 1.0:
            var one_minus_p = 1.0 - p
            var term1 = pow(one_minus_p, gamma) / p
            var term2 = gamma * pow(one_minus_p, gamma - 1) * log(p)
            local_grad = alpha * (term1 + term2)
        else:
            var term1 = pow(p, gamma) / (1.0 - p)
            var term2 = gamma * pow(p, gamma - 1) * log(1.0 - p)
            local_grad = -alpha * (term1 + term2)

        grad._set_float64(i, grad_output._get_float64(i) * local_grad)

    return grad
```

**Use Case**: Handling class imbalance, hard example mining

**Properties**:

- Down-weights easy examples (high confidence correct predictions)
- Focuses on hard examples
- γ=0 reduces to standard cross-entropy

---

## Huber Loss (Smooth L1)

**Forward**:

```text
Huber(pred, target, δ) = {
    0.5(pred - target)²       if |pred - target| ≤ δ
    δ(|pred - target| - 0.5δ) if |pred - target| > δ
}
```

**Gradient**:

```text
∂Huber/∂pred = {
    (pred - target)        if |pred - target| ≤ δ
    δ·sign(pred - target)  if |pred - target| > δ
}
```

**Code**:

```mojo
fn huber_loss_backward(
    grad_output: ExTensor,
    predictions: ExTensor,
    targets: ExTensor,
    delta: Float64 = 1.0
) raises -> ExTensor:
    var grad = zeros_like(predictions)

    for i in range(predictions._numel):
        var diff = predictions._get_float64(i) - targets._get_float64(i)
        var abs_diff = abs(diff)

        var local_grad: Float64
        if abs_diff <= delta:
            local_grad = diff  // Quadratic region
        else:
            local_grad = delta * (1.0 if diff > 0 else -1.0)  // Linear region

        grad._set_float64(i, grad_output._get_float64(i) * local_grad)

    return grad
```

**Use Case**: Robust regression (less sensitive to outliers than MSE)

**Properties**:

- Quadratic for small errors (smooth)
- Linear for large errors (robust)
- Differentiable everywhere (unlike MAE)

---

## Cosine Similarity Loss

**Forward**:

```text
CosineSim(a, b) = (a·b) / (||a|| ||b||)
CosineLoss = 1 - CosineSim(a, b)
```

**Gradient w.r.t. a**:

```text
∂CosineLoss/∂a = -[b/(||a|| ||b||) - a(a·b)/(||a||³ ||b||)]
```

**Code**:

```mojo
fn cosine_loss_backward(
    grad_output: ExTensor, a: ExTensor, b: ExTensor, epsilon: Float64 = 1e-8
) raises -> (ExTensor, ExTensor):
    // Compute norms
    var norm_a = sqrt(sum(multiply(a, a))._get_float64(0) + epsilon)
    var norm_b = sqrt(sum(multiply(b, b))._get_float64(0) + epsilon)
    var dot_ab = sum(multiply(a, b))._get_float64(0)

    // ∂CosineLoss/∂a = -(b/(||a|| ||b||) - a·(a·b)/(||a||³ ||b||))
    var term1 = divide_scalar(b, norm_a * norm_b)
    var term2 = multiply_scalar(a, dot_ab / (norm_a * norm_a * norm_a * norm_b))
    var grad_a = negate(subtract(term1, term2))

    // Symmetric for grad_b
    var term1_b = divide_scalar(a, norm_a * norm_b)
    var term2_b = multiply_scalar(b, dot_ab / (norm_a * norm_b * norm_b * norm_b))
    var grad_b = negate(subtract(term1_b, term2_b))

    return (
        multiply(grad_output, grad_a),
        multiply(grad_output, grad_b)
    )
```

**Use Case**: Semantic similarity, contrastive learning

**Properties**:

- Invariant to vector magnitude
- Range: [-1, 1] (similarity)
- Gradient undefined when either vector is zero

---

## Summary Table

| Loss | Use Case | Range | Properties | Gradient Complexity |
|------|----------|-------|------------|-------------------|
| MSE | Regression | [0, ∞) | Quadratic penalty, outlier-sensitive | Simple |
| MAE | Robust regression | [0, ∞) | Linear penalty, robust | Simple (subgradient) |
| BCE | Binary classification | [0, ∞) | Probabilistic, works with sigmoid | Simple |
| Cross-Entropy | Multi-class | [0, ∞) | Use with softmax, clean gradient | Simple |
| KL Divergence | Distribution match | [0, ∞) | Asymmetric, measures divergence | Medium |
| Hinge | Margin-based | [0, ∞) | Sparse gradients, SVM-style | Medium |
| Focal | Imbalanced data | [0, ∞) | Down-weights easy examples | Complex |
| Huber | Robust regression | [0, ∞) | Smooth, combines MSE + MAE | Medium |
| Cosine | Similarity | [0, 2] | Magnitude-invariant | Complex |

**Loss Selection Guide**:

- **Regression**: MSE (default), MAE (outliers), Huber (robust)
- **Binary Classification**: BCE with sigmoid
- **Multi-class Classification**: Cross-Entropy with softmax
- **Class Imbalance**: Focal Loss, weighted Cross-Entropy
- **Embedding/Similarity**: Cosine Loss, Triplet Loss
- **Distribution Matching**: KL Divergence

**Common Pitfalls**:

- Forgetting to reduce loss to scalar (use mean/sum)
- Using sigmoid with cross-entropy (should use softmax)
- Division by zero in BCE, KL (use epsilon clipping)
- Mixing up target formats (one-hot vs class indices)

---

**Test Files**: `tests/shared/core/test_backward.mojo` (cross-entropy), `tests/shared/core/test_loss.mojo`

**Implementation**: `shared/core/loss.mojo`
