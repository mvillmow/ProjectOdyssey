# Issue #248: [Plan] Softmax GELU - Design and Documentation

## Objective

Design and document modern activation functions: softmax for converting logits to probability distributions in multi-class classification, and GELU (Gaussian Error Linear Unit) for smooth non-linear activation in transformers and modern architectures.

## Deliverables

- Softmax activation specification: `exp(x) / sum(exp(x))` with numerical stability
- GELU activation specification: `x * Phi(x)` where Phi is Gaussian CDF
- API contracts and interfaces for both functions
- Numerical stability requirements and implementation strategies
- Gradient computation specifications for backpropagation

## Success Criteria

- [ ] Softmax outputs sum to 1.0 along specified axis
- [ ] Softmax is numerically stable for large logits
- [ ] GELU produces smooth activation curve
- [ ] Both functions work correctly in forward and backward passes
- [ ] Complete design documentation created
- [ ] API contracts defined
- [ ] Numerical stability strategies documented

## Design Decisions

### 1. Softmax Numerical Stability

**Decision**: Implement log-sum-exp trick for numerical stability.

**Rationale**: Raw softmax computation `exp(x) / sum(exp(x))` can overflow for large values. Subtracting the maximum value before exponentiation prevents overflow while preserving mathematical correctness:

```text
softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
```text

### Implementation Strategy

- Find max value along specified axis
- Subtract max from all elements
- Apply exp and normalize
- Support arbitrary axis specification

### 2. GELU Implementation Approach

**Decision**: Provide both exact and approximate implementations.

### Exact Formula

```text
GELU(x) = x * Phi(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
```text

**Approximate Formula** (faster, used in original paper):

```text
GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
```text

### Rationale

- Exact formula provides theoretical correctness
- Approximate formula offers computational efficiency
- Modern hardware may make exact formula practical
- Let users choose based on accuracy vs speed tradeoffs

### 3. Gradient Computation

**Decision**: Implement analytical gradients for both functions.

### Softmax Gradient

```text
∂softmax(x)_i / ∂x_j = softmax(x)_i * (δ_ij - softmax(x)_j)
```text

where `δ_ij` is Kronecker delta (1 if i=j, 0 otherwise)

### GELU Gradient

```text
∂GELU(x) / ∂x = Phi(x) + x * phi(x)
```text

where `phi(x)` is the Gaussian PDF

**Rationale**: Analytical gradients are more accurate and efficient than numerical approximations.

### 4. API Design

### Softmax Interface

```mojo
fn softmax[dtype: DType](
    tensor: Tensor[dtype],
    axis: Int = -1,
    keepdims: Bool = False
) -> Tensor[dtype]:
    """
    Apply softmax activation along specified axis.

    Args:
        tensor: Input tensor (logits)
        axis: Axis along which to compute softmax (default: -1, last axis)
        keepdims: Whether to keep reduced dimension (default: False)

    Returns:
        Tensor with softmax applied, values sum to 1.0 along axis
    """
```text

### GELU Interface

```mojo
fn gelu[dtype: DType](
    tensor: Tensor[dtype],
    approximate: Bool = False
) -> Tensor[dtype]:
    """
    Apply GELU (Gaussian Error Linear Unit) activation.

    Args:
        tensor: Input tensor
        approximate: Use tanh approximation (faster) vs exact erf (default: False)

    Returns:
        Tensor with GELU activation applied
    """
```text

### 5. Edge Cases and Stability

### Softmax Edge Cases

- Empty tensors: Raise error
- Single element: Return 1.0
- All identical values: Return uniform distribution
- Very large negative values: Handle gracefully (result near 0)
- Very large positive values: Use log-sum-exp trick

### GELU Edge Cases

- Very large positive x: GELU(x) ≈ x
- Very large negative x: GELU(x) ≈ 0
- x = 0: GELU(0) ≈ 0

### 6. Testing Strategy

### Softmax Tests

- Verify outputs sum to 1.0 (within numerical tolerance)
- Test numerical stability with large values (±100, ±1000)
- Verify axis parameter works correctly
- Test gradient computation accuracy
- Compare against reference implementations

### GELU Tests

- Verify exact and approximate versions produce similar results
- Test extreme values (±10, ±100)
- Verify smooth activation curve (no discontinuities)
- Test gradient computation accuracy
- Compare against reference implementations (PyTorch, TensorFlow)

### 7. Performance Considerations

### Softmax

- SIMD optimization for exp and sum operations
- Minimize memory allocations (in-place where possible)
- Optimize for common case (axis=-1, 2D tensors for classification)

### GELU

- Approximate version should be default for performance
- SIMD optimization for polynomial operations in approximation
- Consider lookup tables for erf in exact version

## References

- **Source Plan**: [notes/plan/02-shared-library/01-core-operations/02-activations/03-softmax-gelu/plan.md](../../../plan/02-shared-library/01-core-operations/02-activations/03-softmax-gelu/plan.md)
- **Parent Plan**: [notes/plan/02-shared-library/01-core-operations/02-activations/plan.md](../../../plan/02-shared-library/01-core-operations/02-activations/plan.md)
- **Related Issues**:
  - Issue #249: [Test] Softmax GELU
  - Issue #250: [Impl] Softmax GELU
  - Issue #251: [Package] Softmax GELU
  - Issue #252: [Cleanup] Softmax GELU

- **External References**:
  - GELU Paper: "Gaussian Error Linear Units (GELUs)" - Hendrycks & Gimpel (2016)
  - Softmax: Standard in "Deep Learning" - Goodfellow, Bengio, Courville
  - PyTorch Softmax: [torch.nn.functional.softmax](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html)
  - PyTorch GELU: [torch.nn.functional.gelu](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.gelu.html)

## Implementation Notes

(This section will be populated during Test, Implementation, and Packaging phases)
