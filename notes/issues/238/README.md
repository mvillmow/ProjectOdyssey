# Issue #238: [Plan] ReLU Family - Design and Documentation

## Objective

Design and document the ReLU (Rectified Linear Unit) activation function family including standard ReLU, Leaky ReLU, and PReLU variants. This component provides essential non-linear activation functions widely used in modern deep learning architectures.

## Deliverables

- **ReLU activation function**: Standard ReLU implementing `max(0, x)` for sparse activation
- **Leaky ReLU activation function**: ReLU variant with small negative slope `max(alpha*x, x)` to prevent dying neurons
- **PReLU activation function**: Parametric ReLU with learnable negative slope parameter
- **Gradient computation**: Proper backpropagation support for all three variants
- **API documentation**: Complete interface specification and usage examples
- **Design documentation**: Architecture decisions and implementation strategy

## Success Criteria

- [ ] ReLU correctly zeros negative values
- [ ] Leaky ReLU preserves small negative gradients with configurable alpha parameter
- [ ] PReLU parameter updates work correctly during training
- [ ] All variants handle edge cases (zeros, large values, numerical stability)
- [ ] Gradient computation is mathematically correct for backpropagation
- [ ] API is consistent with other activation functions in the library
- [ ] Documentation includes mathematical formulations and usage examples
- [ ] Design decisions are documented with rationale

## Design Decisions

### 1. Activation Function Variants

### ReLU (Rectified Linear Unit)

- **Formula**: `f(x) = max(0, x)`
- **Gradient**: `f'(x) = 1 if x > 0 else 0`
- **Use Case**: Standard activation for hidden layers, promotes sparsity
- **Implementation**: Simple element-wise maximum operation

### Leaky ReLU

- **Formula**: `f(x) = max(alpha * x, x)` where alpha is typically 0.01
- **Gradient**: `f'(x) = 1 if x > 0 else alpha`
- **Use Case**: Prevents "dying ReLU" problem by allowing small negative gradients
- **Implementation**: Configurable alpha parameter (constant, not learnable)

### PReLU (Parametric ReLU)

- **Formula**: `f(x) = max(alpha * x, x)` where alpha is learned during training
- **Gradient**: `f'(x) = 1 if x > 0 else alpha`, plus gradient w.r.t. alpha
- **Use Case**: Adaptive activation that learns optimal negative slope
- **Implementation**: Requires parameter storage and gradient computation for alpha

### 2. Memory Management Strategy

**Decision**: Use Mojo's ownership model with `borrowed` for forward pass and `owned` for gradient computation.

### Rationale

- Forward pass: Input tensors are borrowed (read-only), no copy needed
- Backward pass: Gradient tensors may need modification, use `owned` where appropriate
- Efficient memory usage for large-scale training

### 3. SIMD Optimization

**Decision**: Implement element-wise operations using SIMD vectorization.

### Rationale

- ReLU family operations are embarrassingly parallel
- SIMD provides significant speedup for tensor operations
- Mojo's SIMD capabilities enable efficient vectorization without assembly
- Target SIMD width based on hardware capabilities (detected at runtime)

### 4. Numerical Stability

**Decision**: Handle edge cases explicitly (zeros, very large values, infinities).

### Rationale

- Zero inputs: Well-defined behavior for all variants
- Large positive values: Pass through unchanged (ReLU) or scaled (Leaky/PReLU)
- Large negative values: Become zero (ReLU) or scaled (Leaky/PReLU)
- Infinities: Preserve mathematical semantics

### 5. API Design

**Decision**: Provide both function-based and struct-based APIs.

### Rationale

- Function API: Simple stateless operations for ReLU and Leaky ReLU
- Struct API: Required for PReLU (needs parameter storage)
- Consistency: All variants available in both forms for uniform interface
- Flexibility: Users can choose appropriate abstraction level

### Proposed API

```mojo
# Function-based API (stateless)
fn relu[dtype: DType](x: Tensor[dtype]) -> Tensor[dtype]
fn leaky_relu[dtype: DType](x: Tensor[dtype], alpha: Scalar[dtype]) -> Tensor[dtype]

# Struct-based API (stateful for PReLU)
struct ReLU:
    fn forward[dtype: DType](self, x: Tensor[dtype]) -> Tensor[dtype]
    fn backward[dtype: DType](self, grad_output: Tensor[dtype], x: Tensor[dtype]) -> Tensor[dtype]

struct LeakyReLU:
    var alpha: Float32
    fn forward[dtype: DType](self, x: Tensor[dtype]) -> Tensor[dtype]
    fn backward[dtype: DType](self, grad_output: Tensor[dtype], x: Tensor[dtype]) -> Tensor[dtype]

struct PReLU:
    var alpha: Tensor[DType.float32]  # Learnable parameter
    fn forward[dtype: DType](self, x: Tensor[dtype]) -> Tensor[dtype]
    fn backward[dtype: DType](self, grad_output: Tensor[dtype], x: Tensor[dtype]) -> (Tensor[dtype], Tensor[dtype])
```text

### 6. Gradient Computation Strategy

### ReLU Gradient

- `grad_input = grad_output if x > 0 else 0`
- No parameter gradients (stateless)

### Leaky ReLU Gradient

- `grad_input = grad_output if x > 0 else alpha * grad_output`
- No parameter gradients (alpha is constant)

### PReLU Gradient

- `grad_input = grad_output if x > 0 else alpha * grad_output`
- `grad_alpha = sum(grad_output * x) if x <= 0 else 0` (gradient w.r.t. alpha)
- Requires accumulating gradient for alpha parameter

### 7. Testing Strategy

### Unit Tests

- Correctness: Verify mathematical formulas for all variants
- Edge cases: Zero, positive, negative, large values
- Gradient checks: Numerical gradient verification
- SIMD correctness: Compare vectorized vs. scalar results

### Property Tests

- ReLU non-negativity: Output is always >= 0
- Leaky ReLU monotonicity: Function is strictly monotonic
- PReLU parameter learning: Alpha converges to optimal value

### Performance Tests

- SIMD speedup: Measure vectorization benefits
- Memory efficiency: No unnecessary allocations

### 8. Integration with Tensor Library

**Decision**: Depend on shared tensor operations for basic operations.

### Rationale

- Reuse existing tensor infrastructure (allocation, indexing)
- Focus activation-specific logic, not tensor mechanics
- Consistent with other activation functions

### Dependencies

- Tensor type from core library
- SIMD operations from Mojo stdlib
- Max/comparison operations from tensor library

## References

### Source Plan

- [ReLU Family Plan](notes/plan/02-shared-library/01-core-operations/02-activations/01-relu-family/plan.md)

### Related Issues

- Issue #239: [Test] ReLU Family - Test Implementation
- Issue #240: [Impl] ReLU Family - Core Implementation
- Issue #241: [Package] ReLU Family - Integration and Packaging
- Issue #242: [Cleanup] ReLU Family - Refactor and Finalize

### Parent Plan

- [Activations Plan](notes/plan/02-shared-library/01-core-operations/02-activations/plan.md)

### Related Documentation

- [Mojo Language Review Specialist](.claude/agents/mojo-language-review-specialist.md) - Mojo coding patterns
- [Implementation Specialist](.claude/agents/implementation-specialist.md) - Implementation guidelines
- [Test Specialist](.claude/agents/test-specialist.md) - Testing approach

## Implementation Notes

(This section will be populated during subsequent phases with findings, challenges, and solutions discovered during implementation.)

### Phase Coordination

- **Planning** (Issue #238): This document
- **Testing** (Issue #239): TDD approach - write tests before implementation
- **Implementation** (Issue #240): Core activation functions and gradient computation
- **Packaging** (Issue #241): Integration with activation module and API exports
- **Cleanup** (Issue #242): Refactoring and finalization based on implementation findings

## Mathematical Background

### ReLU

The Rectified Linear Unit (ReLU) activation function is defined as:

```text
f(x) = max(0, x)
```text

Gradient:

```text
∂f/∂x = { 1  if x > 0
        { 0  if x ≤ 0
```text

### Properties

- Non-linear activation
- Sparse activation (zeros out negative values)
- Computationally efficient
- Can suffer from "dying ReLU" problem (neurons permanently inactive)

### Leaky ReLU

Leaky ReLU introduces a small slope for negative values:

```text
f(x) = max(αx, x)  where α ∈ (0, 1), typically α = 0.01
```text

Gradient:

```text
∂f/∂x = { 1  if x > 0
        { α  if x ≤ 0
```text

### Properties

- Prevents dying ReLU problem
- Small negative gradients allow learning even for negative inputs
- α is a hyperparameter (not learned)

### PReLU (Parametric ReLU)

PReLU makes the negative slope learnable:

```text
f(x) = max(αx, x)  where α is learned during training
```text

Gradients:

```text
∂f/∂x = { 1  if x > 0
        { α  if x ≤ 0

∂f/∂α = { 0  if x > 0
        { x  if x ≤ 0
```text

### Properties

- Adaptive activation function
- Learns optimal negative slope for each layer
- Requires parameter storage and gradient computation
- Can improve model performance on some tasks

## Next Steps

1. **Review and approve** this planning document
1. **Begin Test phase** (Issue #239): Write comprehensive tests following TDD
1. **Coordinate with Implementation** (Issue #240): Use test specifications to guide implementation
1. **Prepare for Packaging** (Issue #241): Plan integration with activation module
1. **Document cleanup items** (Issue #242): Track refactoring needs during implementation
